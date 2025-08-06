# training/trading_env.py

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

# [수정 없음] GATree 모델 및 예측 함수, 상수 임포트
from models.model import (
    GATreePop,
    ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_CLOSE_ALL,
    ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION, ACTION_FLIP_POSITION
)
from training.predictor import predict_population_cuda

# ##############################################################################
#           거래 환경 및 시뮬레이션 핵심 로직 (수정됨)
# ##############################################################################

# --- [제거] 시뮬레이션 환경 설정을 위한 전역 상수 정의 제거 ---
# TAKER_FEE_RATE, MAINTENANCE_MARGIN_RATE 등은 TradingEnvironment 클래스 내부로 이동

# --- [수정 없음] GATree 예측에 필요한 포지션 정수 -> 문자열 변환 맵 ---
INT_TO_POSITION_MAP = {
    0: 'HOLD',
    1: 'SHORT',
    2: 'LONG'
}


class TradingEnvironment:
    """
    [수정] 거래 환경 설정을 생성자에서 받아 초기화하는 방식으로 변경된 거래 시뮬레이션 환경 클래스.
    """
    def __init__(self, chromosomes_size, env_config, device='cpu'):
        self.device = device
        self.chromosomes_size = chromosomes_size
        
        # --- [신규] 시뮬레이션 환경 변수를 config에서 받아 인스턴스 변수로 저장 ---
        self.taker_fee_rate = env_config['taker_fee_rate']
        self.maintenance_margin_rate = env_config['maintenance_margin_rate']
        self.fixed_slippage_rate = env_config['fixed_slippage_rate']
        self.funding_fee_hours = {0, 8, 16} # 펀딩피 시간은 일반적으로 고정이므로 유지

        # --- 나머지 변수들은 기존과 동일 ---
        self.pos_list = torch.zeros(chromosomes_size, dtype=torch.long, device=device)
        self.price_list = torch.full((chromosomes_size,), -1.0, dtype=torch.float32, device=device)
        self.leverage_ratio = torch.full((chromosomes_size,), -1, dtype=torch.int, device=device)
        self.enter_ratio = torch.full((chromosomes_size,), -1.0, dtype=torch.float32, device=device)
        self.additional_count = torch.zeros(chromosomes_size, dtype=torch.long, device=device)
        self.holding_period = torch.zeros(chromosomes_size, dtype=torch.long, device=device)
        self.profit = torch.zeros((chromosomes_size,), dtype=torch.float32, device=device)
        self.last_timestamp = None
        self.sum_returns = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.count_returns = torch.zeros(chromosomes_size, device=device, dtype=torch.int32)
        self.sum_sq_returns = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.sum_neg = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.count_neg = torch.zeros(chromosomes_size, device=device, dtype=torch.int32)
        self.sum_sq_neg = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.total_profit_agg = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.total_loss_agg = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.count_wins = torch.zeros(chromosomes_size, device=device, dtype=torch.int32)
        self.cum_sum = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.running_max = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.max_drawdown = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.compound_value = torch.ones(chromosomes_size, device=device, dtype=torch.float32)

    def _apply_funding_fees(self, current_timestamp, funding_rate):
        if self.last_timestamp is None:
            return
        for hour in self.funding_fee_hours: # [수정] 인스턴스 변수 사용
            funding_time_today = self.last_timestamp.normalize().replace(hour=hour)
            funding_time_next_day = funding_time_today + pd.Timedelta(days=1)
            for ft in [funding_time_today, funding_time_next_day]:
                if self.last_timestamp < ft <= current_timestamp:
                    active_pos_mask = self.pos_list != 0
                    if not active_pos_mask.any():
                        continue
                    funding_pnl = funding_rate * self.leverage_ratio.float()
                    long_mask = (self.pos_list == 2) & active_pos_mask
                    short_mask = (self.pos_list == 1) & active_pos_mask
                    self.profit[long_mask] -= funding_pnl[long_mask] * self.enter_ratio[long_mask]
                    self.profit[short_mask] += funding_pnl[short_mask] * self.enter_ratio[short_mask]
                    break

    def _check_liquidation(self, curr_price):
        active_pos_mask = self.pos_list != 0
        if not active_pos_mask.any():
            return
        pos_price = self.price_list[active_pos_mask]
        leverage = self.leverage_ratio[active_pos_mask].float()
        enter_ratio = self.enter_ratio[active_pos_mask]
        pnl_percent = torch.zeros_like(pos_price)
        long_mask_local = self.pos_list[active_pos_mask] == 2
        short_mask_local = self.pos_list[active_pos_mask] == 1
        pnl_percent[long_mask_local] = (curr_price - pos_price[long_mask_local]) / pos_price[long_mask_local]
        pnl_percent[short_mask_local] = (pos_price[short_mask_local] - curr_price) / pos_price[short_mask_local]
        margin_balance = enter_ratio * (1 + pnl_percent * leverage)
        # [수정] 인스턴스 변수 사용
        maintenance_margin = (enter_ratio * leverage) * self.maintenance_margin_rate
        liquidation_mask_local = margin_balance <= maintenance_margin
        if not liquidation_mask_local.any():
            return
        active_indices = torch.where(active_pos_mask)[0]
        liquidated_indices = active_indices[liquidation_mask_local]
        self.profit[liquidated_indices] -= self.enter_ratio[liquidated_indices]
        self.pos_list[liquidated_indices] = 0
        self.price_list[liquidated_indices] = -1.0
        self.leverage_ratio[liquidated_indices] = -1
        self.enter_ratio[liquidated_indices] = -1.0
        self.additional_count[liquidated_indices] = 0
        self.holding_period[liquidated_indices] = 0

    def _execute_actions(self, actions_tensor, curr_close, curr_high, curr_low):
        # [수정] 인스턴스 변수 사용
        buy_exec_price = curr_close * (1 + self.fixed_slippage_rate)
        sell_exec_price = curr_close * (1 - self.fixed_slippage_rate)
        currently_hold = (self.pos_list == 0)
        currently_short = (self.pos_list == 1)
        currently_long = (self.pos_list == 2)
        for action_type, target_pos, pos_cond, exec_price in [(ACTION_NEW_LONG, 2, currently_hold, buy_exec_price), (ACTION_NEW_SHORT, 1, currently_hold, sell_exec_price)]:
            action_mask = (actions_tensor[:, 0] == action_type) & pos_cond
            if action_mask.any():
                enter_ratio = actions_tensor[action_mask, 1]
                leverage = actions_tensor[action_mask, 2].long()
                # [수정] 인스턴스 변수 사용
                fee = enter_ratio * leverage.float() * self.taker_fee_rate
                self.profit[action_mask] -= fee
                self.pos_list[action_mask] = target_pos
                self.price_list[action_mask] = exec_price
                self.leverage_ratio[action_mask] = leverage
                self.enter_ratio[action_mask] = enter_ratio
                self.additional_count[action_mask] = 0
        action_mask = (actions_tensor[:, 0] == ACTION_CLOSE_ALL) & (self.pos_list != 0)
        if action_mask.any():
            long_close_mask = action_mask & currently_long
            if long_close_mask.any():
                entry_price = self.price_list[long_close_mask]
                leverage = self.leverage_ratio[long_close_mask].float()
                pnl = ((sell_exec_price - entry_price) / entry_price) * leverage * self.enter_ratio[long_close_mask]
                # [수정] 인스턴스 변수 사용
                fee = self.enter_ratio[long_close_mask] * leverage * self.taker_fee_rate
                self.profit[long_close_mask] += (pnl - fee)
            short_close_mask = action_mask & currently_short
            if short_close_mask.any():
                entry_price = self.price_list[short_close_mask]
                leverage = self.leverage_ratio[short_close_mask].float()
                pnl = ((entry_price - buy_exec_price) / entry_price) * leverage * self.enter_ratio[short_close_mask]
                # [수정] 인스턴스 변수 사용
                fee = self.enter_ratio[short_close_mask] * leverage * self.taker_fee_rate
                self.profit[short_close_mask] += (pnl - fee)
            self.pos_list[action_mask] = 0
            self.price_list[action_mask] = -1.0
            self.leverage_ratio[action_mask] = -1
            self.enter_ratio[action_mask] = -1.0
            self.additional_count[action_mask] = 0
        action_mask = (actions_tensor[:, 0] == ACTION_CLOSE_PARTIAL) & (self.pos_list != 0)
        if action_mask.any():
            close_ratio = actions_tensor[action_mask, 1]
            long_p_close_mask = action_mask & currently_long
            if long_p_close_mask.any():
                entry_price = self.price_list[long_p_close_mask]
                leverage = self.leverage_ratio[long_p_close_mask].float()
                ratio = close_ratio[currently_long[action_mask]]
                closed_margin = self.enter_ratio[long_p_close_mask] * ratio
                pnl = ((sell_exec_price - entry_price) / entry_price) * leverage * closed_margin
                # [수정] 인스턴스 변수 사용
                fee = closed_margin * leverage * self.taker_fee_rate
                self.profit[long_p_close_mask] += (pnl - fee)
                self.enter_ratio[long_p_close_mask] -= closed_margin
            short_p_close_mask = action_mask & currently_short
            if short_p_close_mask.any():
                entry_price = self.price_list[short_p_close_mask]
                leverage = self.leverage_ratio[short_p_close_mask].float()
                ratio = close_ratio[currently_short[action_mask]]
                closed_margin = self.enter_ratio[short_p_close_mask] * ratio
                pnl = ((entry_price - buy_exec_price) / entry_price) * leverage * closed_margin
                # [수정] 인스턴스 변수 사용
                fee = closed_margin * leverage * self.taker_fee_rate
                self.profit[short_p_close_mask] += (pnl - fee)
                self.enter_ratio[short_p_close_mask] -= closed_margin
        action_mask = (actions_tensor[:, 0] == ACTION_ADD_POSITION) & (self.pos_list != 0)
        if action_mask.any():
            add_ratio = actions_tensor[action_mask, 1]
            for pos_type, pos_cond, exec_price in [(2, currently_long, buy_exec_price), (1, currently_short, sell_exec_price)]:
                add_mask = action_mask & pos_cond
                if add_mask.any():
                    ratio = add_ratio[pos_cond[action_mask]]
                    # [수정] 인스턴스 변수 사용
                    fee = ratio * self.leverage_ratio[add_mask].float() * self.taker_fee_rate
                    self.profit[add_mask] -= fee
                    old_price = self.price_list[add_mask]
                    old_ratio = self.enter_ratio[add_mask]
                    new_avg_price = (old_price * old_ratio + exec_price * ratio) / (old_ratio + ratio + 1e-9)
                    self.price_list[add_mask] = new_avg_price
                    self.enter_ratio[add_mask] += ratio
                    self.additional_count[add_mask] += 1
        action_mask = (actions_tensor[:, 0] == ACTION_FLIP_POSITION) & (self.pos_list != 0)
        if action_mask.any():
            new_enter_ratio = actions_tensor[action_mask, 1]
            new_leverage = actions_tensor[action_mask, 2].long()
            flip_to_short_mask = action_mask & currently_long
            if flip_to_short_mask.any():
                entry_price = self.price_list[flip_to_short_mask]
                leverage = self.leverage_ratio[flip_to_short_mask].float()
                pnl = ((sell_exec_price - entry_price) / entry_price) * leverage * self.enter_ratio[flip_to_short_mask]
                # [수정] 인스턴스 변수 사용
                fee_close = self.enter_ratio[flip_to_short_mask] * leverage * self.taker_fee_rate
                self.profit[flip_to_short_mask] += (pnl - fee_close)
                ratio = new_enter_ratio[currently_long[action_mask]]
                lev = new_leverage[currently_long[action_mask]]
                # [수정] 인스턴스 변수 사용
                fee_open = ratio * lev.float() * self.taker_fee_rate
                self.profit[flip_to_short_mask] -= fee_open
                self.pos_list[flip_to_short_mask] = 1
                self.price_list[flip_to_short_mask] = sell_exec_price
                self.enter_ratio[flip_to_short_mask] = ratio
                self.leverage_ratio[flip_to_short_mask] = lev
                self.additional_count[flip_to_short_mask] = 0
            flip_to_long_mask = action_mask & currently_short
            if flip_to_long_mask.any():
                entry_price = self.price_list[flip_to_long_mask]
                leverage = self.leverage_ratio[flip_to_long_mask].float()
                pnl = ((entry_price - buy_exec_price) / entry_price) * leverage * self.enter_ratio[flip_to_long_mask]
                # [수정] 인스턴스 변수 사용
                fee_close = self.enter_ratio[flip_to_long_mask] * leverage * self.taker_fee_rate
                self.profit[flip_to_long_mask] += (pnl - fee_close)
                ratio = new_enter_ratio[currently_short[action_mask]]
                lev = new_leverage[currently_short[action_mask]]
                # [수정] 인스턴스 변수 사용
                fee_open = ratio * lev.float() * self.taker_fee_rate
                self.profit[flip_to_long_mask] -= fee_open
                self.pos_list[flip_to_long_mask] = 2
                self.price_list[flip_to_long_mask] = buy_exec_price
                self.enter_ratio[flip_to_long_mask] = ratio
                self.leverage_ratio[flip_to_long_mask] = lev
                self.additional_count[flip_to_long_mask] = 0

    def step(self, market_data, predicted_actions):
        current_timestamp = market_data.name
        curr_close = torch.tensor(market_data['Close'], dtype=torch.float32, device=self.device)
        curr_high = torch.tensor(market_data['High'], dtype=torch.float32, device=self.device)
        curr_low = torch.tensor(market_data['Low'], dtype=torch.float32, device=self.device)
        funding_rate = torch.tensor(market_data['fundingRate'], dtype=torch.float32, device=self.device)
        self._apply_funding_fees(current_timestamp, funding_rate)
        liquidation_price_long = curr_low
        liquidation_price_short = curr_high
        self._check_liquidation(liquidation_price_long)
        self._check_liquidation(liquidation_price_short)
        self._execute_actions(predicted_actions, curr_close, curr_high, curr_low)
        self.holding_period[self.pos_list != 0] += 1
        self.holding_period[self.pos_list == 0] = 0
        non_zero_mask = self.profit != 0
        if non_zero_mask.any():
            self.sum_returns[non_zero_mask] += self.profit[non_zero_mask]
            self.count_returns[non_zero_mask] += 1
            self.sum_sq_returns[non_zero_mask] += self.profit[non_zero_mask] ** 2
            neg_mask = self.profit < 0
            self.sum_neg[neg_mask] += self.profit[neg_mask]
            self.count_neg[neg_mask] += 1
            self.sum_sq_neg[neg_mask] += self.profit[neg_mask] ** 2
            pos_mask = self.profit > 0
            self.total_profit_agg[pos_mask] += self.profit[pos_mask]
            self.total_loss_agg[neg_mask] += -self.profit[neg_mask]
            self.count_wins[pos_mask] += 1
            self.cum_sum += self.profit
            self.running_max = torch.maximum(self.running_max, self.cum_sum)
            current_drawdown = self.running_max - self.cum_sum
            self.max_drawdown = torch.maximum(self.max_drawdown, current_drawdown)
            self.compound_value[non_zero_mask] *= (1 + self.profit[non_zero_mask] / 100.0)
        self.profit.zero_()
        self.last_timestamp = current_timestamp

    def get_final_metrics(self, minimum_date=40):
        count_returns_f = self.count_returns.float()
        mean_returns = torch.where(count_returns_f > 0, self.sum_returns / count_returns_f, torch.full_like(self.sum_returns, -1e9))
        profit_factors = torch.where(self.total_loss_agg > 0, self.total_profit_agg / (self.total_loss_agg + 1e-9), torch.full_like(self.total_profit_agg, -1e9))
        win_rates = torch.where(count_returns_f > 0, self.count_wins.float() / count_returns_f, torch.full_like(count_returns_f, -1e9))
        invalid_mask = self.count_returns < minimum_date
        mean_returns[invalid_mask] = -1e9
        profit_factors[invalid_mask] = -1e9
        win_rates[invalid_mask] = -1e9
        self.max_drawdown[invalid_mask] = 1e9
        self.compound_value[invalid_mask] = -1e9
        metrics = torch.stack([mean_returns, profit_factors, win_rates, self.max_drawdown, self.compound_value], dim=1)
        return metrics.cpu().numpy()

# ##############################################################################
#           유틸리티 및 피트니스 계산 래퍼 함수 (수정됨)
# ##############################################################################

# --- [수정 없음] 전역 기울기 계산 비활성화 ---
torch.set_grad_enabled(False)

def calculate_fitness(metrics, weights):
    """[수정] 피트니스 가중치를 외부 인자로 받도록 변경된 피트니스 계산 함수"""
    chromosomes_size = len(metrics)
    def normalize_metric(metric, higher_is_better=True):
        valid_mask = ~np.isin(metric, [-1e9, 1e9]) & ~np.isnan(metric)
        if not np.any(valid_mask):
            return np.zeros_like(metric)
        valid_metric = metric[valid_mask]
        min_val, max_val = np.min(valid_metric), np.max(valid_metric)
        if min_val == max_val:
            normalized = np.ones_like(metric) * 0.5
        else:
            if higher_is_better:
                normalized = (metric - min_val) / (max_val - min_val + 1e-9)
            else:
                normalized = (max_val - metric) / (max_val - min_val + 1e-9)
        normalized[~valid_mask] = 0.0
        return normalized
    normalized_metrics = np.zeros_like(metrics)
    normalized_metrics[:, 0] = normalize_metric(metrics[:, 0], higher_is_better=True)
    normalized_metrics[:, 1] = normalize_metric(metrics[:, 1], higher_is_better=True)
    normalized_metrics[:, 2] = normalize_metric(metrics[:, 2], higher_is_better=True)
    normalized_metrics[:, 3] = normalize_metric(metrics[:, 3], higher_is_better=False)
    normalized_metrics[:, 4] = normalize_metric(metrics[:, 4], higher_is_better=True)
    
    # [수정] 인자로 받은 가중치 사용
    fitness_values = np.sum(normalized_metrics * np.array(weights), axis=1)
    invalid_trade_mask = metrics[:, 0] == -1e9
    fitness_values[invalid_trade_mask] = -1.0
    return fitness_values


def fitness_fn(
    population: GATreePop,
    data: pd.DataFrame,
    all_feature_names: list,
    start_data_cnt: int,
    stop_data_cnt: int,
    device: str,
    evaluation_config: dict # [신규] 평가 관련 전체 설정을 받음
):
    """
    [수정] evaluation_config를 받아 TradingEnvironment와 get_final_metrics에
    필요한 설정값을 전달하는 메인 피트니스 평가 래퍼 함수.
    """
    env_config = evaluation_config['simulation_env']
    fitness_cfg = evaluation_config['fitness_function']

    # [수정] TradingEnvironment 생성 시 env_config 주입
    environment = TradingEnvironment(
        chromosomes_size=population.pop_size, 
        env_config=env_config, 
        device=device
    )
    
    pbar = tqdm(range(start_data_cnt, stop_data_cnt), desc="Fitness Simulation (Time-driven)")
    
    feature_data = data[all_feature_names]

    for entry_index in pbar:
        market_data_row = data.iloc[entry_index]
        feature_values_series = feature_data.iloc[entry_index]
        
        current_positions_str = [INT_TO_POSITION_MAP[pos.item()] for pos in environment.pos_list]
        
        predicted_actions = predict_population_cuda(
            population=population,
            feature_values=feature_values_series,
            current_positions=current_positions_str,
            device=device
        )
        
        if predicted_actions is None:
            raise RuntimeError("CUDA prediction failed. Check if gatree_cuda module is compiled and available.")
            
        environment.step(market_data_row, predicted_actions)

    pbar.close()
    
    # [수정] get_final_metrics 호출 시 minimum_trades 값을 config에서 가져옴
    final_metrics = environment.get_final_metrics(minimum_date=fitness_cfg['minimum_trades'])
    return final_metrics

# ##############################################################################
#           메인 학습/테스트 루프 (수정됨)
# ##############################################################################

def generation_valid(
    data_1m: pd.DataFrame,
    evolution,
    skip_data_cnt: int,
    valid_skip_data_cnt: int,
    chromosomes_size: int,
    gen_loop: int,
    best_size: int,
    elite_size: int,
    device: str,
    warming_step: int,
    evaluation_config: dict, # [신규] 평가 설정
    output_dir: str,         # [신규] 체크포인트 저장을 위한 경로
    best_profit=None,
    best_chromosomes=None,
    start_gen: int = 0
):
    """[수정] evaluation_config와 output_dir을 인자로 받아 사용하는 메인 학습 루프"""
    
    best_profit = best_profit
    best_chromosomes = best_chromosomes
    # [수정] 체크포인트 저장 경로를 output_dir 기준으로 설정
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if not isinstance(data_1m.index, pd.DatetimeIndex):
        data_1m.index = pd.to_datetime(data_1m.index)

    population = evolution.population
    all_feature_names = population.all_features
    # [신규] 피트니스 가중치를 config에서 미리 추출
    fitness_weights = evaluation_config['fitness_function']['weights']

    for gen_idx in range(start_gen, gen_loop):
        print(f'generation {gen_idx}: ')
        
        # [수정] fitness_fn 호출 시 evaluation_config 전달
        train_metrics = fitness_fn(
            population=population,
            data=data_1m,
            all_feature_names=all_feature_names,
            start_data_cnt=skip_data_cnt,
            stop_data_cnt=valid_skip_data_cnt,
            device=device,
            evaluation_config=evaluation_config
        )
        
        if warming_step <= gen_idx:
            valid_metrics_np = train_metrics[:elite_size]
            valid_index = np.where((valid_metrics_np[:, 4] > 6.0) & (valid_metrics_np[:, 3] < 0.6))[0]
            
            if len(valid_index) > 0:
                new_best_metrics = torch.from_numpy(valid_metrics_np[valid_index])
                new_best_chromosomes = population.population_tensor[valid_index].clone()

                if best_chromosomes is None:
                    best_profit = new_best_metrics
                    best_chromosomes = new_best_chromosomes
                else:
                    best_profit = torch.cat([best_profit, new_best_metrics])
                    best_chromosomes = torch.cat([best_chromosomes, new_best_chromosomes])

                if len(best_chromosomes) > best_size:
                    print('check_discard')
                    # [수정] calculate_fitness 호출 시 weights 전달
                    best_fitness_scores = calculate_fitness(best_profit.numpy(), weights=fitness_weights)
                    top_indices = torch.topk(torch.from_numpy(best_fitness_scores), k=best_size).indices
                    best_profit = best_profit[top_indices]
                    best_chromosomes = best_chromosomes[top_indices]
        
        gen_data = {
            "generation": gen_idx,
            "population_state_dict": population.population_tensor,
            "best_profit": best_profit,
            "best_chromosomes": best_chromosomes,
        }
        
        # [수정] calculate_fitness 호출 시 weights 전달
        train_fitness = calculate_fitness(train_metrics, weights=fitness_weights)
        torch.save(gen_data, os.path.join(checkpoint_dir, f'generation_{gen_idx}.pt'))
        
        evolution.evolve(torch.from_numpy(train_fitness).to(device))
        
    return best_chromosomes, best_profit


def generation_test(
    data_1m: pd.DataFrame,
    population: GATreePop,
    start_data_cnt: int,
    end_data_cnt: int,
    device: str,
    evaluation_config: dict # [신규] 평가 설정
):
    """[수정] 저장된 최적의 GATree 집단을 사용하여 테스트를 수행 (evaluation_config 사용)"""

    if not isinstance(data_1m.index, pd.DatetimeIndex):
        data_1m.index = pd.to_datetime(data_1m.index)

    all_feature_names = population.all_features
    
    # [수정] fitness_fn 호출 시 evaluation_config 전달
    test_metrics = fitness_fn(
        population=population,
        data=data_1m,
        all_feature_names=all_feature_names,
        start_data_cnt=start_data_cnt,
        stop_data_cnt=end_data_cnt,
        device=device,
        evaluation_config=evaluation_config
    )
        
    return test_metrics