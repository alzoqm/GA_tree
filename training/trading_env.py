# /training/trading_env.py

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

from models.model import GATreePop
from models.constants import (
    ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_CLOSE_ALL,
    ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION, ACTION_FLIP_POSITION
)
from training.predictor import build_adjacency_list_cuda, predict_population_cuda

# ##############################################################################
#           거래 환경 및 시뮬레이션 핵심 로직 (수정됨)
# ##############################################################################

# 포지션 정수 -> 문자열 변환 맵
INT_TO_POSITION_MAP = {
    0: 'HOLD',
    1: 'SHORT',
    2: 'LONG'
}


class TradingEnvironment:
    """
    거래 환경 설정을 생성자에서 받아 초기화하는 거래 시뮬레이션 환경 클래스.
    """
    def __init__(self, chromosomes_size, env_config, device='cpu'):
        self.device = device
        self.chromosomes_size = chromosomes_size
        
        self.taker_fee_rate = env_config['taker_fee_rate']
        self.maintenance_margin_rate = env_config['maintenance_margin_rate']
        self.fixed_slippage_rate = env_config['fixed_slippage_rate']
        self.funding_fee_hours = {0, 8, 16}
        self.min_additional_enter_ratio = env_config.get('min_additional_enter_ratio', 0.05)
        self.max_additional_entries = env_config.get('max_additional_entries', 1e9)

        self.pos_list = torch.zeros(chromosomes_size, dtype=torch.long, device=device)
        self.price_list = torch.full((chromosomes_size,), -1.0, dtype=torch.float32, device=device)
        self.leverage_ratio = torch.full((chromosomes_size,), -1, dtype=torch.long, device=device)
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
        for hour in self.funding_fee_hours:
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

    def _check_liquidation(self, curr_low, curr_high):
        # 1. 롱 포지션 청산 검사 (가격 하락 시 위험 -> curr_low 기준)
        long_mask = self.pos_list == 2
        if long_mask.any():
            pos_price = self.price_list[long_mask]
            leverage = self.leverage_ratio[long_mask].float()
            enter_ratio = self.enter_ratio[long_mask]

            pnl_percent = (curr_low - pos_price) / pos_price
            margin_balance = enter_ratio * (1 + pnl_percent * leverage)
            maintenance_margin = (enter_ratio * leverage) * self.maintenance_margin_rate
            
            liquidation_mask_local = margin_balance <= maintenance_margin
            if liquidation_mask_local.any():
                long_indices = torch.where(long_mask)[0]
                liquidated_indices = long_indices[liquidation_mask_local]
                
                self.profit[liquidated_indices] -= self.enter_ratio[liquidated_indices]
                self.pos_list[liquidated_indices] = 0
                self.price_list[liquidated_indices] = -1.0
                self.leverage_ratio[liquidated_indices] = -1
                self.enter_ratio[liquidated_indices] = -1.0
                self.additional_count[liquidated_indices] = 0
                self.holding_period[liquidated_indices] = 0

        # 2. 숏 포지션 청산 검사 (가격 상승 시 위험 -> curr_high 기준)
        short_mask = self.pos_list == 1
        if short_mask.any():
            pos_price = self.price_list[short_mask]
            leverage = self.leverage_ratio[short_mask].float()
            enter_ratio = self.enter_ratio[short_mask]
            
            pnl_percent = (pos_price - curr_high) / pos_price
            margin_balance = enter_ratio * (1 + pnl_percent * leverage)
            maintenance_margin = (enter_ratio * leverage) * self.maintenance_margin_rate

            liquidation_mask_local = margin_balance <= maintenance_margin
            if liquidation_mask_local.any():
                short_indices = torch.where(short_mask)[0]
                liquidated_indices = short_indices[liquidation_mask_local]

                self.profit[liquidated_indices] -= self.enter_ratio[liquidated_indices]
                self.pos_list[liquidated_indices] = 0
                self.price_list[liquidated_indices] = -1.0
                self.leverage_ratio[liquidated_indices] = -1
                self.enter_ratio[liquidated_indices] = -1.0
                self.additional_count[liquidated_indices] = 0
                self.holding_period[liquidated_indices] = 0
    
    # [수정 시작] _execute_actions 함수 전체가 수정되었습니다.
    def _execute_actions(self, actions_tensor, exec_base_price, curr_high, curr_low):
        buy_exec_price = exec_base_price * (1 + self.fixed_slippage_rate)
        sell_exec_price = exec_base_price * (1 - self.fixed_slippage_rate)

        currently_hold = (self.pos_list == 0)
        currently_short = (self.pos_list == 1)
        currently_long = (self.pos_list == 2)

        for action_type, target_pos, pos_cond, exec_price in [(ACTION_NEW_LONG, 2, currently_hold, buy_exec_price), (ACTION_NEW_SHORT, 1, currently_hold, sell_exec_price)]:
            action_mask = (actions_tensor[:, 0] == action_type) & pos_cond
            if action_mask.any():
                enter_ratio = actions_tensor[action_mask, 1]
                leverage = actions_tensor[action_mask, 2].long()
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
                fee = self.enter_ratio[long_close_mask] * leverage * self.taker_fee_rate
                self.profit[long_close_mask] += (pnl - fee)
            short_close_mask = action_mask & currently_short
            if short_close_mask.any():
                entry_price = self.price_list[short_close_mask]
                leverage = self.leverage_ratio[short_close_mask].float()
                pnl = ((entry_price - buy_exec_price) / entry_price) * leverage * self.enter_ratio[short_close_mask]
                fee = self.enter_ratio[short_close_mask] * leverage * self.taker_fee_rate
                self.profit[short_close_mask] += (pnl - fee)
            self.pos_list[action_mask] = 0
            self.price_list[action_mask] = -1.0
            self.leverage_ratio[action_mask] = -1
            self.enter_ratio[action_mask] = -1.0
            self.additional_count[action_mask] = 0

        action_mask = (actions_tensor[:, 0] == ACTION_CLOSE_PARTIAL) & (self.pos_list != 0)
        if action_mask.any():
            long_p_close_mask = action_mask & currently_long
            if long_p_close_mask.any():
                ratio = actions_tensor[long_p_close_mask, 1] # 수정된 부분
                entry_price = self.price_list[long_p_close_mask]
                leverage = self.leverage_ratio[long_p_close_mask].float()
                closed_margin = self.enter_ratio[long_p_close_mask] * ratio
                pnl = ((sell_exec_price - entry_price) / entry_price) * leverage * closed_margin
                fee = closed_margin * leverage * self.taker_fee_rate
                self.profit[long_p_close_mask] += (pnl - fee)
                self.enter_ratio[long_p_close_mask] -= closed_margin
            short_p_close_mask = action_mask & currently_short
            if short_p_close_mask.any():
                ratio = actions_tensor[short_p_close_mask, 1] # 수정된 부분
                entry_price = self.price_list[short_p_close_mask]
                leverage = self.leverage_ratio[short_p_close_mask].float()
                closed_margin = self.enter_ratio[short_p_close_mask] * ratio
                pnl = ((entry_price - buy_exec_price) / entry_price) * leverage * closed_margin
                fee = closed_margin * leverage * self.taker_fee_rate
                self.profit[short_p_close_mask] += (pnl - fee)
                self.enter_ratio[short_p_close_mask] -= closed_margin
        
        action_mask = (actions_tensor[:, 0] == ACTION_ADD_POSITION) & (self.pos_list != 0)
        if action_mask.any():
            for pos_type, pos_cond, exec_price in [(2, currently_long, buy_exec_price), (1, currently_short, sell_exec_price)]:
                add_mask = action_mask & pos_cond & (self.additional_count < self.max_additional_entries)
                if add_mask.any():
                    current_enter_ratio = self.enter_ratio[add_mask]
                    current_requested_ratio = actions_tensor[add_mask, 1] # 수정된 부분
                    
                    intended_add_amount = current_requested_ratio
                    remaining_capital = 1.0 - current_enter_ratio
                    actual_add_ratio = torch.min(intended_add_amount, remaining_capital)
                    valid_add_mask = actual_add_ratio > self.min_additional_enter_ratio
                    
                    if valid_add_mask.any():
                        final_add_mask = torch.where(add_mask)[0][valid_add_mask]
                        final_add_ratio = actual_add_ratio[valid_add_mask]

                        fee = final_add_ratio * self.leverage_ratio[final_add_mask].float() * self.taker_fee_rate
                        self.profit[final_add_mask] -= fee
                        
                        old_price = self.price_list[final_add_mask]
                        old_ratio = self.enter_ratio[final_add_mask]
                        new_avg_price = (old_price * old_ratio + exec_price * final_add_ratio) / (old_ratio + final_add_ratio + 1e-9)
                        
                        self.price_list[final_add_mask] = new_avg_price
                        self.enter_ratio[final_add_mask] += final_add_ratio
                        self.additional_count[final_add_mask] += 1

        action_mask = (actions_tensor[:, 0] == ACTION_FLIP_POSITION) & (self.pos_list != 0)
        if action_mask.any():
            flip_to_short_mask = action_mask & currently_long
            if flip_to_short_mask.any():
                ratio = actions_tensor[flip_to_short_mask, 1] # 수정된 부분
                lev = actions_tensor[flip_to_short_mask, 2].long() # 수정된 부분
                entry_price = self.price_list[flip_to_short_mask]
                leverage = self.leverage_ratio[flip_to_short_mask].float()
                pnl = ((sell_exec_price - entry_price) / entry_price) * leverage * self.enter_ratio[flip_to_short_mask]
                fee_close = self.enter_ratio[flip_to_short_mask] * leverage * self.taker_fee_rate
                self.profit[flip_to_short_mask] += (pnl - fee_close)
                fee_open = ratio * lev.float() * self.taker_fee_rate
                self.profit[flip_to_short_mask] -= fee_open
                self.pos_list[flip_to_short_mask] = 1
                self.price_list[flip_to_short_mask] = sell_exec_price
                self.enter_ratio[flip_to_short_mask] = ratio
                self.leverage_ratio[flip_to_short_mask] = lev
                self.additional_count[flip_to_short_mask] = 0
            flip_to_long_mask = action_mask & currently_short
            if flip_to_long_mask.any():
                ratio = actions_tensor[flip_to_long_mask, 1] # 수정된 부분
                lev = actions_tensor[flip_to_long_mask, 2].long() # 수정된 부분
                entry_price = self.price_list[flip_to_long_mask]
                leverage = self.leverage_ratio[flip_to_long_mask].float()
                pnl = ((entry_price - buy_exec_price) / entry_price) * leverage * self.enter_ratio[flip_to_long_mask]
                fee_close = self.enter_ratio[flip_to_long_mask] * leverage * self.taker_fee_rate
                self.profit[flip_to_long_mask] += (pnl - fee_close)
                fee_open = ratio * lev.float() * self.taker_fee_rate
                self.profit[flip_to_long_mask] -= fee_open
                self.pos_list[flip_to_long_mask] = 2
                self.price_list[flip_to_long_mask] = buy_exec_price
                self.enter_ratio[flip_to_long_mask] = ratio
                self.leverage_ratio[flip_to_long_mask] = lev
                self.additional_count[flip_to_long_mask] = 0
    # [수정 종료]

    # [수정 시작] 보고서 1번 항목: step 함수 시그니처 변경
    def step(self, market_data, next_open_price, predicted_actions):
        current_timestamp = market_data.name
        # curr_close는 더 이상 체결 가격으로 사용되지 않음
        curr_high = torch.tensor(market_data['High'], dtype=torch.float32, device=self.device)
        curr_low = torch.tensor(market_data['Low'], dtype=torch.float32, device=self.device)
        funding_rate = torch.tensor(market_data['fundingRate'], dtype=torch.float32, device=self.device)
        
        self._apply_funding_fees(current_timestamp, funding_rate)
        self._check_liquidation(curr_low, curr_high)
        
        # 체결 기준 가격으로 next_open_price 전달
        self._execute_actions(predicted_actions, next_open_price, curr_high, curr_low)
    # [수정 종료]
        
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
            self.compound_value[non_zero_mask] *= (1 + self.profit[non_zero_mask])
        self.profit.zero_()
        self.last_timestamp = current_timestamp

    def get_final_metrics(self, minimum_date=40):
        count_returns_f = self.count_returns.float()
        mean_returns = torch.where(count_returns_f > 0, self.sum_returns / count_returns_f, torch.full_like(self.sum_returns, -1e9))
        
        # [수정 시작] 보고서 3번 항목: Profit Factor 지표 왜곡 방지
        # 손실이 0일 경우를 대비해 PF의 최대값을 100.0으로 설정하고, 계산된 값에도 상한을 적용
        profit_factors = torch.full_like(self.total_profit_agg, 100.0)
        has_loss_mask = self.total_loss_agg > 0
        if has_loss_mask.any():
            calculated_pf = self.total_profit_agg[has_loss_mask] / (self.total_loss_agg[has_loss_mask] + 1e-9)
            profit_factors[has_loss_mask] = torch.clamp(calculated_pf, max=100.0)
        # [수정 종료]

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
#           유틸리티 및 피트니스 계산 래퍼 함수 (수정 없음)
# ##############################################################################

torch.set_grad_enabled(False)

def calculate_fitness(metrics, weights):
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
    
    fitness_values = np.sum(normalized_metrics * np.array(weights), axis=1)
    invalid_trade_mask = metrics[:, 0] == -1e9
    fitness_values[invalid_trade_mask] = -1.0
    return fitness_values


def fitness_fn(
    population: GATreePop,
    data: pd.DataFrame,
    all_feature_names: list,
    adj_offsets: torch.Tensor,
    adj_indices: torch.Tensor,
    start_data_cnt: int,
    stop_data_cnt: int,
    device: str,
    evaluation_config: dict
):
    """
    [수정 없음]
    """
    env_config = evaluation_config['simulation_env']
    fitness_cfg = evaluation_config['fitness_function']

    environment = TradingEnvironment(
        chromosomes_size=population.pop_size, 
        env_config=env_config, 
        device=device
    )
    
    # [수정 시작] 보고서 1번 항목: 다음 캔들 Open가로 거래하기 위해 루프 범위 수정
    pbar = tqdm(range(start_data_cnt, stop_data_cnt - 1), desc="Fitness Simulation (Time-driven)")
    # [수정 종료]
    
    feature_data = data[all_feature_names]

    for entry_index in pbar:
        # [수정 시작] 보고서 1번 항목: 다음 캔들 Open 가격 조회 및 전달
        market_data_row = data.iloc[entry_index]
        next_open_price = torch.tensor(data['Open'].iloc[entry_index + 1], dtype=torch.float32, device=device)
        feature_values_series = feature_data.iloc[entry_index]
        # [수정 종료]
        
        current_positions_str = [INT_TO_POSITION_MAP[pos.item()] for pos in environment.pos_list]
        
        predicted_actions = predict_population_cuda(
            population=population,
            feature_values=feature_values_series,
            current_positions=current_positions_str,
            adj_offsets=adj_offsets,
            adj_indices=adj_indices,
            device=device
        )
        
        if predicted_actions is None:
            raise RuntimeError("CUDA prediction failed. Check if gatree_cuda module is compiled and available.")
            
        # [수정 시작] 보고서 1번 항목: step 함수에 next_open_price 전달
        environment.step(market_data_row, next_open_price, predicted_actions)
        # [수정 종료]

    pbar.close()
    
    final_metrics = environment.get_final_metrics(minimum_date=fitness_cfg['minimum_trades'])
    return final_metrics

# ##############################################################################
#           메인 학습/테스트 루프 (수정 없음)
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
    evaluation_config: dict,
    output_dir: str,
    best_profit=None,
    best_chromosomes=None,
    start_gen: int = 0
):
    """[수정 없음]"""
    
    best_profit = best_profit
    best_chromosomes = best_chromosomes
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if not isinstance(data_1m.index, pd.DatetimeIndex):
        data_1m.index = pd.to_datetime(data_1m.index)

    population = evolution.population
    all_feature_names = population.all_features
    fitness_weights = evaluation_config['fitness_function']['weights']

    for gen_idx in range(start_gen, gen_loop):
        print(f'generation {gen_idx}: ')
        
        adj_offsets, adj_indices = build_adjacency_list_cuda(population)
        if adj_offsets is None:
            raise RuntimeError("Failed to build adjacency list on GPU.")

        train_metrics = fitness_fn(
            population=population,
            data=data_1m,
            all_feature_names=all_feature_names,
            adj_offsets=adj_offsets,
            adj_indices=adj_indices,
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
        
        train_fitness = calculate_fitness(train_metrics, weights=fitness_weights)
        torch.save(gen_data, os.path.join(checkpoint_dir, f'chrom_generation_{gen_idx}.pt'))
        evolution.population.save(os.path.join(checkpoint_dir, f'generation_{gen_idx}.pt'))
        
        evolution.evolve(torch.from_numpy(train_fitness).to(device))
        population.population_tensor = population.population_tensor.to(device)
        population.reorganize_nodes()
        
    return best_chromosomes, best_profit


def generation_test(
    data_1m: pd.DataFrame,
    population: GATreePop,
    start_data_cnt: int,
    end_data_cnt: int,
    device: str,
    evaluation_config: dict
):
    """[수정 없음]"""

    if not isinstance(data_1m.index, pd.DatetimeIndex):
        data_1m.index = pd.to_datetime(data_1m.index)

    all_feature_names = population.all_features
    
    adj_offsets, adj_indices = build_adjacency_list_cuda(population)
    if adj_offsets is None:
        raise RuntimeError("Failed to build adjacency list on GPU for test.")
        
    test_metrics = fitness_fn(
        population=population,
        data=data_1m,
        all_feature_names=all_feature_names,
        adj_offsets=adj_offsets,
        adj_indices=adj_indices,
        start_data_cnt=start_data_cnt,
        stop_data_cnt=end_data_cnt,
        device=device,
        evaluation_config=evaluation_config
    )
        
    return test_metrics