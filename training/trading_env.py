import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from strategies import BBStrategy
from icecream import ic
from copy import deepcopy
import pandas as pd # 시간 처리를 위해 추가

# ##############################################################################
# 수정된 보고서 기반 신규/수정 코드 영역
# ##############################################################################

# --- 시뮬레이션 환경 설정을 위한 상수 정의 ---
# 바이낸스 USDT-M 선물 시장가(Taker) 수수료 (0.04%)
TAKER_FEE_RATE = 0.0004
# BTCUSDT 유지 증거금률 (0.5%)
MAINTENANCE_MARGIN_RATE = 0.005
# 변동성 기반 슬리피지 계산을 위한 상수 (조정 가능)
SLIPPAGE_CONSTANT = 0.2
# 펀딩 수수료가 부과되는 시간 (UTC)
FUNDING_FEE_HOURS = {0, 8, 16}

class TradingEnvironment:
    """
    비트코인 선물 거래 시뮬레이션 환경을 관리하는 클래스.
    모든 거래 로직, 리스크 관리, 상태 업데이트를 캡슐화합니다.
    """
    def __init__(self, chromosomes_size, device='cpu'):
        self.device = device
        self.chromosomes_size = chromosomes_size

        # --- 포지션 상태 변수 ---
        self.pos_list = torch.zeros(chromosomes_size, dtype=torch.long, device=device)
        self.price_list = torch.full((chromosomes_size,), -1.0, dtype=torch.float32, device=device)
        self.leverage_ratio = torch.full((chromosomes_size,), -1, dtype=torch.int, device=device)
        self.enter_ratio = torch.full((chromosomes_size,), -1.0, dtype=torch.float32, device=device)
        self.additional_count = torch.zeros(chromosomes_size, dtype=torch.long, device=device)
        self.holding_period = torch.zeros(chromosomes_size, dtype=torch.long, device=device)
        
        # --- 시점별 손익 및 시간 추적 ---
        self.profit = torch.zeros((chromosomes_size,), dtype=torch.float32, device=device)
        self.last_timestamp = None

        # --- 누적 성과 통계 변수 ---
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
        """
        시간의 흐름에 따라 펀딩 수수료를 계산하고 적용합니다.
        데이터의 시간 간격과 무관하게 작동합니다.
        """
        if self.last_timestamp is None:
            return

        # 이전 시간과 현재 시간 사이에 펀딩 수수료 지급 시간이 포함되는지 확인
        for hour in FUNDING_FEE_HOURS:
            funding_time_today = self.last_timestamp.normalize().replace(hour=hour)
            funding_time_next_day = funding_time_today + pd.Timedelta(days=1)
            
            # last_timestamp < funding_time <= current_timestamp 인 경우를 체크
            for ft in [funding_time_today, funding_time_next_day]:
                if self.last_timestamp < ft <= current_timestamp:
                    active_pos_mask = self.pos_list != 0
                    if not active_pos_mask.any():
                        continue
                    
                    # 펀딩피는 명목가치에 대해 부과됨
                    # PNL(%) = funding_rate * leverage
                    funding_pnl = funding_rate * self.leverage_ratio.float()
                    
                    # 롱 포지션은 funding_rate와 반대 부호, 숏 포지션은 같은 부호
                    long_mask = (self.pos_list == 2) & active_pos_mask
                    short_mask = (self.pos_list == 1) & active_pos_mask

                    # enter_ratio(투입 증거금) 대비 손익을 계산하여 profit에 누적
                    self.profit[long_mask] -= funding_pnl[long_mask] * self.enter_ratio[long_mask]
                    self.profit[short_mask] += funding_pnl[short_mask] * self.enter_ratio[short_mask]
                    break

    def _check_liquidation(self, curr_price):
        """
        바이낸스 격리 마진 기준 강제 청산 로직을 수행합니다.
        """
        active_pos_mask = self.pos_list != 0
        if not active_pos_mask.any():
            return

        pos_price = self.price_list[active_pos_mask]
        leverage = self.leverage_ratio[active_pos_mask].float()
        enter_ratio = self.enter_ratio[active_pos_mask]

        # 미실현 손익률 (가격 변동률)
        pnl_percent = torch.zeros_like(pos_price)
        
        long_mask_local = self.pos_list[active_pos_mask] == 2
        short_mask_local = self.pos_list[active_pos_mask] == 1
        
        pnl_percent[long_mask_local] = (curr_price - pos_price[long_mask_local]) / pos_price[long_mask_local]
        pnl_percent[short_mask_local] = (pos_price[short_mask_local] - curr_price) / pos_price[short_mask_local]
        
        # 마진 잔고 = 초기 증거금 + 미실현 손익 (초기 증거금 대비 비율로 계산)
        # Margin Balance = Initial Margin * (1 + PNL_percent * Leverage)
        margin_balance = enter_ratio * (1 + pnl_percent * leverage)

        # 유지 증거금 = 명목 가치 * 유지 증거금률
        # Maintenance Margin = (Initial Margin * Leverage) * MM_Rate
        maintenance_margin = (enter_ratio * leverage) * MAINTENANCE_MARGIN_RATE

        # 청산 조건: 마진 잔고가 유지 증거금보다 작거나 같아질 때
        liquidation_mask_local = margin_balance <= maintenance_margin
        if not liquidation_mask_local.any():
            return

        # 청산될 chromosome들의 실제 인덱스
        active_indices = torch.where(active_pos_mask)[0]
        liquidated_indices = active_indices[liquidation_mask_local]

        # 청산 처리: 투입된 증거금(enter_ratio) 전액 손실
        self.profit[liquidated_indices] -= self.enter_ratio[liquidated_indices]
        
        # 상태 초기화
        self.pos_list[liquidated_indices] = 0
        self.price_list[liquidated_indices] = -1.0
        self.leverage_ratio[liquidated_indices] = -1
        self.enter_ratio[liquidated_indices] = -1.0
        self.additional_count[liquidated_indices] = 0
        self.holding_period[liquidated_indices] = 0

    def _calculate_action(self, prob, curr_close, curr_high, curr_low, limit=4, min_enter_ratio=0.1, cut_value=1.0):
        """
        모델의 출력을 바탕으로 실제 거래 액션을 수행하고,
        수수료와 슬리피지를 적용하여 손익을 계산합니다.
        (기존 calculate_action 함수를 클래스 메소드로 통합 및 수정)
        """
        # 변동성 기반 슬리피지 계산
        volatility = (curr_high - curr_low) / (curr_close + 1e-9)
        slippage_rate = volatility * SLIPPAGE_CONSTANT
        
        # action, enter_ratio, leverage 분리
        action = torch.argmax(prob[:, :3], dim=1)
        raw_enter_ratio = prob[:, 3]
        raw_leverage = prob[:, 4]

        enter_enter_ratio = activation_fn(raw_enter_ratio)
        enter_enter_ratio = torch.clamp(enter_enter_ratio, min=min_enter_ratio, max=cut_value)
        
        enter_leverage = activation_fn(raw_leverage) * 124.0
        enter_leverage_int = enter_leverage.int() + 1
        enter_leverage_int = torch.clamp(enter_leverage_int, 5, 125)

        hold_index = (action == 0)
        short_index = (action == 1)
        long_index = (action == 2)

        currently_hold = (self.pos_list == 0)
        currently_short = (self.pos_list == 1)
        currently_long = (self.pos_list == 2)

        # 포지션 청산 (Hold 결정 시)
        for pos_type, close_cond in [(1, currently_short), (2, currently_long)]:
            close_idx = torch.where(close_cond & hold_index)[0]
            if len(close_idx) > 0:
                entry_price = self.price_list[close_idx]
                leverage = self.leverage_ratio[close_idx].float()
                
                # 슬리피지 적용된 체결가
                exec_price = curr_close * (1 - slippage_rate) if pos_type == 1 else curr_close * (1 + slippage_rate)

                pnl_percent = (entry_price - exec_price) / entry_price if pos_type == 1 else (exec_price - entry_price) / entry_price
                
                realized_pnl = pnl_percent * leverage * self.enter_ratio[close_idx]
                fee = self.enter_ratio[close_idx] * leverage * TAKER_FEE_RATE # 청산 수수료
                
                self.profit[close_idx] += (realized_pnl - fee)

                self.pos_list[close_idx] = 0
                self.price_list[close_idx] = -1.0
                self.leverage_ratio[close_idx] = -1
                self.enter_ratio[close_idx] = -1.0
                self.additional_count[close_idx] = 0

        # 반대 포지션 전환 (Flip)
        flip_configs = [
            (currently_short, long_index, 2, 2), # short -> long
            (currently_long, short_index, 1, 1)  # long -> short
        ]
        for current_pos_cond, target_action_cond, target_pos, prob_idx in flip_configs:
            flip_idx = torch.where(current_pos_cond & target_action_cond & (prob[:, prob_idx] >= 0.7))[0]
            if len(flip_idx) > 0:
                # 1. 기존 포지션 청산
                entry_price = self.price_list[flip_idx]
                leverage = self.leverage_ratio[flip_idx].float()
                exec_price_close = curr_close * (1 - slippage_rate) if target_pos == 2 else curr_close * (1 + slippage_rate)
                pnl_percent = (entry_price - exec_price_close) / entry_price if target_pos == 2 else (exec_price_close - entry_price) / entry_price
                realized_pnl = pnl_percent * leverage * self.enter_ratio[flip_idx]
                fee_close = self.enter_ratio[flip_idx] * leverage * TAKER_FEE_RATE
                self.profit[flip_idx] += (realized_pnl - fee_close)
                
                # 2. 신규 포지션 진입
                exec_price_open = curr_close * (1 + slippage_rate) if target_pos == 2 else curr_close * (1 - slippage_rate)
                new_leverage = enter_leverage_int[flip_idx]
                new_enter_ratio = enter_enter_ratio[flip_idx]
                fee_open = new_enter_ratio * new_leverage.float() * TAKER_FEE_RATE
                self.profit[flip_idx] -= fee_open

                self.pos_list[flip_idx] = target_pos
                self.price_list[flip_idx] = exec_price_open
                self.leverage_ratio[flip_idx] = new_leverage
                self.enter_ratio[flip_idx] = new_enter_ratio
                self.additional_count[flip_idx] = 0
        
        # 신규 포지션 진입 (Open)
        open_configs = [
            (short_index, 1, 1), # short
            (long_index, 2, 2)   # long
        ]
        for action_cond, pos_type, prob_idx in open_configs:
            open_idx = torch.where(currently_hold & action_cond & (prob[:, prob_idx] >= 0.7))[0]
            if len(open_idx) > 0:
                exec_price = curr_close * (1 - slippage_rate) if pos_type == 1 else curr_close * (1 + slippage_rate)
                new_leverage = enter_leverage_int[open_idx]
                new_enter_ratio = enter_enter_ratio[open_idx]
                
                fee = new_enter_ratio * new_leverage.float() * TAKER_FEE_RATE # 진입 수수료
                self.profit[open_idx] -= fee
                
                self.pos_list[open_idx] = pos_type
                self.price_list[open_idx] = exec_price
                self.leverage_ratio[open_idx] = new_leverage
                self.enter_ratio[open_idx] = new_enter_ratio
                self.additional_count[open_idx] = 0

        # 추가 진입 (Add)
        add_configs = [
            (currently_short, short_index, 1), # short
            (currently_long, long_index, 2)    # long
        ]
        for current_pos_cond, action_cond, prob_idx in add_configs:
             add_idx = torch.where(current_pos_cond & action_cond & (prob[:, prob_idx] >= 0.7))[0]
             if len(add_idx) > 0:
                can_add_idx = add_idx[self.additional_count[add_idx] < limit]
                if len(can_add_idx) > 0:
                    # 추가 진입은 수수료 외에는 직접적인 손익 발생이 없음 (평단가만 변경)
                    # 실제 구현에서는 추가 수량에 대한 수수료가 발생하나, 복잡성을 고려하여 여기서는 생략하거나
                    # 또는 추가된 enter_ratio 만큼 수수료를 차감할 수 있음. 여기서는 후자를 택함.
                    add_ratio = enter_enter_ratio[can_add_idx]
                    add_ratio = torch.minimum((cut_value - self.enter_ratio[can_add_idx]), add_ratio)
                    add_ratio = torch.clamp(add_ratio, min=0.0)

                    # 추가 진입 수수료
                    fee_add = add_ratio * self.leverage_ratio[can_add_idx].float() * TAKER_FEE_RATE
                    self.profit[can_add_idx] -= fee_add

                    before_price = self.price_list[can_add_idx]
                    before_ratio = self.enter_ratio[can_add_idx]
                    
                    exec_price = curr_close # 추가진입은 슬리피지 간소화
                    after_price = (before_price * before_ratio + exec_price * add_ratio) / (before_ratio + add_ratio)
                    after_ratio = before_ratio + add_ratio
                    
                    self.price_list[can_add_idx] = after_price
                    self.enter_ratio[can_add_idx] = torch.clamp(after_ratio, max=cut_value)
                    self.additional_count[can_add_idx] += 1
    
    def step(self, market_data, prob, prescriptor):
        """ 한 타임스텝을 진행합니다. """
        current_timestamp = market_data.name # 데이터프레임의 인덱스(시간)
        curr_open = torch.tensor(market_data['Open'], dtype=torch.float32, device=self.device)
        curr_close = torch.tensor(market_data['Close'], dtype=torch.float32, device=self.device)
        curr_high = torch.tensor(market_data['High'], dtype=torch.float32, device=self.device)
        curr_low = torch.tensor(market_data['Low'], dtype=torch.float32, device=self.device)
        funding_rate = torch.tensor(market_data['funding_rate'], dtype=torch.float32, device=self.device)
        
        # 1. 펀딩 수수료 적용
        self._apply_funding_fees(current_timestamp, funding_rate)
        
        # 2. 강제 청산 확인
        # 청산은 최악의 경우(Long은 Low, Short는 High)를 가정하여 확인
        self._check_liquidation(curr_low if self.pos_list.any() == 2 else curr_high)

        # 3. 모델의 액션 계산 및 실행 (after_forward 포함)
        now_profit = calculate_now_profit(self.pos_list, self.price_list, self.leverage_ratio, self.enter_ratio, curr_close)
        prob = after_forward(prescriptor, prob, now_profit, self.leverage_ratio, self.enter_ratio, self.pos_list, device=self.device)
        self._calculate_action(prob, curr_close, curr_high, curr_low)

        # 4. 포지션 보유 기간 업데이트
        self.holding_period[self.pos_list != 0] += 1
        self.holding_period[self.pos_list == 0] = 0

        # 5. 이번 스텝의 손익을 누적 통계에 반영
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

            # 누적 수익률 및 MDD 계산용
            self.cum_sum += self.profit
            self.running_max = torch.maximum(self.running_max, self.cum_sum)
            current_drawdown = self.running_max - self.cum_sum
            self.max_drawdown = torch.maximum(self.max_drawdown, current_drawdown)

            # 복리 계산 (상대적 자산 가치)
            self.compound_value[non_zero_mask] *= (1 + self.profit[non_zero_mask] / 100.0)

        # 6. 다음 스텝을 위해 스텝별 profit 초기화 및 시간 업데이트
        self.profit.zero_()
        self.last_timestamp = current_timestamp

    def get_final_metrics(self, minimum_date=40):
        """ 시뮬레이션 종료 후 최종 성과 지표를 계산합니다. """
        count_returns_f = self.count_returns.float()
        
        mean_returns = torch.where(
            count_returns_f > 0, 
            self.sum_returns / count_returns_f, 
            torch.full_like(self.sum_returns, -1e9)
        )
        
        profit_factors = torch.where(
            self.total_loss_agg > 0, 
            self.total_profit_agg / (self.total_loss_agg + 1e-9), 
            torch.full_like(self.total_profit_agg, -1e9)
        )
        
        win_rates = torch.where(
            count_returns_f > 0, 
            self.count_wins.float() / count_returns_f, 
            torch.full_like(count_returns_f, -1e9)
        )
        
        invalid_mask = self.count_returns < minimum_date
        mean_returns[invalid_mask] = -1e9
        profit_factors[invalid_mask] = -1e9
        win_rates[invalid_mask] = -1e9
        self.max_drawdown[invalid_mask] = 1e9 # MDD는 작을수록 좋으므로 큰 값으로 패널티
        self.compound_value[invalid_mask] = -1e9
        
        metrics = torch.stack(
            [mean_returns, profit_factors, win_rates, self.max_drawdown, self.compound_value], 
            dim=1
        )
        return metrics.cpu().numpy()

# ##############################################################################
# 원본 코드 영역 (최소한의 수정)
# ##############################################################################

def days_difference(date1, date2):
    # 날짜 차이 계산
    difference = date2 - date1
    # 일수 반환
    return np.abs(difference / np.timedelta64(1, 'D')).astype(int)

import numpy as np
from scipy.stats import skew, kurtosis

# 전역적으로 기울기 계산 비활성화
torch.set_grad_enabled(False)

# # 예시 activation 함수 (identity)
# def activation_fn(x):
#     return 1 / (1 + torch.exp(-x))

def activation_fn(x: torch.Tensor) -> torch.Tensor:
    device = x.device
    # b = ln(4) 를 사용
    b = torch.log(torch.tensor(4.0, device=device))
    # 분모: exp(2b)-1
    denom = torch.exp(b * 2) - 1
    result = torch.where(
        x <= -1,
        torch.tensor(0.0, device=device),
        torch.where(
            x >= 1,
            torch.tensor(1.0, device=device),
            (torch.exp(b * (x + 1)) - 1) / denom
        )
    )
    return result

def calculate_performance_metrics(returns_list, minimum_date=40):
    """
    원래 누적된 returns_list를 바탕으로 성과 지표(metrics)를 계산하는 함수.
    (참고용으로 남겨두었으며, 본 예제에서는 사용하지 않습니다.)
    """
    chromosomes_size = returns_list.shape[0]

    mean_returns = np.full(chromosomes_size, -1e9)
    sharpe_ratios = np.full(chromosomes_size, -1e9)
    sortino_ratios = np.full(chromosomes_size, -1e9)
    profit_factors = np.full(chromosomes_size, -1e9)
    win_rates = np.full(chromosomes_size, -1e9)
    max_drawdowns = np.full(chromosomes_size, 1e9)
    cumulative_returns = np.full(chromosomes_size, -1e9)  # 누적 수익률

    risk_free_rate = 0.0  # 조정 가능

    num_non_zero = np.count_nonzero(returns_list != 0, axis=1)
    valid_chromosomes = num_non_zero > minimum_date

    if sum(valid_chromosomes) != 0:
        non_zero_returns_list = np.where(returns_list != 0, returns_list, np.nan)

        mean_returns[valid_chromosomes] = np.nanmean(non_zero_returns_list[valid_chromosomes], axis=1)
        std_returns_i = np.nanstd(non_zero_returns_list[valid_chromosomes], axis=1) + 1e-9

        valid_std = (std_returns_i != 0) & (~np.isnan(std_returns_i))
        sharpe_ratios_subset = (mean_returns[valid_chromosomes] - risk_free_rate) / std_returns_i
        sharpe_ratios[valid_chromosomes] = np.where(valid_std, sharpe_ratios_subset, -1e9)
        sharpe_ratios = np.where(np.isnan(sharpe_ratios), -1e9, sharpe_ratios)

        cumulative_returns_raw = np.cumsum(returns_list, axis=1)
        running_max = np.maximum.accumulate(cumulative_returns_raw, axis=1)
        drawdowns = running_max - cumulative_returns_raw
        max_drawdowns[valid_chromosomes] = np.nanmax(drawdowns[valid_chromosomes], axis=1)

        negative_returns = np.where(non_zero_returns_list < 0, non_zero_returns_list, np.nan)
        downside_std = np.nanstd(negative_returns[valid_chromosomes], axis=1) + 1e-9
        valid_downside_std = (downside_std != 0) & (~np.isnan(downside_std))
        sortino_ratios_subset = (mean_returns[valid_chromosomes] - risk_free_rate) / downside_std
        sortino_ratios[valid_chromosomes] = np.where(valid_downside_std, sortino_ratios_subset, -1e9)
        sortino_ratios = np.where(np.isnan(sortino_ratios), -1e9, sortino_ratios)

        total_profit = np.nansum(np.where(non_zero_returns_list > 0, non_zero_returns_list, 0), axis=1)
        total_loss = -np.nansum(np.where(non_zero_returns_list < 0, non_zero_returns_list, 0), axis=1)
        valid_total_loss = (total_loss != 0) & (~np.isnan(total_loss))
        profit_factors[valid_chromosomes] = -1e9
        profit_factors[valid_chromosomes & valid_total_loss] = total_profit[valid_chromosomes & valid_total_loss] / (total_loss[valid_chromosomes & valid_total_loss] + 1e-9)
        profit_factors = np.where(np.isnan(profit_factors), -1e9, profit_factors)

        num_wins = np.nansum(np.where(non_zero_returns_list > 0, 1, 0), axis=1)
        num_trades = num_non_zero
        valid_num_trades = (num_trades != 0) & (~np.isnan(num_trades))
        win_rates[valid_chromosomes] = -1e9
        win_rates[valid_chromosomes & valid_num_trades] = num_wins[valid_chromosomes & valid_num_trades] / num_trades[valid_chromosomes & valid_num_trades]
        win_rates = np.where(np.isnan(win_rates), -1e9, win_rates)

        initial_value = 1.0
        for idx in np.where(valid_chromosomes)[0]:
            clean_returns = returns_list[idx][returns_list[idx] != 0]
            current_value = initial_value
            for ret in clean_returns:
                current_value += current_value * (ret / 100.0)
            cumulative_returns[idx] = current_value

    metrics = np.concatenate([
        np.expand_dims(mean_returns, axis=1),
        np.expand_dims(sharpe_ratios, axis=1),
        np.expand_dims(sortino_ratios, axis=1),
        np.expand_dims(profit_factors, axis=1),
        np.expand_dims(win_rates, axis=1),
        np.expand_dims(max_drawdowns, axis=1),
        np.expand_dims(cumulative_returns, axis=1)
    ], axis=1)
    
    return metrics

class CustomDataset(Dataset):
    def __init__(self, data, data_1d):
        self.data = data
        self.data_1d = data_1d

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data_1d[idx]

def inference(scaled_tensor, scaled_tensor_1d, model, device='cuda:0'):
    dataset = CustomDataset(scaled_tensor, scaled_tensor_1d)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=8, pin_memory=True)
    logits = []
    from tqdm import tqdm

    for data, data_1d in tqdm(dataloader, desc="Inference Progress"):
        data = data.to(torch.float32).to(device)
        data_1d = data_1d.to(torch.float32).to(device)
        logit = model.base_forward(data, data_1d)
        logits.append(logit)
    return logits

def time_based_exit_fn(
    pos_list,
    price_list,
    leverage_ratio,
    enter_ratio,
    additional_count,
    profit,
    curr_price,
    holding_period,
    max_holding_bars,
    device='cpu'
):
    """
    일정 기간 이상 보유한 포지션을 강제 청산하는 함수.
    (기능은 유지하되, 현재 시뮬레이션에서는 호출되지 않음)
    """
    close_indices = torch.where((holding_period > max_holding_bars) & (pos_list != 0))[0]
    if len(close_indices) > 0:
        short_idx = close_indices[pos_list[close_indices] == 1]
        if len(short_idx) > 0:
            realized_pnl = (price_list[short_idx] - curr_price) / price_list[short_idx] * 100.0
            realized_pnl = realized_pnl * leverage_ratio[short_idx] * enter_ratio[short_idx]
            fee = 0.1 * leverage_ratio[short_idx] * enter_ratio[short_idx] # 원본 수수료 로직
            profit[short_idx] += (realized_pnl - fee)
            pos_list[short_idx] = 0
            price_list[short_idx] = -1.0
            leverage_ratio[short_idx] = -1
            enter_ratio[short_idx] = -1.0
            additional_count[short_idx] = 0
            holding_period[short_idx] = 0

        long_idx = close_indices[pos_list[close_indices] == 2]
        if len(long_idx) > 0:
            realized_pnl = (curr_price - price_list[long_idx]) / price_list[long_idx] * 100.0
            realized_pnl = realized_pnl * leverage_ratio[long_idx] * enter_ratio[long_idx]
            fee = 0.1 * leverage_ratio[long_idx] * enter_ratio[long_idx] # 원본 수수료 로직
            profit[long_idx] += (realized_pnl - fee)
            pos_list[long_idx] = 0
            price_list[long_idx] = -1.0
            leverage_ratio[long_idx] = -1
            enter_ratio[long_idx] = -1.0
            additional_count[long_idx] = 0
            holding_period[long_idx] = 0

    return pos_list, price_list, leverage_ratio, enter_ratio, additional_count, profit, holding_period


def calculate_now_profit(pos_list, price_list, leverage_ratio, enter_ratio, curr_price):
    now_profit = torch.zeros_like(pos_list, dtype=torch.float32)
    short_index = torch.where(pos_list == 1)[0]
    long_index = torch.where(pos_list == 2)[0]

    # 현재 미실현 손익률 (수수료, 슬리피지 미반영된 단순 평가)
    short_pnl_percent = (price_list[short_index] - curr_price) / price_list[short_index]
    long_pnl_percent = (curr_price - price_list[long_index]) / price_list[long_index]

    # 증거금 대비 미실현 손익
    now_profit[short_index] = short_pnl_percent * leverage_ratio[short_index].float() * enter_ratio[short_index]
    now_profit[long_index] = long_pnl_percent * leverage_ratio[long_index].float() * enter_ratio[long_index]

    return now_profit

def after_forward(model, prob, now_profit, leverage_ratio, enter_ratio, pos_list, device):
    ch_size = len(now_profit)
    now_profit_tensor = now_profit.unsqueeze(dim=1)
    leverage_ratio_tensor = leverage_ratio.unsqueeze(dim=1).to(torch.float32)
    enter_ratio_tensor = enter_ratio.unsqueeze(dim=1)
    mapping = {0: 0, 1: 1, 2: 2}  # 필요에 따라 조정
    mapped_array = pos_list
    step = torch.arange(0, ch_size * 3, step=3, device=device)

    x = torch.cat([prob, now_profit_tensor / 10, leverage_ratio_tensor / 125, enter_ratio_tensor], dim=1)
    cate_x = mapped_array + step

    x = x.to(torch.float32).to(device)
    cate_x = cate_x.to(device).long()

    after_output = model.after_forward(x=x.squeeze(dim=0), x_cate=cate_x)
    return after_output.squeeze(dim=0)

def calculate_fitness(metrics):
    chromosomes_size = len(metrics)
    
    def normalize_metric(metric, higher_is_better=True):
        valid_indices = metric != -1e9
        valid_metric = metric[valid_indices]
        if len(valid_metric) == 0:
            return np.zeros_like(metric)
        min_val = np.nanmin(valid_metric)
        max_val = np.nanmax(valid_metric)
        if min_val == max_val:
            normalized = np.ones_like(metric) if higher_is_better else np.zeros_like(metric)
        else:
            if higher_is_better:
                normalized = (metric - min_val) / (max_val - min_val + 1e-8)
            else:
                normalized = (max_val - metric) / (max_val - min_val + 1e-8)
        normalized[~valid_indices] = 0.0
        return normalized

    # MDD는 작을수록 좋으므로, 정규화 전에 부호를 바꾼다.
    # 단, 1e9 패널티 값은 그대로 유지해야 함.
    mdd_scores = metrics[:, 3].copy()
    valid_mdd_mask = mdd_scores < 1e9
    mdd_scores[valid_mdd_mask] = -mdd_scores[valid_mdd_mask]

    normalized_metrics = np.zeros_like(metrics)
    higher_is_better_list = [True, True, True, True, True] 
    metrics_to_normalize = [
        metrics[:, 0], # mean_returns
        metrics[:, 1], # profit_factors
        metrics[:, 2], # win_rates
        mdd_scores,    # (negative) max_drawdowns
        metrics[:, 4]  # cumulative_returns
    ]

    for i in range(len(higher_is_better_list)):
        normalized_metrics[:, i] = normalize_metric(metrics_to_normalize[i], higher_is_better=higher_is_better_list[i])

    weights = [0.1, 0.2, 0.15, 0.15, 0.4]
    fitness_values = np.zeros(chromosomes_size)
    for i in range(len(weights)):
        fitness_values += weights[i] * normalized_metrics[:, i]

    fitness_values[metrics[:, 0] == -1e9] = -1e9

    return fitness_values


def fitness_fn(prescriptor, data, probs, entry_index_list, entry_pos_list, skip_data_cnt, start_data_cnt, chromosomes_size, window_size,
               alpha=1., cut_percent=90., device='cpu', stop_cnt=1e9, profit_init=10, limit=4, minimum_date=40):
    """
    리팩토링된 fitness_fn 함수.
    TradingEnvironment 클래스를 사용하여 시뮬레이션을 실행하고 결과를 반환합니다.
    """
    # TradingEnvironment 인스턴스 생성
    environment = TradingEnvironment(chromosomes_size=chromosomes_size, device=device)
    
    # 시뮬레이션 실행
    # tqdm을 사용하여 진행 상황 표시
    pbar = tqdm(range(len(entry_index_list)), desc="Fitness Simulation")
    for data_cnt, entry_index in enumerate(entry_index_list):
        pbar.update(1)
        if data_cnt >= stop_cnt:
            break
        if data_cnt < start_data_cnt:
            continue
        
        # 현재 스텝의 데이터와 모델 예측 확률
        market_data = data.iloc[entry_index]
        prob = torch.tensor(probs[:, data_cnt - skip_data_cnt]).float().to(device)
        
        # 환경의 한 스텝 진행
        environment.step(market_data, prob, prescriptor)

    pbar.close()
    
    # 최종 성과 지표 계산 및 반환
    final_metrics = environment.get_final_metrics(minimum_date=minimum_date)
    return final_metrics


def get_chromosome_key(chromosome):
    quantized_chrom = np.round(chromosome.cpu().numpy(), decimals=6)
    return tuple(quantized_chrom.flatten())

def generation_valid(data_1m, dataset_1m, dataset_1d, prescriptor, evolution,
                     skip_data_cnt, valid_skip_data_cnt, test_skip_data_cnt, chromosomes_size,
                     window_size, gen_loop, best_size, elite_size, profit_init, 
                     entry_index_list=None, entry_pos_list=None,
                     best_profit=None, best_chromosomes=None, start_gen=0, device='cuda:0',
                     warming_step=5):
    
    best_profit = best_profit
    best_chromosomes = best_chromosomes
    temp_dir = 'generation'
    os.makedirs(temp_dir, exist_ok=True)
    
    # 데이터의 인덱스를 datetime으로 변환 (펀딩피 계산용)
    if not isinstance(data_1m.index, pd.DatetimeIndex):
        data_1m.index = pd.to_datetime(data_1m.index)

    for gen_idx in range(start_gen, gen_loop):
        print(f'generation  {gen_idx}: ')

        probs = inference(dataset_1m, dataset_1d, prescriptor, device)
        probs = torch.concat(probs, dim=1)
        probs = probs.squeeze(dim=2)

        train_metrics = fitness_fn(
            prescriptor=prescriptor,
            data=data_1m,
            probs=probs,
            entry_index_list=entry_index_list,
            entry_pos_list=entry_pos_list,
            skip_data_cnt=skip_data_cnt,
            start_data_cnt=skip_data_cnt,
            chromosomes_size=chromosomes_size,
            window_size=window_size,
            device=device,
            stop_cnt=valid_skip_data_cnt,
            limit=4
        )
        if warming_step <= gen_idx:
            if gen_idx != 0:
                valid_metrics_np = train_metrics[:elite_size]
                valid_metrics = torch.from_numpy(valid_metrics_np)
                
                # compound_value > 6 (600% 수익), max_drawdown < 40% 조건
                valid_index = np.where((valid_metrics_np[:, 4] > 6.) & (valid_metrics_np[:, 3] < 40))[0]
                
                if len(valid_index) > 0:
                    valid_metrics = valid_metrics[valid_index]
                    
                    if best_profit is None:
                        best_profit = valid_metrics
                        best_chromosomes, _, _, _ = evolution.flatten_chromosomes()
                        best_chromosomes = torch.tensor(best_chromosomes[:elite_size])[valid_index].clone()
                    else:
                        chromosomes, _, _, _ = evolution.flatten_chromosomes()
                        chromosomes = torch.tensor(chromosomes[:elite_size])[valid_index].clone()
                        
                        # 새로운 chromosome만 추가하기 위한 로직
                        new_fitness_list = []
                        new_chromosomes_list = []
                        for i, t in enumerate(valid_metrics):
                            is_duplicate = any(torch.equal(t, bt) for bt in best_profit)
                            if not is_duplicate:
                                new_fitness_list.append(t)
                                new_chromosomes_list.append(chromosomes[i])

                        if new_fitness_list:
                            new_fitness = torch.stack(new_fitness_list)
                            new_chromosomes_tensor = torch.stack(new_chromosomes_list)
                            best_profit = torch.cat([best_profit, new_fitness])
                            best_chromosomes = torch.cat([best_chromosomes, new_chromosomes_tensor])

                    if len(best_chromosomes) > best_size:
                        print('check_discard')
                        valid_fitness = calculate_fitness(deepcopy(best_profit).numpy())
                        elite_idx, elite_chromosomes = evolution.select_elite(torch.from_numpy(valid_fitness), best_chromosomes, best_size)

                        best_profit = best_profit[elite_idx]
                        best_chromosomes = elite_chromosomes
        
        gen_data = {
            "generation": gen_idx,
            "prescriptor_state_dict": prescriptor.state_dict(),
            "best_profit": best_profit,
            "best_chromosomes": best_chromosomes,
        }
        
        train_fitness = calculate_fitness(train_metrics)
        torch.save(gen_data, os.path.join(temp_dir, f'generation_{gen_idx}.pt')) 
        evolution.evolve(torch.from_numpy(train_fitness))
        prescriptor = prescriptor.to(device)
        
        del probs
    return best_chromosomes, best_profit

def generation_test(data_1m, dataset_1m, dataset_1d, prescriptor, skip_data_cnt,
                     start_data_cnt, end_data_cnt, chromosomes_size,
                     window_size, profit_init, 
                     entry_index_list=None, entry_pos_list=None, device='cuda:0'):
    
    # 데이터의 인덱스를 datetime으로 변환 (펀딩피 계산용)
    if not isinstance(data_1m.index, pd.DatetimeIndex):
        data_1m.index = pd.to_datetime(data_1m.index)

    probs = inference(dataset_1m, dataset_1d, prescriptor, device)
    probs = torch.concat(probs, dim=1)
    probs = probs.squeeze(dim=2)
    
    profit = fitness_fn(
        prescriptor=prescriptor,
        data=data_1m,
        probs=probs,
        entry_index_list=entry_index_list,
        entry_pos_list=entry_pos_list,
        skip_data_cnt=skip_data_cnt,
        start_data_cnt=start_data_cnt,
        chromosomes_size=chromosomes_size,
        window_size=window_size,
        device=device,
        stop_cnt=end_data_cnt,
        limit=4
    )
        
    return profit