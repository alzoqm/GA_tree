# training/trading_env.py
# Unified TradingEnvironment with complete original logic + multi-objective capabilities

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from models.model import GATreePop
from models.constants import (
    ACTION_NOT_FOUND,
    ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_CLOSE_ALL,
    ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION, ACTION_FLIP_POSITION
)
 
# ##############################################################################
#           통합 거래 환경 및 시뮬레이션 (원본 로직 완전 보존 + 다중 목표 기능)
# ##############################################################################


class TradingEnvironment:
    """
    통합 거래 시뮬레이션 환경 클래스 - 원본의 모든 거래 로직 보존 + 다중 목표 최적화 지원
    
    이 클래스는 기존의 모든 거래 규칙, 청산 로직, 액션 실행, 수수료 계산, 
    포지션 관리 등을 그대로 유지하면서 다중 목표 메트릭 계산 기능을 추가합니다.
    """
    def __init__(self, chromosomes_size: int, env_config: dict, 
                 objectives_config: Optional[dict] = None, device='cpu'):
        """
        거래 환경 초기화
        
        Args:
            chromosomes_size: 개체 수
            env_config: 거래 환경 설정
            objectives_config: 다중 목표 설정 (선택적)
            device: 계산 디바이스
        """
        self.device = device
        self.chromosomes_size = chromosomes_size
        
        # === 원본 거래 환경 설정 완전 보존 ===
        self.taker_fee_rate = env_config['taker_fee_rate']
        self.maintenance_margin_rate = env_config['maintenance_margin_rate']
        self.fixed_slippage_rate = env_config['fixed_slippage_rate']
        # Mark price / equity options
        self.use_real_mark_price = env_config.get('use_real_mark_price', False)
        self.equity_mtm_basis = env_config.get('equity_mtm_basis', 'mark_proxy')  # 'mark_price' | 'mark_proxy'
        # Liquidation clearance fee (insurance fund/clearance)
        self.liq_clearance_fee_rate = env_config.get('liq_clearance_fee_rate', 0.0)
        self.funding_fee_hours = {0, 8, 16}
        # Minimum constraints
        self.min_entry_ratio = env_config.get('min_entry_ratio', 0.05)
        self.min_leverage = int(env_config.get('min_leverage', 2))
        self.max_leverage = int(env_config.get('max_leverage', 1_000_000))
        self.min_additional_enter_ratio = env_config.get('min_additional_enter_ratio', 0.05)
        self.max_additional_entries = env_config.get('max_additional_entries', 1e9)

        # === 원본 상태 변수 완전 보존 ===
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
        # Mark-to-market equity tracking
        self.realized_cum = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        # Trade-based stats (vs event-based)
        self.trade_open = torch.zeros(chromosomes_size, device=device, dtype=torch.bool)
        self.trade_realized_pnl = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.trade_win_count = torch.zeros(chromosomes_size, device=device, dtype=torch.int32)
        self.trade_total_count = torch.zeros(chromosomes_size, device=device, dtype=torch.int32)
        # Trade-level aggregates for monitoring/analysis
        self.trade_holding_sum = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.trade_holding_avg = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.curr_win_streak = torch.zeros(chromosomes_size, device=device, dtype=torch.int32)
        self.curr_loss_streak = torch.zeros(chromosomes_size, device=device, dtype=torch.int32)
        self.max_win_streak = torch.zeros(chromosomes_size, device=device, dtype=torch.int32)
        self.max_loss_streak = torch.zeros(chromosomes_size, device=device, dtype=torch.int32)

        # === 다중 목표 최적화 추가 기능 ===
        if objectives_config is None:
            objectives_config = {
                'objectives': ['mean_return', 'max_drawdown', 'sharpe_ratio', 'win_rate'],
                'minimize': [False, True, False, False],
                'risk_free_rate': 0.0,
                'volatility_lookback': 30
            }
        
        self.objectives_config = objectives_config
        self.objective_names = objectives_config.get('objectives', [
            'mean_return', 'max_drawdown', 'sharpe_ratio', 'win_rate'
        ])
        self.minimize_objectives = objectives_config.get('minimize', [
            False, True, False, False  # minimize drawdown, maximize others
        ])
        
        if len(self.minimize_objectives) != len(self.objective_names):
            raise ValueError("minimize list must match objectives list length")
        
        # 다중 목표 메트릭 계산용 추가 추적 변수
        self.risk_free_rate = objectives_config.get('risk_free_rate', 0.0)
        self.lookback_period = objectives_config.get('lookback_period', 252)  # for annualization
        self.volatility_lookback = objectives_config.get('volatility_lookback', 30)
        # Additional advanced metric params
        self.mar = float(objectives_config.get('mar', 0.0))
        self.benchmark_mode = str(objectives_config.get('benchmark_mode', 'close_to_next_open'))
        
        # 고급 메트릭 계산용 히스토리 (메모리 효율적으로 관리)
        self.returns_history = []  # Store return history for rolling calculations
        self.equity_history = []   # Store equity history for advanced metrics
        self.bench_returns_history = []  # Benchmark returns for Information Ratio

        # === 액션 중복 방지(De-duplication) 설정 및 버퍼 ===
        self.enable_action_dedup = bool(env_config.get('enable_action_dedup', False))
        # 기본적으로 부분청산/추가진입은 반복적용이 전략적으로 허용되므로 제외
        self.dedup_exclude_types = set(env_config.get('dedup_exclude_types', [
            ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION
        ]))
        # 앞에서부터 몇 컬럼을 비교할지 (action_type 포함). predictor는 (B,4) 반환.
        self.dedup_compare_cols = int(env_config.get('dedup_compare_cols', 3))
        self.dedup_atol = float(env_config.get('dedup_atol', 1e-6))
        self.dedup_rtol = float(env_config.get('dedup_rtol', 1e-4))
        # 직전 스텝의 액션이 실제 적용된 경우에만 중복 스킵
        self.dedup_require_applied = bool(env_config.get('dedup_require_applied', True))

        # === CUDA 실행 옵션 ===
        # debug_compare_cpu_gpu는 더 이상 사용하지 않지만, 호환을 위해 남겨둡니다.
        self.debug_compare_cpu_gpu = bool(env_config.get('debug_compare_cpu_gpu', False))

        # 이전 스텝의 원본 예측 액션 저장 (B,4)
        self.prev_actions_raw = torch.full(
            (chromosomes_size, 4), float('nan'), dtype=torch.float32, device=device
        )
        # 직전 스텝에서 실제 적용된 개체 마스크
        self.prev_applied_mask = torch.zeros(chromosomes_size, dtype=torch.bool, device=device)

    # ------------------------------------------------------------
    # Step decomposition helpers (no logic change)
    # ------------------------------------------------------------
    @dataclass
    class StepState:
        timestamp: pd.Timestamp
        high: torch.Tensor
        low: torch.Tensor
        close: torch.Tensor
        funding_rate: torch.Tensor
        mark_t: torch.Tensor
        exec_base_price: torch.Tensor | float

    def _extract_step_state(self, market_data, next_open_price) -> "TradingEnvironment.StepState":
        ts = market_data.name
        high = torch.tensor(market_data['High'], dtype=torch.float32, device=self.device)
        low = torch.tensor(market_data['Low'], dtype=torch.float32, device=self.device)
        close = torch.tensor(market_data['Close'], dtype=torch.float32, device=self.device)
        funding_rate = torch.tensor(market_data['fundingRate'], dtype=torch.float32, device=self.device)
        mark_t = (high + low + close) / 3.0
        return TradingEnvironment.StepState(
            timestamp=ts,
            high=high,
            low=low,
            close=close,
            funding_rate=funding_rate,
            mark_t=mark_t,
            exec_base_price=next_open_price,
        )

    def _prepare_and_deduplicate_actions(self, predicted_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = predicted_actions.to(self.device)
        raw_actions = actions.clone()

        if self.enable_action_dedup:
            cols = max(1, min(self.dedup_compare_cols, actions.shape[1], self.prev_actions_raw.shape[1]))
            curr_type = actions[:, 0].long()
            prev_type = self.prev_actions_raw[:, 0].long()
            type_equal = curr_type == prev_type

            if cols > 1:
                same_params = torch.isclose(
                    actions[:, 1:cols],
                    self.prev_actions_raw[:, 1:cols],
                    rtol=self.dedup_rtol,
                    atol=self.dedup_atol,
                    equal_nan=False
                ).all(dim=1)
            else:
                same_params = torch.ones_like(type_equal, dtype=torch.bool)

            if len(self.dedup_exclude_types) > 0:
                exclude_tensor = torch.tensor(list(self.dedup_exclude_types), dtype=torch.long, device=self.device)
                excluded_mask = torch.isin(curr_type, exclude_tensor)
            else:
                excluded_mask = torch.zeros_like(type_equal, dtype=torch.bool)

            duplicate_mask = type_equal & same_params & (~excluded_mask)
            if self.dedup_require_applied:
                duplicate_mask &= self.prev_applied_mask

            if duplicate_mask.any():
                actions[duplicate_mask, :] = 0.0
                actions[duplicate_mask, 0] = float(ACTION_NOT_FOUND)

        return actions, raw_actions

    def _snapshot_state_for_dedup(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pos_list.clone(), self.enter_ratio.clone()

    def _execute_actions_wrapper(self, actions: torch.Tensor, exec_base_price):
        if str(self.device).startswith('cuda'):
            used_cuda = self._execute_actions(actions, exec_base_price)
            if not used_cuda:
                raise RuntimeError("CUDA 확장 모듈을 찾을 수 없습니다. Python 기반 _execute_actions는 제거되었습니다.")
        else:
            raise RuntimeError("device가 cuda가 아닙니다. CUDA만 지원되며 Python 경로는 제거되었습니다.")

    def _update_holding_periods(self):
        self.holding_period[self.pos_list != 0] += 1
        self.holding_period[self.pos_list == 0] = 0

    def _accumulate_realized_pnl_and_stats(self):
        """Accumulate realized P&L and event-based stats.
        CUDA fast-path with safe CPU fallback for parity testing.
        """
        used_cuda = False
        if str(self.device).startswith('cuda'):
            try:
                import gatree_cuda_compat as gatree_cuda
            except Exception:
                gatree_cuda = None
            if gatree_cuda is not None and getattr(gatree_cuda, 'accumulate_realized_stats', None) is not None:
                # Ensure CUDA and contiguous tensors
                profit = self.profit.contiguous()
                realized_cum = self.realized_cum.contiguous()
                sum_returns = self.sum_returns.contiguous()
                count_returns = self.count_returns.contiguous()
                sum_sq_returns = self.sum_sq_returns.contiguous()
                sum_neg = self.sum_neg.contiguous()
                count_neg = self.count_neg.contiguous()
                sum_sq_neg = self.sum_sq_neg.contiguous()
                total_profit_agg = self.total_profit_agg.contiguous()
                total_loss_agg = self.total_loss_agg.contiguous()
                count_wins = self.count_wins.contiguous()
                compound_value = self.compound_value.contiguous()

                gatree_cuda.accumulate_realized_stats(
                    profit,
                    realized_cum,
                    sum_returns,
                    count_returns,
                    sum_sq_returns,
                    sum_neg,
                    count_neg,
                    sum_sq_neg,
                    total_profit_agg,
                    total_loss_agg,
                    count_wins,
                    compound_value,
                )
                # If any tensor got new storage via contiguous(), write back views
                if realized_cum.data_ptr() != self.realized_cum.data_ptr():
                    self.realized_cum = realized_cum
                if sum_returns.data_ptr() != self.sum_returns.data_ptr():
                    self.sum_returns = sum_returns
                if count_returns.data_ptr() != self.count_returns.data_ptr():
                    self.count_returns = count_returns
                if sum_sq_returns.data_ptr() != self.sum_sq_returns.data_ptr():
                    self.sum_sq_returns = sum_sq_returns
                if sum_neg.data_ptr() != self.sum_neg.data_ptr():
                    self.sum_neg = sum_neg
                if count_neg.data_ptr() != self.count_neg.data_ptr():
                    self.count_neg = count_neg
                if sum_sq_neg.data_ptr() != self.sum_sq_neg.data_ptr():
                    self.sum_sq_neg = sum_sq_neg
                if total_profit_agg.data_ptr() != self.total_profit_agg.data_ptr():
                    self.total_profit_agg = total_profit_agg
                if total_loss_agg.data_ptr() != self.total_loss_agg.data_ptr():
                    self.total_loss_agg = total_loss_agg
                if count_wins.data_ptr() != self.count_wins.data_ptr():
                    self.count_wins = count_wins
                if compound_value.data_ptr() != self.compound_value.data_ptr():
                    self.compound_value = compound_value

                used_cuda = True

        if not used_cuda:
            non_zero_mask = self.profit != 0
            if non_zero_mask.any():
                realized_delta = self.profit.clone()
                self.realized_cum += realized_delta
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
                self.compound_value[non_zero_mask] *= (1 + self.profit[non_zero_mask])

    def _compute_unrealized_pnl(self, mark_t: torch.Tensor) -> torch.Tensor:
        open_mask = self.pos_list != 0
        if open_mask.any():
            side = torch.where(self.pos_list == 2, 1.0, -1.0)
            entry = self.price_list
            lev = self.leverage_ratio.float()
            er = self.enter_ratio
            pnl_pct = side * (mark_t - entry) / (entry + 1e-9)
            unreal_vec = pnl_pct * lev * er
            total_unreal = torch.zeros_like(self.sum_returns)
            total_unreal[open_mask] = unreal_vec[open_mask]
        else:
            total_unreal = torch.zeros_like(self.sum_returns)
        return total_unreal

    def _update_equity_and_drawdown(self, total_unreal: torch.Tensor, timestamp):
        equity = self.realized_cum + total_unreal
        self.cum_sum = equity
        self.running_max = torch.maximum(self.running_max, self.cum_sum)
        current_drawdown = self.running_max - self.cum_sum
        self.max_drawdown = torch.maximum(self.max_drawdown, current_drawdown)
        self.profit.zero_()
        self.last_timestamp = timestamp

    def _update_dedup_buffers_after_exec(
        self,
        actions: torch.Tensor,
        raw_actions: torch.Tensor,
        pos_before: torch.Tensor,
        er_before: torch.Tensor,
    ):
        if not self.enable_action_dedup:
            return
        pos_after = self.pos_list
        er_after = self.enter_ratio
        atype_eff = actions[:, 0].long()

        eps = 1e-8
        applied_new_long = (atype_eff == ACTION_NEW_LONG) & (pos_before == 0) & (pos_after == 2)
        applied_new_short = (atype_eff == ACTION_NEW_SHORT) & (pos_before == 0) & (pos_after == 1)
        applied_close_all = (atype_eff == ACTION_CLOSE_ALL) & (pos_before != 0) & (pos_after == 0)
        applied_flip = (atype_eff == ACTION_FLIP_POSITION) & (
            ((pos_before == 2) & (pos_after == 1)) | ((pos_before == 1) & (pos_after == 2))
        )
        applied_close_partial = (atype_eff == ACTION_CLOSE_PARTIAL) & (pos_before != 0) & (er_after < er_before - eps)
        applied_add = (atype_eff == ACTION_ADD_POSITION) & (pos_before != 0) & (er_after > er_before + eps)

        applied_mask = (
            applied_new_long | applied_new_short | applied_close_all |
            applied_flip | applied_close_partial | applied_add
        )

        self.prev_applied_mask = applied_mask
        cols_copy = min(self.prev_actions_raw.shape[1], raw_actions.shape[1])
        self.prev_actions_raw[:, :cols_copy] = raw_actions[:, :cols_copy]

    def _update_return_histories(self, pre_equity: torch.Tensor, market_data, exec_base_price):
        post_equity = self._calculate_current_equity(market_data)
        returns = (post_equity - pre_equity) / (pre_equity + 1e-9)
        self.returns_history.append(returns)
        self.equity_history.append(post_equity)
        if self.benchmark_mode == 'close_to_next_open':
            curr_close = torch.tensor(market_data['Close'], dtype=torch.float32, device=self.device)
            bench_r = (exec_base_price - curr_close) / (curr_close + 1e-9)
            if isinstance(bench_r, torch.Tensor) and bench_r.dim() == 0:
                bench_r_val = bench_r.item()
                bench_vec = torch.full((self.chromosomes_size,), bench_r_val, dtype=torch.float32, device=self.device)
            else:
                bench_vec = bench_r if isinstance(bench_r, torch.Tensor) else torch.full((self.chromosomes_size,), float(bench_r), dtype=torch.float32, device=self.device)
            self.bench_returns_history.append(bench_vec)

        max_history = max(self.volatility_lookback, self.lookback_period)
        if len(self.returns_history) > max_history:
            self.returns_history.pop(0)
            self.equity_history.pop(0)
        if len(self.bench_returns_history) > max_history:
            self.bench_returns_history.pop(0)

    def _apply_funding_fees(self, current_timestamp, funding_rate):
        """원본 펀딩비 로직 완전 보존"""
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

    def _check_liquidation(self, mark_low, mark_high):
        """원본 청산 로직 완전 보존"""
        # Liquidation check using mark price approximation (not last price).
        # Conservative proxy: longs tested against bar low, shorts against bar high.
        # 1. 롱 포지션 청산 검사 (가격 하락 시 위험 -> mark_low 기준)
        long_mask = self.pos_list == 2
        if long_mask.any():
            pos_price = self.price_list[long_mask]
            leverage = self.leverage_ratio[long_mask].float()
            enter_ratio = self.enter_ratio[long_mask]

            pnl_percent = (mark_low - pos_price) / pos_price
            margin_balance = enter_ratio * (1 + pnl_percent * leverage)
            maintenance_margin = (enter_ratio * leverage) * self.maintenance_margin_rate
            
            liquidation_mask_local = margin_balance <= maintenance_margin
            if liquidation_mask_local.any():
                long_indices = torch.where(long_mask)[0]
                liquidated_indices = long_indices[liquidation_mask_local]
                # Initial margin loss
                self.profit[liquidated_indices] -= self.enter_ratio[liquidated_indices]
                # Liquidation clearance fee: notional * rate ≈ (enter_ratio * leverage) * rate
                notional = self.enter_ratio[liquidated_indices] * self.leverage_ratio[liquidated_indices].float()
                liq_fee = notional * self.liq_clearance_fee_rate
                self.profit[liquidated_indices] -= liq_fee
                # Trade-based accounting: add realized loss then settle trade
                self.trade_realized_pnl[liquidated_indices] += (-self.enter_ratio[liquidated_indices] - liq_fee)
                # Settle trade
                self.trade_total_count[liquidated_indices] += 1
                win_mask = self.trade_realized_pnl[liquidated_indices] > 0
                if win_mask.any():
                    idx_win = liquidated_indices[win_mask]
                    self.trade_win_count[idx_win] += 1
                # Record trade-close monitoring stats
                self._record_trade_close(liquidated_indices, self.trade_realized_pnl[liquidated_indices])
                self.trade_holding_avg[liquidated_indices] = self.trade_holding_sum[liquidated_indices] / torch.clamp(self.trade_total_count[liquidated_indices].float(), min=1.0)
                self.trade_open[liquidated_indices] = False
                self.trade_realized_pnl[liquidated_indices] = 0.0
                self.pos_list[liquidated_indices] = 0
                self.price_list[liquidated_indices] = -1.0
                self.leverage_ratio[liquidated_indices] = -1
                self.enter_ratio[liquidated_indices] = -1.0
                self.additional_count[liquidated_indices] = 0
                self.holding_period[liquidated_indices] = 0

        # 2. 숏 포지션 청산 검사 (가격 상승 시 위험 -> mark_high 기준)
        short_mask = self.pos_list == 1
        if short_mask.any():
            pos_price = self.price_list[short_mask]
            leverage = self.leverage_ratio[short_mask].float()
            enter_ratio = self.enter_ratio[short_mask]
            
            pnl_percent = (pos_price - mark_high) / pos_price
            margin_balance = enter_ratio * (1 + pnl_percent * leverage)
            maintenance_margin = (enter_ratio * leverage) * self.maintenance_margin_rate

            liquidation_mask_local = margin_balance <= maintenance_margin
            if liquidation_mask_local.any():
                short_indices = torch.where(short_mask)[0]
                liquidated_indices = short_indices[liquidation_mask_local]
                # Initial margin loss
                self.profit[liquidated_indices] -= self.enter_ratio[liquidated_indices]
                # Liquidation clearance fee
                notional = self.enter_ratio[liquidated_indices] * self.leverage_ratio[liquidated_indices].float()
                liq_fee = notional * self.liq_clearance_fee_rate
                self.profit[liquidated_indices] -= liq_fee
                # Trade-based accounting and settlement
                self.trade_realized_pnl[liquidated_indices] += (-self.enter_ratio[liquidated_indices] - liq_fee)
                self.trade_total_count[liquidated_indices] += 1
                win_mask = self.trade_realized_pnl[liquidated_indices] > 0
                if win_mask.any():
                    idx_win = liquidated_indices[win_mask]
                    self.trade_win_count[idx_win] += 1
                # Record trade-close monitoring stats
                self._record_trade_close(liquidated_indices, self.trade_realized_pnl[liquidated_indices])
                self.trade_holding_avg[liquidated_indices] = self.trade_holding_sum[liquidated_indices] / torch.clamp(self.trade_total_count[liquidated_indices].float(), min=1.0)
                self.trade_open[liquidated_indices] = False
                self.trade_realized_pnl[liquidated_indices] = 0.0
                self.pos_list[liquidated_indices] = 0
                self.price_list[liquidated_indices] = -1.0
                self.leverage_ratio[liquidated_indices] = -1
                self.enter_ratio[liquidated_indices] = -1.0
                self.additional_count[liquidated_indices] = 0
                self.holding_period[liquidated_indices] = 0
    
    # === 원본 액션 실행 로직 완전 보존 ===
    def _open_new_positions(self, actions_tensor, currently_hold, buy_exec_price, sell_exec_price):
        """원본 신규 포지션 로직 완전 보존"""
        for action_type, target_pos, exec_price in [
            (ACTION_NEW_LONG, 2, buy_exec_price),
            (ACTION_NEW_SHORT, 1, sell_exec_price),
        ]:
            action_mask = (actions_tensor[:, 0] == action_type) & currently_hold
            if not action_mask.any():
                continue
            enter_ratio = actions_tensor[action_mask, 1]
            # Enforce minimum entry ratio and cap at remaining capacity (for new pos, remaining is 1.0)
            enter_ratio = torch.clamp(enter_ratio, min=self.min_entry_ratio, max=1.0)
            # Enforce leverage bounds
            leverage = torch.clamp(
                actions_tensor[action_mask, 2].long(),
                min=self.min_leverage,
                max=self.max_leverage
            )
            fee = enter_ratio * leverage.float() * self.taker_fee_rate
            self.profit[action_mask] -= fee
            self.pos_list[action_mask] = target_pos
            self.price_list[action_mask] = exec_price
            self.leverage_ratio[action_mask] = leverage
            self.enter_ratio[action_mask] = enter_ratio
            self.additional_count[action_mask] = 0
            # Trade-based: open new trade
            self.trade_open[action_mask] = True
            self.trade_realized_pnl[action_mask] = 0.0

    def _close_all_positions(self, actions_tensor, currently_long, currently_short, buy_exec_price, sell_exec_price):
        """원본 전량 청산 로직 완전 보존"""
        action_mask = (actions_tensor[:, 0] == ACTION_CLOSE_ALL) & (self.pos_list != 0)
        if not action_mask.any():
            return
        long_close_mask = action_mask & currently_long
        if long_close_mask.any():
            entry_price = self.price_list[long_close_mask]
            leverage = self.leverage_ratio[long_close_mask].float()
            pnl = ((sell_exec_price - entry_price) / entry_price) * leverage * self.enter_ratio[long_close_mask]
            fee = self.enter_ratio[long_close_mask] * leverage * self.taker_fee_rate
            realized = pnl - fee
            self.profit[long_close_mask] += realized
            # Trade-based: accumulate and settle
            self.trade_realized_pnl[long_close_mask] += realized
            self.trade_total_count[long_close_mask] += 1
            win_mask = self.trade_realized_pnl[long_close_mask] > 0
            if win_mask.any():
                idx_win = torch.where(long_close_mask)[0][win_mask]
                self.trade_win_count[idx_win] += 1
            # Record trade-close monitoring stats
            idxs_close = torch.where(long_close_mask)[0]
            self._record_trade_close(idxs_close, self.trade_realized_pnl[idxs_close])
            self.trade_holding_avg[idxs_close] = self.trade_holding_sum[idxs_close] / torch.clamp(self.trade_total_count[idxs_close].float(), min=1.0)
            self.trade_open[long_close_mask] = False
            self.trade_realized_pnl[long_close_mask] = 0.0
        short_close_mask = action_mask & currently_short
        if short_close_mask.any():
            entry_price = self.price_list[short_close_mask]
            leverage = self.leverage_ratio[short_close_mask].float()
            pnl = ((entry_price - buy_exec_price) / entry_price) * leverage * self.enter_ratio[short_close_mask]
            fee = self.enter_ratio[short_close_mask] * leverage * self.taker_fee_rate
            realized = pnl - fee
            self.profit[short_close_mask] += realized
            # Trade-based: accumulate and settle
            self.trade_realized_pnl[short_close_mask] += realized
            self.trade_total_count[short_close_mask] += 1
            win_mask = self.trade_realized_pnl[short_close_mask] > 0
            if win_mask.any():
                idx_win = torch.where(short_close_mask)[0][win_mask]
                self.trade_win_count[idx_win] += 1
            # Record trade-close monitoring stats
            idxs_close = torch.where(short_close_mask)[0]
            self._record_trade_close(idxs_close, self.trade_realized_pnl[idxs_close])
            self.trade_holding_avg[idxs_close] = self.trade_holding_sum[idxs_close] / torch.clamp(self.trade_total_count[idxs_close].float(), min=1.0)
            self.trade_open[short_close_mask] = False
            self.trade_realized_pnl[short_close_mask] = 0.0
        self.pos_list[action_mask] = 0
        self.price_list[action_mask] = -1.0
        self.leverage_ratio[action_mask] = -1
        self.enter_ratio[action_mask] = -1.0
        self.additional_count[action_mask] = 0

    def _close_partial_positions(self, actions_tensor, currently_long, currently_short, buy_exec_price, sell_exec_price):
        """원본 부분 청산 로직 완전 보존"""
        action_mask = (actions_tensor[:, 0] == ACTION_CLOSE_PARTIAL) & (self.pos_list != 0)
        if not action_mask.any():
            return
        long_p_close_mask = action_mask & currently_long
        if long_p_close_mask.any():
            idxs = torch.where(long_p_close_mask)[0]
            ratio = actions_tensor[long_p_close_mask, 1]
            entry_price = self.price_list[long_p_close_mask]
            leverage = self.leverage_ratio[long_p_close_mask].float()
            current_er = self.enter_ratio[long_p_close_mask]
            remaining_after = current_er * (1.0 - ratio)
            close_all_sub = remaining_after <= self.min_entry_ratio
            partial_sub = ~close_all_sub

            # Full close for those that would leave <= min_entry_ratio
            if close_all_sub.any():
                cidx = idxs[close_all_sub]
                eprice = self.price_list[cidx]
                lev = self.leverage_ratio[cidx].float()
                er = self.enter_ratio[cidx]
                pnl = ((sell_exec_price - eprice) / eprice) * lev * er
                fee = er * lev * self.taker_fee_rate
                realized = pnl - fee
                self.profit[cidx] += realized
                # Trade-based: accumulate and settle
                self.trade_realized_pnl[cidx] += realized
                self.trade_total_count[cidx] += 1
                win_mask = self.trade_realized_pnl[cidx] > 0
                if win_mask.any():
                    self.trade_win_count[cidx[win_mask]] += 1
                # Record trade-close monitoring stats
                self._record_trade_close(cidx, self.trade_realized_pnl[cidx])
                self.trade_holding_avg[cidx] = self.trade_holding_sum[cidx] / torch.clamp(self.trade_total_count[cidx].float(), min=1.0)
                self.trade_open[cidx] = False
                self.trade_realized_pnl[cidx] = 0.0
                self.pos_list[cidx] = 0
                self.price_list[cidx] = -1.0
                self.leverage_ratio[cidx] = -1
                self.enter_ratio[cidx] = -1.0
                self.additional_count[cidx] = 0

            # Partial close for the rest
            if partial_sub.any():
                pidx = idxs[partial_sub]
                pratio = ratio[partial_sub]
                eprice = entry_price[partial_sub]
                lev = leverage[partial_sub]
                closed_margin = self.enter_ratio[pidx] * pratio
                pnl = ((sell_exec_price - eprice) / eprice) * lev * closed_margin
                fee = closed_margin * lev * self.taker_fee_rate
                realized = pnl - fee
                self.profit[pidx] += realized
                # Trade-based: partial close accumulates only
                self.trade_realized_pnl[pidx] += realized
                self.enter_ratio[pidx] -= closed_margin
        short_p_close_mask = action_mask & currently_short
        if short_p_close_mask.any():
            idxs = torch.where(short_p_close_mask)[0]
            ratio = actions_tensor[short_p_close_mask, 1]
            entry_price = self.price_list[short_p_close_mask]
            leverage = self.leverage_ratio[short_p_close_mask].float()
            current_er = self.enter_ratio[short_p_close_mask]
            remaining_after = current_er * (1.0 - ratio)
            close_all_sub = remaining_after <= self.min_entry_ratio
            partial_sub = ~close_all_sub

            # Full close for those that would leave <= min_entry_ratio
            if close_all_sub.any():
                cidx = idxs[close_all_sub]
                eprice = self.price_list[cidx]
                lev = self.leverage_ratio[cidx].float()
                er = self.enter_ratio[cidx]
                pnl = ((eprice - buy_exec_price) / eprice) * lev * er
                fee = er * lev * self.taker_fee_rate
                realized = pnl - fee
                self.profit[cidx] += realized
                # Trade-based: accumulate and settle
                self.trade_realized_pnl[cidx] += realized
                self.trade_total_count[cidx] += 1
                win_mask = self.trade_realized_pnl[cidx] > 0
                if win_mask.any():
                    self.trade_win_count[cidx[win_mask]] += 1
                # Record trade-close monitoring stats
                self._record_trade_close(cidx, self.trade_realized_pnl[cidx])
                self.trade_holding_avg[cidx] = self.trade_holding_sum[cidx] / torch.clamp(self.trade_total_count[cidx].float(), min=1.0)
                self.trade_open[cidx] = False
                self.trade_realized_pnl[cidx] = 0.0
                self.pos_list[cidx] = 0
                self.price_list[cidx] = -1.0
                self.leverage_ratio[cidx] = -1
                self.enter_ratio[cidx] = -1.0
                self.additional_count[cidx] = 0

            # Partial close for the rest
            if partial_sub.any():
                pidx = idxs[partial_sub]
                pratio = ratio[partial_sub]
                eprice = entry_price[partial_sub]
                lev = leverage[partial_sub]
                closed_margin = self.enter_ratio[pidx] * pratio
                pnl = ((eprice - buy_exec_price) / eprice) * lev * closed_margin
                fee = closed_margin * lev * self.taker_fee_rate
                realized = pnl - fee
                self.profit[pidx] += realized
                # Trade-based: partial close accumulates only
                self.trade_realized_pnl[pidx] += realized
                self.enter_ratio[pidx] -= closed_margin

    def _add_to_positions(self, actions_tensor, currently_long, currently_short, buy_exec_price, sell_exec_price):
        """원본 포지션 추가 로직 완전 보존"""
        action_mask = (actions_tensor[:, 0] == ACTION_ADD_POSITION) & (self.pos_list != 0)
        if not action_mask.any():
            return
        for pos_cond, exec_price in [
            (currently_long, buy_exec_price),
            (currently_short, sell_exec_price),
        ]:
            add_mask = action_mask & pos_cond & (self.additional_count < self.max_additional_entries)
            if not add_mask.any():
                continue
            current_enter_ratio = self.enter_ratio[add_mask]
            current_requested_ratio = actions_tensor[add_mask, 1]
            # Remaining capacity to add
            remaining_capital = torch.clamp(1.0 - current_enter_ratio, min=0.0)
            # Require at least 5% remaining capacity; otherwise skip adding
            capacity_ok_mask = remaining_capital >= self.min_additional_enter_ratio
            if not capacity_ok_mask.any():
                continue
            # Enforce minimum add size (5%) when capacity allows
            desired_add = torch.maximum(
                current_requested_ratio,
                torch.full_like(current_requested_ratio, self.min_additional_enter_ratio)
            )
            actual_add_ratio = torch.min(desired_add, remaining_capital)
            valid_add_mask = (actual_add_ratio > 0) & capacity_ok_mask
            if not valid_add_mask.any():
                continue
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

    def _flip_positions(self, actions_tensor, currently_long, currently_short, buy_exec_price, sell_exec_price):
        """원본 포지션 전환 로직 완전 보존"""
        action_mask = (actions_tensor[:, 0] == ACTION_FLIP_POSITION) & (self.pos_list != 0)
        if not action_mask.any():
            return
        flip_to_short_mask = action_mask & currently_long
        if flip_to_short_mask.any():
            ratio = actions_tensor[flip_to_short_mask, 1]
            # Enforce minimum entry ratio for the new flipped position
            ratio = torch.clamp(ratio, min=self.min_entry_ratio, max=1.0)
            # Enforce leverage bounds for the new flipped position
            lev = torch.clamp(
                actions_tensor[flip_to_short_mask, 2].long(),
                min=self.min_leverage,
                max=self.max_leverage
            )
            entry_price = self.price_list[flip_to_short_mask]
            leverage = self.leverage_ratio[flip_to_short_mask].float()
            pnl = ((sell_exec_price - entry_price) / entry_price) * leverage * self.enter_ratio[flip_to_short_mask]
            fee_close = self.enter_ratio[flip_to_short_mask] * leverage * self.taker_fee_rate
            realized_close = pnl - fee_close
            self.profit[flip_to_short_mask] += realized_close
            # Trade-based: close current trade and settle
            self.trade_realized_pnl[flip_to_short_mask] += realized_close
            self.trade_total_count[flip_to_short_mask] += 1
            win_mask = self.trade_realized_pnl[flip_to_short_mask] > 0
            if win_mask.any():
                idx_win = torch.where(flip_to_short_mask)[0][win_mask]
                self.trade_win_count[idx_win] += 1
            # Record trade-close monitoring stats
            idxs_close = torch.where(flip_to_short_mask)[0]
            self._record_trade_close(idxs_close, self.trade_realized_pnl[idxs_close])
            self.trade_holding_avg[idxs_close] = self.trade_holding_sum[idxs_close] / torch.clamp(self.trade_total_count[idxs_close].float(), min=1.0)
            self.trade_open[flip_to_short_mask] = False
            self.trade_realized_pnl[flip_to_short_mask] = 0.0
            fee_open = ratio * lev.float() * self.taker_fee_rate
            self.profit[flip_to_short_mask] -= fee_open
            self.pos_list[flip_to_short_mask] = 1
            self.price_list[flip_to_short_mask] = sell_exec_price
            self.enter_ratio[flip_to_short_mask] = ratio
            self.leverage_ratio[flip_to_short_mask] = lev
            self.additional_count[flip_to_short_mask] = 0
            # Trade-based: open new trade for flipped position
            self.trade_open[flip_to_short_mask] = True
            self.trade_realized_pnl[flip_to_short_mask] = 0.0
        flip_to_long_mask = action_mask & currently_short
        if flip_to_long_mask.any():
            ratio = actions_tensor[flip_to_long_mask, 1]
            ratio = torch.clamp(ratio, min=self.min_entry_ratio, max=1.0)
            lev = torch.clamp(
                actions_tensor[flip_to_long_mask, 2].long(),
                min=self.min_leverage,
                max=self.max_leverage
            )
            entry_price = self.price_list[flip_to_long_mask]
            leverage = self.leverage_ratio[flip_to_long_mask].float()
            pnl = ((entry_price - buy_exec_price) / entry_price) * leverage * self.enter_ratio[flip_to_long_mask]
            fee_close = self.enter_ratio[flip_to_long_mask] * leverage * self.taker_fee_rate
            realized_close = pnl - fee_close
            self.profit[flip_to_long_mask] += realized_close
            # Trade-based: close and settle
            self.trade_realized_pnl[flip_to_long_mask] += realized_close
            self.trade_total_count[flip_to_long_mask] += 1
            win_mask = self.trade_realized_pnl[flip_to_long_mask] > 0
            if win_mask.any():
                idx_win = torch.where(flip_to_long_mask)[0][win_mask]
                self.trade_win_count[idx_win] += 1
            # Record trade-close monitoring stats
            idxs_close = torch.where(flip_to_long_mask)[0]
            self._record_trade_close(idxs_close, self.trade_realized_pnl[idxs_close])
            self.trade_holding_avg[idxs_close] = self.trade_holding_sum[idxs_close] / torch.clamp(self.trade_total_count[idxs_close].float(), min=1.0)
            self.trade_open[flip_to_long_mask] = False
            self.trade_realized_pnl[flip_to_long_mask] = 0.0
            fee_open = ratio * lev.float() * self.taker_fee_rate
            self.profit[flip_to_long_mask] -= fee_open
            self.pos_list[flip_to_long_mask] = 2
            self.price_list[flip_to_long_mask] = buy_exec_price
            self.enter_ratio[flip_to_long_mask] = ratio
            self.leverage_ratio[flip_to_long_mask] = lev
            self.additional_count[flip_to_long_mask] = 0
            # Trade-based: open new trade for flipped position
            self.trade_open[flip_to_long_mask] = True
            self.trade_realized_pnl[flip_to_long_mask] = 0.0


    def _execute_actions(self, actions_tensor, exec_base_price):
        """CUDA 커널로 액션 실행 (Thread-per-individual)."""
        try:
            import gatree_cuda_compat as gatree_cuda
        except Exception:
            gatree_cuda = None
        if gatree_cuda is None or getattr(gatree_cuda, 'execute_actions_batch', None) is None:
            # Fallback to CPU path if CUDA extension not available
            return False

        # Only run on CUDA
        if not str(self.device).startswith('cuda'):
            return False

        # Ensure tensors are on CUDA and contiguous
        actions = actions_tensor.to(self.device).contiguous()
        buy_exec_price = float((exec_base_price * (1 + self.fixed_slippage_rate)).item() if isinstance(exec_base_price, torch.Tensor) else exec_base_price * (1 + self.fixed_slippage_rate))
        sell_exec_price = float((exec_base_price * (1 - self.fixed_slippage_rate)).item() if isinstance(exec_base_price, torch.Tensor) else exec_base_price * (1 - self.fixed_slippage_rate))

        # CPU 비교 디버그는 제거됨 (Python 경로 삭제)

        # Launch CUDA batch executor
        gatree_cuda.execute_actions_batch(
            actions,
            self.pos_list, self.price_list, self.leverage_ratio, self.enter_ratio,
            self.additional_count, self.profit,
            self.trade_open, self.trade_realized_pnl, self.trade_total_count, self.trade_win_count,
            self.trade_holding_sum, self.trade_holding_avg,
            self.curr_win_streak, self.curr_loss_streak, self.max_win_streak, self.max_loss_streak,
            self.holding_period,
            float(buy_exec_price), float(sell_exec_price),
            float(self.taker_fee_rate), float(self.min_entry_ratio),
            int(self.min_leverage), int(self.max_leverage),
            float(self.min_additional_enter_ratio), int(self.max_additional_entries)
        )

        # no-op

        return True

    def _calculate_current_equity(self, market_data) -> torch.Tensor:
        """현재 에쿼티 계산 (미실현 손익 포함)"""
        curr_high = torch.tensor(market_data['High'], dtype=torch.float32, device=self.device)
        curr_low = torch.tensor(market_data['Low'], dtype=torch.float32, device=self.device)
        curr_close = torch.tensor(market_data['Close'], dtype=torch.float32, device=self.device)
        mark_t = (curr_high + curr_low + curr_close) / 3.0
        
        # Calculate unrealized P&L for open positions
        open_mask = self.pos_list != 0
        unrealized_pnl = torch.zeros_like(self.realized_cum)
        
        if open_mask.any():
            side = torch.where(self.pos_list == 2, 1.0, -1.0)  # long=+1, short=-1
            entry = self.price_list
            lev = self.leverage_ratio.float()
            er = self.enter_ratio
            pnl_pct = side * (mark_t - entry) / (entry + 1e-9)
            unrealized_pnl[open_mask] = (pnl_pct * lev * er)[open_mask]
        
        # Total equity = realized + unrealized, starting from 1.0
        equity = 1.0 + self.realized_cum + unrealized_pnl
        return equity

    def step(self, market_data, next_open_price, predicted_actions):
        """
        원본 step 함수 완전 보존 + 다중 목표용 히스토리 추가 추적
        """
        pre_equity = self._calculate_current_equity(market_data)
        state = self._extract_step_state(market_data, next_open_price)
        self._apply_funding_fees(state.timestamp, state.funding_rate)
        self._check_liquidation(state.low, state.high)
        actions, raw_actions = self._prepare_and_deduplicate_actions(predicted_actions)
        pos_before, er_before = self._snapshot_state_for_dedup()
        self._execute_actions_wrapper(actions, state.exec_base_price)
        self._update_holding_periods()
        self._accumulate_realized_pnl_and_stats()
        total_unreal = self._compute_unrealized_pnl(state.mark_t)
        self._update_equity_and_drawdown(total_unreal, state.timestamp)
        self._update_dedup_buffers_after_exec(actions, raw_actions, pos_before, er_before)
        self._update_return_histories(pre_equity, market_data, state.exec_base_price)

    def get_final_metrics(self, minimum_date=40):
        """원본 메트릭 계산 완전 보존"""
        count_returns_f = self.count_returns.float()
        mean_returns = torch.where(count_returns_f > 0, self.sum_returns / count_returns_f, torch.full_like(self.sum_returns, -1e9))
        
        # Profit Factor 지표 왜곡 방지
        profit_factors = torch.full_like(self.total_profit_agg, 100.0)
        has_loss_mask = self.total_loss_agg > 0
        if has_loss_mask.any():
            calculated_pf = self.total_profit_agg[has_loss_mask] / (self.total_loss_agg[has_loss_mask] + 1e-9)
            profit_factors[has_loss_mask] = torch.clamp(calculated_pf, max=100.0)

        win_rates = torch.where(count_returns_f > 0, self.count_wins.float() / count_returns_f, torch.full_like(count_returns_f, -1e9))
        # Trade-based win rate (per open->final close)
        trade_counts_f = self.trade_total_count.float()
        trade_win_rates = torch.where(trade_counts_f > 0, self.trade_win_count.float() / trade_counts_f, torch.full_like(trade_counts_f, -1e9))
        invalid_mask = self.count_returns < minimum_date
        mean_returns[invalid_mask] = -1e9
        profit_factors[invalid_mask] = -1e9
        win_rates[invalid_mask] = -1e9
        self.max_drawdown[invalid_mask] = 1e9
        self.compound_value[invalid_mask] = -1e9
        # Append trade-based win rate as an extra column for reporting
        metrics = torch.stack([mean_returns, profit_factors, win_rates, self.max_drawdown, self.compound_value, trade_win_rates], dim=1)
        return metrics.cpu().numpy()

    # === 다중 목표 최적화 추가 메서드들 ===
    def _calculate_sharpe_ratio(self, returns: List[torch.Tensor]) -> torch.Tensor:
        """샤프 비율 계산"""
        if len(returns) < 2:
            return torch.zeros(self.chromosomes_size, device=self.device)
        
        returns_tensor = torch.stack(returns)  # [time_steps, chromosomes_size]
        
        # Calculate excess returns (assuming risk-free rate is annualized)
        period_rf_rate = self.risk_free_rate / self.lookback_period
        excess_returns = returns_tensor - period_rf_rate
        
        # Calculate mean and std of excess returns
        mean_excess = excess_returns.mean(dim=0)  # [chromosomes_size]
        std_excess = excess_returns.std(dim=0, unbiased=True)  # [chromosomes_size]
        
        # Sharpe ratio with numerical stability
        sharpe_ratios = torch.where(
            std_excess > 1e-8,
            mean_excess / std_excess * np.sqrt(self.lookback_period),  # Annualized
            torch.zeros_like(mean_excess)
        )
        
        return sharpe_ratios
    
    def _calculate_volatility(self, returns: List[torch.Tensor]) -> torch.Tensor:
        """변동성 계산"""
        if len(returns) < 2:
            return torch.zeros(self.chromosomes_size, device=self.device)
        
        returns_tensor = torch.stack(returns)  # [time_steps, chromosomes_size]
        volatilities = returns_tensor.std(dim=0, unbiased=True) * np.sqrt(self.lookback_period)
        
        return volatilities
    
    def _calculate_calmar_ratio(self, returns: List[torch.Tensor]) -> torch.Tensor:
        """칼마 비율 계산 (연간 수익률 / 최대 드로다운)"""
        if len(returns) < 2:
            return torch.zeros(self.chromosomes_size, device=self.device)
        
        returns_tensor = torch.stack(returns)  # [time_steps, chromosomes_size]
        
        # Annualized return
        total_return = (1 + returns_tensor).prod(dim=0) - 1
        annualized_return = (1 + total_return) ** (self.lookback_period / len(returns)) - 1
        
        # Max drawdown (from current max_drawdown tracking)
        max_dd = torch.clamp(self.max_drawdown, min=1e-8)  # Avoid division by zero
        
        calmar_ratios = annualized_return / max_dd
        
        return calmar_ratios
    
    def _calculate_sortino_ratio(self, returns: List[torch.Tensor]) -> torch.Tensor:
        """소르티노 비율 계산 (하방 편차 사용)"""
        if len(returns) < 2:
            return torch.zeros(self.chromosomes_size, device=self.device)
        
        returns_tensor = torch.stack(returns)  # [time_steps, chromosomes_size]
        
        # Calculate excess returns
        period_rf_rate = self.risk_free_rate / self.lookback_period
        excess_returns = returns_tensor - period_rf_rate
        
        # Downside deviation (only negative returns)
        negative_returns = torch.where(excess_returns < 0, excess_returns, torch.zeros_like(excess_returns))
        downside_std = torch.sqrt(torch.mean(negative_returns ** 2, dim=0))
        
        # Mean excess return
        mean_excess = excess_returns.mean(dim=0)
        
        # Sortino ratio with numerical stability
        sortino_ratios = torch.where(
            downside_std > 1e-8,
            mean_excess / downside_std * np.sqrt(self.lookback_period),  # Annualized
            torch.zeros_like(mean_excess)
        )
        
        return sortino_ratios

    def _calculate_omega_ratio(self, returns: List[torch.Tensor], mar: float) -> torch.Tensor:
        """오메가 비율 계산 (MAR 기준 이익/손실 비)"""
        if len(returns) < 2:
            return torch.zeros(self.chromosomes_size, device=self.device)
        ret = torch.stack(returns)  # [T, B]
        period_mar = mar / self.lookback_period
        excess = ret - period_mar
        gains = torch.clamp(excess, min=0.0).sum(dim=0)
        losses = torch.clamp(-excess, min=0.0).sum(dim=0)
        omega = torch.where(losses > 1e-12, gains / (losses + 1e-12), torch.zeros_like(gains))
        return omega

    def _calculate_information_ratio(self, returns: List[torch.Tensor], bench_returns: List[torch.Tensor]) -> torch.Tensor:
        """정보 비율 계산 (전략-벤치마크 초과수익 평균 / 추적오차)"""
        if len(returns) < 2 or len(bench_returns) < 2:
            return torch.zeros(self.chromosomes_size, device=self.device)
        t = min(len(returns), len(bench_returns))
        ret = torch.stack(returns[-t:])
        bench = torch.stack(bench_returns[-t:])
        active = ret - bench
        mean_active = active.mean(dim=0)
        tracking_err = active.std(dim=0, unbiased=True)
        ir = torch.where(tracking_err > 1e-8, mean_active / tracking_err * np.sqrt(self.lookback_period), torch.zeros_like(mean_active))
        return ir

    def _calculate_t_stat(self, returns: List[torch.Tensor]) -> torch.Tensor:
        """스텝 수익률의 t-통계량 계산"""
        if len(returns) < 2:
            return torch.zeros(self.chromosomes_size, device=self.device)
        ret = torch.stack(returns)
        mean = ret.mean(dim=0)
        std = ret.std(dim=0, unbiased=True)
        n = ret.shape[0]
        t = torch.where(std > 1e-8, mean / (std / (torch.sqrt(torch.tensor(float(n), device=self.device)))) , torch.zeros_like(mean))
        return t

    def _record_trade_close(self, indices: torch.Tensor, realized_vec: torch.Tensor):
        """거래 종료 시 보유기간 합산 및 연승/연패 스트릭 갱신"""
        if indices.numel() == 0:
            return
        is_win = realized_vec > 0
        if is_win.any():
            win_idx = indices[is_win]
            self.curr_win_streak[win_idx] += 1
            self.max_win_streak[win_idx] = torch.maximum(self.max_win_streak[win_idx], self.curr_win_streak[win_idx])
            self.curr_loss_streak[win_idx] = 0
        if (~is_win).any():
            loss_idx = indices[~is_win]
            self.curr_loss_streak[loss_idx] += 1
            self.max_loss_streak[loss_idx] = torch.maximum(self.max_loss_streak[loss_idx], self.curr_loss_streak[loss_idx])
            self.curr_win_streak[loss_idx] = 0
        # Holding period accumulation for closed trades
        hp = self.holding_period[indices].float()
        self.trade_holding_sum[indices] += hp

    def _calculate_psr(self, returns: List[torch.Tensor], sr_benchmark: float = 0.0) -> torch.Tensor:
        """
        Probabilistic Sharpe Ratio (Bailey & López de Prado) 계산
        - SR_hat: 비연환산 샤프(기간 수익률 기준)
        - S: 왜도, K: (피어슨) 첨도
        - PSR = Phi( ((SR_hat - SR*) * sqrt(N-1)) / sqrt(1 - S*SR_hat + ((K-1)/4)*SR_hat^2) )
        """
        if len(returns) < 3:
            return torch.zeros(self.chromosomes_size, device=self.device)

        # Excess returns (per-step)
        ret = torch.stack(returns)  # [T, B]
        period_rf = self.risk_free_rate / self.lookback_period
        ex = ret - period_rf

        n = ex.shape[0]
        # Sample moments
        mean = ex.mean(dim=0)
        std = ex.std(dim=0, unbiased=True)
        # Guard for zero-variance cases
        safe_mask = std > 1e-12
        sr_hat = torch.zeros(self.chromosomes_size, device=self.device)
        sr_hat[safe_mask] = mean[safe_mask] / std[safe_mask]

        # Centralize for higher moments
        xc = ex - mean
        m2 = (xc.pow(2).sum(dim=0)) / (n - 1)
        m3 = (xc.pow(3).sum(dim=0)) / (n - 1)
        m4 = (xc.pow(4).sum(dim=0)) / (n - 1)

        # Skewness and (Pearson) kurtosis
        # S = m3 / m2^(3/2), K = m4 / m2^2
        eps = 1e-12
        denom_s = torch.clamp(m2, min=eps).pow(1.5)
        denom_k = torch.clamp(m2, min=eps).pow(2.0)
        S = m3 / denom_s
        K = m4 / denom_k

        # Denominator of PSR expression
        denom = torch.sqrt(torch.clamp(1.0 - S * sr_hat + ((K - 1.0) / 4.0) * (sr_hat ** 2), min=eps))
        num = (sr_hat - sr_benchmark) * torch.sqrt(torch.tensor(float(n - 1), device=self.device))

        z = torch.zeros_like(sr_hat)
        valid = denom > 0
        z[valid] = num[valid] / denom[valid]

        # Standard normal CDF via erf
        psr = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
        psr = torch.clamp(psr, 0.0, 1.0)
        psr[~safe_mask] = 0.0
        return psr

    def get_multi_objective_metrics(self, minimum_date: int = 40) -> np.ndarray:
        """
        다중 목표 메트릭 계산 (원본 메트릭 + 고급 메트릭)
        """
        # Get base metrics from original method
        base_metrics = self.get_final_metrics(minimum_date)
        
        # base_metrics columns: [mean_returns, profit_factors, win_rates, max_drawdown, compound_value, trade_win_rates]
        mean_returns = torch.tensor(base_metrics[:, 0], device=self.device)
        profit_factors = torch.tensor(base_metrics[:, 1], device=self.device)
        win_rates = torch.tensor(base_metrics[:, 2], device=self.device)
        max_drawdown = torch.tensor(base_metrics[:, 3], device=self.device)
        compound_value = torch.tensor(base_metrics[:, 4], device=self.device)
        trade_win_rates = torch.tensor(base_metrics[:, 5], device=self.device)
        
        # Calculate additional advanced metrics if we have return history
        if len(self.returns_history) >= 2:
            sharpe_ratios = self._calculate_sharpe_ratio(self.returns_history)
            volatilities = self._calculate_volatility(self.returns_history)
            calmar_ratios = self._calculate_calmar_ratio(self.returns_history)
            sortino_ratios = self._calculate_sortino_ratio(self.returns_history)
            omega_ratios = self._calculate_omega_ratio(self.returns_history, self.mar)
            information_ratios = self._calculate_information_ratio(self.returns_history, self.bench_returns_history)
        else:
            # Fallback for insufficient data
            sharpe_ratios = torch.zeros_like(mean_returns)
            volatilities = torch.zeros_like(mean_returns)
            calmar_ratios = torch.zeros_like(mean_returns)
            sortino_ratios = torch.zeros_like(mean_returns)
            omega_ratios = torch.zeros_like(mean_returns)
            information_ratios = torch.zeros_like(mean_returns)
        
        # Build objective dictionary
        available_objectives = {
            'mean_return': mean_returns,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratios,
            'win_rate': win_rates,
            'profit_factor': profit_factors,
            'compound_value': compound_value,
            'trade_win_rate': trade_win_rates,
            'volatility': volatilities,
            'calmar_ratio': calmar_ratios,
            'sortino_ratio': sortino_ratios,
            'omega_ratio': omega_ratios,
            'information_ratio': information_ratios
        }
        
        # Select and arrange objectives according to configuration
        objectives_list = []
        for i, obj_name in enumerate(self.objective_names):
            if obj_name not in available_objectives:
                raise ValueError(f"Unknown objective: {obj_name}. Available: {list(available_objectives.keys())}")
            
            objective_values = available_objectives[obj_name]
            
            # Apply minimization/maximization transformation
            if self.minimize_objectives[i]:
                # For minimization objectives, use as-is (assuming lower is better)
                objectives_list.append(objective_values)
            else:
                # For maximization objectives, negate for minimization framework
                objectives_list.append(-objective_values)
        
        # Stack objectives
        objectives_tensor = torch.stack(objectives_list, dim=1)  # [chromosomes_size, num_objectives]
        
        # Handle invalid individuals (same logic as base class)
        invalid_mask = torch.tensor(base_metrics[:, 0] == -1e9, device=self.device)
        
        # Set invalid individuals to worst possible values for each objective
        for i in range(len(self.objective_names)):
            if self.minimize_objectives[i]:
                # For minimization: set to large positive value
                objectives_tensor[invalid_mask, i] = 1e9
            else:
                # For maximization (negated): set to large positive value (worst for negated)
                objectives_tensor[invalid_mask, i] = 1e9
        
        return objectives_tensor.cpu().numpy()

    def get_objective_names(self) -> List[str]:
        """설정된 목표 이름들 반환"""
        return self.objective_names.copy()
    
    def get_minimization_flags(self) -> List[bool]:
        """각 목표의 최소화 플래그 반환"""
        return self.minimize_objectives.copy()
