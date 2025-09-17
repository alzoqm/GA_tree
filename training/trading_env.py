# /training/trading_env.py
# Unified TradingEnvironment with complete original logic + multi-objective capabilities

import torch
import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from models.model import GATreePop
from models.constants import (
    ACTION_NOT_FOUND,
    ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_CLOSE_ALL,
    ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION, ACTION_FLIP_POSITION
)

import gatree_cuda_compat as gatree_cuda
 
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
                 objectives_config: Optional[dict] = None, device='cpu',
                 max_nodes: Optional[int] = None):
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
        self.max_nodes = int(max_nodes) if max_nodes is not None else int(env_config.get('max_nodes', 0))
        
        # === 원본 거래 환경 설정 완전 보존 ===
        self.taker_fee_rate = env_config['taker_fee_rate']
        self.maintenance_margin_rate = env_config['maintenance_margin_rate']
        
        # === 슬리피지 설정 (동적 슬리피지 지원) ===
        slippage_config = env_config.get('slippage', {})
        self.slippage_mode = slippage_config.get('mode', 'fixed')
        
        if self.slippage_mode == 'fixed':
            # 기존 고정 슬리피지 (backward compatibility)
            self.fixed_slippage_rate = slippage_config.get('fixed_rate', env_config.get('fixed_slippage_rate', 0.0005))
        elif self.slippage_mode == 'relative_volatility':
            # 상대 변동성 기반 동적 슬리피지
            self.base_slippage_rate = slippage_config.get('base_rate', 0.0001)
            self.atr_ma_window = slippage_config.get('atr_ma_window', 50)
            self.min_slippage = slippage_config.get('min_slippage', 0.00005)
            self.max_slippage = slippage_config.get('max_slippage', 0.002)
            
            # ATR 컬럼 설정 (사용자 지정 또는 자동 감지)
            self.atr_column_config = slippage_config.get('atr_column')
            self.atr_ma_column_config = slippage_config.get('atr_ma_column')
            self.atr_pattern = slippage_config.get('atr_pattern', r'ATR_\d+.*')
            self.atr_ma_pattern = slippage_config.get('atr_ma_pattern', r'ATR_\d+.*_MA_\d+')
            
            # 실제 컬럼명은 첫 번째 step에서 감지됨
            self.atr_column = None
            self.atr_ma_column = None
            self._columns_detected = False
            # If ATR-MA column is absent in data, compute runtime rolling mean
            self._use_runtime_atr_ma = False
            self._atr_history = None
        else:
            raise ValueError(f"Unknown slippage mode: {self.slippage_mode}")
        
        # 기존 호환성을 위해 fixed_slippage_rate를 유지 (fixed 모드에서만 사용)
        if not hasattr(self, 'fixed_slippage_rate'):
            self.fixed_slippage_rate = 0.0005
        # Mark price / equity options
        self.use_real_mark_price = env_config.get('use_real_mark_price', False)
        self.equity_mtm_basis = env_config.get('equity_mtm_basis', 'mark_proxy')  # 'mark_price' | 'mark_proxy'
        # Liquidation clearance fee (insurance fund/clearance)
        self.liq_clearance_fee_rate = env_config.get('liq_clearance_fee_rate', 0.0)
        self.funding_fee_hours = {0, 8, 16}
        # Minimum constraints
        self.min_entry_ratio = float(env_config.get('min_entry_ratio', 0.05))
        self.max_entry_ratio = float(env_config.get('max_entry_ratio', 1.0))
        if not (0.0 < self.min_entry_ratio <= 1.0):
            raise ValueError(f"min_entry_ratio must be in (0, 1], got {self.min_entry_ratio}")
        if self.max_entry_ratio <= 0:
            raise ValueError(f"max_entry_ratio must be positive, got {self.max_entry_ratio}")
        if self.max_entry_ratio < self.min_entry_ratio:
            raise ValueError(
                f"max_entry_ratio ({self.max_entry_ratio}) must be >= min_entry_ratio ({self.min_entry_ratio})"
            )
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
        self.equity_base = float(env_config.get('equity_base', 1.0))
        self.sum_returns = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.count_returns = torch.zeros(chromosomes_size, device=device, dtype=torch.int32)
        self.sum_sq_returns = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.sum_neg = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.count_neg = torch.zeros(chromosomes_size, device=device, dtype=torch.int32)
        self.sum_sq_neg = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.total_profit_agg = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.total_loss_agg = torch.zeros(chromosomes_size, device=device, dtype=torch.float32)
        self.count_wins = torch.zeros(chromosomes_size, device=device, dtype=torch.int32)
        self.cum_sum = torch.full(
            (chromosomes_size,), self.equity_base, device=device, dtype=torch.float32
        )
        self.running_max = torch.full(
            (chromosomes_size,), self.equity_base, device=device, dtype=torch.float32
        )
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
                'objectives': ['mean_return', 'max_drawdown', 'psr', 'win_rate'],
                'minimize': [False, True, False, False],
                'risk_free_rate': 0.0,
                'volatility_lookback': 30
            }
        
        self.objectives_config = objectives_config
        self.objective_names = objectives_config.get('objectives', [
            'mean_return', 'max_drawdown', 'psr', 'win_rate'
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
        self._dedup_exclude_tensor_cuda: Optional[torch.Tensor] = None
        if self.enable_action_dedup and str(self.device).startswith('cuda'):
            self._refresh_dedup_exclude_buffers()

        # === CUDA 실행 옵션 ===
        # debug_compare_cpu_gpu는 더 이상 사용하지 않지만, 호환을 위해 남겨둡니다.
        self.debug_compare_cpu_gpu = bool(env_config.get('debug_compare_cpu_gpu', False))

        # 이전 스텝의 원본 예측 액션 저장 (B,4)
        self.prev_actions_raw = torch.full(
            (chromosomes_size, 4), float('nan'), dtype=torch.float32, device=device
        )
        # 직전 스텝에서 실제 적용된 개체 마스크
        self.prev_applied_mask = torch.zeros(chromosomes_size, dtype=torch.bool, device=device)

        # === Action-node attribution buffers ===
        if self.max_nodes and self.max_nodes > 0 and str(self.device).startswith('cuda'):
            B = self.chromosomes_size
            N = self.max_nodes
            i32 = dict(dtype=torch.int32, device=self.device)
            f32 = dict(dtype=torch.float32, device=self.device)
            b8  = dict(dtype=torch.bool, device=self.device)
            i8  = dict(dtype=torch.int8, device=self.device)
            # Per-node accumulators
            self.node_attempt_count = torch.zeros((B, N), **i32)
            self.node_apply_count   = torch.zeros((B, N), **i32)
            self.node_noop_count    = torch.zeros((B, N), **i32)
            self.node_stall_count   = torch.zeros((B, N), **i32)
            self.node_win_count     = torch.zeros((B, N), **i32)
            self.node_realized_pnl_sum = torch.zeros((B, N), **f32)
            self.node_exposure_sum     = torch.zeros((B, N), **f32)
            # Lot ledger (L = N)
            L = N
            self.lot_open      = torch.zeros((B, L), **b8)
            self.lot_node_idx  = torch.full((B, L), -1, **i32)
            self.lot_side      = torch.zeros((B, L), **i8)
            self.lot_entry     = torch.full((B, L), -1.0, **f32)
            self.lot_exposure  = torch.zeros((B, L), **f32)
            self.lot_open_step = torch.zeros((B, L), **i32)
            # Last predicted node indices (per step)
            self._last_action_node_idx = torch.full((B,), -1, **i32)
            self._last_dedup_mask = torch.zeros((B,), dtype=torch.bool, device=self.device)
            # Step counter for holding periods
            self._sim_step = 0
        else:
            # Fallback placeholders on CPU or when max_nodes unknown
            self.node_attempt_count = None
            self.node_apply_count = None
            self.node_noop_count = None
            self.node_win_count = None
            self.node_realized_pnl_sum = None
            self.node_exposure_sum = None
            self.lot_open = None
            self.lot_node_idx = None
            self.lot_side = None
            self.lot_entry = None
            self.lot_exposure = None
            self.lot_open_step = None
            self._last_action_node_idx = None
            self._last_dedup_mask = None
            self._sim_step = 0

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
        mark_low: torch.Tensor
        mark_high: torch.Tensor
        exec_base_price: torch.Tensor | float

    def _extract_step_state(self, market_data, next_open_price) -> "TradingEnvironment.StepState":
        ts = market_data.name
        high = torch.tensor(market_data['High'], dtype=torch.float32, device=self.device)
        low = torch.tensor(market_data['Low'], dtype=torch.float32, device=self.device)
        close = torch.tensor(market_data['Close'], dtype=torch.float32, device=self.device)
        funding_rate = torch.tensor(market_data['fundingRate'], dtype=torch.float32, device=self.device)

        def _safe_get(key, default):
            if key in market_data and pd.notna(market_data[key]):
                return float(market_data[key])
            return float(default)

        typical = (float(market_data['High']) + float(market_data['Low']) + float(market_data['Close'])) / 3.0
        mark_t = torch.tensor(_safe_get('markPrice_est', typical), dtype=torch.float32, device=self.device)
        mark_low = torch.tensor(_safe_get('markLow_est', float(market_data['Low'])), dtype=torch.float32, device=self.device)
        mark_high = torch.tensor(_safe_get('markHigh_est', float(market_data['High'])), dtype=torch.float32, device=self.device)
        return TradingEnvironment.StepState(
            timestamp=ts,
            high=high,
            low=low,
            close=close,
            funding_rate=funding_rate,
            mark_t=mark_t,
            mark_low=mark_low,
            mark_high=mark_high,
            exec_base_price=next_open_price,
        )

    def _refresh_dedup_exclude_buffers(self):
        if not self.enable_action_dedup:
            self._dedup_exclude_tensor_cuda = None
            return
        if not str(self.device).startswith('cuda'):
            raise RuntimeError("CUDA device required for action deduplication")
        values = sorted(int(v) for v in self.dedup_exclude_types)
        self._dedup_exclude_tensor_cuda = torch.tensor(
            values,
            dtype=torch.int32,
            device=self.device
        ) if len(values) > 0 else torch.empty(0, dtype=torch.int32, device=self.device)

    def _deduplicate_actions_cuda(
        self,
        actions: torch.Tensor,
        raw_actions: torch.Tensor,
        dedup_cols: int
    ) -> torch.Tensor:
        try:
            import gatree_cuda_compat as gatree_cuda
        except Exception:
            gatree_cuda = None

        if gatree_cuda is None:
            raise RuntimeError("gatree CUDA extension is required for action deduplication")

        kernel = getattr(gatree_cuda, 'prepare_and_deduplicate_actions', None)
        if kernel is None:
            raise RuntimeError("prepare_and_deduplicate_actions kernel not available in gatree CUDA extension")

        if (
            self._dedup_exclude_tensor_cuda is None
            or self._dedup_exclude_tensor_cuda.device != actions.device
        ):
            self._refresh_dedup_exclude_buffers()

        exclude_tensor = (
            self._dedup_exclude_tensor_cuda
            if self._dedup_exclude_tensor_cuda is not None
            else torch.empty(0, dtype=torch.int32, device=self.device)
        )

        prev_actions = self.prev_actions_raw
        prev_applied = self.prev_applied_mask
        last_node_idx = self._last_action_node_idx if self._last_action_node_idx is not None else None
        attempt = self.node_attempt_count if self.node_attempt_count is not None else None
        stall = getattr(self, 'node_stall_count', None)

        duplicate_mask = kernel(
            actions,
            raw_actions,
            prev_actions,
            prev_applied,
            exclude_tensor,
            int(dedup_cols),
            float(self.dedup_rtol),
            float(self.dedup_atol),
            bool(self.dedup_require_applied),
            int(ACTION_NOT_FOUND),
            last_node_idx,
            attempt,
            stall,
            int(self.max_nodes) if self.max_nodes else 0
        )

        if not isinstance(duplicate_mask, torch.Tensor):
            raise RuntimeError("CUDA deduplication kernel did not return a tensor")

        return duplicate_mask

    def _prepare_and_deduplicate_actions(self, predicted_actions) -> Tuple[torch.Tensor, torch.Tensor]:
        # Accept either tensor or (actions, node_idx)
        if isinstance(predicted_actions, tuple) or isinstance(predicted_actions, list):
            actions_tensor, node_idx = predicted_actions
            actions = actions_tensor.to(self.device)
            if self._last_action_node_idx is not None:
                # store raw predicted node indices for attribution
                self._last_action_node_idx = node_idx.to(device=self.device, dtype=torch.int32)
        else:
            actions = predicted_actions.to(self.device)

        if not str(self.device).startswith('cuda'):
            raise RuntimeError("TradingEnvironment requires CUDA device for action deduplication")

        actions = actions.contiguous()
        raw_actions = actions.clone()

        if self.enable_action_dedup:
            cols = max(1, min(self.dedup_compare_cols, actions.shape[1], self.prev_actions_raw.shape[1]))
            duplicate_mask = self._deduplicate_actions_cuda(actions, raw_actions, cols)
            if self._last_dedup_mask is None:
                self._last_dedup_mask = duplicate_mask.clone()
            elif self._last_dedup_mask.shape != duplicate_mask.shape:
                self._last_dedup_mask = duplicate_mask.clone()
            else:
                self._last_dedup_mask.copy_(duplicate_mask)
        else:
            duplicate_mask = torch.zeros(actions.shape[0], dtype=torch.bool, device=self.device)
            if self._last_dedup_mask is None:
                self._last_dedup_mask = duplicate_mask.clone()
            elif self._last_dedup_mask.shape != duplicate_mask.shape:
                self._last_dedup_mask = duplicate_mask.clone()
            else:
                self._last_dedup_mask.zero_()

        return actions, raw_actions

    def _snapshot_state_for_dedup(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pos_list.clone(), self.enter_ratio.clone()

    def _execute_actions_wrapper(self, actions: torch.Tensor, exec_base_price, market_data_row=None):
        try:
            import gatree_cuda_compat as gatree_cuda
        except Exception:
            gatree_cuda = None
        # 동적 슬리피지 계산 (market_data_row가 제공된 경우)
        slippage_rate = None
        if market_data_row is not None:
            slippage_rate = self._calculate_dynamic_slippage(market_data_row)
        
        if str(self.device).startswith('cuda'):
            # Prefer extended kernel when buffers are available
            if self.lot_open is not None and hasattr(gatree_cuda, 'execute_actions_with_attrib') and getattr(gatree_cuda, 'execute_actions_with_attrib') is not None and self._last_action_node_idx is not None:
                buy_exec = float(exec_base_price) * (1.0 + (slippage_rate if slippage_rate is not None else self.fixed_slippage_rate))
                sell_exec = float(exec_base_price) * (1.0 - (slippage_rate if slippage_rate is not None else self.fixed_slippage_rate))
                # Ensure contiguous views
                pos_list = self.pos_list.contiguous(); price_list = self.price_list.contiguous()
                leverage_ratio = self.leverage_ratio.contiguous(); enter_ratio = self.enter_ratio.contiguous()
                additional_count = self.additional_count.contiguous(); profit = self.profit.contiguous()
                trade_open = self.trade_open.contiguous(); trade_realized_pnl = self.trade_realized_pnl.contiguous()
                trade_total_count = self.trade_total_count.contiguous(); trade_win_count = self.trade_win_count.contiguous()
                trade_holding_sum = self.trade_holding_sum.contiguous(); trade_holding_avg = self.trade_holding_avg.contiguous()
                curr_win_streak = self.curr_win_streak.contiguous(); curr_loss_streak = self.curr_loss_streak.contiguous()
                max_win_streak = self.max_win_streak.contiguous(); max_loss_streak = self.max_loss_streak.contiguous()
                holding_period = self.holding_period.contiguous()

                gatree_cuda.execute_actions_with_attrib(
                    actions,
                    self._last_action_node_idx,
                    pos_list, price_list, leverage_ratio, enter_ratio, additional_count,
                    profit, trade_open, trade_realized_pnl, trade_total_count, trade_win_count,
                    trade_holding_sum, trade_holding_avg,
                    curr_win_streak, curr_loss_streak, max_win_streak, max_loss_streak,
                    holding_period,
                    self.lot_open, self.lot_node_idx, self.lot_side, self.lot_entry, self.lot_exposure, self.lot_open_step,
                    self.node_attempt_count if self.node_attempt_count is not None else torch.zeros(1, device=self.device, dtype=torch.int32),
                    self.node_apply_count, self.node_noop_count, self.node_win_count, self.node_realized_pnl_sum, self.node_exposure_sum,
                    int(self._sim_step),
                    float(buy_exec), float(sell_exec),
                    float(self.taker_fee_rate), float(self.min_entry_ratio), float(self.max_entry_ratio),
                    int(self.min_leverage), int(self.max_leverage),
                    float(self.min_additional_enter_ratio), int(self.max_additional_entries)
                )
            else:
                used_cuda = self._execute_actions(actions, exec_base_price, slippage_rate)
                if not used_cuda:
                    raise RuntimeError("CUDA 확장 모듈을 찾을 수 없습니다. Python 기반 _execute_actions는 제거되었습니다.")
        else:
            raise RuntimeError("device가 cuda가 아닙니다. CUDA만 지원되며 Python 경로는 제거되었습니다.")


    def _update_holding_periods(self):
        kernel = getattr(gatree_cuda, 'update_holding_periods', None)

        pos_list = self.pos_list.contiguous()
        holding_period = self.holding_period.contiguous()

        kernel(pos_list, holding_period)

        if holding_period.data_ptr() != self.holding_period.data_ptr():
            self.holding_period = holding_period

    def _accumulate_realized_pnl_and_stats(self):
        """Accumulate realized P&L and event-based stats using CUDA only."""
        if not str(self.device).startswith('cuda'):
            raise RuntimeError("device가 cuda가 아닙니다. CUDA만 지원되며 Python 경로는 제거되었습니다.")
        try:
            import gatree_cuda_compat as gatree_cuda
        except Exception:
            gatree_cuda = None
        if gatree_cuda is None or getattr(gatree_cuda, 'accumulate_realized_stats', None) is None:
            raise RuntimeError("CUDA 확장 모듈을 찾을 수 없습니다. Python 기반 집계 경로는 제거되었습니다.")

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

    def _compute_unrealized_pnl(self, mark_t: torch.Tensor) -> torch.Tensor:
        if not str(self.device).startswith('cuda'):
            raise RuntimeError("TradingEnvironment requires CUDA device for unrealized PnL computation")

        try:
            import gatree_cuda_compat as gatree_cuda
        except Exception:
            gatree_cuda = None

        kernel = getattr(gatree_cuda, 'compute_unrealized_pnl', None) if gatree_cuda is not None else None
        if kernel is None:
            raise RuntimeError("compute_unrealized_pnl CUDA kernel is unavailable")

        pos_list = self.pos_list.contiguous()
        price_list = self.price_list.contiguous()
        leverage_ratio = self.leverage_ratio.contiguous()
        enter_ratio = self.enter_ratio.contiguous()

        if isinstance(mark_t, torch.Tensor):
            mark_scalar = float(mark_t.item())
        else:
            mark_scalar = float(mark_t)

        unreal = kernel(
            pos_list,
            price_list,
            leverage_ratio,
            enter_ratio,
            float(mark_scalar)
        )

        if not isinstance(unreal, torch.Tensor):
            raise RuntimeError("compute_unrealized_pnl CUDA kernel did not return a tensor")

        expected_shape = self.sum_returns.shape
        if unreal.shape != expected_shape:
            raise RuntimeError(
                f"CUDA unrealized PnL kernel returned shape {unreal.shape}, expected {expected_shape}"
            )

        if unreal.dtype != self.sum_returns.dtype or unreal.device != self.sum_returns.device:
            unreal = unreal.to(device=self.sum_returns.device, dtype=self.sum_returns.dtype)

        return unreal

    def _update_equity_and_drawdown(self, total_unreal: torch.Tensor, timestamp):
        try:
            import gatree_cuda_compat as gatree_cuda_local
        except Exception:
            gatree_cuda_local = None

        kernel = getattr(gatree_cuda_local, 'update_equity_and_drawdown', None) if gatree_cuda_local is not None else None
        if kernel is None:
            raise RuntimeError("CUDA update_equity_and_drawdown kernel is not available")

        if not str(self.device).startswith('cuda'):
            raise RuntimeError("CUDA device가 아닙니다. update_equity_and_drawdown는 GPU 전용입니다.")

        realized_cum = self.realized_cum.contiguous()
        total_unreal = total_unreal.contiguous()
        cum_sum = self.cum_sum.contiguous()
        running_max = self.running_max.contiguous()
        max_drawdown = self.max_drawdown.contiguous()

        kernel(
            realized_cum,
            total_unreal,
            cum_sum,
            running_max,
            max_drawdown,
            float(self.equity_base)
        )

        if realized_cum.data_ptr() != self.realized_cum.data_ptr():
            self.realized_cum = realized_cum
        if cum_sum.data_ptr() != self.cum_sum.data_ptr():
            self.cum_sum = cum_sum
        if running_max.data_ptr() != self.running_max.data_ptr():
            self.running_max = running_max
        if max_drawdown.data_ptr() != self.max_drawdown.data_ptr():
            self.max_drawdown = max_drawdown

        self.profit.zero_()
        self.last_timestamp = timestamp

    def _detect_atr_columns(self, market_data_row) -> Tuple[str, str]:
        """
        시장 데이터에서 사용 가능한 ATR 및 ATR-MA 컬럼을 자동 감지합니다.
        
        Args:
            market_data_row: 시장 데이터 행 (pandas Series)
            
        Returns:
            Tuple[str, str]: (atr_column_name, atr_ma_column_name)
            
        Raises:
            ValueError: 적절한 ATR 컬럼을 찾지 못한 경우
        """
        available_columns = list(market_data_row.index)
        
        # 1. 사용자가 명시적으로 지정한 컬럼 사용
        if self.atr_column_config and self.atr_ma_column_config:
            if (self.atr_column_config in available_columns and 
                self.atr_ma_column_config in available_columns):
                print(f"Using user-specified ATR columns: {self.atr_column_config}, {self.atr_ma_column_config}")
                return self.atr_column_config, self.atr_ma_column_config
            else:
                print(f"Warning: User-specified columns not found. Falling back to auto-detection.")
        
        # 2. 패턴 매칭을 통한 자동 감지
        atr_candidates = [col for col in available_columns if re.match(self.atr_pattern, col)]
        atr_ma_candidates = [col for col in available_columns if re.match(self.atr_ma_pattern, col)]
        
        if not atr_candidates:
            raise ValueError(
                f"No ATR columns found matching pattern '{self.atr_pattern}'. "
                f"Available columns: {[col for col in available_columns if 'ATR' in col]}"
            )
        
        if not atr_ma_candidates:
            # No ATR-MA columns present; fall back to runtime rolling mean
            best_atr = None
            preferred_windows = [14, 7, 21, 28, 20, 10]
            for window in preferred_windows:
                for atr_col in atr_candidates:
                    if f'_{window}' in atr_col or atr_col.endswith(str(window)):
                        best_atr = atr_col
                        break
                if best_atr:
                    break
            if not best_atr:
                best_atr = sorted(atr_candidates)[0]
            print(
                f"Auto-detected ATR column: {best_atr}; ATR-MA column not found. "
                f"Using runtime rolling mean (window={self.atr_ma_window})."
            )
            self._use_runtime_atr_ma = True
            return best_atr, None
        
        # 3. 최적의 ATR/ATR-MA 쌍 찾기
        best_atr = None
        best_atr_ma = None
        
        # ATR 컬럼 우선순위 (일반적인 윈도우 크기 순서)
        preferred_windows = [14, 7, 21, 28, 20, 10]
        
        for window in preferred_windows:
            for atr_col in atr_candidates:
                if f'_{window}' in atr_col or atr_col.endswith(str(window)):
                    # ATR 컬럼과 동일 타임프레임 접미사를 고려한 MA 컬럼 매칭 지원
                    # 예: ATR_14_5m <-> ATR_14_MA_50_5m
                    m = re.match(r'^(.*)_(\d+[mhd])$', atr_col)
                    if m:
                        atr_base, tf_suffix = m.group(1), m.group(2)
                    else:
                        atr_base, tf_suffix = atr_col, None
                    # 기본 규칙과 접미사 규칙을 모두 허용
                    matching_ma = [ma_col for ma_col in atr_ma_candidates if (
                        ma_col.startswith(atr_col + '_MA_') or
                        (ma_col.startswith(atr_base + '_MA_') and (tf_suffix is None or ma_col.endswith('_' + tf_suffix)))
                    )]
                    
                    # 우선적으로 설정된 MA 윈도우와 일치하는 컬럼 찾기
                    preferred_ma = [ma_col for ma_col in matching_ma 
                                    if f'_MA_{self.atr_ma_window}' in ma_col]
                    
                    if preferred_ma:
                        best_atr = atr_col
                        best_atr_ma = preferred_ma[0]
                        break
                    elif matching_ma:
                        # 선호하는 MA 윈도우가 없으면 첫 번째 매칭 사용
                        best_atr = atr_col
                        best_atr_ma = matching_ma[0]
                        break
            
            if best_atr and best_atr_ma:
                break
        
        # 4. 기본 매칭 시도 (우선순위 매칭이 실패한 경우)
        if not best_atr or not best_atr_ma:
            # 첫 번째 ATR 컬럼과 매칭되는 MA 컬럼 찾기 (타임프레임 접미사 고려)
            for atr_col in sorted(atr_candidates):
                m = re.match(r'^(.*)_(\d+[mhd])$', atr_col)
                if m:
                    atr_base, tf_suffix = m.group(1), m.group(2)
                else:
                    atr_base, tf_suffix = atr_col, None
                matching_ma = [ma_col for ma_col in atr_ma_candidates if (
                    ma_col.startswith(atr_col + '_MA_') or
                    (ma_col.startswith(atr_base + '_MA_') and (tf_suffix is None or ma_col.endswith('_' + tf_suffix)))
                )]
                if matching_ma:
                    best_atr = atr_col
                    best_atr_ma = matching_ma[0]
                    break
        
        if not best_atr or not best_atr_ma:
            raise ValueError(
                f"Could not find matching ATR/ATR-MA column pair. "
                f"ATR candidates: {atr_candidates}, ATR-MA candidates: {atr_ma_candidates}"
            )
        
        print(f"Auto-detected ATR columns: {best_atr}, {best_atr_ma}")
        return best_atr, best_atr_ma

    def _calculate_dynamic_slippage(self, market_data_row) -> float:
        """
        상대 변동성 기반 동적 슬리피지를 계산합니다.
        
        Args:
            market_data_row: 현재 시점의 시장 데이터 행 (pandas Series)
        
        Returns:
            float: 계산된 슬리피지 비율
        """
        if self.slippage_mode != 'relative_volatility':
            return self.fixed_slippage_rate
        
        # 첫 번째 호출 시 컬럼 감지
        if not self._columns_detected:
            try:
                self.atr_column, self.atr_ma_column = self._detect_atr_columns(market_data_row)
                self._columns_detected = True
            except ValueError as e:
                print(f"ATR column detection failed: {e}")
                print("Falling back to fixed slippage mode.")
                return self.fixed_slippage_rate
        
        try:
            # ATR과 ATR 이동평균 값 추출
            current_atr = market_data_row.get(self.atr_column)
            if not self._use_runtime_atr_ma:
                atr_ma = market_data_row.get(self.atr_ma_column)
            else:
                # Compute ATR moving average on the fly using a rolling buffer
                if current_atr is None or pd.isna(current_atr):
                    return self.base_slippage_rate
                if self._atr_history is None:
                    self._atr_history = []
                self._atr_history.append(float(current_atr))
                # Keep last 'atr_ma_window' values
                maxlen = int(self.atr_ma_window) if self.atr_ma_window else 50
                if len(self._atr_history) > maxlen:
                    self._atr_history = self._atr_history[-maxlen:]
                atr_ma = float(np.mean(self._atr_history)) if len(self._atr_history) > 0 else None
            
            # 데이터 유효성 검사
            if current_atr is None or atr_ma is None:
                print(f"Warning: Missing ATR data columns ({self.atr_column}, {self.atr_ma_column}). Using base slippage rate.")
                return self.base_slippage_rate
            
            if pd.isna(current_atr) or pd.isna(atr_ma) or atr_ma <= 0:
                return self.base_slippage_rate
            
            # 변동성 비율 계산: 현재 ATR / ATR 이동평균
            volatility_ratio = float(current_atr) / float(atr_ma)
            
            # 동적 슬리피지 계산: 기본 비율 * 변동성 비율
            dynamic_slippage = self.base_slippage_rate * volatility_ratio
            
            # 최소/최대 한계 적용
            final_slippage = max(self.min_slippage, min(self.max_slippage, dynamic_slippage))
            
            return final_slippage
            
        except Exception as e:
            print(f"Error in dynamic slippage calculation: {e}. Using base slippage rate.")
            return self.base_slippage_rate

    def _update_dedup_buffers_after_exec(
        self,
        actions: torch.Tensor,
        raw_actions: torch.Tensor,
        pos_before: torch.Tensor,
        er_before: torch.Tensor,
    ):
        if not self.enable_action_dedup:
            return

        if not str(self.device).startswith('cuda'):
            raise RuntimeError("Action deduplication update requires CUDA device")

        try:
            import gatree_cuda_compat as gatree_cuda
        except Exception:
            gatree_cuda = None

        kernel = None
        if gatree_cuda is not None:
            kernel = getattr(gatree_cuda, 'update_dedup_buffers_after_exec', None)

        if kernel is None:
            raise RuntimeError("update_dedup_buffers_after_exec kernel not available")

        prev_actions = self.prev_actions_raw
        if not prev_actions.is_contiguous():
            prev_actions = prev_actions.contiguous()
            self.prev_actions_raw = prev_actions

        prev_applied = self.prev_applied_mask
        if not prev_applied.is_contiguous():
            prev_applied = prev_applied.contiguous()
            self.prev_applied_mask = prev_applied

        cols_copy = min(self.prev_actions_raw.shape[1], raw_actions.shape[1])

        kernel(
            actions.contiguous(),
            raw_actions.contiguous(),
            pos_before.contiguous(),
            er_before.contiguous(),
            self.pos_list.contiguous(),
            self.enter_ratio.contiguous(),
            self.prev_actions_raw,
            self.prev_applied_mask,
            int(cols_copy),
            1e-8,
            ACTION_NEW_LONG,
            ACTION_NEW_SHORT,
            ACTION_CLOSE_ALL,
            ACTION_CLOSE_PARTIAL,
            ACTION_ADD_POSITION,
            ACTION_FLIP_POSITION,
        )

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
        """CUDA 커널로 청산 체크를 수행. Lot/노드 어트리뷰션 버퍼가 있으면 노드 단위로도 집계."""
        if not str(self.device).startswith('cuda'):
            raise RuntimeError("device가 cuda가 아닙니다. CUDA만 지원되며 Python 경로는 제거되었습니다.")
        try:
            import gatree_cuda_compat as gatree_cuda
        except Exception:
            gatree_cuda = None
        if gatree_cuda is None or getattr(gatree_cuda, 'check_liquidation_batch', None) is None:
            raise RuntimeError("CUDA 확장 모듈을 찾을 수 없습니다. Python 기반 _check_liquidation는 제거되었습니다.")

        # Ensure CUDA and contiguous tensors for in-place kernel updates
        pos_list = self.pos_list.contiguous()
        price_list = self.price_list.contiguous()
        leverage_ratio = self.leverage_ratio.contiguous()
        enter_ratio = self.enter_ratio.contiguous()
        additional_count = self.additional_count.contiguous()
        holding_period = self.holding_period.contiguous()
        profit = self.profit.contiguous()
        trade_open = self.trade_open.contiguous()
        trade_realized_pnl = self.trade_realized_pnl.contiguous()
        trade_total_count = self.trade_total_count.contiguous()
        trade_win_count = self.trade_win_count.contiguous()
        trade_holding_sum = self.trade_holding_sum.contiguous()
        trade_holding_avg = self.trade_holding_avg.contiguous()
        curr_win_streak = self.curr_win_streak.contiguous()
        curr_loss_streak = self.curr_loss_streak.contiguous()
        max_win_streak = self.max_win_streak.contiguous()
        max_loss_streak = self.max_loss_streak.contiguous()

        mark_low_f = float(mark_low.item() if isinstance(mark_low, torch.Tensor) else mark_low)
        mark_high_f = float(mark_high.item() if isinstance(mark_high, torch.Tensor) else mark_high)

        used_attrib = False
        if (
            self.lot_open is not None and
            hasattr(gatree_cuda, 'check_liquidation_with_attrib') and
            getattr(gatree_cuda, 'check_liquidation_with_attrib') is not None and
            self.node_realized_pnl_sum is not None and self.node_exposure_sum is not None
        ):
            gatree_cuda.check_liquidation_with_attrib(
                pos_list,
                price_list,
                leverage_ratio,
                enter_ratio,
                additional_count,
                holding_period,
                profit,
                trade_open,
                trade_realized_pnl,
                trade_total_count,
                trade_win_count,
                trade_holding_sum,
                trade_holding_avg,
                curr_win_streak,
                curr_loss_streak,
                max_win_streak,
                max_loss_streak,
                self.lot_open, self.lot_node_idx, self.lot_side, self.lot_entry, self.lot_exposure, self.lot_open_step,
                self.node_win_count, self.node_realized_pnl_sum, self.node_exposure_sum,
                float(mark_low_f),
                float(mark_high_f),
                float(self.maintenance_margin_rate),
                float(self.liq_clearance_fee_rate),
            )
            used_attrib = True
        else:
            gatree_cuda.check_liquidation_batch(
                pos_list,
                price_list,
                leverage_ratio,
                enter_ratio,
                additional_count,
                holding_period,
                profit,
                trade_open,
                trade_realized_pnl,
                trade_total_count,
                trade_win_count,
                trade_holding_sum,
                trade_holding_avg,
                curr_win_streak,
                curr_loss_streak,
                max_win_streak,
                max_loss_streak,
                float(mark_low_f),
                float(mark_high_f),
                float(self.maintenance_margin_rate),
                float(self.liq_clearance_fee_rate),
            )

        # If any tensor got new storage via contiguous(), write back views
        if pos_list.data_ptr() != self.pos_list.data_ptr():
            self.pos_list = pos_list
        if price_list.data_ptr() != self.price_list.data_ptr():
            self.price_list = price_list
        if leverage_ratio.data_ptr() != self.leverage_ratio.data_ptr():
            self.leverage_ratio = leverage_ratio
        if enter_ratio.data_ptr() != self.enter_ratio.data_ptr():
            self.enter_ratio = enter_ratio
        if additional_count.data_ptr() != self.additional_count.data_ptr():
            self.additional_count = additional_count
        if holding_period.data_ptr() != self.holding_period.data_ptr():
            self.holding_period = holding_period
        if profit.data_ptr() != self.profit.data_ptr():
            self.profit = profit
        if trade_open.data_ptr() != self.trade_open.data_ptr():
            self.trade_open = trade_open
        if trade_realized_pnl.data_ptr() != self.trade_realized_pnl.data_ptr():
            self.trade_realized_pnl = trade_realized_pnl
        if trade_total_count.data_ptr() != self.trade_total_count.data_ptr():
            self.trade_total_count = trade_total_count
        if trade_win_count.data_ptr() != self.trade_win_count.data_ptr():
            self.trade_win_count = trade_win_count
        if trade_holding_sum.data_ptr() != self.trade_holding_sum.data_ptr():
            self.trade_holding_sum = trade_holding_sum
        if trade_holding_avg.data_ptr() != self.trade_holding_avg.data_ptr():
            self.trade_holding_avg = trade_holding_avg
        if curr_win_streak.data_ptr() != self.curr_win_streak.data_ptr():
            self.curr_win_streak = curr_win_streak
        if curr_loss_streak.data_ptr() != self.curr_loss_streak.data_ptr():
            self.curr_loss_streak = curr_loss_streak
        if max_win_streak.data_ptr() != self.max_win_streak.data_ptr():
            self.max_win_streak = max_win_streak
        if max_loss_streak.data_ptr() != self.max_loss_streak.data_ptr():
            self.max_loss_streak = max_loss_streak

        # no wrapper/fallback. CUDA only.
    
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
            enter_ratio = torch.clamp(enter_ratio, min=self.min_entry_ratio, max=self.max_entry_ratio)
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
            remaining_capital = torch.clamp(self.max_entry_ratio - current_enter_ratio, min=0.0)
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
            ratio = torch.clamp(ratio, min=self.min_entry_ratio, max=self.max_entry_ratio)
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
            ratio = torch.clamp(ratio, min=self.min_entry_ratio, max=self.max_entry_ratio)
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


    def _execute_actions(self, actions_tensor, exec_base_price, slippage_rate=None):
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

        # Use provided slippage rate or fall back to fixed rate
        if slippage_rate is None:
            slippage_rate = self.fixed_slippage_rate

        # Ensure tensors are on CUDA and contiguous
        actions = actions_tensor.to(self.device).contiguous()
        buy_exec_price = float((exec_base_price * (1 + slippage_rate)).item() if isinstance(exec_base_price, torch.Tensor) else exec_base_price * (1 + slippage_rate))
        sell_exec_price = float((exec_base_price * (1 - slippage_rate)).item() if isinstance(exec_base_price, torch.Tensor) else exec_base_price * (1 - slippage_rate))

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
            float(self.taker_fee_rate), float(self.min_entry_ratio), float(self.max_entry_ratio),
            int(self.min_leverage), int(self.max_leverage),
            float(self.min_additional_enter_ratio), int(self.max_additional_entries)
        )

        # no-op

        return True

    def _calculate_current_equity(self, market_data) -> torch.Tensor:
        """현재 에쿼티 계산 (미실현 손익 포함) - CUDA 전용 경로"""
        if not str(self.device).startswith('cuda'):
            raise RuntimeError("device가 cuda가 아닙니다. CUDA만 지원되며 Python 경로는 제거되었습니다.")

        try:
            import gatree_cuda_compat as gatree_cuda
        except Exception:
            gatree_cuda = None
        if gatree_cuda is None or getattr(gatree_cuda, 'compute_equity_batch', None) is None:
            raise RuntimeError("CUDA 확장 모듈을 찾을 수 없습니다. Python 기반 _calculate_current_equity는 제거되었습니다.")

        # 스칼라 typical price 계산 (파이썬 스칼라만 사용)
        h = float(market_data['High'])
        l = float(market_data['Low'])
        c = float(market_data['Close'])
        mark_t_scalar = (h + l + c) / 3.0

        # Ensure CUDA and contiguous tensors
        pos_list = self.pos_list.contiguous()
        price_list = self.price_list.contiguous()
        leverage_ratio = self.leverage_ratio.contiguous()
        enter_ratio = self.enter_ratio.contiguous()
        realized_cum = self.realized_cum.contiguous()

        equity = gatree_cuda.compute_equity_batch(
            pos_list, price_list, leverage_ratio, enter_ratio, realized_cum,
            float(mark_t_scalar), float(self.equity_base)
        )
        return equity

    def step(self, market_data, next_open_price, predicted_actions):
        """
        원본 step 함수 완전 보존 + 다중 목표용 히스토리 추가 추적
        """
        # Increment simulation step for attribution holding periods
        self._sim_step += 1
        pre_equity = self._calculate_current_equity(market_data)
        state = self._extract_step_state(market_data, next_open_price)
        self._apply_funding_fees(state.timestamp, state.funding_rate)
        # Use mark-based range for liquidation checks when available
        self._check_liquidation(state.mark_low, state.mark_high)
        actions, raw_actions = self._prepare_and_deduplicate_actions(predicted_actions)
        pos_before, er_before = self._snapshot_state_for_dedup()
        self._execute_actions_wrapper(actions, state.exec_base_price, market_data)
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
        self.max_drawdown[invalid_mask] = 1.0
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
            # Advanced statistical metrics
            # PSR benchmark SR* can be tuned via objectives_config (default 0.0)
            psr_sr_benchmark = float(self.objectives_config.get('psr_sr_benchmark', 0.0))
            psr_vals = self._calculate_psr(self.returns_history, sr_benchmark=psr_sr_benchmark)
            t_stats = self._calculate_t_stat(self.returns_history)
        else:
            # Fallback for insufficient data
            sharpe_ratios = torch.zeros_like(mean_returns)
            volatilities = torch.zeros_like(mean_returns)
            calmar_ratios = torch.zeros_like(mean_returns)
            sortino_ratios = torch.zeros_like(mean_returns)
            omega_ratios = torch.zeros_like(mean_returns)
            information_ratios = torch.zeros_like(mean_returns)
            psr_vals = torch.zeros_like(mean_returns)
            t_stats = torch.zeros_like(mean_returns)
        
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
            'information_ratio': information_ratios,
            # Added advanced statistical objectives
            'psr': psr_vals,           # Probabilistic Sharpe Ratio (0..1)
            't_stat': t_stats          # t-statistic of per-step returns
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
