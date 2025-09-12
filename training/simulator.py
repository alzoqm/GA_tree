# fitness/simulator.py

import numpy as np
import torch
import pandas as pd
from typing import List, Optional
from tqdm import tqdm

from training.trading_env import TradingEnvironment
from analysis.event_logger import EventLogger, EventRecord, WindowRecord
from training.predictor import build_adjacency_list_cuda, predict_population_cuda
from models.model import GATreePop
from models.constants import (
    COL_NODE_TYPE, COL_DEPTH, NODE_TYPE_UNUSED,
)
from models.constants import ACTION_FLIP_POSITION

torch.set_grad_enabled(False)


class FitnessSimulator:
    """Run trading simulation and return multi-objective metrics."""

    def __init__(self):
        pass

    def _step_and_log_events(
        self,
        *,
        environment: TradingEnvironment,
        market_data_row: pd.Series,
        next_open_price: torch.Tensor,
        predicted_actions: torch.Tensor,
        B: int,
        device: str,
        generation_id: int,
        entry_index: int,
        evt_logger: EventLogger,
        trade_id: np.ndarray,
        trade_seq: np.ndarray,
        win_start_step: np.ndarray,
        win_start_time: np.ndarray,
    ) -> None:
        """Apply one environment step and log event/window records.

        Mutates `trade_id`, `trade_seq`, `win_start_step`, and `win_start_time` in place.
        """
        # Snapshot pre-state for logging
        pos_before = environment.pos_list.clone()
        lev_before = environment.leverage_ratio.clone()
        er_before = environment.enter_ratio.clone()
        add_before = environment.additional_count.clone()

        environment.step(market_data_row, next_open_price, predicted_actions)

        # Post-state and applied actions
        pos_after = environment.pos_list.clone()
        applied_mask = environment.get_last_applied_mask()
        last_actions, last_raw = environment.get_last_actions()
        # Ensure tensors on CPU for logging
        if applied_mask is None:
            applied_mask = torch.zeros((B,), dtype=torch.bool, device=device)
        applied_mask_cpu = applied_mask.detach().to('cpu').numpy()
        if last_actions is None:
            last_actions = torch.zeros((B, predicted_actions.size(1)), dtype=torch.float32, device=device)
        last_actions_cpu = last_actions.detach().to('cpu').numpy()
        pos_before_cpu = pos_before.detach().to('cpu').numpy()
        pos_after_cpu = pos_after.detach().to('cpu').numpy()
        er_before_cpu = er_before.detach().to('cpu').numpy()
        lev_before_cpu = lev_before.detach().to('cpu').numpy()
        add_before_cpu = add_before.detach().to('cpu').numpy()

        ts_str = str(market_data_row.name)

        # Vectorized window close/open determination
        atype_vec = (
            last_actions_cpu[:, 0].astype(np.int64)
            if last_actions_cpu.shape[1] >= 1
            else np.zeros((B,), dtype=np.int64)
        )
        flip_mask = applied_mask_cpu & (atype_vec == ACTION_FLIP_POSITION) & (pos_before_cpu != 0)
        open_mask = ((pos_before_cpu == 0) & (pos_after_cpu != 0)) | flip_mask
        close_mask = ((pos_before_cpu != 0) & (pos_after_cpu == 0)) | flip_mask

        valid_close = close_mask & (trade_id > 0) & (win_start_step >= 0)
        if np.any(valid_close):
            idx = np.where(valid_close)[0]
            windows_df = pd.DataFrame(
                {
                    'generation': generation_id,
                    'chromosome': idx.astype(np.int64),
                    'trade_id': trade_id[idx].astype(np.int64),
                    'start_step': win_start_step[idx].astype(np.int64),
                    'start_time': win_start_time[idx].astype(str),
                    'end_step': np.full(idx.shape[0], entry_index, dtype=np.int64),
                    'end_time': np.full(idx.shape[0], ts_str, dtype=object),
                }
            )
            evt_logger.log_windows_df(windows_df)
            # reset indicators (next open increments id as needed)
            win_start_step[idx] = -1
            win_start_time[idx] = None

        if np.any(open_mask):
            idxo = np.where(open_mask)[0]
            trade_id[idxo] = trade_id[idxo] + 1
            trade_seq[idxo] = 0
            win_start_step[idxo] = entry_index
            # store original timestamp object for start_time; keep as string for consistency
            win_start_time[idxo] = ts_str

        # Vectorized applied event logging
        if np.any(applied_mask_cpu):
            idxe = np.where(applied_mask_cpu)[0]
            acts = last_actions_cpu[idxe, :]
            # action parameters (default to 0 if absent)
            atype_e = (
                acts[:, 0].astype(np.int64)
                if acts.shape[1] >= 1
                else np.zeros((idxe.shape[0],), dtype=np.int64)
            )
            p2_e = (
                acts[:, 1].astype(np.float32)
                if acts.shape[1] >= 2
                else np.zeros((idxe.shape[0],), dtype=np.float32)
            )
            p3_e = (
                acts[:, 2].astype(np.float32)
                if acts.shape[1] >= 3
                else np.zeros((idxe.shape[0],), dtype=np.float32)
            )
            p4_e = (
                acts[:, 3].astype(np.float32)
                if acts.shape[1] >= 4
                else np.zeros((idxe.shape[0],), dtype=np.float32)
            )
            node_e = (
                acts[:, 4].astype(np.int64)
                if acts.shape[1] >= 5
                else np.full((idxe.shape[0],), -1, dtype=np.int64)
            )
            # trade_seq increments for these indices
            trade_seq[idxe] = trade_seq[idxe] + 1
            events_df = pd.DataFrame(
                {
                    'generation': generation_id,
                    'chromosome': idxe.astype(np.int64),
                    'step_index': np.full(idxe.shape[0], entry_index, dtype=np.int64),
                    'timestamp': np.full(idxe.shape[0], ts_str, dtype=object),
                    'action_node_id': node_e,
                    'action_type': atype_e,
                    'param2': p2_e,
                    'param3': p3_e,
                    'param4': p4_e,
                    'applied': np.full(idxe.shape[0], True, dtype=bool),
                    'pos_before': pos_before_cpu[idxe].astype(np.int64),
                    'pos_after': pos_after_cpu[idxe].astype(np.int64),
                    'lev_before': lev_before_cpu[idxe].astype(np.int64),
                    'er_before': er_before_cpu[idxe].astype(np.float32),
                    'add_count_before': add_before_cpu[idxe].astype(np.int64),
                    'trade_id': trade_id[idxe].astype(np.int64),
                    'trade_seq': trade_seq[idxe].astype(np.int64),
                }
            )
            evt_logger.log_events_df(events_df)

    def simulate(
        self,
        population: GATreePop,
        data: pd.DataFrame,
        all_feature_names: List[str],
        start_data_cnt: int,
        stop_data_cnt: int,
        device: str,
        evaluation_config: dict,
        return_returns: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        env_config = evaluation_config['simulation_env']

        # Pull the requested objective set (could be IS/OOS config); may include
        # simulator-only complexity metrics. We'll filter for the environment.
        requested_cfg = evaluation_config.get(
            'multi_objectives',
            evaluation_config.get(
                'objectives',
                {
                    'objectives': ['mean_return', 'max_drawdown', 'psr', 'win_rate'],
                    'minimize': [False, True, False, False],
                    'risk_free_rate': 0.0,
                    'volatility_lookback': 30,
                },
            ),
        )

        # Environment-supported objective names
        SUPPORTED_ENV_OBJECTIVES = {
            'mean_return', 'max_drawdown', 'sharpe_ratio', 'psr', 't_stat', 'win_rate',
            'profit_factor', 'compound_value', 'trade_win_rate', 'volatility',
            'calmar_ratio', 'sortino_ratio', 'omega_ratio', 'information_ratio'
        }
        # Simulator-side complexity metrics
        SUPPORTED_COMPLEXITY_OBJECTIVES = {'comp_size_norm', 'comp_depth_norm', 'complexity'}

        req_names = list(requested_cfg.get('objectives', []))
        req_minimize = list(requested_cfg.get('minimize', []))
        if req_names and (len(req_minimize) != len(req_names)):
            if len(req_minimize) < len(req_names):
                req_minimize = req_minimize + [True] * (len(req_names) - len(req_minimize))

        env_names = [n for n in req_names if n in SUPPORTED_ENV_OBJECTIVES]
        env_minimize = [req_minimize[req_names.index(n)] for n in env_names]
        extra_names = [n for n in req_names if n not in SUPPORTED_ENV_OBJECTIVES]
        unknown = [n for n in extra_names if n not in SUPPORTED_COMPLEXITY_OBJECTIVES]
        if unknown:
            raise ValueError(f"Unknown objective(s) not supported: {unknown}")

        # Filtered config for the environment (no simulator-only objectives)
        objectives_config = dict(requested_cfg)
        objectives_config['objectives'] = env_names
        objectives_config['minimize'] = env_minimize

        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Build adjacency list for CUDA operations (once per simulate)
        adj_offsets, adj_indices = build_adjacency_list_cuda(population, strict_overflow=False)
        if adj_offsets is None:
            raise RuntimeError("Failed to build adjacency list on GPU.")

        # Create environment
        environment = TradingEnvironment(
            chromosomes_size=population.pop_size,
            env_config=env_config,
            objectives_config=objectives_config,
            device=device,
        )

        # Pre-extract features to numpy for fast access
        feature_data = data[all_feature_names]
        feature_np = feature_data.to_numpy(dtype=np.float32, copy=False)

        B = population.pop_size
        N = population.max_nodes
        F = feature_np.shape[1]

        # Try to keep full features on GPU to avoid H2D per step
        features_all_gpu = None
        if str(device).startswith('cuda'):
            try:
                features_all_gpu = torch.as_tensor(feature_np, dtype=torch.float32, device=device)
            except RuntimeError:
                features_all_gpu = None

        # Preallocate reusable GPU buffers
        # Allocate 5 columns: [:,4] will carry action_node_id when available
        results = torch.empty((B, 5), dtype=torch.float32, device=device)
        bfs = torch.empty((B, N), dtype=torch.int32, device=device)
        visited = torch.empty((B, N), dtype=torch.int32, device=device)
        positions = torch.empty((B,), dtype=torch.int32, device=device)
        features = torch.empty((F,), dtype=torch.float32, device=device)

        # LUT: env.pos_list (0: HOLD, 1: SHORT, 2: LONG) -> ROOT_BRANCH_*
        from models.constants import ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT, ROOT_BRANCH_LONG
        lut = torch.tensor([ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT, ROOT_BRANCH_LONG], dtype=torch.int32, device=device)

        # Monotonic step id for kernel visited generation marking
        step_id = 1

        # --- Event logging setup ---
        evt_logger = EventLogger(output_dir="results")
        try:
            generation_id = int(evaluation_config.get('generation', 0))
        except Exception:
            generation_id = 0
        # Trade tracking per chromosome (CPU-side, vectorized)
        trade_id = np.zeros((B,), dtype=np.int64)
        trade_seq = np.zeros((B,), dtype=np.int64)
        win_start_step = np.full((B,), -1, dtype=np.int64)
        win_start_time = np.empty((B,), dtype=object)
        win_start_time[:] = None

        pbar = tqdm(range(start_data_cnt, stop_data_cnt - 1), desc="Sim", leave=False)
        for entry_index in pbar:
            market_data_row = data.iloc[entry_index]
            next_open_price = torch.as_tensor(
                data['Open'].iloc[entry_index + 1], dtype=torch.float32, device=device
            )

            # Update features and positions into preallocated buffers
            if features_all_gpu is not None:
                features.copy_(features_all_gpu[entry_index])
            else:
                features.copy_(torch.from_numpy(feature_np[entry_index]).to(device))

            positions.copy_(lut[environment.pos_list])

            # Increment step id each iteration to avoid clearing visited
            step_id += 1
            predicted_actions = predict_population_cuda(
                population=population,
                feature_values=None,
                current_positions=positions,
                adj_offsets=adj_offsets,
                adj_indices=adj_indices,
                device=device,
                results=results,
                bfs=bfs,
                visited=visited,
                positions_tensor=positions,
                features_tensor=features,
                step_id=step_id,
                validate=False,
                sync=False,
            )
            if predicted_actions is None:
                raise RuntimeError(
                    "CUDA prediction failed. Check if gatree_cuda module is compiled and available."
                )

            self._step_and_log_events(
                environment=environment,
                market_data_row=market_data_row,
                next_open_price=next_open_price,
                predicted_actions=predicted_actions,
                B=B,
                device=device,
                generation_id=generation_id,
                entry_index=entry_index,
                evt_logger=evt_logger,
                trade_id=trade_id,
                trade_seq=trade_seq,
                win_start_step=win_start_step,
                win_start_time=win_start_time,
            )
        pbar.close()

        minimum_trades = evaluation_config.get('fitness_function', {}).get('minimum_trades', 40)
        wfo_min_trades = evaluation_config.get('min_trades_per_window', None)
        if wfo_min_trades is not None:
            minimum_trades = wfo_min_trades

        env_objectives = environment.get_multi_objective_metrics(minimum_date=minimum_trades)
        # Flush event/window logs to disk
        try:
            evt_logger.flush()
        except Exception:
            pass
        try:
            verbose = bool(evaluation_config.get('verbose', False)) if isinstance(evaluation_config, dict) else False
        except Exception:
            verbose = False
        if verbose:
            print('completed multi_objective_metrics')

        # Assemble final objective matrix in the exact requested order,
        # appending simulator-side complexity metrics when asked.
        B = population.pop_size
        if env_objectives is None:
            env_objectives = np.zeros((B, 0), dtype=np.float32)
        # Derive invalid mask from env outputs
        if env_objectives.size > 0:
            invalid_mask = np.any(
                np.isin(env_objectives, [-1e9, 1e9]) | ~np.isfinite(env_objectives), axis=1
            )
        else:
            invalid_mask = np.zeros((B,), dtype=bool)

        # Complexity components computation as needed
        need_size = 'comp_size_norm' in req_names or 'complexity' in req_names
        need_depth = 'comp_depth_norm' in req_names or 'complexity' in req_names
        comp_cols: dict[str, np.ndarray] = {}
        comp_size_np = None
        comp_depth_np = None

        if need_size or need_depth:
            pt = population.population_tensor
            used = (pt[:, :, COL_NODE_TYPE] != float(NODE_TYPE_UNUSED))
            used_f = used.to(dtype=torch.float32)

            if need_size:
                denom = max(1, population.max_nodes - 3)
                size_used = used.sum(dim=1).to(dtype=torch.float32)
                comp_size = torch.clamp((size_used - 3.0) / float(denom), min=0.0, max=1.0)
                comp_size_np = comp_size.detach().to('cpu').numpy().astype(np.float32)
                comp_size_np[invalid_mask] = 1e9
                if 'comp_size_norm' in req_names:
                    comp_cols['comp_size_norm'] = comp_size_np

            if need_depth:
                denom_d = max(1, population.max_depth)
                depth_vals = pt[:, :, COL_DEPTH]
                depth_masked = depth_vals * used_f
                max_depth_vals = torch.amax(depth_masked, dim=1)
                comp_depth = torch.clamp(max_depth_vals / float(denom_d), min=0.0, max=1.0)
                comp_depth_np = comp_depth.detach().to('cpu').numpy().astype(np.float32)
                comp_depth_np[invalid_mask] = 1e9
                if 'comp_depth_norm' in req_names:
                    comp_cols['comp_depth_norm'] = comp_depth_np

        # Composite single complexity metric
        if 'complexity' in req_names:
            cx_cfg = evaluation_config.get('complexity', {}) if isinstance(evaluation_config, dict) else {}
            mode = str(cx_cfg.get('mode', 'weighted')).lower()
            w_size = float(cx_cfg.get('size_weight', 0.5))
            w_depth = float(cx_cfg.get('depth_weight', 0.5))

            if comp_size_np is None:
                comp_size_np = np.zeros((B,), dtype=np.float32)
            if comp_depth_np is None:
                comp_depth_np = np.zeros((B,), dtype=np.float32)

            if mode == 'max':
                comp = np.maximum(comp_size_np, comp_depth_np).astype(np.float32)
            elif mode in ('geomean', 'geo', 'gmean'):
                comp = np.sqrt(np.maximum(comp_size_np, 0.0) * np.maximum(comp_depth_np, 0.0)).astype(np.float32)
            else:  # weighted average (default)
                denom_w = (w_size + w_depth) if (w_size + w_depth) != 0 else 1.0
                comp = ((w_size * comp_size_np + w_depth * comp_depth_np) / denom_w).astype(np.float32)
            comp[invalid_mask] = 1e9
            comp_cols['complexity'] = comp

        # Reorder into requested order
        env_idx = {name: i for i, name in enumerate(env_names)}
        columns: list[np.ndarray] = []
        for name in req_names:
            if name in env_idx:
                columns.append(env_objectives[:, env_idx[name]].astype(np.float32, copy=False))
            else:
                arr = comp_cols.get(name, None)
                if arr is None:
                    raise ValueError(f"Objective '{name}' not produced by env or simulator.")
                columns.append(arr)

        objectives = np.stack(columns, axis=1) if columns else np.zeros((B, 0), dtype=np.float32)

        if return_returns:
            # Stack step-level returns: list[Tensor[Timestep x B]] -> [T, B]
            try:
                if len(environment.returns_history) > 0:
                    ret_stack = torch.stack(environment.returns_history).cpu().numpy()
                else:
                    ret_stack = np.zeros((0, population.pop_size), dtype=np.float32)
            except Exception:
                ret_stack = np.zeros((0, population.pop_size), dtype=np.float32)
            return objectives, ret_stack
        else:
            return objectives


def fitness_fn(
    population: GATreePop,
    data: pd.DataFrame,
    all_feature_names: list,
    start_data_cnt: int,
    stop_data_cnt: int,
    device: str,
    evaluation_config: dict,
) -> np.ndarray:
    """Compatibility wrapper for the old function-style API."""
    sim = FitnessSimulator()
    return sim.simulate(
        population=population,
        data=data,
        all_feature_names=all_feature_names,
        start_data_cnt=start_data_cnt,
        stop_data_cnt=stop_data_cnt,
        device=device,
        evaluation_config=evaluation_config,
    )
