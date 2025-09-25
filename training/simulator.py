# /training/simulator.py

import numpy as np
import torch
import pandas as pd
from typing import List, Optional
from tqdm import tqdm

from training.trading_env import TradingEnvironment
from training.predictor import build_adjacency_list_cuda, predict_population_cuda
from models.model import GATreePop
from models.constants import (
    COL_NODE_TYPE,
    COL_PARENT_IDX,
    COL_DEPTH,
    COL_PARAM_1,
    NODE_TYPE_UNUSED,
    NODE_TYPE_ROOT_BRANCH,
    ROOT_BRANCH_LONG,
    ROOT_BRANCH_HOLD,
    ROOT_BRANCH_SHORT,
)

torch.set_grad_enabled(False)


def _compute_root_branch_balance_penalty(population: GATreePop) -> torch.Tensor:
    """Return per-tree penalties (0~1) for imbalance across root branches."""

    pt = population.population_tensor
    if pt.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=pt.device)

    node_types = pt[:, :, COL_NODE_TYPE]
    parent_idx = pt[:, :, COL_PARENT_IDX].to(torch.long)
    depths = pt[:, :, COL_DEPTH].to(torch.long)

    used_mask = node_types != float(NODE_TYPE_UNUSED)
    non_root_mask = used_mask & (parent_idx >= 0)

    B, N = parent_idx.shape
    branch_ids = torch.full((B, N), -1, dtype=torch.long, device=pt.device)

    root_mask = (node_types == float(NODE_TYPE_ROOT_BRANCH)) & (parent_idx == -1)
    if root_mask.any():
        branch_ids[root_mask] = pt[:, :, COL_PARAM_1].to(torch.long)[root_mask]

    max_depth = int(depths.max().item()) if depths.numel() > 0 else 0
    parent_idx_safe = torch.where(parent_idx >= 0, parent_idx, torch.zeros_like(parent_idx))

    for depth_val in range(1, max_depth + 1):
        depth_mask = used_mask & (depths == depth_val)
        if not depth_mask.any():
            continue
        parent_branch = branch_ids.gather(1, parent_idx_safe.clamp_max(N - 1))
        parent_branch = torch.where(parent_idx >= 0, parent_branch, torch.full_like(parent_branch, -1))
        branch_ids[depth_mask] = parent_branch[depth_mask]

    total_nodes = non_root_mask.sum(dim=1)
    total_nodes_clamped = torch.clamp(total_nodes, min=1)
    total_nodes_float = total_nodes_clamped.to(torch.float32)

    counts = []
    for branch_type in (ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT):
        branch_mask = non_root_mask & (branch_ids == branch_type)
        counts.append(branch_mask.sum(dim=1).to(torch.float32))
    counts = torch.stack(counts, dim=1) if counts else torch.zeros((B, 0), dtype=torch.float32, device=pt.device)

    ratios = torch.where(
        total_nodes_float.unsqueeze(1) > 0,
        counts / total_nodes_float.unsqueeze(1),
        torch.zeros_like(counts),
    )

    dev = torch.abs(ratios - (1.0 / 3.0))
    penalty = dev.sum(dim=1) / (4.0 / 3.0)
    penalty = torch.where(total_nodes == 0, torch.ones_like(penalty), penalty)
    return penalty.clamp_(0.0, 1.0).to(torch.float32)


class FitnessSimulator:
    """Run trading simulation and return multi-objective metrics."""

    def __init__(self):
        pass

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
        return_env_state: bool = False,
        return_node_stats: bool = False,
    ) -> (
        np.ndarray
        | tuple[np.ndarray, np.ndarray]
        | tuple[np.ndarray, dict]
        | tuple[np.ndarray, np.ndarray, dict]
        | tuple[np.ndarray, np.ndarray, dict, dict]
        | tuple[np.ndarray, dict, dict]
    ):
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
            'calmar_ratio', 'sortino_ratio', 'omega_ratio', 'information_ratio', 'turnover'
        }
        # Simulator-side complexity metrics
        SUPPORTED_COMPLEXITY_OBJECTIVES = {
            'comp_size_norm',
            'comp_depth_norm',
            'complexity',
            'root_balance_penalty',
        }

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
            max_nodes=population.max_nodes,
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
        results = torch.empty((B, 4), dtype=torch.float32, device=device)
        bfs = torch.empty((B, N), dtype=torch.int32, device=device)
        visited = torch.empty((B, N), dtype=torch.int32, device=device)
        positions = torch.empty((B,), dtype=torch.int32, device=device)
        features = torch.empty((F,), dtype=torch.float32, device=device)

        # LUT: env.pos_list (0: HOLD, 1: SHORT, 2: LONG) -> ROOT_BRANCH_*
        lut = torch.tensor([ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT, ROOT_BRANCH_LONG], dtype=torch.int32, device=device)

        # Monotonic step id for kernel visited generation marking
        step_id = 1

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
            predicted_actions, predicted_nodes = predict_population_cuda(
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

            environment.step(market_data_row, next_open_price, (predicted_actions, predicted_nodes))
        pbar.close()

        minimum_trades = evaluation_config.get('fitness_function', {}).get('minimum_trades', 40)
        wfo_min_trades = evaluation_config.get('min_trades_per_window', None)
        if wfo_min_trades is not None:
            minimum_trades = wfo_min_trades

        env_objectives = environment.get_multi_objective_metrics(minimum_date=minimum_trades)
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

        need_size = 'comp_size_norm' in req_names or 'complexity' in req_names
        need_depth = 'comp_depth_norm' in req_names or 'complexity' in req_names
        cx_cfg: dict[str, object] = {}
        min_nodes = 0
        if (need_size or need_depth) and isinstance(evaluation_config, dict):
            cx_cfg = evaluation_config.get('complexity', {}) or {}
            min_nodes_cfg = cx_cfg.get('min_nodes', cx_cfg.get('min_node', 0))
            try:
                min_nodes = int(float(min_nodes_cfg))
            except (TypeError, ValueError):
                min_nodes = 0
            min_nodes = max(0, min_nodes)
        size_used_np: np.ndarray | None = None
        small_complexity_mask_np: np.ndarray | None = None
        need_root_balance = 'root_balance_penalty' in req_names
        comp_cols: dict[str, np.ndarray] = {}
        comp_size_np = None
        comp_depth_np = None

        # Derive invalid mask from env outputs
        if env_objectives.size > 0:
            invalid_mask = np.any(
                np.isin(env_objectives, [-1e9, 1e9]) | ~np.isfinite(env_objectives), axis=1
            )
        else:
            invalid_mask = np.zeros((B,), dtype=bool)

        if need_root_balance:
            balance_penalty = _compute_root_branch_balance_penalty(population)
            balance_penalty_np = balance_penalty.detach().to('cpu').numpy().astype(np.float32)
            balance_penalty_np[invalid_mask] = 1e9
            comp_cols['root_balance_penalty'] = balance_penalty_np

        # Complexity components computation as needed
        if need_size or need_depth:
            pt = population.population_tensor
            used = (pt[:, :, COL_NODE_TYPE] != float(NODE_TYPE_UNUSED))
            used_f = used.to(dtype=torch.float32)
            size_used = used.sum(dim=1).to(dtype=torch.float32)
            if min_nodes > 0:
                small_complexity_mask = size_used <= float(min_nodes)
            else:
                small_complexity_mask = torch.zeros_like(size_used, dtype=torch.bool)
            size_used_np = size_used.detach().to('cpu').numpy().astype(np.float32)
            small_complexity_mask_np = small_complexity_mask.detach().to('cpu').numpy() if min_nodes > 0 else np.zeros((B,), dtype=bool)

            if need_size:
                denom = float(population.max_nodes - min_nodes)
                if denom <= 0:
                    comp_size = torch.zeros_like(size_used)
                else:
                    comp_size = torch.clamp((size_used - float(min_nodes)) / denom, min=0.0, max=1.0)
                if min_nodes > 0:
                    comp_size = torch.where(small_complexity_mask, torch.zeros_like(comp_size), comp_size)
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
                if min_nodes > 0:
                    comp_depth = torch.where(small_complexity_mask, torch.zeros_like(comp_depth), comp_depth)
                comp_depth_np = comp_depth.detach().to('cpu').numpy().astype(np.float32)
                comp_depth_np[invalid_mask] = 1e9
                if 'comp_depth_norm' in req_names:
                    comp_cols['comp_depth_norm'] = comp_depth_np

        # Composite single complexity metric
        if 'complexity' in req_names:
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
            if min_nodes > 0:
                if small_complexity_mask_np is None and size_used_np is not None:
                    small_complexity_mask_np = size_used_np <= float(min_nodes)
                if small_complexity_mask_np is not None:
                    comp[small_complexity_mask_np] = 0.0
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

        # Optionally return node-level CUDA attribution stats for pruning
        if isinstance(evaluation_config, dict):
            config_wants_node_stats = bool(evaluation_config.get('return_node_stats', False))
        else:
            config_wants_node_stats = False
        return_node_stats = bool(return_node_stats) or config_wants_node_stats

        def _collect_returns() -> np.ndarray:
            try:
                if len(environment.returns_history) > 0:
                    return torch.stack(environment.returns_history).cpu().numpy()
            except Exception:
                pass
            return np.zeros((0, population.pop_size), dtype=np.float32)

        def _collect_node_stats() -> dict:
            try:
                return {
                    'node_attempt_count': environment.node_attempt_count,
                    'node_apply_count': environment.node_apply_count,
                    'node_noop_count': environment.node_noop_count,
                    'node_stall_count': getattr(environment, 'node_stall_count', None),
                    'node_win_count': environment.node_win_count,
                    'node_realized_pnl_sum': environment.node_realized_pnl_sum,
                    'node_exposure_sum': environment.node_exposure_sum,
                    'total_steps': int(max(0, stop_data_cnt - start_data_cnt - 1)),
                }
            except Exception:
                return {
                    'node_attempt_count': None,
                    'node_apply_count': None,
                    'node_noop_count': None,
                    'node_stall_count': None,
                    'node_win_count': None,
                    'node_realized_pnl_sum': None,
                    'node_exposure_sum': None,
                    'total_steps': int(max(0, stop_data_cnt - start_data_cnt - 1)),
                }

        def _collect_env_state() -> dict:
            def _to_numpy(t: torch.Tensor) -> np.ndarray:
                return t.detach().to('cpu').numpy().astype(np.float32, copy=False)

            state: dict[str, object] = {}
            try:
                state.update(
                    sum_returns=_to_numpy(environment.sum_returns),
                    count_returns=_to_numpy(environment.count_returns),
                    total_profit_agg=_to_numpy(environment.total_profit_agg),
                    total_loss_agg=_to_numpy(environment.total_loss_agg),
                    count_wins=_to_numpy(environment.count_wins),
                    trade_total_count=_to_numpy(environment.trade_total_count),
                    trade_win_count=_to_numpy(environment.trade_win_count),
                )
            except Exception:
                pass

            try:
                state['max_drawdown'] = _to_numpy(environment.max_drawdown)
            except Exception:
                pass
            try:
                state['compound_value'] = _to_numpy(environment.compound_value)
            except Exception:
                pass

            # Meta configuration captured for downstream metric recomputation
            try:
                state['risk_free_rate'] = float(getattr(environment, 'risk_free_rate', 0.0))
            except Exception:
                state['risk_free_rate'] = 0.0
            try:
                state['lookback_period'] = int(getattr(environment, 'lookback_period', 252))
            except Exception:
                state['lookback_period'] = 252
            try:
                state['mar'] = float(getattr(environment, 'mar', 0.0))
            except Exception:
                state['mar'] = 0.0

            # Benchmark returns (if provided)
            bench_returns: list[np.ndarray] = []
            try:
                for tensor in getattr(environment, 'bench_returns_history', []) or []:
                    bench_returns.append(tensor.detach().to('cpu').numpy().astype(np.float32, copy=False))
            except Exception:
                bench_returns = []
            state['bench_returns'] = bench_returns

            # Total steps processed this run (used for turnover)
            try:
                state['total_steps'] = int(max(0, len(environment.returns_history)))
            except Exception:
                state['total_steps'] = 0

            return state

        payload: list[object] = [objectives]
        if return_returns:
            payload.append(_collect_returns())
        if return_node_stats:
            payload.append(_collect_node_stats())
        if return_env_state:
            payload.append(_collect_env_state())

        if len(payload) == 1:
            return objectives
        if len(payload) == 2:
            return payload[0], payload[1]
        if len(payload) == 3:
            return payload[0], payload[1], payload[2]
        return payload[0], payload[1], payload[2], payload[3]


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
