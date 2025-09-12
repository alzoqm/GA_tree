# fitness/simulator.py

import numpy as np
import torch
import pandas as pd
from typing import List, Optional
from tqdm import tqdm

from training.trading_env import TradingEnvironment
from training.predictor import build_adjacency_list_cuda, predict_population_cuda
from models.model import GATreePop
from models.constants import (
    COL_NODE_TYPE, COL_DEPTH, NODE_TYPE_UNUSED,
)

torch.set_grad_enabled(False)


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
        from models.constants import ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT, ROOT_BRANCH_LONG
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
