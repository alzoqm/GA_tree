# fitness/simulator.py

import numpy as np
import torch
import pandas as pd
from typing import List, Optional
from tqdm import tqdm

from training.trading_env import TradingEnvironment
from training.predictor import build_adjacency_list_cuda, predict_population_cuda
from models.model import GATreePop

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
        objectives_config = evaluation_config.get(
            'multi_objectives',
            evaluation_config.get(
                'objectives',
                {
                    'objectives': ['mean_return', 'max_drawdown', 'sharpe_ratio', 'win_rate'],
                    'minimize': [False, True, False, False],
                    'risk_free_rate': 0.0,
                    'volatility_lookback': 30,
                },
            ),
        )

        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Build adjacency list for CUDA operations (once per simulate call)
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

        # Pre-extract features to numpy for fast access, and preallocate GPU buffers
        feature_data = data[all_feature_names]
        feature_np = feature_data.to_numpy(dtype=np.float32, copy=False)

        B = population.pop_size
        N = population.max_nodes
        F = feature_np.shape[1]

        # Preallocate reusable GPU buffers
        results = torch.empty((B, 4), dtype=torch.float32, device=device)
        bfs = torch.empty((B, N), dtype=torch.int32, device=device)
        visited = torch.zeros((B, N), dtype=torch.int32, device=device)
        positions = torch.empty((B,), dtype=torch.int32, device=device)
        features = torch.empty((F,), dtype=torch.float32, device=device)

        # LUT: env.pos_list mapping -> ROOT_BRANCH_* constants
        # env.pos_list: 0=HOLD, 1=SHORT, 2=LONG
        from models.constants import ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT, ROOT_BRANCH_LONG
        lut = torch.tensor([ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT, ROOT_BRANCH_LONG], dtype=torch.int32, device=device)

        pbar = tqdm(range(start_data_cnt, stop_data_cnt - 1), desc="Sim", leave=False)
        for entry_index in pbar:
            market_data_row = data.iloc[entry_index]
            next_open_price = torch.as_tensor(
                data['Open'].iloc[entry_index + 1], dtype=torch.float32, device=device
            )

            # Fill features buffer for this step
            # Note: from_numpy creates a CPU tensor view; then copy to GPU buffer
            step_feat_cpu = torch.from_numpy(feature_np[entry_index])
            features.copy_(step_feat_cpu.to(device=device))

            # Map env positions to ROOT_BRANCH_* on GPU
            positions.copy_(lut[environment.pos_list])

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
                validate=False,
                sync=False,
            )
            if predicted_actions is None:
                raise RuntimeError(
                    "CUDA prediction failed. Check if gatree_cuda module is compiled and available."
                )

            environment.step(market_data_row, next_open_price, predicted_actions)
        pbar.close()

        minimum_trades = evaluation_config.get('fitness_function', {}).get('minimum_trades', 40)
        wfo_min_trades = evaluation_config.get('min_trades_per_window', None)
        if wfo_min_trades is not None:
            minimum_trades = wfo_min_trades

        objectives = environment.get_multi_objective_metrics(minimum_date=minimum_trades)

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
