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
    ) -> np.ndarray:
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

        # Build adjacency list for CUDA operations
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

        feature_data = data[all_feature_names]
        pbar = tqdm(range(start_data_cnt, stop_data_cnt - 1), desc="Sim", leave=False)
        for entry_index in pbar:
            market_data_row = data.iloc[entry_index]
            next_open_price = torch.tensor(
                data['Open'].iloc[entry_index + 1], dtype=torch.float32, device=device
            )

            feature_values_series = feature_data.iloc[entry_index]

            current_positions_str = [
                {0: 'HOLD', 1: 'SHORT', 2: 'LONG'}[pos.item()] for pos in environment.pos_list
            ]

            predicted_actions = predict_population_cuda(
                population=population,
                feature_values=feature_values_series,
                current_positions=current_positions_str,
                adj_offsets=adj_offsets,
                adj_indices=adj_indices,
                device=device,
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
