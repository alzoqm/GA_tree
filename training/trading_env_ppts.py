# /training/trading_env_ppts.py - Multi-Objective Trading Environment for PPTS

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
from training.multi_objective_trading_env import MultiObjectiveTradingEnvironment, multi_objective_fitness_fn

# ##############################################################################
#           Multi-Objective Trading Environment for PPTS
# ##############################################################################

# Position mapping for CUDA prediction (kept for compatibility)
INT_TO_POSITION_MAP = {
    0: 'HOLD',
    1: 'SHORT', 
    2: 'LONG'
}

class PPTSTradingEnvironment(MultiObjectiveTradingEnvironment):
    """
    Multi-Objective Trading Environment for PPTS (Parameterless Pareto Tournament Selection).
    Extends MultiObjectiveTradingEnvironment to provide multi-objective fitness vectors
    instead of single fitness scores.
    """
    
    def __init__(self, chromosomes_size, env_config, objectives_config=None, device='cpu'):
        # Set default multi-objective configuration if not provided
        if objectives_config is None:
            objectives_config = {
                'objectives': ['mean_return', 'max_drawdown', 'sharpe_ratio', 'win_rate'],
                'minimize': [False, True, False, False],  # minimize drawdown, maximize others
                'risk_free_rate': 0.0,
                'volatility_lookback': 30
            }
        
        super().__init__(chromosomes_size, env_config, objectives_config, device)

    def get_ppts_metrics(self, minimum_date=40):
        """
        Get multi-objective metrics for PPTS selection.
        Returns objective vectors instead of single fitness values.
        
        Args:
            minimum_date (int): Minimum number of trading periods required for valid metrics
            
        Returns:
            objectives: [chromosomes_size, num_objectives] array of objective values (for minimization)
        """
        return self.get_multi_objective_metrics(minimum_date)

# ##############################################################################
#           Multi-Objective Fitness Functions for PPTS
# ##############################################################################

torch.set_grad_enabled(False)

def ppts_fitness_fn(
    population: GATreePop,
    data: pd.DataFrame,
    all_feature_names: list,
    adj_offsets: torch.Tensor,
    adj_indices: torch.Tensor,
    start_data_cnt: int,
    stop_data_cnt: int,
    device: str,
    evaluation_config: dict
) -> np.ndarray:
    """
    Multi-objective fitness function for PPTS.
    Returns objective vectors instead of single fitness values.
    
    Args:
        population: GATreePop population
        data: Market data DataFrame
        all_feature_names: List of feature names
        adj_offsets: Adjacency list offsets
        adj_indices: Adjacency list indices
        start_data_cnt: Start index for evaluation
        stop_data_cnt: End index for evaluation
        device: Device for computations
        evaluation_config: Evaluation configuration
        
    Returns:
        objectives: [population_size, num_objectives] array of objective values (for minimization)
    """
    env_config = evaluation_config['simulation_env']
    objectives_config = evaluation_config.get('multi_objectives', {
        'objectives': ['mean_return', 'max_drawdown', 'sharpe_ratio', 'win_rate'],
        'minimize': [False, True, False, False],
        'risk_free_rate': 0.0,
        'volatility_lookback': 30
    })

    environment = PPTSTradingEnvironment(
        chromosomes_size=population.pop_size,
        env_config=env_config,
        objectives_config=objectives_config,
        device=device
    )
    
    pbar = tqdm(range(start_data_cnt, stop_data_cnt - 1), desc="PPTS Multi-Objective Simulation")
    feature_data = data[all_feature_names]

    for entry_index in pbar:
        market_data_row = data.iloc[entry_index]
        next_open_price = torch.tensor(data['Open'].iloc[entry_index + 1], dtype=torch.float32, device=device)
        feature_values_series = feature_data.iloc[entry_index]
        
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
            
        environment.step(market_data_row, next_open_price, predicted_actions)

    pbar.close()
    
    minimum_trades = evaluation_config.get('fitness_function', {}).get('minimum_trades', 40)
    objectives = environment.get_ppts_metrics(minimum_date=minimum_trades)
    
    return objectives

# ##############################################################################
#           Main Training/Test Loop Functions for PPTS
# ##############################################################################

def ppts_generation_valid(
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
    best_objectives=None,
    best_chromosomes=None,
    start_gen: int = 0
):
    """
    Multi-objective training loop for PPTS.
    Uses objective vectors instead of single fitness scores.
    """
    
    best_objectives = best_objectives
    best_chromosomes = best_chromosomes
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if not isinstance(data_1m.index, pd.DatetimeIndex):
        data_1m.index = pd.to_datetime(data_1m.index)

    population = evolution.population
    all_feature_names = population.all_features

    for gen_idx in range(start_gen, gen_loop):
        print(f'generation {gen_idx}: ')
        
        adj_offsets, adj_indices = build_adjacency_list_cuda(population, strict_overflow=False)
        if adj_offsets is None:
            raise RuntimeError("Failed to build adjacency list on GPU.")

        # Get multi-objective metrics instead of single fitness
        train_objectives = ppts_fitness_fn(
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
        
        # Use PPTS selection directly with objectives
        # The evolution engine should handle multi-objective selection
        evolution.evolve(torch.from_numpy(train_objectives).to(device))
        population.population_tensor = population.population_tensor.to(device)
        population.reorganize_nodes()
        
        # Save checkpoint data
        gen_data = {
            "generation": gen_idx,
            "population_state_dict": population.population_tensor,
            "train_objectives": train_objectives,
        }
        
        torch.save(gen_data, os.path.join(checkpoint_dir, f'chrom_generation_{gen_idx}.pt'))
        evolution.population.save(os.path.join(checkpoint_dir, f'generation_{gen_idx}.pt'))
        
    # For PPTS, we return the final population instead of "best" individuals
    # since multi-objective optimization doesn't have a single "best"
    return population.population_tensor, train_objectives


def ppts_generation_test(
    data_1m: pd.DataFrame,
    population: GATreePop,
    start_data_cnt: int,
    end_data_cnt: int,
    device: str,
    evaluation_config: dict
) -> np.ndarray:
    """
    Multi-objective test evaluation for PPTS.
    Returns objective vectors instead of single metrics.
    """

    if not isinstance(data_1m.index, pd.DatetimeIndex):
        data_1m.index = pd.to_datetime(data_1m.index)

    all_feature_names = population.all_features
    
    adj_offsets, adj_indices = build_adjacency_list_cuda(population, strict_overflow=False)
    if adj_offsets is None:
        raise RuntimeError("Failed to build adjacency list on GPU for test.")
        
    test_objectives = ppts_fitness_fn(
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
        
    return test_objectives


# Utility functions for backward compatibility
def calculate_fitness(objectives, weights=None):
    """
    Convert multi-objective values to single fitness for compatibility.
    This is a fallback function for systems expecting single fitness values.
    
    Args:
        objectives: [N, M] array of objectives or [N] array of single values
        weights: Optional weights for combining objectives
        
    Returns:
        fitness: [N] array of single fitness values
    """
    if objectives.ndim == 1:
        # Already single-objective
        return objectives
    
    if weights is None:
        # Default equal weights
        weights = np.ones(objectives.shape[1]) / objectives.shape[1]
    
    # Convert minimization objectives to maximization (negate)
    # and combine with weights
    fitness_values = np.sum(-objectives * np.array(weights), axis=1)
    
    return fitness_values