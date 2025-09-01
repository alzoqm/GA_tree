# training/multi_objective_fitness.py

import numpy as np
import torch
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from .multi_objective_trading_env import MultiObjectiveTradingEnvironment, multi_objective_fitness_fn


class MultiObjectiveFitnessEvaluator:
    """
    Multi-objective fitness evaluator that provides utilities for managing
    and evaluating multiple objectives in the genetic algorithm context.
    """
    
    def __init__(self, objectives_config: dict):
        """
        Initialize the multi-objective fitness evaluator.
        
        Args:
            objectives_config (dict): Configuration for objectives including:
                - objectives: List of objective names
                - minimize: List of booleans indicating minimization direction
                - weights: Optional weights for weighted sum evaluation
                - normalization: Normalization strategy ('minmax', 'zscore', 'none')
        """
        self.objectives_config = objectives_config
        self.objective_names = objectives_config['objectives']
        self.minimize_objectives = objectives_config['minimize']
        self.num_objectives = len(self.objective_names)
        
        # Validation
        if len(self.minimize_objectives) != self.num_objectives:
            raise ValueError("minimize list must match objectives list length")
        
        # Optional configuration
        self.weights = objectives_config.get('weights', None)
        self.normalization = objectives_config.get('normalization', 'minmax')
        
        # Statistics tracking for normalization
        self.objective_stats = {}
        self.history = []
    
    def evaluate_population(
        self, 
        population,
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
        Evaluate population using multi-objective fitness function.
        
        Returns:
            objectives: [population_size, num_objectives] array of objective values
        """
        return multi_objective_fitness_fn(
            population=population,
            data=data,
            all_feature_names=all_feature_names,
            adj_offsets=adj_offsets,
            adj_indices=adj_indices,
            start_data_cnt=start_data_cnt,
            stop_data_cnt=stop_data_cnt,
            device=device,
            evaluation_config=evaluation_config
        )
    
    def normalize_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """
        Normalize objectives according to the configured strategy.
        
        Args:
            objectives: [population_size, num_objectives] raw objective values
            
        Returns:
            normalized_objectives: [population_size, num_objectives] normalized values
        """
        if self.normalization == 'none':
            return objectives
        
        normalized = objectives.copy()
        
        for i in range(self.num_objectives):
            obj_values = objectives[:, i]
            valid_mask = ~np.isin(obj_values, [-1e9, 1e9]) & ~np.isnan(obj_values)
            
            if not np.any(valid_mask):
                # No valid values - keep as is
                continue
            
            valid_values = obj_values[valid_mask]
            
            if self.normalization == 'minmax':
                min_val, max_val = np.min(valid_values), np.max(valid_values)
                if min_val != max_val:
                    normalized[valid_mask, i] = (obj_values[valid_mask] - min_val) / (max_val - min_val)
                else:
                    normalized[valid_mask, i] = 0.5
            
            elif self.normalization == 'zscore':
                mean_val, std_val = np.mean(valid_values), np.std(valid_values)
                if std_val > 1e-8:
                    normalized[valid_mask, i] = (obj_values[valid_mask] - mean_val) / std_val
                else:
                    normalized[valid_mask, i] = 0.0
            
            # Update statistics for future use
            if valid_mask.any():
                self.objective_stats[self.objective_names[i]] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'min': np.min(valid_values),
                    'max': np.max(valid_values)
                }
        
        return normalized
    
    def calculate_weighted_fitness(self, objectives: np.ndarray) -> np.ndarray:
        """
        Calculate weighted fitness scores from multi-objective values.
        
        Args:
            objectives: [population_size, num_objectives] objective values
            
        Returns:
            fitness: [population_size] weighted fitness scores
        """
        if self.weights is None:
            # Equal weights
            weights = np.ones(self.num_objectives) / self.num_objectives
        else:
            weights = np.array(self.weights)
            if len(weights) != self.num_objectives:
                raise ValueError("Weights length must match number of objectives")
        
        # Normalize objectives first
        normalized = self.normalize_objectives(objectives)
        
        # For minimization objectives, we want lower values to contribute positively to fitness
        # For maximization objectives (stored as negated), we want higher absolute values
        fitness_contrib = normalized.copy()
        
        for i in range(self.num_objectives):
            if self.minimize_objectives[i]:
                # Minimization: lower normalized value = higher fitness contribution
                fitness_contrib[:, i] = 1.0 - fitness_contrib[:, i]
            # For maximization objectives stored as negated values, 
            # higher normalized values already represent better fitness
        
        # Weighted sum
        fitness = np.sum(fitness_contrib * weights, axis=1)
        
        # Handle invalid individuals
        invalid_mask = np.any(np.isin(objectives, [-1e9, 1e9]), axis=1)
        fitness[invalid_mask] = -1.0
        
        return fitness
    
    def pareto_rank(self, objectives: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Pareto ranking and crowding distance for objectives.
        
        Args:
            objectives: [population_size, num_objectives] objective values
            
        Returns:
            ranks: [population_size] Pareto rank (0 = best front)
            distances: [population_size] crowding distances
        """
        population_size = len(objectives)
        ranks = np.zeros(population_size, dtype=int)
        distances = np.zeros(population_size)
        
        # Handle invalid individuals
        valid_mask = ~np.any(np.isin(objectives, [-1e9, 1e9]), axis=1)
        valid_objectives = objectives[valid_mask]
        
        if len(valid_objectives) == 0:
            return ranks, distances
        
        # Fast non-dominated sorting
        fronts = []
        domination_counts = np.zeros(len(valid_objectives))
        dominated_solutions = [[] for _ in range(len(valid_objectives))]
        
        # Calculate domination relationships
        for i in range(len(valid_objectives)):
            for j in range(len(valid_objectives)):
                if i != j:
                    if self._dominates(valid_objectives[i], valid_objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(valid_objectives[j], valid_objectives[i]):
                        domination_counts[i] += 1
            
            if domination_counts[i] == 0:
                if len(fronts) == 0:
                    fronts.append([])
                fronts[0].append(i)
        
        # Build subsequent fronts
        front_idx = 0
        while front_idx < len(fronts) and fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            front_idx += 1
        
        # Assign ranks and calculate crowding distances
        valid_indices = np.where(valid_mask)[0]
        
        for front_num, front in enumerate(fronts):
            if not front:
                continue
            
            # Assign ranks
            for idx in front:
                ranks[valid_indices[idx]] = front_num
            
            # Calculate crowding distances for this front
            front_objectives = valid_objectives[front]
            front_distances = self._crowding_distance(front_objectives)
            
            for i, idx in enumerate(front):
                distances[valid_indices[idx]] = front_distances[i]
        
        # Set invalid individuals to worst rank
        ranks[~valid_mask] = len(fronts)
        
        return ranks, distances
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """
        Check if obj1 Pareto dominates obj2.
        
        Args:
            obj1: First objective vector
            obj2: Second objective vector
            
        Returns:
            True if obj1 dominates obj2
        """
        # obj1 dominates obj2 if:
        # - obj1 is <= obj2 in all objectives (for minimization)
        # - obj1 is < obj2 in at least one objective
        
        all_better_or_equal = True
        at_least_one_better = False
        
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:  # Worse in this objective
                all_better_or_equal = False
                break
            elif obj1[i] < obj2[i]:  # Better in this objective
                at_least_one_better = True
        
        return all_better_or_equal and at_least_one_better
    
    def _crowding_distance(self, front_objectives: np.ndarray) -> np.ndarray:
        """
        Calculate crowding distance for a front.
        
        Args:
            front_objectives: [front_size, num_objectives] objectives for one front
            
        Returns:
            distances: [front_size] crowding distances
        """
        front_size = len(front_objectives)
        if front_size <= 2:
            return np.full(front_size, float('inf'))
        
        distances = np.zeros(front_size)
        
        for obj_idx in range(self.num_objectives):
            # Sort by this objective
            sorted_indices = np.argsort(front_objectives[:, obj_idx])
            
            # Boundary points get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate range for normalization
            obj_range = (front_objectives[sorted_indices[-1], obj_idx] - 
                        front_objectives[sorted_indices[0], obj_idx])
            
            if obj_range > 1e-8:
                for i in range(1, front_size - 1):
                    distance = (front_objectives[sorted_indices[i + 1], obj_idx] - 
                              front_objectives[sorted_indices[i - 1], obj_idx]) / obj_range
                    distances[sorted_indices[i]] += distance
        
        return distances
    
    def get_objective_summary(self, objectives: np.ndarray) -> Dict:
        """
        Generate summary statistics for objectives.
        
        Args:
            objectives: [population_size, num_objectives] objective values
            
        Returns:
            summary: Dictionary with summary statistics
        """
        summary = {
            'objective_names': self.objective_names,
            'minimize_objectives': self.minimize_objectives,
            'population_size': len(objectives),
            'statistics': {}
        }
        
        for i, obj_name in enumerate(self.objective_names):
            obj_values = objectives[:, i]
            valid_mask = ~np.isin(obj_values, [-1e9, 1e9]) & ~np.isnan(obj_values)
            
            if np.any(valid_mask):
                valid_values = obj_values[valid_mask]
                summary['statistics'][obj_name] = {
                    'mean': float(np.mean(valid_values)),
                    'std': float(np.std(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'median': float(np.median(valid_values)),
                    'valid_count': int(np.sum(valid_mask)),
                    'minimize': self.minimize_objectives[i]
                }
            else:
                summary['statistics'][obj_name] = {
                    'mean': None, 'std': None, 'min': None, 'max': None,
                    'median': None, 'valid_count': 0,
                    'minimize': self.minimize_objectives[i]
                }
        
        return summary


def create_multi_objective_evaluator(objectives_config: dict) -> MultiObjectiveFitnessEvaluator:
    """
    Factory function to create a multi-objective fitness evaluator.
    
    Args:
        objectives_config (dict): Configuration for objectives
        
    Returns:
        evaluator: MultiObjectiveFitnessEvaluator instance
    """
    return MultiObjectiveFitnessEvaluator(objectives_config)


# Predefined objective configurations
PRESET_OBJECTIVES = {
    'balanced': {
        'objectives': ['mean_return', 'max_drawdown', 'sharpe_ratio', 'win_rate'],
        'minimize': [False, True, False, False],
        'weights': [0.3, 0.3, 0.2, 0.2],
        'normalization': 'minmax'
    },
    
    'return_focused': {
        'objectives': ['mean_return', 'compound_value', 'max_drawdown'],
        'minimize': [False, False, True],
        'weights': [0.4, 0.4, 0.2],
        'normalization': 'minmax'
    },
    
    'risk_focused': {
        'objectives': ['max_drawdown', 'volatility', 'sharpe_ratio', 'sortino_ratio'],
        'minimize': [True, True, False, False],
        'weights': [0.3, 0.2, 0.25, 0.25],
        'normalization': 'minmax'
    },
    
    'comprehensive': {
        'objectives': ['mean_return', 'max_drawdown', 'sharpe_ratio', 'win_rate', 'profit_factor', 'calmar_ratio'],
        'minimize': [False, True, False, False, False, False],
        'weights': [0.2, 0.2, 0.2, 0.15, 0.15, 0.1],
        'normalization': 'minmax'
    }
}


def get_preset_objectives(preset_name: str) -> dict:
    """
    Get a predefined objective configuration.
    
    Args:
        preset_name (str): Name of the preset ('balanced', 'return_focused', 'risk_focused', 'comprehensive')
        
    Returns:
        objectives_config: Configuration dictionary
    """
    if preset_name not in PRESET_OBJECTIVES:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_OBJECTIVES.keys())}")
    
    return PRESET_OBJECTIVES[preset_name].copy()