# training/multi_objective_training.py

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

from models.model import GATreePop
from evolution.multi_objective_evolution import MultiObjectiveEvolution
from evolution.Selection import PPTSSelection
from evolution.Crossover import ChainCrossover, SubtreeCrossover, NodeCrossover, RootBranchCrossover
from evolution.Mutation import (
    ChainMutation, AddNodeMutation, DeleteNodeMutation, NodeParamMutation,
    ReinitializeNodeMutation, AddSubtreeMutation, DeleteSubtreeMutation
)

from .multi_objective_trading_env import MultiObjectiveTradingEnvironment, multi_objective_fitness_fn
from .multi_objective_fitness import MultiObjectiveFitnessEvaluator, get_preset_objectives
from .predictor import build_adjacency_list_cuda


def multi_objective_generation_training(
    data_1m: pd.DataFrame,
    evolution: MultiObjectiveEvolution,
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
    objectives_config: dict,
    best_archive=None,
    start_gen: int = 0
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    """
    Multi-objective genetic algorithm training loop.
    
    Args:
        data_1m: Market data DataFrame
        evolution: MultiObjectiveEvolution instance
        skip_data_cnt: Start index for training data
        valid_skip_data_cnt: End index for training data
        chromosomes_size: Population size
        gen_loop: Number of generations
        best_size: Size of best individuals archive
        elite_size: Number of elites per generation
        device: Device for computations
        warming_step: Generation to start archive management
        evaluation_config: Evaluation configuration
        output_dir: Output directory for checkpoints
        objectives_config: Multi-objective configuration
        best_archive: Existing archive (optional)
        start_gen: Starting generation (for resuming)
        
    Returns:
        best_chromosomes: Best individuals archive
        best_objectives: Corresponding objective values
    """
    
    # Setup directories
    checkpoint_dir = os.path.join(output_dir, 'multi_objective_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize fitness evaluator
    evaluator = MultiObjectiveFitnessEvaluator(objectives_config)
    
    # Initialize archive
    if best_archive is None:
        best_chromosomes = None
        best_objectives = None
    else:
        best_chromosomes, best_objectives = best_archive
    
    # Setup data
    if not isinstance(data_1m.index, pd.DatetimeIndex):
        data_1m.index = pd.to_datetime(data_1m.index)
    
    population = evolution.population
    all_feature_names = population.all_features
    
    # Generation loop
    for gen_idx in range(start_gen, gen_loop):
        print(f'Generation {gen_idx}:')
        
        # Build adjacency list for CUDA operations
        adj_offsets, adj_indices = build_adjacency_list_cuda(population, strict_overflow=False)
        if adj_offsets is None:
            raise RuntimeError("Failed to build adjacency list on GPU.")
        
        # Evaluate population using multi-objective fitness
        train_objectives = multi_objective_fitness_fn(
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
        
        # Convert to torch tensor for evolution
        train_objectives_tensor = torch.from_numpy(train_objectives).to(device)
        
        # Archive management after warming period
        if gen_idx >= warming_step:
            # Find valid individuals
            valid_mask = ~np.any(np.isin(train_objectives, [-1e9, 1e9]), axis=1)
            
            if np.any(valid_mask):
                valid_objectives = train_objectives[valid_mask]
                valid_chromosomes = population.population_tensor[valid_mask].clone()
                
                # Find Pareto front
                pareto_front_mask = evaluator._find_pareto_front(valid_objectives)
                if np.any(pareto_front_mask):
                    pareto_objectives = valid_objectives[pareto_front_mask]
                    pareto_chromosomes = valid_chromosomes[pareto_front_mask]
                    
                    print(f"  Found {len(pareto_objectives)} Pareto optimal solutions")
                    
                    # Update archive
                    if best_chromosomes is None:
                        best_objectives = pareto_objectives.copy()
                        best_chromosomes = pareto_chromosomes.clone()
                    else:
                        # Combine with existing archive
                        combined_objectives = np.vstack([best_objectives, pareto_objectives])
                        combined_chromosomes = torch.cat([best_chromosomes, pareto_chromosomes])
                        
                        # Find non-dominated solutions in combined set
                        combined_pareto_mask = evaluator._find_pareto_front(combined_objectives)
                        best_objectives = combined_objectives[combined_pareto_mask]
                        best_chromosomes = combined_chromosomes[combined_pareto_mask]
                        
                        # Limit archive size
                        if len(best_objectives) > best_size:
                            # Use diversity-based selection to reduce archive
                            distances = evaluator._calculate_crowding_distances(best_objectives)
                            selected_indices = np.argsort(distances)[-best_size:]
                            best_objectives = best_objectives[selected_indices]
                            best_chromosomes = best_chromosomes[selected_indices]
                    
                    print(f"  Archive size: {len(best_objectives)}")
        
        # Generation statistics
        generation_summary = evaluator.get_objective_summary(train_objectives)
        print(f"  Valid individuals: {generation_summary['statistics'].get('mean_return', {}).get('valid_count', 0)}")
        
        # Print objective statistics
        for obj_name, stats in generation_summary['statistics'].items():
            if stats['valid_count'] > 0:
                direction = "↓" if stats['minimize'] else "↑"
                print(f"    {obj_name}: {stats['mean']:.4f}±{stats['std']:.4f} "
                      f"[{stats['min']:.4f}, {stats['max']:.4f}] {direction}")
        
        # Save checkpoint
        checkpoint_data = {
            "generation": gen_idx,
            "population_state_dict": population.population_tensor.cpu(),
            "best_objectives": best_objectives,
            "best_chromosomes": best_chromosomes.cpu() if best_chromosomes is not None else None,
            "generation_stats": generation_summary,
            "objectives_config": objectives_config,
            "evolution_stats": evolution.get_generation_statistics()
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'multi_obj_generation_{gen_idx}.pt')
        torch.save(checkpoint_data, checkpoint_path)
        
        # Evolve population
        evolution.evolve(train_objectives_tensor)
        population.population_tensor = population.population_tensor.to(device)
        population.reorganize_nodes()
        
        print()  # Empty line for readability
    
    return best_chromosomes, best_objectives


def multi_objective_generation_test(
    data_1m: pd.DataFrame,
    population: GATreePop,
    start_data_cnt: int,
    end_data_cnt: int,
    device: str,
    evaluation_config: dict,
    objectives_config: dict
) -> Tuple[np.ndarray, Dict]:
    """
    Test multi-objective population on data.
    
    Args:
        data_1m: Market data DataFrame
        population: Population to test
        start_data_cnt: Start index for test data
        end_data_cnt: End index for test data
        device: Device for computations
        evaluation_config: Evaluation configuration
        objectives_config: Multi-objective configuration
        
    Returns:
        test_objectives: [population_size, num_objectives] test results
        test_summary: Summary statistics
    """
    
    if not isinstance(data_1m.index, pd.DatetimeIndex):
        data_1m.index = pd.to_datetime(data_1m.index)
    
    all_feature_names = population.all_features
    
    # Build adjacency list
    adj_offsets, adj_indices = build_adjacency_list_cuda(population, strict_overflow=False)
    if adj_offsets is None:
        raise RuntimeError("Failed to build adjacency list on GPU for test.")
    
    # Evaluate on test data
    test_objectives = multi_objective_fitness_fn(
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
    
    # Generate summary
    evaluator = MultiObjectiveFitnessEvaluator(objectives_config)
    test_summary = evaluator.get_objective_summary(test_objectives)
    
    return test_objectives, test_summary


def create_multi_objective_evolution(
    population: GATreePop,
    selection_config: dict,
    crossover_config: dict,
    mutation_config: dict,
    evolution_config: dict
) -> MultiObjectiveEvolution:
    """
    Factory function to create a multi-objective evolution engine.
    
    Args:
        population: GATree population
        selection_config: Selection configuration
        crossover_config: Crossover configuration  
        mutation_config: Mutation configuration
        evolution_config: Evolution configuration
        
    Returns:
        evolution: MultiObjectiveEvolution instance
    """
    
    # Create selection operator (default to PPTS)
    selection_type = selection_config.get('type', 'ppts')
    if selection_type == 'ppts':
        selection = PPTSSelection(
            chunk_size=selection_config.get('chunk_size', 8192),
            generator=torch.Generator().manual_seed(selection_config.get('seed', 42))
        )
    else:
        raise ValueError(f"Unsupported selection type: {selection_type}")
    
    # Create crossover operator
    crossover_operators = []
    for op_config in crossover_config['operators']:
        op_type = op_config['type']
        op_prob = op_config['probability']
        
        if op_type == 'subtree':
            crossover_operators.append((SubtreeCrossover(), op_prob))
        elif op_type == 'node':
            crossover_operators.append((NodeCrossover(), op_prob))
        elif op_type == 'root_branch':
            crossover_operators.append((RootBranchCrossover(), op_prob))
        else:
            raise ValueError(f"Unknown crossover type: {op_type}")
    
    crossover = ChainCrossover(crossover_operators)
    
    # Create mutation operator
    mutation_operators = []
    for op_config in mutation_config['operators']:
        op_type = op_config['type']
        op_prob = op_config['probability']
        
        if op_type == 'add_node':
            mutation_operators.append((AddNodeMutation(), op_prob))
        elif op_type == 'delete_node':
            mutation_operators.append((DeleteNodeMutation(), op_prob))
        elif op_type == 'node_param':
            mutation_operators.append((NodeParamMutation(), op_prob))
        elif op_type == 'reinitialize_node':
            mutation_operators.append((ReinitializeNodeMutation(), op_prob))
        elif op_type == 'add_subtree':
            mutation_operators.append((AddSubtreeMutation(), op_prob))
        elif op_type == 'delete_subtree':
            mutation_operators.append((DeleteSubtreeMutation(), op_prob))
        else:
            raise ValueError(f"Unknown mutation type: {op_type}")
    
    mutation = ChainMutation(mutation_operators)
    
    # Create evolution engine
    evolution = MultiObjectiveEvolution(
        population=population,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        parent_size=evolution_config['parent_size'],
        num_elites=evolution_config['num_elites'],
        archive_size=evolution_config.get('archive_size', None),
        diversity_preservation=evolution_config.get('diversity_preservation', True)
    )
    
    return evolution


# Example configuration templates
DEFAULT_MULTI_OBJECTIVE_CONFIG = {
    'objectives': ['mean_return', 'max_drawdown', 'sharpe_ratio', 'win_rate'],
    'minimize': [False, True, False, False],
    'weights': [0.3, 0.3, 0.2, 0.2],  # For weighted sum evaluation
    'normalization': 'minmax',
    'risk_free_rate': 0.0,
    'volatility_lookback': 30
}

DEFAULT_SELECTION_CONFIG = {
    'type': 'ppts',
    'chunk_size': 8192,
    'seed': 42
}

DEFAULT_EVOLUTION_CONFIG = {
    'parent_size': 50,
    'num_elites': 10,
    'archive_size': 100,
    'diversity_preservation': True
}


def run_multi_objective_training_example():
    """
    Example usage of the multi-objective training system.
    This function demonstrates how to set up and run multi-objective optimization.
    """
    print("Multi-Objective Trading GA Training Example")
    print("=" * 50)
    
    # This is a template - actual implementation would need:
    # - Real market data loading
    # - Proper population initialization
    # - Complete configuration setup
    # - Error handling and logging
    
    print("Configuration templates created successfully!")
    print("Key components:")
    print("- PPTS Selection for multi-objective optimization")
    print("- Multi-objective trading environment") 
    print("- Archive management for Pareto front preservation")
    print("- Comprehensive objective evaluation")
    print()
    print("To use this system:")
    print("1. Load your market data into a pandas DataFrame")
    print("2. Initialize a GATreePop population") 
    print("3. Configure objectives using get_preset_objectives() or custom config")
    print("4. Create evolution engine with create_multi_objective_evolution()")
    print("5. Run training with multi_objective_generation_training()")


if __name__ == "__main__":
    run_multi_objective_training_example()