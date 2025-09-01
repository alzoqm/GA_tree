# evolution/multi_objective_evolution.py

import torch
import numpy as np
from typing import Tuple, Optional, Union, List, Dict
from copy import deepcopy

from models.model import GATreePop
from .evolution import Evolution
from .Selection.base import BaseSelection
from .Selection.ppts import PPTSSelection
from .Crossover.base import BaseCrossover
from .Mutation.base import BaseMutation


class MultiObjectiveEvolution(Evolution):
    """
    Multi-Objective Evolution Engine extending the base Evolution class.
    
    This class integrates PPTS (Parameterless Batched Pareto Tournament Selection)
    for multi-objective optimization while maintaining compatibility with existing
    crossover and mutation operators. Provides archive management and diversity
    preservation mechanisms.
    """
    
    def __init__(self, 
                 population: GATreePop,
                 selection: BaseSelection,
                 crossover: BaseCrossover,
                 mutation: BaseMutation,
                 parent_size: int,
                 num_elites: int = 2,
                 archive_size: Optional[int] = None,
                 diversity_preservation: bool = True):
        """
        Initialize Multi-Objective Evolution Engine.
        
        Args:
            population (GATreePop): The population to evolve
            selection (BaseSelection): Selection operator (preferably PPTSSelection)
            crossover (BaseCrossover): Crossover operator
            mutation (BaseMutation): Mutation operator
            parent_size (int): Size of mating pool
            num_elites (int): Number of elites to preserve
            archive_size (Optional[int]): Size of external archive (None = no archive)
            diversity_preservation (bool): Enable diversity preservation mechanisms
        """
        # Initialize base class
        super().__init__(population, selection, crossover, mutation, parent_size, num_elites)
        
        # Multi-objective specific configuration
        self.archive_size = archive_size
        self.diversity_preservation = diversity_preservation
        
        # External archive for storing non-dominated solutions
        self.archive_objectives = None  # [archive_size, num_objectives]
        self.archive_individuals = None  # [archive_size, chromosome_size, ...]
        self.archive_count = 0
        
        # Generation statistics
        self.generation_stats = []
        
        # Check if selection is PPTS for optimal multi-objective performance
        self.using_ppts = isinstance(selection, PPTSSelection)
        if not self.using_ppts:
            print("Warning: Using non-PPTS selection for multi-objective optimization. "
                  "Consider using PPTSSelection for better performance.")
    
    def evolve_multi_objective(self, objectives: torch.Tensor) -> Dict:
        """
        Perform one generation of multi-objective evolution.
        
        Args:
            objectives: [population_size, num_objectives] tensor of objective values (minimization)
            
        Returns:
            evolution_stats: Dictionary containing evolution statistics
        """
        if not self.population.initialized:
            raise RuntimeError("Population is not initialized. Call make_population() first.")
        
        pop_size = self.population.pop_size
        device = objectives.device
        
        # Store generation statistics
        gen_stats = self._compute_generation_statistics(objectives)
        
        # Update external archive if enabled
        if self.archive_size is not None:
            self._update_archive(objectives)
        
        # 1. Elite Selection using multi-objective selection
        if objectives.dim() == 2 and objectives.shape[1] > 1:
            # True multi-objective case
            elite_indices = self.selection.select_elites(objectives, self.num_elites)
        else:
            # Single-objective fallback
            fitness = -objectives.squeeze() if objectives.dim() > 1 else -objectives
            elite_indices = self.selection.select_elites(fitness, self.num_elites)
        
        # Move elites to GPU
        elite_chromosomes_gpu = self.population.population_tensor[elite_indices.to('cpu')].clone().to(device)
        
        # 2. Offspring generation
        num_offspring = pop_size - self.num_elites
        if num_offspring > 0:
            # 2a. Mating pool selection
            if objectives.dim() == 2 and objectives.shape[1] > 1:
                mating_pool_indices = self.selection.select_elites(objectives, self.parent_size)
                mating_pool_objectives = objectives[mating_pool_indices]
            else:
                fitness = -objectives.squeeze() if objectives.dim() > 1 else -objectives
                mating_pool_indices = self.selection.select_elites(fitness, self.parent_size)
                mating_pool_objectives = fitness[mating_pool_indices]
            
            # 2b. Parent selection for crossover
            if objectives.dim() == 2 and objectives.shape[1] > 1:
                relative_parent_indices = self.selection.pick_parents(mating_pool_objectives, num_offspring * 2)
            else:
                relative_parent_indices = self.selection.pick_parents(mating_pool_objectives, num_offspring * 2)
            
            absolute_parent_indices = mating_pool_indices[relative_parent_indices]
            
            # Move parents to GPU
            parents_gpu = self.population.population_tensor[absolute_parent_indices.to('cpu')].to(device)
            
            # 2c. Crossover
            offspring_chromosomes_gpu = self.crossover(parents_gpu)
            
            # 2d. Mutation
            mutated_offspring_gpu = self.mutation(offspring_chromosomes_gpu)
        else:
            mutated_offspring_gpu = torch.empty((0, *self.population.population_tensor.shape[1:]), device=device)
        
        # 3. Diversity enhancement (if enabled)
        if self.diversity_preservation and num_offspring > 0:
            mutated_offspring_gpu = self._enhance_diversity(mutated_offspring_gpu, elite_chromosomes_gpu, device)
        
        # 4. Form next generation
        new_population_tensor_gpu = torch.cat([elite_chromosomes_gpu, mutated_offspring_gpu], dim=0)
        
        # 5. Update population
        self.population.population_tensor.copy_(new_population_tensor_gpu)
        
        # Store statistics
        gen_stats['archive_size'] = self.archive_count if self.archive_size else 0
        gen_stats['diversity_score'] = self._calculate_diversity_score(new_population_tensor_gpu)
        self.generation_stats.append(gen_stats)
        
        return gen_stats
    
    def evolve(self, fitness_or_objectives: torch.Tensor):
        """
        Unified evolve method supporting both single and multi-objective optimization.
        
        Args:
            fitness_or_objectives: [N] for single-objective or [N, M] for multi-objective
        """
        if fitness_or_objectives.dim() == 1:
            # Single-objective: use base class method
            super().evolve(fitness_or_objectives)
        else:
            # Multi-objective: use multi-objective evolution
            self.evolve_multi_objective(fitness_or_objectives)
    
    def _compute_generation_statistics(self, objectives: torch.Tensor) -> Dict:
        """
        Compute statistics for the current generation.
        
        Args:
            objectives: [population_size, num_objectives] objective values
            
        Returns:
            stats: Dictionary of generation statistics
        """
        objectives_np = objectives.cpu().numpy()
        
        # Basic statistics
        stats = {
            'population_size': len(objectives_np),
            'num_objectives': objectives_np.shape[1] if objectives_np.ndim > 1 else 1,
            'generation': len(self.generation_stats),
        }
        
        # Handle invalid individuals
        if objectives_np.ndim > 1:
            valid_mask = ~np.any(np.isin(objectives_np, [-1e9, 1e9]), axis=1)
        else:
            valid_mask = ~np.isin(objectives_np, [-1e9, 1e9])
        
        stats['valid_individuals'] = int(np.sum(valid_mask))
        
        if stats['valid_individuals'] > 0:
            valid_objectives = objectives_np[valid_mask]
            
            if valid_objectives.ndim == 1:
                valid_objectives = valid_objectives.reshape(-1, 1)
            
            # Per-objective statistics
            stats['objective_stats'] = {}
            for i in range(valid_objectives.shape[1]):
                obj_values = valid_objectives[:, i]
                stats['objective_stats'][f'obj_{i}'] = {
                    'mean': float(np.mean(obj_values)),
                    'std': float(np.std(obj_values)),
                    'min': float(np.min(obj_values)),
                    'max': float(np.max(obj_values)),
                    'median': float(np.median(obj_values))
                }
            
            # Pareto front analysis
            if valid_objectives.shape[1] > 1:
                pareto_front_mask = self._find_pareto_front(valid_objectives)
                stats['pareto_front_size'] = int(np.sum(pareto_front_mask))
                stats['pareto_front_ratio'] = stats['pareto_front_size'] / stats['valid_individuals']
                
                if np.any(pareto_front_mask):
                    pareto_objectives = valid_objectives[pareto_front_mask]
                    stats['pareto_front_stats'] = {}
                    for i in range(pareto_objectives.shape[1]):
                        obj_values = pareto_objectives[:, i]
                        stats['pareto_front_stats'][f'obj_{i}'] = {
                            'mean': float(np.mean(obj_values)),
                            'min': float(np.min(obj_values)),
                            'max': float(np.max(obj_values))
                        }
            else:
                # Single objective - best individual
                best_idx = np.argmin(valid_objectives[:, 0])
                stats['best_objective'] = float(valid_objectives[best_idx, 0])
        
        return stats
    
    def _update_archive(self, objectives: torch.Tensor):
        """
        Update the external archive with non-dominated solutions.
        
        Args:
            objectives: [population_size, num_objectives] current population objectives
        """
        if objectives.dim() == 1:
            return  # No archive for single-objective
        
        objectives_np = objectives.cpu().numpy()
        population_tensor_np = self.population.population_tensor.cpu().numpy()
        
        # Filter valid individuals
        valid_mask = ~np.any(np.isin(objectives_np, [-1e9, 1e9]), axis=1)
        if not np.any(valid_mask):
            return
        
        valid_objectives = objectives_np[valid_mask]
        valid_individuals = population_tensor_np[valid_mask]
        
        # Initialize archive if first time
        if self.archive_objectives is None:
            max_archive = min(self.archive_size, len(valid_objectives))
            self.archive_objectives = np.full((self.archive_size, objectives_np.shape[1]), 1e9)
            self.archive_individuals = np.zeros((self.archive_size, *population_tensor_np.shape[1:]))
            
            # Add initial non-dominated solutions
            pareto_mask = self._find_pareto_front(valid_objectives)
            pareto_objectives = valid_objectives[pareto_mask]
            pareto_individuals = valid_individuals[pareto_mask]
            
            n_add = min(len(pareto_objectives), self.archive_size)
            self.archive_objectives[:n_add] = pareto_objectives[:n_add]
            self.archive_individuals[:n_add] = pareto_individuals[:n_add]
            self.archive_count = n_add
            
            return
        
        # Combine current archive with new candidates
        current_archive_objectives = self.archive_objectives[:self.archive_count]
        current_archive_individuals = self.archive_individuals[:self.archive_count]
        
        combined_objectives = np.vstack([current_archive_objectives, valid_objectives])
        combined_individuals = np.vstack([current_archive_individuals, valid_individuals])
        
        # Find non-dominated solutions in combined set
        pareto_mask = self._find_pareto_front(combined_objectives)
        pareto_objectives = combined_objectives[pareto_mask]
        pareto_individuals = combined_individuals[pareto_mask]
        
        # If we have too many, use diversity-based selection
        if len(pareto_objectives) > self.archive_size:
            selected_indices = self._select_diverse_subset(pareto_objectives, self.archive_size)
            pareto_objectives = pareto_objectives[selected_indices]
            pareto_individuals = pareto_individuals[selected_indices]
        
        # Update archive
        self.archive_count = len(pareto_objectives)
        self.archive_objectives[:self.archive_count] = pareto_objectives
        self.archive_individuals[:self.archive_count] = pareto_individuals
        
        # Clear unused slots
        if self.archive_count < self.archive_size:
            self.archive_objectives[self.archive_count:] = 1e9
    
    def _find_pareto_front(self, objectives: np.ndarray) -> np.ndarray:
        """
        Find Pareto front using fast non-dominated sorting.
        
        Args:
            objectives: [population_size, num_objectives] objective values
            
        Returns:
            pareto_mask: [population_size] boolean mask of Pareto front members
        """
        n = len(objectives)
        pareto_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not pareto_mask[i]:
                continue
            for j in range(n):
                if i != j and pareto_mask[j]:
                    if self._dominates(objectives[j], objectives[i]):
                        pareto_mask[i] = False
                        break
        
        return pareto_mask
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 Pareto dominates obj2 (minimization)."""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def _select_diverse_subset(self, objectives: np.ndarray, target_size: int) -> np.ndarray:
        """
        Select diverse subset using crowding distance.
        
        Args:
            objectives: [population_size, num_objectives] objective values
            target_size: Number of solutions to select
            
        Returns:
            selected_indices: Indices of selected solutions
        """
        if len(objectives) <= target_size:
            return np.arange(len(objectives))
        
        # Calculate crowding distances
        distances = self._calculate_crowding_distances(objectives)
        
        # Select individuals with highest crowding distances
        selected_indices = np.argsort(distances)[-target_size:]
        
        return selected_indices
    
    def _calculate_crowding_distances(self, objectives: np.ndarray) -> np.ndarray:
        """Calculate crowding distances for diversity estimation."""
        n, m = objectives.shape
        distances = np.zeros(n)
        
        for obj_idx in range(m):
            sorted_indices = np.argsort(objectives[:, obj_idx])
            
            # Boundary points get maximum distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Normalize by range
            obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]
            if obj_range > 1e-8:
                for i in range(1, n - 1):
                    distance = (objectives[sorted_indices[i + 1], obj_idx] - 
                              objectives[sorted_indices[i - 1], obj_idx]) / obj_range
                    distances[sorted_indices[i]] += distance
        
        return distances
    
    def _enhance_diversity(self, offspring: torch.Tensor, elites: torch.Tensor, device: str) -> torch.Tensor:
        """
        Enhance population diversity through additional mechanisms.
        
        Args:
            offspring: [num_offspring, ...] offspring chromosomes
            elites: [num_elites, ...] elite chromosomes
            device: Device for tensor operations
            
        Returns:
            enhanced_offspring: Diversity-enhanced offspring
        """
        if not self.diversity_preservation or len(offspring) == 0:
            return offspring
        
        # Simple diversity enhancement: add some random individuals if diversity is low
        combined = torch.cat([elites, offspring], dim=0)
        diversity = self._calculate_diversity_score(combined)
        
        # If diversity is low, replace some offspring with random individuals
        if diversity < 0.1:  # Threshold for low diversity
            num_random = max(1, len(offspring) // 10)  # Replace 10% with random
            
            # Generate random individuals (simplified - in practice, use proper initialization)
            random_individuals = torch.randn_like(offspring[:num_random])
            
            # Replace lowest-performing offspring with random ones
            offspring[-num_random:] = random_individuals
        
        return offspring
    
    def _calculate_diversity_score(self, population: torch.Tensor) -> float:
        """
        Calculate a diversity score for the population.
        
        Args:
            population: [population_size, ...] population tensor
            
        Returns:
            diversity_score: Diversity measure (0-1, higher = more diverse)
        """
        if len(population) < 2:
            return 1.0
        
        # Simple diversity measure: average pairwise distance
        # Flatten population for distance calculation
        flat_pop = population.view(len(population), -1)
        
        # Calculate pairwise distances (sample for efficiency)
        sample_size = min(50, len(population))
        indices = torch.randperm(len(population))[:sample_size]
        sample_pop = flat_pop[indices]
        
        # Compute average pairwise L2 distance
        distances = torch.pdist(sample_pop)
        avg_distance = distances.mean().item() if len(distances) > 0 else 0.0
        
        # Normalize by rough estimate of maximum possible distance
        max_distance = torch.norm(torch.ones_like(sample_pop[0])) * 2
        diversity_score = min(1.0, avg_distance / (max_distance + 1e-8))
        
        return diversity_score
    
    def get_archive_solutions(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get solutions from the external archive.
        
        Returns:
            archive_objectives: [archive_count, num_objectives] archive objectives
            archive_individuals: [archive_count, ...] archive individuals
        """
        if self.archive_objectives is None or self.archive_count == 0:
            return None, None
        
        return (self.archive_objectives[:self.archive_count].copy(),
                self.archive_individuals[:self.archive_count].copy())
    
    def get_generation_statistics(self) -> List[Dict]:
        """Get all generation statistics."""
        return self.generation_stats.copy()
    
    def reset_archive(self):
        """Reset the external archive."""
        self.archive_objectives = None
        self.archive_individuals = None
        self.archive_count = 0
    
    def save_state(self, filepath: str):
        """Save the evolution state including archive."""
        state = {
            'population_tensor': self.population.population_tensor,
            'archive_objectives': self.archive_objectives,
            'archive_individuals': self.archive_individuals,
            'archive_count': self.archive_count,
            'generation_stats': self.generation_stats,
            'archive_size': self.archive_size,
            'diversity_preservation': self.diversity_preservation
        }
        torch.save(state, filepath)
    
    def load_state(self, filepath: str):
        """Load the evolution state including archive."""
        state = torch.load(filepath)
        
        self.population.population_tensor = state['population_tensor']
        self.archive_objectives = state.get('archive_objectives')
        self.archive_individuals = state.get('archive_individuals')
        self.archive_count = state.get('archive_count', 0)
        self.generation_stats = state.get('generation_stats', [])
        
        # Update configuration if saved
        if 'archive_size' in state:
            self.archive_size = state['archive_size']
        if 'diversity_preservation' in state:
            self.diversity_preservation = state['diversity_preservation']