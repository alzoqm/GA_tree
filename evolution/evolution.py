# evolution/evolution.py
# Unified Evolution class with complete original logic + multi-objective capabilities

import torch
import numpy as np
from typing import Tuple, Optional, Union, List, Dict
from copy import deepcopy

from models.model import GATreePop
from .Selection.base import BaseSelection
from .Selection.ppts import PPTSSelection
from .Crossover.base import BaseCrossover
from .Mutation.base import BaseMutation


class Evolution:
    """
    통합 진화 엔진 - 원본 진화 로직 완전 보존 + 다중 목표 최적화 지원
    
    이 클래스는 기존의 모든 집단 진화, 선택, 교차, 변이 로직을 그대로 유지하면서
    다중 목표 최적화, PPTS 선택, 아카이브 관리 등의 기능을 추가합니다.
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
        진화 엔진 초기화
        
        Args:
            population (GATreePop): 진화시킬 GATree 집단 객체
            selection (BaseSelection): 선택 연산자 (PPTS 권장)
            crossover (BaseCrossover): 교차 연산자
            mutation (BaseMutation): 변이 연산자
            parent_size (int): 교배에 참여할 상위 개체 집단(교배 풀)의 크기
            num_elites (int): 다음 세대로 그대로 전달될 엘리트 개체의 수
            archive_size (Optional[int]): 외부 아카이브 크기 (None = 아카이브 없음)
            diversity_preservation (bool): 다양성 보존 메커니즘 활성화
        """
        # === 원본 타입 검증 완전 보존 ===
        if not isinstance(population, GATreePop):
            raise TypeError("population must be an instance of GATreePop.")
        if not isinstance(selection, BaseSelection):
            raise TypeError("selection must be an instance of a class inheriting from BaseSelection.")
        if not isinstance(crossover, BaseCrossover):
            raise TypeError("crossover must be an instance of a class inheriting from BaseCrossover.")
        if not isinstance(mutation, BaseMutation):
            raise TypeError("mutation must be an instance of a class inheriting from BaseMutation.")
        if not (0 <= num_elites < population.pop_size):
            raise ValueError("num_elites must be a non-negative integer smaller than pop_size.")
            
        # === 원본 parent_size 유효성 검사 완전 보존 ===
        if not isinstance(parent_size, int):
            raise TypeError("parent_size must be an integer.")
        if not (num_elites <= parent_size <= population.pop_size):
            raise ValueError("parent_size must be between num_elites and pop_size (inclusive).")
        # 자손을 만들어야 하는데, 부모가 2명 미만인 경우 방지
        num_offspring_check = population.pop_size - num_elites
        if num_offspring_check > 0 and parent_size < 2:
            raise ValueError("parent_size must be at least 2 to create offspring.")
            
        # === 원본 기본 속성 완전 보존 ===
        self.population = population
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.parent_size = parent_size
        self.num_elites = num_elites

        # === 다중 목표 최적화 추가 기능 ===
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

    def evolve(self, objectives: torch.Tensor) -> Optional[Dict]:
        """
        통합 진화 메서드 - 다중 목표 최적화 전용
        
        원본의 모든 GPU 텐서 연산과 집단 관리 로직을 보존하면서
        다중 목표 선택과 아카이브 관리 기능을 통합합니다.
        
        Args:
            objectives: [population_size, num_objectives] 목표 값들
            
        Returns:
            evolution_stats: 진화 통계 (선택적)
        """
        if not self.population.initialized:
            raise RuntimeError("Population is not initialized. Call make_population() first.")
        
        pop_size = self.population.pop_size
        device = objectives.device  # 연산이 수행될 GPU 장치

        # Ensure objectives are 2D for multi-objective processing
        if objectives.dim() == 1:
            objectives = objectives.unsqueeze(1)  # Convert [N] to [N, 1]

        # Store generation statistics
        gen_stats = self._compute_generation_statistics(objectives)
        
        # Update external archive if enabled
        if self.archive_size is not None:
            self._update_archive(objectives)

        # === 원본 엘리트 선택 로직 보존 (다중 목표 대응) ===
        # PPTS나 다중 목표 지원 선택자를 사용하여 엘리트 선택
        elite_indices = self.selection.select_elites(objectives, self.num_elites)
        # 인덱스를 population 텐서의 디바이스로 맞춰 불필요한 CPU 왕복 제거
        pop_tensor = self.population.population_tensor
        elite_indices_on_pop = elite_indices.to(pop_tensor.device)
        elite_chromosomes_gpu = pop_tensor.index_select(0, elite_indices_on_pop).clone()
        if elite_chromosomes_gpu.device != device:
            elite_chromosomes_gpu = elite_chromosomes_gpu.to(device)
        
        # === 원본 자식(offspring) 생성 로직 완전 보존 ===
        num_offspring = pop_size - self.num_elites
        if num_offspring > 0:
            # 2a. 교배 풀(Mating Pool) 선택 (GPU에서 수행)
            mating_pool_indices = self.selection.select_elites(objectives, self.parent_size)
            mating_pool_objectives = objectives[mating_pool_indices]
            
            # 2b. 실제 부모 선택 (GPU에서 수행)
            relative_parent_indices = self.selection.pick_parents(mating_pool_objectives, num_offspring * 2)
            absolute_parent_indices = mating_pool_indices[relative_parent_indices]
            
            # 부모 인덱스를 population 텐서 디바이스로 정렬 후 슬라이스
            abs_parent_idx_on_pop = absolute_parent_indices.to(pop_tensor.device)
            parents_gpu = pop_tensor.index_select(0, abs_parent_idx_on_pop)
            if parents_gpu.device != device:
                parents_gpu = parents_gpu.to(device)

            # 2c. 교차 (Crossover) - GPU In, GPU Out (원본 로직 보존)
            offspring_chromosomes_gpu = self.crossover(parents_gpu)
            
            # 2d. 변이 (Mutation) - GPU In, GPU Out (원본 로직 보존)  
            mutated_offspring_gpu = self.mutation(offspring_chromosomes_gpu)
            
        else:
            mutated_offspring_gpu = torch.empty((0, *self.population.population_tensor.shape[1:]), device=device)

        # === 다중 목표 다양성 향상 (선택적) ===
        if self.diversity_preservation and num_offspring > 0:
            mutated_offspring_gpu = self._enhance_diversity(mutated_offspring_gpu, elite_chromosomes_gpu, device)

        # === 원본 다음 세대 구성 로직 완전 보존 ===
        # 모든 GPU 연산 결과를 결합
        new_population_tensor_gpu = torch.cat([elite_chromosomes_gpu, mutated_offspring_gpu], dim=0)
        
        # === 원본 집단 업데이트 로직 완전 보존 ===
        # 새로운 세대를 population 텐서로 복사 (디바이스 간 복사 없이 가능하면 동일 디바이스에서 수행)
        self.population.population_tensor.copy_(new_population_tensor_gpu)

        # Store statistics
        if gen_stats is not None:
            gen_stats['archive_size'] = self.archive_count if self.archive_size else 0
            gen_stats['diversity_score'] = self._calculate_diversity_score(new_population_tensor_gpu)
            self.generation_stats.append(gen_stats)
            return gen_stats
        
        return None
    
    def _compute_generation_statistics(self, objectives: torch.Tensor) -> Dict:
        """
        현재 세대 통계 계산
        
        Args:
            objectives: [population_size, num_objectives] 목표 값들
            
        Returns:
            stats: 세대 통계 딕셔너리
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
            
            # Pareto front analysis for multi-objective
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
        외부 아카이브를 비지배 솔루션으로 업데이트
        
        Args:
            objectives: [population_size, num_objectives] 현재 집단의 목표 값들
        """
        if objectives.dim() == 1:
            return  # Single-objective에서는 아카이브 사용 안함
        
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
        고속 비지배 정렬을 사용한 파레토 프론트 찾기
        
        Args:
            objectives: [population_size, num_objectives] 목표 값들
            
        Returns:
            pareto_mask: [population_size] 파레토 프론트 멤버의 불린 마스크
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
        """obj1이 obj2를 파레토 지배하는지 확인 (최소화 기준)"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def _select_diverse_subset(self, objectives: np.ndarray, target_size: int) -> np.ndarray:
        """
        크라우딩 거리를 사용한 다양한 부분집합 선택
        
        Args:
            objectives: [population_size, num_objectives] 목표 값들
            target_size: 선택할 솔루션 수
            
        Returns:
            selected_indices: 선택된 솔루션들의 인덱스
        """
        if len(objectives) <= target_size:
            return np.arange(len(objectives))
        
        # Calculate crowding distances
        distances = self._calculate_crowding_distances(objectives)
        
        # Select individuals with highest crowding distances
        selected_indices = np.argsort(distances)[-target_size:]
        
        return selected_indices
    
    def _calculate_crowding_distances(self, objectives: np.ndarray) -> np.ndarray:
        """다양성 추정을 위한 크라우딩 거리 계산"""
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
        추가 메커니즘을 통한 집단 다양성 향상
        
        Args:
            offspring: [num_offspring, ...] 자손 염색체들
            elites: [num_elites, ...] 엘리트 염색체들
            device: 텐서 연산용 디바이스
            
        Returns:
            enhanced_offspring: 다양성이 향상된 자손들
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
        집단의 다양성 점수 계산
        
        Args:
            population: [population_size, ...] 집단 텐서
            
        Returns:
            diversity_score: 다양성 측정값 (0-1, 높을수록 더 다양함)
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
        외부 아카이브에서 솔루션들 가져오기
        
        Returns:
            archive_objectives: [archive_count, num_objectives] 아카이브 목표값들
            archive_individuals: [archive_count, ...] 아카이브 개체들
        """
        if self.archive_objectives is None or self.archive_count == 0:
            return None, None
        
        return (self.archive_objectives[:self.archive_count].copy(),
                self.archive_individuals[:self.archive_count].copy())
    
    def get_generation_statistics(self) -> List[Dict]:
        """모든 세대 통계 가져오기"""
        return self.generation_stats.copy()
    
    def reset_archive(self):
        """외부 아카이브 재설정"""
        self.archive_objectives = None
        self.archive_individuals = None
        self.archive_count = 0
    
    def save_state(self, filepath: str):
        """아카이브를 포함한 진화 상태 저장"""
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
        """아카이브를 포함한 진화 상태 로드"""
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
