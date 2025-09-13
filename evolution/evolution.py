# /evolution/evolution.py
# Unified Evolution class with complete original logic + multi-objective capabilities

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict

from models.model import GATreePop
from .Selection.base import BaseSelection
from .Selection.ppts import PPTSSelection
from .Crossover.base import BaseCrossover
from .Mutation.base import BaseMutation
# [변경] 최적화된 Pareto 랭킹을 위해 MultiObjectiveFitnessEvaluator 임포트
from training.multi_objective_fitness import MultiObjectiveFitnessEvaluator


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
        
        # [신규] Pareto 계산을 위한 유틸리티 평가기 인스턴스화
        # 랭킹 계산 자체는 objective 이름/방향 설정과 무관하므로 빈 설정으로 초기화
        self.evaluator = MultiObjectiveFitnessEvaluator({'objectives': [], 'minimize': []})
        
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
        # [변경] NumPy 변환을 GPU에서 바로 처리 가능한 PyTorch 텐서로 유지
        objectives_tensor = objectives
        
        # Basic statistics
        stats = {
            'population_size': len(objectives_tensor),
            'num_objectives': objectives_tensor.shape[1] if objectives_tensor.ndim > 1 else 1,
            'generation': len(self.generation_stats),
        }
        
        # Handle invalid individuals
        if objectives_tensor.ndim > 1:
            valid_mask = ~torch.any(torch.isin(objectives_tensor, torch.tensor([-1e9, 1e9], device=objectives_tensor.device)), dim=1)
        else:
            valid_mask = ~torch.isin(objectives_tensor, torch.tensor([-1e9, 1e9], device=objectives_tensor.device))
        
        stats['valid_individuals'] = int(torch.sum(valid_mask).item())
        
        if stats['valid_individuals'] > 0:
            valid_objectives = objectives_tensor[valid_mask]
            
            if valid_objectives.dim() == 1:
                valid_objectives = valid_objectives.view(-1, 1)
            
            # Per-objective statistics
            stats['objective_stats'] = {}
            for i in range(valid_objectives.shape[1]):
                obj_values = valid_objectives[:, i]
                stats['objective_stats'][f'obj_{i}'] = {
                    'mean': float(obj_values.mean().item()),
                    'std': float(obj_values.std().item()),
                    'min': float(obj_values.min().item()),
                    'max': float(obj_values.max().item()),
                    'median': float(torch.median(obj_values).item())
                }
            
            # Pareto front analysis for multi-objective
            if valid_objectives.shape[1] > 1:
                # [변경] 느린 _find_pareto_front 대신 최적화된 pareto_rank 사용
                ranks, _ = self.evaluator.pareto_rank(valid_objectives.cpu().numpy(), device=valid_objectives.device)
                pareto_front_mask = (ranks == 0)
                
                stats['pareto_front_size'] = int(np.sum(pareto_front_mask))
                stats['pareto_front_ratio'] = stats['pareto_front_size'] / stats['valid_individuals']
                
                if np.any(pareto_front_mask):
                    pareto_objectives = valid_objectives[pareto_front_mask]
                    stats['pareto_front_stats'] = {}
                    for i in range(pareto_objectives.shape[1]):
                        obj_values = pareto_objectives[:, i]
                        stats['pareto_front_stats'][f'obj_{i}'] = {
                            'mean': float(obj_values.mean().item()),
                            'min': float(obj_values.min().item()),
                            'max': float(obj_values.max().item())
                        }
            else:
                # Single objective - best individual
                best_idx = torch.argmin(valid_objectives[:, 0])
                stats['best_objective'] = float(valid_objectives[best_idx, 0].item())
        
        return stats
    
    def _update_archive(self, objectives: torch.Tensor):
        """
        외부 아카이브를 비지배 솔루션으로 업데이트
        
        Args:
            objectives: [population_size, num_objectives] 현재 집단의 목표 값들
        """
        if objectives.dim() == 1:
            return  # Single-objective에서는 아카이브 사용 안함
        
        population_tensor = self.population.population_tensor
        device = objectives.device

        # Filter valid individuals
        valid_mask = ~torch.any(torch.isin(objectives, torch.tensor([-1e9, 1e9], device=device)), dim=1)
        if not torch.any(valid_mask):
            return
        
        valid_objectives = objectives[valid_mask]
        valid_individuals = population_tensor[valid_mask]
        
        # Initialize archive if first time
        if self.archive_objectives is None:
            self.archive_objectives = torch.full((self.archive_size, objectives.shape[1]), 1e9, device=device)
            self.archive_individuals = torch.zeros((self.archive_size, *population_tensor.shape[1:]), device=device)
            
            # [변경] 최적화된 pareto_rank 사용
            ranks, _ = self.evaluator.pareto_rank(valid_objectives.cpu().numpy(), device=device)
            pareto_mask = torch.from_numpy(ranks == 0).to(device)
            
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
        
        combined_objectives = torch.cat([current_archive_objectives, valid_objectives], dim=0)
        combined_individuals = torch.cat([current_archive_individuals, valid_individuals], dim=0)
        
        # [변경] 최적화된 pareto_rank 및 crowding distance 사용
        ranks, distances = self.evaluator.pareto_rank(combined_objectives.cpu().numpy(), device=device)
        pareto_mask = torch.from_numpy(ranks == 0).to(device)
        
        pareto_objectives = combined_objectives[pareto_mask]
        pareto_individuals = combined_individuals[pareto_mask]
        pareto_distances = torch.from_numpy(distances[pareto_mask.cpu().numpy()]).to(device)
        
        # If we have too many, use diversity-based selection (crowding distance)
        if len(pareto_objectives) > self.archive_size:
            # Sort by distance (descending) and take top `archive_size`
            sorted_indices = torch.argsort(pareto_distances, descending=True)
            selected_indices = sorted_indices[:self.archive_size]
            pareto_objectives = pareto_objectives[selected_indices]
            pareto_individuals = pareto_individuals[selected_indices]
        
        # Update archive
        self.archive_count = len(pareto_objectives)
        self.archive_objectives[:self.archive_count] = pareto_objectives
        self.archive_individuals[:self.archive_count] = pareto_individuals
        
        # Clear unused slots
        if self.archive_count < self.archive_size:
            self.archive_objectives[self.archive_count:] = 1e9
    
    # [삭제] 아래의 느린 Python 기반 Pareto 및 거리 계산 함수들을 제거합니다.
    # _find_pareto_front, _dominates, _select_diverse_subset, _calculate_crowding_distances
    
    def _enhance_diversity(self, offspring: torch.Tensor, elites: torch.Tensor, device: torch.device) -> torch.Tensor:
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
    
    def get_archive_solutions(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        외부 아카이브에서 솔루션들 가져오기
        
        Returns:
            archive_objectives: [archive_count, num_objectives] 아카이브 목표값들 (Tensor)
            archive_individuals: [archive_count, ...] 아카이브 개체들 (Tensor)
        """
        if self.archive_objectives is None or self.archive_count == 0:
            return None, None
        
        return (self.archive_objectives[:self.archive_count].clone(),
                self.archive_individuals[:self.archive_count].clone())
    
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
            'archive_objectives': self.archive_objectives.cpu() if self.archive_objectives is not None else None,
            'archive_individuals': self.archive_individuals.cpu() if self.archive_individuals is not None else None,
            'archive_count': self.archive_count,
            'generation_stats': self.generation_stats,
            'archive_size': self.archive_size,
            'diversity_preservation': self.diversity_preservation
        }
        torch.save(state, filepath)
    
    def load_state(self, filepath: str):
        """아카이브를 포함한 진화 상태 로드"""
        state = torch.load(filepath)
        device = self.population.population_tensor.device
        
        self.population.population_tensor = state['population_tensor'].to(device)
        self.archive_objectives = state.get('archive_objectives').to(device) if state.get('archive_objectives') is not None else None
        self.archive_individuals = state.get('archive_individuals').to(device) if state.get('archive_individuals') is not None else None
        self.archive_count = state.get('archive_count', 0)
        self.generation_stats = state.get('generation_stats', [])
        
        # Update configuration if saved
        if 'archive_size' in state:
            self.archive_size = state['archive_size']
        if 'diversity_preservation' in state:
            self.diversity_preservation = state['diversity_preservation']
