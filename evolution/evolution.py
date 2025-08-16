# evolution/evolution.py
import torch
from copy import deepcopy

# 프로젝트 구조에 따라 올바른 경로에서 GATreePop을 임포트해야 합니다.
# 이 파일이 프로젝트 루트에서 실행된다고 가정합니다.
from models.model import GATreePop 
from .Selection.base import BaseSelection
from .Crossover.base import BaseCrossover
from .Mutation.base import BaseMutation

class Evolution:
    """
    GATreePop 집단을 진화시키는 메인 컨트롤러 클래스.
    """
    def __init__(self, 
                 population: GATreePop, 
                 selection: BaseSelection, 
                 crossover: BaseCrossover, 
                 mutation: BaseMutation,
                 parent_size: int,
                 num_elites: int = 2):
        """
        Evolution 엔진 초기화.

        Args:
            population (GATreePop): 진화시킬 GATree 집단 객체.
            selection (BaseSelection): 선택 연산자.
            crossover (BaseCrossover): 교차 연산자.
            mutation (BaseMutation): 변이 연산자.
            parent_size (int): 교배에 참여할 자격이 있는 상위 개체 집단(교배 풀)의 크기.
            num_elites (int): 다음 세대로 그대로 전달될 엘리트 개체의 수.
        """
        # --- 타입 검증 ---
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
            
        # --- [신규] parent_size 유효성 검사 ---
        if not isinstance(parent_size, int):
            raise TypeError("parent_size must be an integer.")
        if not (num_elites <= parent_size <= population.pop_size):
            raise ValueError("parent_size must be between num_elites and pop_size (inclusive).")
        # 자손을 만들어야 하는데, 부모가 2명 미만인 경우 방지
        num_offspring_check = population.pop_size - num_elites
        if num_offspring_check > 0 and parent_size < 2:
            raise ValueError("parent_size must be at least 2 to create offspring.")
            
        self.population = population
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.parent_size = parent_size
        self.num_elites = num_elites

    def evolve(self, fitness: torch.Tensor):
        """
        [수정됨] 한 세대의 진화를 수행합니다. 모든 텐서 연산은 GPU에서 처리됩니다.

        Args:
            fitness (torch.Tensor): 현재 집단의 각 개체에 대한 적합도 점수 (GPU에 위치해야 함).
        """
        if not self.population.initialized:
            raise RuntimeError("Population is not initialized. Call make_population() first.")
        
        pop_size = self.population.pop_size
        device = fitness.device # 연산이 수행될 GPU 장치

        # 1. 엘리트 선택 (Elitism)
        # elite_indices는 fitness와 같은 device(GPU)에 생성됨
        elite_indices = self.selection.select_elites(fitness, self.num_elites)
        # CPU에 있는 population_tensor에서 엘리트를 선택한 후, GPU로 이동
        elite_chromosomes_gpu = self.population.population_tensor[elite_indices.to('cpu')].clone().to(device)
        
        # 2. 자식(offspring) 생성
        num_offspring = pop_size - self.num_elites
        if num_offspring > 0:
            # 2a. 교배 풀(Mating Pool) 선택 (GPU에서 수행)
            mating_pool_indices = self.selection.select_elites(fitness, self.parent_size)
            mating_pool_fitness = fitness[mating_pool_indices]
            
            # 2b. 실제 부모 선택 (GPU에서 수행)
            relative_parent_indices = self.selection.pick_parents(mating_pool_fitness, num_offspring * 2)
            absolute_parent_indices = mating_pool_indices[relative_parent_indices]
            
            # CPU의 population_tensor에서 부모를 선택하고, 즉시 GPU로 이동
            parents_gpu = self.population.population_tensor[absolute_parent_indices.to('cpu')].to(device)

            # 2c. 교차 (Crossover) - GPU In, GPU Out
            offspring_chromosomes_gpu = self.crossover(parents_gpu)
            
            # 2d. 변이 (Mutation) - GPU In, GPU Out
            mutated_offspring_gpu = self.mutation(offspring_chromosomes_gpu)
            
        else:
            mutated_offspring_gpu = torch.empty((0, *self.population.population_tensor.shape[1:]), device=device)

        # 3. 다음 세대 구성 (모두 GPU에서 수행)
        new_population_tensor_gpu = torch.cat([elite_chromosomes_gpu, mutated_offspring_gpu], dim=0)
        
        # 4. 집단 업데이트 (GPU -> CPU로 최종 데이터 복사)
        self.population.population_tensor.copy_(new_population_tensor_gpu)