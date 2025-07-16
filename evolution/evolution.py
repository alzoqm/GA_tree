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
        한 세대의 진화를 수행합니다.

        Args:
            fitness (torch.Tensor): 현재 집단의 각 개체에 대한 적합도 점수. (pop_size,)
        """
        if not self.population.initialized:
            raise RuntimeError("Population is not initialized. Call make_population() first.")
        
        pop_size = self.population.pop_size
        
        # 1. 엘리트 선택 (Elitism)
        elite_indices = self.selection.select_elites(fitness, self.num_elites)
        # 엘리트 개체의 텐서 데이터를 깊은 복사하여 보존
        elite_chromosomes = self.population.population_tensor[elite_indices].clone()
        
        # 2. 자식(offspring) 생성
        num_offspring = pop_size - self.num_elites
        if num_offspring > 0:
            # --- [수정] 부모 선택 로직 전체 변경 ---
            
            # 2a. 교배 풀(Mating Pool) 선택
            # 전체 집단에서 적합도가 높은 parent_size개의 고유한 개체를 선택하여 교배 풀을 구성합니다.
            mating_pool_indices = self.selection.select_elites(fitness, self.parent_size)
            
            # 2b. 교배 풀 내에서 실제 교배할 부모 선택
            # 교배 풀에 속한 개체들의 적합도 정보만 추출합니다.
            mating_pool_fitness = fitness[mating_pool_indices]
            
            # 교배 풀의 적합도를 이용해, 자손을 생성할 부모를 (중복 허용하여) 선택합니다.
            # 이때 반환되는 인덱스는 '교배 풀 내에서의 상대적 인덱스'입니다.
            relative_parent_indices = self.selection.pick_parents(
                mating_pool_fitness, num_offspring * 2
            )
            
            # 상대적 인덱스를 전체 집단에서의 절대적 인덱스로 변환합니다.
            absolute_parent_indices = mating_pool_indices[relative_parent_indices]
            
            # 최종 선택된 부모의 텐서 데이터를 가져옵니다.
            parents = self.population.population_tensor[absolute_parent_indices]

            # --- 수정된 로직 끝 ---
            
            # 2c. 교차 (Crossover)
            offspring_chromosomes = self.crossover(parents)
            
            # 2d. 변이 (Mutation)
            mutated_offspring = self.mutation(offspring_chromosomes)
        else:
            mutated_offspring = torch.empty((0, *self.population.population_tensor.shape[1:]))

        # 3. 다음 세대 구성
        # 엘리트와 변이된 자식들을 합쳐 새로운 집단 텐서를 만듦
        new_population_tensor = torch.cat([elite_chromosomes, mutated_offspring], dim=0)
        
        # 4. 집단 업데이트
        # GATreePop의 텐서 데이터를 새로운 세대의 텐서로 덮어씀
        self.population.population_tensor.copy_(new_population_tensor)
        # print(f"Evolution complete. {self.num_elites} elites preserved. {num_offspring} new offsprings created from a mating pool of size {self.parent_size}.")