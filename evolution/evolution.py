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
                 num_elites: int = 2):
        """
        Evolution 엔진 초기화.

        Args:
            population (GATreePop): 진화시킬 GATree 집단 객체.
            selection (BaseSelection): 선택 연산자.
            crossover (BaseCrossover): 교차 연산자.
            mutation (BaseMutation): 변이 연산자.
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
            
        self.population = population
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
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
            # 2a. 부모 선택
            # 교차를 위해 2개씩 짝지어 부모를 선택해야 하므로 num_offspring * 2 만큼 선택
            parent_indices = self.selection.pick_parents(fitness, num_offspring * 2)
            parents = self.population.population_tensor[parent_indices]
            
            # 2b. 교차 (Crossover)
            # crossover.__call__은 (num_parents, ...) -> (num_offspring, ...) 형태를 따라야 함
            offspring_chromosomes = self.crossover(parents)
            
            # 2c. 변이 (Mutation)
            mutated_offspring = self.mutation(offspring_chromosomes)
        else:
            mutated_offspring = torch.empty((0, *self.population.population_tensor.shape[1:]))

        # 3. 다음 세대 구성
        # 엘리트와 변이된 자식들을 합쳐 새로운 집단 텐서를 만듦
        new_population_tensor = torch.cat([elite_chromosomes, mutated_offspring], dim=0)
        
        # 4. 집단 업데이트
        # GATreePop의 텐서 데이터를 새로운 세대의 텐서로 덮어씀
        self.population.population_tensor.copy_(new_population_tensor)
        print(f"Evolution complete. {self.num_elites} elites preserved. {num_offspring} new offsprings created.")