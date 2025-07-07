import torch
from typing import List
from .base import BaseMutation

class ChainMutation(BaseMutation):
    """
    여러 변이 연산자들을 순차적으로 적용하는 체인 클래스.
    """
    def __init__(self, mutations: List[BaseMutation]):
        """
        적용할 변이 연산자 리스트로 초기화합니다.
        
        Args:
            mutations (List[BaseMutation]): 순차적으로 적용할 BaseMutation 객체의 리스트.
        """
        super().__init__(prob=1.0) # 체인 자체의 확률은 1, 내부 연산자가 각자의 확률을 가짐
        self.mutations = mutations

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        """
        주어진 개체들에 대해 모든 변이 연산을 순서대로 적용합니다.
        """
        mutated_chromosomes = chromosomes
        for mutation_op in self.mutations:
            mutated_chromosomes = mutation_op(mutated_chromosomes)
        return mutated_chromosomes