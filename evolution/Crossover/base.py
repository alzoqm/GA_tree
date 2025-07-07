# evolution/Crossover/base.py
import torch
from abc import ABC, abstractmethod

class BaseCrossover(ABC):
    """
    유전 알고리즘의 '교차' 연산을 위한 추상 기반 클래스.
    """
    def __init__(self, rate: float = 0.8):
        self.rate = rate

    @abstractmethod
    def __call__(self, parents: torch.Tensor) -> torch.Tensor:
        """
        부모 개체들의 텐서를 받아 교차를 통해 생성된 자식 개체들의 텐서를 반환합니다.

        Args:
            parents (torch.Tensor): 부모 개체들의 정보를 담은 텐서. 
                                    Shape: (num_parents, max_nodes, node_dim).

        Returns:
            torch.Tensor: 생성된 자식 개체들의 정보를 담은 텐서.
                          Shape: (num_offspring, max_nodes, node_dim).
        """
        pass