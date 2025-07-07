import torch
from abc import ABC, abstractmethod

class BaseMutation(ABC):
    """
    유전 알고리즘의 '변이' 연산을 위한 추상 기반 클래스.
    """
    def __init__(self, prob: float = 0.1):
        self.prob = prob

    @abstractmethod
    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        """
        개체(들)의 텐서를 받아 변이가 적용된 텐서를 반환합니다.

        Args:
            chromosomes (torch.Tensor): 변이를 적용할 개체(들)의 텐서.
                                        Shape: (N, max_nodes, node_dim), where N >= 1.

        Returns:
            torch.Tensor: 변이가 적용된 개체(들)의 텐서.
        """
        pass