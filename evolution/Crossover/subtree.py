# evolution/Crossover/subtree.py
import torch
from .base import BaseCrossover

class SubtreeCrossover(BaseCrossover):
    """
    두 트리의 서브트리(가지)를 교환하는 교차 연산자.
    (세부 로직은 추후 구현 예정)
    """
    def __call__(self, parents: torch.Tensor) -> torch.Tensor:
        """
        서브트리 교차를 수행합니다.
        """
        raise NotImplementedError("SubtreeCrossover logic is not yet implemented.")