import torch
from .base import BaseMutation

class AddNodeMutation(BaseMutation):
    """트리에 새로운 노드를 추가하는 변이. (구조만 구현)"""
    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("AddNodeMutation logic is not yet implemented.")