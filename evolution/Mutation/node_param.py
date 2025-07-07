import torch
from .base import BaseMutation

class NodeParamMutation(BaseMutation):
    """노드의 파라미터를 변경하는 변이. (구조만 구현)"""
    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("NodeParamMutation logic is not yet implemented.")