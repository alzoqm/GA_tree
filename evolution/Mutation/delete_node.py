import torch
from .base import BaseMutation

class DeleteNodeMutation(BaseMutation):
    """트리에서 노드(와 그 서브트리)를 삭제하는 변이. (구조만 구현)"""
    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DeleteNodeMutation logic is not yet implemented.")