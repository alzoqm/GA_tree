# evolution/Crossover/node.py
import torch
import random
from .base import BaseCrossover
from typing import Dict, Tuple

# 프로젝트 구조에 따라 model.py에서 상수 임포트
from models.constants import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_PARAM_1, NODE_TYPE_UNUSED, 
    NODE_TYPE_ROOT_BRANCH,  # <--- [수정 1] 누락된 상수 추가
    NODE_TYPE_DECISION, NODE_TYPE_ACTION, ROOT_BRANCH_LONG, 
    ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT
)

class NodeCrossover(BaseCrossover):
    """
    한 부모의 구조는 유지한 채, 다른 부모의 노드 파라미터를 이식하는 교차 연산자.
    'free' 모드: 노드 타입만 맞으면 교환.
    'context' 모드: 루트 분기 및 노드 타입이 모두 맞아야 교환.
    """
    def __init__(self, rate: float = 0.8, mode: str = 'free'):
        super().__init__(rate)
        if mode not in ['free', 'context']:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'free' or 'context'.")
        self.mode = mode

        # GATree 텐서 구조 상수
        self.COL_NODE_TYPE = COL_NODE_TYPE
        self.COL_PARENT_IDX = COL_PARENT_IDX
        self.COL_PARAM_1 = COL_PARAM_1
        self.NODE_TYPE_DECISION = NODE_TYPE_DECISION
        self.NODE_TYPE_ACTION = NODE_TYPE_ACTION
        self.BRANCH_TYPES = [ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT]

    def __call__(self, parents: torch.Tensor) -> torch.Tensor:
        num_offspring = parents.shape[0] // 2
        max_nodes, node_dim = parents.shape[1], parents.shape[2]
        
        children = torch.empty((num_offspring, max_nodes, node_dim), 
                               dtype=parents.dtype, device=parents.device)
        
        parent_pairs = parents.view(num_offspring, 2, max_nodes, node_dim)

        for i, (p1, p2) in enumerate(parent_pairs):
            if torch.rand(1).item() < self.rate:
                child1, child2 = self._perform_crossover_pair(p1, p2)
                children[i] = child1 if torch.rand(1).item() < 0.5 else child2
            else:
                children[i] = p1.clone() if torch.rand(1).item() < 0.5 else p2.clone()
        
        return children

    def _perform_crossover_pair(self, p1: torch.Tensor, p2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """두 부모에 대해 노드 파라미터 교환을 수행합니다."""
        child1, child2 = p1.clone(), p2.clone()

        for node_type in [self.NODE_TYPE_DECISION, self.NODE_TYPE_ACTION]:
            if self.mode == 'free':
                p1_mask = (p1[:, self.COL_NODE_TYPE] == node_type)
                p2_mask = (p2[:, self.COL_NODE_TYPE] == node_type)
                self._swap_node_params(child1, child2, p1_mask, p2_mask)
            else: # context mode
                for branch_type in self.BRANCH_TYPES:
                    p1_context_mask = self._get_contextual_mask(p1, node_type, branch_type)
                    p2_context_mask = self._get_contextual_mask(p2, node_type, branch_type)
                    if p1_context_mask.any() and p2_context_mask.any():
                        self._swap_node_params(child1, child2, p1_context_mask, p2_context_mask)

        return child1, child2

    def _swap_node_params(self, c1: torch.Tensor, c2: torch.Tensor, 
                          p1_mask: torch.Tensor, p2_mask: torch.Tensor):
        """주어진 마스크에 해당하는 노드들의 파라미터를 교환합니다."""
        p1_indices = p1_mask.nonzero(as_tuple=True)[0]
        p2_indices = p2_mask.nonzero(as_tuple=True)[0]

        if p1_indices.numel() == 0 or p2_indices.numel() == 0:
            return

        max_swaps = min(p1_indices.numel(), p2_indices.numel())
        if max_swaps == 0: return
        
        # 최소 1개, 최대 절반까지 교환하도록 수정 (k가 1이 되는 것을 보장)
        k_upper_bound = max(1, max_swaps // 2) 
        k = torch.randint(1, k_upper_bound + 1, (1,)).item()
        k = min(k, max_swaps)

        p1_swap_indices = p1_indices[torch.randperm(p1_indices.numel())[:k]]
        p2_swap_indices = p2_indices[torch.randperm(p2_indices.numel())[:k]]

        p1_params = c1[p1_swap_indices, self.COL_PARAM_1:].clone()
        p2_params = c2[p2_swap_indices, self.COL_PARAM_1:].clone()
        
        c1[p1_swap_indices, self.COL_PARAM_1:] = p2_params
        c2[p2_swap_indices, self.COL_PARAM_1:] = p1_params


    def _get_contextual_mask(self, tree: torch.Tensor, node_type: int, branch_type: int) -> torch.Tensor:
        """특정 루트 분기와 노드 타입에 해당하는 노드 마스크를 반환합니다."""
        type_mask = (tree[:, self.COL_NODE_TYPE] == node_type)
        if not type_mask.any():
            return torch.zeros_like(type_mask)
        
        candidate_indices = type_mask.nonzero(as_tuple=True)[0]
        
        context_indices = [
            idx.item() for idx in candidate_indices
            if self._get_root_branch_type(tree, idx.item()) == branch_type
        ]

        if not context_indices:
            return torch.zeros_like(type_mask)

        final_mask = torch.zeros_like(type_mask)
        final_mask[torch.tensor(context_indices, device=tree.device)] = True
        return final_mask

    def _get_root_branch_type(self, tree: torch.Tensor, node_idx: int) -> int:
        # <--- [수정 2] 함수 로직을 더 간결하고 명확하게 개선
        """주어진 노드의 최상위 조상(루트 분기)의 타입을 반환합니다."""
        current_idx = node_idx
        # 부모 인덱스가 -1이 될 때까지 (즉, 최상위 노드에 도달할 때까지) 거슬러 올라감
        while int(tree[current_idx, self.COL_PARENT_IDX].item()) != -1:
            current_idx = int(tree[current_idx, self.COL_PARENT_IDX].item())
        
        # 이제 current_idx는 루트 분기 노드를 가리키므로, 그 타입을 반환
        return int(tree[current_idx, self.COL_PARAM_1].item())