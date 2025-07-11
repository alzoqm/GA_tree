# evolution/Crossover/root_branch.py
import torch
import random
from .base import BaseCrossover
from ..Mutation.utils import find_subtree_nodes, find_empty_slots
from typing import Tuple

from models.model import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, COL_PARAM_1, NODE_INFO_DIM,
    NODE_TYPE_UNUSED, NODE_TYPE_ROOT_BRANCH, ROOT_BRANCH_LONG,
    ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT
)

class RootBranchCrossover(BaseCrossover):
    """
    부모들의 루트 분기(LONG/HOLD/SHORT)를 재조합하여 새로운 자식을 생성합니다.
    예: 자식의 LONG 분기는 p1에서, SHORT 분기는 p2에서 가져옵니다.
    """
    def __init__(self, rate: float = 0.8, max_nodes: int = 100):
        super().__init__(rate)
        self.max_nodes = max_nodes
        self.BRANCH_MAP = {
            ROOT_BRANCH_LONG: 0,
            ROOT_BRANCH_HOLD: 1,
            ROOT_BRANCH_SHORT: 2
        }

    def __call__(self, parents: torch.Tensor) -> torch.Tensor:
        num_offspring = parents.shape[0] // 2
        max_nodes, node_dim = parents.shape[1], parents.shape[2]
        
        children = torch.empty((num_offspring, max_nodes, node_dim), 
                               dtype=parents.dtype, device=parents.device)
        
        parent_pairs = parents.view(num_offspring, 2, max_nodes, node_dim)

        for i, (p1, p2) in enumerate(parent_pairs):
            if torch.rand(1).item() < self.rate:
                children[i] = self._perform_crossover_pair(p1, p2)
            else:
                children[i] = p1.clone() if torch.rand(1).item() < 0.5 else p2.clone()
        
        return children

    def _perform_crossover_pair(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """두 부모의 루트 분기를 재조합하여 하나의 자식을 생성합니다."""
        child = torch.zeros_like(p1)
        
        # 1. 자식 트리에 3개의 루트 분기 노드 생성
        child_next_idx = 0
        for branch_type, branch_idx in self.BRANCH_MAP.items():
            child[branch_idx, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
            child[branch_idx, COL_PARENT_IDX] = -1
            child[branch_idx, COL_DEPTH] = 0
            child[branch_idx, COL_PARAM_1] = branch_type
            child_next_idx += 1

        # 2. 각 분기별로 기증 부모를 선택하고 서브트리 복사
        for branch_type, dest_parent_idx in self.BRANCH_MAP.items():
            donor = p1 if random.random() < 0.5 else p2
            source_parent_idx = self.BRANCH_MAP[branch_type]

            # 기증 부모의 해당 분기 바로 아래 자식들을 찾음
            source_children_indices = (donor[:, COL_PARENT_IDX] == source_parent_idx).nonzero(as_tuple=True)[0]
            
            for source_child_idx in source_children_indices:
                # 복사 전 유효성 검사 (max_nodes 초과 여부)
                subtree_nodes = find_subtree_nodes(donor, source_child_idx.item())
                if child_next_idx + len(subtree_nodes) > self.max_nodes:
                    continue # 공간 부족, 이 서브트리 복사 포기
                
                # 서브트리 복사 실행
                copied_nodes_count = self._copy_subtree(child, child_next_idx, donor, source_child_idx.item(), dest_parent_idx)
                child_next_idx += copied_nodes_count
        
        return child

    def _copy_subtree(self, dest_tree, dest_start_idx, source_tree, source_root_idx, dest_parent_idx):
        """
        source_tree의 서브트리를 dest_tree의 빈 공간으로 복사합니다.
        _transplant_one_way와 유사하지만, 빈 공간에 새로 쓰는 방식입니다.
        """
        nodes_to_copy = find_subtree_nodes(source_tree, source_root_idx)
        if not nodes_to_copy:
            return 0
        
        indices_to_copy = torch.tensor(nodes_to_copy, dtype=torch.long, device=source_tree.device)
        
        new_slots = torch.arange(dest_start_idx, dest_start_idx + len(nodes_to_copy))
        
        old_to_new_map = {old: new for old, new in zip(nodes_to_copy, new_slots.tolist())}
        
        # 깊이 차이 계산
        dest_parent_depth = dest_tree[dest_parent_idx, COL_DEPTH].item()
        source_root_depth = source_tree[source_root_idx, COL_DEPTH].item()
        depth_offset = (dest_parent_depth + 1) - source_root_depth

        # 노드 데이터 복사 및 정보 업데이트
        dest_tree[new_slots] = source_tree[indices_to_copy]
        dest_tree[new_slots, COL_DEPTH] += depth_offset

        # 부모 인덱스 재연결
        for old_idx, new_idx in old_to_new_map.items():
            old_parent_idx = int(source_tree[old_idx, COL_PARENT_IDX].item())
            if old_idx == source_root_idx:
                dest_tree[new_idx, COL_PARENT_IDX] = dest_parent_idx
            else:
                dest_tree[new_idx, COL_PARENT_IDX] = old_to_new_map[old_parent_idx]
        
        return len(nodes_to_copy)