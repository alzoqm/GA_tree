# evolution/Mutation/delete_node.py (수정된 최종본)
import torch
import random
from .base import BaseMutation
from .utils import update_subtree_depth
from typing import Dict, Any

# model.py에서 상수 임포트
from models.model import (
    COL_NODE_TYPE, COL_PARENT_IDX, NODE_TYPE_UNUSED, NODE_TYPE_ROOT_BRANCH,
    NODE_TYPE_DECISION, NODE_TYPE_ACTION
)

class DeleteNodeMutation(BaseMutation):
    """
    중간 Decision 노드 하나를 제거하고 그 자식들을 조부모에게 연결(Splicing)하는 변이.
    (자식 타입 혼재 불가 제약조건을 검사하도록 수정됨)
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None):
        super().__init__(prob)
        if config is None:
            raise ValueError("DeleteNodeMutation requires a 'config' dictionary.")
        self.max_children = config['max_children']

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        mutated_chromosomes = chromosomes.clone()

        for i in range(mutated_chromosomes.shape[0]):
            if random.random() >= self.prob:
                continue

            tree = mutated_chromosomes[i]

            # 삭제 가능한 노드 찾기: Decision 타입이고, 루트 분기의 자식이 아닌 노드
            decision_mask = tree[:, COL_NODE_TYPE] == NODE_TYPE_DECISION
            
            parent_indices = tree[:, COL_PARENT_IDX].long()
            valid_parent_mask = parent_indices != -1
            parent_types = torch.zeros_like(tree[:, 0], dtype=torch.long)
            if valid_parent_mask.any():
                parent_types[valid_parent_mask] = tree[parent_indices[valid_parent_mask], COL_NODE_TYPE]
            
            not_root_child_mask = parent_types != NODE_TYPE_ROOT_BRANCH
            
            candidate_indices = (decision_mask & not_root_child_mask).nonzero(as_tuple=True)[0]
            
            if len(candidate_indices) == 0:
                continue

            node_to_delete_idx = candidate_indices[torch.randint(len(candidate_indices), (1,))].item()
            parent_idx = int(tree[node_to_delete_idx, COL_PARENT_IDX].item())
            
            children_of_deleted_indices = (tree[:, COL_PARENT_IDX] == node_to_delete_idx).nonzero(as_tuple=True)[0]
            
            # --- 제약 조건 검사 추가 (핵심 수정 사항) ---
            
            # 1. 자식 개수 검사
            children_of_parent_indices = (tree[:, COL_PARENT_IDX] == parent_idx).nonzero(as_tuple=True)[0]
            if (len(children_of_parent_indices) - 1 + len(children_of_deleted_indices)) > self.max_children:
                continue # 자식 수 초과, 변이 취소

            # 2. 자식 타입 혼재 검사
            # 삭제될 노드의 자식들 중 Action 노드가 있는지 확인
            has_action_child = False
            if len(children_of_deleted_indices) > 0:
                if (tree[children_of_deleted_indices, COL_NODE_TYPE] == NODE_TYPE_ACTION).any():
                    has_action_child = True
            
            # 만약 삭제될 노드가 Action 자식을 가지고 있다면, 부모의 다른 자식들은 없어야만 한다.
            # (즉, 부모의 자식이 삭제될 노드 하나뿐이었어야 함)
            if has_action_child and len(children_of_parent_indices) > 1:
                continue # Action 노드와 다른 Decision 노드가 섞이게 되므로 변이 취소
            
            # --- 변이 수행 ---
            # 자식들을 조부모에게 재연결
            if len(children_of_deleted_indices) > 0:
                tree[children_of_deleted_indices, COL_PARENT_IDX] = parent_idx
                # 자식들과 그 서브트리의 깊이 1 감소
                for child_idx_tensor in children_of_deleted_indices:
                    update_subtree_depth(tree, child_idx_tensor.item(), -1)

            # 노드 삭제
            tree[node_to_delete_idx].zero_()
            tree[node_to_delete_idx, COL_NODE_TYPE] = NODE_TYPE_UNUSED
        
        return mutated_chromosomes
