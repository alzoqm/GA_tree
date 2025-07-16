# evolution/Mutation/delete_subtree.py (수정됨)
import torch
import random
from .base import BaseMutation
from .utils import find_subtree_nodes
from typing import Dict, Any

# model.py에서 상수 임포트
from models.model import (
    COL_NODE_TYPE, COL_PARENT_IDX, NODE_TYPE_UNUSED, NODE_TYPE_ROOT_BRANCH,
    NODE_TYPE_DECISION
)

class DeleteSubtreeMutation(BaseMutation):
    """
    트리에서 서브트리를 삭제하는 변이.
    (수정) 노드 수가 많은 '복잡한' 서브트리를 우선적으로 삭제합니다.
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None):
        super().__init__(prob)
        self.config = config if config is not None else {}

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        mutated_chromosomes = chromosomes.clone()

        for i in range(mutated_chromosomes.shape[0]):
            if random.random() >= self.prob:
                continue

            tree = mutated_chromosomes[i]

            # --- 1. 삭제 가능한 서브트리의 루트 노드 후보 찾기 ---
            active_mask = tree[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED
            not_root_branch_mask = tree[:, COL_NODE_TYPE] != NODE_TYPE_ROOT_BRANCH
            candidate_indices = (active_mask & not_root_branch_mask).nonzero(as_tuple=True)[0]
            
            if len(candidate_indices) == 0:
                continue

            # --- 2. [신규] 우선순위 기반 서브트리 선택 ---
            # 점수 계산: 각 후보를 루트로 하는 서브트리의 노드 수
            scores = torch.tensor([
                len(find_subtree_nodes(tree, idx.item())) for idx in candidate_indices
            ], device=tree.device, dtype=torch.float)
            
            if scores.sum() == 0: # 모든 서브트리가 0 노드인 경우(이론상 거의 없음)
                continue
                
            # 가중치(scores)에 따라 삭제할 서브트리 루트를 확률적으로 선택
            winner_position = torch.multinomial(scores, num_samples=1).item()
            subtree_root_idx = candidate_indices[winner_position].item()
            # --- 우선순위 선택 로직 끝 ---
            
            parent_idx = int(tree[subtree_root_idx, COL_PARENT_IDX].item())

            # 3. 서브트리 삭제 (기존 로직과 동일)
            nodes_to_delete = find_subtree_nodes(tree, subtree_root_idx)
            if not nodes_to_delete:
                continue

            indices_tensor = torch.tensor(nodes_to_delete, dtype=torch.long, device=tree.device)
            tree[indices_tensor].zero_()
            tree[indices_tensor, COL_NODE_TYPE] = NODE_TYPE_UNUSED

            # 4. 후처리: 연쇄적 고아 노드 제거 (기존 로직과 동일)
            current_node_idx = parent_idx
            while current_node_idx != -1: 
                node_type = tree[current_node_idx, COL_NODE_TYPE].item()
                if node_type != NODE_TYPE_DECISION:
                    break

                children_mask = tree[:, COL_PARENT_IDX] == current_node_idx
                active_children_mask = children_mask & (tree[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED)
                
                if not active_children_mask.any().item():
                    parent_of_current = int(tree[current_node_idx, COL_PARENT_IDX].item())
                    tree[current_node_idx].zero_()
                    tree[current_node_idx, COL_NODE_TYPE] = NODE_TYPE_UNUSED
                    current_node_idx = parent_of_current
                else:
                    break

        return mutated_chromosomes