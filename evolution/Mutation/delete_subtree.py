# evolution/Mutation/delete_subtree.py (수정 완료)
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
    (수정) ROOT_BRANCH의 마지막 남은 자식 서브트리는 삭제하지 않도록 보호합니다.
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
            scores = torch.tensor([
                len(find_subtree_nodes(tree, idx.item())) for idx in candidate_indices
            ], device=tree.device, dtype=torch.float)
            
            if scores.sum() == 0:
                continue
                
            winner_position = torch.multinomial(scores, num_samples=1).item()
            subtree_root_idx = candidate_indices[winner_position].item()
            # --- 우선순위 선택 로직 끝 ---
            
            # --- [신규] 가드레일: ROOT_BRANCH가 고아가 되는 것을 방지 ---
            parent_idx = int(tree[subtree_root_idx, COL_PARENT_IDX].item())
            if parent_idx != -1 and tree[parent_idx, COL_NODE_TYPE].item() == NODE_TYPE_ROOT_BRANCH:
                # 만약 삭제하려는 서브트리의 부모가 ROOT_BRANCH이고, 
                # 그 ROOT_BRANCH의 자식이 단 하나뿐이라면(즉, 삭제될 서브트리뿐이라면)
                # 이 변이는 해당 분기 자체를 없애므로 건너뜁니다.
                if (tree[:, COL_PARENT_IDX] == parent_idx).sum().item() == 1:
                    continue # 현재 개체에 대한 변이를 건너뛰고 다음 개체로 넘어감
            # --- [신규] 가드레일 끝 ---

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