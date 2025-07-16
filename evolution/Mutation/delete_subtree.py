# evolution/Mutation/delete_subtree.py (최종 수정 완료)
import torch
import random
from .base import BaseMutation
from .utils import find_subtree_nodes
from typing import Dict, Any, Set

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

    def _get_protected_nodes(self, tree: torch.Tensor) -> Set[int]:
        """
        ROOT_BRANCH의 유일한 자식 노드를 찾아 "보호된 노드" 집합으로 반환합니다.
        이 노드들은 서브트리 삭제의 루트가 될 수 없습니다.
        """
        protected = set()
        # 모든 ROOT_BRANCH 노드를 찾습니다.
        root_branch_indices = (tree[:, COL_NODE_TYPE] == NODE_TYPE_ROOT_BRANCH).nonzero(as_tuple=True)[0]
        
        for rb_idx_tensor in root_branch_indices:
            rb_idx = rb_idx_tensor.item()
            # 각 ROOT_BRANCH의 자식들을 찾습니다.
            children_indices = (tree[:, COL_PARENT_IDX] == rb_idx).nonzero(as_tuple=True)[0]
            
            # 자식이 단 하나뿐이라면, 그 자식 노드를 보호 대상으로 추가합니다.
            if len(children_indices) == 1:
                protected.add(children_indices[0].item())
        return protected

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        mutated_chromosomes = chromosomes.clone()

        for i in range(mutated_chromosomes.shape[0]):
            if random.random() >= self.prob:
                continue

            tree = mutated_chromosomes[i]

            # --- 1. [수정] 보호된 노드를 제외한 삭제 후보 찾기 ---
            
            # 1a. [신규] 보호해야 할 노드 목록을 가져옵니다.
            protected_nodes = self._get_protected_nodes(tree)

            # 1b. 삭제 가능한 모든 잠재적 후보를 찾습니다 (ROOT_BRANCH 제외).
            active_mask = tree[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED
            not_root_branch_mask = tree[:, COL_NODE_TYPE] != NODE_TYPE_ROOT_BRANCH
            potential_candidates = (active_mask & not_root_branch_mask).nonzero(as_tuple=True)[0]
            
            if len(potential_candidates) == 0:
                continue

            # 1c. [신규] 잠재적 후보 중에서 보호된 노드를 필터링하여 최종 후보 목록을 만듭니다.
            final_candidates = [
                idx.item() for idx in potential_candidates if idx.item() not in protected_nodes
            ]

            if not final_candidates:
                continue
            
            candidate_indices = torch.tensor(final_candidates, dtype=torch.long, device=tree.device)

            # --- 2. 우선순위 기반 서브트리 선택 ---
            scores = torch.tensor([
                len(find_subtree_nodes(tree, idx.item())) for idx in candidate_indices
            ], device=tree.device, dtype=torch.float)
            
            if scores.sum() == 0:
                continue
                
            winner_position = torch.multinomial(scores, num_samples=1).item()
            subtree_root_idx = candidate_indices[winner_position].item()
            
            # --- 3. 서브트리 삭제 ---
            parent_idx = int(tree[subtree_root_idx, COL_PARENT_IDX].item())
            nodes_to_delete = find_subtree_nodes(tree, subtree_root_idx)
            
            if not nodes_to_delete:
                continue

            indices_tensor = torch.tensor(nodes_to_delete, dtype=torch.long, device=tree.device)
            tree[indices_tensor].zero_()
            tree[indices_tensor, COL_NODE_TYPE] = NODE_TYPE_UNUSED

            # --- 4. 후처리: 연쇄적 고아 노드 제거 ---
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