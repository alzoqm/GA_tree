# evolution/Mutation/delete_node.py (수정됨)
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
    중간 Decision 노드를 제거하고 자식들을 조부모에게 연결하는 변이.
    (수정) 자식 노드를 많이 가진 '복잡한 연결점'을 우선적으로 삭제합니다.
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None, max_delete_node: int = 5):
        super().__init__(prob)
        if config is None:
            raise ValueError("DeleteNodeMutation requires a 'config' dictionary.")
        if max_delete_node < 1:
            raise ValueError("max_delete_node must be at least 1.")
            
        self.max_children = config['max_children']
        self.max_delete_node = max_delete_node

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        mutated_chromosomes = chromosomes.clone()

        for i in range(mutated_chromosomes.shape[0]):
            if random.random() >= self.prob:
                continue

            tree = mutated_chromosomes[i]
            
            nodes_to_delete_count = random.randint(1, self.max_delete_node)

            for _ in range(nodes_to_delete_count):
                # --- 1. 삭제 가능한 노드 후보 찾기 ---
                decision_mask = tree[:, COL_NODE_TYPE] == NODE_TYPE_DECISION
                
                parent_indices = tree[:, COL_PARENT_IDX].long()
                valid_parent_mask = parent_indices != -1
                parent_types = torch.zeros_like(tree[:, 0], dtype=torch.long)
                if valid_parent_mask.any():
                    parent_types[valid_parent_mask] = tree[parent_indices[valid_parent_mask], COL_NODE_TYPE].long()
                
                not_root_child_mask = parent_types != NODE_TYPE_ROOT_BRANCH
                
                candidate_indices_all = (decision_mask & not_root_child_mask).nonzero(as_tuple=True)[0]
                
                if len(candidate_indices_all) == 0:
                    break
                
                # --- 2. [신규] 유효성 검사 및 점수 계산 ---
                valid_candidates = []
                scores = []
                for node_idx_tensor in candidate_indices_all:
                    node_idx = node_idx_tensor.item()
                    parent_idx = int(tree[node_idx, COL_PARENT_IDX].item())

                    children_of_deleted = (tree[:, COL_PARENT_IDX] == node_idx).nonzero(as_tuple=True)[0]
                    children_of_parent = (tree[:, COL_PARENT_IDX] == parent_idx).nonzero(as_tuple=True)[0]
                    
                    # 제약 조건 검사 1: 최대 자식 수 초과 여부
                    if (len(children_of_parent) - 1 + len(children_of_deleted)) > self.max_children:
                        continue
                    
                    # 제약 조건 검사 2: 자식 타입 혼재 여부
                    has_action_child = False
                    if len(children_of_deleted) > 0 and (tree[children_of_deleted, COL_NODE_TYPE] == NODE_TYPE_ACTION).any():
                        has_action_child = True
                    if has_action_child and len(children_of_parent) > 1:
                        continue

                    valid_candidates.append(node_idx)
                    # 점수 계산: 자식 수가 많을수록 높은 점수
                    scores.append(float(len(children_of_deleted)))

                if not valid_candidates:
                    continue # 유효한 삭제 후보가 없으면 다음 시도로

                # --- 3. [신규] 우선순위 기반 삭제 노드 선택 ---
                scores_tensor = torch.tensor(scores, device=tree.device, dtype=torch.float)
                scores_tensor += 1e-6 # 0점 방지
                winner_position = torch.multinomial(scores_tensor, num_samples=1).item()
                node_to_delete_idx = valid_candidates[winner_position]
                # --- 우선순위 선택 로직 끝 ---

                # --- 4. 변이 수행 ---
                parent_idx = int(tree[node_to_delete_idx, COL_PARENT_IDX].item())
                children_of_deleted_indices = (tree[:, COL_PARENT_IDX] == node_to_delete_idx).nonzero(as_tuple=True)[0]

                if len(children_of_deleted_indices) > 0:
                    tree[children_of_deleted_indices, COL_PARENT_IDX] = parent_idx
                    for child_idx_tensor in children_of_deleted_indices:
                        update_subtree_depth(tree, child_idx_tensor.item(), -1)

                tree[node_to_delete_idx].zero_()
                tree[node_to_delete_idx, COL_NODE_TYPE] = NODE_TYPE_UNUSED
                # print(f"Chromosome {i}: Deleted node {node_to_delete_idx} (Priority: High)")

        return mutated_chromosomes