# evolution/Mutation/delete_node.py (수정 완료)
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
    (수정) ROOT_BRANCH의 직계 자식도 삭제 가능하지만, ROOT_BRANCH가 고아가 되는 것은 방지합니다.
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
                # --- 1. [수정] 삭제 가능한 노드 후보 찾기 (ROOT_BRANCH 자식 포함) ---
                decision_mask = tree[:, COL_NODE_TYPE] == NODE_TYPE_DECISION
                candidate_indices_all = decision_mask.nonzero(as_tuple=True)[0]
                
                if len(candidate_indices_all) == 0:
                    break
                
                # --- 2. [신규] 유효성 검사 및 점수 계산 ---
                valid_candidates = []
                scores = []
                for node_idx_tensor in candidate_indices_all:
                    node_idx = node_idx_tensor.item()
                    parent_idx = int(tree[node_idx, COL_PARENT_IDX].item())

                    # --- [신규] 가드레일: ROOT_BRANCH가 고아가 되는 것을 방지 ---
                    # 삭제될 노드의 부모가 ROOT_BRANCH인 경우 특별 검사를 수행합니다.
                    if tree[parent_idx, COL_NODE_TYPE].item() == NODE_TYPE_ROOT_BRANCH:
                        # 삭제될 노드가 ROOT_BRANCH의 유일한 자식인지 확인
                        is_only_child = (tree[:, COL_PARENT_IDX] == parent_idx).sum().item() == 1
                        # 삭제될 노드 자신에게 자식이 없는지 확인
                        has_no_children = not (tree[:, COL_PARENT_IDX] == node_idx).any()
                        
                        # 유일한 자식이면서 자신도 자식이 없다면, 이 변이는 ROOT_BRANCH를 고아로 만듭니다.
                        # 이런 경우, 이 노드는 삭제 후보에서 제외합니다.
                        if is_only_child and has_no_children:
                            continue # 다음 후보로 넘어감
                    # --- [신규] 가드레일 끝 ---

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
                    continue 

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

        return mutated_chromosomes