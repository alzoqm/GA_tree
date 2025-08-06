# evolution/Mutation/add_node.py (수정됨)
import torch
import random
from .base import BaseMutation
from .utils import find_empty_slots, update_subtree_depth, get_subtree_max_depth, _create_random_decision_params
from typing import Dict, Any

# model.py에서 상수 임포트
from models.constants import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, NODE_TYPE_UNUSED, NODE_TYPE_ROOT_BRANCH,
    NODE_TYPE_DECISION
)

class AddNodeMutation(BaseMutation):
    """
    트리에 새로운 Decision 노드를 삽입하는 변이. (구조 변경)
    (수정) 깊이가 얕은 '단순한 경로'에 우선적으로 노드를 추가합니다.
    (수정) ROOT_BRANCH 바로 아래에도 노드를 삽입할 수 있도록 허용합니다.
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None, max_add_nodes: int = 5):
        super().__init__(prob)
        if config is None:
            raise ValueError("AddNodeMutation requires a 'config' dictionary.")
        if not isinstance(max_add_nodes, int) or max_add_nodes < 1:
            raise ValueError("max_add_nodes must be a positive integer.")
            
        self.config = config
        self.max_depth = config['max_depth']
        self.max_add_nodes = max_add_nodes

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        mutated_chromosomes = chromosomes.clone()
        
        for i in range(mutated_chromosomes.shape[0]):
            if random.random() >= self.prob:
                continue

            tree = mutated_chromosomes[i]
            
            num_to_add = random.randint(1, self.max_add_nodes)
            nodes_added = 0
            
            # 여러 번 시도하여 가능한 많이 추가
            for _ in range(num_to_add * 5): 
                if nodes_added >= num_to_add:
                    break

                # --- 1. 삽입 가능한 엣지(자식 노드) 찾기 ---
                active_mask = tree[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED
                
                # [변경] Root Branch 노드 자체는 자식이 될 수 없으므로 제외
                is_not_root_branch_node_mask = tree[:, COL_NODE_TYPE] != NODE_TYPE_ROOT_BRANCH
                
                # [변경] 부모가 있는 모든 노드가 대상이 됨 (기존 not_root_child_mask 제거)
                has_parent_mask = tree[:, COL_PARENT_IDX] != -1
                
                candidate_indices = (active_mask & is_not_root_branch_node_mask & has_parent_mask).nonzero(as_tuple=True)[0]
                
                if len(candidate_indices) == 0:
                    break 

                # --- 2. [신규] 우선순위 기반 위치 선택 ---
                valid_candidates = []
                scores = []
                
                for child_idx_tensor in candidate_indices:
                    child_idx = child_idx_tensor.item()
                    
                    # 유효성 검사: 최대 깊이 초과 여부
                    if get_subtree_max_depth(tree, child_idx) + 1 >= self.max_depth:
                        continue
                    
                    valid_candidates.append(child_idx)
                    
                    # 점수 계산: 깊이가 얕을수록 높은 점수 부여 (1 / (depth + 1))
                    parent_idx = int(tree[child_idx, COL_PARENT_IDX].item())
                    parent_depth = tree[parent_idx, COL_DEPTH].item()
                    scores.append(1.0 / (parent_depth + 1))

                if not valid_candidates:
                    continue # 더 이상 유효한 후보가 없으면 종료
                
                # 가중치(scores)에 따라 삽입할 위치(child_idx)를 확률적으로 선택
                scores_tensor = torch.tensor(scores, device=tree.device, dtype=torch.float)
                winner_position = torch.multinomial(scores_tensor, num_samples=1).item()
                child_idx = valid_candidates[winner_position]
                parent_idx = int(tree[child_idx, COL_PARENT_IDX].item())
                # --- 우선순위 선택 로직 끝 ---

                # 3. 유효성 검사 및 변이 수행
                empty_slots = find_empty_slots(tree, 1)
                if not empty_slots:
                    break 
                new_node_idx = empty_slots[0]
                
                parent_depth = tree[parent_idx, COL_DEPTH].item()
                tree[new_node_idx, COL_NODE_TYPE] = NODE_TYPE_DECISION
                tree[new_node_idx, COL_PARENT_IDX] = parent_idx
                tree[new_node_idx, COL_DEPTH] = parent_depth + 1
                
                _create_random_decision_params(tree, new_node_idx, self.config)
                
                tree[child_idx, COL_PARENT_IDX] = new_node_idx
                
                update_subtree_depth(tree, child_idx, 1)

                nodes_added += 1

        return mutated_chromosomes