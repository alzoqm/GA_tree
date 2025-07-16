# evolution/Mutation/add_subtree.py (수정됨)
import torch
import random
from .base import BaseMutation
from .utils import find_empty_slots, create_random_node
from typing import Dict, Any, Tuple

# model.py에서 상수 임포트
from models.model import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, NODE_TYPE_UNUSED, NODE_TYPE_DECISION, 
    NODE_TYPE_ACTION, NODE_TYPE_ROOT_BRANCH
)

class AddSubtreeMutation(BaseMutation):
    """
    트리에 새로운 랜덤 서브트리를 추가하는 변이.
    (수정) 깊이가 얕고 자식 추가 공간이 많은 '덜 발달한' 위치를 우선적으로 선택합니다.
    (수정) ROOT_BRANCH 노드 아래에도 직접 추가할 수 있도록 허용합니다.
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None,
                 node_count_range: Tuple[int, int] = (2, 5)):
        super().__init__(prob)
        if config is None:
            raise ValueError("AddSubtreeMutation requires a 'config' dictionary.")
        self.config = config
        self.max_depth = config['max_depth']
        self.max_children = config['max_children']
        self.node_count_range = node_count_range

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        mutated_chromosomes = chromosomes.clone()

        for i in range(mutated_chromosomes.shape[0]):
            if random.random() >= self.prob:
                continue
                
            tree = mutated_chromosomes[i]
            
            # --- 1. 유효한 부모 찾기 및 점수 계산 ---
            # [변경] DECISION 노드 또는 ROOT_BRANCH 노드를 부모 후보로 선택
            parent_type_mask = (tree[:, COL_NODE_TYPE] == NODE_TYPE_DECISION) | (tree[:, COL_NODE_TYPE] == NODE_TYPE_ROOT_BRANCH)
            depth_mask = tree[:, COL_DEPTH] < self.max_depth - 1
            
            candidate_parents_indices = (parent_type_mask & depth_mask).nonzero(as_tuple=True)[0]
            
            valid_parents = []
            scores = []
            for p_idx_tensor in candidate_parents_indices:
                p_idx = p_idx_tensor.item()
                children_indices = (tree[:, COL_PARENT_IDX] == p_idx).nonzero(as_tuple=True)[0]
                num_children = len(children_indices)
                
                if num_children < self.max_children:
                    # [신규] 자식으로 Action 노드를 가지는지 확인 (Action 노드를 가지면 Decision 추가 불가)
                    if num_children > 0 and (tree[children_indices, COL_NODE_TYPE] == NODE_TYPE_ACTION).any():
                        continue
                    
                    valid_parents.append(p_idx)
                    # [신규] 점수 계산 로직
                    # 점수 = (남은 자식 슬롯 수) / (깊이 + 1)
                    # -> 자식 추가 공간이 많고, 깊이가 얕을수록 높은 점수를 받음
                    parent_depth = tree[p_idx, COL_DEPTH].item()
                    score = (self.max_children - num_children) / (parent_depth + 1.0)
                    scores.append(score)

            if not valid_parents:
                continue

            # --- 2. [신규] 우선순위 기반 부모 선택 ---
            scores_tensor = torch.tensor(scores, device=tree.device, dtype=torch.float)
            scores_tensor += 1e-6 # 0점 방지
            winner_position = torch.multinomial(scores_tensor, num_samples=1).item()
            parent_idx = valid_parents[winner_position]
            
            # --- 3. 예산 및 슬롯 확인 ---
            budget = random.randint(*self.node_count_range)
            empty_slots = find_empty_slots(tree, budget)
            if len(empty_slots) < budget:
                continue

            # --- 4. 서브트리 성장 로직 (기존과 동일) ---
            open_list = [parent_idx]
            nodes_created = 0
            
            while nodes_created < budget and open_list:
                current_parent_idx = random.choice(open_list)
                open_list.remove(current_parent_idx)

                parent_depth = int(tree[current_parent_idx, COL_DEPTH].item())
                
                force_action = (parent_depth + 1 >= self.max_depth - 1)
                add_action_node = force_action or (random.random() < 0.5)

                children_of_current_parent_mask = tree[:, COL_PARENT_IDX] == current_parent_idx
                num_existing_children = children_of_current_parent_mask.sum().item()

                if add_action_node:
                    if num_existing_children > 0:
                        continue
                
                if add_action_node:
                    new_node_idx = create_random_node(tree, current_parent_idx, NODE_TYPE_ACTION, self.config)
                    if new_node_idx != -1:
                        nodes_created += 1
                else:
                    max_can_add = self.max_children - num_existing_children
                    num_children_to_add = random.randint(1, max(1, max_can_add))
                    num_children_to_add = min(num_children_to_add, budget - nodes_created)
                    
                    for _ in range(num_children_to_add):
                        new_node_idx = create_random_node(tree, current_parent_idx, NODE_TYPE_DECISION, self.config)
                        if new_node_idx != -1:
                            nodes_created += 1
                            open_list.append(new_node_idx)
                        else:
                            nodes_created = budget
                            break
            
            for dangling_parent_idx in open_list:
                children_mask = tree[:, COL_PARENT_IDX] == dangling_parent_idx
                if not children_mask.any():
                    create_random_node(tree, dangling_parent_idx, NODE_TYPE_ACTION, self.config)

        return mutated_chromosomes