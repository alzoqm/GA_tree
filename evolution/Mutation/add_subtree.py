# evolution/Mutation/add_subtree.py (수정된 최종본)
import torch
import random
from .base import BaseMutation
from .utils import find_empty_slots, create_random_node
from typing import Dict, Any, Tuple

from models.model import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, NODE_TYPE_DECISION, NODE_TYPE_ACTION
)

class AddSubtreeMutation(BaseMutation):
    """
    트리에 새로운 랜덤 서브트리(가지)를 추가하는 변이.
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
            
            # --- 유효한 부모 찾기 (이전과 동일) ---
            decision_mask = tree[:, COL_NODE_TYPE] == NODE_TYPE_DECISION
            depth_mask = tree[:, COL_DEPTH] < self.max_depth - 1
            
            candidate_parents = (decision_mask & depth_mask).nonzero(as_tuple=True)[0]
            
            valid_parents = []
            for p_idx_tensor in candidate_parents:
                p_idx = p_idx_tensor.item()
                children_indices = (tree[:, COL_PARENT_IDX] == p_idx).nonzero(as_tuple=True)[0]
                
                if len(children_indices) < self.max_children:
                    has_action_child = False
                    if len(children_indices) > 0:
                        if (tree[children_indices, COL_NODE_TYPE] == NODE_TYPE_ACTION).any():
                            has_action_child = True
                    if not has_action_child:
                        valid_parents.append(p_idx)

            if not valid_parents:
                continue

            parent_idx = random.choice(valid_parents)
            
            budget = random.randint(*self.node_count_range)
            empty_slots = find_empty_slots(tree, budget)
            if len(empty_slots) < budget:
                continue

            # --- 서브트리 성장 로직 (버그 수정) ---
            open_list = [parent_idx]
            nodes_created = 0
            
            while nodes_created < budget and open_list:
                current_parent_idx = random.choice(open_list)
                open_list.remove(current_parent_idx)

                parent_depth = int(tree[current_parent_idx, COL_DEPTH].item())
                
                # *** BUG FIX START ***
                # 1. 자식의 타입을 먼저 결정한다.
                # 깊이가 거의 다 찼으면 무조건 Action, 아니면 확률적으로 결정
                force_action = (parent_depth + 1 >= self.max_depth - 1)
                add_action_node = force_action or (random.random() < 0.5)

                if add_action_node:
                    # 2-A. 단일 Action 노드를 추가
                    new_node_idx = create_random_node(tree, current_parent_idx, NODE_TYPE_ACTION, self.config)
                    if new_node_idx != -1:
                        nodes_created += 1
                    # 이 부모는 터미널이므로 더 이상 성장하지 않음
                    # open_list에서 이미 제거되었으므로 추가 작업 불필요
                else:
                    # 2-B. Decision 노드 그룹을 추가
                    children_of_parent = (tree[:, COL_PARENT_IDX] == current_parent_idx).sum().item()
                    max_can_add = self.max_children - children_of_parent
                    
                    # 추가할 자식 수 결정 (남은 예산과 슬롯 내에서)
                    num_children_to_add = random.randint(1, max_can_add)
                    num_children_to_add = min(num_children_to_add, budget - nodes_created)
                    
                    for _ in range(num_children_to_add):
                        # 무조건 Decision 노드만 생성
                        new_node_idx = create_random_node(tree, current_parent_idx, NODE_TYPE_DECISION, self.config)
                        if new_node_idx != -1:
                            nodes_created += 1
                            open_list.append(new_node_idx) # 다음 성장을 위해 open_list에 추가
                        else: # 슬롯 부족
                            nodes_created = budget # 루프 강제 중단
                            break
                # *** BUG FIX END ***
        
        return mutated_chromosomes
