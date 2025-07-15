# evolution/Mutation/add_subtree.py (최종 수정본)
import torch
import random
from .base import BaseMutation
from .utils import find_empty_slots, create_random_node
from typing import Dict, Any, Tuple

# model.py에서 상수 임포트
from models.model import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, NODE_TYPE_UNUSED, NODE_TYPE_DECISION, NODE_TYPE_ACTION
)

class AddSubtreeMutation(BaseMutation):
    """
    트리에 새로운 랜덤 서브트리(가지)를 추가하는 변이.
    (최종 수정: '자식 타입 혼재' 및 '미종결 리프 노드' 문제 완벽 해결)
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
            
            # --- 1. 유효한 부모 찾기 ---
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
            
            # --- 2. 예산 및 슬롯 확인 ---
            budget = random.randint(*self.node_count_range)
            empty_slots = find_empty_slots(tree, budget)
            if len(empty_slots) < budget:
                continue

            # --- 3. 서브트리 성장 로직 (2단계 접근) ---
            # 1단계: 주 성장 루프
            open_list = [parent_idx]
            nodes_created = 0
            
            while nodes_created < budget and open_list:
                current_parent_idx = random.choice(open_list)
                open_list.remove(current_parent_idx)

                parent_depth = int(tree[current_parent_idx, COL_DEPTH].item())
                
                # 3-1. 자식 타입 '전략' 결정
                force_action = (parent_depth + 1 >= self.max_depth - 1)
                add_action_node = force_action or (random.random() < 0.5)

                # 3-2. (★★★ 최종 수정: 자격 검사 ★★★)
                # 전략을 실행하기 전, 부모의 현재 상태와 충돌하지 않는지 확인
                children_of_current_parent_mask = tree[:, COL_PARENT_IDX] == current_parent_idx
                num_existing_children = children_of_current_parent_mask.sum().item()

                if add_action_node:
                    # 'ACTION 추가' 전략의 자격 검사: 부모에게 자식이 없어야만 가능
                    if num_existing_children > 0:
                        # 자격 미달. 이 부모에 대한 시도를 포기하고 다음 루프로 넘어감.
                        # 이 부모는 open_list에 다시 추가하지 않으므로 자연스럽게 배제됨.
                        continue
                
                # 'DECISION 추가' 전략은 valid_parents 선정 시 이미 자격이 검증됨.

                # 3-3. 자격 검사를 통과한 후, 전략 실행
                if add_action_node:
                    new_node_idx = create_random_node(tree, current_parent_idx, NODE_TYPE_ACTION, self.config)
                    if new_node_idx != -1:
                        nodes_created += 1
                else:
                    max_can_add = self.max_children - num_existing_children
                    num_children_to_add = random.randint(1, max(1, max_can_add)) # max_can_add가 0이 될 수 있으므로 max(1,...)
                    num_children_to_add = min(num_children_to_add, budget - nodes_created)
                    
                    for _ in range(num_children_to_add):
                        new_node_idx = create_random_node(tree, current_parent_idx, NODE_TYPE_DECISION, self.config)
                        if new_node_idx != -1:
                            nodes_created += 1
                            open_list.append(new_node_idx)
                        else:
                            nodes_created = budget
                            break
            
            # 2단계: 필수 후처리 (미종결 리프 노드 방지)
            for dangling_parent_idx in open_list:
                children_mask = tree[:, COL_PARENT_IDX] == dangling_parent_idx
                if not children_mask.any():
                    create_random_node(tree, dangling_parent_idx, NODE_TYPE_ACTION, self.config)

        return mutated_chromosomes