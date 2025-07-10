# evolution/Mutation/add_node.py
import torch
import random
from .base import BaseMutation
from .utils import find_empty_slots, update_subtree_depth, get_subtree_max_depth
from typing import Dict, Any

# model.py에서 상수 임포트
from models.model import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, NODE_TYPE_UNUSED, NODE_TYPE_ROOT_BRANCH,
    NODE_TYPE_DECISION, NODE_TYPE_ACTION
)

class AddNodeMutation(BaseMutation):
    """
    트리에 새로운 Decision 노드를 삽입하는 변이. (구조 변경)
    기존 부모-자식 연결(엣지) 사이에 새 노드를 추가합니다.
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None):
        super().__init__(prob)
        if config is None:
            raise ValueError("AddNodeMutation requires a 'config' dictionary.")
        self.config = config
        self.max_depth = config['max_depth']

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        mutated_chromosomes = chromosomes.clone()
        
        for i in range(mutated_chromosomes.shape[0]):
            if random.random() >= self.prob:
                continue

            tree = mutated_chromosomes[i]
            
            # 삽입 가능한 엣지(자식 노드) 찾기
            # 루트 분기의 직접적인 자식이 아닌 모든 활성 노드
            active_mask = tree[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED
            parent_indices = tree[:, COL_PARENT_IDX].long()
            # parent가 -1이 아닌 노드만 선택
            valid_parent_mask = parent_indices != -1
            parent_types = torch.zeros_like(tree[:, 0], dtype=torch.long)
            parent_types[valid_parent_mask] = tree[parent_indices[valid_parent_mask], COL_NODE_TYPE]
            
            not_root_child_mask = parent_types != NODE_TYPE_ROOT_BRANCH
            
            candidate_indices = (active_mask & not_root_child_mask).nonzero(as_tuple=True)[0]
            
            if len(candidate_indices) == 0:
                continue

            # 후보 중 하나를 무작위로 선택
            child_idx = candidate_indices[torch.randint(len(candidate_indices), (1,))].item()
            parent_idx = int(tree[child_idx, COL_PARENT_IDX].item())
            
            # 유효성 검사 1: 빈 슬롯이 있는가?
            empty_slots = find_empty_slots(tree, 1)
            if not empty_slots:
                continue
            new_node_idx = empty_slots[0]
            
            # 유효성 검사 2: 최대 깊이를 초과하지 않는가?
            # 새 노드는 parent_depth+1에 위치, 기존 child와 그 서브트리는 깊이가 1 증가함
            # 따라서 child 서브트리의 최대 깊이가 max_depth를 넘으면 안됨
            if get_subtree_max_depth(tree, child_idx) + 1 > self.max_depth:
                continue

            # 변이 수행
            # 1. 새 Decision 노드 생성
            from .utils import create_random_node # 순환참조 방지
            create_random_node(tree, parent_idx, NODE_TYPE_DECISION, self.config)
            
            # 2. 기존 자식의 부모를 새 노드로 재연결
            tree[child_idx, COL_PARENT_IDX] = new_node_idx
            
            # 3. 기존 자식과 그 서브트리의 깊이 1 증가
            update_subtree_depth(tree, child_idx, 1)

        return mutated_chromosomes
