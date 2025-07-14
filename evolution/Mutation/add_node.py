# evolution/Mutation/add_node.py (수정 완료)
import torch
import random
from .base import BaseMutation
from .utils import find_empty_slots, update_subtree_depth, get_subtree_max_depth, _create_random_decision_params
from typing import Dict, Any

# model.py에서 상수 임포트
from models.model import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, NODE_TYPE_UNUSED, NODE_TYPE_ROOT_BRANCH,
    NODE_TYPE_DECISION, NODE_TYPE_ACTION
)


# 추가된 부모의 최대 자식 수를 확인해야 할듯
class AddNodeMutation(BaseMutation):
    """
    트리에 새로운 Decision 노드를 삽입하는 변이. (구조 변경)
    기존 부모-자식 연결(엣지) 사이에 새 노드를 추가합니다.
    한 번의 변이 호출 당, 1개에서 max_add_nodes개 사이의 노드를 무작위로 추가합니다.
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None, max_add_nodes: int = 5):
        """
        AddNodeMutation 초기화.

        Args:
            prob (float): 이 변이가 각 염색체에 적용될 확률.
            config (Dict[str, Any]): 최대 깊이 등 트리 제약조건이 포함된 설정 딕셔너리.
            max_add_nodes (int): 한 번의 변이에서 추가될 노드의 최대 개수.
        """
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
            # 확률에 따라 이 개체에 변이를 적용할지 결정
            if random.random() >= self.prob:
                continue

            tree = mutated_chromosomes[i]
            
            # 1. 이번 변이에서 추가할 노드의 개수를 랜덤하게 결정
            num_to_add = random.randint(1, self.max_add_nodes)
            nodes_added = 0
            
            # 2. 결정된 개수만큼 노드를 추가하기 위해 루프 실행
            # (무한 루프 방지를 위해 최대 시도 횟수를 제한)
            for _ in range(num_to_add * 5): # 시도 횟수에 여유를 줌
                if nodes_added >= num_to_add:
                    break # 목표한 개수만큼 노드를 추가했으면 종료

                # 3. 삽입 가능한 엣지(자식 노드) 찾기
                # 트리가 계속 변하므로, 매번 후보를 새로 찾아야 함
                active_mask = tree[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED
                parent_indices = tree[:, COL_PARENT_IDX].long()
                
                valid_parent_mask = parent_indices != -1
                parent_types = torch.zeros_like(tree[:, 0], dtype=torch.long)
                if valid_parent_mask.any():
                    parent_types[valid_parent_mask] = tree[parent_indices[valid_parent_mask], COL_NODE_TYPE].long()
                
                not_root_child_mask = parent_types != NODE_TYPE_ROOT_BRANCH
                
                candidate_indices = (active_mask & not_root_child_mask).nonzero(as_tuple=True)[0]
                
                if len(candidate_indices) == 0:
                    break # 더 이상 삽입할 유효한 위치가 없으면 종료

                # 4. 삽입 위치 및 유효성 검사
                child_idx = candidate_indices[torch.randint(len(candidate_indices), (1,))].item()
                parent_idx = int(tree[child_idx, COL_PARENT_IDX].item())
                
                # 유효성 검사 1: 빈 슬롯이 있는가?
                empty_slots = find_empty_slots(tree, 1)
                if not empty_slots:
                    break # 빈 슬롯이 없으면 더 이상 추가 불가
                new_node_idx = empty_slots[0]
                
                # 유효성 검사 2: 최대 깊이를 초과하지 않는가?
                # 새 노드가 들어갈 깊이 = 부모 깊이 + 1
                # 기존 자식의 서브트리 최대 깊이 = get_subtree_max_depth(tree, child_idx)
                # 새 노드가 추가되면 기존 자식의 서브트리 깊이가 1 증가하므로,
                # 최종적으로 예상되는 최대 깊이는 (부모 깊이 + 1) + (기존 자식의 상대적 깊이)
                if get_subtree_max_depth(tree, child_idx) + 1 > self.max_depth:
                    continue # 깊이 초과. 다른 후보를 찾아 재시도
                
                # 5. 변이 수행 (노드 삽입)
                # 5-1. 찾은 빈 슬롯(new_node_idx)에 새 Decision 노드 정보 직접 설정
                parent_depth = tree[parent_idx, COL_DEPTH].item()
                tree[new_node_idx, COL_NODE_TYPE] = NODE_TYPE_DECISION
                tree[new_node_idx, COL_PARENT_IDX] = parent_idx
                tree[new_node_idx, COL_DEPTH] = parent_depth + 1
                
                # 5-2. 새 노드의 파라미터 생성
                _create_random_decision_params(tree, new_node_idx, self.config)
                
                # 5-3. 기존 자식의 부모를 새 노드로 재연결
                tree[child_idx, COL_PARENT_IDX] = new_node_idx
                
                # 5-4. 기존 자식과 그 서브트리의 깊이 1 증가
                update_subtree_depth(tree, child_idx, 1)

                nodes_added += 1

        return mutated_chromosomes