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
    중간 Decision 노드를 무작위 개수(1 ~ max_delete_node)만큼 제거하고
    그 자식들을 조부모에게 연결(Splicing)하는 변이.
    (자식 타입 혼재 불가 제약조건을 검사하도록 수정됨)
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None, max_delete_node: int = 5):
        """
        Args:
            prob (float): 각 염색체에 이 변이가 적용될 확률.
            max_delete_node (int): 변이가 적용될 때 삭제할 최대 노드 수.
                                   실제 삭제되는 노드 수는 1과 이 값 사이에서 무작위로 결정됩니다.
            config (Dict[str, Any]): 모델 설정 딕셔너리. 'max_children' 키가 필요합니다.
        """
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
            
            # 이 염색체에서 삭제할 노드의 수를 무작위로 결정
            nodes_to_delete_count = random.randint(1, self.max_delete_node)

            for _ in range(nodes_to_delete_count):
                # --- 매 반복마다 삭제 가능한 노드를 다시 찾아야 함 ---
                # 트리가 매번 변하기 때문
                
                # 삭제 가능한 노드 찾기: Decision 타입이고, 루트 분기의 자식이 아닌 노드
                decision_mask = tree[:, COL_NODE_TYPE] == NODE_TYPE_DECISION
                
                parent_indices = tree[:, COL_PARENT_IDX].long()
                valid_parent_mask = parent_indices != -1
                parent_types = torch.zeros_like(tree[:, 0], dtype=torch.long)
                if valid_parent_mask.any():
                    parent_types[valid_parent_mask] = tree[parent_indices[valid_parent_mask], COL_NODE_TYPE].long()
                
                not_root_child_mask = parent_types != NODE_TYPE_ROOT_BRANCH
                
                candidate_indices = (decision_mask & not_root_child_mask).nonzero(as_tuple=True)[0]
                
                # 더 이상 삭제할 수 있는 노드가 없으면 중단
                if len(candidate_indices) == 0:
                    break

                # 삭제할 노드를 후보 중에서 무작위로 선택
                node_to_delete_idx = candidate_indices[torch.randint(len(candidate_indices), (1,))].item()
                parent_idx = int(tree[node_to_delete_idx, COL_PARENT_IDX].item())
                
                children_of_deleted_indices = (tree[:, COL_PARENT_IDX] == node_to_delete_idx).nonzero(as_tuple=True)[0]
                
                # --- 제약 조건 검사 ---
                
                # 1. 자식 개수 검사
                children_of_parent_indices = (tree[:, COL_PARENT_IDX] == parent_idx).nonzero(as_tuple=True)[0]
                if (len(children_of_parent_indices) - 1 + len(children_of_deleted_indices)) > self.max_children:
                    # 이번 삭제는 건너뛰고 다음 삭제 시도 (또는 루프 종료)
                    continue

                # 2. 자식 타입 혼재 검사
                has_action_child = False
                if len(children_of_deleted_indices) > 0:
                    if (tree[children_of_deleted_indices, COL_NODE_TYPE] == NODE_TYPE_ACTION).any():
                        has_action_child = True
                
                if has_action_child and len(children_of_parent_indices) > 1:
                    # 이번 삭제는 건너뛰고 다음 삭제 시도
                    continue
                
                # --- 변이 수행 ---
                # 자식들을 조부모에게 재연결
                if len(children_of_deleted_indices) > 0:
                    tree[children_of_deleted_indices, COL_PARENT_IDX] = parent_idx
                    # 자식들과 그 서브트리의 깊이 1 감소
                    for child_idx_tensor in children_of_deleted_indices:
                        update_subtree_depth(tree, child_idx_tensor.item(), -1)

                # 노드 삭제
                tree[node_to_delete_idx].zero_()
                tree[node_to_delete_idx, COL_NODE_TYPE] = NODE_TYPE_UNUSED
                print(f"Chromosome {i}: Deleted node {node_to_delete_idx}")

        return mutated_chromosomes