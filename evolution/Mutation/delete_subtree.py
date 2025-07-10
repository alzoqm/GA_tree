# evolution/Mutation/delete_subtree.py
import torch
import random
from .base import BaseMutation
from .utils import find_subtree_nodes, create_random_node
from typing import Dict, Any

# model.py에서 상수 임포트
from models.model import (
    COL_NODE_TYPE, COL_PARENT_IDX, NODE_TYPE_UNUSED, NODE_TYPE_ROOT_BRANCH,
    NODE_TYPE_DECISION, NODE_TYPE_ACTION
)

class DeleteSubtreeMutation(BaseMutation):
    """
    트리에서 서브트리(가지) 전체를 삭제하는 변이. (구조 변경)
    삭제 후 논리적 완결성을 보장하기 위한 후처리 로직을 포함합니다.
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None):
        super().__init__(prob)
        if config is None:
            raise ValueError("DeleteSubtreeMutation requires a 'config' dictionary for post-processing.")
        self.config = config

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        mutated_chromosomes = chromosomes.clone()

        for i in range(mutated_chromosomes.shape[0]):
            if random.random() >= self.prob:
                continue

            tree = mutated_chromosomes[i]

            # 삭제 가능한 서브트리의 루트 노드 찾기
            # (루트 분기가 아닌 모든 활성 노드)
            active_mask = tree[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED
            not_root_branch_mask = tree[:, COL_NODE_TYPE] != NODE_TYPE_ROOT_BRANCH
            
            candidate_indices = (active_mask & not_root_branch_mask).nonzero(as_tuple=True)[0]
            
            if len(candidate_indices) == 0:
                continue

            # 후보 중 하나를 무작위로 선택하여 서브트리의 루트로 지정
            subtree_root_idx = candidate_indices[torch.randint(len(candidate_indices), (1,))].item()
            
            # 후처리를 위해 부모 노드 정보 저장
            parent_idx = int(tree[subtree_root_idx, COL_PARENT_IDX].item())

            # 1. 삭제할 서브트리의 모든 노드 찾기
            nodes_to_delete = find_subtree_nodes(tree, subtree_root_idx)
            if not nodes_to_delete:
                continue # 서브트리가 비어있는 경우 (이론상 발생 안함)

            # 2. 서브트리 삭제 (관련 노드들을 UNUSED로 변경)
            indices_tensor = torch.tensor(nodes_to_delete, dtype=torch.long, device=tree.device)
            tree[indices_tensor].zero_()
            tree[indices_tensor, COL_NODE_TYPE] = NODE_TYPE_UNUSED

            # 3. 후처리: "고아 부모" 문제 해결
            # 삭제된 서브트리의 부모가 Decision 노드이고, 이 삭제로 인해 자식이 모두 사라졌다면,
            # 트리의 논리적 완결성을 위해 새로운 랜덤 Action 노드를 자식으로 추가해준다.
            if parent_idx != -1:
                parent_type = tree[parent_idx, COL_NODE_TYPE].item()
                children_of_parent = (tree[:, COL_PARENT_IDX] == parent_idx).sum().item()

                if parent_type == NODE_TYPE_DECISION and children_of_parent == 0:
                    # 새로운 Action 노드를 추가하여 경로를 완성
                    create_random_node(tree, parent_idx, NODE_TYPE_ACTION, self.config)

        return mutated_chromosomes
