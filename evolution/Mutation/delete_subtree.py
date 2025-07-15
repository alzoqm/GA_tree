import torch
import random
from .base import BaseMutation
from .utils import find_subtree_nodes
from typing import Dict, Any

# model.py에서 상수 임포트
from models.model import (
    COL_NODE_TYPE, COL_PARENT_IDX, NODE_TYPE_UNUSED, NODE_TYPE_ROOT_BRANCH,
    NODE_TYPE_DECISION
)

class DeleteSubtreeMutation(BaseMutation):
    """
    트리에서 서브트리(가지) 전체를 삭제하는 변이. (구조 변경)
    삭제 후 '고아 결정 노드'가 연쇄적으로 발생하는 것을 방지하는
    '연쇄 가지치기(Cascading Pruning)' 후처리 로직을 포함합니다.
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None):
        super().__init__(prob)
        if config is None:
            # config는 이제 직접적으로 사용되지 않지만, 다른 모듈과의 일관성을 위해 유지합니다.
            self.config = {}
        else:
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

            # 3. 후처리: 연쇄적 고아 노드 제거 (Cascading Pruning)
            # 삭제된 서브트리의 부모로부터 시작하여 루트 방향으로 거슬러 올라가며,
            # 자식이 없는 Decision 노드를 연쇄적으로 정리합니다.
            current_node_idx = parent_idx
            while current_node_idx != -1: # 루트의 부모(-1)에 도달할 때까지 반복
                
                # 3-1. 현재 노드가 유효한 검사 대상인지 확인 (Decision 타입이어야 함)
                node_type = tree[current_node_idx, COL_NODE_TYPE].item()
                if node_type != NODE_TYPE_DECISION:
                    # 검사 대상이 Decision 노드가 아니면 연쇄 반응은 중단됨
                    break

                # 3-2. 현재 노드에 활성 자식이 있는지 확인
                children_mask = tree[:, COL_PARENT_IDX] == current_node_idx
                active_children_mask = children_mask & (tree[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED)
                has_active_children = active_children_mask.any().item()

                # 3-3. 고아 노드 판단 및 처리
                if not has_active_children:
                    # 자식이 없으므로 이 노드는 고아 노드임. 삭제 처리.
                    # 다음 검사 대상을 위해 부모 인덱스를 미리 저장
                    parent_of_current = int(tree[current_node_idx, COL_PARENT_IDX].item())
                    
                    # 현재 노드 삭제
                    tree[current_node_idx].zero_()
                    tree[current_node_idx, COL_NODE_TYPE] = NODE_TYPE_UNUSED
                    
                    # 다음 검사 대상을 이 노드의 부모로 설정하여 연쇄 반응을 이어감
                    current_node_idx = parent_of_current
                else:
                    # 활성 자식이 존재하므로 고아 노드가 아님. 연쇄 반응 중단.
                    break

        return mutated_chromosomes