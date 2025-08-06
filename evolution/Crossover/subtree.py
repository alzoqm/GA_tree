# evolution/Crossover/subtree.py
import torch
import random
from .base import BaseCrossover
from typing import Dict, Tuple, Optional, List

# 프로젝트 구조에 따라 model.py에서 상수 임포트
from models.constants import (
    ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT
)

class SubtreeCrossover(BaseCrossover):
    """
    두 트리의 서브트리(가지)를 교환하는 교차 연산자.
    '자유 교차'와 '문맥-인식 교차' 두 가지 모드를 지원합니다.
    - free: 포지션 분기(LONG/HOLD/SHORT)에 관계없이 서브트리를 교환. (탐색 극대화)
    - context: 동일한 포지션 분기에 속한 서브트리끼리만 교환. (활용 강화)
    """
    def __init__(self, 
                 rate: float = 0.8, 
                 max_nodes: int = 100, 
                 max_depth: int = 3, 
                 max_retries: int = 5,
                 mode: str = 'free'):
        """
        SubtreeCrossover 초기화.

        Args:
            rate (float): 교차가 일어날 확률.
            max_nodes (int): 트리가 가질 수 있는 최대 노드 수.
            max_depth (int): 트리의 최대 깊이.
            max_retries (int): 유효한 교차점을 찾기 위한 최대 재시도 횟수.
            mode (str): 교차 방식. 'free' 또는 'context'.
        """
        super().__init__(rate)
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_retries = max_retries

        if mode not in ['free', 'context']:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'free' or 'context'.")
        self.mode = mode
        
        # GATree의 텐서 구조와 일치하는 상수 정의
        self.COL_NODE_TYPE = 0
        self.COL_PARENT_IDX = 1
        self.COL_DEPTH = 2
        self.COL_PARAM_1 = 3 # 루트 분기 타입 저장에 사용
        self.NODE_TYPE_UNUSED = 0
        self.NODE_TYPE_ROOT_BRANCH = 1
        
        # 문맥-인식 교차에서 사용할 분기 타입 리스트
        self.BRANCH_TYPES = [ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT]

    def __call__(self, parents: torch.Tensor) -> torch.Tensor:
        """
        부모 개체 쌍에 대해 선택된 모드로 서브트리 교차를 수행하여 자식 개체를 생성합니다.
        """
        num_offspring = parents.shape[0] // 2
        node_dim = parents.shape[2]
        
        children = torch.empty((num_offspring, self.max_nodes, node_dim), 
                               dtype=parents.dtype, device=parents.device)
        
        parent_pairs = parents.view(num_offspring, 2, self.max_nodes, node_dim)

        for i, (p1, p2) in enumerate(parent_pairs):
            if torch.rand(1).item() < self.rate:
                # self.mode에 따라 적절한 교차 메소드 호출
                if self.mode == 'free':
                    child1, child2 = self._perform_crossover_pair_free(p1, p2)
                else: # context
                    child1, child2 = self._perform_crossover_pair_context(p1, p2)
                
                children[i] = child1 if torch.rand(1).item() < 0.5 else child2
            else:
                children[i] = p1.clone() if torch.rand(1).item() < 0.5 else p2.clone()
        
        return children

    # --- 방법론 1: 자유 교차 (기존 방식) ---
    def _perform_crossover_pair_free(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """모든 유효한 노드들 사이에서 자유롭게 교차를 시도합니다."""
        p1_candidates = self._get_valid_crossover_points(parent1)
        p2_candidates = self._get_valid_crossover_points(parent2)

        return self._intelligent_retry_crossover(parent1, parent2, p1_candidates, p2_candidates)

    # --- 방법론 2: 문맥-인식 교차 (새로운 방식) ---
    def _perform_crossover_pair_context(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """동일한 루트 분기(LONG/HOLD/SHORT) 내에서만 교차를 시도합니다."""
        # 공평한 시도를 위해 분기 타입 순서를 섞음
        shuffled_branches = self.BRANCH_TYPES[:]
        random.shuffle(shuffled_branches)
        
        for branch_type in shuffled_branches:
            # 각 부모 트리에서 해당 분기에 속하는 교차점만 필터링
            p1_candidates = self._get_contextual_crossover_points(parent1, branch_type)
            p2_candidates = self._get_contextual_crossover_points(parent2, branch_type)

            # 두 부모 모두 해당 문맥에서 교차점이 존재할 경우에만 시도
            if p1_candidates.numel() > 0 and p2_candidates.numel() > 0:
                # 지능적 재시도 로직을 호출하여 실제 교차 수행
                child1, child2 = self._intelligent_retry_crossover(parent1, parent2, p1_candidates, p2_candidates)
                
                # 교차에 성공했다면(부모와 다른 자식이 반환되었다면) 즉시 결과를 반환
                if not (torch.equal(child1, parent1) and torch.equal(child2, parent2)):
                    return child1, child2

        # 모든 문맥에서 교차에 실패한 경우, 부모 복사본 반환
        return parent1.clone(), parent2.clone()

    # --- 핵심 교차 엔진 (리팩토링된 공통 로직) ---
    def _intelligent_retry_crossover(self, parent1: torch.Tensor, parent2: torch.Tensor, 
                                     p1_candidates: torch.Tensor, p2_candidates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        주어진 후보군 내에서 지능적 재시도 메커니즘을 사용하여 교차를 수행하는 핵심 엔진.
        """
        if p1_candidates.numel() == 0 or p2_candidates.numel() == 0:
            return parent1.clone(), parent2.clone()

        for _ in range(self.max_retries):
            if p1_candidates.numel() == 0 or p2_candidates.numel() == 0:
                break 

            p1_idx = p1_candidates[torch.randint(len(p1_candidates), (1,))].item()
            p2_idx = p2_candidates[torch.randint(len(p2_candidates), (1,))].item()

            s1_info = self._get_subtree_info(parent1, p1_idx)
            s2_info = self._get_subtree_info(parent2, p2_idx)
            
            p1_nodes_count = (parent1[:, self.COL_NODE_TYPE] != self.NODE_TYPE_UNUSED).sum().item()
            p2_nodes_count = (parent2[:, self.COL_NODE_TYPE] != self.NODE_TYPE_UNUSED).sum().item()

            p1_can_receive, p1_fail_reason = self._validate_transplant(parent1, p1_idx, p1_nodes_count, s1_info, s2_info)
            p2_can_receive, p2_fail_reason = self._validate_transplant(parent2, p2_idx, p2_nodes_count, s2_info, s1_info)

            if p1_can_receive and p2_can_receive:
                return self._execute_transplant(parent1, p1_idx, s1_info, parent2, p2_idx, s2_info)
            else:
                if not p1_can_receive:
                    p1_candidates = p1_candidates[p1_candidates != (p1_idx if p1_fail_reason == 'DEPTH' else p2_idx)]
                if not p2_can_receive:
                    p2_candidates = p2_candidates[p2_candidates != (p2_idx if p2_fail_reason == 'DEPTH' else p1_idx)]
        
        return parent1.clone(), parent2.clone()

    # --- 헬퍼 함수들 ---
    def _get_valid_crossover_points(self, tree_tensor: torch.Tensor) -> torch.Tensor:
        """루트 분기를 제외한 모든 활성 노드의 인덱스를 반환합니다."""
        active_nodes_mask = tree_tensor[:, self.COL_NODE_TYPE] != self.NODE_TYPE_UNUSED
        not_root_branch_mask = tree_tensor[:, self.COL_PARENT_IDX] != -1
        return (active_nodes_mask & not_root_branch_mask).nonzero(as_tuple=True)[0]
        
    def _get_contextual_crossover_points(self, tree_tensor: torch.Tensor, branch_type: int) -> torch.Tensor:
        """특정 루트 분기 타입에 속하는 유효한 교차점 인덱스만 반환합니다."""
        all_candidates = self._get_valid_crossover_points(tree_tensor)
        if all_candidates.numel() == 0:
            return all_candidates

        # 각 후보 노드의 루트 분기 타입을 찾아서 필터링
        contextual_indices = [
            idx.item() for idx in all_candidates 
            if self._get_root_branch_type(tree_tensor, idx.item()) == branch_type
        ]
        return torch.tensor(contextual_indices, dtype=torch.long, device=tree_tensor.device)

    def _get_root_branch_type(self, tree_tensor: torch.Tensor, node_idx: int) -> int:
        """주어진 노드의 최상위 조상(루트 분기)의 타입을 반환합니다."""
        current_idx = node_idx
        while True:
            parent_idx = int(tree_tensor[current_idx, self.COL_PARENT_IDX].item())
            if parent_idx == -1:
                # 현재 노드가 루트 분기 노드임
                return int(tree_tensor[current_idx, self.COL_PARAM_1].item())
            current_idx = parent_idx

    # _get_subtree_info, _validate_transplant, _execute_transplant, _transplant_one_way
    # 함수들은 변경 없이 그대로 사용하면 되므로 여기에 다시 붙여넣습니다.

    def _get_subtree_info(self, tree_tensor: torch.Tensor, root_idx: int) -> Dict:
        """BFS를 사용하여 서브트리의 정보(인덱스, 노드 수, 상대 깊이)를 추출합니다."""
        q = [root_idx]
        visited = {root_idx}
        indices = [root_idx]
        
        head = 0
        while head < len(q):
            current_idx = q[head]
            head += 1
            
            children_mask = tree_tensor[:, self.COL_PARENT_IDX] == current_idx
            children_indices = children_mask.nonzero(as_tuple=True)[0]
            
            for child_idx in children_indices:
                child_idx = child_idx.item()
                if child_idx not in visited:
                    visited.add(child_idx)
                    q.append(child_idx)
                    indices.append(child_idx)

        indices_tensor = torch.tensor(indices, dtype=torch.long, device=tree_tensor.device)
        subtree_depths = tree_tensor[indices_tensor, self.COL_DEPTH]
        root_depth = tree_tensor[root_idx, self.COL_DEPTH]
        
        return {
            "indices": indices_tensor,
            "node_count": len(indices),
            "max_relative_depth": (subtree_depths - root_depth).max().item() if len(indices) > 0 else 0
        }

    def _validate_transplant(self, recipient_tensor: torch.Tensor, r_idx: int, 
                             recipient_nodes_count: int, s_r_info: Dict, s_d_info: Dict) -> Tuple[bool, Optional[str]]:
        """한 방향의 이식(transplant)이 유효한지 검증합니다."""
        r_parent_idx = int(recipient_tensor[r_idx, self.COL_PARENT_IDX].item())
        insertion_depth = recipient_tensor[r_parent_idx, self.COL_DEPTH].item() + 1
        new_max_depth = insertion_depth + s_d_info["max_relative_depth"]
        if new_max_depth > self.max_depth:
            return False, "DEPTH"

        new_node_count = recipient_nodes_count - s_r_info["node_count"] + s_d_info["node_count"]
        if new_node_count > self.max_nodes:
            return False, "NODE_COUNT"
            
        return True, None

    def _execute_transplant(self, p1: torch.Tensor, p1_idx: int, s1_info: Dict, 
                            p2: torch.Tensor, p2_idx: int, s2_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """실제로 두 서브트리를 교환하여 두 개의 새로운 자식을 생성합니다."""
        child1 = self._transplant_one_way(p1, p1_idx, s1_info, p2, p2_idx, s2_info)
        child2 = self._transplant_one_way(p2, p2_idx, s2_info, p1, p1_idx, s1_info)
        return child1, child2
    
    def _transplant_one_way(self, recipient: torch.Tensor, r_idx: int, s_r_info: Dict,
                            donor: torch.Tensor, d_idx: int, s_d_info: Dict) -> torch.Tensor:
        """한쪽 방향으로 서브트리를 이식하는 내부 헬퍼 함수."""
        new_child = recipient.clone()

        new_child[s_r_info["indices"]].zero_()
        new_child[s_r_info["indices"], self.COL_NODE_TYPE] = self.NODE_TYPE_UNUSED

        empty_slots = (new_child[:, self.COL_NODE_TYPE] == self.NODE_TYPE_UNUSED).nonzero(as_tuple=True)[0]
        new_slots = empty_slots[:s_d_info["node_count"]]
        
        old_to_new_map = {old.item(): new.item() for old, new in zip(s_d_info["indices"], new_slots)}
        
        r_parent_idx = int(recipient[r_idx, self.COL_PARENT_IDX].item())
        insertion_depth = recipient[r_parent_idx, self.COL_DEPTH].item() + 1
        depth_offset = insertion_depth - donor[d_idx, self.COL_DEPTH].item()
        
        for old_idx_tensor in s_d_info["indices"]:
            old_idx = old_idx_tensor.item()
            new_idx = old_to_new_map[old_idx]
            
            new_child[new_idx] = donor[old_idx]
            new_child[new_idx, self.COL_DEPTH] += depth_offset
            
            old_parent_idx = int(donor[old_idx, self.COL_PARENT_IDX].item())
            if old_idx == d_idx:
                new_child[new_idx, self.COL_PARENT_IDX] = r_parent_idx
            else:
                new_child[new_idx, self.COL_PARENT_IDX] = old_to_new_map[old_parent_idx]
                
        return new_child