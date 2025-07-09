# evolution/Crossover/subtree.py
import torch
from .base import BaseCrossover
from typing import Dict, Tuple, Optional

class SubtreeCrossover(BaseCrossover):
    """
    두 트리의 서브트리(가지)를 교환하는 교차 연산자.
    교차 시도 실패 시, 실패 원인(노드 수, 깊이)을 분석하여 지능적으로 재시도합니다.
    """
    def __init__(self, 
                 rate: float = 0.8, 
                 max_nodes: int = 100, 
                 max_depth: int = 3, 
                 max_retries: int = 5):
        """
        SubtreeCrossover 초기화.

        Args:
            rate (float): 교차가 일어날 확률.
            max_nodes (int): 트리가 가질 수 있는 최대 노드 수.
            max_depth (int): 트리의 최대 깊이.
            max_retries (int): 유효한 교차점을 찾기 위한 최대 재시도 횟수.
        """
        super().__init__(rate)
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_retries = max_retries

        # GATree의 텐서 구조와 일치하는 상수 정의
        self.COL_NODE_TYPE = 0
        self.COL_PARENT_IDX = 1
        self.COL_DEPTH = 2
        self.NODE_TYPE_UNUSED = 0

    def __call__(self, parents: torch.Tensor) -> torch.Tensor:
        """
        부모 개체 쌍에 대해 서브트리 교차를 수행하여 자식 개체를 생성합니다.
        Evolution.py의 요구사항에 맞춰 한 쌍의 부모로부터 하나의 자식을 생성합니다.

        Args:
            parents (torch.Tensor): 부모 개체들의 텐서. (num_parents, max_nodes, node_dim)
                                    num_parents는 짝수여야 합니다.

        Returns:
            torch.Tensor: 생성된 자식 개체들의 텐서. (num_offspring, max_nodes, node_dim)
        """
        num_offspring = parents.shape[0] // 2
        node_dim = parents.shape[2]
        
        children = torch.empty((num_offspring, self.max_nodes, node_dim), 
                               dtype=parents.dtype, device=parents.device)
        
        # 부모를 2개씩 짝지어 처리
        parent_pairs = parents.view(num_offspring, 2, self.max_nodes, node_dim)

        for i, (p1, p2) in enumerate(parent_pairs):
            if torch.rand(1).item() < self.rate:
                # _perform_crossover_pair는 두 개의 자식(child1, child2)을 반환
                child1, child2 = self._perform_crossover_pair(p1, p2)
                # 두 자식 중 하나를 무작위로 선택하여 다음 세대로 전달
                children[i] = child1 if torch.rand(1).item() < 0.5 else child2
            else:
                # 교차가 일어나지 않으면 부모 중 하나를 그대로 복사
                children[i] = p1.clone() if torch.rand(1).item() < 0.5 else p2.clone()
        
        return children

    def _perform_crossover_pair(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """지능적 재시도 메커니즘을 사용하여 한 쌍의 부모를 교차시킵니다."""
        
        p1_candidates = self._get_valid_crossover_points(parent1)
        p2_candidates = self._get_valid_crossover_points(parent2)

        if p1_candidates.numel() == 0 or p2_candidates.numel() == 0:
            return parent1.clone(), parent2.clone() # 교차 가능한 지점 없음

        for _ in range(self.max_retries):
            if p1_candidates.numel() == 0 or p2_candidates.numel() == 0:
                break # 시도할 후보가 더 이상 없음

            # 1. 교차점 무작위 선택
            p1_idx = p1_candidates[torch.randint(len(p1_candidates), (1,))].item()
            p2_idx = p2_candidates[torch.randint(len(p2_candidates), (1,))].item()

            # 2. 서브트리 정보 추출
            s1_info = self._get_subtree_info(parent1, p1_idx)
            s2_info = self._get_subtree_info(parent2, p2_idx)
            
            p1_nodes_count = (parent1[:, self.COL_NODE_TYPE] != self.NODE_TYPE_UNUSED).sum().item()
            p2_nodes_count = (parent2[:, self.COL_NODE_TYPE] != self.NODE_TYPE_UNUSED).sum().item()

            # 3. 양방향 유효성 검증
            p1_can_receive, p1_fail_reason = self._validate_transplant(parent1, p1_idx, p1_nodes_count, s1_info, s2_info)
            p2_can_receive, p2_fail_reason = self._validate_transplant(parent2, p2_idx, p2_nodes_count, s2_info, s1_info)

            # 4. 결과에 따른 분기
            if p1_can_receive and p2_can_receive:
                # 성공! 실제 교환 로직 실행 후 반환
                return self._execute_transplant(parent1, p1_idx, s1_info, parent2, p2_idx, s2_info)
            else:
                # 실패! 실패 원인에 따라 후보군 조정
                if not p1_can_receive:
                    if p1_fail_reason == 'DEPTH':
                        p1_candidates = p1_candidates[p1_candidates != p1_idx]
                    elif p1_fail_reason == 'NODE_COUNT':
                        p2_candidates = p2_candidates[p2_candidates != p2_idx]
                
                if not p2_can_receive:
                    if p2_fail_reason == 'DEPTH':
                        p2_candidates = p2_candidates[p2_candidates != p2_idx]
                    elif p2_fail_reason == 'NODE_COUNT':
                        p1_candidates = p1_candidates[p1_candidates != p1_idx]
        
        # 최대 재시도 후에도 실패 시, 부모 복사본 반환
        return parent1.clone(), parent2.clone()

    def _get_valid_crossover_points(self, tree_tensor: torch.Tensor) -> torch.Tensor:
        """루트 분기를 제외한 모든 활성 노드의 인덱스를 반환합니다."""
        active_nodes_mask = tree_tensor[:, self.COL_NODE_TYPE] != self.NODE_TYPE_UNUSED
        not_root_branch_mask = tree_tensor[:, self.COL_PARENT_IDX] != -1
        valid_indices = (active_nodes_mask & not_root_branch_mask).nonzero(as_tuple=True)[0]
        return valid_indices

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
        # 깊이 검증
        r_parent_idx = int(recipient_tensor[r_idx, self.COL_PARENT_IDX].item())
        insertion_depth = recipient_tensor[r_parent_idx, self.COL_DEPTH].item() + 1
        new_max_depth = insertion_depth + s_d_info["max_relative_depth"]
        if new_max_depth > self.max_depth:
            return False, "DEPTH"

        # 노드 수 검증
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

        # 1. 기존 서브트리 제거 (슬롯 비우기)
        new_child[s_r_info["indices"]].zero_()
        new_child[s_r_info["indices"], self.COL_NODE_TYPE] = self.NODE_TYPE_UNUSED

        # 2. 이식할 빈 슬롯 찾기
        empty_slots = (new_child[:, self.COL_NODE_TYPE] == self.NODE_TYPE_UNUSED).nonzero(as_tuple=True)[0]
        new_slots = empty_slots[:s_d_info["node_count"]]
        
        # 3. 인덱스 매핑 (기증자 트리의 옛 인덱스 -> 수혜자 트리의 새 인덱스)
        old_to_new_map = {old.item(): new.item() for old, new in zip(s_d_info["indices"], new_slots)}
        
        # 4. 깊이 오프셋 계산
        r_parent_idx = int(recipient[r_idx, self.COL_PARENT_IDX].item())
        insertion_depth = recipient[r_parent_idx, self.COL_DEPTH].item() + 1
        depth_offset = insertion_depth - donor[d_idx, self.COL_DEPTH].item()
        
        # 5. 서브트리 노드 복사 및 재연결
        for old_idx_tensor in s_d_info["indices"]:
            old_idx = old_idx_tensor.item()
            new_idx = old_to_new_map[old_idx]
            
            # 노드 데이터 복사
            new_child[new_idx] = donor[old_idx]
            
            # 깊이 업데이트
            new_child[new_idx, self.COL_DEPTH] += depth_offset
            
            # 부모 인덱스 업데이트
            old_parent_idx = int(donor[old_idx, self.COL_PARENT_IDX].item())
            if old_idx == d_idx:
                # 서브트리의 루트는 수혜자의 이식 지점 부모와 연결
                new_child[new_idx, self.COL_PARENT_IDX] = r_parent_idx
            else:
                # 서브트리 내부 노드는 새로운 부모 인덱스로 연결
                new_child[new_idx, self.COL_PARENT_IDX] = old_to_new_map[old_parent_idx]
                
        return new_child