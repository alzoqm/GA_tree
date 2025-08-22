# evolution/Crossover/root_branch.py (수정된 최종 코드)
import torch
from .base import BaseCrossover
from typing import Tuple

from models.constants import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, COL_PARAM_1,
    NODE_TYPE_UNUSED, NODE_TYPE_ROOT_BRANCH, NODE_TYPE_ACTION, ROOT_BRANCH_LONG,
    ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT, ACTION_CLOSE_ALL
)

# C++/CUDA로 구현된 헬퍼 함수 임포트
try:
    import gatree_cuda
except ImportError:
    print("="*60)
    print(">>> 경고: 'gatree_cuda' 모듈을 찾을 수 없습니다.")
    print(">>> C++/CUDA 코드를 먼저 컴파일해야 합니다.")
    print(">>> 프로젝트 루트에서 다음 명령을 실행하세요:")
    print(">>> python setup.py build_ext --inplace")
    print("="*60)
    gatree_cuda = None

class RootBranchCrossover(BaseCrossover):
    """
    [개선된 버전]
    GPU 병렬 처리에 최적화된 루트 분기 교차 연산자.
    부모들의 루트 분기(LONG/HOLD/SHORT)를 재조합하여 전체 자식 배치를 한 번의 텐서 연산으로 생성합니다.
    """
    def __init__(self, rate: float = 0.8, max_nodes: int = 100):
        super().__init__(rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Crossover rate must be between 0.0 and 1.0, but got {rate}")
        self.max_nodes = max_nodes

    def __call__(self, parents: torch.Tensor) -> torch.Tensor:
        """
        [개선 1] 전체 부모 배치에 대한 교차 연산을 루프 없이 수행합니다.
        """
        num_offspring = parents.shape[0] // 2
        if num_offspring == 0:
            return torch.empty((0, *parents.shape[1:]), dtype=parents.dtype, device=parents.device)

        # 부모 쌍 생성 (num_offspring, 2, max_nodes, node_dim)
        parent_pairs = parents.view(num_offspring, 2, -1, parents.shape[-1])
        p1_batch = parent_pairs[:, 0]
        p2_batch = parent_pairs[:, 1]
        
        children = torch.empty_like(p1_batch)
        rand_vals = torch.rand(num_offspring, device=parents.device)
        
        # 1. Crossover를 수행할 자식 결정
        crossover_mask = rand_vals < self.rate
        num_crossover = crossover_mask.sum().item()

        if num_crossover > 0:
            p1_to_cross = p1_batch[crossover_mask]
            p2_to_cross = p2_batch[crossover_mask]
            crossed_children = self._perform_crossover_batch(p1_to_cross, p2_to_cross)
            children[crossover_mask] = crossed_children

        # 2. P1을 그대로 복제할 자식 결정
        clone_p1_mask = (rand_vals >= self.rate) & (rand_vals < self.rate + (1.0 - self.rate) / 2.0)
        if clone_p1_mask.any():
            children[clone_p1_mask] = p1_batch[clone_p1_mask].clone()

        # 3. P2를 그대로 복제할 자식 결정
        clone_p2_mask = rand_vals >= self.rate + (1.0 - self.rate) / 2.0
        if clone_p2_mask.any():
            children[clone_p2_mask] = p2_batch[clone_p2_mask].clone()
            
        return children

    def _perform_crossover_batch(self, p1_batch: torch.Tensor, p2_batch: torch.Tensor) -> torch.Tensor:
        """
        [수정됨] '배치' 단위로 루트 분기 교차를 수행하며, 입력과 출력이 모두 GPU 텐서입니다.
        """
        if gatree_cuda is None:
            raise RuntimeError("gatree_cuda module is not loaded. Cannot perform crossover.")

        num_to_cross = p1_batch.shape[0]
        device = p1_batch.device

        # 모든 텐서를 GPU에 생성
        child_batch = torch.zeros_like(p1_batch)
        root_indices = torch.arange(3, device=device)
        child_batch[:, root_indices, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
        child_batch[:, root_indices, COL_PARENT_IDX] = -1
        child_batch[:, root_indices, COL_DEPTH] = 0
        
        branch_types_tensor = torch.tensor([ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT], device=device, dtype=p1_batch.dtype)
        child_batch[:, root_indices, COL_PARAM_1] = branch_types_tensor

        donor_map = torch.randint(0, 2, (num_to_cross, 3), device=device, dtype=torch.int32)
        
        bfs_queue_buffer = torch.empty((num_to_cross, self.max_nodes), dtype=torch.int32, device=device)
        result_indices_buffer = torch.empty((num_to_cross, self.max_nodes), dtype=torch.int32, device=device)
        old_to_new_map_buffer = torch.empty((num_to_cross, self.max_nodes), dtype=torch.int32, device=device)
        
        # GPU 텐서들로 CUDA 커널 호출 (입력 p1_batch, p2_batch는 이미 GPU에 있음)
        gatree_cuda.copy_branches_batch(
            child_batch, p1_batch, p2_batch, donor_map,
            bfs_queue_buffer, result_indices_buffer, old_to_new_map_buffer
        )
        
        # Fix any root branches that have no children by adding default ACTION nodes
        self._fix_empty_root_branches(child_batch)
        
        # Fix any orphaned DECISION nodes that became leaves
        self._fix_orphaned_decision_nodes(child_batch)
        
        # Validate trees after CUDA root-branch crossover (if available)
        try:
            if gatree_cuda is not None and child_batch.is_cuda:
                gatree_cuda.validate_trees(child_batch.contiguous())
                print('complete root branch crossover')
        except Exception:
            import traceback
            raise RuntimeError(f"gatree_cuda.validate_trees failed after root_branch crossover.\n{traceback.format_exc()}")

        return child_batch # GPU 텐서를 그대로 반환

    def _fix_empty_root_branches(self, child_batch: torch.Tensor):
        """
        Fix any root branches that have no children by adding default ACTION nodes.
        This ensures that root branches are never leaf nodes, which would violate
        the tree structure constraint that only ACTION nodes can be leaves.
        """
        device = child_batch.device
        batch_size, max_nodes, node_dim = child_batch.shape
        
        for b in range(batch_size):
            # Check each root branch (indices 0, 1, 2) for children
            for root_idx in range(3):
                has_children = False
                
                # Check if this root branch has any children
                for i in range(3, max_nodes):
                    if child_batch[b, i, COL_NODE_TYPE] != NODE_TYPE_UNUSED:
                        if child_batch[b, i, COL_PARENT_IDX] == root_idx:
                            has_children = True
                            break
                
                # If no children found, add a default ACTION node
                if not has_children:
                    # Find next available index
                    next_available_idx = None
                    for i in range(3, max_nodes):
                        if child_batch[b, i, COL_NODE_TYPE] == NODE_TYPE_UNUSED:
                            next_available_idx = i
                            break
                    
                    if next_available_idx is not None:
                        child_batch[b, next_available_idx, COL_NODE_TYPE] = NODE_TYPE_ACTION
                        child_batch[b, next_available_idx, COL_PARENT_IDX] = root_idx
                        child_batch[b, next_available_idx, COL_DEPTH] = 1
                        child_batch[b, next_available_idx, COL_PARAM_1] = ACTION_CLOSE_ALL  # Safe default action
                        
                        # Clear other parameters
                        for col in range(4, node_dim):
                            child_batch[b, next_available_idx, col] = 0.0
                    else:
                        # Tree is full, need to find and fix the issue differently
                        # Strategy: Find a DECISION node that became a leaf and convert it to ACTION
                        from models.constants import NODE_TYPE_DECISION
                        found_node = False
                        for i in range(max_nodes-1, 2, -1):  # Search backwards
                            if child_batch[b, i, COL_NODE_TYPE] == NODE_TYPE_DECISION:
                                # Check if this DECISION node is a leaf (no children)
                                has_children = False
                                for j in range(3, max_nodes):
                                    if (child_batch[b, j, COL_NODE_TYPE] != NODE_TYPE_UNUSED and
                                        child_batch[b, j, COL_PARENT_IDX] == i):
                                        has_children = True
                                        break
                                
                                if not has_children:
                                    # Convert this leaf DECISION to ACTION and reassign to empty root
                                    child_batch[b, i, COL_NODE_TYPE] = NODE_TYPE_ACTION
                                    child_batch[b, i, COL_PARENT_IDX] = root_idx
                                    child_batch[b, i, COL_DEPTH] = 1
                                    child_batch[b, i, COL_PARAM_1] = ACTION_CLOSE_ALL
                                    for col in range(4, node_dim):
                                        child_batch[b, i, col] = 0.0
                                    found_node = True
                                    break
                        
                        if not found_node:
                            # Last resort: find ANY ACTION node and reassign it
                            for i in range(max_nodes-1, 2, -1):
                                if child_batch[b, i, COL_NODE_TYPE] == NODE_TYPE_ACTION:
                                    child_batch[b, i, COL_PARENT_IDX] = root_idx
                                    child_batch[b, i, COL_DEPTH] = 1
                                    child_batch[b, i, COL_PARAM_1] = ACTION_CLOSE_ALL
                                    for col in range(4, node_dim):
                                        child_batch[b, i, col] = 0.0
                                    found_node = True
                                    break

    def _fix_orphaned_decision_nodes(self, child_batch: torch.Tensor):
        """
        Fix DECISION nodes that have become leaves (have no children).
        Convert them to ACTION nodes since only ACTION nodes can be leaves.
        """
        from models.constants import NODE_TYPE_DECISION, NODE_TYPE_ACTION
        
        batch_size, max_nodes, node_dim = child_batch.shape
        
        for b in range(batch_size):
            # Find DECISION nodes that are leaves
            for i in range(3, max_nodes):  # Skip root branches
                if child_batch[b, i, COL_NODE_TYPE] == NODE_TYPE_DECISION:
                    # Check if this DECISION node has any children
                    has_children = False
                    for j in range(3, max_nodes):
                        if (child_batch[b, j, COL_NODE_TYPE] != NODE_TYPE_UNUSED and
                            child_batch[b, j, COL_PARENT_IDX] == i):
                            has_children = True
                            break
                    
                    # If no children, convert to ACTION node
                    if not has_children:
                        child_batch[b, i, COL_NODE_TYPE] = NODE_TYPE_ACTION
                        child_batch[b, i, COL_PARAM_1] = ACTION_CLOSE_ALL  # Safe default action
                        # Clear other parameters that are not needed for ACTION nodes
                        for col in range(4, node_dim):
                            child_batch[b, i, col] = 0.0
