# evolution/Crossover/root_branch.py (수정된 최종 코드)
import torch
from .base import BaseCrossover
from typing import Tuple

from models.constants import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, COL_PARAM_1,
    NODE_TYPE_UNUSED, NODE_TYPE_ROOT_BRANCH, ROOT_BRANCH_LONG,
    ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT
)

# C++/CUDA로 구현된 헬퍼 함수 임포트
try:
    import gatree_cuda
except ImportError:
    print("="*60)
    print(">>> 경고: 'gatree_cuda' 모듈을 찾을 수 없습니다.")
    print(">>> C++/CUDA 코드를 먼저 컴파일해야 합니다.")
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
        [개선 2] '배치' 단위로 루트 분기 교차를 수행합니다.
        이 과정의 핵심은 C++/CUDA 커널로 위임됩니다.
        """
        if gatree_cuda is None:
            raise RuntimeError("gatree_cuda module is not loaded. Cannot perform crossover.")

        num_to_cross = p1_batch.shape[0]
        device = p1_batch.device

        # 1. 자식 텐서 초기화 (모든 노드를 UNUSED로 설정)
        child_batch = torch.zeros_like(p1_batch)

        # 2. 모든 자식의 루트 분기 노드(0, 1, 2번 인덱스)를 한 번에 생성
        root_indices = torch.arange(3, device=device)
        child_batch[:, root_indices, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
        child_batch[:, root_indices, COL_PARENT_IDX] = -1
        child_batch[:, root_indices, COL_DEPTH] = 0
        
        branch_types_tensor = torch.tensor([ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT], device=device)
        child_batch[:, root_indices, COL_PARAM_1] = branch_types_tensor

        # 3. 각 자식의 각 브랜치에 대해 어떤 부모로부터 유전받을지 랜덤하게 결정
        # donor_map: (num_to_cross, 3) 크기. 0은 p1, 1은 p2를 의미.
        donor_map = torch.randint(0, 2, (num_to_cross, 3), device=device, dtype=torch.int32)
        
        # ======================================================================
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ C++/CUDA 구현 호출 부분 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # ======================================================================
        # CUDA 커널이 사용할 스크래치 버퍼를 Python에서 생성하여 전달
        # 각 스레드(자식)는 BFS 큐, 결과 인덱스, old_to_new 맵을 위한 공간이 필요
        scratch_buffer_size_per_thread = self.max_nodes * 3 
        scratch_buffer = torch.empty(
            (num_to_cross, scratch_buffer_size_per_thread), 
            dtype=torch.int32, 
            device=device
        )
        
        gatree_cuda.copy_branches_batch(
            child_batch, p1_batch, p2_batch, donor_map, scratch_buffer
        )
        # ======================================================================
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ C++/CUDA 구현 호출 부분 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        # ======================================================================

        return child_batch