# evolution/Crossover/subtree.py (수정된 최종본)
import torch
from .base import BaseCrossover
from typing import Tuple

# 프로젝트 상수 통일
from models.constants import (
    ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT
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


class SubtreeCrossover(BaseCrossover):
    """
    [개선된 배치 버전 - 메모리 안전]
    GPU 병렬 처리에 최적화된 서브트리 교차 연산자.
    모든 임시 배열을 Python에서 텐서 버퍼로 생성하여 커널에 전달함으로써 메모리 안전성을 보장합니다.
    """
    def __init__(
        self,
        rate: float = 0.8,
        max_nodes: int = 100,
        max_depth: int = 3,
        max_retries: int = 5,
        mode: str = "free",
    ):
        super().__init__(rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Crossover rate must be between 0.0 and 1.0, but got {rate}")
        if mode not in ["free", "context"]:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'free' or 'context'.")

        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_retries = max_retries
        self.mode = mode
        self.BRANCH_TYPES = [ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT]

    def __call__(self, parents: torch.Tensor) -> torch.Tensor:
        """
        [개선 1] 전체 부모 배치에 대한 교차 연산을 루프 없이 수행합니다.
        parents: (batch=2*num_offspring, max_nodes, node_dim)
        return : (num_offspring, max_nodes, node_dim)
        """
        num_offspring = parents.shape[0] // 2
        if num_offspring == 0:
            return torch.empty((0, *parents.shape[1:]), dtype=parents.dtype, device=parents.device)

        # 부모 쌍 묶기: (num_offspring, 2, max_nodes, node_dim)
        parent_pairs = parents.view(num_offspring, 2, -1, parents.shape[-1])
        p1_batch = parent_pairs[:, 0]
        p2_batch = parent_pairs[:, 1]

        # 결과 버퍼
        children = torch.empty_like(p1_batch)

        # 어떤 자식이 교차 대상인지 결정
        rand_vals = torch.rand(num_offspring, device=parents.device)
        crossover_mask = rand_vals < self.rate

        # --- 1) 교차 대상 샘플들만 모아서 배치 교차 수행 ---
        if crossover_mask.any():
            crossed = self._perform_crossover_batch(
                p1_batch[crossover_mask], p2_batch[crossover_mask]
            )
            children[crossover_mask] = crossed

        # --- 2) 나머지: P1/P2 원본을 1-rate를 절반씩 복제 ---
        clone_p1_mask = (rand_vals >= self.rate) & (
            rand_vals < self.rate + (1.0 - self.rate) / 2.0
        )
        if clone_p1_mask.any():
            children[clone_p1_mask] = p1_batch[clone_p1_mask].clone()

        clone_p2_mask = rand_vals >= self.rate + (1.0 - self.rate) / 2.0
        if clone_p2_mask.any():
            children[clone_p2_mask] = p2_batch[clone_p2_mask].clone()

        return children


    def _perform_crossover_batch(self, p1_batch: torch.Tensor, p2_batch: torch.Tensor) -> torch.Tensor:
        """
        [개선 2] '배치' 단위로 서브트리 교차를 수행하고,
        각 샘플마다 child1/child2 중 랜덤하게 하나를 반환합니다.
        """
        if gatree_cuda is None:
            raise RuntimeError("gatree_cuda module is not loaded. Cannot perform subtree crossover.")

        num_to_cross = p1_batch.shape[0]
        device = p1_batch.device

        # 출력 버퍼 (커널이 채움)
        child1_out = torch.empty_like(p1_batch)
        child2_out = torch.empty_like(p2_batch)

        # [수정] 커널 내부의 모든 임시 배열을 Python에서 버퍼로 생성
        bfs_queue_buffer = torch.empty((num_to_cross, self.max_nodes), dtype=torch.int32, device=device)
        result_indices_buffer = torch.empty((num_to_cross, self.max_nodes), dtype=torch.int32, device=device)
        old_to_new_map_buffer = torch.empty((num_to_cross, self.max_nodes), dtype=torch.int32, device=device)
        
        # [신규] 후보군 저장을 위한 추가 버퍼
        p1_candidates_buffer = torch.empty((num_to_cross, self.max_nodes), dtype=torch.int32, device=device)
        p2_candidates_buffer = torch.empty((num_to_cross, self.max_nodes), dtype=torch.int32, device=device)


        # context 모드 공정성 확보를 위한 분기 순열(샘플별로 LONG/HOLD/SHORT 순서를 랜덤화)
        if self.mode == "context":
            branch_types = torch.tensor(self.BRANCH_TYPES, device=device, dtype=torch.int32)
            # (3,) → (num_to_cross, 3)로 expand 후 각 행을 독립적으로 셔플
            branch_perm = branch_types.repeat(num_to_cross, 1)
            # 셔플 인덱스 생성
            shuffle_idx = torch.stack([torch.randperm(3, device=device) for _ in range(num_to_cross)], dim=0)
            branch_perm = torch.gather(branch_perm, 1, shuffle_idx)
        else:
            branch_perm = torch.empty((0, 3), dtype=torch.int32, device=device)  # 더미

        # [수정] CUDA 함수 호출 시 추가 버퍼 전달
        gatree_cuda.subtree_crossover_batch(
            child1_out,
            child2_out,
            p1_batch,
            p2_batch,
            1 if self.mode == "context" else 0,
            int(self.max_depth),
            int(self.max_nodes),
            int(self.max_retries),
            branch_perm,
            bfs_queue_buffer,
            result_indices_buffer,
            old_to_new_map_buffer,
            p1_candidates_buffer, # <--- 신규 버퍼 전달
            p2_candidates_buffer  # <--- 신규 버퍼 전달
        )

        # child1/child2 중 하나를 선택(각 샘플 0.5 확률)
        select_mask = (torch.rand(num_to_cross, 1, 1, device=device) < 0.5)
        final_children = torch.where(select_mask, child1_out, child2_out)
        return final_children