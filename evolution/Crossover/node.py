# evolution/Crossover/node.py
import torch
from typing import Tuple
from .base import BaseCrossover

# 프로젝트 구조에 따라 model.py에서 상수 임포트
from models.constants import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_PARAM_1, NODE_TYPE_UNUSED, 
    NODE_TYPE_ROOT_BRANCH,
    NODE_TYPE_DECISION, NODE_TYPE_ACTION, ROOT_BRANCH_LONG, 
    ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT
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

class NodeCrossover(BaseCrossover):
    """
    [개선된 버전]
    GPU 병렬 처리에 최적화된 노드 교차 연산자.
    전체 자식 배치를 한 번의 텐서 연산으로 처리합니다.
    'free' 모드: 노드 타입만 맞으면 교환.
    'context' 모드: 루트 분기 및 노드 타입이 모두 맞아야 교환.
    """
    def __init__(self, rate: float = 0.8, mode: str = 'free'):
        super().__init__(rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Crossover rate must be between 0.0 and 1.0, but got {rate}")
        if mode not in ['free', 'context']:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'free' or 'context'.")
        
        self.mode = mode
        
        # GATree 텐서 구조 상수
        self.NODE_TYPE_DECISION = NODE_TYPE_DECISION
        self.NODE_TYPE_ACTION = NODE_TYPE_ACTION
        self.BRANCH_TYPES = [ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT]

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

    def _perform_crossover_batch(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        [수정됨] '배치' 단위로 교차를 수행하며, 입력과 출력이 모두 GPU 텐서입니다.
        """
        if gatree_cuda is None:
            raise RuntimeError("gatree_cuda module is not loaded. Cannot perform crossover.")
        
        # 입력 p1, p2는 이미 GPU 텐서라고 가정
        child1, child2 = p1.clone(), p2.clone()
        
        for node_type in [self.NODE_TYPE_DECISION, self.NODE_TYPE_ACTION]:
            if self.mode == 'free':
                p1_mask = (p1[:, :, COL_NODE_TYPE] == node_type)
                p2_mask = (p2[:, :, COL_NODE_TYPE] == node_type)
                gatree_cuda.swap_node_params(child1, child2, p1_mask, p2_mask)
            else: # context mode
                for branch_type in self.BRANCH_TYPES:
                    p1_context_mask = gatree_cuda.get_contextual_mask(p1, node_type, branch_type)
                    p2_context_mask = gatree_cuda.get_contextual_mask(p2, node_type, branch_type)
                    gatree_cuda.swap_node_params(child1, child2, p1_context_mask, p2_context_mask)
        
        num_to_cross = p1.shape[0]
        selection_mask = torch.rand(num_to_cross, 1, 1, device=p1.device) < 0.5
        final_children = torch.where(selection_mask, child1, child2)

        # Validate trees after CUDA node crossover (if available)
        try:
            if gatree_cuda is not None and final_children.is_cuda:
                gatree_cuda.validate_trees(final_children.contiguous())
        except Exception:
            import traceback
            raise RuntimeError(f"gatree_cuda.validate_trees failed after node crossover.\n{traceback.format_exc()}")

        return final_children # GPU 텐서를 그대로 반환
