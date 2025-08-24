# --- START OF FILE evolution/Mutation/reinitialize_node.py ---

# evolution/Mutation/reinitialize_node.py (수정된 최종본)
import torch
from .base import BaseMutation
from .utils import _create_random_action_params, _create_random_decision_params
from typing import Dict, Any

try:
    import gatree_cuda_compat as gatree_cuda
except ImportError:
    gatree_cuda = None

# model.py에서 상수 임포트
from models.constants import (
    COL_NODE_TYPE, COL_PARAM_1, NODE_TYPE_DECISION, NODE_TYPE_ACTION
)

class ReinitializeNodeMutation(BaseMutation):
    """[수정] 노드의 파라미터를 완전히 새로운 랜덤 값으로 교체합니다. (CUDA 가속화 적용)"""
    def __init__(self, prob: float = 0.05, config: Dict[str, Any] = None):
        super().__init__(prob)
        if config is None:
            raise ValueError("ReinitializeNodeMutation requires a 'config' dictionary.")
        self.config = config

        # [신규] CUDA 커널에 전달할 데이터 미리 준비
        all_features = self.config['all_features']
        feature_num = self.config['feature_num']
        feature_comparison_map = self.config.get('feature_comparison_map', {})
        feature_bool = self.config.get('feature_bool', [])

        self.feature_num_indices_list = [all_features.index(name) for name in feature_num.keys()]
        self.feature_min_vals_list = [float(feature_num[name][0]) for name in feature_num.keys()]
        self.feature_max_vals_list = [float(feature_num[name][1]) for name in feature_num.keys()]
        
        comp_keys = list(feature_comparison_map.keys())
        self.feature_comparison_indices_list = [all_features.index(name) for name in comp_keys]
        self.feature_bool_indices_list = [all_features.index(name) for name in feature_bool]

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        if gatree_cuda is None:
            raise ImportError("gatree_cuda module is not compiled or available.")

        device = chromosomes.device

        feature_num_indices = torch.tensor(self.feature_num_indices_list, dtype=torch.int32, device=device)
        feature_min_vals = torch.tensor(self.feature_min_vals_list, dtype=torch.float32, device=device)
        feature_max_vals = torch.tensor(self.feature_max_vals_list, dtype=torch.float32, device=device)
        feature_comparison_indices = torch.tensor(self.feature_comparison_indices_list, dtype=torch.int32, device=device)
        feature_bool_indices = torch.tensor(self.feature_bool_indices_list, dtype=torch.int32, device=device)
        
        # [수정] ReinitializeNodeMutation 전용 CUDA 함수 호출
        gatree_cuda.reinitialize_node_mutate(
            chromosomes,
            self.prob,
            feature_num_indices,
            feature_min_vals,
            feature_max_vals,
            feature_comparison_indices,
            feature_bool_indices
        )
        # Validate trees after CUDA reinitialize-node mutation (if available)
        try:
            if gatree_cuda is not None and chromosomes.is_cuda:
                gatree_cuda.validate_trees(chromosomes.contiguous())
                print('complete reinit mutation')
        except Exception:
            import traceback
            raise RuntimeError(f"gatree_cuda.validate_trees failed after reinit mutation.\n{traceback.format_exc()}")

        return chromosomes
