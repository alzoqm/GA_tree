# --- START OF FILE evolution/Mutation/node_param.py ---

import torch
import random
from .base import BaseMutation
from typing import Dict, Any

try:
    import gatree_cuda
except ImportError:
    gatree_cuda = None


from models.constants import (
    COL_NODE_TYPE, COL_PARAM_1, COL_PARAM_2, COL_PARAM_3, COL_PARAM_4,
    NODE_TYPE_DECISION, NODE_TYPE_ACTION, COMP_TYPE_FEAT_NUM, COMP_TYPE_FEAT_BOOL,
    OP_GTE, OP_LTE, 
    # [신규] 새로운 Action 상수 임포트
    ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_CLOSE_PARTIAL, 
    ACTION_ADD_POSITION, ACTION_FLIP_POSITION
)

class NodeParamMutation(BaseMutation):
    """[수정] 노드의 파라미터를 미세 조정하는 변이. (CUDA 가속화 적용)"""
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None,
                 noise_ratio: float = 0.1, leverage_change: int = 5):
        super().__init__(prob)
        if config is None:
            raise ValueError("NodeParamMutation requires a 'config' dictionary.")
        self.config = config
        self.noise_ratio = noise_ratio
        self.leverage_change = leverage_change
        
        # [신규] CUDA 커널에 전달할 데이터 미리 준비
        all_features = self.config['all_features']
        feature_num = self.config['feature_num']
        self.feature_num_indices_list = [all_features.index(name) for name in feature_num.keys()]
        self.feature_min_vals_list = [float(feature_num[name][0]) for name in feature_num.keys()]
        self.feature_max_vals_list = [float(feature_num[name][1]) for name in feature_num.keys()]

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        if gatree_cuda is None:
            raise ImportError("gatree_cuda module not compiled or available.")
        
        device = chromosomes.device
        
        feature_num_indices = torch.tensor(self.feature_num_indices_list, dtype=torch.int32, device=device)
        feature_min_vals = torch.tensor(self.feature_min_vals_list, dtype=torch.float32, device=device)
        feature_max_vals = torch.tensor(self.feature_max_vals_list, dtype=torch.float32, device=device)

        # [수정] NodeParamMutation 전용 CUDA 함수 호출
        gatree_cuda.node_param_mutate(
            chromosomes,
            self.prob,
            self.noise_ratio,
            self.leverage_change,
            feature_num_indices,
            feature_min_vals,
            feature_max_vals
        )
        
        return chromosomes