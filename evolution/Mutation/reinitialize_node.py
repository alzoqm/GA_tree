# evolution/Mutation/reinitialize_node.py
import torch
from .base import BaseMutation
from .utils import _create_random_action_params, _create_random_decision_params
from typing import Dict, Any

# model.py에서 상수 임포트
from models.model import (
    COL_NODE_TYPE, COL_PARAM_1, NODE_TYPE_DECISION, NODE_TYPE_ACTION
)

class ReinitializeNodeMutation(BaseMutation):
    """
    노드의 파라미터를 완전히 새로운 랜덤 값으로 교체합니다. (값 기반 변이)
    지역 최적해 탈출에 도움을 줍니다.
    """
    def __init__(self, prob: float = 0.05, config: Dict[str, Any] = None):
        super().__init__(prob)
        if config is None:
            raise ValueError("ReinitializeNodeMutation requires a 'config' dictionary.")
        self.config = config

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        mutated_chromosomes = chromosomes.clone()
        pop_size, max_nodes, _ = chromosomes.shape

        types = chromosomes[:, :, COL_NODE_TYPE]
        mutable_type_mask = (types == NODE_TYPE_DECISION) | (types == NODE_TYPE_ACTION)
        prob_mask = torch.rand(pop_size, max_nodes, device=chromosomes.device) < self.prob
        mutation_mask = mutable_type_mask & prob_mask

        if not mutation_mask.any():
            return mutated_chromosomes

        mutation_indices = mutation_mask.nonzero()

        for chrom_idx, node_idx in mutation_indices:
            node = mutated_chromosomes[chrom_idx, node_idx]
            node_type = int(node[COL_NODE_TYPE].item())

            # 파라미터 부분만 0으로 초기화
            node[COL_PARAM_1:] = 0.0

            if node_type == NODE_TYPE_ACTION:
                _create_random_action_params(node, 0) # node가 1D 텐서이므로 node_idx=0
            elif node_type == NODE_TYPE_DECISION:
                # node는 1D 텐서이므로, utils 함수에 맞게 2D로 잠시 변환
                _create_random_decision_params(node.unsqueeze(0), 0, self.config)

        return mutated_chromosomes
