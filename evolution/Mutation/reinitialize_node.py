# --- START OF FILE evolution/Mutation/reinitialize_node.py ---

# evolution/Mutation/reinitialize_node.py (수정된 최종본)
import torch
from .base import BaseMutation
from .utils import _create_random_action_params, _create_random_decision_params
from typing import Dict, Any

# model.py에서 상수 임포트
from models.constants import (
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
            tree = mutated_chromosomes[chrom_idx] 
            node = tree[node_idx]
            
            node_type = int(node[COL_NODE_TYPE].item())

            # 파라미터 부분만 0으로 초기화
            node[COL_PARAM_1:] = 0.0

            if node_type == NODE_TYPE_ACTION:
                # 수정된 _create_random_action_params 함수를 호출하여
                # 문맥에 맞는 새로운 랜덤 파라미터로 교체합니다.
                _create_random_action_params(tree, node_idx)
            elif node_type == NODE_TYPE_DECISION:
                _create_random_decision_params(tree, node_idx, self.config)

        return mutated_chromosomes