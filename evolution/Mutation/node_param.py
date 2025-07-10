# evolution/Mutation/node_param.py
import torch
import random
from .base import BaseMutation
from typing import Dict, Any

# model.py에서 상수 임포트
from models.model import (
    COL_NODE_TYPE, COL_PARAM_1, COL_PARAM_2, COL_PARAM_3, COL_PARAM_4,
    NODE_TYPE_DECISION, NODE_TYPE_ACTION, COMP_TYPE_FEAT_NUM,
    OP_GT, OP_LT, OP_EQ, POS_TYPE_LONG, POS_TYPE_SHORT
)

class NodeParamMutation(BaseMutation):
    """
    노드의 파라미터를 미세 조정하는 변이. (값 기반 변이)
    트리 구조는 변경하지 않고 기존 해의 근방을 탐색합니다.
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None,
                 noise_ratio: float = 0.1, leverage_change: int = 5):
        """
        Args:
            prob (float): 각 노드에 변이가 적용될 확률.
            config (Dict): GATree 생성에 사용된 설정(feature_num, feature_pair 등).
            noise_ratio (float): 숫자 비교값에 적용될 노이즈의 크기 비율.
            leverage_change (int): 레버리지 값의 최대 변경폭.
        """
        super().__init__(prob)
        if config is None:
            raise ValueError("NodeParamMutation requires a 'config' dictionary.")
        self.config = config
        self.noise_ratio = noise_ratio
        self.leverage_change = leverage_change
        self.operators = [OP_GT, OP_LT, OP_EQ]

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        """
        벡터화된 연산을 사용하여 효율적으로 파라미터 변이를 적용합니다.
        """
        mutated_chromosomes = chromosomes.clone()
        pop_size, max_nodes, _ = chromosomes.shape
        
        # 1. 변이 대상 노드 마스크 생성
        types = chromosomes[:, :, COL_NODE_TYPE]
        # 변이 가능한 노드 타입: Decision 또는 Action
        mutable_type_mask = (types == NODE_TYPE_DECISION) | (types == NODE_TYPE_ACTION)
        # 각 노드에 대한 확률적 마스크
        prob_mask = torch.rand(pop_size, max_nodes, device=chromosomes.device) < self.prob
        # 최종 변이 대상 마스크
        mutation_mask = mutable_type_mask & prob_mask
        
        # 변이 대상이 없으면 조기 종료
        if not mutation_mask.any():
            return mutated_chromosomes

        # 2. 변이 대상 인덱스 추출
        mutation_indices = mutation_mask.nonzero() # shape: (N, 2)
        
        # 3. 선택적 순회를 통한 변이 적용
        for chrom_idx, node_idx in mutation_indices:
            node = mutated_chromosomes[chrom_idx, node_idx]
            node_type = int(node[COL_NODE_TYPE].item())

            if node_type == NODE_TYPE_ACTION:
                param_to_mutate = random.randint(1, 3)
                if param_to_mutate == 1: # 포지션 토글
                    node[COL_PARAM_1] = POS_TYPE_SHORT if node[COL_PARAM_1] == POS_TYPE_LONG else POS_TYPE_LONG
                elif param_to_mutate == 2: # 비중
                    noise = (torch.randn(1) * 0.1).item()
                    node[COL_PARAM_2] = torch.clamp(node[COL_PARAM_2] + noise, 0.0, 1.0)
                else: # 레버리지
                    change = random.randint(-self.leverage_change, self.leverage_change)
                    node[COL_PARAM_3] = torch.clamp(node[COL_PARAM_3] + change, 1, 100)

            elif node_type == NODE_TYPE_DECISION:
                param_to_mutate = random.randint(1, 2)
                if param_to_mutate == 1: # 연산자 변경
                    current_op = int(node[COL_PARAM_2].item())
                    new_op = random.choice([op for op in self.operators if op != current_op])
                    node[COL_PARAM_2] = new_op
                else: # 값/피처 변경
                    comp_type = int(node[COL_PARAM_3].item())
                    if comp_type == COMP_TYPE_FEAT_NUM:
                        # 숫자 값 변경
                        feat_idx = int(node[COL_PARAM_1].item())
                        feat_name = self.config['all_features'][feat_idx]
                        min_val, max_val = self.config['feature_num'][feat_name]
                        noise = (random.uniform(-1, 1) * self.noise_ratio * (max_val - min_val))
                        node[COL_PARAM_4] = torch.clamp(node[COL_PARAM_4] + noise, min_val, max_val)
        
        return mutated_chromosomes
