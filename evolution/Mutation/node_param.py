# --- START OF FILE evolution/Mutation/node_param.py ---

import torch
import random
from .base import BaseMutation
from typing import Dict, Any

from models.constants import (
    COL_NODE_TYPE, COL_PARAM_1, COL_PARAM_2, COL_PARAM_3, COL_PARAM_4,
    NODE_TYPE_DECISION, NODE_TYPE_ACTION, COMP_TYPE_FEAT_NUM, COMP_TYPE_FEAT_BOOL,
    OP_GTE, OP_LTE, 
    # [신규] 새로운 Action 상수 임포트
    ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_CLOSE_PARTIAL, 
    ACTION_ADD_POSITION, ACTION_FLIP_POSITION
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
        self.operators = [OP_GTE, OP_LTE]

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        """
        [수정] 벡터화된 연산을 사용하여 효율적으로 파라미터 변이를 적용합니다.
        Action 노드 변이 로직이 새로운 Action 체계에 맞게 수정되었습니다.
        """
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

            if node_type == NODE_TYPE_ACTION:
                # [수정] Action Type에 따라 유효한 파라미터만 변경
                action_type = int(node[COL_PARAM_1].item())

                if action_type in [ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_FLIP_POSITION]:
                    # Size(비중) 또는 Leverage 변경
                    param_to_mutate = random.choice([2, 3])
                    if param_to_mutate == 2: # Size
                        noise = (torch.randn(1) * 0.1).item()
                        node[COL_PARAM_2] = torch.clamp(node[COL_PARAM_2] + noise, 0.0, 1.0)
                    else: # Leverage
                        change = random.randint(-self.leverage_change, self.leverage_change)
                        node[COL_PARAM_3] = torch.clamp(node[COL_PARAM_3] + change, 1, 100)
                
                elif action_type in [ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION]:
                    # Ratio 또는 Add Size 변경 (PARAM_2만 유효)
                    noise = (torch.randn(1) * 0.1).item()
                    node[COL_PARAM_2] = torch.clamp(node[COL_PARAM_2] + noise, 0.0, 1.0)
                
                # ACTION_CLOSE_ALL은 변경할 파라미터가 없으므로 아무것도 하지 않음

            elif node_type == NODE_TYPE_DECISION:
                comp_type = int(node[COL_PARAM_3].item())
                
                if comp_type == COMP_TYPE_FEAT_BOOL:
                    node[COL_PARAM_4] = 1.0 - node[COL_PARAM_4]
                else:
                    param_to_mutate = random.randint(1, 2)
                    if param_to_mutate == 1:
                        current_op = int(node[COL_PARAM_2].item())
                        current_op_idx = self.operators.index(current_op)
                        new_op = self.operators[1 - current_op_idx]
                        node[COL_PARAM_2] = new_op
                    else:
                        if comp_type == COMP_TYPE_FEAT_NUM:
                            feat_idx = int(node[COL_PARAM_1].item())
                            feat_name = self.config['all_features'][feat_idx]
                            min_val, max_val = self.config['feature_num'][feat_name]

                            # [수정 시작] YAML에서 읽어온 값이 문자열일 수 있으므로 float으로 강제 변환
                            min_val_f = float(min_val)
                            max_val_f = float(max_val)
                            
                            noise_range = (max_val_f - min_val_f) * self.noise_ratio
                            noise = random.uniform(-noise_range, noise_range)
                            node[COL_PARAM_4] = torch.clamp(node[COL_PARAM_4] + noise, min_val_f, max_val_f)
                            # [수정 끝]
        
        return mutated_chromosomes