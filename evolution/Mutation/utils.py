# --- START OF FILE evolution/Mutation/utils.py ---

import torch
import random
from typing import List, Dict, Any

from models.constants import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, COL_PARAM_1, COL_PARAM_2,
    COL_PARAM_3, COL_PARAM_4, NODE_TYPE_UNUSED, NODE_TYPE_DECISION,
    NODE_TYPE_ACTION, COMP_TYPE_FEAT_NUM, COMP_TYPE_FEAT_FEAT, COMP_TYPE_FEAT_BOOL,
    OP_GTE, OP_LTE, 
    # [신규] 새로운 Action 상수 임포트
    ROOT_BRANCH_HOLD, ROOT_BRANCH_LONG, ROOT_BRANCH_SHORT,
    ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_CLOSE_ALL, ACTION_CLOSE_PARTIAL,
    ACTION_ADD_POSITION, ACTION_FLIP_POSITION
)

def find_subtree_nodes(tree_tensor: torch.Tensor, root_idx: int) -> List[int]:
    """
    주어진 루트 노드에서 시작하는 서브트리에 포함된 모든 노드의 인덱스를 BFS로 찾습니다.
    """
    if tree_tensor[root_idx, COL_NODE_TYPE] == NODE_TYPE_UNUSED:
        return []
    
    q = [root_idx]
    visited = {root_idx}
    head = 0
    while head < len(q):
        current_idx = q[head]
        head += 1
        
        children_mask = tree_tensor[:, COL_PARENT_IDX] == current_idx
        children_indices = children_mask.nonzero(as_tuple=True)[0]
        
        for child_idx_tensor in children_indices:
            child_idx = child_idx_tensor.item()
            if child_idx not in visited:
                visited.add(child_idx)
                q.append(child_idx)
    return q

def update_subtree_depth(tree_tensor: torch.Tensor, root_idx: int, delta: int):
    """
    주어진 루트 노드와 그 모든 자손 노드의 깊이를 delta만큼 조정합니다.
    """
    nodes_to_update = find_subtree_nodes(tree_tensor, root_idx)
    if not nodes_to_update:
        return
        
    indices_tensor = torch.tensor(nodes_to_update, dtype=torch.long, device=tree_tensor.device)
    tree_tensor[indices_tensor, COL_DEPTH] += delta

def get_subtree_max_depth(tree_tensor: torch.Tensor, root_idx: int) -> int:
    """서브트리의 최대 절대 깊이를 반환합니다."""
    subtree_nodes = find_subtree_nodes(tree_tensor, root_idx)
    if not subtree_nodes:
        return int(tree_tensor[root_idx, COL_DEPTH].item())
    
    indices_tensor = torch.tensor(subtree_nodes, dtype=torch.long, device=tree_tensor.device)
    max_depth = tree_tensor[indices_tensor, COL_DEPTH].max().item()
    return int(max_depth)

def find_empty_slots(tree_tensor: torch.Tensor, count: int = 1) -> List[int]:
    """
    트리 텐서에서 비어있는 슬롯(UNUSED)의 인덱스를 'count'개 만큼 찾아 반환합니다.
    """
    empty_mask = tree_tensor[:, COL_NODE_TYPE] == NODE_TYPE_UNUSED
    empty_indices = empty_mask.nonzero(as_tuple=True)[0]
    if len(empty_indices) < count:
        return []
    return empty_indices[:count].tolist()

def _create_random_action_params(tree_tensor: torch.Tensor, node_idx: int):
    """[전면 수정] 주어진 Action 노드에 문맥에 맞는 랜덤 파라미터를 채웁니다."""
    parent_idx = int(tree_tensor[node_idx, COL_PARENT_IDX].item())
    if parent_idx == -1: return # 부모가 없으면(이론상 불가) 아무것도 안 함

    # 1. 부모로부터 루트 분기 타입을 결정
    current_idx = parent_idx
    while tree_tensor[current_idx, COL_PARENT_IDX].item() != -1:
        current_idx = int(tree_tensor[current_idx, COL_PARENT_IDX].item())
    root_branch_type = tree_tensor[current_idx, COL_PARAM_1].item()

    # 2. 루트 분기 타입(문맥)에 따라 가능한 Action 리스트를 정의
    if root_branch_type == ROOT_BRANCH_HOLD:
        possible_actions = [ACTION_NEW_LONG, ACTION_NEW_SHORT]
    elif root_branch_type == ROOT_BRANCH_LONG:
        possible_actions = [ACTION_CLOSE_ALL, ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION, ACTION_FLIP_POSITION]
    elif root_branch_type == ROOT_BRANCH_SHORT:
        possible_actions = [ACTION_CLOSE_ALL, ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION, ACTION_FLIP_POSITION]
    else:
        possible_actions = []

    if not possible_actions:
        # 생성할 유효한 액션이 없으면 노드를 UNUSED로 변경 (호출한 쪽에서 처리)
        tree_tensor[node_idx, COL_NODE_TYPE] = NODE_TYPE_UNUSED
        return

    # 3. 선택된 Action 타입에 따라 파라미터를 랜덤하게 생성
    chosen_action = random.choice(possible_actions)
    tree_tensor[node_idx, COL_PARAM_1] = chosen_action

    if chosen_action in [ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_FLIP_POSITION]:
        tree_tensor[node_idx, COL_PARAM_2] = random.random()  # Size or New Size
        tree_tensor[node_idx, COL_PARAM_3] = random.randint(1, 100) # Leverage
    elif chosen_action == ACTION_CLOSE_PARTIAL:
        tree_tensor[node_idx, COL_PARAM_2] = random.random()  # Close Ratio
    elif chosen_action == ACTION_ADD_POSITION:
        tree_tensor[node_idx, COL_PARAM_2] = random.random()  # Add Size Ratio

def _create_random_decision_params(tree_tensor: torch.Tensor, node_idx: int, config: Dict[str, Any]):
    """주어진 Decision 노드에 랜덤 파라미터를 채웁니다."""
    comp_type_choices = [COMP_TYPE_FEAT_NUM]
    if config.get('feature_comparison_map'):
        comp_type_choices.append(COMP_TYPE_FEAT_FEAT)
    if config.get('feature_bool'):
        comp_type_choices.append(COMP_TYPE_FEAT_BOOL)

    comp_type = random.choice(comp_type_choices)
    tree_tensor[node_idx, COL_PARAM_3] = comp_type

    if comp_type == COMP_TYPE_FEAT_NUM or comp_type == COMP_TYPE_FEAT_FEAT:
        tree_tensor[node_idx, COL_PARAM_2] = random.choice([OP_GTE, OP_LTE])

    all_features = config['all_features']

    if comp_type == COMP_TYPE_FEAT_NUM:
        feature_num = config['feature_num']
        feat_name = random.choice(list(feature_num.keys()))
        feat_idx = all_features.index(feat_name)
        
        min_val, max_val = feature_num[feat_name]
        
        # [수정 시작] YAML에서 읽어온 값이 문자열일 수 있으므로 float으로 강제 변환
        min_val_f = float(min_val)
        max_val_f = float(max_val)
        comp_val = random.uniform(min_val_f, max_val_f)
        # [수정 끝]

        tree_tensor[node_idx, COL_PARAM_1] = feat_idx
        tree_tensor[node_idx, COL_PARAM_4] = comp_val
        
    elif comp_type == COMP_TYPE_FEAT_FEAT:
        feature_comparison_map = config['feature_comparison_map']
        possible_feat1 = [k for k, v in feature_comparison_map.items() if v]
        
        if not possible_feat1:
            tree_tensor[node_idx, COL_PARAM_3] = COMP_TYPE_FEAT_NUM
            feature_num = config['feature_num']
            feat_name = random.choice(list(feature_num.keys()))
            feat_idx = all_features.index(feat_name)
            min_val, max_val = feature_num[feat_name]
            
            # [수정 시작] YAML에서 읽어온 값이 문자열일 수 있으므로 float으로 강제 변환
            min_val_f = float(min_val)
            max_val_f = float(max_val)
            comp_val = random.uniform(min_val_f, max_val_f)
            # [수정 끝]

            tree_tensor[node_idx, COL_PARAM_1] = feat_idx
            tree_tensor[node_idx, COL_PARAM_4] = comp_val
            return

        feat1_name = random.choice(possible_feat1)
        feat2_name = random.choice(feature_comparison_map[feat1_name])
        
        feat1_idx = all_features.index(feat1_name)
        feat2_idx = all_features.index(feat2_name)
        
        tree_tensor[node_idx, COL_PARAM_1] = feat1_idx
        tree_tensor[node_idx, COL_PARAM_4] = feat2_idx

    elif comp_type == COMP_TYPE_FEAT_BOOL:
        feature_bool = config['feature_bool']
        feat_name = random.choice(feature_bool)
        feat_idx = all_features.index(feat_name)
        comp_val = random.choice([0.0, 1.0])

        tree_tensor[node_idx, COL_PARAM_1] = feat_idx
        tree_tensor[node_idx, COL_PARAM_4] = comp_val

def create_random_node(tree_tensor: torch.Tensor, parent_idx: int, node_type: int, config: Dict[str, Any]) -> int:
    """
    지정된 부모 아래에 특정 타입의 랜덤 노드를 생성하고 새 노드의 인덱스를 반환합니다.
    """
    empty_slots = find_empty_slots(tree_tensor, 1)
    if not empty_slots:
        return -1
    new_node_idx = empty_slots[0]
    
    parent_depth = tree_tensor[parent_idx, COL_DEPTH].item()
    
    tree_tensor[new_node_idx, COL_NODE_TYPE] = node_type
    tree_tensor[new_node_idx, COL_PARENT_IDX] = parent_idx
    tree_tensor[new_node_idx, COL_DEPTH] = parent_depth + 1
    
    if node_type == NODE_TYPE_ACTION:
        _create_random_action_params(tree_tensor, new_node_idx)
    elif node_type == NODE_TYPE_DECISION:
        _create_random_decision_params(tree_tensor, new_node_idx, config)
        
    return new_node_idx