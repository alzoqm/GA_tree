import torch
import random
from typing import List, Dict, Any

from models.model import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, COL_PARAM_1, COL_PARAM_2,
    COL_PARAM_3, COL_PARAM_4, NODE_TYPE_UNUSED, NODE_TYPE_DECISION,
    NODE_TYPE_ACTION, COMP_TYPE_FEAT_NUM, COMP_TYPE_FEAT_FEAT, COMP_TYPE_FEAT_BOOL,
    OP_GTE, OP_LTE, POS_TYPE_LONG, POS_TYPE_SHORT
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
    """주어진 Action 노드에 랜덤 파라미터를 채웁니다."""
    tree_tensor[node_idx, COL_PARAM_1] = random.choice([POS_TYPE_LONG, POS_TYPE_SHORT])
    tree_tensor[node_idx, COL_PARAM_2] = random.random()
    tree_tensor[node_idx, COL_PARAM_3] = random.randint(1, 100)

def _create_random_decision_params(tree_tensor: torch.Tensor, node_idx: int, config: Dict[str, Any]):
    """[수정] 주어진 Decision 노드에 랜덤 파라미터를 채웁니다."""
    comp_type_choices = [COMP_TYPE_FEAT_NUM, COMP_TYPE_FEAT_FEAT]
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
        comp_val = random.uniform(min_val, max_val)
        
        tree_tensor[node_idx, COL_PARAM_1] = feat_idx
        tree_tensor[node_idx, COL_PARAM_4] = comp_val
        
    elif comp_type == COMP_TYPE_FEAT_FEAT:
        feature_pair = config['feature_pair']
        feat1_name, feat2_name = random.sample(feature_pair, 2)
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