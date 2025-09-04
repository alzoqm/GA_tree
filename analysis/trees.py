from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from models.constants import (
    COL_NODE_TYPE, COL_PARAM_1, COL_PARAM_2, COL_PARAM_3, COL_PARAM_4,
    NODE_TYPE_DECISION,
    COMP_TYPE_FEAT_NUM, COMP_TYPE_FEAT_FEAT, COMP_TYPE_FEAT_BOOL,
)


def _iter_decision_nodes(tree_tensor: torch.Tensor):
    mask = (tree_tensor[:, COL_NODE_TYPE] == NODE_TYPE_DECISION)
    idxs = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    for i in idxs:
        yield i, tree_tensor[i]


def feature_usage_counts(population_tensor: torch.Tensor,
                         all_features: List[str]) -> Dict[str, int]:
    """Count feature usage across all trees (LHS and RHS where applicable)."""
    counts: Counter = Counter()
    B = population_tensor.shape[0]
    for b in range(B):
        tree = population_tensor[b]
        for _, node in _iter_decision_nodes(tree):
            comp = int(node[COL_PARAM_3].item())
            lhs_idx = int(node[COL_PARAM_1].item())
            if 0 <= lhs_idx < len(all_features):
                counts[all_features[lhs_idx]] += 1
            if comp == COMP_TYPE_FEAT_FEAT:
                rhs_idx = int(node[COL_PARAM_4].item())
                if 0 <= rhs_idx < len(all_features):
                    counts[all_features[rhs_idx]] += 1
    return dict(counts)


def comparison_type_mix(population_tensor: torch.Tensor) -> Dict[str, int]:
    """Return counts for each comparison type across all decision nodes."""
    mix = defaultdict(int)
    B = population_tensor.shape[0]
    for b in range(B):
        tree = population_tensor[b]
        for _, node in _iter_decision_nodes(tree):
            comp = int(node[COL_PARAM_3].item())
            if comp == COMP_TYPE_FEAT_NUM:
                mix['feat_num'] += 1
            elif comp == COMP_TYPE_FEAT_FEAT:
                mix['feat_feat'] += 1
            elif comp == COMP_TYPE_FEAT_BOOL:
                mix['feat_bool'] += 1
    return dict(mix)


def feature_interaction_matrix(population_tensor: torch.Tensor,
                               all_features: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Build LHS–RHS co-occurrence matrix for FEAT–FEAT nodes.

    Returns (matrix, feature_names) where matrix[i,j] counts occurrences of
    feature_i >=/<= feature_j (direction-agnostic for counting purposes).
    """
    n = len(all_features)
    mat = np.zeros((n, n), dtype=np.int64)
    B = population_tensor.shape[0]
    for b in range(B):
        tree = population_tensor[b]
        for _, node in _iter_decision_nodes(tree):
            comp = int(node[COL_PARAM_3].item())
            if comp != COMP_TYPE_FEAT_FEAT:
                continue
            i = int(node[COL_PARAM_1].item())
            j = int(node[COL_PARAM_4].item())
            if 0 <= i < n and 0 <= j < n:
                mat[i, j] += 1
    return mat, all_features


def threshold_distributions(population_tensor: torch.Tensor,
                            all_features: List[str]) -> Dict[str, List[float]]:
    """Collect numeric thresholds per feature for FEAT–NUM nodes."""
    data: Dict[str, List[float]] = defaultdict(list)
    B = population_tensor.shape[0]
    for b in range(B):
        tree = population_tensor[b]
        for _, node in _iter_decision_nodes(tree):
            comp = int(node[COL_PARAM_3].item())
            if comp != COMP_TYPE_FEAT_NUM:
                continue
            fidx = int(node[COL_PARAM_1].item())
            thr = float(node[COL_PARAM_4].item())
            if 0 <= fidx < len(all_features):
                data[all_features[fidx]].append(thr)
    return dict(data)

