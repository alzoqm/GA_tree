import os
import sys
import math
import torch
import pandas as pd
import pytest

from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.model import GATreePop
from models import constants as C
from training.predictor import build_adjacency_list_cuda, predict_population_cuda


def _skip_if_no_cuda_ext():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available for tests.")
    try:
        import gatree_cuda_compat  # noqa: F401
    except Exception:
        pytest.skip("gatree_cuda_compat extension not available.")


def _make_empty_population(B: int, N: int) -> GATreePop:
    """Create a minimal GATreePop with a CUDA tensor; caller fills nodes."""
    pop = GATreePop(
        pop_size=B,
        max_nodes=N,
        max_depth=8,
        max_children=10**9,  # not used post-init; just a placeholder
        feature_num={},
        feature_comparison_map={},
        feature_bool=[],
        all_features=[],
    )
    # Create CUDA tensor shaped (B,N,D)
    D = int(C.NODE_INFO_DIM)
    t = torch.zeros((B, N, D), dtype=torch.float32, device='cuda')
    pop.population_tensor = t
    pop.initialized = True
    return pop


def _set_root(tree: torch.Tensor, idx: int, branch_type: int):
    tree[idx, C.COL_NODE_TYPE] = float(C.NODE_TYPE_ROOT_BRANCH)
    tree[idx, C.COL_PARENT_IDX] = -1.0
    tree[idx, C.COL_DEPTH] = 0.0
    tree[idx, C.COL_PARAM_1] = float(branch_type)


def _set_action(tree: torch.Tensor, idx: int, parent: int, action_type: int, depth: int = 1):
    tree[idx, C.COL_NODE_TYPE] = float(C.NODE_TYPE_ACTION)
    tree[idx, C.COL_PARENT_IDX] = float(parent)
    tree[idx, C.COL_DEPTH] = float(depth)
    tree[idx, C.COL_PARAM_1] = float(action_type)


def _set_decision(tree: torch.Tensor, idx: int, parent: int, depth: int = 1):
    tree[idx, C.COL_NODE_TYPE] = float(C.NODE_TYPE_DECISION)
    tree[idx, C.COL_PARENT_IDX] = float(parent)
    tree[idx, C.COL_DEPTH] = float(depth)


@pytest.mark.cuda
def test_predict_children_row_offset_per_tree():
    """
    Ensure per-tree children matrix is correctly offset in predict kernel.
    Two trees with different action types under LONG root should yield different
    predictions when positions=['LONG','LONG'].
    """
    _skip_if_no_cuda_ext()

    B, N = 2, 8
    pop = _make_empty_population(B, N)
    t = pop.population_tensor

    # Tree 0 roots
    _set_root(t[0], 0, C.ROOT_BRANCH_LONG)
    _set_root(t[0], 1, C.ROOT_BRANCH_HOLD)
    _set_root(t[0], 2, C.ROOT_BRANCH_SHORT)
    # Tree 1 roots
    _set_root(t[1], 0, C.ROOT_BRANCH_LONG)
    _set_root(t[1], 1, C.ROOT_BRANCH_HOLD)
    _set_root(t[1], 2, C.ROOT_BRANCH_SHORT)

    # Attach at least one child to every root to satisfy validate_trees
    # Tree 0: LONG->ACTION_CLOSE_ALL, HOLD->ACTION_CLOSE_PARTIAL, SHORT->ACTION_CLOSE_ALL
    _set_action(t[0], 3, parent=0, action_type=C.ACTION_CLOSE_ALL)
    _set_action(t[0], 4, parent=1, action_type=C.ACTION_CLOSE_PARTIAL)
    _set_action(t[0], 5, parent=2, action_type=C.ACTION_CLOSE_ALL)
    # Tree 1: LONG->ACTION_ADD_POSITION (different!), HOLD/SHORT arbitrary valid actions
    _set_action(t[1], 3, parent=1, action_type=C.ACTION_CLOSE_ALL)
    _set_action(t[1], 4, parent=0, action_type=C.ACTION_ADD_POSITION)  # different under LONG
    _set_action(t[1], 5, parent=2, action_type=C.ACTION_CLOSE_PARTIAL)

    # Build adjacency (no strict overflow)
    adj_off, adj_idx = build_adjacency_list_cuda(pop, strict_overflow=False)

    # Predict with LONG position for both trees
    feats = pd.Series({})  # no features used by ACTION
    positions = ['LONG', 'LONG']
    out = predict_population_cuda(pop, feats, positions, adj_off, adj_idx, device='cuda')

    # First column is action type
    a0 = int(out[0, 0].item())
    a1 = int(out[1, 0].item())
    assert a0 == int(C.ACTION_CLOSE_ALL), "Tree 0 action type mismatch"
    assert a1 == int(C.ACTION_ADD_POSITION), "Tree 1 should differ and match its own children row"
    assert a0 != a1, "Per-tree children not respected; predictions identical unexpectedly"


@pytest.mark.cuda
def test_build_adjacency_handles_large_parent_degree_without_overflow():
    """
    With strict_overflow=False, a parent having many children should not cause errors.
    Verify CSR offsets reflect the correct degree and children are present.
    """
    _skip_if_no_cuda_ext()

    B, N = 1, 80
    pop = _make_empty_population(B, N)
    t = pop.population_tensor

    # Roots
    _set_root(t[0], 0, C.ROOT_BRANCH_LONG)
    _set_root(t[0], 1, C.ROOT_BRANCH_HOLD)
    _set_root(t[0], 2, C.ROOT_BRANCH_SHORT)

    # Add many DECISION children to LONG root to create a high-degree parent
    # and give each decision a single ACTION child to satisfy leaf/action invariant.
    degree = 20
    dec_indices = list(range(3, 3 + degree))
    act_indices = list(range(3 + degree, 3 + 2 * degree))
    for di, ai in zip(dec_indices, act_indices):
        _set_decision(t[0], di, parent=0, depth=1)
        _set_action(t[0], ai, parent=di, action_type=C.ACTION_CLOSE_ALL, depth=2)

    # Also ensure HOLD/SHORT have at least one child
    _set_action(t[0], 70, parent=1, action_type=C.ACTION_CLOSE_PARTIAL)
    _set_action(t[0], 71, parent=2, action_type=C.ACTION_ADD_POSITION)

    # Build adjacency
    adj_off, adj_idx = build_adjacency_list_cuda(pop, strict_overflow=False)

    # Check shapes
    assert adj_off.shape == (B, N + 1)
    assert adj_idx.shape[0] == B

    # Degree at parent 0 should be `degree`
    deg0 = int(adj_off[0, 1].item() - adj_off[0, 0].item())
    assert deg0 == degree, f"Expected degree {degree} at root LONG, got {deg0}"

    # Children set for LONG root should contain all DECISION child indices
    start = int(adj_off[0, 0].item())
    end = int(adj_off[0, 1].item())
    got = sorted([int(x.item()) for x in adj_idx[0, start:end]])
    assert got == sorted(dec_indices), "Adjacency children mismatch for high-degree parent"
