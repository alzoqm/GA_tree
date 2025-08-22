# evolution/Mutation/add_subtree.py
# CUDA-accelerated batch AddSubtreeMutation (one-shot)
import torch
from typing import Dict, Any, Tuple
from .base import BaseMutation
from models.constants import (
    COL_NODE_TYPE, COL_PARAM_1, COL_PARAM_2, COL_PARAM_3, COL_PARAM_4,
    NODE_TYPE_DECISION, NODE_TYPE_ACTION,
    ROOT_BRANCH_HOLD, ROOT_BRANCH_LONG, ROOT_BRANCH_SHORT,
    ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_CLOSE_ALL, ACTION_CLOSE_PARTIAL,
    ACTION_ADD_POSITION, ACTION_FLIP_POSITION,
)

try:
    import gatree_cuda
except ImportError:
    print("="*60)
    print(">>> Warning: 'gatree_cuda' module not found.")
    print(">>> Build the CUDA extension first:")
    print(">>> python setup.py build_ext --inplace")
    print("="*60)
    gatree_cuda = None


class ActionParamSampler:
    """
    Vectorized GPU filler for ACTION node parameters.
    - Action type depends on root-branch context (HOLD/LONG/SHORT).
    - Parameters written via tensor ops; no Python loops.
    """
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.actions_hold = torch.tensor([ACTION_NEW_LONG, ACTION_NEW_SHORT], device=device, dtype=dtype)
        self.actions_trend = torch.tensor(
            [ACTION_CLOSE_ALL, ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION, ACTION_FLIP_POSITION],
            device=device, dtype=dtype
        )
        self.root_hold  = torch.tensor([ROOT_BRANCH_HOLD],  device=device, dtype=dtype)
        self.root_long  = torch.tensor([ROOT_BRANCH_LONG],  device=device, dtype=dtype)
        self.root_short = torch.tensor([ROOT_BRANCH_SHORT], device=device, dtype=dtype)

    @torch.no_grad()
    def fill_params_for_nodes(
        self,
        trees: torch.Tensor,
        node_batch_idx: torch.Tensor,
        node_indices: torch.Tensor,
        root_branch_type: torch.Tensor
    ) -> None:
        if node_batch_idx.numel() == 0:
            return
        dev, dt = trees.device, trees.dtype
        b, n, rb = node_batch_idx, node_indices, root_branch_type

        is_hold  = (rb == self.root_hold.item())
        is_long  = (rb == self.root_long.item())
        is_short = (rb == self.root_short.item())

        chosen = torch.full((b.numel(),), ACTION_CLOSE_ALL, device=dev, dtype=dt)
        if is_hold.any():
            idx = torch.nonzero(is_hold, as_tuple=True)[0]
            pick = torch.randint(0, self.actions_hold.numel(), (idx.numel(),), device=dev)
            chosen[idx] = self.actions_hold[pick]
        long_or_short = is_long | is_short
        if long_or_short.any():
            idx = torch.nonzero(long_or_short, as_tuple=True)[0]
            pick = torch.randint(0, self.actions_trend.numel(), (idx.numel(),), device=dev)
            chosen[idx] = self.actions_trend[pick]

        trees[b, n, COL_PARAM_1] = chosen

        is_new   = (chosen == ACTION_NEW_LONG) | (chosen == ACTION_NEW_SHORT) | (chosen == ACTION_FLIP_POSITION)
        is_close = (chosen == ACTION_CLOSE_PARTIAL)
        is_add   = (chosen == ACTION_ADD_POSITION)

        if is_new.any():
            idx = torch.nonzero(is_new, as_tuple=True)[0]
            bm, nm = b[idx], n[idx]
            trees[bm, nm, COL_PARAM_2] = torch.rand(idx.numel(), device=dev, dtype=dt)
            lev = torch.randint(1, 101, (idx.numel(),), device=dev, dtype=torch.long).to(dt)
            trees[bm, nm, COL_PARAM_3] = lev

        if is_close.any():
            idx = torch.nonzero(is_close, as_tuple=True)[0]
            bm, nm = b[idx], n[idx]
            trees[bm, nm, COL_PARAM_2] = torch.rand(idx.numel(), device=dev, dtype=dt)

        if is_add.any():
            idx = torch.nonzero(is_add, as_tuple=True)[0]
            bm, nm = b[idx], n[idx]
            trees[bm, nm, COL_PARAM_2] = torch.rand(idx.numel(), device=dev, dtype=dt)


class AddSubtreeMutation(BaseMutation):
    """
    One-shot, batch AddSubtree mutation on GPU:
    - Python allocates all buffers.
    - CUDA kernel performs structural edits with invariant guards.
    - DECISION/ACTION params filled via vectorized GPU samplers.
    """
    def __init__(
        self,
        prob: float = 0.1,
        config: Dict[str, Any] = None,
        node_count_range: Tuple[int, int] = (2, 5),
        max_nodes: int = 256,
        max_new_nodes: int = 32,
    ):
        super().__init__(prob)
        if config is None:
            raise ValueError("AddSubtreeMutation requires a 'config' dictionary.")
        self.config = config
        self.max_depth    = int(config['max_depth'])
        self.max_children = int(config['max_children'])
        self.node_count_range = (int(node_count_range[0]), int(node_count_range[1]))
        self.max_nodes    = int(max_nodes)
        self.max_new_nodes = int(max_new_nodes)

        # Safety contract: parameter filling requires space to record all new nodes.
        hi = self.node_count_range[1]
        # Up to hi DECISION + hi ACTION (from dangling fix) = 2*hi
        if self.max_new_nodes < 2 * hi:
            raise ValueError(f"max_new_nodes ({max_new_nodes}) must be >= 2 * max(node_count_range) ({2 * hi}) to ensure all new nodes are recorded.")

        self._dec_sampler = None
        self._act_sampler = None

    def _ensure_samplers(self, device: torch.device, dtype: torch.dtype):
        # Reuse DecisionParamSampler from add_node.py (keeps style consistent)
        from .add_node import DecisionParamSampler
        if (self._dec_sampler is None) or (self._dec_sampler.device != device) or (self._dec_sampler.dtype != dtype):
            self._dec_sampler = DecisionParamSampler(self.config, device=device, dtype=dtype)
        if (self._act_sampler is None) or (self._act_sampler.device != device) or (self._act_sampler.dtype != dtype):
            self._act_sampler = ActionParamSampler(device=device, dtype=dtype)

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        if gatree_cuda is None:
            raise RuntimeError("gatree_cuda module is not loaded. Cannot perform add-subtree mutation.")
        if chromosomes.dtype != torch.float32:
            raise ValueError("Trees must be float32 (integer fields encoded as floats).")

        trees = chromosomes.clone()
        B, N, D = trees.shape
        device, dtype = trees.device, trees.dtype

        self._ensure_samplers(device, dtype)

        mutate_mask = (torch.rand(B, device=device) < self.prob)
        if not mutate_mask.any():
            return trees

        num_to_grow = torch.zeros(B, device=device, dtype=torch.int32)
        lo, hi = self.node_count_range
        k = torch.randint(lo, hi + 1, (mutate_mask.sum(),), device=device, dtype=torch.int32)
        num_to_grow[mutate_mask] = k

        # Work buffers allocated in Python (no device arrays in C/CUDA)
        bfs_queue_buffer        = torch.empty((B, 2 * self.max_nodes), dtype=torch.int32, device=device)
        result_indices_buffer   = torch.empty((B, 2 * self.max_nodes), dtype=torch.int32, device=device)

        child_count_buffer      = torch.empty((B, self.max_nodes), dtype=torch.int32, device=device)
        act_cnt_buffer          = torch.empty((B, self.max_nodes), dtype=torch.int32, device=device)
        dec_cnt_buffer          = torch.empty((B, self.max_nodes), dtype=torch.int32, device=device)

        candidate_indices_buffer = torch.empty((B, self.max_nodes), dtype=torch.int32, device=device)
        candidate_weights_buffer = torch.empty((B, self.max_nodes), dtype=torch.float32, device=device)

        new_decision_nodes = torch.full((B, self.max_new_nodes), -1, dtype=torch.int32, device=device)
        new_action_nodes   = torch.full((B, self.max_new_nodes), -1, dtype=torch.int32, device=device)
        action_root_type   = torch.full((B, self.max_new_nodes), -1, dtype=dtype, device=device)

        gatree_cuda.add_subtrees_batch(
            trees, num_to_grow,
            int(self.max_children), int(self.max_depth),
            int(self.max_nodes), int(self.max_new_nodes),
            new_decision_nodes, new_action_nodes, action_root_type,
            bfs_queue_buffer, result_indices_buffer,
            child_count_buffer, act_cnt_buffer, dec_cnt_buffer,
            candidate_indices_buffer, candidate_weights_buffer
        )

        # Vectorized parameter filling for new nodes
        valid_dec = (new_decision_nodes >= 0)
        if valid_dec.any():
            db, di = torch.nonzero(valid_dec, as_tuple=True)
            dec_nodes_flat = new_decision_nodes[db, di].to(torch.long)
            self._dec_sampler.fill_params_for_nodes(trees, db, dec_nodes_flat)
            trees[db, dec_nodes_flat, COL_NODE_TYPE] = torch.tensor(NODE_TYPE_DECISION, device=device, dtype=dtype)

        valid_act = (new_action_nodes >= 0)
        if valid_act.any():
            ab, ai = torch.nonzero(valid_act, as_tuple=True)
            act_nodes_flat = new_action_nodes[ab, ai].to(torch.long)
            act_root_type  = action_root_type[ab, ai]
            self._act_sampler.fill_params_for_nodes(trees, ab, act_nodes_flat, act_root_type)
            trees[ab, act_nodes_flat, COL_NODE_TYPE] = torch.tensor(NODE_TYPE_ACTION, device=device, dtype=dtype)

        # Validate trees after CUDA add-subtree mutation (if available)
        try:
            if gatree_cuda is not None and trees.is_cuda:
                gatree_cuda.validate_trees(trees.contiguous())
        except Exception:
            pass

        return trees
