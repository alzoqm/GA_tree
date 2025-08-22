# evolution/Mutation/delete_subtree.py
import torch
from typing import Dict, Any
from .base import BaseMutation
from models.constants import (
    COL_NODE_TYPE, COL_PARENT_IDX, NODE_TYPE_UNUSED,
)

try:
    import gatree_cuda
except ImportError:
    raise RuntimeError(
        "gatree_cuda module not built. Run: python setup.py build_ext --inplace"
    )


class DeleteSubtreeMutation(BaseMutation):
    """
    CUDA-accelerated batch Delete-Subtree Mutation (one-shot).
    - Single kernel per batch; Python only allocates buffers and applies masks.
    - R1–R3 repairs + G1–G2 guards guarantee structural invariants by construction.
    - Supports setting parent_idx=-1 for UNUSED nodes as requested.
    """

    def __init__(
        self,
        prob: float = 0.1,
        config: Dict[str, Any] = None,
        max_nodes: int = 256,
        alpha: float = 1.0,
        ensure_action_left: bool = True,
        set_unused_parent_idx: bool = True,
    ):
        super().__init__(prob)
        if config is None:
            raise ValueError("DeleteSubtreeMutation requires a 'config' dictionary.")
        self.config = config
        self.max_nodes = int(max_nodes)
        self.alpha = float(alpha)
        self.ensure_action_left = bool(ensure_action_left)
        self.set_unused_parent_idx = bool(set_unused_parent_idx)

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        if chromosomes.dtype != torch.float32:
            raise ValueError("Trees must be float32 (integer fields encoded as floats).")
        if not chromosomes.is_cuda:
            raise ValueError("Trees must be on CUDA device.")

        trees = chromosomes.clone().contiguous()
        B, N, D = trees.shape
        if N > self.max_nodes:
            raise ValueError(f"max_nodes({self.max_nodes}) < N({N}). Increase max_nodes.")

        device = trees.device

        mutate_mask = (torch.rand(B, device=device) < self.prob).to(torch.int32)
        if mutate_mask.sum().item() == 0:
            return trees

        # ---- Work buffers (all on device) ----
        child_count_buffer       = torch.empty((B, N), dtype=torch.int32, device=device)
        act_cnt_buffer           = torch.empty((B, N), dtype=torch.int32, device=device)
        dec_cnt_buffer           = torch.empty((B, N), dtype=torch.int32, device=device)

        candidate_indices_buffer = torch.empty((B, N), dtype=torch.int32, device=device)
        candidate_weights_buffer = torch.empty((B, N), dtype=torch.float32, device=device)

        bfs_queue_buffer         = torch.empty((B, 2 * N), dtype=torch.int32, device=device)
        result_indices_buffer    = torch.empty((B, 2 * N), dtype=torch.int32, device=device)

        deletion_mask_buffer     = torch.zeros((B, N), dtype=torch.int32, device=device)
        repair_mask_buffer       = torch.zeros((B, N), dtype=torch.int32, device=device)
        chosen_roots_buffer      = torch.full((B,), -1, dtype=torch.int32, device=device)

        gatree_cuda.delete_subtrees_batch(
            trees,
            mutate_mask,
            int(self.max_nodes),
            float(self.alpha),
            int(self.ensure_action_left),
            child_count_buffer,
            act_cnt_buffer,
            dec_cnt_buffer,
            candidate_indices_buffer,
            candidate_weights_buffer,
            bfs_queue_buffer,
            result_indices_buffer,
            deletion_mask_buffer,
            repair_mask_buffer,
            chosen_roots_buffer
        )

        # ---- One-shot vectorized deletion with optional parent_idx reset ----
        final_delete = (deletion_mask_buffer | repair_mask_buffer) > 0
        if final_delete.any():
            b_idx, n_idx = torch.nonzero(final_delete, as_tuple=True)
            trees[b_idx, n_idx, :] = 0.0
            trees[b_idx, n_idx, COL_NODE_TYPE] = trees.new_tensor(NODE_TYPE_UNUSED)
            
            # Optional: Set parent_idx to -1 for UNUSED nodes (as requested by user)
            if self.set_unused_parent_idx:
                trees[b_idx, n_idx, COL_PARENT_IDX] = -1.0

        # Validate trees after CUDA delete-subtree mutation (if available)
        try:
            if gatree_cuda is not None and trees.is_cuda:
                gatree_cuda.validate_trees(trees.contiguous())
                print('complete delete subtree mutation')
        except Exception:
            import traceback
            raise RuntimeError(f"gatree_cuda.validate_trees failed after delete_subtree mutation.\n{traceback.format_exc()}")

        return trees
