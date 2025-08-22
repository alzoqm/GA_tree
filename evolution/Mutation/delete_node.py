# evolution/Mutation/delete_node.py
# CUDA-accelerated batch DeleteNodeMutation implementation (one-shot)
import torch
from typing import Dict, Any
from .base import BaseMutation

from models.constants import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH,
    NODE_TYPE_UNUSED, NODE_TYPE_DECISION, NODE_TYPE_ROOT_BRANCH, NODE_TYPE_ACTION,
)

try:
    import gatree_cuda
except ImportError:
    print("=" * 60)
    print(">>> Warning: 'gatree_cuda' module not found.")
    print(">>> Build the CUDA extension first:")
    print(">>> python setup.py build_ext --inplace")
    print("=" * 60)
    gatree_cuda = None


class DeleteNodeMutation(BaseMutation):
    """
    Batch delete-node mutation (GPU).
    All invariants are guarded inside the CUDA kernel. Python allocates work buffers and
    triggers a single kernel call.
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None,
                 max_delete_nodes: int = 5, max_nodes: int = 256):
        super().__init__(prob)
        if config is None:
            raise ValueError("DeleteNodeMutation requires a 'config' dictionary.")
        if not isinstance(max_delete_nodes, int) or max_delete_nodes < 1:
            raise ValueError("max_delete_nodes must be a positive integer.")

        self.config = config
        self.max_children = int(config['max_children'])
        self.max_depth    = int(config['max_depth'])
        self.max_nodes    = int(max_nodes)
        self.max_delete_nodes = int(max_delete_nodes)

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        if gatree_cuda is None:
            raise RuntimeError("gatree_cuda module is not loaded. Cannot perform delete-node mutation.")

        trees = chromosomes.clone()  # (B, max_nodes, node_dim)
        B, N, D = trees.shape
        device = trees.device
        dtype  = trees.dtype

        mutate_mask = (torch.rand(B, device=device) < self.prob)
        if not mutate_mask.any():
            return trees

        num_to_delete = torch.zeros(B, device=device, dtype=torch.int32)
        k = torch.randint(1, self.max_delete_nodes + 1, (mutate_mask.sum(),), device=device, dtype=torch.int32)
        num_to_delete[mutate_mask] = k

        # ---- Work buffers (ALL allocated in Python; no device arrays in kernels) ----
        bfs_queue_buffer        = torch.empty((B, 2 * self.max_nodes), dtype=torch.int32, device=device)
        result_indices_buffer   = torch.empty((B, 2 * self.max_nodes), dtype=torch.int32, device=device)

        child_count_buffer      = torch.empty((B, self.max_nodes), dtype=torch.int32, device=device)
        act_cnt_buffer          = torch.empty((B, self.max_nodes), dtype=torch.int32, device=device)
        dec_cnt_buffer          = torch.empty((B, self.max_nodes), dtype=torch.int32, device=device)

        candidate_indices_buffer  = torch.empty((B, self.max_nodes), dtype=torch.int32, device=device)
        candidate_weights_buffer  = torch.empty((B, self.max_nodes), dtype=torch.float32, device=device)

        deleted_nodes = torch.full((B, self.max_delete_nodes), -1, dtype=torch.int32, device=device)

        gatree_cuda.delete_nodes_batch(
            trees,
            num_to_delete,
            int(self.max_children),
            int(self.max_depth),
            int(self.max_nodes),
            int(self.max_delete_nodes),
            deleted_nodes,
            bfs_queue_buffer,
            result_indices_buffer,
            child_count_buffer,
            act_cnt_buffer,
            dec_cnt_buffer,
            candidate_indices_buffer,
            candidate_weights_buffer
        )

        # Optional: develop-time validator (comment out in production)
        # valid = _validate_tree_constraints(trees)
        # assert valid.all(), "Post-delete invariants violated"

        # Validate trees after CUDA delete-node mutation (if available)
        try:
            if gatree_cuda is not None and trees.is_cuda:
                gatree_cuda.validate_trees(trees.contiguous())
                print('complete delete node mutation')
        except Exception:
            import traceback
            raise RuntimeError(f"gatree_cuda.validate_trees failed after delete_node mutation.\n{traceback.format_exc()}")

        return trees


# Optional vector validator for development/debugging only.
# NOTE: Keep it local to this file to avoid import cycles.
def _validate_tree_constraints(trees: torch.Tensor) -> torch.Tensor:
    B, N, D = trees.shape
    dev = trees.device
    t  = trees[..., COL_NODE_TYPE].long()
    p  = trees[..., COL_PARENT_IDX].long()
    d  = trees[..., COL_DEPTH]
    used = (t != NODE_TYPE_UNUSED)

    edge_mask = (p >= 0) & used
    eb, ei = torch.nonzero(edge_mask, as_tuple=True)
    ep = p[eb, ei]

    Z = B * N
    flat_parent = eb * N + ep
    ones = torch.ones_like(flat_parent, dtype=torch.float32, device=dev)

    child_cnt = torch.zeros((B, N), device=dev, dtype=torch.float32).view(-1)
    child_cnt.index_add_(0, flat_parent, ones)
    child_cnt = child_cnt.view(B, N)

    act_edge = edge_mask & (t == NODE_TYPE_ACTION)
    ab, ai = torch.nonzero(act_edge, as_tuple=True); ap = p[ab, ai]
    flat_parent_act = ab * N + ap
    act_cnt = torch.zeros((B, N), device=dev, dtype=torch.float32).view(-1)
    act_cnt.index_add_(0, flat_parent_act, torch.ones_like(flat_parent_act, dtype=torch.float32, device=dev))
    act_cnt = act_cnt.view(B, N)

    dec_edge = edge_mask & (t == NODE_TYPE_DECISION)
    db, di = torch.nonzero(dec_edge, as_tuple=True); dp = p[db, di]
    flat_parent_dec = db * N + dp
    dec_cnt = torch.zeros((B, N), device=dev, dtype=torch.float32).view(-1)
    dec_cnt.index_add_(0, flat_parent_dec, torch.ones_like(flat_parent_dec, dtype=torch.float32, device=dev))
    dec_cnt = dec_cnt.view(B, N)

    mix_violation   = (act_cnt > 0) & (dec_cnt > 0)
    leaf_violation  = (child_cnt == 0) & used & (t != NODE_TYPE_ACTION)
    multi_act_viols = (act_cnt > 1)
    single_act_par  = (act_cnt > 0) & (child_cnt != 1)
    inter_violation = (child_cnt > 0) & used & (t != NODE_TYPE_ROOT_BRANCH) & (t != NODE_TYPE_DECISION)
    action_has_kids = (t == NODE_TYPE_ACTION) & (child_cnt > 0)

    pd = torch.zeros_like(p, dtype=d.dtype); pd[edge_mask] = d[eb, ep]
    depth_violation = torch.zeros((B, N), device=dev, dtype=torch.bool)
    depth_violation[edge_mask] = (d[edge_mask] != (pd[edge_mask] + 1))

    root_mask = (t == NODE_TYPE_ROOT_BRANCH) & used
    root_orphan = root_mask & (child_cnt == 0)

    any_violation = (
        mix_violation | leaf_violation | multi_act_viols |
        single_act_par | inter_violation | action_has_kids |
        depth_violation | root_orphan
    )
    return (~any_violation).view(B, N).all(dim=1)
