# evolution/Mutation/add_node.py
# CUDA-accelerated batch AddNodeMutation implementation
import torch
from typing import Dict, Any

from .base import BaseMutation
from models.constants import (
    COL_NODE_TYPE, COL_PARAM_1, COL_PARAM_2, COL_PARAM_3, COL_PARAM_4,
    NODE_TYPE_DECISION, OP_GTE, OP_LTE,
    COMP_TYPE_FEAT_NUM, COMP_TYPE_FEAT_FEAT, COMP_TYPE_FEAT_BOOL
)

# C++/CUDA extension
try:
    import gatree_cuda
except ImportError:
    print("="*60)
    print(">>> Warning: 'gatree_cuda' module not found.")
    print(">>> Build the CUDA extension first:")
    print(">>> python setup.py build_ext --inplace")
    print("="*60)
    gatree_cuda = None

class DecisionParamSampler:
    """
    GPU-vectorized parameter filler for newly inserted DECISION nodes.
    - Chooses comparison type among {NUM, FEAT_FEAT, FEAT_BOOL} depending on config.
    - Samples features/ops/values entirely with tensor ops on GPU.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

        all_features = config['all_features']
        self.all_feat_to_idx = {name: i for i, name in enumerate(all_features)}

        # Numeric features
        feature_num = config.get('feature_num', {})
        num_feat_names = list(feature_num.keys())
        self.has_num = len(num_feat_names) > 0
        if self.has_num:
            num_feat_idx = [self.all_feat_to_idx[n] for n in num_feat_names]
            mins = [float(feature_num[n][0]) for n in num_feat_names]
            maxs = [float(feature_num[n][1]) for n in num_feat_names]
            self.num_feat_idx = torch.tensor(num_feat_idx, device=device, dtype=torch.long)
            self.num_feat_min = torch.tensor(mins, device=device, dtype=self.dtype)
            self.num_feat_max = torch.tensor(maxs, device=device, dtype=self.dtype)

        # Feature-feature pairs (flattened)
        feature_comparison = config.get('feature_comparison', [])
        pairs = []
        for i, f1 in enumerate(feature_comparison):
            for j, f2 in enumerate(feature_comparison):
                if i != j:  # Different features can be compared
                    pairs.append((self.all_feat_to_idx[f1], self.all_feat_to_idx[f2]))
        self.has_pairs = len(pairs) > 0
        if self.has_pairs:
            self.pair_f1 = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
            self.pair_f2 = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)

        # Boolean features
        feat_bool = config.get('feature_bool', [])
        self.has_bool = len(feat_bool) > 0
        if self.has_bool:
            bool_idx = [self.all_feat_to_idx[n] for n in feat_bool]
            self.bool_feat_idx = torch.tensor(bool_idx, device=device, dtype=torch.long)

        # Operators
        self.ops = torch.tensor([OP_GTE, OP_LTE], device=device, dtype=self.dtype)

        # Types
        self.ct_num  = torch.tensor([COMP_TYPE_FEAT_NUM],  device=device, dtype=self.dtype)
        self.ct_pair = torch.tensor([COMP_TYPE_FEAT_FEAT], device=device, dtype=self.dtype)
        self.ct_bool = torch.tensor([COMP_TYPE_FEAT_BOOL], device=device, dtype=self.dtype)

    @torch.no_grad()
    def fill_params_for_nodes(
        self,
        trees: torch.Tensor,
        node_batch_idx: torch.Tensor,  # (M,)
        node_indices: torch.Tensor     # (M,)
    ) -> None:
        """Fill params for the given DECISION nodes in one shot."""
        if node_batch_idx.numel() == 0:
            return
        dev, dt = trees.device, trees.dtype
        M = node_batch_idx.numel()

        choices = []
        if self.has_num:   choices.append(COMP_TYPE_FEAT_NUM)
        if self.has_pairs: choices.append(COMP_TYPE_FEAT_FEAT)
        if self.has_bool:  choices.append(COMP_TYPE_FEAT_BOOL)
        if not choices:
            choices = [COMP_TYPE_FEAT_NUM]

        choices_t = torch.tensor(choices, device=dev, dtype=dt)
        comp_type = choices_t[torch.randint(0, len(choices), (M,), device=dev)]

        op = self.ops[torch.randint(0, 2, (M,), device=dev)]

        # NUM
        if self.has_num:
            pick_num = torch.randint(0, self.num_feat_idx.numel(), (M,), device=dev)
            num_feat_idx = self.num_feat_idx[pick_num]
            vmin = self.num_feat_min[pick_num]
            vmax = self.num_feat_max[pick_num]
            comp_val = torch.rand(M, device=dev, dtype=dt) * (vmax - vmin) + vmin

        # PAIR
        if self.has_pairs:
            pick_pair = torch.randint(0, self.pair_f1.numel(), (M,), device=dev)
            f1 = self.pair_f1[pick_pair]
            f2 = self.pair_f2[pick_pair]

        # BOOL
        if self.has_bool:
            pick_bool = torch.randint(0, self.bool_feat_idx.numel(), (M,), device=dev)
            bool_feat = self.bool_feat_idx[pick_bool]
            bool_val  = (torch.randint(0, 2, (M,), device=dev, dtype=torch.long)).to(dt)

        b = node_batch_idx
        n = node_indices

        trees[b, n, COL_PARAM_3] = comp_type

        num_mask = (comp_type == COMP_TYPE_FEAT_NUM)
        if num_mask.any():
            bm, nm = b[num_mask], n[num_mask]
            trees[bm, nm, COL_PARAM_1] = num_feat_idx[num_mask].to(dt)
            trees[bm, nm, COL_PARAM_2] = op[num_mask]
            trees[bm, nm, COL_PARAM_4] = comp_val[num_mask]

        pair_mask = (comp_type == COMP_TYPE_FEAT_FEAT)
        if pair_mask.any():
            bm, nm = b[pair_mask], n[pair_mask]
            trees[bm, nm, COL_PARAM_1] = f1[pair_mask].to(dt)
            trees[bm, nm, COL_PARAM_2] = op[pair_mask]
            trees[bm, nm, COL_PARAM_4] = f2[pair_mask].to(dt)

        bool_mask = (comp_type == COMP_TYPE_FEAT_BOOL)
        if bool_mask.any():
            bm, nm = b[bool_mask], n[bool_mask]
            trees[bm, nm, COL_PARAM_1] = bool_feat[bool_mask].to(dt)
            trees[bm, nm, COL_PARAM_4] = bool_val[bool_mask]


class AddNodeMutation(BaseMutation):
    """
    Batch add-node mutation (GPU).
    - One CUDA call performs all edge-splits, rewiring, and depth updates.
    - Then we fill DECISION params for all newly inserted nodes in a single vectorized pass.
    """
    def __init__(self, prob: float = 0.1, config: Dict[str, Any] = None,
                 max_add_nodes: int = 5, max_nodes: int = 256):
        super().__init__(prob)
        if config is None:
            raise ValueError("AddNodeMutation requires a 'config' dictionary.")
        if not isinstance(max_add_nodes, int) or max_add_nodes < 1:
            raise ValueError("max_add_nodes must be a positive integer.")
        self.config = config
        self.max_depth = int(config['max_depth'])
        self.max_add_nodes = int(max_add_nodes)
        self.max_nodes = int(max_nodes)
        self._sampler = None

    def _ensure_sampler(self, device: torch.device, dtype: torch.dtype):
        if (self._sampler is None) or (self._sampler.device != device) or (self._sampler.dtype != dtype):
            self._sampler = DecisionParamSampler(self.config, device=device, dtype=dtype)

    def __call__(self, chromosomes: torch.Tensor) -> torch.Tensor:
        if gatree_cuda is None:
            raise RuntimeError("gatree_cuda module is not loaded. Cannot perform add-node mutation.")

        trees = chromosomes.clone()  # (B, max_nodes, node_dim)
        B = trees.shape[0]
        device, dtype = trees.device, trees.dtype

        self._ensure_sampler(device, dtype)

        mutate_mask = (torch.rand(B, device=device) < self.prob)
        if not mutate_mask.any():
            return trees

        num_to_add = torch.zeros(B, device=device, dtype=torch.int32)
        k = torch.randint(1, self.max_add_nodes + 1, (mutate_mask.sum(),), device=device, dtype=torch.int32)
        num_to_add[mutate_mask] = k

        # Work buffers — ALL allocated in Python as requested (no device VLA).
        bfs_queue_buffer       = torch.empty((B, 2 * self.max_nodes), dtype=torch.int32, device=device)
        result_indices_buffer  = torch.empty((B, 2 * self.max_nodes), dtype=torch.int32, device=device)
        old_to_new_map_buffer  = torch.empty((B, self.max_nodes),   dtype=torch.int32, device=device)

        new_node_indices = torch.full((B, self.max_add_nodes), -1, dtype=torch.int32, device=device)

        # Single CUDA call for the whole batch
        gatree_cuda.add_decision_nodes_batch(
            trees,
            num_to_add,
            int(self.max_depth),
            int(self.max_nodes),
            int(self.max_add_nodes),
            new_node_indices,
            bfs_queue_buffer, result_indices_buffer, old_to_new_map_buffer
        )

        # Vectorized param fill for DECISION nodes
        valid = (new_node_indices >= 0)
        if valid.any():
            rows, cols = torch.nonzero(valid, as_tuple=True)
            new_nodes_flat = new_node_indices[rows, cols].to(torch.long)
            self._sampler.fill_params_for_nodes(trees, rows, new_nodes_flat)
            # Ensure node type is DECISION (kernel already set it; redundant but safe)
            trees[rows, new_nodes_flat, COL_NODE_TYPE] = torch.tensor(NODE_TYPE_DECISION, device=device, dtype=dtype)

        return trees