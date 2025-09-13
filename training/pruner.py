import torch
from typing import Dict

try:
    import gatree_cuda_compat as gatree_cuda
except ImportError:
    gatree_cuda = None


class PopulationPruner:
    """CUDA-based population pruner using node-level attribution stats."""

    def __init__(self, pruning_cfg: Dict):
        self.cfg = pruning_cfg or {}

    def is_enabled(self) -> bool:
        return bool(self.cfg.get('enabled', False))

    def prune_in_place(self, population, stats: Dict):
        if gatree_cuda is None or getattr(gatree_cuda, 'prune_mark_and_cleanup', None) is None:
            raise RuntimeError("CUDA pruning kernel not available. Build extensions first.")

        # Extract stats tensors, ensure CUDA and contiguous
        def to_i32(x):
            return x.to(device=population.population_tensor.device, dtype=torch.int32).contiguous()
        def to_f32(x):
            return x.to(device=population.population_tensor.device, dtype=torch.float32).contiguous()

        node_attempt_count   = to_i32(stats['node_attempt_count'])
        node_apply_count     = to_i32(stats['node_apply_count'])
        node_noop_count      = to_i32(stats['node_noop_count'])
        node_stall_count     = to_i32(stats.get('node_stall_count', torch.zeros_like(node_attempt_count)))
        node_win_count       = to_i32(stats['node_win_count'])
        node_realized_pnl_sum= to_f32(stats['node_realized_pnl_sum'])
        node_exposure_sum    = to_f32(stats['node_exposure_sum'])
        total_steps          = int(stats.get('total_steps', 0))

        # Thresholds
        k_min      = int(self.cfg.get('k_min', 3))
        tau_use    = float(self.cfg.get('tau_use', 0.01))
        tau_ineff  = float(self.cfg.get('tau_ineff', 0.8))
        k_perf_min = int(self.cfg.get('k_perf_min', 3))
        tau_wr     = float(self.cfg.get('tau_wr', 0.45))
        k_harm     = int(self.cfg.get('k_harm', 5))
        delta_harm = float(self.cfg.get('delta_harm', 0.001))
        m_min      = int(self.cfg.get('m_min', 1))
        w_use      = float(self.cfg.get('w_use', 0.2))
        w_ineff    = float(self.cfg.get('w_ineff', 0.2))
        w_pnl      = float(self.cfg.get('w_pnl', 0.5))
        w_wr       = float(self.cfg.get('w_wr', 0.1))

        # In-place prune mark + cleanup
        gatree_cuda.prune_mark_and_cleanup(
            population.population_tensor,
            node_attempt_count,
            node_apply_count,
            node_noop_count,
            node_stall_count,
            node_win_count,
            node_realized_pnl_sum,
            node_exposure_sum,
            total_steps,
            k_min,
            tau_use,
            tau_ineff,
            k_perf_min,
            tau_wr,
            k_harm,
            delta_harm,
            m_min,
            w_use,
            w_ineff,
            w_pnl,
            w_wr,
        )

        # Compact in-place using existing reorganize kernel
        B, N, _ = population.population_tensor.shape
        active_counts = torch.zeros((B,), dtype=torch.int32, device=population.population_tensor.device)
        old_to_new    = torch.full((B * N,), -1, dtype=torch.int32, device=population.population_tensor.device)
        gatree_cuda.reorganize_population(population.population_tensor, active_counts, old_to_new)

        # Validate
        gatree_cuda.validate_trees(population.population_tensor.contiguous())

