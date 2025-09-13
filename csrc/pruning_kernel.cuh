#pragma once
#include <torch/extension.h>

// Prune ACTION leaves based on per-node attribution stats and thresholds, then
// clean up by dropping DECISION nodes that no longer lead to any ACTION leaf.
// Finally, mark nodes to UNUSED in-place. Intended to be followed by
// reorganize_population_with_arrays_cuda for compaction.

struct PruneThresholds {
    int    total_steps;      // IS steps for eff_use
    int    k_min;            // minimum apply to consider used
    float  tau_use;          // eff_use minimum
    float  tau_ineff;        // ineff ratio maximum
    int    k_perf_min;       // minimum apply for performance gate
    float  tau_wr;           // win-rate minimum
    int    k_harm;           // frequent harmful threshold
    float  delta_harm;       // harmful pnl_per_exposure threshold (<= -delta)
    int    m_min;            // per-branch minimum actions to keep
    float  w_use;            // score weights
    float  w_ineff;
    float  w_pnl;
    float  w_wr;
};

void prune_mark_and_cleanup_cuda(
    torch::Tensor population_tensor,
    // per-node stats (B,N)
    torch::Tensor node_attempt_count,
    torch::Tensor node_apply_count,
    torch::Tensor node_noop_count,
    torch::Tensor node_stall_count,
    torch::Tensor node_win_count,
    torch::Tensor node_realized_pnl_sum,
    torch::Tensor node_exposure_sum,
    PruneThresholds thr
);

