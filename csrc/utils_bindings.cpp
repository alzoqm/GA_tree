// csrc/utils_bindings.cpp
// This file contains the Python bindings for various utility kernels, including
// population reorganization, initialization, and validation.

#include <torch/extension.h>
#include "reorganize_kernel.cuh"
#include "validate_kernel.cuh"
#include "make_population.cuh"
#include "pruning_kernel.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance CUDA utility kernels for GATree";

    // --- Population Reorganization ---
    // Canonical API: reorganize_population uses pre-allocated arrays for performance and stability.
    m.def("reorganize_population", &reorganize_population_with_arrays_cuda, 
          "Reorganizes the population tensor on the GPU using pre-allocated arrays.",
          py::arg("population_tensor"), py::arg("active_counts_per_tree"), py::arg("old_gid_to_new_gid_map"));

    // --- Population Initialization ---

    // This binding exposes the init_population_cuda function to Python.
    m.def("init_population_cuda", &init_population_cuda,
          "Initializes a batch of trees on the GPU with invariant guarantees.",
          py::arg("trees"), py::arg("total_budget"),
          py::arg("max_children"), py::arg("max_depth"), py::arg("max_nodes"),
          py::arg("bfs_q"), py::arg("scratch"), py::arg("child_cnt"),
          py::arg("act_cnt"), py::arg("dec_cnt"), py::arg("cand_idx"), py::arg("cand_w"),
          py::arg("num_feat_indices"), py::arg("num_feat_minmax"),
          py::arg("bool_feat_indices"), py::arg("ff_pairs"),
          py::arg("long_actions"), py::arg("hold_actions"), py::arg("short_actions"));

    // --- Validation ---

    // This binding exposes the validate_trees_or_throw_cuda function to Python.
    m.def("validate_trees", &validate_trees_or_throw_cuda,
          "Validates tree structural constraints on the GPU and throws an exception if any violations are found.",
          py::arg("trees"));

    // --- Pruning ---
    m.def("prune_mark_and_cleanup", [](torch::Tensor population_tensor,
                                        torch::Tensor node_attempt_count,
                                        torch::Tensor node_apply_count,
                                        torch::Tensor node_noop_count,
                                        torch::Tensor node_stall_count,
                                        torch::Tensor node_win_count,
                                        torch::Tensor node_realized_pnl_sum,
                                        torch::Tensor node_exposure_sum,
                                        int total_steps,
                                        int k_min,
                                        float tau_use,
                                        float tau_ineff,
                                        int k_perf_min,
                                        float tau_wr,
                                        int k_harm,
                                        float delta_harm,
                                        int m_min,
                                        float w_use,
                                        float w_ineff,
                                        float w_pnl,
                                        float w_wr) {
        PruneThresholds thr{total_steps, k_min, tau_use, tau_ineff, k_perf_min, tau_wr, k_harm, delta_harm, m_min, w_use, w_ineff, w_pnl, w_wr};
        prune_mark_and_cleanup_cuda(
            population_tensor,
            node_attempt_count,
            node_apply_count,
            node_noop_count,
            node_stall_count,
            node_win_count,
            node_realized_pnl_sum,
            node_exposure_sum,
            thr
        );
    },
    "Prune ACTION leaves by CUDA using attribution stats, then cleanup DECISION dead-ends.",
    py::arg("population_tensor"),
    py::arg("node_attempt_count"),
    py::arg("node_apply_count"),
    py::arg("node_noop_count"),
    py::arg("node_stall_count"),
    py::arg("node_win_count"),
    py::arg("node_realized_pnl_sum"),
    py::arg("node_exposure_sum"),
    py::arg("total_steps"),
    py::arg("k_min"),
    py::arg("tau_use"),
    py::arg("tau_ineff"),
    py::arg("k_perf_min"),
    py::arg("tau_wr"),
    py::arg("k_harm"),
    py::arg("delta_harm"),
    py::arg("m_min"),
    py::arg("w_use"),
    py::arg("w_ineff"),
    py::arg("w_pnl"),
    py::arg("w_wr"));

}
