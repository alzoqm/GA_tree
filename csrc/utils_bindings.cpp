// csrc/utils_bindings.cpp
// This file contains the Python bindings for various utility kernels, including
// population reorganization, initialization, and validation.

#include <torch/extension.h>
#include "reorganize_kernel.cuh"
#include "validate_kernel.cuh"
#include "make_population.cuh"

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

}
