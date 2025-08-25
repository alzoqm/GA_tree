// csrc/utils_bindings.cpp
#include <torch/extension.h>
#include "reorganize_kernel.cuh"
#include "validate_kernel.cuh"
#include "make_population.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance CUDA utility kernels for GATree";

    // --- 재구성 함수 바인딩 ---
    m.def("reorganize_population", &reorganize_population_cuda, 
          "Reorganize the population tensor on GPU to remove fragmentation.");
    
    m.def("reorganize_population_with_arrays", &reorganize_population_with_arrays_cuda, 
          "Reorganize the population tensor on GPU using pre-allocated arrays from Python.",
          py::arg("population_tensor"), py::arg("active_counts_per_tree"), py::arg("old_gid_to_new_gid_map"));

    // --- GPU Population Initialization with invariant guarantees ---
    m.def("init_population_cuda", &init_population_cuda,
          "Initialize a batch of trees on CUDA with invariant guarantees.",
          py::arg("trees"), py::arg("total_budget"),
          py::arg("max_children"), py::arg("max_depth"), py::arg("max_nodes"),
          py::arg("bfs_q"), py::arg("scratch"), py::arg("child_cnt"),
          py::arg("act_cnt"), py::arg("dec_cnt"), py::arg("cand_idx"), py::arg("cand_w"),
          py::arg("num_feat_indices"), py::arg("num_feat_minmax"),
          py::arg("bool_feat_indices"), py::arg("ff_pairs"),
          py::arg("long_actions"), py::arg("hold_actions"), py::arg("short_actions"));

    // --- Validation: checks structural constraints and throws if violations found ---
    m.def("validate_trees", &validate_trees_or_throw_cuda,
          "Validate tree structural constraints on GPU; throws on invalid trees.",
          py::arg("trees"));

}