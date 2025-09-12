// csrc/predict_bindings.cpp
// This file contains the Python bindings for the CUDA prediction and adjacency list kernels.
// It uses pybind11 to expose the C++/CUDA functions to Python.

#include <torch/extension.h>
#include "predict_kernel.cuh"
#include "adjacency_builder.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance CUDA prediction kernels for GATree";

    // --- Adjacency List Creation ---

    // Step 1: Count children, create CSR offsets, and detect overflow.
    m.def("count_and_create_offsets", &count_and_create_offsets_cuda,
          "Counts children, builds CSR offsets, and returns a per-tree overflow mask.",
          py::arg("population_tensor"),
          py::arg("max_children"));

    // Step 2: Fill the child indices array and optionally sort.
    m.def("fill_child_indices", &fill_child_indices_cuda,
          "Fills CSR child indices and optionally sorts them for each parent.",
          py::arg("population_tensor"),
          py::arg("offset_array"),
          py::arg("child_indices"),
          py::arg("max_children"),
          py::arg("sort_children") = true);

    // --- Adjacency List Validation ---
    m.def("validate_adjacency", &validate_adjacency_cuda,
          "Validates the CSR structure and writes a per-tree violation bitmask.",
          py::arg("population_tensor"),
          py::arg("offset_array"),
          py::arg("child_indices"),
          py::arg("out_violation_mask"));

    // --- Adjacency Dictionary Builder ---

    // This function builds adjacency dictionaries for all trees in parallel on the GPU.
    m.def("build_adjacency_dicts", &build_adjacency_dicts_cuda,
          "Builds adjacency dictionaries for all trees in parallel.",
          py::arg("trees"),
          py::arg("next_idxs"));

    // This is a memory-optimized version of the adjacency dictionary builder.
    m.def("build_adjacency_dicts_optimized", &build_adjacency_dicts_optimized_cuda,
          "A memory-optimized function to build adjacency dictionaries.",
          py::arg("trees"),
          py::arg("next_idxs"));

    // --- Prediction ---

    // This function performs GPU-accelerated prediction for a batch of trees.
    m.def("predict", &predict_cuda, 
          "Performs GPU-accelerated prediction for a batch of trees with safety guarantees.\n"
          "results tensor may have >=4 columns. If >=5, column[4] will store the selected ACTION node index.",
          py::arg("trees"), py::arg("features"), py::arg("positions"), 
          py::arg("offsets"), py::arg("children"), py::arg("results"),
          py::arg("bfs_q"), py::arg("visited_gen"), py::arg("step_id"),
          py::arg("action_node_idx"));

}
