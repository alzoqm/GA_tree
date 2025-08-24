// csrc/predict_bindings.cpp
#include <torch/extension.h>
#include "predict_kernel.cuh"
#include "adjacency_builder.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance CUDA prediction kernels for GATree";

    // --- Adjacency (Step 1: counts + offsets + overflow mask) ---
    m.def("count_and_create_offsets", &count_and_create_offsets_cuda,
          "Step 1: Count children, build CSR offsets, and return per-tree overflow mask",
          py::arg("population_tensor"),
          py::arg("max_children"));

    // --- Adjacency (Step 2: fill + optional sort) ---
    m.def("fill_child_indices", &fill_child_indices_cuda,
          "Step 2: Fill CSR child indices and (optionally) sort per parent",
          py::arg("population_tensor"),
          py::arg("offset_array"),
          py::arg("child_indices"),
          py::arg("max_children"),
          py::arg("sort_children") = true);

    // --- Validator ---
    m.def("validate_adjacency", &validate_adjacency_cuda,
          "Validate CSR/structure; writes per-tree violation bitmask (0 means OK)",
          py::arg("population_tensor"),
          py::arg("offset_array"),
          py::arg("child_indices"),
          py::arg("max_children"),
          py::arg("max_depth"),
          py::arg("out_violation_mask"));

    // --- Optimized prediction function with safety guarantees ---
    m.def("predict", &predict_cuda, 
          "GPU-accelerated prediction for GA-Tree populations with safety guarantees",
          py::arg("trees"), py::arg("features"), py::arg("positions"), 
          py::arg("offsets"), py::arg("children"), py::arg("results"),
          py::arg("bfs_q"), py::arg("visited"));
}