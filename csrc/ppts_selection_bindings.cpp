// csrc/ppts_selection_bindings.cpp
// This file contains the Python bindings for the Parameterless Pareto Tournament Selection (PPTS) kernels.
// It uses pybind11 to expose the C++/CUDA functions to Python, allowing for high-performance
// multi-objective selection to be called directly from Python.

#include <torch/extension.h>
#include "ppts_selection.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance CUDA PPTS (Parameterless Pareto Tournament Selection) for multi-objective optimization";

    // This binding exposes the ppts_select_elites_cuda function to Python.
    // It is used for selecting elite individuals with high selection pressure.
    m.def("ppts_select_elites", &ppts_select_elites_cuda,
          "Selects elite individuals using PPTS with high selection pressure.",
          py::arg("objectives"),
          py::arg("num_elites"), 
          py::arg("chunk_size") = -1,
          py::arg("rng_seed") = 0,
          R"pbdoc(
          Selects elite individuals using Parameterless Pareto Tournament Selection (PPTS).

          This function performs multi-objective selection without fixed weights, using
          larger tournament sizes for higher selection pressure on elite selection.

          Args:
              objectives (torch.Tensor): A [N, M] float32 CUDA tensor of minimization objectives.
              num_elites (int): The number of elites to select.
              chunk_size (int, optional): The tournament batch size for memory management. Defaults to -1 (auto).
              rng_seed (int, optional): The random seed for deterministic selection. Defaults to 0.

          Returns:
              torch.Tensor: A [num_elites] int64 tensor of selected elite indices (unique).

          Note:
              - Objectives are assumed to be for minimization (lower is better).
              - The tournament size is automatically computed as 2 * ceil(log2(N)) for elites.
              - The results are deduplicated to ensure unique elite selection.
              - This function is optimized for high performance, providing a significant speedup over Python implementations.
          )pbdoc");

    // This binding exposes the ppts_pick_parents_cuda function to Python.
    // It is used for selecting parent individuals with standard selection pressure.
    m.def("ppts_pick_parents", &ppts_pick_parents_cuda,
          "Selects parent individuals using PPTS with standard selection pressure.",
          py::arg("objectives"),
          py::arg("num_parents"),
          py::arg("chunk_size") = -1, 
          py::arg("rng_seed") = 0,
          R"pbdoc(
          Selects parent individuals using Parameterless Pareto Tournament Selection (PPTS).

          This function performs multi-objective parent selection with standard tournament
          sizes, allowing duplicates for genetic algorithm breeding operations.

          Args:
              objectives (torch.Tensor): A [N, M] float32 CUDA tensor of minimization objectives.
              num_parents (int): The number of parents to select.
              chunk_size (int, optional): The tournament batch size for memory management. Defaults to -1 (auto).
              rng_seed (int, optional): The random seed for deterministic selection. Defaults to 0.

          Returns:
              torch.Tensor: A [num_parents] int64 tensor of selected parent indices (duplicates are allowed).

          Note:
              - Objectives are assumed to be for minimization (lower is better).
              - The tournament size is automatically computed as ceil(log2(N)) for parents.
              - Duplicates are allowed and expected for parent selection.
              - This function is optimized for large populations and uses chunked processing for memory efficiency.
          )pbdoc");
}