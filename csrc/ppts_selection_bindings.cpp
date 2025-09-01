// csrc/ppts_selection_bindings.cpp - Python bindings for PPTS selection
#include <torch/extension.h>
#include "ppts_selection.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance CUDA PPTS (Parameterless Pareto Tournament Selection) for multi-objective optimization";

    // Elite selection with PPTS using larger tournament size for higher selection pressure
    m.def("ppts_select_elites", &ppts_select_elites_cuda,
          "Select elites using PPTS with high selection pressure",
          py::arg("objectives"),
          py::arg("num_elites"), 
          py::arg("chunk_size") = -1,
          py::arg("rng_seed") = 0,
          R"pbdoc(
          Select elite individuals using Parameterless Pareto Tournament Selection (PPTS).
          
          This function performs multi-objective selection without fixed weights, using
          larger tournament sizes for higher selection pressure on elite selection.
          
          Args:
              objectives (torch.Tensor): [N, M] float32 CUDA tensor of minimization objectives
              num_elites (int): Number of elites to select
              chunk_size (int, optional): Tournament batch size for memory management (-1 = auto)  
              rng_seed (int, optional): Random seed for deterministic selection (default: 0)
              
          Returns:
              torch.Tensor: [num_elites] int64 tensor of selected elite indices (unique)
              
          Note:
              - Objectives are assumed to be for minimization (lower is better)
              - Tournament size is auto-computed as 2 * ceil(log2(N)) for elites
              - Results are deduplicated to ensure unique elite selection
              - Performance optimized for 20x+ speedup over Python implementation
          )pbdoc");

    // Parent selection with PPTS using standard tournament size (duplicates allowed)
    m.def("ppts_pick_parents", &ppts_pick_parents_cuda,
          "Select parents using PPTS with standard selection pressure",
          py::arg("objectives"),
          py::arg("num_parents"),
          py::arg("chunk_size") = -1, 
          py::arg("rng_seed") = 0,
          R"pbdoc(
          Select parent individuals using Parameterless Pareto Tournament Selection (PPTS).
          
          This function performs multi-objective parent selection with standard tournament
          sizes, allowing duplicates for genetic algorithm breeding operations.
          
          Args:
              objectives (torch.Tensor): [N, M] float32 CUDA tensor of minimization objectives
              num_parents (int): Number of parents to select  
              chunk_size (int, optional): Tournament batch size for memory management (-1 = auto)
              rng_seed (int, optional): Random seed for deterministic selection (default: 0)
              
          Returns:
              torch.Tensor: [num_parents] int64 tensor of selected parent indices (duplicates allowed)
              
          Note:
              - Objectives are assumed to be for minimization (lower is better)
              - Tournament size is auto-computed as ceil(log2(N)) for parents
              - Duplicates are allowed and expected for parent selection
              - Optimized for populations of 500k+ individuals with chunked processing
          )pbdoc");
}