// csrc/ppts_selection.cuh
// This header file declares the CUDA functions for the Parameterless Pareto Tournament Selection (PPTS) algorithm.

#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

/**
 * @brief Selects elite individuals using PPTS with high selection pressure.
 *
 * This function uses a larger tournament size to increase the selection pressure,
 * making it suitable for selecting elite individuals.
 *
 * @param objectives A [N, M] float32 CUDA tensor of minimization objectives.
 * @param num_elites The number of elites to select.
 * @param chunk_size The tournament batch size for memory management. Defaults to -1 (auto).
 * @param rng_seed The random seed for deterministic selection. Defaults to 0.
 * @return A [num_elites] int64 tensor of selected elite indices.
 */
torch::Tensor ppts_select_elites_cuda(
    const torch::Tensor& objectives,
    int64_t num_elites,
    int64_t chunk_size = -1,
    int64_t rng_seed = 0
);

/**
 * @brief Selects parent individuals using PPTS with standard selection pressure.
 *
 * This function uses a standard tournament size and allows for duplicate selections,
 * making it suitable for selecting parents for breeding.
 *
 * @param objectives A [N, M] float32 CUDA tensor of minimization objectives.
 * @param num_parents The number of parents to select (duplicates are allowed).
 * @param chunk_size The tournament batch size for memory management. Defaults to -1 (auto).
 * @param rng_seed The random seed for deterministic selection. Defaults to 0.
 * @return A [num_parents] int64 tensor of selected parent indices.
 */
torch::Tensor ppts_pick_parents_cuda(
    const torch::Tensor& objectives,
    int64_t num_parents, 
    int64_t chunk_size = -1,
    int64_t rng_seed = 0
);

/**
 * @brief The low-level fused tournament kernel for maximum performance.
 *
 * This function is for internal use only and is called by the elite and parent selection functions.
 *
 * @param objectives A pointer to the objectives tensor.
 * @param winner_indices A pointer to the output tensor for the winner indices.
 * @param tournament_indices A pointer to the pre-sampled tournament indices.
 * @param N The population size.
 * @param M The number of objectives.
 * @param num_tournaments The number of tournaments to run.
 * @param k The tournament size.
 * @param rng_seed The random seed for tie-breaking.
 * @param stream The CUDA stream to use.
 */
void launch_ppts_tournaments_kernel(
    const float* objectives,
    int64_t* winner_indices,
    const int32_t* tournament_indices,
    int N,
    int M,
    int num_tournaments,
    int k,
    int64_t rng_seed,
    cudaStream_t stream = 0
);