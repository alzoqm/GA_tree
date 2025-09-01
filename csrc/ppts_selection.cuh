// csrc/ppts_selection.cuh - High-Performance PPTS Selection Kernels
#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

/**
 * High-Performance CUDA Implementation of Parameterless Pareto Tournament Selection (PPTS)
 * 
 * This module provides GPU-accelerated selection for multi-objective optimization with:
 * - Fused tournament operations for maximum memory throughput
 * - Zero-copy shared memory optimization for 20x+ performance improvement
 * - Memory-aware chunking for populations of 500k+ individuals
 * - Exact semantic equivalence to Python implementation
 * 
 * Architecture:
 * - One block per tournament with k ≈ log₂(N) participants
 * - Shared memory for tournament data to minimize global memory access
 * - Warp-level primitives for Pareto dominance and ε-lexicase computation
 * - Pre-allocated memory pools to eliminate device allocations
 * 
 * Performance targets:
 * - 20x speedup over Python/Torch implementation for large N
 * - >80% memory bandwidth utilization
 * - >90% theoretical GPU occupancy
 * - Support for 500k+ population sizes
 */

/**
 * Elite selection using PPTS with larger tournament size for higher selection pressure
 * 
 * @param objectives    [N, M] float32 CUDA tensor (minimization objectives)
 * @param num_elites    Number of elites to select
 * @param chunk_size    Tournament batch size for memory management (-1 = auto)
 * @param rng_seed      Random seed for deterministic selection
 * @return              [num_elites] int64 tensor of selected elite indices
 */
torch::Tensor ppts_select_elites_cuda(
    const torch::Tensor& objectives,
    int64_t num_elites,
    int64_t chunk_size = -1,
    int64_t rng_seed = 0
);

/**
 * Parent selection using PPTS with standard tournament size
 * 
 * @param objectives    [N, M] float32 CUDA tensor (minimization objectives)  
 * @param num_parents   Number of parents to select (duplicates allowed)
 * @param chunk_size    Tournament batch size for memory management (-1 = auto)
 * @param rng_seed      Random seed for deterministic selection
 * @return              [num_parents] int64 tensor of selected parent indices
 */
torch::Tensor ppts_pick_parents_cuda(
    const torch::Tensor& objectives,
    int64_t num_parents, 
    int64_t chunk_size = -1,
    int64_t rng_seed = 0
);

/**
 * Low-level fused tournament kernel for maximum performance
 * Internal use only - called by elite/parent selection functions
 */
void launch_ppts_tournaments_kernel(
    const float* objectives,          // [N, M] objectives (minimization)
    int64_t* winner_indices,         // [num_tournaments] output winners
    const int32_t* tournament_indices, // [num_tournaments, k] pre-sampled indices
    int N,                           // Population size
    int M,                           // Number of objectives
    int num_tournaments,             // Number of tournaments to run
    int k,                           // Tournament size
    int64_t rng_seed,               // Random seed for tie-breaking
    cudaStream_t stream = 0
);