// csrc/reorganize_kernel.cuh
// This header file declares the CUDA functions for reorganizing the population.

#pragma once
#include <torch/extension.h>

/**
 * @brief Reorganizes the population by removing unused nodes and compacting the trees.
 *
 * @param population_tensor The population tensor to be reorganized in-place.
 */
void reorganize_population_cuda(torch::Tensor population_tensor);

/**
 * @brief Reorganizes the population using pre-allocated arrays.
 *
 * @param population_tensor The population tensor to be reorganized in-place.
 * @param active_counts_per_tree A tensor to store the number of active nodes in each tree.
 * @param old_gid_to_new_gid_map A tensor to map old global indices to new global indices.
 */
void reorganize_population_with_arrays_cuda(
    torch::Tensor population_tensor,
    torch::Tensor active_counts_per_tree,
    torch::Tensor old_gid_to_new_gid_map
);