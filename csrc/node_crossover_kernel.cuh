// csrc/node_crossover_kernel.cuh
// This header file declares the CUDA functions for node-level crossover operations.
// These functions are implemented in node_crossover_kernel.cu and are called
// from the C++ bindings.

#pragma once
#include <torch/extension.h>

/**
 * @brief Generates a mask for nodes that match a specific node type and are
 *        within a specific root branch context.
 *
 * @param trees The population tensor.
 * @param output_mask The output mask tensor to be filled.
 * @param node_type The target node type to mask.
 * @param branch_type The target root branch type.
 */
void get_contextual_mask_cuda(
    const torch::Tensor& trees,
    torch::Tensor& output_mask,
    int node_type,
    int branch_type
);

/**
 * @brief Swaps the parameters of a random number of nodes between two
 *        populations, based on the provided masks.
 *
 * @param c1 The first population tensor.
 * @param c2 The second population tensor.
 * @param p1_mask The mask for the first population.
 * @param p2_mask The mask for the second population.
 */
void swap_node_params_cuda(
    torch::Tensor& c1,
    torch::Tensor& c2,
    const torch::Tensor& p1_mask,
    const torch::Tensor& p2_mask
);