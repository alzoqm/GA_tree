// csrc/subtree_crossover_kernel.cuh
// This header file declares the CUDA function for performing subtree crossover.

#pragma once
#include <torch/extension.h>

/**
 * @brief Performs subtree crossover on a batch of trees.
 *
 * @param child1_out The first batch of child trees to be created.
 * @param child2_out The second batch of child trees to be created.
 * @param p1_batch The first batch of parent trees.
 * @param p2_batch The second batch of parent trees.
 * @param mode The crossover mode (0 for free, 1 for context-aware).
 * @param max_nodes The maximum number of nodes per tree.
 * @param max_retries The maximum number of retries to create a valid child.
 * @param branch_perm A permutation of branches for context-aware crossover.
 * @param bfs_queue_buffer A buffer for the BFS queue.
 * @param result_indices_buffer A buffer to store the result of the BFS.
 * @param old_to_new_map_buffer A buffer for mapping old to new node indices.
 * @param p1_candidates_buffer A buffer for candidate nodes from the first parent.
 * @param p2_candidates_buffer A buffer for candidate nodes from the second parent.
 */
void subtree_crossover_batch_cuda(
    torch::Tensor& child1_out,
    torch::Tensor& child2_out,
    const torch::Tensor& p1_batch,
    const torch::Tensor& p2_batch,
    int mode,
    int max_nodes,
    int max_retries,
    const torch::Tensor& branch_perm,
    torch::Tensor& bfs_queue_buffer,
    torch::Tensor& result_indices_buffer,
    torch::Tensor& old_to_new_map_buffer,
    torch::Tensor& p1_candidates_buffer,
    torch::Tensor& p2_candidates_buffer
);