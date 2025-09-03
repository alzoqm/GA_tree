// csrc/root_crossover_kernel.cuh
// This header file declares the CUDA functions for root branch crossover operations.

#pragma once
#include <torch/extension.h>

/**
 * @brief Copies branches from parent trees to child trees in a batch.
 *
 * @param child_batch The batch of child trees to be modified.
 * @param p1_batch The first batch of parent trees.
 * @param p2_batch The second batch of parent trees.
 * @param donor_map A map indicating which parent to use for each branch.
 * @param bfs_queue_buffer A buffer for the BFS queue.
 * @param result_indices_buffer A buffer to store the result of the BFS.
 * @param old_to_new_map_buffer A buffer for mapping old to new node indices.
 */
void copy_branches_batch_cuda(
    torch::Tensor& child_batch,
    const torch::Tensor& p1_batch,
    const torch::Tensor& p2_batch,
    const torch::Tensor& donor_map,
    torch::Tensor& bfs_queue_buffer,
    torch::Tensor& result_indices_buffer,
    torch::Tensor& old_to_new_map_buffer
);

/**
 * @brief Repairs the tree structure after a root branch crossover operation.
 *
 * @param trees The batch of trees to be repaired.
 * @param child_count_buffer A buffer for child counts.
 * @param act_cnt_buffer A buffer for action child counts.
 * @param dec_cnt_buffer A buffer for decision child counts.
 * @param bfs_queue_buffer A buffer for the BFS queue.
 * @param result_indices_buffer A buffer to store the result of the BFS.
 */
void repair_after_root_branch_cuda(
    torch::Tensor& trees,
    torch::Tensor& child_count_buffer,
    torch::Tensor& act_cnt_buffer,
    torch::Tensor& dec_cnt_buffer,
    torch::Tensor& bfs_queue_buffer,
    torch::Tensor& result_indices_buffer
);

/**
 * @brief Performs root branch crossover by sampling from a parent pool.
 *
 * @param child_batch The batch of child trees to be created.
 * @param parent_pool The pool of parent trees to sample from.
 * @param max_nodes The maximum number of nodes per tree.
 * @param max_retries The maximum number of retries to create a valid child.
 * @param bfs_queue_buffer A buffer for the BFS queue.
 * @param result_indices_buffer A buffer to store the result of the BFS.
 * @param old_to_new_map_buffer A buffer for mapping old to new node indices.
 */
void root_branch_sampling_crossover_cuda(
    torch::Tensor& child_batch,
    const torch::Tensor& parent_pool,
    int max_nodes,
    int max_retries,
    torch::Tensor& bfs_queue_buffer,
    torch::Tensor& result_indices_buffer,
    torch::Tensor& old_to_new_map_buffer
);