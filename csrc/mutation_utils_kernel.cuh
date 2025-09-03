// csrc/mutation_utils_kernel.cuh
// This header file declares the CUDA utility functions for mutation operations.
// These functions are implemented in mutation_utils_kernel.cu and are called
// from the C++ bindings.

#pragma once
#include <torch/extension.h>

/**
 * @brief Finds all nodes in a subtree for a batch of trees.
 *
 * @param trees The population tensor (B, max_nodes, node_dim).
 * @param root_indices The root indices of the subtrees to find (-1 to skip).
 * @param max_nodes The maximum number of nodes per tree.
 * @param bfs_queue_buffer A buffer for the BFS queue.
 * @param result_indices_buffer A buffer to store the result.
 * @param out_counts The number of nodes found in each subtree.
 */
void find_subtree_nodes_batch_cuda(
    const torch::Tensor& trees,            // (B, max_nodes, node_dim)
    const torch::Tensor& root_indices,     // (B,) int32, -1 means skip
    int max_nodes,
    torch::Tensor bfs_queue_buffer,        // (B, 2*max_nodes) int32
    torch::Tensor result_indices_buffer,   // (B, 2*max_nodes) int32
    torch::Tensor out_counts               // (B,) int32: number of collected nodes per tree
);

/**
 * @brief Updates the depth of all nodes in a subtree by a given delta.
 *
 * @param trees The population tensor (modified in-place).
 * @param root_indices The root indices of the subtrees to update.
 * @param delta The value to add to the depth of each node.
 * @param max_nodes The maximum number of nodes per tree.
 * @param bfs_queue_buffer A buffer for the BFS queue.
 * @param result_indices_buffer A buffer to store the result of the BFS.
 */
void update_subtree_depth_batch_cuda(
    torch::Tensor trees,                   // in/out
    const torch::Tensor& root_indices,     // (B,) int32
    int delta,
    int max_nodes,
    torch::Tensor bfs_queue_buffer,        // (B, 2*max_nodes) int32
    torch::Tensor result_indices_buffer    // (B, 2*max_nodes) int32
);

/**
 * @brief Gets the maximum depth of a subtree.
 *
 * @param trees The population tensor.
 * @param root_indices The root indices of the subtrees.
 * @param max_nodes The maximum number of nodes per tree.
 * @param bfs_queue_buffer A buffer for the BFS queue.
 * @param result_indices_buffer A buffer to store the result of the BFS.
 * @param out_max_depths The maximum depth of each subtree.
 */
void get_subtree_max_depth_batch_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& root_indices,     // (B,) int32
    int max_nodes,
    torch::Tensor bfs_queue_buffer,        // (B, 2*max_nodes) int32
    torch::Tensor result_indices_buffer,   // (B, 2*max_nodes) int32
    torch::Tensor out_max_depths           // (B,) int32
);

/**
 * @brief Finds the first 'count' empty (UNUSED) slots in each tree.
 *
 * @param trees The population tensor.
 * @param count The number of empty slots to find per tree.
 * @param max_nodes The maximum number of nodes per tree.
 * @param out_indices A tensor to store the indices of the empty slots.
 */
void find_empty_slots_batch_cuda(
    const torch::Tensor& trees,            // (B, max_nodes, node_dim)
    int count,                             // #slots per tree
    int max_nodes,
    torch::Tensor out_indices              // (B, count) int32, -1 fill if insufficient
);