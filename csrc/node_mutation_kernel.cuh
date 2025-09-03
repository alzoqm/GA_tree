// csrc/node_mutation_kernel.cuh
// This header file declares the CUDA functions for node-level mutation operations,
// including adding and deleting nodes.

#pragma once
#include <torch/extension.h>

/**
 * @brief Adds decision nodes to a batch of trees.
 *
 * This function performs an add-node mutation by splitting an existing edge and
 * inserting a new decision node. It ensures that the tree structure remains valid.
 *
 * @param trees The population tensor.
 * @param num_to_add The number of nodes to add to each tree.
 * @param max_nodes The maximum number of nodes per tree.
 * @param max_add_nodes The maximum number of nodes that can be added per tree.
 * @param out_new_node_indices A tensor to store the indices of the new nodes.
 * @param bfs_queue_buffer A buffer for the BFS queue.
 * @param result_indices_buffer A buffer to store the result of the BFS.
 * @param old_to_new_map_buffer A buffer for mapping old to new node indices.
 * @param candidate_indices_buffer A buffer for candidate node indices.
 * @param candidate_weights_buffer A buffer for candidate node weights.
 */
void add_decision_nodes_batch_cuda(
    torch::Tensor trees,
    torch::Tensor num_to_add,
    int max_nodes,
    int max_add_nodes,
    torch::Tensor out_new_node_indices,
    torch::Tensor bfs_queue_buffer,
    torch::Tensor result_indices_buffer,
    torch::Tensor old_to_new_map_buffer,
    torch::Tensor candidate_indices_buffer,
    torch::Tensor candidate_weights_buffer
);

/**
 * @brief Deletes nodes from a batch of trees.
 *
 * This function performs a delete-node mutation by removing a node and reconnecting
 * its children to its parent. It ensures that the tree structure remains valid.
 *
 * @param trees The population tensor.
 * @param num_to_delete The number of nodes to delete from each tree.
 * @param max_nodes The maximum number of nodes per tree.
 * @param max_delete_nodes The maximum number of nodes that can be deleted per tree.
 * @param deleted_nodes A tensor to store the indices of the deleted nodes.
 * @param bfs_queue_buffer A buffer for the BFS queue.
 * @param result_indices_buffer A buffer to store the result of the BFS.
 * @param child_count_buffer A buffer for child counts.
 * @param act_cnt_buffer A buffer for action child counts.
 * @param dec_cnt_buffer A buffer for decision child counts.
 * @param candidate_indices_buffer A buffer for candidate node indices.
 * @param candidate_weights_buffer A buffer for candidate node weights.
 */
void delete_nodes_batch_cuda(
    torch::Tensor trees,
    torch::Tensor num_to_delete,
    int max_nodes,
    int max_delete_nodes,
    torch::Tensor deleted_nodes,
    torch::Tensor bfs_queue_buffer,
    torch::Tensor result_indices_buffer,
    torch::Tensor child_count_buffer,
    torch::Tensor act_cnt_buffer,
    torch::Tensor dec_cnt_buffer,
    torch::Tensor candidate_indices_buffer,
    torch::Tensor candidate_weights_buffer
);
