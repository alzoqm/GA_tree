// csrc/subtree_mutation_kernel.cuh
// This header file declares the CUDA functions for performing subtree mutations.

#pragma once
#include <torch/extension.h>

/**
 * @brief Adds subtrees to a batch of trees.
 *
 * @param trees The population tensor.
 * @param num_to_grow The number of nodes to grow for each tree.
 * @param max_nodes The maximum number of nodes per tree.
 * @param max_new_nodes The maximum number of new nodes that can be added.
 * @param new_decision_nodes A tensor to store the indices of new decision nodes.
 * @param new_action_nodes A tensor to store the indices of new action nodes.
 * @param action_root_branch_type A tensor to store the root branch type of new action nodes.
 * @param bfs_queue_buffer A buffer for the BFS queue.
 * @param result_indices_buffer A buffer to store the result of the BFS.
 * @param child_count_buffer A buffer for child counts.
 * @param act_cnt_buffer A buffer for action child counts.
 * @param dec_cnt_buffer A buffer for decision child counts.
 * @param candidate_indices_buffer A buffer for candidate node indices.
 * @param candidate_weights_buffer A buffer for candidate node weights.
 */
void add_subtrees_batch_cuda(
    torch::Tensor trees,
    torch::Tensor num_to_grow,
    int max_nodes,
    int max_new_nodes,
    torch::Tensor new_decision_nodes,
    torch::Tensor new_action_nodes,
    torch::Tensor action_root_branch_type,
    torch::Tensor bfs_queue_buffer,
    torch::Tensor result_indices_buffer,
    torch::Tensor child_count_buffer,
    torch::Tensor act_cnt_buffer,
    torch::Tensor dec_cnt_buffer,
    torch::Tensor candidate_indices_buffer,
    torch::Tensor candidate_weights_buffer
);

/**
 * @brief Deletes subtrees from a batch of trees.
 *
 * @param trees The population tensor.
 * @param mutate_mask_i32 A mask indicating which trees to mutate.
 * @param max_nodes The maximum number of nodes per tree.
 * @param alpha An exponent for weighting subtree size.
 * @param ensure_action_left A flag to ensure at least one action node is left.
 * @param child_count_buffer A buffer for child counts.
 * @param act_cnt_buffer A buffer for action child counts.
 * @param dec_cnt_buffer A buffer for decision child counts.
 * @param candidate_indices_buffer A buffer for candidate node indices.
 * @param candidate_weights_buffer A buffer for candidate node weights.
 * @param bfs_queue_buffer A buffer for the BFS queue.
 * @param result_indices_buffer A buffer to store the result of the BFS.
 * @param deletion_mask_buffer A buffer for the deletion mask.
 * @param repair_mask_buffer A buffer for the repair mask.
 * @param chosen_roots_buffer A buffer to store the chosen roots for deletion.
 */
void delete_subtrees_batch_cuda(
    torch::Tensor trees,
    torch::Tensor mutate_mask_i32,
    int max_nodes,
    float alpha,
    int ensure_action_left,
    torch::Tensor child_count_buffer,
    torch::Tensor act_cnt_buffer,
    torch::Tensor dec_cnt_buffer,
    torch::Tensor candidate_indices_buffer,
    torch::Tensor candidate_weights_buffer,
    torch::Tensor bfs_queue_buffer,
    torch::Tensor result_indices_buffer,
    torch::Tensor deletion_mask_buffer,
    torch::Tensor repair_mask_buffer,
    torch::Tensor chosen_roots_buffer
);

/**
 * @brief A critical repair function to ensure that no root branch is left without children.
 *
 * @param trees The population tensor.
 */
void critical_repair_batch_cuda(
    torch::Tensor trees
);