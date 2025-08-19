// csrc/node_mutation_kernel.cuh
#pragma once
#include <torch/extension.h>

void add_decision_nodes_batch_cuda(
    torch::Tensor trees,                 // (B, max_nodes, node_dim) float32
    torch::Tensor num_to_add,            // (B,) int32
    int max_depth,
    int max_nodes,
    int max_add_nodes,                   // max_add_nodes per tree (for indexing)
    torch::Tensor out_new_node_indices,  // (B, max_add_nodes) int32, -1 on fail
    torch::Tensor bfs_queue_buffer,      // (B, 2*max_nodes) int32
    torch::Tensor result_indices_buffer, // (B, 2*max_nodes) int32
    torch::Tensor old_to_new_map_buffer  // (B, max_nodes) int32
);

void delete_nodes_batch_cuda(
    torch::Tensor trees,                    // (B, N, D) float32, in/out
    torch::Tensor num_to_delete,            // (B,) int32
    int max_children,
    int max_depth,
    int max_nodes,
    int max_delete_nodes,
    torch::Tensor deleted_nodes,            // (B, K) int32, initialized to -1
    torch::Tensor bfs_queue_buffer,         // (B, 2N) int32
    torch::Tensor result_indices_buffer,    // (B, 2N) int32
    torch::Tensor child_count_buffer,       // (B, N) int32
    torch::Tensor act_cnt_buffer,           // (B, N) int32
    torch::Tensor dec_cnt_buffer,           // (B, N) int32
    torch::Tensor candidate_indices_buffer, // (B, N) int32
    torch::Tensor candidate_weights_buffer  // (B, N) float32
);