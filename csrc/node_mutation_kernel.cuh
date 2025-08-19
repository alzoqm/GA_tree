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