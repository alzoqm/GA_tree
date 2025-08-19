// csrc/mutation_utils_kernel.cuh
#pragma once
#include <torch/extension.h>

// GPU variants of utils.py functions for batched operations
void find_subtree_nodes_batch_cuda(
    const torch::Tensor& trees,            // (B, max_nodes, node_dim)
    const torch::Tensor& root_indices,     // (B,) int32, -1 means skip
    int max_nodes,
    torch::Tensor bfs_queue_buffer,        // (B, 2*max_nodes) int32
    torch::Tensor result_indices_buffer,   // (B, 2*max_nodes) int32
    torch::Tensor out_counts               // (B,) int32: number of collected nodes per tree
);

void update_subtree_depth_batch_cuda(
    torch::Tensor trees,                   // in/out
    const torch::Tensor& root_indices,     // (B,) int32
    int delta,
    int max_nodes,
    torch::Tensor bfs_queue_buffer,        // (B, 2*max_nodes) int32
    torch::Tensor result_indices_buffer    // (B, 2*max_nodes) int32
);

void get_subtree_max_depth_batch_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& root_indices,     // (B,) int32
    int max_nodes,
    torch::Tensor bfs_queue_buffer,        // (B, 2*max_nodes) int32
    torch::Tensor result_indices_buffer,   // (B, 2*max_nodes) int32
    torch::Tensor out_max_depths           // (B,) int32
);

void find_empty_slots_batch_cuda(
    const torch::Tensor& trees,            // (B, max_nodes, node_dim)
    int count,                             // #slots per tree
    int max_nodes,
    torch::Tensor out_indices              // (B, count) int32, -1 fill if insufficient
);