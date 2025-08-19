// csrc/subtree_mutation_kernel.cuh
#pragma once
#include <torch/extension.h>

// Batch AddSubtree mutation (one kernel). All large buffers are allocated in Python.
// Dtypes: trees=float32; num_to_grow=int32; outputs as documented below.
void add_subtrees_batch_cuda(
    torch::Tensor trees,                    // (B, N, D) float32
    torch::Tensor num_to_grow,              // (B,) int32
    int max_children,
    int max_depth,
    int max_nodes,
    int max_new_nodes,
    torch::Tensor new_decision_nodes,       // (B, K) int32, init -1
    torch::Tensor new_action_nodes,         // (B, K) int32, init -1
    torch::Tensor action_root_branch_type,  // (B, K) float32, init -1
    torch::Tensor bfs_queue_buffer,         // (B, 2*N) int32
    torch::Tensor result_indices_buffer,    // (B, 2*N) int32 (scratch: freelist and counters)
    torch::Tensor child_count_buffer,       // (B, N) int32
    torch::Tensor act_cnt_buffer,           // (B, N) int32
    torch::Tensor dec_cnt_buffer,           // (B, N) int32
    torch::Tensor candidate_indices_buffer, // (B, N) int32
    torch::Tensor candidate_weights_buffer  // (B, N) float32
);