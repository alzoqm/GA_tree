// csrc/subtree_crossover_kernel.cuh
#pragma once
#include <torch/extension.h>

// Subtree crossover related CUDA kernel functions
void subtree_crossover_batch_cuda(
    torch::Tensor& child1_out,
    torch::Tensor& child2_out,
    const torch::Tensor& p1_batch,
    const torch::Tensor& p2_batch,
    int mode,
    int max_depth,
    int max_nodes,
    int max_retries,
    const torch::Tensor& branch_perm,
    torch::Tensor& bfs_queue_buffer,
    torch::Tensor& result_indices_buffer,
    torch::Tensor& old_to_new_map_buffer,
    torch::Tensor& p1_candidates_buffer,
    torch::Tensor& p2_candidates_buffer
);