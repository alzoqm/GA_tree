// csrc/root_crossover_kernel.cuh
#pragma once
#include <torch/extension.h>

// Root branch crossover related CUDA kernel functions
void copy_branches_batch_cuda(
    torch::Tensor& child_batch,
    const torch::Tensor& p1_batch,
    const torch::Tensor& p2_batch,
    const torch::Tensor& donor_map,
    torch::Tensor& bfs_queue_buffer,
    torch::Tensor& result_indices_buffer,
    torch::Tensor& old_to_new_map_buffer
);

void repair_after_root_branch_cuda(
    torch::Tensor& trees,
    torch::Tensor& child_count_buffer,
    torch::Tensor& act_cnt_buffer,
    torch::Tensor& dec_cnt_buffer,
    torch::Tensor& bfs_queue_buffer,
    torch::Tensor& result_indices_buffer
);