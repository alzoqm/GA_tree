// csrc/crossover_kernel.cuh
#pragma once
#include <torch/extension.h>

// C++ 래퍼 함수 (Python에서 호출)
void get_contextual_mask_cuda(
    const torch::Tensor& trees,
    torch::Tensor& output_mask,
    int node_type,
    int branch_type
);

void swap_node_params_cuda(
    torch::Tensor& c1,
    torch::Tensor& c2,
    const torch::Tensor& p1_mask,
    const torch::Tensor& p2_mask
);