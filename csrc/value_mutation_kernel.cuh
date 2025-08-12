// /csrc/value_mutation_kernel.cuh
#pragma once

#include <torch/extension.h>
#include <vector>
#include <map>
#include <string>

// [신규] NodeParamMutation을 위한 C++ 래퍼 함수 선언
void node_param_mutate_cuda(
    torch::Tensor population,
    float mutation_prob,
    float noise_ratio,
    int leverage_change,
    torch::Tensor feature_num_indices,
    torch::Tensor feature_min_vals,
    torch::Tensor feature_max_vals
);

// [신규] ReinitializeNodeMutation을 위한 C++ 래퍼 함수 선언
void reinitialize_node_mutate_cuda(
    torch::Tensor population,
    float mutation_prob,
    torch::Tensor feature_num_indices,
    torch::Tensor feature_min_vals,
    torch::Tensor feature_max_vals,
    torch::Tensor feature_comparison_indices,
    torch::Tensor feature_bool_indices
);