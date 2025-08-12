// /csrc/value_mutation_kernel.cuh (변경 없음)
#pragma once
#include <torch/extension.h>
void _launch_mutation_kernel_cpp(
    torch::Tensor& population, 
    bool is_reinitialize, 
    float mutation_prob,
    float noise_ratio, 
    int leverage_change,
    torch::Tensor& feature_num_indices, 
    torch::Tensor& feature_min_vals, 
    torch::Tensor& feature_max_vals,
    torch::Tensor& feature_comparison_indices, 
    torch::Tensor& feature_bool_indices
);