// csrc/value_mutation_kernel.cuh
// This header file declares the C++ wrapper function for launching the value mutation kernel.

#pragma once
#include <torch/extension.h>

/**
 * @brief Launches the value mutation kernel.
 *
 * @param population The population tensor.
 * @param is_reinitialize A flag to indicate whether to reinitialize the node.
 * @param mutation_prob The mutation probability.
 * @param noise_ratio The noise ratio for the mutation.
 * @param leverage_change The change in leverage.
 * @param feature_num_indices The indices of numerical features.
 * @param feature_min_vals The minimum values of numerical features.
 * @param feature_max_vals The maximum values of numerical features.
 * @param feature_comparison_indices The indices of features for comparison.
 * @param feature_bool_indices The indices of boolean features.
 */
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