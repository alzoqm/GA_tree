// csrc/validate_kernel.cuh
// This header file declares the CUDA function for validating the structural integrity of the trees.

#pragma once
#include <torch/extension.h>

/**
 * @brief Validates a batch of trees on the GPU and throws an exception if any violations are found.
 *
 * @param trees The population tensor to be validated.
 */
void validate_trees_or_throw_cuda(const torch::Tensor& trees);

