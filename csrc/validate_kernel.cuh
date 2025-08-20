// csrc/validate_kernel.cuh
#pragma once
#include <torch/extension.h>

// Validate a batch of trees on GPU and throw if violations are found.
// Input: trees (B, N, D=float32) with D == NODE_INFO_DIM
void validate_trees_or_throw_cuda(const torch::Tensor& trees);

