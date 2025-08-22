// /csrc/reorganize_kernel.cuh (변경 없음)
#pragma once
#include <torch/extension.h>
void reorganize_population_cuda(torch::Tensor population_tensor);
void reorganize_population_with_arrays_cuda(
    torch::Tensor population_tensor,
    torch::Tensor active_counts_per_tree,
    torch::Tensor old_gid_to_new_gid_map
);