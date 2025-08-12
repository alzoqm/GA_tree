// /csrc/reorganize_kernel.cuh (변경 없음)
#pragma once
#include <torch/extension.h>
void reorganize_population_cuda(torch::Tensor population_tensor);