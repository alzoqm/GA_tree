// csrc/predict_kernel.cuh
#pragma once

#include <torch/extension.h>

void launch_predict_kernel(
    const float* population_ptr,
    const float* features_ptr,
    const long* positions_ptr,
    const int* next_indices_ptr, // [수정] 파라미터 추가
    float* results_ptr,
    int pop_size,
    int max_nodes,
    int num_features
);