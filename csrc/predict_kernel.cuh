// csrc/predict_kernel.cuh
#pragma once

#include <torch/extension.h>

void launch_predict_kernel(
    const float* population_ptr,
    const float* features_ptr,
    const long* positions_ptr,
    float* results_ptr,
    int pop_size,
    int max_nodes,
    int num_features
);