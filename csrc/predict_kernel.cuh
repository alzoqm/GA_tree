// csrc/predict_kernel.cuh (수정됨)
#pragma once

#include <torch/extension.h>

void launch_predict_kernel(
    const float* population_ptr,
    const float* features_ptr,
    const long* positions_ptr,
    const int* next_indices_ptr,
    // [신규] 인접 리스트 포인터 추가
    const int* offset_ptr,
    const int* child_indices_ptr,
    float* results_ptr,
    int* bfs_queue_buffer_ptr,
    int pop_size,
    int max_nodes,
    int num_features
);