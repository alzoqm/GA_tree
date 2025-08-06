// csrc/predict_kernel.cuh
#pragma once

#include <torch/extension.h>

void launch_predict_kernel(
    const float* population_ptr,
    const float* features_ptr,
    const long* positions_ptr,
    const int* next_indices_ptr,
    float* results_ptr,
    int* bfs_queue_buffer_ptr, // [신규] BFS 큐 버퍼 포인터 추가
    int pop_size,
    int max_nodes,
    int num_features
);