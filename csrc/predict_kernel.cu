// --- START OF FILE csrc/predict_kernel.cu ---

// csrc/predict_kernel.cu
#include <cuda_runtime.h>
#include "constants.h"

// --- Device-level Helper Function --- (수정 없음)
__device__ bool evaluate_node_device(
    const float* node_data,
    const float* feature_values) {

    int comp_type = static_cast<int>(node_data[COL_PARAM_3]);
    int feat1_idx = static_cast<int>(node_data[COL_PARAM_1]);
    float val1 = feature_values[feat1_idx];
    float val2 = node_data[COL_PARAM_4];

    if (comp_type == COMP_TYPE_FEAT_FEAT) {
        int feat2_idx = static_cast<int>(node_data[COL_PARAM_4]);
        val2 = feature_values[feat2_idx];
    }
    
    if (comp_type == COMP_TYPE_FEAT_BOOL) {
        return val1 == val2;
    }

    int op = static_cast<int>(node_data[COL_PARAM_2]);
    switch(op) {
        case OP_GTE: return val1 >= val2;
        case OP_LTE: return val1 <= val2;
    }

    return false;
}


// --- Main CUDA Kernel ---
__global__ void predict_kernel(
    const float* population_ptr,
    const float* features_ptr, // 이 포인터는 이제 (num_features) 크기의 1D 배열을 가리킵니다.
    const long* positions_ptr,
    const int* next_indices_ptr,
    float* results_ptr,
    int pop_size,
    int max_nodes,
    int num_features) {

    // 1. Thread-to-Tree Mapping
    const int tree_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tree_idx >= pop_size) {
        return;
    }

    // 2. Setup pointers and variables for the current tree
    const float* tree_data = population_ptr + tree_idx * max_nodes * NODE_INFO_DIM;
    
    // [수정] features 포인터를 인덱싱하지 않고 그대로 사용합니다.
    // 모든 스레드는 동일한 features 포인터를 공유합니다.
    const float* features = features_ptr;
    
    float* result_out = results_ptr + tree_idx * 4;
    const int next_idx = next_indices_ptr[tree_idx];

    // 3. Find the starting node (root branch)
    long start_pos_type = positions_ptr[tree_idx];
    int start_node_idx = -1;
    for (int i = 0; i < 3; ++i) { 
        const float* node = tree_data + i * NODE_INFO_DIM;
        if (static_cast<int>(node[COL_NODE_TYPE]) == NODE_TYPE_ROOT_BRANCH &&
            static_cast<int>(node[COL_PARAM_1]) == start_pos_type) {
            start_node_idx = i;
            break;
        }
    }

    // 기본 결과값(HOLD) 설정
    result_out[0] = ACTION_NOT_FOUND;
    result_out[1] = 0.0f;
    result_out[2] = 0.0f;
    result_out[3] = 0.0f;

    if (start_node_idx == -1) {
        return;
    }

    // 4. BFS(너비 우선 탐색)를 위한 로컬 원형 큐
    int bfs_queue[2048];
    int queue_head = 0;
    int queue_tail = 0;

    if (queue_tail < 2048) {
        bfs_queue[queue_tail++] = start_node_idx;
    }
    
    bool found_action = false;

    // 5. BFS 루프 시작 (수정 없음)
    while (queue_head < queue_tail && !found_action) {
        int current_node_idx = bfs_queue[queue_head++];

        for (int child_idx = 0; child_idx < next_idx; ++child_idx) {
            const float* child_node_data = tree_data + child_idx * NODE_INFO_DIM;

            if (static_cast<int>(child_node_data[COL_PARENT_IDX]) == current_node_idx) {
                int child_node_type = static_cast<int>(child_node_data[COL_NODE_TYPE]);

                if (child_node_type == NODE_TYPE_ACTION) {
                    result_out[0] = child_node_data[COL_PARAM_1];
                    result_out[1] = child_node_data[COL_PARAM_2];
                    result_out[2] = child_node_data[COL_PARAM_3];
                    result_out[3] = child_node_data[COL_PARAM_4];
                    found_action = true;
                    break;
                }
                
                else if (child_node_type == NODE_TYPE_DECISION) {
                    if (evaluate_node_device(child_node_data, features)) {
                        if (queue_tail < 2048) {
                            bfs_queue[queue_tail++] = child_idx;
                        }
                    }
                }
            }
        }
    }
}

// --- Kernel Launcher --- (수정 없음)
void launch_predict_kernel(
    const float* population_ptr,
    const float* features_ptr,
    const long* positions_ptr,
    const int* next_indices_ptr,
    float* results_ptr,
    int pop_size,
    int max_nodes,
    int num_features) {

    if (pop_size == 0) return;

    const int threads_per_block = 256;
    const int num_blocks = (pop_size + threads_per_block - 1) / threads_per_block;

    predict_kernel<<<num_blocks, threads_per_block>>>(
        population_ptr,
        features_ptr,
        positions_ptr,
        next_indices_ptr,
        results_ptr,
        pop_size,
        max_nodes,
        num_features
    );
}

// --- END OF FILE csrc/predict_kernel.cu ---