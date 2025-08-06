// csrc/predict_kernel.cu (수정됨)
#include <cuda_runtime.h>
#include "constants.h"

// [신규] 공유 메모리에 캐싱할 피처의 최대 개수.
// 실제 피처 수가 이보다 많으면 에러가 발생합니다. (C++ 래퍼에서 체크)
constexpr int MAX_FEATURES_IN_SHARED_MEM = 1024;

// --- Device-level Helper Function ---
__device__ bool evaluate_node_device(
    const float* node_data,
    const float* feature_values) { // 이제 feature_values는 공유 메모리 포인터가 됩니다.

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


// --- [수정] Main CUDA Kernel ---
__global__ void predict_kernel(
    const float* population_ptr,
    const float* features_ptr,
    const long* positions_ptr,
    const int* next_indices_ptr,
    // [신규] 인접 리스트 포인터
    const int* offset_ptr,
    const int* child_indices_ptr,
    float* results_ptr,
    int* bfs_queue_buffer,
    int pop_size,
    int max_nodes,
    int num_features) {

    // [신규] 공유 메모리 선언
    __shared__ float feature_cache[MAX_FEATURES_IN_SHARED_MEM];

    // 1. Thread-to-Tree Mapping
    const int tree_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tree_idx >= pop_size) {
        return;
    }
    
    // [신규] 블록의 첫 번째 스레드가 피처 값을 공유 메모리로 캐싱
    if (threadIdx.x < num_features) {
        feature_cache[threadIdx.x] = features_ptr[threadIdx.x];
    }
    __syncthreads(); // 블록 내 모든 스레드가 캐싱 완료까지 대기

    // 2. Setup pointers and variables for the current tree
    const float* tree_data = population_ptr + tree_idx * max_nodes * NODE_INFO_DIM;
    float* result_out = results_ptr + tree_idx * 4;
    const int next_idx = next_indices_ptr[tree_idx];
    // [신규] 현재 트리에 해당하는 오프셋 배열 부분 포인터 설정
    const int* tree_offset_ptr = offset_ptr + tree_idx * max_nodes;

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

    result_out[0] = ACTION_NOT_FOUND;
    result_out[1] = 0.0f;
    result_out[2] = 0.0f;
    result_out[3] = 0.0f;

    if (start_node_idx == -1) {
        return;
    }

    // 4. BFS를 위한 큐 관리 변수 설정
    int* bfs_queue = bfs_queue_buffer + tree_idx * max_nodes;
    int queue_head = 0;
    int queue_tail = 0;

    if (queue_tail < max_nodes) {
        bfs_queue[queue_tail++] = start_node_idx;
    }
    
    bool found_action = false;

    // 5. BFS 루프 시작
    while (queue_head < queue_tail && !found_action) {
        int current_node_idx = bfs_queue[queue_head++];

        // [수정] 비효율적인 선형 탐색을 인접 리스트 조회로 변경 (핵심 최적화)
        int start_offset = tree_offset_ptr[current_node_idx];
        int end_offset = tree_offset_ptr[current_node_idx + 1];

        for (int i = start_offset; i < end_offset; ++i) {
            int child_idx = child_indices_ptr[i];
            const float* child_node_data = tree_data + child_idx * NODE_INFO_DIM;
            int child_node_type = static_cast<int>(child_node_data[COL_NODE_TYPE]);

            if (child_node_type == NODE_TYPE_ACTION) {
                result_out[0] = child_node_data[COL_PARAM_1];
                result_out[1] = child_node_data[COL_PARAM_2];
                result_out[2] = child_node_data[COL_PARAM_3];
                result_out[3] = child_node_data[COL_PARAM_4];
                found_action = true;
                break; // Action을 찾았으므로 더 이상 자식을 볼 필요 없음
            }
            else if (child_node_type == NODE_TYPE_DECISION) {
                // [수정] 공유 메모리에 캐시된 피처 값을 사용
                if (evaluate_node_device(child_node_data, feature_cache)) {
                    if (queue_tail < max_nodes) {
                        bfs_queue[queue_tail++] = child_idx;
                    }
                }
            }
        }
    }
}

// --- [수정] Kernel Launcher ---
void launch_predict_kernel(
    const float* population_ptr,
    const float* features_ptr,
    const long* positions_ptr,
    const int* next_indices_ptr,
    // [신규] 인접 리스트 포인터
    const int* offset_ptr,
    const int* child_indices_ptr,
    float* results_ptr,
    int* bfs_queue_buffer_ptr,
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
        offset_ptr,
        child_indices_ptr,
        results_ptr,
        bfs_queue_buffer_ptr,
        pop_size,
        max_nodes,
        num_features
    );
}