// csrc/predict_kernel.cu
#include <cuda_runtime.h>
#include "constants.h"

// --- Device-level Helper Function ---
// Python의 _evaluate_node와 동일한 로직
__device__ bool evaluate_node_device(
    const float* node_data,
    const float* feature_values) {

    int comp_type = static_cast<int>(node_data[COL_PARAM_3]);
    int feat1_idx = static_cast<int>(node_data[COL_PARAM_1]);
    float val1 = feature_values[feat1_idx];
    float val2 = node_data[COL_PARAM_4]; // feat-num 또는 feat-bool의 비교값

    // FEAT_FEAT 타입일 경우에만 val2를 피쳐 값에서 가져옴
    if (comp_type == COMP_TYPE_FEAT_FEAT) {
        int feat2_idx = static_cast<int>(node_data[COL_PARAM_4]);
        val2 = feature_values[feat2_idx];
    }
    
    // FEAT_BOOL 타입일 경우, 동등 비교만 수행
    if (comp_type == COMP_TYPE_FEAT_BOOL) {
        return val1 == val2;
    }

    // FEAT_NUM, FEAT_FEAT 타입일 경우, 연산자 기반 비교
    int op = static_cast<int>(node_data[COL_PARAM_2]);
    switch(op) {
        case OP_GTE: return val1 >= val2;
        case OP_LTE: return val1 <= val2;
    }

    return false;
}


// --- Main CUDA Kernel ---
// [전면 수정] Python의 predict (BFS) 메소드와 동일한 로직
__global__ void predict_kernel(
    const float* population_ptr,
    const float* features_ptr,
    const long* positions_ptr,
    const int* next_indices_ptr, // [추가] 각 트리의 실제 노드 수를 담은 포인터
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
    const float* features = features_ptr + tree_idx * num_features;
    float* result_out = results_ptr + tree_idx * 3; // (pos, size, leverage)
    const int next_idx = next_indices_ptr[tree_idx]; // [추가] 이 트리의 실제 노드 수

    // 3. Find the starting node (root branch)
    long start_pos_type = positions_ptr[tree_idx];
    int start_node_idx = -1;
    // Root branches는 항상 0, 1, 2에 위치
    for (int i = 0; i < 3; ++i) { 
        const float* node = tree_data + i * NODE_INFO_DIM;
        if (static_cast<int>(node[COL_NODE_TYPE]) == NODE_TYPE_ROOT_BRANCH &&
            static_cast<int>(node[COL_PARAM_1]) == start_pos_type) {
            start_node_idx = i;
            break;
        }
    }

    // 기본 HOLD 결과값 설정
    result_out[0] = ACTION_DEFAULT_HOLD; // Python에서 'HOLD'로 해석될 값
    result_out[1] = 0.0f;
    result_out[2] = 0.0f;

    if (start_node_idx == -1) {
        return; // 시작 노드를 못 찾으면 기본 HOLD 값으로 종료
    }

    // 4. BFS(너비 우선 탐색)를 위한 로컬 원형 큐
    // Python의 _bfs_queue와 동일한 역할
    int bfs_queue[2048]; // max_nodes에 맞춰 충분히 큰 크기 할당
    int queue_head = 0;
    int queue_tail = 0;

    // 4.1. 시작 노드 삽입(Enqueue)
    if (queue_tail < 2048) {
        bfs_queue[queue_tail++] = start_node_idx;
    }
    
    bool found_action = false;

    // 5. BFS 루프 시작 (큐가 빌 때까지)
    while (queue_head < queue_tail && !found_action) {
        // 5.1. 큐에서 현재 노드 추출(Dequeue)
        int current_node_idx = bfs_queue[queue_head++];

        // 5.2. 자식 노드 탐색 (성능 최적화: max_nodes 대신 next_idx까지 순회)
        for (int child_idx = 0; child_idx < next_idx; ++child_idx) {
            const float* child_node_data = tree_data + child_idx * NODE_INFO_DIM;

            if (static_cast<int>(child_node_data[COL_PARENT_IDX]) == current_node_idx) {
                int child_node_type = static_cast<int>(child_node_data[COL_NODE_TYPE]);

                // 5.2.1. Action 노드 발견 시: 즉시 결과 저장 및 탐색 종료
                if (child_node_type == NODE_TYPE_ACTION) {
                    result_out[0] = child_node_data[COL_PARAM_1]; // Position type
                    result_out[1] = child_node_data[COL_PARAM_2]; // Size
                    result_out[2] = child_node_data[COL_PARAM_3]; // Leverage
                    found_action = true;
                    break; // 자식 탐색 루프 종료
                }
                
                // 5.2.2. Decision 노드인 경우: 조건 평가 후 성공 시 큐에 추가
                else if (child_node_type == NODE_TYPE_DECISION) {
                    if (evaluate_node_device(child_node_data, features)) {
                        // 큐 오버플로우 방지
                        if (queue_tail < 2048) {
                            bfs_queue[queue_tail++] = child_idx; // Enqueue
                        }
                    }
                }
            }
        }
    }
    // 루프가 모두 종료될 때까지 Action을 못 찾았으면, 미리 설정된 HOLD 값이 유지됨
}

// --- Kernel Launcher ---
void launch_predict_kernel(
    const float* population_ptr,
    const float* features_ptr,
    const long* positions_ptr,
    const int* next_indices_ptr, // [추가]
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
        next_indices_ptr, // [추가]
        results_ptr,
        pop_size,
        max_nodes,
        num_features
    );
}