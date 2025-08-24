#ifndef CROSSOVER_UTILS_CUH
#define CROSSOVER_UTILS_CUH

#include "crossover_kernel.cuh"
#include "constants.h"

// ==============================================================================
//                       BFS Helper Function for Subtree Collection
// ==============================================================================
__device__ inline int find_subtree_nodes_device(
    const float* tree_ptr,
    int root_idx,
    int* queue_buffer,      // 스레드별 BFS 큐
    int* result_buffer,     // 스레드별 결과 배열
    int max_nodes
) {
    if (root_idx < 0 || root_idx >= max_nodes) return 0;
    if ((int)tree_ptr[root_idx * NODE_INFO_DIM + COL_NODE_TYPE] == NODE_TYPE_UNUSED) return 0;

    int queue_front = 0;
    int queue_back = 0;
    int result_count = 0;

    // BFS 초기화: 루트 노드를 큐에 추가
    queue_buffer[queue_back++] = root_idx;

    while (queue_front < queue_back && result_count < max_nodes) {
        int current_idx = queue_buffer[queue_front++];
        result_buffer[result_count++] = current_idx;

        // 현재 노드의 모든 자식을 찾아 큐에 추가
        for (int i = 0; i < max_nodes; i++) {
            if ((int)tree_ptr[i * NODE_INFO_DIM + COL_PARENT_IDX] == current_idx &&
                (int)tree_ptr[i * NODE_INFO_DIM + COL_NODE_TYPE] != NODE_TYPE_UNUSED) {
                if (queue_back < max_nodes) {
                    queue_buffer[queue_back++] = i;
                }
            }
        }
    }

    return result_count;
}

#endif // CROSSOVER_UTILS_CUH