// csrc/crossover_kernel.cu
#include "crossover_kernel.cuh"
#include "constants.h"
#include <curand_kernel.h>

// ==============================================================================
//           커널 1: 컨텍스트 마스크 생성 (Contextual Mask Generation)
// ==============================================================================
__global__ void get_contextual_mask_kernel(
    const float* trees_ptr,
    bool* output_mask_ptr,
    int batch_size,
    int max_nodes,
    int node_type_target,
    int branch_type_target)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_nodes = batch_size * max_nodes;
    if (gid >= total_nodes) return;

    const int batch_idx = gid / max_nodes;
    const int node_idx = gid % max_nodes;

    const float* node_data = trees_ptr + gid * NODE_INFO_DIM;
    
    // 1. 노드 타입이 일치하는지 확인
    if ((int)node_data[COL_NODE_TYPE] == node_type_target) {
        
        // 2. 루트까지 부모를 거슬러 올라가기
        int current_idx = node_idx;
        int parent_idx = (int)node_data[COL_PARENT_IDX];

        while (parent_idx != -1) {
            current_idx = parent_idx;
            parent_idx = (int)trees_ptr[(batch_idx * max_nodes + current_idx) * NODE_INFO_DIM + COL_PARENT_IDX];
        }

        // 3. 루트 분기 타입이 일치하는지 확인
        const float* root_data = trees_ptr + (batch_idx * max_nodes + current_idx) * NODE_INFO_DIM;
        if ((int)root_data[COL_PARAM_1] == branch_type_target) {
            output_mask_ptr[gid] = true;
        }
    }
}

// ==============================================================================
//         커널 2: 노드 파라미터 교환 (Node Parameter Swap)
// ==============================================================================
__global__ void swap_node_params_kernel(
    float* c1_ptr,
    float* c2_ptr,
    const bool* p1_mask_ptr,
    const bool* p2_mask_ptr,
    int batch_size,
    int max_nodes)
{
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // --- 1. 교환 가능한 노드 수 계산 ---
    int p1_count = 0;
    int p2_count = 0;
    for (int i = 0; i < max_nodes; ++i) {
        if (p1_mask_ptr[batch_idx * max_nodes + i]) p1_count++;
        if (p2_mask_ptr[batch_idx * max_nodes + i]) p2_count++;
    }

    int max_swaps = min(p1_count, p2_count);
    if (max_swaps == 0) return;

    // --- 2. 교환할 개수 'k' 결정 ---
    curandState state;
    curand_init(batch_idx, 0, 0, &state);
    int k_upper = max(1, max_swaps / 2);
    int k = (int)(curand_uniform(&state) * k_upper) + 1;
    k = min(k, max_swaps);

    // --- 3. 교환 수행 (단일 스레드 내 루프) ---
    // 이 방식은 병렬성은 낮지만, 구현이 간단하고 메모리 오류 위험이 적습니다.
    for (int i = 0; i < k; ++i) {
        // p1에서 랜덤 인덱스 선택
        int p1_rand_n = (int)(curand_uniform(&state) * p1_count);
        int p1_swap_idx = -1;
        int current_n = 0;
        for (int j = 0; j < max_nodes; ++j) {
            if (p1_mask_ptr[batch_idx * max_nodes + j]) {
                if (current_n == p1_rand_n) {
                    p1_swap_idx = j;
                    break;
                }
                current_n++;
            }
        }

        // p2에서 랜덤 인덱스 선택
        int p2_rand_n = (int)(curand_uniform(&state) * p2_count);
        int p2_swap_idx = -1;
        current_n = 0;
        for (int j = 0; j < max_nodes; ++j) {
            if (p2_mask_ptr[batch_idx * max_nodes + j]) {
                if (current_n == p2_rand_n) {
                    p2_swap_idx = j;
                    break;
                }
                current_n++;
            }
        }

        if (p1_swap_idx != -1 && p2_swap_idx != -1) {
            // 파라미터(COL_PARAM_1 ~ 끝) 교환
            for (int param_col = COL_PARAM_1; param_col < NODE_INFO_DIM; ++param_col) {
                float* p1_addr = c1_ptr + (batch_idx * max_nodes + p1_swap_idx) * NODE_INFO_DIM + param_col;
                float* p2_addr = c2_ptr + (batch_idx * max_nodes + p2_swap_idx) * NODE_INFO_DIM + param_col;
                
                float temp = *p1_addr;
                *p1_addr = *p2_addr;
                *p2_addr = temp;
            }
        }
    }
}

__device__ int find_subtree_nodes_device(
    const float* tree_ptr,
    int root_idx,
    int* queue_buffer,      // 스레드별 BFS 큐
    int* result_indices,    // 스레드별 결과 저장 버퍼
    int max_nodes)
{
    if (tree_ptr[root_idx * NODE_INFO_DIM + COL_NODE_TYPE] == NODE_TYPE_UNUSED) {
        return 0;
    }
    
    int head = 0, tail = 0;
    queue_buffer[tail++] = root_idx;
    result_indices[0] = root_idx;
    int count = 1;

    while (head < tail) {
        int current_idx = queue_buffer[head++];
        
        // 부모-자식 관계는 전역 메모리를 순회하며 찾아야 함
        for (int i = 0; i < max_nodes; ++i) {
            if ((int)tree_ptr[i * NODE_INFO_DIM + COL_PARENT_IDX] == current_idx) {
                if (tail < max_nodes) {
                    queue_buffer[tail++] = i;
                    result_indices[count++] = i;
                } else {
                    // 큐 버퍼가 꽉 차면 탐색 중단 (에러 방지)
                    return count;
                }
            }
        }
    }
    return count;
}


// ==============================================================================
//       [신규] 커널 3: RootBranchCrossover를 위한 배치 복사 커널
// ==============================================================================
__global__ void copy_branches_kernel(
    float* child_batch_ptr,
    const float* p1_batch_ptr,
    const float* p2_batch_ptr,
    const int* donor_map_ptr,
    int* bfs_queue_buffer_ptr,      // <--- [수정]
    int* result_indices_buffer_ptr, // <--- [수정]
    int* old_to_new_map_buffer_ptr, // <--- [수정]
    int batch_size,
    int max_nodes)
{
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // --- 1. 스레드별 포인터 및 버퍼 설정 ---
    float* child_ptr = child_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    const float* p1_ptr = p1_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    const float* p2_ptr = p2_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    
    // [수정] 각 버퍼에 대한 포인터를 직접 계산
    int* my_queue = bfs_queue_buffer_ptr + batch_idx * max_nodes;
    int* my_results = result_indices_buffer_ptr + batch_idx * max_nodes;
    int* my_old_to_new_map = old_to_new_map_buffer_ptr + batch_idx * max_nodes;

    int child_next_idx = 3; // 루트 분기(0,1,2) 다음부터 채움

    // --- 2. 3개의 브랜치(LONG, HOLD, SHORT)에 대해 순차적으로 복사 ---
    for (int b_idx = 0; b_idx < 3; ++b_idx) {
        int donor_choice = donor_map_ptr[batch_idx * 3 + b_idx];
        const float* donor_ptr = (donor_choice == 0) ? p1_ptr : p2_ptr;

        // --- 3. 기증 부모에서 해당 브랜치의 직계 자식(서브트리 루트) 찾기 ---
        for (int donor_node_idx = 0; donor_node_idx < max_nodes; ++donor_node_idx) {
            if ((int)donor_ptr[donor_node_idx * NODE_INFO_DIM + COL_PARENT_IDX] == b_idx) {
                int subtree_root_idx = donor_node_idx;
                
                // --- 4. 서브트리 정보 수집 및 유효성 검사 ---
                int subtree_size = find_subtree_nodes_device(donor_ptr, subtree_root_idx, my_queue, my_results, max_nodes);

                if (child_next_idx + subtree_size > max_nodes) {
                    continue; // 공간 부족 시 이 서브트리는 건너뜀
                }

                // --- 5. 서브트리 복사 ---
                // a. old_to_new_map 생성 (전체 맵 초기화)
                for(int i=0; i<max_nodes; ++i) my_old_to_new_map[i] = -1;
                for (int k = 0; k < subtree_size; ++k) {
                    my_old_to_new_map[my_results[k]] = child_next_idx + k;
                }

                // b. 깊이 오프셋 계산
                float dest_parent_depth = child_ptr[b_idx * NODE_INFO_DIM + COL_DEPTH];
                float source_root_depth = donor_ptr[subtree_root_idx * NODE_INFO_DIM + COL_DEPTH];
                float depth_offset = (dest_parent_depth + 1) - source_root_depth;

                // c. 노드 데이터 복사 및 업데이트
                for (int k = 0; k < subtree_size; ++k) {
                    int old_idx = my_results[k];
                    int new_idx = my_old_to_new_map[old_idx];
                    
                    const float* src_node_data = donor_ptr + old_idx * NODE_INFO_DIM;
                    float* dest_node_data = child_ptr + new_idx * NODE_INFO_DIM;

                    for(int d=0; d<NODE_INFO_DIM; ++d) dest_node_data[d] = src_node_data[d];

                    dest_node_data[COL_DEPTH] += depth_offset;
                    
                    int old_parent_idx = (int)src_node_data[COL_PARENT_IDX];
                    if (old_idx == subtree_root_idx) {
                        dest_node_data[COL_PARENT_IDX] = (float)b_idx; // 새 부모는 루트 분기
                    } else {
                        dest_node_data[COL_PARENT_IDX] = (float)my_old_to_new_map[old_parent_idx];
                    }
                }
                child_next_idx += subtree_size;
            }
        }
    }
}


// ==============================================================================
//                       C++ 래퍼 함수 (커널 런처)
// ==============================================================================
void get_contextual_mask_cuda(const torch::Tensor& trees, torch::Tensor& output_mask, int node_type, int branch_type) {
    const int batch_size = trees.size(0);
    const int max_nodes = trees.size(1);
    const int total_nodes = batch_size * max_nodes;
    if (total_nodes == 0) return;

    const int threads = 256;
    const int blocks = (total_nodes + threads - 1) / threads;
    
    get_contextual_mask_kernel<<<blocks, threads>>>(
        trees.data_ptr<float>(),
        output_mask.data_ptr<bool>(),
        batch_size, max_nodes, node_type, branch_type);
}

void swap_node_params_cuda(torch::Tensor& c1, torch::Tensor& c2, const torch::Tensor& p1_mask, const torch::Tensor& p2_mask) {
    const int batch_size = c1.size(0);
    if (batch_size == 0) return;
    const int max_nodes = c1.size(1);

    swap_node_params_kernel<<<batch_size, 1>>>(
        c1.data_ptr<float>(),
        c2.data_ptr<float>(),
        p1_mask.data_ptr<bool>(),
        p2_mask.data_ptr<bool>(),
        batch_size, max_nodes);
}

void copy_branches_batch_cuda(
    torch::Tensor& child_batch,
    const torch::Tensor& p1_batch,
    const torch::Tensor& p2_batch,
    const torch::Tensor& donor_map,
    torch::Tensor& bfs_queue_buffer,      // <--- [수정]
    torch::Tensor& result_indices_buffer, // <--- [수정]
    torch::Tensor& old_to_new_map_buffer  // <--- [수정]
)
{
    const int batch_size = child_batch.size(0);
    if (batch_size == 0) return;
    const int max_nodes = child_batch.size(1);

    // 각 스레드 블록이 하나의 자식을 처리
    copy_branches_kernel<<<batch_size, 1>>>(
        child_batch.data_ptr<float>(),
        p1_batch.data_ptr<float>(),
        p2_batch.data_ptr<float>(),
        donor_map.data_ptr<int>(),
        bfs_queue_buffer.data_ptr<int>(),      // <--- [수정]
        result_indices_buffer.data_ptr<int>(), // <--- [수정]
        old_to_new_map_buffer.data_ptr<int>(), // <--- [수정]
        batch_size,
        max_nodes
    );
}