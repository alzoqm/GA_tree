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
    if (root_idx < 0 || root_idx >= max_nodes || tree_ptr[root_idx * NODE_INFO_DIM + COL_NODE_TYPE] == NODE_TYPE_UNUSED) {
        return 0;
    }
    
    int head = 0, tail = 0;
    queue_buffer[tail++] = root_idx;
    result_indices[0] = root_idx;
    int count = 1;

    while (head < tail) {
        int current_idx = queue_buffer[head++];
        
        for (int i = 0; i < max_nodes; ++i) {
            if ((int)tree_ptr[i * NODE_INFO_DIM + COL_PARENT_IDX] == current_idx) {
                if (tail < max_nodes) { // 버퍼 오버플로우 방지
                    queue_buffer[tail++] = i;
                    if (count < max_nodes) { // 버퍼 오버플로우 방지
                        result_indices[count++] = i;
                    }
                } else {
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
    int* bfs_queue_buffer_ptr,
    int* result_indices_buffer_ptr,
    int* old_to_new_map_buffer_ptr,
    int batch_size,
    int max_nodes)
{
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // [안전성] child 버퍼 중 non-root 노드만 UNUSED로 초기화하여 잔존 쓰레기 노드 방지
    // 루트 브랜치(0,1,2)는 Python에서 이미 설정되었으므로 건드리지 않음
    float* child_ptr_init = child_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    for (int i = 3; i < max_nodes; ++i) {  // Start from index 3, skip root branches (0,1,2)
        float* nd = child_ptr_init + i * NODE_INFO_DIM;
        nd[COL_NODE_TYPE] = NODE_TYPE_UNUSED;
        // 부모/깊이/파라미터는 필요 시 이후 복사에서 덮어씀
    }

    // --- 1. 스레드별 포인터 및 버퍼 설정 ---
    float* child_ptr = child_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    const float* p1_ptr = p1_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    const float* p2_ptr = p2_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    
    int* my_queue = bfs_queue_buffer_ptr + batch_idx * max_nodes;
    int* my_results = result_indices_buffer_ptr + batch_idx * max_nodes;
    int* my_old_to_new_map = old_to_new_map_buffer_ptr + batch_idx * max_nodes;

    int child_next_idx = 3; 

    // --- 2. 3개의 브랜치(LONG, HOLD, SHORT)에 대해 순차적으로 복사 ---
    for (int b_idx = 0; b_idx < 3; ++b_idx) {
        int donor_choice = donor_map_ptr[batch_idx * 3 + b_idx];
        const float* donor_ptr = (donor_choice == 0) ? p1_ptr : p2_ptr;

        for (int donor_node_idx = 0; donor_node_idx < max_nodes; ++donor_node_idx) {
            if ((int)donor_ptr[donor_node_idx * NODE_INFO_DIM + COL_PARENT_IDX] == b_idx) {
                int subtree_root_idx = donor_node_idx;
                
                int subtree_size = find_subtree_nodes_device(donor_ptr, subtree_root_idx, my_queue, my_results, max_nodes);

                if (child_next_idx + subtree_size > max_nodes) {
                    continue; 
                }

                for(int i=0; i<max_nodes; ++i) my_old_to_new_map[i] = -1;
                for (int k = 0; k < subtree_size; ++k) {
                    my_old_to_new_map[my_results[k]] = child_next_idx + k;
                }

                float dest_parent_depth = child_ptr[b_idx * NODE_INFO_DIM + COL_DEPTH];
                float source_root_depth = donor_ptr[subtree_root_idx * NODE_INFO_DIM + COL_DEPTH];
                float depth_offset = (dest_parent_depth + 1) - source_root_depth;

                for (int k = 0; k < subtree_size; ++k) {
                    int old_idx = my_results[k];
                    int new_idx = my_old_to_new_map[old_idx];
                    
                    const float* src_node_data = donor_ptr + old_idx * NODE_INFO_DIM;
                    float* dest_node_data = child_ptr + new_idx * NODE_INFO_DIM;

                    for(int d=0; d<NODE_INFO_DIM; ++d) dest_node_data[d] = src_node_data[d];

                    dest_node_data[COL_DEPTH] += depth_offset;
                    
                    int old_parent_idx = (int)src_node_data[COL_PARENT_IDX];
                    if (old_idx == subtree_root_idx) {
                        dest_node_data[COL_PARENT_IDX] = (float)b_idx;
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
//       [수정된] 커널 4: SubtreeCrossover를 위한 배치 커널 및 헬퍼 함수
// ==============================================================================

__device__ int get_root_branch_type_device(const float* tree_ptr, int node_idx, int max_nodes) {
    int current_idx = node_idx;
    while (current_idx >= 0 && current_idx < max_nodes) {
        int parent_idx = (int)tree_ptr[current_idx * NODE_INFO_DIM + COL_PARENT_IDX];
        if (parent_idx == -1) {
            return (int)tree_ptr[current_idx * NODE_INFO_DIM + COL_PARAM_1];
        }
        current_idx = parent_idx;
    }
    return -1; // Error or root not found
}

__device__ int get_active_node_count(const float* tree_ptr, int max_nodes) {
    int count = 0;
    for (int i = 0; i < max_nodes; ++i) {
        if (tree_ptr[i * NODE_INFO_DIM + COL_NODE_TYPE] != NODE_TYPE_UNUSED) {
            count++;
        }
    }
    return count;
}

__device__ int get_max_relative_depth(const float* tree_ptr, int* indices, int count, int max_nodes) {
    if (count == 0) return 0;
    int root_node_idx = indices[0];
    if (root_node_idx < 0 || root_node_idx >= max_nodes) return 0;

    float root_depth = tree_ptr[root_node_idx * NODE_INFO_DIM + COL_DEPTH];
    float max_abs_depth = root_depth;
    for (int i = 1; i < count; ++i) {
        int current_node_idx = indices[i];
        if (current_node_idx < 0 || current_node_idx >= max_nodes) continue;
        float current_depth = tree_ptr[current_node_idx * NODE_INFO_DIM + COL_DEPTH];
        if (current_depth > max_abs_depth) {
            max_abs_depth = current_depth;
        }
    }
    return (int)(max_abs_depth - root_depth);
}

__device__ bool would_violate_tree_structure(
    const float* child_ptr, int parent_idx, int new_subtree_root_type, int max_nodes)
{
    if (parent_idx < 0 || parent_idx >= max_nodes) return true;
    
    // Count existing children by type
    int action_children = 0;
    int decision_children = 0;
    
    for (int i = 0; i < max_nodes; ++i) {
        if ((int)child_ptr[i * NODE_INFO_DIM + COL_PARENT_IDX] == parent_idx) {
            int child_type = (int)child_ptr[i * NODE_INFO_DIM + COL_NODE_TYPE];
            if (child_type == NODE_TYPE_ACTION) {
                action_children++;
            } else if (child_type == NODE_TYPE_DECISION) {
                decision_children++;
            }
        }
    }
    
    // Check if adding new subtree would violate rules
    if (new_subtree_root_type == NODE_TYPE_ACTION) {
        // Rule 1: No mixed children (action + decision)
        if (decision_children > 0) return true;
        
        // Rule 2: Parent with action child must have exactly one child
        if (action_children > 0) return true; // Already has action child, can't add another
    } else if (new_subtree_root_type == NODE_TYPE_DECISION) {
        // Rule 1: No mixed children (action + decision)  
        if (action_children > 0) return true;
    }
    
    return false;
}

__device__ bool transplant_one_way_device(
    float* child_ptr, const float* recipient_ptr, const float* donor_ptr,
    int r_idx, int d_idx, int* r_indices, int r_count, int* d_indices, int d_count,
    int* my_old_to_new_map, int* my_empty_slots_buffer, int max_nodes)
{
    // 1. recipient로부터 child를 초기화
    for(int i=0; i < max_nodes * NODE_INFO_DIM; ++i) child_ptr[i] = recipient_ptr[i];

    // 2. child에서 제거될 서브트리 영역을 비움
    for (int i = 0; i < r_count; ++i) {
        int idx_to_clear = r_indices[i];
        if (idx_to_clear >= 0 && idx_to_clear < max_nodes) {
            child_ptr[idx_to_clear * NODE_INFO_DIM + COL_NODE_TYPE] = NODE_TYPE_UNUSED;
        }
    }

    // 3. 비어있는 슬롯 찾기
    int empty_count = 0;
    for (int i = 0; i < max_nodes && empty_count < d_count; ++i) {
        if (child_ptr[i * NODE_INFO_DIM + COL_NODE_TYPE] == NODE_TYPE_UNUSED) {
            my_empty_slots_buffer[empty_count++] = i;
        }
    }
    if (empty_count < d_count) return false; // 공간 부족 시 이식 중단

    // 4. old_to_new_map 생성
    for (int i = 0; i < max_nodes; ++i) my_old_to_new_map[i] = -1;
    for (int i = 0; i < d_count; ++i) {
        my_old_to_new_map[d_indices[i]] = my_empty_slots_buffer[i];
    }
    
    // 5. 깊이 오프셋 계산
    int r_parent_idx = (int)recipient_ptr[r_idx * NODE_INFO_DIM + COL_PARENT_IDX];
    if(r_parent_idx < 0 || r_parent_idx >= max_nodes) return false; // 유효하지 않은 부모
    float insertion_depth = recipient_ptr[r_parent_idx * NODE_INFO_DIM + COL_DEPTH] + 1.0f;
    float depth_offset = insertion_depth - donor_ptr[d_idx * NODE_INFO_DIM + COL_DEPTH];

    // 5.5. Check if transplantation would violate tree structure rules
    int donor_root_type = (int)donor_ptr[d_idx * NODE_INFO_DIM + COL_NODE_TYPE];
    if (would_violate_tree_structure(child_ptr, r_parent_idx, donor_root_type, max_nodes)) {
        return false; // Abort transplantation to avoid invalid structure
    }

    // 6. 노드 복사 및 업데이트
    for (int i = 0; i < d_count; ++i) {
        int old_idx = d_indices[i];
        int new_idx = my_old_to_new_map[old_idx];
        if (new_idx == -1) continue;

        const float* src_node = donor_ptr + old_idx * NODE_INFO_DIM;
        float* dest_node = child_ptr + new_idx * NODE_INFO_DIM;

        for(int d=0; d<NODE_INFO_DIM; ++d) dest_node[d] = src_node[d];
        dest_node[COL_DEPTH] += depth_offset;
        
        int old_parent_idx = (int)src_node[COL_PARENT_IDX];
        if (old_idx == d_idx) {
            dest_node[COL_PARENT_IDX] = (float)r_parent_idx;
        } else {
            dest_node[COL_PARENT_IDX] = (float)my_old_to_new_map[old_parent_idx];
        }
    }
    return true; // Success
}


__global__ void subtree_crossover_kernel(
    float* child1_out_ptr, float* child2_out_ptr,
    const float* p1_batch_ptr, const float* p2_batch_ptr,
    int mode, int max_depth, int max_nodes, int max_retries,
    const int* branch_perm_ptr,
    int* bfs_queue_buffer_ptr, int* result_indices_buffer_ptr, int* old_to_new_map_buffer_ptr,
    int* p1_candidates_buffer_ptr, int* p2_candidates_buffer_ptr,
    int batch_size)
{
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // --- 1. 스레드별 포인터 및 버퍼 설정 ---
    float* c1_ptr = child1_out_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    float* c2_ptr = child2_out_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    const float* p1_ptr = p1_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    const float* p2_ptr = p2_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;

    // [중요 수정] p1, p2 서브트리를 동시에 담기 위해 버퍼를 2*max_nodes로 사용
    //   - 래퍼에서 bfs_queue_buffer/result_indices_buffer는 [batch, 2*max_nodes]로 할당되어야 함
    int* my_queue1   = bfs_queue_buffer_ptr     + batch_idx * (2 * max_nodes);
    int* my_queue2   = my_queue1                + max_nodes;
    int* my_results1 = result_indices_buffer_ptr+ batch_idx * (2 * max_nodes);
    int* my_results2 = my_results1              + max_nodes;

    int* my_old_to_new_map = old_to_new_map_buffer_ptr + batch_idx * max_nodes;
    int* p1_candidates = p1_candidates_buffer_ptr + batch_idx * max_nodes;
    int* p2_candidates = p2_candidates_buffer_ptr + batch_idx * max_nodes;

    curandState state;
    curand_init(batch_idx, 0, 0, &state);

    bool success = false;
    for (int retry = 0; retry < max_retries && !success; ++retry) {
        // --- 2. 교차 후보군 선택 ---
        int p1_cand_count = 0, p2_cand_count = 0;
        
        if (mode == 0) { // free mode
            for(int i=0; i<max_nodes; ++i) {
                if(p1_ptr[i*NODE_INFO_DIM + COL_PARENT_IDX] != -1) p1_candidates[p1_cand_count++] = i;
                if(p2_ptr[i*NODE_INFO_DIM + COL_PARENT_IDX] != -1) p2_candidates[p2_cand_count++] = i;
            }
        } else { // context mode
            int branch_type_to_try = branch_perm_ptr[batch_idx * 3 + (retry % 3)];
            for(int i=0; i<max_nodes; ++i) {
                if(p1_ptr[i*NODE_INFO_DIM + COL_PARENT_IDX] != -1 && get_root_branch_type_device(p1_ptr, i, max_nodes) == branch_type_to_try) p1_candidates[p1_cand_count++] = i;
                if(p2_ptr[i*NODE_INFO_DIM + COL_PARENT_IDX] != -1 && get_root_branch_type_device(p2_ptr, i, max_nodes) == branch_type_to_try) p2_candidates[p2_cand_count++] = i;
            }
        }

        if (p1_cand_count == 0 || p2_cand_count == 0) continue;
        
        int p1_idx = p1_candidates[(int)(curand_uniform(&state) * p1_cand_count)];
        int p2_idx = p2_candidates[(int)(curand_uniform(&state) * p2_cand_count)];

        // --- 3. 서브트리 정보 수집 ---
        int s1_count = find_subtree_nodes_device(p1_ptr, p1_idx, my_queue1, my_results1, max_nodes);
        int s2_count = find_subtree_nodes_device(p2_ptr, p2_idx, my_queue2, my_results2, max_nodes);

        int p1_total_nodes = get_active_node_count(p1_ptr, max_nodes);
        int p2_total_nodes = get_active_node_count(p2_ptr, max_nodes);
        
        // --- 4. 제약 조건 검증 ---
        int p1_parent_idx = (int)p1_ptr[p1_idx*NODE_INFO_DIM + COL_PARENT_IDX];
        if(p1_parent_idx < 0 || p1_parent_idx >= max_nodes) continue;
        float p1_ins_depth = p1_ptr[p1_parent_idx*NODE_INFO_DIM + COL_DEPTH] + 1;
        if(p1_ins_depth + get_max_relative_depth(p2_ptr, my_results2, s2_count, max_nodes) > max_depth) continue;
        if(p1_total_nodes - s1_count + s2_count > max_nodes) continue;

        int p2_parent_idx = (int)p2_ptr[p2_idx*NODE_INFO_DIM + COL_PARENT_IDX];
        if(p2_parent_idx < 0 || p2_parent_idx >= max_nodes) continue;
        float p2_ins_depth = p2_ptr[p2_parent_idx*NODE_INFO_DIM + COL_DEPTH] + 1;
        if(p2_ins_depth + get_max_relative_depth(p1_ptr, my_results1, s1_count, max_nodes) > max_depth) continue;
        if(p2_total_nodes - s2_count + s1_count > max_nodes) continue;

        // --- 5. 이식 수행 ---
        bool transplant1_success = transplant_one_way_device(c1_ptr, p1_ptr, p2_ptr,
            p1_idx, p2_idx,
            my_results1, s1_count,
            my_results2, s2_count,
            my_old_to_new_map, my_queue1, max_nodes);

        bool transplant2_success = transplant_one_way_device(c2_ptr, p2_ptr, p1_ptr,
            p2_idx, p1_idx,
            my_results2, s2_count,
            my_results1, s1_count,
            my_old_to_new_map, my_queue2, max_nodes);
        
        success = transplant1_success && transplant2_success;
    }

    if (!success) {
        for(int i=0; i < max_nodes * NODE_INFO_DIM; ++i) c1_ptr[i] = p1_ptr[i];
        for(int i=0; i < max_nodes * NODE_INFO_DIM; ++i) c2_ptr[i] = p2_ptr[i];
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
}

void copy_branches_batch_cuda(
    torch::Tensor& child_batch,
    const torch::Tensor& p1_batch,
    const torch::Tensor& p2_batch,
    const torch::Tensor& donor_map,
    torch::Tensor& bfs_queue_buffer,
    torch::Tensor& result_indices_buffer,
    torch::Tensor& old_to_new_map_buffer
)
{
    const int batch_size = child_batch.size(0);
    if (batch_size == 0) return;
    const int max_nodes = child_batch.size(1);

    copy_branches_kernel<<<batch_size, 1>>>(
        child_batch.data_ptr<float>(),
        p1_batch.data_ptr<float>(),
        p2_batch.data_ptr<float>(),
        donor_map.data_ptr<int>(),
        bfs_queue_buffer.data_ptr<int>(),
        result_indices_buffer.data_ptr<int>(),
        old_to_new_map_buffer.data_ptr<int>(),
        batch_size,
        max_nodes
    );
    cudaDeviceSynchronize();
}

// [수정된] SubtreeCrossover 래퍼 함수
void subtree_crossover_batch_cuda(
    torch::Tensor& child1_out,
    torch::Tensor& child2_out,
    const torch::Tensor& p1_batch,
    const torch::Tensor& p2_batch,
    int mode,
    int max_depth,
    int max_nodes,
    int max_retries,
    const torch::Tensor& branch_perm,
    torch::Tensor& bfs_queue_buffer,
    torch::Tensor& result_indices_buffer,
    torch::Tensor& old_to_new_map_buffer,
    torch::Tensor& p1_candidates_buffer,
    torch::Tensor& p2_candidates_buffer)
{
    const int batch_size = p1_batch.size(0);
    if (batch_size == 0) return;

    // [중요] 버퍼 크기 검증: 큐/결과는 2*max_nodes, 그 외는 >= max_nodes
    TORCH_CHECK(bfs_queue_buffer.dim() >= 2 && bfs_queue_buffer.size(1) >= 2*max_nodes,
                "bfs_queue_buffer must have second dim >= 2*max_nodes");
    TORCH_CHECK(result_indices_buffer.dim() >= 2 && result_indices_buffer.size(1) >= 2*max_nodes,
                "result_indices_buffer must have second dim >= 2*max_nodes");
    TORCH_CHECK(old_to_new_map_buffer.dim() >= 2 && old_to_new_map_buffer.size(1) >= max_nodes,
                "old_to_new_map_buffer must have second dim >= max_nodes");
    TORCH_CHECK(p1_candidates_buffer.dim() >= 2 && p1_candidates_buffer.size(1) >= max_nodes,
                "p1_candidates_buffer must have second dim >= max_nodes");
    TORCH_CHECK(p2_candidates_buffer.dim() >= 2 && p2_candidates_buffer.size(1) >= max_nodes,
                "p2_candidates_buffer must have second dim >= max_nodes");

    subtree_crossover_kernel<<<batch_size, 1>>>(
        child1_out.data_ptr<float>(),
        child2_out.data_ptr<float>(),
        p1_batch.data_ptr<float>(),
        p2_batch.data_ptr<float>(),
        mode, max_depth, max_nodes, max_retries,
        branch_perm.data_ptr<int>(),
        bfs_queue_buffer.data_ptr<int>(),
        result_indices_buffer.data_ptr<int>(),
        old_to_new_map_buffer.data_ptr<int>(),
        p1_candidates_buffer.data_ptr<int>(),
        p2_candidates_buffer.data_ptr<int>(),
        batch_size
    );
    cudaDeviceSynchronize();
}