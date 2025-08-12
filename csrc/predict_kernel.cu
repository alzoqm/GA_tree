// /csrc/reorganize_kernel.cu (수정된 최종 코드)
#include "reorganize_kernel.cuh"
#include "constants.h"
#include <algorithm> // for std::min

// ==============================================================================
//   커널 1: 각 트리의 활성 노드 수를 세고, old_gid -> new_gid 매핑을 계산합니다.
//   [수정] 병렬 스캔(Prefix Sum) 로직의 레이스 컨디션 버그를 수정했습니다.
// ==============================================================================
__global__ void compute_remap_indices_kernel(
    const float* population_ptr,
    int* active_counts_per_tree_ptr, // [tree_idx] = num_active_nodes
    int* old_gid_to_new_gid_map_ptr, // [old_gid] = new_gid
    int pop_size,
    int max_nodes)
{
    // 각 스레드 블록이 하나의 트리(개체)를 담당합니다.
    const int tree_idx = blockIdx.x;
    if (tree_idx >= pop_size) return;
    
    // --- 1. 공유 메모리를 사용하여 블록 내 스캔(Prefix Sum) 수행 ---
    extern __shared__ int local_scan_buffer[]; // max_nodes 크기
    
    // 각 스레드가 자신의 로컬 노드를 담당 (Grid-Stride Loop)
    for (int i = threadIdx.x; i < max_nodes; i += blockDim.x) {
        const int old_gid = tree_idx * max_nodes + i;
        bool is_active = (population_ptr[old_gid * NODE_INFO_DIM + COL_NODE_TYPE] != NODE_TYPE_UNUSED);
        local_scan_buffer[i] = is_active ? 1 : 0;
    }
    __syncthreads();

    // --- 2. [수정된 부분] 블록 내 Prefix Sum (Scan) ---
    // Blelloch 스캔 알고리즘을 올바르게 구현하여 레이스 컨디션을 제거합니다.
    for (int offset = 1; offset < max_nodes; offset *= 2) {
        int i = threadIdx.x;
        // 스레드마다 독립적으로 처리할 수 있도록 임시 변수 사용
        float temp_val = 0;
        if (i >= offset) {
            temp_val = local_scan_buffer[i - offset];
        }
        __syncthreads(); // 모든 스레드가 이전 단계의 값을 읽을 때까지 대기

        if (i >= offset) {
             local_scan_buffer[i] += temp_val;
        }
        __syncthreads(); // 모든 스레드가 현재 단계의 계산을 마칠 때까지 대기
    }
    
    // --- 3. 최종 결과 계산 및 전역 메모리에 쓰기 ---
    for (int i = threadIdx.x; i < max_nodes; i += blockDim.x) {
        const int old_gid = tree_idx * max_nodes + i;
        
        bool is_active = (population_ptr[old_gid * NODE_INFO_DIM + COL_NODE_TYPE] != NODE_TYPE_UNUSED);
        if (is_active) {
            int new_local_idx = local_scan_buffer[i] - 1;
            old_gid_to_new_gid_map_ptr[old_gid] = tree_idx * max_nodes + new_local_idx;
        } else {
            old_gid_to_new_gid_map_ptr[old_gid] = -1; // 비활성 노드는 -1로 표시
        }
    }
    __syncthreads();
    
    // 블록의 0번 스레드가 해당 트리의 총 활성 노드 수를 기록
    if (threadIdx.x == 0 && max_nodes > 0) {
        active_counts_per_tree_ptr[tree_idx] = local_scan_buffer[max_nodes - 1];
    }
}

// ==============================================================================
//   커널 2: 계산된 매핑을 사용하여 노드 데이터를 새 위치로 복사하고 부모 인덱스를 업데이트합니다.
//   (이 커널은 원본과 동일하며, 수정할 필요가 없습니다.)
// ==============================================================================
__global__ void reorganize_and_update_kernel(
    const float* original_pop_ptr,
    const int* old_gid_to_new_gid_map_ptr,
    float* new_pop_ptr,
    int pop_size,
    int max_nodes)
{
    const int old_gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (old_gid >= pop_size * max_nodes) return;

    // 이 노드가 활성 노드일 경우에만 재구성 작업 수행
    if (original_pop_ptr[old_gid * NODE_INFO_DIM + COL_NODE_TYPE] != NODE_TYPE_UNUSED) {
        const int new_gid = old_gid_to_new_gid_map_ptr[old_gid];
        if (new_gid == -1) return;
        
        // 1. 노드 데이터 전체를 새 위치로 복사
        for (int i = 0; i < NODE_INFO_DIM; ++i) {
            new_pop_ptr[new_gid * NODE_INFO_DIM + i] = original_pop_ptr[old_gid * NODE_INFO_DIM + i];
        }

        // 2. 부모 인덱스를 새로운 gid로 업데이트
        int old_parent_local_idx = (int)original_pop_ptr[old_gid * NODE_INFO_DIM + COL_PARENT_IDX];
        
        if (old_parent_local_idx != -1) {
            int tree_idx = old_gid / max_nodes;
            int old_parent_gid = tree_idx * max_nodes + old_parent_local_idx;
            int new_parent_gid = old_gid_to_new_gid_map_ptr[old_parent_gid];
            
            if (new_parent_gid != -1) {
                int new_parent_local_idx = new_parent_gid % max_nodes;
                new_pop_ptr[new_gid * NODE_INFO_DIM + COL_PARENT_IDX] = (float)new_parent_local_idx;
            } else {
                 new_pop_ptr[new_gid * NODE_INFO_DIM + COL_PARENT_IDX] = -1.0f;
            }
        }
    }
}


// ==============================================================================
//                      C++ 래퍼 함수 (수정 없음)
// ==============================================================================
void reorganize_population_cuda(torch::Tensor population_tensor) {
    TORCH_CHECK(population_tensor.is_cuda(), "Population tensor must be on a CUDA device for reorganize.");
    TORCH_CHECK(population_tensor.is_contiguous(), "Population tensor must be contiguous.");

    const int pop_size = population_tensor.size(0);
    const int max_nodes = population_tensor.size(1);
    const int total_nodes = pop_size * max_nodes;
    
    if (total_nodes == 0) return;

    auto int_options = torch::TensorOptions().device(population_tensor.device()).dtype(torch::kInt32);

    // --- 1. 커널 실행에 필요한 임시 텐서 할당 ---
    torch::Tensor active_counts_per_tree = torch::zeros({pop_size}, int_options);
    torch::Tensor old_gid_to_new_gid_map = torch::empty({total_nodes}, int_options);
    
    // --- 2. 커널 1 실행: 매핑 정보 계산 ---
    const int threads_per_block_scan = std::min(max_nodes, 1024);
    const int num_blocks_scan = pop_size;
    const size_t shared_mem_size = max_nodes * sizeof(int);

    compute_remap_indices_kernel<<<num_blocks_scan, threads_per_block_scan, shared_mem_size>>>(
        population_tensor.data_ptr<float>(),
        active_counts_per_tree.data_ptr<int>(),
        old_gid_to_new_gid_map.data_ptr<int>(),
        pop_size,
        max_nodes
    );
    // cudaDeviceSynchronize(); // 디버깅 시 에러 확인에 필요할 수 있음

    // --- 3. 커널 2 실행: 실제 재구성 ---
    torch::Tensor new_population = torch::full_like(population_tensor, NODE_TYPE_UNUSED);
    
    const int threads_per_block_reorg = 256;
    const int num_blocks_reorg = (total_nodes + threads_per_block_reorg - 1) / threads_per_block_reorg;

    reorganize_and_update_kernel<<<num_blocks_reorg, threads_per_block_reorg>>>(
        population_tensor.data_ptr<float>(),
        old_gid_to_new_gid_map.data_ptr<int>(),
        new_population.data_ptr<float>(),
        pop_size,
        max_nodes
    );
    // cudaDeviceSynchronize(); // 디버깅 시 에러 확인에 필요할 수 있음

    // --- 4. 원본 텐서에 결과 복사 ---
    population_tensor.copy_(new_population);
}