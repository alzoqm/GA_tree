// /csrc/reorganize_kernel.cu (신규 파일)
#include "reorganize_kernel.cuh"
#include "constants.h"

// ==============================================================================
//   커널 1: 각 트리의 활성 노드 수를 세고, old_gid -> new_gid 매핑을 계산합니다.
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
    
    // 각 스레드가 자신의 로컬 노드 담당
    const int local_node_idx = threadIdx.x;
    const int old_gid = tree_idx * max_nodes + local_node_idx;

    // 활성 노드이면 1, 아니면 0을 공유 메모리에 기록
    if (local_node_idx < max_nodes) {
        bool is_active = (population_ptr[old_gid * NODE_INFO_DIM + COL_NODE_TYPE] != NODE_TYPE_UNUSED);
        local_scan_buffer[local_node_idx] = is_active ? 1 : 0;
    }
    __syncthreads();

    // --- 2. 블록 내 Prefix Sum (Scan) ---
    // 효율적인 병렬 스캔 알고리즘 (Blelloch Scan)
    for (int offset = 1; offset < max_nodes; offset *= 2) {
        if (local_node_idx >= offset) {
            int prev_val = local_scan_buffer[local_node_idx - offset];
            __syncthreads();
            local_scan_buffer[local_node_idx] += prev_val;
        }
        __syncthreads();
    }
    
    // --- 3. 최종 결과 계산 및 전역 메모리에 쓰기 ---
    if (local_node_idx < max_nodes) {
        int new_local_idx = local_scan_buffer[local_node_idx] - 1;
        
        // 이 노드가 활성 노드인 경우에만 매핑 정보를 기록
        if (population_ptr[old_gid * NODE_INFO_DIM + COL_NODE_TYPE] != NODE_TYPE_UNUSED) {
            old_gid_to_new_gid_map_ptr[old_gid] = tree_idx * max_nodes + new_local_idx;
        } else {
            old_gid_to_new_gid_map_ptr[old_gid] = -1; // 비활성 노드는 -1로 표시
        }
    }
    __syncthreads();
    
    // 블록의 0번 스레드가 해당 트리의 총 활성 노드 수를 기록
    if (local_node_idx == 0 && max_nodes > 0) {
        active_counts_per_tree_ptr[tree_idx] = local_scan_buffer[max_nodes - 1];
    }
}

// ==============================================================================
//   커널 2: 계산된 매핑을 사용하여 노드 데이터를 새 위치로 복사하고 부모 인덱스를 업데이트합니다.
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
            int new_parent_local_idx = new_parent_gid % max_nodes;
            new_pop_ptr[new_gid * NODE_INFO_DIM + COL_PARENT_IDX] = (float)new_parent_local_idx;
        }
    }
}


// ==============================================================================
//                      C++ 래퍼 함수
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
    const int threads_per_block_scan = max_nodes; // 각 블록이 트리 하나를 담당하므로, 스레드는 max_nodes만큼 필요
    const int num_blocks_scan = pop_size;
    const size_t shared_mem_size = max_nodes * sizeof(int);

    compute_remap_indices_kernel<<<num_blocks_scan, threads_per_block_scan, shared_mem_size>>>(
        population_tensor.data_ptr<float>(),
        active_counts_per_tree.data_ptr<int>(),
        old_gid_to_new_gid_map.data_ptr<int>(),
        pop_size,
        max_nodes
    );
    cudaDeviceSynchronize();

    // --- 3. 커널 2 실행: 실제 재구성 ---
    // 결과를 받을 새로운 임시 텐서 생성 (0으로 초기화)
    torch::Tensor new_population = torch::zeros_like(population_tensor);
    
    const int threads_per_block_reorg = 256;
    const int num_blocks_reorg = (total_nodes + threads_per_block_reorg - 1) / threads_per_block_reorg;

    reorganize_and_update_kernel<<<num_blocks_reorg, threads_per_block_reorg>>>(
        population_tensor.data_ptr<float>(),
        old_gid_to_new_gid_map.data_ptr<int>(),
        new_population.data_ptr<float>(),
        pop_size,
        max_nodes
    );
    cudaDeviceSynchronize();

    // --- 4. 원본 텐서에 결과 복사 ---
    population_tensor.copy_(new_population);
}