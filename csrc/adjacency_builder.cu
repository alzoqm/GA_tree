// csrc/adjacency_builder.cu (수정된 파일)
#include "adjacency_builder.cuh"
#include "constants.h"
#include <cuda_runtime.h>

// ==============================================================================
//           커널 1: 각 노드의 자식 수를 병렬로 계산하는 커널 (변경 없음)
// ==============================================================================
__global__ void count_children_kernel(
    const float* population_ptr,
    int* child_counts_ptr,
    int pop_size,
    int max_nodes) {

    const int total_nodes = pop_size * max_nodes;
    const int node_gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_gid >= total_nodes) {
        return;
    }

    const float* node_data = population_ptr + node_gid * NODE_INFO_DIM;
    
    if (node_data[COL_NODE_TYPE] != NODE_TYPE_UNUSED) {
        int parent_idx = static_cast<int>(node_data[COL_PARENT_IDX]);
        if (parent_idx != -1) {
            int tree_idx = node_gid / max_nodes;
            int parent_gid = tree_idx * max_nodes + parent_idx;
            atomicAdd(&child_counts_ptr[parent_gid], 1);
        }
    }
}

// ==============================================================================
//        커널 2: CSR 형식의 인접 리스트를 병렬로 채우는 커널 (변경 없음)
// ==============================================================================
__global__ void fill_adjacency_list_kernel(
    const float* population_ptr,
    const int* offset_ptr,
    int* temp_offset_ptr,
    int* child_indices_ptr,
    int pop_size,
    int max_nodes) {

    const int total_nodes = pop_size * max_nodes;
    const int node_gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_gid >= total_nodes) {
        return;
    }

    const float* node_data = population_ptr + node_gid * NODE_INFO_DIM;

    if (node_data[COL_NODE_TYPE] != NODE_TYPE_UNUSED) {
        int parent_idx = static_cast<int>(node_data[COL_PARENT_IDX]);
        if (parent_idx != -1) {
            int tree_idx = node_gid / max_nodes;
            int parent_gid = tree_idx * max_nodes + parent_idx;
            int placement_offset = atomicAdd(&temp_offset_ptr[parent_gid], 1);
            int local_node_idx = node_gid % max_nodes;
            child_indices_ptr[placement_offset] = local_node_idx;
        }
    }
}


// ==============================================================================
// [신규] 커널 3 헬퍼 함수: 공유 메모리 내에서 Bitonic Sort의 한 단계를 수행
// ==============================================================================
__device__ void bitonic_sort_step(int* data, int j, int k, int thread_id) {
    int ixj = thread_id ^ j;

    if (ixj > thread_id) {
        if ((thread_id & k) == 0) { // 오름차순
            if (data[thread_id] > data[ixj]) {
                int temp = data[thread_id];
                data[thread_id] = data[ixj];
                data[ixj] = temp;
            }
        } else { // 내림차순
            if (data[thread_id] < data[ixj]) {
                int temp = data[thread_id];
                data[thread_id] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

// ==============================================================================
// [전면 수정] 커널 3: 각 부모의 자식 리스트를 Bitonic Sort로 병렬 정렬하는 커널
// ==============================================================================
__global__ void sort_children_kernel(
    const int* offset_ptr,
    int* child_indices_ptr,
    int pop_size,
    int max_nodes,
    int max_children) {

    // 정렬을 위한 공유 메모리 할당
    extern __shared__ int s_children[];

    // 각 스레드 블록이 하나의 부모 노드를 담당합니다.
    const int parent_gid = blockIdx.x;
    if (parent_gid >= pop_size * max_nodes) return;

    const int start_idx = offset_ptr[parent_gid];
    const int end_idx = offset_ptr[parent_gid + 1];
    const int num_children = end_idx - start_idx;

    if (num_children <= 1) return;

    // 1. 전역 메모리 -> 공유 메모리로 자식 리스트 복사
    int local_tid = threadIdx.x;
    if (local_tid < num_children) {
        s_children[local_tid] = child_indices_ptr[start_idx + local_tid];
    }
    __syncthreads();

    // 2. Bitonic Sort 수행 (공유 메모리 내에서)
    for (int k = 2; k <= blockDim.x; k <<= 1) { // blockDim.x는 max_children의 다음 2의 거듭제곱
        for (int j = k >> 1; j > 0; j = j >> 1) {
            __syncthreads(); // 이전 스텝의 모든 비교/교환이 끝날 때까지 대기
            bitonic_sort_step(s_children, j, k, local_tid);
        }
    }
    __syncthreads();

    // 3. 정렬된 공유 메모리 -> 전역 메모리로 결과 복사
    if (local_tid < num_children) {
        child_indices_ptr[start_idx + local_tid] = s_children[local_tid];
    }
}


// ==============================================================================
//         1단계 C++ 래퍼 함수: 카운팅 및 오프셋 생성 (변경 없음)
// ==============================================================================
std::pair<long, torch::Tensor> count_and_create_offsets_cuda(const torch::Tensor& population_tensor) {
    TORCH_CHECK(population_tensor.is_cuda(), "Population tensor must be on a CUDA device");
    TORCH_CHECK(population_tensor.is_contiguous(), "Population tensor must be contiguous");
    
    const int pop_size = population_tensor.size(0);
    const int max_nodes = population_tensor.size(1);
    const int total_nodes = pop_size * max_nodes;
    auto options = torch::TensorOptions().device(population_tensor.device()).dtype(torch::kInt32);

    torch::Tensor child_counts = torch::zeros({total_nodes}, options);
    
    const int threads_per_block = 256;
    const int num_blocks = (total_nodes + threads_per_block - 1) / threads_per_block;

    count_children_kernel<<<num_blocks, threads_per_block>>>(
        population_tensor.data_ptr<float>(),
        child_counts.data_ptr<int>(),
        pop_size,
        max_nodes
    );
    cudaDeviceSynchronize();

    torch::Tensor offset_array = torch::zeros({total_nodes + 1}, options);
    offset_array.slice(0, 1, total_nodes + 1) = torch::cumsum(child_counts, 0, torch::kInt32);
    
    long total_children = offset_array.index({-1}).item<long>();

    return {total_children, offset_array};
}

// [수정] 헬퍼 함수: x보다 크거나 같은 가장 작은 2의 거듭제곱을 계산
unsigned int next_power_of_2(unsigned int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


// ==============================================================================
// [수정] 3단계 정렬 커널을 호출하는 C++ 런처 함수
// ==============================================================================
void launch_sort_children_kernel(
    const int* offset_ptr,
    int* child_indices_ptr,
    int pop_size,
    int max_nodes,
    int max_children) {
    
    const int total_parent_nodes = pop_size * max_nodes;
    if (total_parent_nodes == 0) return;

    // [수정] Bitonic Sort를 위한 스레드 블록 및 공유 메모리 설정
    // 스레드 수는 max_children보다 크거나 같은 최소 2의 거듭제곱으로 설정
    const int threads_per_block = next_power_of_2(max_children);
    const int num_blocks = total_parent_nodes; // 각 블록이 부모 노드 하나를 담당

    // 공유 메모리 크기 계산
    const size_t shared_mem_size = threads_per_block * sizeof(int);

    sort_children_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        offset_ptr,
        child_indices_ptr,
        pop_size,
        max_nodes,
        max_children
    );
    cudaDeviceSynchronize();
}

// ==============================================================================
//            2단계 C++ 래퍼 함수: 인덱스 내용 채우기 (수정됨)
// ==============================================================================
void fill_child_indices_cuda(
    const torch::Tensor& population_tensor,
    const torch::Tensor& offset_array,
    torch::Tensor& child_indices, // Python에서 생성된 텐서를 참조로 받음
    int max_children // [수정] max_children 파라미터 추가
) {
    TORCH_CHECK(population_tensor.is_cuda(), "Population tensor must be on a CUDA device");
    TORCH_CHECK(offset_array.is_cuda(), "Offset array must be on a CUDA device");
    TORCH_CHECK(child_indices.is_cuda(), "Child indices must be on a CUDA device");

    const int pop_size = population_tensor.size(0);
    const int max_nodes = population_tensor.size(1);
    const int total_nodes = pop_size * max_nodes;
    long total_children = child_indices.size(0);

    // C++ 내부에서 임시로 사용할 텐서 생성 (atomicAdd를 위함)
    torch::Tensor temp_offset = offset_array.clone();

    if (total_children > 0) {
        const int threads_per_block_fill = 256;
        const int num_blocks_fill = (total_nodes + threads_per_block_fill - 1) / threads_per_block_fill;
        
        // Step 2a: 비결정적으로 자식 리스트를 채웁니다. (기존 로직)
        fill_adjacency_list_kernel<<<num_blocks_fill, threads_per_block_fill>>>(
            population_tensor.data_ptr<float>(),
            offset_array.data_ptr<int>(),
            temp_offset.data_ptr<int>(),
            child_indices.data_ptr<int>(),
            pop_size,
            max_nodes
        );
        cudaDeviceSynchronize(); // 채우기 완료까지 대기

        // [수정] Step 2b: 채워진 자식 리스트를 병렬로 정렬합니다.
        launch_sort_children_kernel(
            offset_array.data_ptr<int>(),
            child_indices.data_ptr<int>(),
            pop_size,
            max_nodes,
            max_children // 추가된 파라미터 전달
        );
        // cudaDeviceSynchronize()는 launch_sort_children_kernel 내부에 이미 있으므로 생략 가능
    }
}