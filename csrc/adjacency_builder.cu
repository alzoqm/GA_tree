// csrc/adjacency_builder.cu (수정됨)
#include "adjacency_builder.cuh"
#include "constants.h"

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
//         [신규] 1단계 C++ 래퍼 함수: 카운팅 및 오프셋 생성
// ==============================================================================
std::pair<long, torch::Tensor> count_and_create_offsets_cuda(const torch::Tensor& population_tensor) {
    TORCH_CHECK(population_tensor.is_cuda(), "Population tensor must be on a CUDA device");
    TORCH_CHECK(population_tensor.is_contiguous(), "Population tensor must be contiguous");
    
    const int pop_size = population_tensor.size(0);
    const int max_nodes = population_tensor.size(1);
    const int total_nodes = pop_size * max_nodes;
    auto options = torch::TensorOptions().device(population_tensor.device()).dtype(torch::kInt32);

    // C++ 내부에서 임시로 사용할 텐서 생성 (함수 종료 시 소멸)
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

    // Python으로 반환할 offset_array 텐서 생성
    torch::Tensor offset_array = torch::zeros({total_nodes + 1}, options);
    offset_array.slice(0, 1, total_nodes + 1) = torch::cumsum(child_counts, 0, torch::kInt32);
    
    // 총 자식 수를 long 타입으로 계산
    long total_children = offset_array.index({-1}).item<long>();

    // 계산된 값들을 pair로 묶어 Python에 반환
    return {total_children, offset_array};
}


// ==============================================================================
//            [신규] 2단계 C++ 래퍼 함수: 인덱스 내용 채우기
// ==============================================================================
void fill_child_indices_cuda(
    const torch::Tensor& population_tensor,
    const torch::Tensor& offset_array,
    torch::Tensor& child_indices // Python에서 생성된 텐서를 참조로 받음
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
        const int threads_per_block = 256;
        const int num_blocks = (total_nodes + threads_per_block - 1) / threads_per_block;
        
        fill_adjacency_list_kernel<<<num_blocks, threads_per_block>>>(
            population_tensor.data_ptr<float>(),
            offset_array.data_ptr<int>(),
            temp_offset.data_ptr<int>(),
            child_indices.data_ptr<int>(),
            pop_size,
            max_nodes
        );
        cudaDeviceSynchronize();
    }
}