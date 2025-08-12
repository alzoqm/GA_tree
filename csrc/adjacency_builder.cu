// csrc/adjacency_builder.cu (수정된 파일)
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
// [신규] 커널 3: 각 부모의 자식 리스트를 정렬하는 커널
// ==============================================================================
__global__ void sort_children_kernel(
    const int* offset_ptr,
    int* child_indices_ptr,
    int pop_size,
    int max_nodes) {

    // 각 스레드는 하나의 부모 노드를 담당합니다.
    const int parent_gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_parent_nodes = pop_size * max_nodes;

    if (parent_gid >= total_parent_nodes) {
        return;
    }

    // `offset_ptr`을 사용하여 이 부모의 자식 리스트 범위를 찾습니다.
    const int start_idx = offset_ptr[parent_gid];
    const int end_idx = offset_ptr[parent_gid + 1];
    const int num_children = end_idx - start_idx;

    // 자식이 1개 이하면 정렬할 필요가 없습니다.
    if (num_children <= 1) {
        return;
    }

    // 자식 리스트의 시작 포인터를 가져옵니다.
    int* children_list = child_indices_ptr + start_idx;

    // 간단한 삽입 정렬(Insertion Sort)을 수행합니다.
    // 자식의 수는 일반적으로 매우 적기 때문에 삽입 정렬이 효율적입니다.
    for (int i = 1; i < num_children; i++) {
        int key = children_list[i];
        int j = i - 1;
        while (j >= 0 && children_list[j] > key) {
            children_list[j + 1] = children_list[j];
            j = j - 1;
        }
        children_list[j + 1] = key;
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

// ==============================================================================
// [신규] 3단계 정렬 커널을 호출하는 C++ 런처 함수
// ==============================================================================
void launch_sort_children_kernel(
    const int* offset_ptr,
    int* child_indices_ptr,
    int pop_size,
    int max_nodes) {
    
    const int total_parent_nodes = pop_size * max_nodes;
    if (total_parent_nodes == 0) return;

    const int threads_per_block = 256;
    const int num_blocks = (total_parent_nodes + threads_per_block - 1) / threads_per_block;

    sort_children_kernel<<<num_blocks, threads_per_block>>>(
        offset_ptr,
        child_indices_ptr,
        pop_size,
        max_nodes
    );
}

// ==============================================================================
//            2단계 C++ 래퍼 함수: 인덱스 내용 채우기 (수정됨)
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
        
        // Step 2a: 비결정적으로 자식 리스트를 채웁니다. (기존 로직)
        fill_adjacency_list_kernel<<<num_blocks, threads_per_block>>>(
            population_tensor.data_ptr<float>(),
            offset_array.data_ptr<int>(),
            temp_offset.data_ptr<int>(),
            child_indices.data_ptr<int>(),
            pop_size,
            max_nodes
        );
        cudaDeviceSynchronize(); // 채우기 완료까지 대기

        // [수정] Step 2b: 채워진 자식 리스트를 결정적으로 정렬합니다.
        launch_sort_children_kernel(
            offset_array.data_ptr<int>(),
            child_indices.data_ptr<int>(),
            pop_size,
            max_nodes
        );
        cudaDeviceSynchronize(); // 정렬 완료까지 대기
    }
}