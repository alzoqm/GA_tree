// csrc/predict.cpp (수정된 최종 전체 코드)

#include <torch/extension.h>
#include "predict_kernel.cuh"
#include "adjacency_builder.cuh"
#include "constants.h"

// 공유 메모리 캐싱을 위한 피처 수 한계 (컴파일 타임 상수)
constexpr int MAX_FEATURES_IN_SHARED_MEM_CPP = 1024;

// ==============================================================================
//           Helper 1: 예측 커널에 전달될 텐서 유효성 검사 함수
// ==============================================================================
void check_predict_tensors(
    const torch::Tensor& population,
    const torch::Tensor& features,
    const torch::Tensor& positions,
    const torch::Tensor& next_indices,
    const torch::Tensor& offset_array,
    const torch::Tensor& child_indices,
    const torch::Tensor& results,
    const torch::Tensor& bfs_queue_buffer) {

    // --- 1. Device 검사 ---
    TORCH_CHECK(population.is_cuda() && features.is_cuda() && positions.is_cuda() &&
                next_indices.is_cuda() && offset_array.is_cuda() && child_indices.is_cuda() &&
                results.is_cuda() && bfs_queue_buffer.is_cuda(),
                "All input tensors for prediction must be on a CUDA device");

    // --- 2. Data Type 검사 ---
    TORCH_CHECK(population.scalar_type() == torch::kFloat32, "Population tensor must be of type float32");
    TORCH_CHECK(features.scalar_type() == torch::kFloat32, "Features tensor must be of type float32");
    TORCH_CHECK(positions.scalar_type() == torch::kInt64, "Positions tensor must be of type int64 (long)");
    TORCH_CHECK(next_indices.scalar_type() == torch::kInt32, "Next_indices tensor must be of type int32");
    TORCH_CHECK(offset_array.scalar_type() == torch::kInt32, "Offset array must be of type int32");
    TORCH_CHECK(child_indices.scalar_type() == torch::kInt32, "Child indices must be of type int32");
    TORCH_CHECK(results.scalar_type() == torch::kFloat32, "Results tensor must be of type float32");
    TORCH_CHECK(bfs_queue_buffer.scalar_type() == torch::kInt32, "BFS queue buffer must be of type int32");

    // --- 3. Contiguity 검사 ---
    TORCH_CHECK(population.is_contiguous(), "Population tensor must be contiguous");
    TORCH_CHECK(features.is_contiguous(), "Features tensor must be contiguous");
    TORCH_CHECK(positions.is_contiguous(), "Positions tensor must be contiguous");
    TORCH_CHECK(next_indices.is_contiguous(), "Next_indices tensor must be contiguous");
    TORCH_CHECK(offset_array.is_contiguous(), "Offset array must be contiguous");
    TORCH_CHECK(child_indices.is_contiguous(), "Child indices must be contiguous");
    TORCH_CHECK(results.is_contiguous(), "Results tensor must be contiguous");
    TORCH_CHECK(bfs_queue_buffer.is_contiguous(), "BFS queue buffer must be contiguous");

    // --- 4. Dimension 및 Size 검사 ---
    TORCH_CHECK(population.dim() == 3, "Population tensor must be 3D");
    TORCH_CHECK(features.dim() == 1, "Features tensor must be 1D");
    TORCH_CHECK(positions.dim() == 1, "Positions tensor must be 1D");
    TORCH_CHECK(next_indices.dim() == 1, "Next_indices tensor must be 1D");
    TORCH_CHECK(offset_array.dim() == 1, "Offset array must be 1D");
    TORCH_CHECK(child_indices.dim() == 1, "Child indices must be 1D");
    TORCH_CHECK(results.dim() == 2, "Results tensor must be 2D");
    TORCH_CHECK(bfs_queue_buffer.dim() == 2, "BFS queue buffer must be 2D");

    const int pop_size = population.size(0);
    const int max_nodes = population.size(1);

    TORCH_CHECK(features.size(0) <= MAX_FEATURES_IN_SHARED_MEM_CPP, 
                "Number of features exceeds shared memory limit defined in C++");
    TORCH_CHECK(positions.size(0) == pop_size, "Positions tensor pop_size mismatch");
    TORCH_CHECK(next_indices.size(0) == pop_size, "Next_indices tensor pop_size mismatch");
    TORCH_CHECK(offset_array.size(0) == (pop_size * max_nodes + 1), "Offset array size mismatch");
    TORCH_CHECK(results.size(0) == pop_size, "Results tensor pop_size mismatch");
    TORCH_CHECK(bfs_queue_buffer.size(0) == pop_size, "BFS queue buffer pop_size mismatch");
    TORCH_CHECK(population.size(2) == NODE_INFO_DIM, "Population tensor node_dim mismatch");
    TORCH_CHECK(results.size(1) == 4, "Results tensor must have 4 columns");
    TORCH_CHECK(bfs_queue_buffer.size(1) == max_nodes, "BFS queue buffer max_nodes mismatch");
}

// ==============================================================================
//           Helper 2: CUDA 커널을 호출하는 C++ 래퍼 함수
// ==============================================================================
void predict_cuda(
    torch::Tensor population_tensor,
    torch::Tensor features_tensor,
    torch::Tensor positions_tensor,
    torch::Tensor next_indices_tensor,
    torch::Tensor offset_array_tensor,
    torch::Tensor child_indices_tensor,
    torch::Tensor results_tensor,
    torch::Tensor bfs_queue_buffer) {
    
    // 1. 모든 입력 텐서 유효성 검사
    check_predict_tensors(population_tensor, features_tensor, positions_tensor, next_indices_tensor,
                          offset_array_tensor, child_indices_tensor, results_tensor, bfs_queue_buffer);

    // 2. 커널 실행에 필요한 파라미터 추출
    const int pop_size = population_tensor.size(0);
    const int max_nodes = population_tensor.size(1);
    const int num_features = features_tensor.size(0);

    // 3. `predict_kernel.cuh`에 선언된 커널 런처 함수 호출
    launch_predict_kernel(
        population_tensor.data_ptr<float>(),
        features_tensor.data_ptr<float>(),
        positions_tensor.data_ptr<long>(),
        next_indices_tensor.data_ptr<int>(),
        offset_array_tensor.data_ptr<int>(),
        child_indices_tensor.data_ptr<int>(),
        results_tensor.data_ptr<float>(),
        bfs_queue_buffer.data_ptr<int>(),
        pop_size,
        max_nodes,
        num_features
    );
}

// ==============================================================================
//               Pybind11 모듈 정의: Python과 C++ 연결
// ==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance CUDA kernels for GATree evaluation";

    // --- 인접 리스트 생성 관련 함수 바인딩 ---
    m.def("count_and_create_offsets", &count_and_create_offsets_cuda, 
          "Step 1: Counts children for each node and returns total child count and the offset array.");

    m.def("fill_child_indices", &fill_child_indices_cuda, 
          "Step 2: Fills a pre-allocated child_indices tensor using the offset array.",
          py::arg("population_tensor"), 
          py::arg("offset_array"), 
          py::arg("child_indices")); // 'child_indices'는 in-place로 수정됨

    // --- 예측 함수 바인딩 ---
    m.def("predict", &predict_cuda, 
          "GATree Prediction on CUDA using a pre-built adjacency list");
}