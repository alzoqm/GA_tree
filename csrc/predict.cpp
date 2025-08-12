// csrc/predict.cpp (수정된 최종 전체 코드)

#include <torch/extension.h>
#include <random> // [수정] <random> 헤더 추가
#include "predict_kernel.cuh"
#include "adjacency_builder.cuh"
#include "value_mutation_kernel.cuh"
#include "reorganize_kernel.cuh"
#include "constants.h"


// ==============================================================================
//           Helper 1: 예측 커널에 전달될 텐서 유효성 검사 함수 (변경 없음)
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
//           Helper 2: CUDA 커널을 호출하는 C++ 래퍼 함수 (변경 없음)
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
//           [신규] Helper 3: 변이 커널을 위한 시드 생성 및 실행 헬퍼
// ==============================================================================

// 고품질 난수 생성을 위한 정적 객체
static std::random_device rd;
static std::mt19937_64 gen(rd());
static std::uniform_int_distribution<unsigned long long> distrib;

void _launch_mutation_kernel_cpp(
    torch::Tensor& population, bool is_reinitialize, float mutation_prob,
    float noise_ratio, int leverage_change,
    torch::Tensor& feature_num_indices, torch::Tensor& feature_min_vals, torch::Tensor& feature_max_vals,
    torch::Tensor& feature_comparison_indices, torch::Tensor& feature_bool_indices
) {
    const int pop_size = population.size(0);
    const int max_nodes = population.size(1);
    const int total_nodes = pop_size * max_nodes;

    if (total_nodes == 0) return;

    auto options = torch::TensorOptions().device(population.device()).dtype(torch::kInt64);
    torch::Tensor curand_states = torch::empty({total_nodes, sizeof(curandStatePhilox4_32_10_t) / sizeof(int64_t)}, options);

    const int threads = 256;
    const int blocks = (total_nodes + threads - 1) / threads;

    // [수정] 고품질 시드 생성
    unsigned long long seed = distrib(gen);

    init_curand_states_kernel<<<blocks, threads>>>(
        seed, // 생성된 시드 사용
        pop_size, max_nodes, (curandStatePhilox4_32_10_t*)curand_states.data_ptr()
    );
    cudaDeviceSynchronize();

    value_mutation_kernel<<<blocks, threads>>>(
        population.data_ptr<float>(),
        (curandStatePhilox4_32_10_t*)curand_states.data_ptr(),
        is_reinitialize, mutation_prob, noise_ratio, leverage_change,
        feature_num_indices.data_ptr<int>(), feature_min_vals.data_ptr<float>(), feature_max_vals.data_ptr<float>(), feature_num_indices.size(0),
        feature_comparison_indices.data_ptr<int>(), feature_comparison_indices.size(0),
        feature_bool_indices.data_ptr<int>(), feature_bool_indices.size(0),
        pop_size, max_nodes
    );
    cudaDeviceSynchronize();
}


// ==============================================================================
//               Pybind11 모듈 정의: Python과 C++ 연결
// ==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance CUDA kernels for GATree";

    // --- 구조 관련 함수 ---
    m.def("count_and_create_offsets", &count_and_create_offsets_cuda, "Build adjacency list step 1");
    
    // [수정] fill_child_indices 바인딩 수정 (max_children 받도록)
    m.def("fill_child_indices", &fill_child_indices_cuda, 
          "Build adjacency list step 2: fill and sort child indices",
          py::arg("population_tensor"),
          py::arg("offset_array"),
          py::arg("child_indices"),
          py::arg("max_children")
    );

    // --- 예측 함수 ---
    m.def("predict", &predict_cuda, "GATree Prediction on CUDA");

    // --- [수정] 값 기반 돌연변이 함수 바인딩 ---
    m.def("node_param_mutate", 
        [](torch::Tensor population, float mutation_prob, float noise_ratio, int leverage_change,
           torch::Tensor feature_num_indices, torch::Tensor feature_min_vals, torch::Tensor feature_max_vals) {
            
            auto empty_int_tensor = torch::empty({0}, torch::dtype(torch::kInt32).device(population.device()));
            _launch_mutation_kernel_cpp(
                population, false, mutation_prob, noise_ratio, leverage_change,
                feature_num_indices, feature_min_vals, feature_max_vals,
                empty_int_tensor, empty_int_tensor
            );
        },
        "Perform NodeParamMutation on GPU.",
        py::arg("population"),
        py::arg("mutation_prob"),
        py::arg("noise_ratio"),
        py::arg("leverage_change"),
        py::arg("feature_num_indices"),
        py::arg("feature_min_vals"),
        py::arg("feature_max_vals")
    );

    m.def("reinitialize_node_mutate", 
        [](torch::Tensor population, float mutation_prob,
           torch::Tensor feature_num_indices, torch::Tensor feature_min_vals, torch::Tensor feature_max_vals,
           torch::Tensor feature_comparison_indices, torch::Tensor feature_bool_indices) {
           
           _launch_mutation_kernel_cpp(
               population, true, mutation_prob, 0.0, 0,
               feature_num_indices, feature_min_vals, feature_max_vals,
               feature_comparison_indices, feature_bool_indices
           );
        },
        "Perform ReinitializeNodeMutation on GPU.",
        py::arg("population"),
        py::arg("mutation_prob"),
        py::arg("feature_num_indices"),
        py::arg("feature_min_vals"),
        py::arg("feature_max_vals"),
        py::arg("feature_comparison_indices"),
        py::arg("feature_bool_indices")
    );
    
    // --- [신규] 재구성 함수 바인딩 ---
    m.def("reorganize_population", &reorganize_population_cuda, 
          "Reorganize the population tensor on GPU to remove fragmentation.");
}