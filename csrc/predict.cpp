// csrc/predict.cpp (수정된 최종 전체 코드)

#include <torch/extension.h>
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

    // Device, Data Type, Contiguity, Dimension 검사 (이전과 동일)
    TORCH_CHECK(population.is_cuda() && features.is_cuda() && positions.is_cuda() &&
                next_indices.is_cuda() && offset_array.is_cuda() && child_indices.is_cuda() &&
                results.is_cuda() && bfs_queue_buffer.is_cuda(),
                "All input tensors for prediction must be on a CUDA device");
    TORCH_CHECK(population.dim() == 3, "Population tensor must be 3D");
    const int pop_size = population.size(0);
    const int max_nodes = population.size(1);
    TORCH_CHECK(positions.size(0) == pop_size, "Positions tensor pop_size mismatch");
    TORCH_CHECK(results.size(0) == pop_size, "Results tensor pop_size mismatch");
    TORCH_CHECK(population.size(2) == NODE_INFO_DIM, "Population tensor node_dim mismatch");
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

    check_predict_tensors(population_tensor, features_tensor, positions_tensor, next_indices_tensor,
                          offset_array_tensor, child_indices_tensor, results_tensor, bfs_queue_buffer);

    const int pop_size = population_tensor.size(0);
    const int max_nodes = population_tensor.size(1);
    const int num_features = features_tensor.size(0);

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
    m.doc() = "High-performance CUDA kernels for GATree";

    // --- 구조 관련 함수 ---
    m.def("count_and_create_offsets", &count_and_create_offsets_cuda, "Build adjacency list step 1");
    
    m.def("fill_child_indices", &fill_child_indices_cuda, 
          "Build adjacency list step 2: fill and sort child indices",
          py::arg("population_tensor"),
          py::arg("offset_array"),
          py::arg("child_indices"),
          py::arg("max_children")
    );

    // --- 예측 함수 ---
    m.def("predict", &predict_cuda, "GATree Prediction on CUDA");

    // --- 값 기반 돌연변이 함수 바인딩 ---
    m.def("node_param_mutate", 
        [](torch::Tensor& population, float mutation_prob, float noise_ratio, int leverage_change,
           torch::Tensor& feature_num_indices, torch::Tensor& feature_min_vals, torch::Tensor& feature_max_vals) {
            
            auto empty_int_tensor = torch::empty({0}, torch::dtype(torch::kInt32).device(population.device()));
            _launch_mutation_kernel_cpp(
                population, false, mutation_prob, noise_ratio, leverage_change,
                feature_num_indices, feature_min_vals, feature_max_vals,
                empty_int_tensor, empty_int_tensor
            );
        },
        "Perform NodeParamMutation on GPU.",
        py::arg("population"), py::arg("mutation_prob"), py::arg("noise_ratio"),
        py::arg("leverage_change"), py::arg("feature_num_indices"),
        py::arg("feature_min_vals"), py::arg("feature_max_vals")
    );

    m.def("reinitialize_node_mutate", 
        [](torch::Tensor& population, float mutation_prob,
           torch::Tensor& feature_num_indices, torch::Tensor& feature_min_vals, torch::Tensor& feature_max_vals,
           torch::Tensor& feature_comparison_indices, torch::Tensor& feature_bool_indices) {
           
           _launch_mutation_kernel_cpp(
               population, true, mutation_prob, 0.0, 0,
               feature_num_indices, feature_min_vals, feature_max_vals,
               feature_comparison_indices, feature_bool_indices
           );
        },
        "Perform ReinitializeNodeMutation on GPU.",
        py::arg("population"), py::arg("mutation_prob"),
        py::arg("feature_num_indices"), py::arg("feature_min_vals"), py::arg("feature_max_vals"),
        py::arg("feature_comparison_indices"), py::arg("feature_bool_indices")
    );
    
    // --- 재구성 함수 바인딩 ---
    m.def("reorganize_population", &reorganize_population_cuda, 
          "Reorganize the population tensor on GPU to remove fragmentation.");
}