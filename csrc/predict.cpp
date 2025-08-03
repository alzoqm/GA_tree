// --- START OF FILE csrc/predict.cpp ---

// csrc/predict.cpp
#include <torch/extension.h>
#include "predict_kernel.cuh"
#include "constants.h"

// --- [수정] Input Validation and Tensor Checks ---
void check_tensors(
    const torch::Tensor& population,
    const torch::Tensor& features,
    const torch::Tensor& positions,
    const torch::Tensor& next_indices,
    const torch::Tensor& results) {

    // Device check
    TORCH_CHECK(population.device().is_cuda(), "Population tensor must be on a CUDA device");
    TORCH_CHECK(features.device().is_cuda(), "Features tensor must be on a CUDA device");
    TORCH_CHECK(positions.device().is_cuda(), "Positions tensor must be on a CUDA device");
    TORCH_CHECK(next_indices.device().is_cuda(), "Next_indices tensor must be on a CUDA device");
    TORCH_CHECK(results.device().is_cuda(), "Results tensor must be on a CUDA device");

    // Datatype check
    TORCH_CHECK(population.scalar_type() == torch::kFloat32, "Population tensor must be of type float32");
    TORCH_CHECK(features.scalar_type() == torch::kFloat32, "Features tensor must be of type float32");
    TORCH_CHECK(positions.scalar_type() == torch::kInt64, "Positions tensor must be of type int64 (long)");
    TORCH_CHECK(next_indices.scalar_type() == torch::kInt32, "Next_indices tensor must be of type int32");
    TORCH_CHECK(results.scalar_type() == torch::kFloat32, "Results tensor must be of type float32");

    // Contiguity check
    TORCH_CHECK(population.is_contiguous(), "Population tensor must be contiguous");
    TORCH_CHECK(features.is_contiguous(), "Features tensor must be contiguous");
    TORCH_CHECK(positions.is_contiguous(), "Positions tensor must be contiguous");
    TORCH_CHECK(next_indices.is_contiguous(), "Next_indices tensor must be contiguous");
    TORCH_CHECK(results.is_contiguous(), "Results tensor must be contiguous");

    // Dimension check
    TORCH_CHECK(population.dim() == 3, "Population tensor must be 3D");
    // [수정] features 텐서는 이제 1차원입니다.
    TORCH_CHECK(features.dim() == 1, "Features tensor must be 1D");
    TORCH_CHECK(positions.dim() == 1, "Positions tensor must be 1D");
    TORCH_CHECK(next_indices.dim() == 1, "Next_indices tensor must be 1D");
    TORCH_CHECK(results.dim() == 2, "Results tensor must be 2D");

    // Size consistency check
    int pop_size = population.size(0);
    // [수정] features 텐서에 대한 pop_size 일관성 체크는 제거합니다.
    TORCH_CHECK(positions.size(0) == pop_size, "Positions tensor pop_size mismatch");
    TORCH_CHECK(next_indices.size(0) == pop_size, "Next_indices tensor pop_size mismatch");
    TORCH_CHECK(results.size(0) == pop_size, "Results tensor pop_size mismatch");
    TORCH_CHECK(population.size(2) == NODE_INFO_DIM, "Population tensor node_dim mismatch");
    TORCH_CHECK(results.size(1) == 4, "Results tensor must have 4 columns");
}


// --- [수정] C++ Wrapper for the CUDA Kernel ---
void predict_cuda(
    torch::Tensor population_tensor,
    torch::Tensor features_tensor,
    torch::Tensor positions_tensor,
    torch::Tensor next_indices_tensor,
    torch::Tensor results_tensor) {

    check_tensors(population_tensor, features_tensor, positions_tensor, next_indices_tensor, results_tensor);

    int pop_size = population_tensor.size(0);
    int max_nodes = population_tensor.size(1);
    // [수정] num_features는 1D features_tensor의 크기에서 가져옵니다.
    int num_features = features_tensor.size(0);

    launch_predict_kernel(
        population_tensor.data_ptr<float>(),
        features_tensor.data_ptr<float>(),
        positions_tensor.data_ptr<long>(),
        next_indices_tensor.data_ptr<int>(),
        results_tensor.data_ptr<float>(),
        pop_size,
        max_nodes,
        num_features
    );
}

// --- Pybind11 Module Definition --- (수정 없음)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("predict", &predict_cuda, "GATree Prediction on CUDA for a population (in-place)");
}

// --- END OF FILE csrc/predict.cpp ---