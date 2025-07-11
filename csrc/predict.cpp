// csrc/predict.cpp
#include <torch/extension.h>
#include "predict_kernel.cuh"
#include "constants.h"

// --- Input Validation and Tensor Checks ---
void check_tensors(
    const torch::Tensor& population,
    const torch::Tensor& features,
    const torch::Tensor& positions,
    const torch::Tensor& results) { // results 텐서 체크 추가

    // Device check
    TORCH_CHECK(population.device().is_cuda(), "Population tensor must be on a CUDA device");
    TORCH_CHECK(features.device().is_cuda(), "Features tensor must be on a CUDA device");
    TORCH_CHECK(positions.device().is_cuda(), "Positions tensor must be on a CUDA device");
    TORCH_CHECK(results.device().is_cuda(), "Results tensor must be on a CUDA device"); // 추가

    // Datatype check
    TORCH_CHECK(population.scalar_type() == torch::kFloat32, "Population tensor must be of type float32");
    TORCH_CHECK(features.scalar_type() == torch::kFloat32, "Features tensor must be of type float32");
    TORCH_CHECK(positions.scalar_type() == torch::kInt64, "Positions tensor must be of type int64 (long)");
    TORCH_CHECK(results.scalar_type() == torch::kFloat32, "Results tensor must be of type float32"); // 추가

    // Contiguity check
    TORCH_CHECK(population.is_contiguous(), "Population tensor must be contiguous");
    TORCH_CHECK(features.is_contiguous(), "Features tensor must be contiguous");
    TORCH_CHECK(positions.is_contiguous(), "Positions tensor must be contiguous");
    TORCH_CHECK(results.is_contiguous(), "Results tensor must be contiguous"); // 추가

    // Dimension check
    TORCH_CHECK(population.dim() == 3, "Population tensor must be 3D");
    TORCH_CHECK(features.dim() == 2, "Features tensor must be 2D");
    TORCH_CHECK(positions.dim() == 1, "Positions tensor must be 1D");
    TORCH_CHECK(results.dim() == 2, "Results tensor must be 2D"); // 추가

    // Size consistency check
    int pop_size = population.size(0);
    TORCH_CHECK(features.size(0) == pop_size, "Features tensor pop_size mismatch");
    TORCH_CHECK(positions.size(0) == pop_size, "Positions tensor pop_size mismatch");
    TORCH_CHECK(results.size(0) == pop_size, "Results tensor pop_size mismatch"); // 추가
    TORCH_CHECK(population.size(2) == NODE_INFO_DIM, "Population tensor node_dim mismatch");
    TORCH_CHECK(results.size(1) == 3, "Results tensor must have 3 columns"); // 추가
}


// --- C++ Wrapper for the CUDA Kernel ---
// 이제 반환 타입이 void 이고, results_tensor를 인자로 받습니다.
void predict_cuda(
    torch::Tensor population_tensor,
    torch::Tensor features_tensor,
    torch::Tensor positions_tensor,
    torch::Tensor results_tensor) { // results_tensor 추가

    check_tensors(population_tensor, features_tensor, positions_tensor, results_tensor);

    int pop_size = population_tensor.size(0);
    int max_nodes = population_tensor.size(1);
    int num_features = features_tensor.size(1);

    // 출력 텐서 생성 로직 제거

    // Launch the CUDA kernel
    launch_predict_kernel(
        population_tensor.data_ptr<float>(),
        features_tensor.data_ptr<float>(),
        positions_tensor.data_ptr<long>(),
        results_tensor.data_ptr<float>(), // 전달받은 텐서의 포인터 사용
        pop_size,
        max_nodes,
        num_features
    );
}

// --- Pybind11 Module Definition ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("predict", &predict_cuda, "GATree Prediction on CUDA for a population (in-place)");
}