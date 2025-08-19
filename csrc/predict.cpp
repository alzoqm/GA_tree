// csrc/predict.cpp (수정된 최종 전체 코드)

#include <torch/extension.h>
#include "predict_kernel.cuh"
#include "adjacency_builder.cuh"
#include "value_mutation_kernel.cuh"
#include "reorganize_kernel.cuh"
#include "constants.h"
#include "crossover_kernel.cuh" 

// NEW includes
#include "node_mutation_kernel.cuh"
#include "mutation_utils_kernel.cuh"


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

    // Device, Data Type, Contiguity, Dimension 검사
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

    // --- 교차(Crossover) 관련 CUDA 함수 바인딩 ---
    m.def("get_contextual_mask",
        [](const torch::Tensor& trees, int node_type, int branch_type) {
            auto options = torch::TensorOptions().device(trees.device()).dtype(torch::kBool);
            auto output_mask = torch::zeros({trees.size(0), trees.size(1)}, options);
            get_contextual_mask_cuda(trees, output_mask, node_type, branch_type);
            return output_mask;
        },
        "Get a mask for nodes matching a specific type and root branch context.",
        py::arg("trees"), py::arg("node_type"), py::arg("branch_type")
    );

    m.def("swap_node_params", &swap_node_params_cuda,
        "Swap node parameters between two populations based on given masks.",
        py::arg("c1"), py::arg("c2"), py::arg("p1_mask"), py::arg("p2_mask")
    );

    m.def("copy_branches_batch", &copy_branches_batch_cuda,
        "Performs root branch crossover on a batch of parents using CUDA.",
        py::arg("child_batch"), 
        py::arg("p1_batch"), 
        py::arg("p2_batch"), 
        py::arg("donor_map"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("old_to_new_map_buffer")
    );
    
    // [수정된] SubtreeCrossover 바인딩 (시그니처 변경)
    m.def("subtree_crossover_batch", &subtree_crossover_batch_cuda,
        "Performs subtree crossover on a batch of parents using CUDA.",
        py::arg("child1_out"),
        py::arg("child2_out"),
        py::arg("p1_batch"),
        py::arg("p2_batch"),
        py::arg("mode"),
        py::arg("max_depth"),
        py::arg("max_nodes"),
        py::arg("max_retries"),
        py::arg("branch_perm"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("old_to_new_map_buffer"),
        py::arg("p1_candidates_buffer"), // <--- 신규 인자
        py::arg("p2_candidates_buffer")  // <--- 신규 인자
    );

    // --- NEW: add-node mutation ---
    m.def("add_decision_nodes_batch", &add_decision_nodes_batch_cuda,
        "Batch add-node mutation (edge split) with invariant guards.",
        py::arg("trees"),
        py::arg("num_to_add"),
        py::arg("max_depth"),
        py::arg("max_nodes"),
        py::arg("max_add_nodes"),
        py::arg("out_new_node_indices"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("old_to_new_map_buffer"));

    // --- NEW: mutation utils (GPU variants of utils.py) ---
    m.def("find_subtree_nodes_batch", &find_subtree_nodes_batch_cuda,
        "Collect subtree nodes for each (b, root_idx) into result buffer.",
        py::arg("trees"),
        py::arg("root_indices"),           // (B,) int32 or (-1 for skip)
        py::arg("max_nodes"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("out_counts"));            // (B,) int32

    m.def("update_subtree_depth_batch", &update_subtree_depth_batch_cuda,
        "Delta-add to depth for each subtree.",
        py::arg("trees"),
        py::arg("root_indices"),           // (B,)
        py::arg("delta"),
        py::arg("max_nodes"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"));

    m.def("get_subtree_max_depth_batch", &get_subtree_max_depth_batch_cuda,
        "Return max depth over subtree for each (b,root).",
        py::arg("trees"),
        py::arg("root_indices"),           // (B,)
        py::arg("max_nodes"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("out_max_depths"));        // (B,) int32

    m.def("find_empty_slots_batch", &find_empty_slots_batch_cuda,
        "Return first 'count' UNUSED slots per tree into out_indices; -1 if not enough.",
        py::arg("trees"),
        py::arg("count"),                  // int
        py::arg("max_nodes"),
        py::arg("out_indices"));           // (B, count) int32
}