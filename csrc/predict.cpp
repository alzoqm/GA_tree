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
#include "subtree_mutation_kernel.cuh"
#include "mutation_utils_kernel.cuh"
#include "validate_kernel.cuh"
#include "make_population.cuh"


// ==============================================================================
//           Optimized Prediction Function - Updated for New API
// ==============================================================================
// NOTE: The new predict_cuda function is already defined in predict_kernel.cuh
// and implemented in predict_kernel.cu. This wrapper is no longer needed as
// the validation and implementation are now integrated.

// ==============================================================================
//               Pybind11 모듈 정의: Python과 C++ 연결
// ==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance CUDA kernels for GATree";

    // --- Adjacency (Step 1: counts + offsets + overflow mask) ---
    m.def("count_and_create_offsets", &count_and_create_offsets_cuda,
          "Step 1: Count children, build CSR offsets, and return per-tree overflow mask",
          py::arg("population_tensor"),
          py::arg("max_children"));

    // --- Adjacency (Step 2: fill + optional sort) ---
    m.def("fill_child_indices", &fill_child_indices_cuda,
          "Step 2: Fill CSR child indices and (optionally) sort per parent",
          py::arg("population_tensor"),
          py::arg("offset_array"),
          py::arg("child_indices"),
          py::arg("max_children"),
          py::arg("sort_children") = true);

    // --- Validator ---
    m.def("validate_adjacency", &validate_adjacency_cuda,
          "Validate CSR/structure; writes per-tree violation bitmask (0 means OK)",
          py::arg("population_tensor"),
          py::arg("offset_array"),
          py::arg("child_indices"),
          py::arg("max_children"),
          py::arg("max_depth"),
          py::arg("out_violation_mask"));

    // --- Optimized prediction function with safety guarantees ---
    m.def("predict", &predict_cuda, 
          "GPU-accelerated prediction for GA-Tree populations with safety guarantees",
          py::arg("trees"), py::arg("features"), py::arg("positions"), 
          py::arg("offsets"), py::arg("children"), py::arg("results"),
          py::arg("bfs_q"), py::arg("visited"));

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

    m.def("delete_nodes_batch", &delete_nodes_batch_cuda,
        "Batch delete-node mutation with invariant guards.",
        py::arg("trees"),
        py::arg("num_to_delete"),
        py::arg("max_children"),
        py::arg("max_depth"),
        py::arg("max_nodes"),
        py::arg("max_delete_nodes"),
        py::arg("deleted_nodes"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("child_count_buffer"),
        py::arg("act_cnt_buffer"),
        py::arg("dec_cnt_buffer"),
        py::arg("candidate_indices_buffer"),
        py::arg("candidate_weights_buffer"));

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

    // --- NEW: AddSubtree batch mutation ---
    m.def("add_subtrees_batch", &add_subtrees_batch_cuda,
        "Batch add-subtree mutation with invariant guards.",
        py::arg("trees"), py::arg("num_to_grow"),
        py::arg("max_children"), py::arg("max_depth"), py::arg("max_nodes"), py::arg("max_new_nodes"),
        py::arg("new_decision_nodes"), py::arg("new_action_nodes"), py::arg("action_root_branch_type"),
        py::arg("bfs_queue_buffer"), py::arg("result_indices_buffer"),
        py::arg("child_count_buffer"), py::arg("act_cnt_buffer"), py::arg("dec_cnt_buffer"),
        py::arg("candidate_indices_buffer"), py::arg("candidate_weights_buffer"));

    // --- NEW: Delete-Subtree batch mutation ---
    m.def("delete_subtrees_batch", &delete_subtrees_batch_cuda,
        "Batch delete-subtree mutation with invariant guards and repairs.",
        py::arg("trees"),
        py::arg("mutate_mask_i32"),
        py::arg("max_nodes"),
        py::arg("alpha"),
        py::arg("ensure_action_left"),
        py::arg("child_count_buffer"),
        py::arg("act_cnt_buffer"),
        py::arg("dec_cnt_buffer"),
        py::arg("candidate_indices_buffer"),
        py::arg("candidate_weights_buffer"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("deletion_mask_buffer"),
        py::arg("repair_mask_buffer"),
        py::arg("chosen_roots_buffer")
    );

    // --- GPU Population Initialization with invariant guarantees ---
    m.def("init_population_cuda", &init_population_cuda,
          "Initialize a batch of trees on CUDA with invariant guarantees.",
          py::arg("trees"), py::arg("total_budget"),
          py::arg("max_children"), py::arg("max_depth"), py::arg("max_nodes"),
          py::arg("bfs_q"), py::arg("scratch"), py::arg("child_cnt"),
          py::arg("act_cnt"), py::arg("dec_cnt"), py::arg("cand_idx"), py::arg("cand_w"),
          py::arg("num_feat_indices"), py::arg("num_feat_minmax"),
          py::arg("bool_feat_indices"), py::arg("ff_pairs"),
          py::arg("long_actions"), py::arg("hold_actions"), py::arg("short_actions"));

    // --- Validation: checks structural constraints and throws if violations found ---
    m.def("validate_trees", &validate_trees_or_throw_cuda,
          "Validate tree structural constraints on GPU; throws on invalid trees.",
          py::arg("trees"));
}