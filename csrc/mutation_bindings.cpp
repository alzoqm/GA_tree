// csrc/mutation_bindings.cpp
#include <torch/extension.h>
#include "value_mutation_kernel.cuh"
#include "node_mutation_kernel.cuh"
#include "subtree_mutation_kernel.cuh"
#include "mutation_utils_kernel.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance CUDA mutation kernels for GATree";

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

    // --- NEW: Critical repair function (CUDA replacement for slow Python loops) ---
    m.def("critical_repair_batch", &critical_repair_batch_cuda,
        "Critical repair: Ensure no root branch is left without children (CUDA replacement for slow Python loops).",
        py::arg("trees")
    );

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