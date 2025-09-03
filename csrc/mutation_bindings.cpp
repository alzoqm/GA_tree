// csrc/mutation_bindings.cpp
// This file contains the Python bindings for the CUDA mutation kernels.
// It uses pybind11 to expose the C++/CUDA functions to Python, allowing for
// high-performance mutations to be called directly from Python.

#include <torch/extension.h>
#include "value_mutation_kernel.cuh"
#include "node_mutation_kernel.cuh"
#include "subtree_mutation_kernel.cuh"
#include "mutation_utils_kernel.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance CUDA mutation kernels for GATree";

    // Bindings for value-based mutation functions

    // This binding exposes the NodeParamMutation kernel to Python.
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
        "Performs NodeParamMutation on the GPU. This mutation adjusts the parameters of existing nodes.",
        py::arg("population"), py::arg("mutation_prob"), py::arg("noise_ratio"),
        py::arg("leverage_change"), py::arg("feature_num_indices"),
        py::arg("feature_min_vals"), py::arg("feature_max_vals")
    );

    // This binding exposes the ReinitializeNodeMutation kernel to Python.
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
        "Performs ReinitializeNodeMutation on the GPU. This mutation re-initializes a node from scratch.",
        py::arg("population"), py::arg("mutation_prob"),
        py::arg("feature_num_indices"), py::arg("feature_min_vals"), py::arg("feature_max_vals"),
        py::arg("feature_comparison_indices"), py::arg("feature_bool_indices")
    );

    // Bindings for structural mutation functions

    // This binding exposes the add_decision_nodes_batch_cuda kernel to Python.
    m.def("add_decision_nodes_batch", &add_decision_nodes_batch_cuda,
        "Performs batch add-node mutation (edge splitting) with invariant guards.",
        py::arg("trees"),
        py::arg("num_to_add"),
        py::arg("max_nodes"),
        py::arg("max_add_nodes"),
        py::arg("out_new_node_indices"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("old_to_new_map_buffer"),
        py::arg("candidate_indices_buffer"),
        py::arg("candidate_weights_buffer"));

    // This binding exposes the delete_nodes_batch_cuda kernel to Python.
    m.def("delete_nodes_batch", &delete_nodes_batch_cuda,
        "Performs batch delete-node mutation with invariant guards.",
        py::arg("trees"),
        py::arg("num_to_delete"),
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

    // This binding exposes the add_subtrees_batch_cuda kernel to Python.
    m.def("add_subtrees_batch", &add_subtrees_batch_cuda,
        "Performs batch add-subtree mutation with invariant guards.",
        py::arg("trees"), py::arg("num_to_grow"),
        py::arg("max_nodes"), py::arg("max_new_nodes"),
        py::arg("new_decision_nodes"), py::arg("new_action_nodes"), py::arg("action_root_branch_type"),
        py::arg("bfs_queue_buffer"), py::arg("result_indices_buffer"),
        py::arg("child_count_buffer"), py::arg("act_cnt_buffer"), py::arg("dec_cnt_buffer"),
        py::arg("candidate_indices_buffer"), py::arg("candidate_weights_buffer"));

    // This binding exposes the delete_subtrees_batch_cuda kernel to Python.
    m.def("delete_subtrees_batch", &delete_subtrees_batch_cuda,
        "Performs batch delete-subtree mutation with invariant guards and repairs.",
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

    // Binding for the critical repair function.
    m.def("critical_repair_batch", &critical_repair_batch_cuda,
        "Ensures no root branch is left without children, replacing slow Python loops.",
        py::arg("trees")
    );

    // Bindings for mutation utility functions.

    // This binding exposes the find_subtree_nodes_batch_cuda kernel to Python.
    m.def("find_subtree_nodes_batch", &find_subtree_nodes_batch_cuda,
        "Collects all nodes in a subtree for each (batch_index, root_index) pair.",
        py::arg("trees"),
        py::arg("root_indices"),
        py::arg("max_nodes"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("out_counts"));

    // This binding exposes the update_subtree_depth_batch_cuda kernel to Python.
    m.def("update_subtree_depth_batch", &update_subtree_depth_batch_cuda,
        "Adds a delta to the depth of all nodes in a subtree.",
        py::arg("trees"),
        py::arg("root_indices"),
        py::arg("delta"),
        py::arg("max_nodes"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"));

    // This binding exposes the get_subtree_max_depth_batch_cuda kernel to Python.
    m.def("get_subtree_max_depth_batch", &get_subtree_max_depth_batch_cuda,
        "Returns the maximum depth of a subtree for each (batch_index, root_index) pair.",
        py::arg("trees"),
        py::arg("root_indices"),
        py::arg("max_nodes"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("out_max_depths"));

    // This binding exposes the find_empty_slots_batch_cuda kernel to Python.
    m.def("find_empty_slots_batch", &find_empty_slots_batch_cuda,
        "Finds the first 'count' unused slots in each tree.",
        py::arg("trees"),
        py::arg("count"),
        py::arg("max_nodes"),
        py::arg("out_indices"));
}
