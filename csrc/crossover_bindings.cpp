// csrc/crossover_bindings.cpp
// This file contains the Python bindings for the CUDA crossover kernels.
// It uses pybind11 to expose the C++/CUDA functions to Python.

#include <torch/extension.h>
#include "crossover_kernel.cuh"
#include "node_crossover_kernel.cuh"
#include "subtree_crossover_kernel.cuh"
#include "root_crossover_kernel.cuh"

// The PYBIND11_MODULE macro creates a Python module.
// The first argument is the name of the module, and the second is a variable
// that represents the module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance CUDA crossover kernels for GATree";

    // --- Crossover-related CUDA function bindings ---

    // This binding exposes the get_contextual_mask_cuda function to Python.
    // It creates a new tensor for the output mask and then calls the CUDA kernel.
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

    // This is a direct binding to the get_contextual_mask_cuda function.
    // It is more efficient as it avoids creating a new tensor for the output mask.
    m.def("get_contextual_mask_cuda", &get_contextual_mask_cuda,
        "Get a mask for nodes matching a specific type and root branch context (direct).",
        py::arg("trees"), py::arg("output_mask"), py::arg("node_type"), py::arg("branch_type")
    );

    // This binding exposes the swap_node_params_cuda function to Python.
    m.def("swap_node_params", &swap_node_params_cuda,
        "Swap node parameters between two populations based on given masks.",
        py::arg("c1"), py::arg("c2"), py::arg("p1_mask"), py::arg("p2_mask")
    );

    // This is a direct binding to the swap_node_params_cuda function.
    m.def("swap_node_params_cuda", &swap_node_params_cuda,
        "Swap node parameters between two populations based on given masks (direct).",
        py::arg("c1"), py::arg("c2"), py::arg("p1_mask"), py::arg("p2_mask")
    );

    // This binding exposes the copy_branches_batch_cuda function to Python.
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

    // This is a direct binding to the copy_branches_batch_cuda function.
    m.def("copy_branches_batch_cuda", &copy_branches_batch_cuda,
        "Performs root branch crossover on a batch of parents using CUDA (direct).",
        py::arg("child_batch"), 
        py::arg("p1_batch"), 
        py::arg("p2_batch"), 
        py::arg("donor_map"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("old_to_new_map_buffer")
    );

    // This binding exposes the repair_after_root_branch_cuda function to Python.
    m.def("repair_after_root_branch", &repair_after_root_branch_cuda,
        "Repair structural invariants after root-branch crossover using preallocated buffers.",
        py::arg("trees"),
        py::arg("child_count_buffer"),
        py::arg("act_cnt_buffer"),
        py::arg("dec_cnt_buffer"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"));

    // This is a direct binding to the repair_after_root_branch_cuda function.
    m.def("repair_after_root_branch_cuda", &repair_after_root_branch_cuda,
        "Repair structural invariants after root-branch crossover using preallocated buffers (direct).",
        py::arg("trees"),
        py::arg("child_count_buffer"),
        py::arg("act_cnt_buffer"),
        py::arg("dec_cnt_buffer"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"));

    // This binding exposes the root_branch_sampling_crossover_cuda function to Python.
    m.def("root_branch_sampling_crossover", &root_branch_sampling_crossover_cuda,
        "Parent-pool sampling root branch crossover with proactive size validation and retry mechanism.",
        py::arg("child_batch"),
        py::arg("parent_pool"),
        py::arg("max_nodes"),
        py::arg("max_retries"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("old_to_new_map_buffer"));

    // This is a direct binding to the root_branch_sampling_crossover_cuda function.
    m.def("root_branch_sampling_crossover_cuda", &root_branch_sampling_crossover_cuda,
        "Parent-pool sampling root branch crossover with proactive size validation and retry mechanism (direct).",
        py::arg("child_batch"),
        py::arg("parent_pool"),
        py::arg("max_nodes"),
        py::arg("max_retries"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("old_to_new_map_buffer"));
    
    // This binding exposes the subtree_crossover_batch_cuda function to Python.
    m.def("subtree_crossover_batch", &subtree_crossover_batch_cuda,
        "Performs subtree crossover on a batch of parents using CUDA.",
        py::arg("child1_out"),
        py::arg("child2_out"),
        py::arg("p1_batch"),
        py::arg("p2_batch"),
        py::arg("mode"),
        py::arg("max_nodes"),
        py::arg("max_retries"),
        py::arg("branch_perm"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("old_to_new_map_buffer"),
        py::arg("p1_candidates_buffer"),
        py::arg("p2_candidates_buffer")
    );

    // This is a direct binding to the subtree_crossover_batch_cuda function.
    m.def("subtree_crossover_batch_cuda", &subtree_crossover_batch_cuda,
        "Performs subtree crossover on a batch of parents using CUDA (direct).",
        py::arg("child1_out"),
        py::arg("child2_out"),
        py::arg("p1_batch"),
        py::arg("p2_batch"),
        py::arg("mode"),
        py::arg("max_nodes"),
        py::arg("max_retries"),
        py::arg("branch_perm"),
        py::arg("bfs_queue_buffer"),
        py::arg("result_indices_buffer"),
        py::arg("old_to_new_map_buffer"),
        py::arg("p1_candidates_buffer"),
        py::arg("p2_candidates_buffer")
    );
}