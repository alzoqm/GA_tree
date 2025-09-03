// csrc/adjacency_builder.cuh
// This header file declares the functions for building and validating tree
// adjacency structures on the GPU. These functions are implemented in
// adjacency_builder.cu and are called from the C++ bindings.

#pragma once
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <tuple>

namespace py = pybind11;

/**
 * @brief Pass 1: Counts children per parent and creates CSR offsets.
 * 
 * This function is the first pass in a two-pass process to build a Compressed
 * Sparse Row (CSR) representation of the tree adjacencies. It also detects
 * if any parent node has more children than a specified maximum.
 * 
 * @param trees A (B, N, D) float32 CUDA tensor representing a batch of trees.
 * @param max_children The maximum number of children a parent node is allowed to have.
 * 
 * @return A tuple containing:
 *         - total_children: A long integer with the total number of child nodes across all trees.
 *         - offsets_flat: A (B*N+1) int32 CUDA tensor for the CSR representation.
 *         - overflow_mask: A (B) int32 CUDA tensor where a 1 indicates that a tree has at least one parent with too many children.
 */
std::tuple<long, torch::Tensor, torch::Tensor> count_and_create_offsets_cuda(
    const torch::Tensor& trees,
    int max_children
);

/**
 * @brief Pass 2: Fills the child_indices array and optionally sorts children.
 * 
 * This is the second pass of the CSR construction. It populates the child_indices
 * array. It can dynamically switch between a shared-memory-based implementation
 * and a global-memory-based one depending on the available resources and the
 * problem size.
 *
 * @param trees A (B, N, D) float32 CUDA tensor representing a batch of trees.
 * @param offsets_flat The (B*N+1) int32 CUDA CSR offset tensor from the first pass.
 * @param child_indices A (total_children) int32 CUDA tensor to be filled with child indices.
 * @param max_children The maximum number of children a parent node is allowed to have.
 * @param sort_children A boolean indicating whether to sort the children of each parent by their index.
 */
void fill_child_indices_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& offsets_flat,
    torch::Tensor&       child_indices,
    int                  max_children,
    bool                 sort_children = true
);

/**
 * @brief Validates the structural integrity of the trees using the CSR representation.
 * 
 * This function checks for various structural violations in the trees, such as mixed
 * child types, incorrect depths, and invalid parent-child relationships. The results
 * are returned as a bitmask for each tree.
 * 
 * Violations (ViolBits):
 *  - V_MIXED_CHILDREN:   A parent has both ACTION and DECISION children.
 *  - V_LEAF_NOT_ACTION:  A leaf node is not an ACTION node (roots are excluded).
 *  - V_ACTION_HAS_CHILD: An ACTION node has one or more children.
 *  - V_SINGLE_ACTION_BR: A parent has action children, but the degree is not 1.
 *  - V_DEPTH_MISMATCH:   A child's depth is not parent's depth + 1.
 *  - V_CHILD_OVERFLOW:   A parent has more children than max_children.
 *  - V_BAD_PARENT:       A child index is out of the valid range [0, N).
 *  - V_ROOT_BROKEN:      A node with parent == -1 is not a ROOT_BRANCH or its depth is not 0.
 *  - V_ROOT_LEAF:        A root node (parent == -1) has no children.
 * 
 * @param trees A (B, N, D) float32 CUDA tensor representing a batch of trees.
 * @param offsets_flat The (B*N+1) int32 CUDA CSR offset tensor.
 * @param child_indices The (total_children) int32 CUDA tensor of child indices.
 * @param out_violation_mask A (B) int32 CUDA tensor to store the violation bitmasks.
 */
void validate_adjacency_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& offsets_flat,
    const torch::Tensor& child_indices,
    torch::Tensor        out_violation_mask
);

/**
 * @brief Builds adjacency dictionaries for synchronization with Python.
 * 
 * This function provides a way to get the tree adjacency information back to Python.
 * It builds the adjacency lists for all trees in parallel on the GPU.
 * 
 * @param trees A (B, N, D) float32 CUDA tensor representing a batch of trees.
 * @param next_idxs A (B) int32 CUDA tensor indicating the number of active nodes in each tree.
 * 
 * @return A tuple containing:
 *         - offsets: A (B*N+1) int32 CUDA CSR offset tensor.
 *         - children: A (total_children) int32 CUDA tensor of child node indices.
 */
std::tuple<torch::Tensor, torch::Tensor> build_adjacency_dicts_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& next_idxs
);

/**
 * @brief A memory-optimized function to build and return adjacency dictionaries directly as Python objects.
 * 
 * This function is designed to be highly efficient by avoiding Python loops for dictionary
 * construction. It builds the adjacency dictionaries in C++ and returns them as a Python
 * list of dictionaries.
 * 
 * @param trees A (B, N, D) float32 CUDA tensor representing a batch of trees.
 * @param next_idxs A (B) int32 CUDA tensor indicating the number of active nodes in each tree.
 * 
 * @return A py::list containing B Python dictionaries, where each dictionary represents the
 *         adjacency list of a tree (parent_idx -> [child_indices]).
 */
py::list build_adjacency_dicts_optimized_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& next_idxs
);