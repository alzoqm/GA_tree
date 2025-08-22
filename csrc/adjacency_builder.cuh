// csrc/adjacency_builder.cuh
#pragma once
#include <torch/extension.h>
#include <tuple>

/**
 * Pass 1: Count children and produce CSR offsets, with overflow detection.
 * 
 * @param trees          (B,N,D) float32 CUDA contiguous
 * @param max_children   Maximum allowed children per parent
 * @return (total_children, offsets_flat, overflow_mask)
 *         total_children: long
 *         offsets_flat:   (B*N+1) int32 CUDA
 *         overflow_mask:  (B)     int32 CUDA (0 if OK, 1 if any parent in tree overflowed)
 */
std::tuple<long, torch::Tensor, torch::Tensor> count_and_create_offsets_cuda(
    const torch::Tensor& trees,
    int max_children
);

/**
 * Pass 2: Fill child_indices and (optionally) sort per parent.
 * 
 * - Uses dynamic shared memory only (no fixed arrays)
 * - Automatically disables sorting if shared memory budget cannot accommodate it
 * - Automatically switches to a global-cursor mode if tiling would cause excessive rescans
 *
 * @param trees           (B,N,D) float32 CUDA contiguous
 * @param offsets_flat    (B*N+1) int32 CUDA
 * @param child_indices   (total_children) int32 CUDA (output filled in-place)
 * @param max_children    Maximum allowed children per parent
 * @param sort_children   Whether to sort child lists per parent (default true)
 */
void fill_child_indices_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& offsets_flat,
    torch::Tensor&       child_indices,
    int                  max_children,
    bool                 sort_children = true
);

/**
 * Validator: CSR-based structural checks. Writes per-tree bitmask (0 means OK).
 * 
 * Bits (ViolBits):
 *  - V_MIXED_CHILDREN   (1<<0)  parent has both ACTION and DECISION children
 *  - V_LEAF_NOT_ACTION  (1<<1)  leaf that is not ACTION (excluding roots)
 *  - V_ACTION_HAS_CHILD (1<<2)  ACTION node with any child
 *  - V_SINGLE_ACTION_BR (1<<3)  parent has action children but deg != 1 (single-action rule)
 *  - V_DEPTH_MISMATCH   (1<<4)  child's depth != parent's + 1, or >= max_depth
 *  - V_CHILD_OVERFLOW   (1<<5)  degree > max_children
 *  - V_BAD_PARENT       (1<<6)  child index out of [0,N)
 *  - V_ROOT_BROKEN      (1<<7)  parent == -1 but node not ROOT_BRANCH or depth != 0
 *  - V_ROOT_LEAF        (1<<8)  root (parent == -1) has no children (optional diagnostic)
 */
void validate_adjacency_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& offsets_flat,
    const torch::Tensor& child_indices,
    int                  max_children,
    int                  max_depth,
    torch::Tensor        out_violation_mask
);