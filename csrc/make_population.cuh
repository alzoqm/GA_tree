// csrc/make_population.cuh
// This header file declares the CUDA function for initializing a population of trees.

#pragma once
#include <torch/extension.h>

/**
 * @brief Initializes a batch of GA-Trees on the GPU with strict invariant enforcement.
 *
 * This function generates a population of trees in parallel on the GPU. It ensures that the
 * generated trees are structurally valid and adhere to a set of predefined constraints.
 * The key features of this function are:
 *
 * - **Invariant Enforcement**: Guarantees that there is no mixing of DECISION and ACTION nodes
 *   under the same parent, all leaves are ACTION nodes, and all intermediate nodes are DECISION nodes.
 * - **Single-Action Parent Rule**: If a parent has an ACTION child, it must have exactly one child.
 * - **Resource Limits**: Respects the maximum depth and number of children per node.
 * - **Minimal Tree Guarantee**: Ensures that each tree has at least a root and an action node.
 * - **Safe Memory Access**: Uses a frontier reservation guard to prevent memory access issues.
 * - **Infinite Loop Protection**: Includes an iteration limit to prevent infinite loops.
 *
 * @param trees The population tensor to be initialized (B, N, D).
 * @param total_budget The number of nodes to generate for each tree (B,).
 * @param max_children The maximum number of children a node can have.
 * @param max_depth The maximum depth of the trees.
 * @param max_nodes The maximum number of nodes per tree (must be equal to N).
 * @param bfs_q A buffer for the Breadth-First Search queue (B, 2*N).
 * @param scratch A scratch buffer for the freelist and other temporary data (B, 2*N).
 * @param child_cnt A buffer to store the child count for each node (B, N).
 * @param act_cnt A buffer to store the action child count for each node (B, N).
 * @param dec_cnt A buffer to store the decision child count for each node (B, N).
 * @param cand_idx A buffer for candidate indices (B, N).
 * @param cand_w A buffer for candidate weights (B, N).
 * @param num_feat_indices Indices of numerical features.
 * @param num_feat_minmax Min/max ranges for numerical features.
 * @param bool_feat_indices Indices of boolean features.
 * @param ff_pairs Pairs of features for feature-feature comparisons.
 * @param long_actions Allowed actions for the LONG context.
 * @param hold_actions Allowed actions for the HOLD context.
 * @param short_actions Allowed actions for the SHORT context.
 */
void init_population_cuda(
    torch::Tensor trees,            // (B,N,D) float32, CUDA, contiguous
    torch::Tensor total_budget,     // (B,)    int32,  CUDA
    int max_children,
    int max_depth,
    int max_nodes,
    // Work buffers (all CUDA, preallocated in Python)
    torch::Tensor bfs_q,            // (B, 2N) int32
    torch::Tensor scratch,          // (B, 2N) int32
    torch::Tensor child_cnt,        // (B, N)  int32
    torch::Tensor act_cnt,          // (B, N)  int32
    torch::Tensor dec_cnt,          // (B, N)  int32
    torch::Tensor cand_idx,         // (B, N)  int32
    torch::Tensor cand_w,           // (B, N)  float32
    // Feature tables
    torch::Tensor num_feat_indices, // (Kn,)   int32
    torch::Tensor num_feat_minmax,  // (Kn,2)  float32
    torch::Tensor bool_feat_indices,// (Kb,)   int32
    torch::Tensor ff_pairs,         // (P,2)   int32
    // Action allow-lists per root-context
    torch::Tensor long_actions,     // (La,)   int32
    torch::Tensor hold_actions,     // (Ha,)   int32
    torch::Tensor short_actions     // (Sa,)   int32
);