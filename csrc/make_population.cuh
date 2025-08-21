// csrc/make_population.cuh
#pragma once
#include <torch/extension.h>

/**
 * GPU-parallel GA-Tree population initialization with strict invariant enforcement
 * 
 * This function initializes a batch of GA-Tree population on GPU with the following guarantees:
 * - No mixing of DECISION and ACTION under the same parent
 * - If a parent has an ACTION child, it has exactly one child (single-action parent rule)
 * - All leaves are ACTION nodes
 * - All intermediate nodes are DECISION nodes  
 * - Depth/arity limits respected (max_depth, max_children)
 * - Minimal tree guaranteed (≥2 nodes; root→ACTION form permitted)
 * - Safe memory access with frontier reservation guards
 * - Infinite loop protection with iteration limits
 * 
 * @param trees            (B,N,D) float32, CUDA, contiguous - population tensor
 * @param total_budget     (B,) int32, CUDA - node budget per tree
 * @param max_children     Maximum children per node
 * @param max_depth        Maximum tree depth
 * @param max_nodes        Maximum nodes per tree (must equal N)
 * @param bfs_q           (B, 2N) int32, CUDA - BFS queue buffer
 * @param scratch         (B, 2N) int32, CUDA - scratch buffer for freelist
 * @param child_cnt       (B, N) int32, CUDA - child count per node
 * @param act_cnt         (B, N) int32, CUDA - action child count per node
 * @param dec_cnt         (B, N) int32, CUDA - decision child count per node
 * @param cand_idx        (B, N) int32, CUDA - candidate indices buffer
 * @param cand_w          (B, N) float32, CUDA - candidate weights buffer
 * @param num_feat_indices (Kn,) int32, CUDA - numeric feature indices (may be empty)
 * @param num_feat_minmax  (Kn,2) float32, CUDA - numeric feature min/max ranges
 * @param bool_feat_indices (Kb,) int32, CUDA - boolean feature indices (may be empty)
 * @param ff_pairs         (P,2) int32, CUDA - feature-feature pairs (may be empty)
 * @param long_actions     (La,) int32, CUDA - allowed actions for LONG context
 * @param hold_actions     (Ha,) int32, CUDA - allowed actions for HOLD context
 * @param short_actions    (Sa,) int32, CUDA - allowed actions for SHORT context
 */
void init_population_cuda(
    torch::Tensor trees,            // (B,N,D) float32, CUDA, contiguous
    torch::Tensor total_budget,     // (B,)    int32,  CUDA
    int max_children,
    int max_depth,
    int max_nodes,
    // Work buffers (all CUDA, preallocated in Python)
    torch::Tensor bfs_q,            // (B, 2N) int32
    torch::Tensor scratch,          // (B, 2N) int32  (SCR_FC=0, SCR_CUR=1, SCR_CC=2, payload from 3..)
    torch::Tensor child_cnt,        // (B, N)  int32
    torch::Tensor act_cnt,          // (B, N)  int32
    torch::Tensor dec_cnt,          // (B, N)  int32
    torch::Tensor cand_idx,         // (B, N)  int32
    torch::Tensor cand_w,           // (B, N)  float32
    // Feature tables
    torch::Tensor num_feat_indices, // (Kn,)   int32   indices in ALL_FEATURES
    torch::Tensor num_feat_minmax,  // (Kn,2)  float32
    torch::Tensor bool_feat_indices,// (Kb,)   int32
    torch::Tensor ff_pairs,         // (P,2)   int32   (feat1, feat2)
    // Action allow-lists per root-context
    torch::Tensor long_actions,     // (La,)   int32
    torch::Tensor hold_actions,     // (Ha,)   int32
    torch::Tensor short_actions     // (Sa,)   int32
);