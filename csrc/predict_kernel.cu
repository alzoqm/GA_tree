// csrc/predict_kernel.cu
// This file contains the CUDA kernel for performing predictions with a GA-Tree.
// The kernel is optimized for performance and includes safety guarantees.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include "constants.h"
#include "predict_kernel.cuh"

// Macro for checking CUDA errors.
#define CUDA_CHECK_ERRORS() \
  do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  }} while (0)

// Utility functions for type conversion and pointer arithmetic.
__device__ __forceinline__ int   f2i(float x){ return __float2int_rn(x); }
__device__ __forceinline__ float i2f(int   x){ return static_cast<float>(x); }
__device__ __forceinline__ bool  valid_idx(int idx, int N){ return idx >= 0 && idx < N; }

__device__ __forceinline__ const float* node_ptr(const float* base, int D, int i){
    return base + (size_t)i * D;
}

/**
 * @brief Evaluates a decision node with robust bounds checking.
 *
 * @param nb A pointer to the node data.
 * @param feat A pointer to the feature array.
 * @param F The number of features.
 * @return true if the node's condition is satisfied, false otherwise.
 */
__device__ __forceinline__ bool eval_node(
    const float* __restrict__ nb,
    const float* __restrict__ feat,
    int F
){
    const int comp_type = f2i(nb[COL_PARAM_3]);
    const int feat1     = f2i(nb[COL_PARAM_1]);

    if (feat1 < 0 || feat1 >= F) return false;
    const float v1 = feat[feat1];

    float v2 = nb[COL_PARAM_4];

    if (comp_type == COMP_TYPE_FEAT_FEAT){
        const int feat2 = f2i(nb[COL_PARAM_4]);
        if (feat2 < 0 || feat2 >= F) return false;
        v2 = feat[feat2];
    } else if (comp_type == COMP_TYPE_FEAT_BOOL){
        return v1 == v2;
    }

    const int op = f2i(nb[COL_PARAM_2]);
    if (op == OP_GTE) return v1 >= v2;
    if (op == OP_LTE) return v1 <= v2;
    return false;
}

/**
 * @brief The main prediction kernel. Each block processes one tree.
 *
 * This kernel uses a serial BFS traversal within each block for robustness.
 * It includes several safety guarantees, such as bounds checking, a visited bitmap
 * to prevent infinite loops, and a step counter.
 */
__global__ void k_predict_batch(
    const float* __restrict__ trees,   int B, int N, int D,
    const float* __restrict__ features,int F,
    const int*   __restrict__ positions,
    const int*   __restrict__ offsets,
    const int*   __restrict__ children,int Emax,
    float*       __restrict__ results,
    int*         __restrict__ bfs_q,
    int*         __restrict__ visited_gen,
    const int    step_id,
    int*         __restrict__ out_action_node_idx
){
    const int b = blockIdx.x;
    if (b >= B) return;

    const float* tree = trees + (size_t)b * N * D;
    const int*   off  = offsets + (size_t)b * (N + 1);
    const int*   col  = children + (size_t)b * Emax;
    float*       rout = results + (size_t)b * 4;
    int*         q    = bfs_q       + (size_t)b * N;
    int*         vis  = visited_gen + (size_t)b * N;

    if (threadIdx.x == 0){
        rout[0] = i2f(ACTION_NOT_FOUND);
        rout[1] = 0.0f;
        rout[2] = 0.0f;
        rout[3] = 0.0f;
        q[0]    = -1;
        if (out_action_node_idx) out_action_node_idx[b] = -1;
    }
    __syncthreads();

    const int eN = off[N];
    if (eN < 0 || eN > Emax) return;

    int start = -1;
    if (threadIdx.x == 0){
        const int want = positions[b];
        for (int i = 0; i < 3; ++i){
            if (!valid_idx(i, N)) continue;
            const float* nb = node_ptr(tree, D, i);
            const int type = f2i(nb[COL_NODE_TYPE]);
            if (type == NODE_TYPE_ROOT_BRANCH && f2i(nb[COL_PARAM_1]) == want){
                start = i;
                break;
            }
        }
        if (start >= 0){
            q[0] = start;
            vis[start] = step_id;
        }
    }
    __syncthreads();

    if (q[0] < 0) return;

    if (threadIdx.x == 0){
        int qh = 0, qt = 1;
        int steps = 0;
        bool found = false;

        while (qh < qt && steps < N && !found){
            const int u = q[qh++];
            if (!valid_idx(u, N)) { 
                ++steps; 
                continue; 
            }

            int s = off[u];
            int e = (u + 1 <= N) ? off[u + 1] : s;
            if (s < 0) s = 0;
            if (e < s) e = s;
            if (e > eN) e = eN;

            for (int p = s; p < e; ++p){
                const int v = col[p];
                if (!valid_idx(v, N)) continue;

                const float* nb = node_ptr(tree, D, v);
                const int nt = f2i(nb[COL_NODE_TYPE]);

                if (nt == NODE_TYPE_ACTION){
                    rout[0] = nb[COL_PARAM_1];
                    rout[1] = nb[COL_PARAM_2];
                    rout[2] = nb[COL_PARAM_3];
                    rout[3] = nb[COL_PARAM_4];
                    if (out_action_node_idx) out_action_node_idx[b] = v;
                    found = true;
                    break;
                }
                else if (nt == NODE_TYPE_DECISION){
                    if (eval_node(nb, features, F) && vis[v] != step_id){
                        vis[v] = step_id;
                        if (qt < N) q[qt++] = v;
                    }
                }
            }
            ++steps;
        }
    }
}

/**
 * @brief The host-side launcher for the prediction kernel.
 */
void predict_cuda(
    torch::Tensor trees,
    torch::Tensor features,
    torch::Tensor positions,
    torch::Tensor offsets,
    torch::Tensor children,
    torch::Tensor results,
    torch::Tensor bfs_q,
    torch::Tensor visited_gen,
    int step_id,
    torch::Tensor action_node_idx
){
    TORCH_CHECK(trees.is_cuda() && trees.dtype()==torch::kFloat32 && trees.dim()==3 && trees.is_contiguous(), "trees must be (B,N,D) float32 CUDA contiguous");
    TORCH_CHECK(features.is_cuda() && features.dtype()==torch::kFloat32 && features.dim()==1, "features must be (F,) float32 CUDA");
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==torch::kInt32 && positions.dim()==1, "positions must be (B,) int32 CUDA");
    TORCH_CHECK(offsets.is_cuda() && offsets.dtype()==torch::kInt32 && offsets.dim()==2, "offsets must be (B,N+1) int32 CUDA");
    TORCH_CHECK(children.is_cuda() && children.dtype()==torch::kInt32 && children.dim()==2, "children must be (B,Emax) int32 CUDA");
    TORCH_CHECK(results.is_cuda() && results.dtype()==torch::kFloat32 && results.dim()==2 && results.size(1)==4, "results must be (B,4) float32 CUDA");
    TORCH_CHECK(bfs_q.is_cuda() && bfs_q.dtype()==torch::kInt32 && bfs_q.dim()==2, "bfs_q must be (B,N) int32 CUDA");
    TORCH_CHECK(visited_gen.is_cuda() && visited_gen.dtype()==torch::kInt32 && visited_gen.dim()==2, "visited_gen must be (B,N) int32 CUDA");
    TORCH_CHECK(action_node_idx.is_cuda() && action_node_idx.dtype()==torch::kInt32 && action_node_idx.dim()==1, "action_node_idx must be (B,) int32 CUDA");

    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);

    TORCH_CHECK(D == NODE_INFO_DIM, "Node dimension must match NODE_INFO_DIM");
    TORCH_CHECK(offsets.size(0)==B && offsets.size(1)==N+1, "offsets must be (B,N+1)");
    TORCH_CHECK(children.size(0)==B, "children must have first dim B");
    TORCH_CHECK(bfs_q.size(0)==B && bfs_q.size(1)==N, "bfs_q must be (B,N)");
    TORCH_CHECK(visited_gen.size(0)==B && visited_gen.size(1)==N, "visited_gen must be (B,N)");
    TORCH_CHECK(results.size(0)==B, "results must have first dim B");
    TORCH_CHECK(positions.size(0)==B, "positions must have first dim B");
    TORCH_CHECK(action_node_idx.size(0)==B, "action_node_idx must have first dim B");

    const int Emax = children.size(1);
    const int F = features.size(0);
    TORCH_CHECK(Emax >= 0, "Emax must be >= 0");
    TORCH_CHECK(F >= 0, "Feature count must be >= 0");

    if (B == 0) return;

    dim3 grid(B), block(1);

    k_predict_batch<<<grid, block>>>(
        trees.data_ptr<float>(), B, N, D,
        features.data_ptr<float>(), F,
        positions.data_ptr<int>(),
        offsets.data_ptr<int>(),
        children.data_ptr<int>(), Emax,
        results.data_ptr<float>(),
        bfs_q.data_ptr<int>(),
        visited_gen.data_ptr<int>(),
        step_id,
        action_node_idx.data_ptr<int>()
    );

    CUDA_CHECK_ERRORS();
}
