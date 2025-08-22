// csrc/predict_kernel.cu - Optimized GA-Tree Prediction Kernel
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "constants.h"
#include "predict_kernel.cuh"

#define CUDA_CHECK_ERRORS() \
  do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  }} while (0)

// Utility functions (matching make_population style)
__device__ __forceinline__ int   f2i(float x){ return __float2int_rn(x); }
__device__ __forceinline__ float i2f(int   x){ return static_cast<float>(x); }
__device__ __forceinline__ bool  valid_idx(int idx, int N){ return idx >= 0 && idx < N; }

__device__ __forceinline__ const float* node_ptr(const float* base, int D, int i){ 
    return base + (size_t)i * D; 
}

/**
 * Robust node evaluation with aggressive bounds checking
 * 
 * @param nb           Node data pointer (7 floats)
 * @param feat         Feature array
 * @param F            Number of features
 * @return             true if node condition is satisfied
 */
__device__ __forceinline__ bool eval_node(
    const float* __restrict__ nb,
    const float* __restrict__ feat,
    int F
){
    const int comp_type = f2i(nb[COL_PARAM_3]);
    const int feat1     = f2i(nb[COL_PARAM_1]);
    
    // Bounds check for first feature
    if (feat1 < 0 || feat1 >= F) return false;
    const float v1 = feat[feat1];

    float v2 = nb[COL_PARAM_4];
    
    // Handle feature-feature comparison
    if (comp_type == COMP_TYPE_FEAT_FEAT){
        const int feat2 = f2i(nb[COL_PARAM_4]);
        if (feat2 < 0 || feat2 >= F) return false;
        v2 = feat[feat2];
    } else if (comp_type == COMP_TYPE_FEAT_BOOL){
        // Boolean comparison: exact equality for {0.0, 1.0}
        return v1 == v2;
    }

    // Numeric comparison
    const int op = f2i(nb[COL_PARAM_2]);
    if (op == OP_GTE) return v1 >= v2;
    if (op == OP_LTE) return v1 <= v2;
    return false;
}

/**
 * Main prediction kernel: One block per tree architecture
 * 
 * Each block handles one tree with serial BFS in thread 0 for robustness.
 * All buffers are pre-allocated and passed from Python to prevent fixed-size limitations.
 * 
 * Safety guarantees:
 * - Aggressive bounds checking on all array accesses
 * - Visited bitmap prevents infinite loops
 * - Step counter limits traversal to N iterations max
 * - CSR validation ensures well-formed adjacency
 */
__global__ void k_predict_batch(
    const float* __restrict__ trees,   int B, int N, int D,
    const float* __restrict__ features,int F,
    const int*   __restrict__ positions,
    const int*   __restrict__ offsets, // (B, N+1)
    const int*   __restrict__ children,int Emax,
    float*       __restrict__ results, // (B,4)
    int*         __restrict__ bfs_q,   // (B,N)
    int*         __restrict__ visited  // (B,N)
){
    const int b = blockIdx.x;
    if (b >= B) return;

    // Per-tree data pointers
    const float* tree = trees + (size_t)b * N * D;
    const int*   off  = offsets + (size_t)b * (N + 1);
    const int*   col  = children + (size_t)b * Emax;
    float*       rout = results + (size_t)b * 4;
    int*         q    = bfs_q   + (size_t)b * N;
    int*         vis  = visited + (size_t)b * N;

    // Initialize buffers in parallel within block
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        vis[i] = 0;
        q[i]   = -1;
    }
    __syncthreads();

    // Initialize result to "not found" state
    if (threadIdx.x == 0){
        rout[0] = i2f(ACTION_NOT_FOUND);
        rout[1] = 0.0f;
        rout[2] = 0.0f;
        rout[3] = 0.0f;
    }
    __syncthreads();

    // CSR validation: Check edge count is within bounds
    const int eN = off[N];  // Total edge count for this tree
    if (eN < 0 || eN > Emax) return;  // Early return on malformed CSR

    // Root selection (thread 0 only for deterministic behavior)
    int start = -1;
    if (threadIdx.x == 0){
        const int want = positions[b];
        
        // Search for matching root branch in nodes 0, 1, 2
        for (int i = 0; i < 3; ++i){
            if (!valid_idx(i, N)) continue;
            
            const float* nb = node_ptr(tree, D, i);
            const int type = f2i(nb[COL_NODE_TYPE]);
            
            if (type == NODE_TYPE_ROOT_BRANCH && f2i(nb[COL_PARAM_1]) == want){
                start = i;
                break;
            }
        }
        
        // Seed BFS queue if valid root found
        if (start >= 0){
            q[0] = start;
            vis[start] = 1;
        }
    }
    __syncthreads();

    // Early return if no valid root found
    if (q[0] < 0) return;

    // BFS traversal (thread 0 only for simplicity and correctness)
    if (threadIdx.x == 0){
        int qh = 0, qt = 1;  // queue head, tail
        int steps = 0;
        bool found = false;

        // Main BFS loop with safety guards
        while (qh < qt && steps < N && !found){
            const int u = q[qh++];
            
            // Bounds check current node
            if (!valid_idx(u, N)) { 
                ++steps; 
                continue; 
            }

            // CSR row bounds: [s, e) where s = off[u], e = off[u+1]
            int s = off[u];
            int e = (u + 1 <= N) ? off[u + 1] : s;
            
            // Clamp to valid ranges
            if (s < 0) s = 0;
            if (e < s) e = s;
            if (e > eN) e = eN;

            // Process all children in CSR row
            for (int p = s; p < e; ++p){
                const int v = col[p];
                
                // Bounds check child index
                if (!valid_idx(v, N)) continue;

                const float* nb = node_ptr(tree, D, v);
                const int nt = f2i(nb[COL_NODE_TYPE]);

                if (nt == NODE_TYPE_ACTION){
                    // Found action node - extract parameters and exit
                    rout[0] = nb[COL_PARAM_1];  // action type
                    rout[1] = nb[COL_PARAM_2];  // param 2
                    rout[2] = nb[COL_PARAM_3];  // param 3
                    rout[3] = nb[COL_PARAM_4];  // param 4
                    found = true;
                    break;
                }
                else if (nt == NODE_TYPE_DECISION){
                    // Evaluate decision node and add to queue if true and unvisited
                    if (eval_node(nb, features, F) && !vis[v]){
                        vis[v] = 1;
                        if (qt < N) q[qt++] = v;  // Queue capacity guard
                    }
                }
            }
            ++steps;  // Increment step counter for infinite loop protection
        }
    }
}

/**
 * Host launcher with comprehensive validation (matching make_population style)
 */
void predict_cuda(
    torch::Tensor trees,
    torch::Tensor features,
    torch::Tensor positions,
    torch::Tensor offsets,
    torch::Tensor children,
    torch::Tensor results,
    torch::Tensor bfs_q,
    torch::Tensor visited
){
    // Comprehensive input validation
    TORCH_CHECK(trees.is_cuda() && trees.dtype()==torch::kFloat32 && trees.dim()==3 && trees.is_contiguous(),
                "trees must be (B,N,D) float32 CUDA contiguous");
    TORCH_CHECK(features.is_cuda() && features.dtype()==torch::kFloat32 && features.dim()==1,
                "features must be (F,) float32 CUDA");
    TORCH_CHECK(positions.is_cuda() && positions.dtype()==torch::kInt32 && positions.dim()==1,
                "positions must be (B,) int32 CUDA");
    TORCH_CHECK(offsets.is_cuda() && offsets.dtype()==torch::kInt32 && offsets.dim()==2,
                "offsets must be (B,N+1) int32 CUDA");
    TORCH_CHECK(children.is_cuda() && children.dtype()==torch::kInt32 && children.dim()==2,
                "children must be (B,Emax) int32 CUDA");
    TORCH_CHECK(results.is_cuda() && results.dtype()==torch::kFloat32 && results.dim()==2 && results.size(1)==4,
                "results must be (B,4) float32 CUDA");
    TORCH_CHECK(bfs_q.is_cuda() && bfs_q.dtype()==torch::kInt32 && bfs_q.dim()==2,
                "bfs_q must be (B,N) int32 CUDA");
    TORCH_CHECK(visited.is_cuda() && visited.dtype()==torch::kInt32 && visited.dim()==2,
                "visited must be (B,N) int32 CUDA");

    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);
    
    // Validate tensor dimensions
    TORCH_CHECK(D == NODE_INFO_DIM, "Node dimension must match NODE_INFO_DIM");
    TORCH_CHECK(offsets.size(0)==B && offsets.size(1)==N+1, "offsets must be (B,N+1)");
    TORCH_CHECK(children.size(0)==B, "children must have first dim B");
    TORCH_CHECK(bfs_q.size(0)==B && bfs_q.size(1)==N, "bfs_q must be (B,N)");
    TORCH_CHECK(visited.size(0)==B && visited.size(1)==N, "visited must be (B,N)");
    TORCH_CHECK(results.size(0)==B, "results must have first dim B");
    TORCH_CHECK(positions.size(0)==B, "positions must have first dim B");

    const int Emax = children.size(1);
    const int F = features.size(0);
    TORCH_CHECK(Emax >= 0, "Emax must be >= 0");
    TORCH_CHECK(F >= 0, "Feature count must be >= 0");

    if (B == 0) return;  // Early return for empty batch

    // Launch configuration: One block per tree
    dim3 grid(B), block(128);
    
    k_predict_batch<<<grid, block>>>(
        trees.data_ptr<float>(), B, N, D,
        features.data_ptr<float>(), F,
        positions.data_ptr<int>(),
        offsets.data_ptr<int>(),
        children.data_ptr<int>(), Emax,
        results.data_ptr<float>(),
        bfs_q.data_ptr<int>(),
        visited.data_ptr<int>()
    );
    
    CUDA_CHECK_ERRORS();
}