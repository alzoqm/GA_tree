// csrc/adjacency_builder.cu
// This file contains CUDA kernels and functions for building adjacency information
// from a batch of trees represented as tensors. This is a critical step for
// efficiently processing tree structures on the GPU.

#include "adjacency_builder.cuh"
#include "constants.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>

// Macro for checking CUDA errors. This is a common utility for debugging CUDA code.
#define CUDA_CHECK_ERRORS() 
  do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { 
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); 
  }} while (0)

// Utility device functions for type conversion and pointer arithmetic.
// These are inlined for performance.
__device__ __forceinline__ int   f2i(float x){ return __float2int_rn(x); }
__device__ __forceinline__ float i2f(int   x){ return static_cast<float>(x); }
__device__ __forceinline__ bool valid_idx(int idx, int N){ return idx >= 0 && idx < N; }
__device__ __forceinline__ const float* node_ptr(const float* base, int D, int i){
    return base + (size_t)i * D;
}

// Calculates the next power of 2 for a given integer.
// This is useful for determining buffer sizes for algorithms like bitonic sort.
static inline int next_pow2_int(int x){
    int v = 1; while (v < x) v <<= 1; return (v <= 0) ? 1 : v;
}

// Gets the available shared memory per block. This is important for tuning
// kernel launch parameters and selecting the right algorithm.
static inline int get_shared_mem_budget_bytes(){
    int dev = -1, bytes = 0, got = 0;
    if (cudaGetDevice(&dev) == cudaSuccess){
        // Try to get the opt-in shared memory size first.
        if (cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) == cudaSuccess && bytes > 0){
            got = 1;
        } else if (cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlock, dev) == cudaSuccess && bytes > 0){
            got = 1;
        }
    }
    // If we couldn't get the shared memory size, use a conservative default.
    if (!got || bytes <= 0) bytes = 48 * 1024; // 48KB
    return bytes;
}

// -----------------------------------------------------------------------------
// Pass 1: Count children per parent and detect overflow.
// This is the first step in building the adjacency list in CSR format.
// Each block processes one tree.
// -----------------------------------------------------------------------------
__global__ void k_count_children_per_tree(
    const float* __restrict__ trees, // Input: (B, N, D) tensor of trees
    int B, int N, int D,             // Dimensions of the trees tensor
    int* __restrict__ child_counts   // Output: (B, N) tensor of child counts
){
    const int b = blockIdx.x; // Each block handles one tree in the batch.
    if (b >= B) return;

    const float* tree = trees + (size_t)b * N * D;
    int* cc          = child_counts + (size_t)b * N;

    // Initialize child counts to 0 for this tree.
    for (int i = threadIdx.x; i < N; i += blockDim.x) cc[i] = 0;
    __syncthreads();

    // Each thread iterates through the nodes of the tree and atomically increments
    // the child count of the parent node.
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        const float* nb = node_ptr(tree, D, i);
        int nt = f2i(nb[COL_NODE_TYPE]); if (nt == NODE_TYPE_UNUSED) continue;
        int p  = f2i(nb[COL_PARENT_IDX]);
        if (!valid_idx(p, N)) continue; // Parent index -1 is for root nodes.
        atomicAdd(&cc[p], 1);
    }
}

// Kernel to check if any parent has more children than max_children.
__global__ void k_check_overflow_per_tree(
    const int* __restrict__ child_counts, // Input: (B, N) tensor of child counts
    int B, int N, int max_children,       // Dimensions and max_children limit
    int* __restrict__ overflow_mask       // Output: (B) mask indicating overflow
){
    const int b = blockIdx.x;
    if (b >= B) return;
    const int* cc = child_counts + (size_t)b * N;

    // Use shared memory to communicate within the block.
    __shared__ int s_flag;
    if (threadIdx.x == 0) s_flag = 0;
    __syncthreads();

    // Each thread checks a subset of the nodes. If an overflow is found,
    // a flag in shared memory is set.
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        if (cc[i] > max_children) atomicExch(&s_flag, 1);
    }
    __syncthreads();

    // The first thread in the block writes the overflow flag to global memory.
    if (threadIdx.x == 0) overflow_mask[b] = s_flag;
}

// This function orchestrates the first pass of building the adjacency list.
// It calls the k_count_children_per_tree and k_check_overflow_per_tree kernels
// and then computes the CSR offsets array.
std::tuple<long, torch::Tensor, torch::Tensor> count_and_create_offsets_cuda(
    const torch::Tensor& trees,
    int max_children
){
    // Input validation.
    TORCH_CHECK(trees.is_cuda() && trees.dtype()==torch::kFloat32 && trees.dim()==3 && trees.is_contiguous(),
                "trees must be (B,N,D) float32 CUDA contiguous");
    TORCH_CHECK(max_children >= 0, "max_children must be >= 0");

    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);
    auto i32 = torch::TensorOptions().device(trees.device()).dtype(torch::kInt32);

    // Allocate output tensors.
    auto child_counts  = torch::empty({B, N}, i32);
    auto overflow_mask = torch::empty({B}, i32);

    c10::cuda::CUDAGuard guard(trees.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // Launch kernels.
    dim3 grid(B), block(256);
    k_count_children_per_tree<<<grid, block, 0, stream>>>(
        trees.data_ptr<float>(), B, N, D,
        child_counts.data_ptr<int>()
    );
    CUDA_CHECK_ERRORS();

    k_check_overflow_per_tree<<<grid, block, 0, stream>>>(
        child_counts.data_ptr<int>(), B, N, max_children, overflow_mask.data_ptr<int>()
    );
    CUDA_CHECK_ERRORS();

    // Compute CSR offsets by flattening the child_counts tensor and performing a cumulative sum.
    auto counts_flat = child_counts.reshape({B * N});
    auto offsets     = torch::empty({B * N + 1}, i32);
    offsets.index_put_({0}, 0);
    offsets.slice(0, 1, B * N + 1).copy_(torch::cumsum(counts_flat, 0, torch::kInt32));
    long total_children = offsets.index({-1}).item<long>();
    return {total_children, offsets, overflow_mask};
}

// -----------------------------------------------------------------------------
// Pass 2A: Fill child indices and optionally sort them using shared memory.
// This is the preferred method when there is enough shared memory.
// -----------------------------------------------------------------------------
__global__ void k_fill_sort_shared(
    const float* __restrict__ trees, // (B,N,D)
    int B, int N, int D,
    const int*  __restrict__ off,    // (B*N+1) CSR offsets
    int*        __restrict__ chi,    // (total_children) child indices
    int max_children,
    int parents_tile,                 // Number of parents to process in a tile
    int sort_buf,                     // Shared memory sort buffer size
    bool sort_children                // Whether to sort the children
){
    extern __shared__ int s_mem[];
    int* s_wcursor = s_mem;                  // Write cursors for each parent in the tile
    int* s_sort    = s_mem + parents_tile;   // Sort buffer
    const int b = blockIdx.x;
    if (b >= B) return;

    const float* tree = trees + (size_t)b * N * D;
    const int INT_INF = std::numeric_limits<int>::max();

    // Process parents in tiles to make use of shared memory.
    for (int p0 = 0; p0 < N; p0 += parents_tile){
        int tile = parents_tile;
        if (p0 + tile > N) tile = N - p0;
        if (tile <= 0) break;

        // 1) Initialize write cursors for this tile from global memory.
        for (int t = threadIdx.x; t < tile; t += blockDim.x){
            s_wcursor[t] = off[(size_t)b * N + (p0 + t)];
        }
        __syncthreads();

        // 2) Fill child indices. Each thread scans all nodes and if a node's
        //    parent is in the current tile, it adds the node to the child list.
        for (int i = threadIdx.x; i < N; i += blockDim.x){
            const float* nb = node_ptr(tree, D, i);
            int nt = f2i(nb[COL_NODE_TYPE]); if (nt == NODE_TYPE_UNUSED) continue;
            int p  = f2i(nb[COL_PARENT_IDX]);
            if (p < p0 || p >= p0 + tile) continue;
            if (!valid_idx(p, N)) continue;
            int lp  = p - p0;
            int pos = atomicAdd(&s_wcursor[lp], 1);
            int end = off[(size_t)b * N + p + 1];
            if (pos < end){
                chi[pos] = i;
            }
        }
        __syncthreads();

        // 3) Sort children for each parent in the tile.
        //    This uses a bitonic sort implemented in shared memory.
        if (sort_children && sort_buf >= 2){
            for (int lp = 0; lp < tile; ++lp){
                int p     = p0 + lp;
                int start = off[(size_t)b * N + p];
                int end   = off[(size_t)b * N + p + 1];
                int wrote = s_wcursor[lp]; if (wrote > end) wrote = end;
                int deg   = wrote - start;
                if (deg <= 1){ __syncthreads(); continue; }

                // Determine the size of the sort buffer needed.
                int T = 1; while (T < deg && T < sort_buf) T <<= 1;
                if (T < 2) T = 2; if (T > sort_buf) T = sort_buf;

                // Load children into shared memory, padding with infinity.
                for (int i = threadIdx.x; i < T; i += blockDim.x){
                    int v = (i < deg) ? chi[start + i] : INT_INF;
                    if (i < deg){
                        if (!valid_idx(v, N)) v = INT_INF; // Defensive check
                    }
                    s_sort[i] = v;
                }
                __syncthreads();

                // Bitonic sort network.
                for (int k = 2; k <= T; k <<= 1){
                    for (int j = k >> 1; j > 0; j >>= 1){
                        for (int idx = threadIdx.x; idx < T; idx += blockDim.x){
                            int ixj = idx ^ j;
                            if (ixj > idx){
                                bool up = ((idx & k) == 0);
                                int a = s_sort[idx], b = s_sort[ixj];
                                if ((up && a > b) || (!up && a < b)){
                                    s_sort[idx] = b; s_sort[ixj] = a;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }

                // Store sorted children back to global memory.
                for (int i = threadIdx.x; i < deg; i += blockDim.x){
                    chi[start + i] = s_sort[i];
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------------
// Pass 2B: Fill and optionally sort using global memory cursors.
// This is a fallback for when shared memory is limited.
// -----------------------------------------------------------------------------
__global__ void k_fill_sort_global(
    const float* __restrict__ trees, // (B,N,D)
    int B, int N, int D,
    const int*  __restrict__ off,    // (B*N+1)
    int*        __restrict__ chi,    // (total_children)
    int*        __restrict__ curs,   // (B*N) global cursors
    int max_children,
    int sort_buf,                     // shared sort capacity (#ints)
    bool sort_children
){
    extern __shared__ int s_sort[]; // size = sort_buf
    const int b = blockIdx.x;
    if (b >= B) return;

    const float* tree = trees + (size_t)b * N * D;
    const int INT_INF = std::numeric_limits<int>::max();

    // 1) Fill child indices using global atomic operations on cursors.
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        const float* nb = node_ptr(tree, D, i);
        int nt = f2i(nb[COL_NODE_TYPE]); if (nt == NODE_TYPE_UNUSED) continue;
        int p  = f2i(nb[COL_PARENT_IDX]);
        if (!valid_idx(p, N)) continue;
        int pos = atomicAdd(&curs[(size_t)b * N + p], 1);
        int end = off[(size_t)b * N + p + 1];
        if (pos < end){
            chi[pos] = i;
        }
    }
    __syncthreads();

    // 2) Sort children for each parent. This is similar to the shared memory
    //    version, but it iterates through all parents.
    if (sort_children && sort_buf >= 2){
        for (int p = 0; p < N; ++p){
            int start = off[(size_t)b * N + p];
            int end   = off[(size_t)b * N + p + 1];
            int deg   = end - start;
            if (deg <= 1){ __syncthreads(); continue; }

            int T = 1; while (T < deg && T < sort_buf) T <<= 1;
            if (T < 2) T = 2; if (T > sort_buf) T = sort_buf;

            // Load children into shared memory.
            for (int i = threadIdx.x; i < T; i += blockDim.x){
                int v = (i < deg) ? chi[start + i] : INT_INF;
                if (i < deg){
                    if (!valid_idx(v, N)) v = INT_INF;
                }
                s_sort[i] = v;
            }
            __syncthreads();

            // Bitonic sort.
            for (int k = 2; k <= T; k <<= 1){
                for (int j = k >> 1; j > 0; j >>= 1){
                    for (int idx = threadIdx.x; idx < T; idx += blockDim.x){
                        int ixj = idx ^ j;
                        if (ixj > idx){
                            bool up = ((idx & k) == 0);
                            int a = s_sort[idx], b = s_sort[ixj];
                            if ((up && a > b) || (!up && a < b)){
                                s_sort[idx] = b; s_sort[ixj] = a;
                            }
                        }
                    }
                    __syncthreads();
                }
            }

            // Store sorted children back to global memory.
            for (int i = threadIdx.x; i < deg; i += blockDim.x){
                chi[start + i] = s_sort[i];
            }
            __syncthreads();
        }
    }
}

// This function orchestrates the second pass of building the adjacency list.
// It decides whether to use the shared memory or global memory version of the
// fill and sort kernel based on available shared memory and problem size.
void fill_child_indices_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& offsets_flat,
    torch::Tensor&       child_indices,
    int                  max_children,
    bool                 sort_children
){
    // Input validation
    TORCH_CHECK(trees.is_cuda() && trees.dtype()==torch::kFloat32 && trees.dim()==3 && trees.is_contiguous(),
                "trees must be (B,N,D) float32 CUDA contiguous");
    TORCH_CHECK(offsets_flat.is_cuda() && offsets_flat.dtype()==torch::kInt32 && offsets_flat.dim()==1,
                "offsets_flat must be (B*N+1) int32 CUDA");
    TORCH_CHECK(child_indices.is_cuda() && child_indices.dtype()==torch::kInt32 && child_indices.dim()==1,
                "child_indices must be (total_children) int32 CUDA");
    TORCH_CHECK(offsets_flat.device() == trees.device(), "offsets must be on same device as trees");
    TORCH_CHECK(child_indices.device() == trees.device(), "child_indices must be on same device as trees");
    TORCH_CHECK(max_children >= 0, "max_children must be >= 0");

    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);
    TORCH_CHECK(offsets_flat.size(0) == B * N + 1, "offsets size mismatch");

    if (child_indices.numel() == 0) return;

    TORCH_CHECK(child_indices.numel() == offsets_flat.index({-1}).item<int>(),
                "child_indices length must match offsets[-1]");

    // Plan shared memory usage.
    const int shm_budget_bytes = get_shared_mem_budget_bytes();
    int max_ints = shm_budget_bytes / (int)sizeof(int);
    if (max_ints < 2) max_ints = 2;

    int sort_buf = next_pow2_int(std::max(1, max_children));
    int sort_buf_max = max_ints - 1;
    if (sort_buf > sort_buf_max){
        // Not enough shared memory for sorting, so disable it.
        sort_children = false;
        sort_buf = 1;
    }
    int parents_tile = max_ints - sort_buf;
    if (parents_tile < 1) parents_tile = 1;
    if (parents_tile > N) parents_tile = N;

    // Heuristic to decide whether to use the shared memory or global memory kernel.
    bool use_global = (parents_tile < (N >> 3)); // threshold = N/8

    c10::cuda::CUDAGuard guard(trees.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(B), block(256);

    if (!use_global){
        // Use the shared memory version.
        const size_t shm_bytes = (size_t)(parents_tile + sort_buf) * sizeof(int);
        k_fill_sort_shared<<<grid, block, shm_bytes, stream>>>(
            trees.data_ptr<float>(), B, N, D,
            offsets_flat.data_ptr<int>(),
            child_indices.data_ptr<int>(),
            max_children,
            parents_tile,
            sort_buf,
            sort_children
        );
        CUDA_CHECK_ERRORS();
    } else {
        // Use the global memory version.
        auto curs = offsets_flat.index({torch::indexing::Slice(0, B * N)}).clone();
        const size_t shm_bytes = (size_t)(sort_children ? sort_buf : 1) * sizeof(int);
        k_fill_sort_global<<<grid, block, shm_bytes, stream>>>(
            trees.data_ptr<float>(), B, N, D,
            offsets_flat.data_ptr<int>(),
            child_indices.data_ptr<int>(),
            curs.data_ptr<int>(),
            max_children,
            sort_buf,
            sort_children
        );
        CUDA_CHECK_ERRORS();
    }
}

// -----------------------------------------------------------------------------
// Validator: Checks for structural violations in the trees.
// -----------------------------------------------------------------------------
enum ViolBits {
  V_MIXED_CHILDREN   = 1<<0, // Parent has both action and decision children
  V_LEAF_NOT_ACTION  = 1<<1, // Leaf node is not an action node
  V_ACTION_HAS_CHILD = 1<<2, // Action node has children
  V_SINGLE_ACTION_BR = 1<<3, // Action branch has more than one child
  V_DEPTH_MISMATCH   = 1<<4, // Child depth is not parent depth + 1
  V_CHILD_OVERFLOW   = 1<<5, // Too many children for a parent
  V_BAD_PARENT       = 1<<6, // Invalid parent index
  V_ROOT_BROKEN      = 1<<7, // Root node is not of type ROOT_BRANCH
  V_ROOT_LEAF        = 1<<8  // Root node is a leaf
};

// Kernel to validate the adjacency information.
__global__ void k_validate_adjacency(
  const float* __restrict__ trees, int B, int N, int D,
  const int*   __restrict__ off,
  const int*   __restrict__ chi,
  int*         __restrict__ out_mask
){
  const int b = blockIdx.x;
  if (b >= B) return;

  const float* tree = trees + (size_t)b * N * D;
  int local_mask = 0;

  // Each thread checks a subset of the parents.
  for (int p = threadIdx.x; p < N; p += blockDim.x){
    int s = off[(size_t)b * N + p];
    int e = off[(size_t)b * N + p + 1];
    int deg = e - s;

    int act=0, dec=0;
    for (int k = s; k < e; ++k){
      int c = chi[k];
      if (!valid_idx(c, N)){ local_mask |= V_BAD_PARENT; continue; }
      int ct = f2i(node_ptr(tree, D, c)[COL_NODE_TYPE]);
      if (ct == NODE_TYPE_ACTION)        ++act;
      else if (ct == NODE_TYPE_DECISION) ++dec;

      int pd = f2i(node_ptr(tree, D, p)[COL_DEPTH]);
      int cd = f2i(node_ptr(tree, D, c)[COL_DEPTH]);
      if (cd != pd + 1) local_mask |= V_DEPTH_MISMATCH;
    }

    // Check for violations.
    if (act>0 && dec>0) local_mask |= V_MIXED_CHILDREN;
    if (act>0 && deg!=1) local_mask |= V_SINGLE_ACTION_BR;

    int pt = f2i(node_ptr(tree, D, p)[COL_NODE_TYPE]);
    if (pt == NODE_TYPE_ACTION && deg>0) local_mask |= V_ACTION_HAS_CHILD;

    int parent = f2i(node_ptr(tree, D, p)[COL_PARENT_IDX]);
    if (deg==0 && pt != NODE_TYPE_ACTION){
        if (parent >= 0) local_mask |= V_LEAF_NOT_ACTION;
    }
    if (parent == -1){
        if (pt != NODE_TYPE_ROOT_BRANCH) local_mask |= V_ROOT_BROKEN;
        int depth = f2i(node_ptr(tree, D, p)[COL_DEPTH]);
        if (depth != 0) local_mask |= V_DEPTH_MISMATCH;
        if (deg == 0)   local_mask |= V_ROOT_LEAF;
    } else if (parent < -1 || parent >= N){
        local_mask |= V_BAD_PARENT;
    }
  }

  // Reduce the local violation masks into a single mask for the tree.
  __shared__ int s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  atomicOr(&s, local_mask);
  __syncthreads();
  if (threadIdx.x == 0) out_mask[b] = s;
}

// This function orchestrates the validation of the adjacency information.
void validate_adjacency_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& offsets_flat,
    const torch::Tensor& child_indices,
    torch::Tensor        out_violation_mask
){
    // Input validation
    TORCH_CHECK(trees.is_cuda() && trees.dtype()==torch::kFloat32 && trees.dim()==3 && trees.is_contiguous(),
                "trees must be (B,N,D) float32 CUDA contiguous");
    TORCH_CHECK(offsets_flat.is_cuda() && offsets_flat.dtype()==torch::kInt32 && offsets_flat.dim()==1,
                "offsets_flat must be (B*N+1) int32 CUDA");
    TORCH_CHECK(child_indices.is_cuda() && child_indices.dtype()==torch::kInt32 && child_indices.dim()==1,
                "child_indices must be (total_children) int32 CUDA");
    TORCH_CHECK(out_violation_mask.is_cuda() && out_violation_mask.dtype()==torch::kInt32 && out_violation_mask.dim()==1,
                "out_violation_mask must be (B,) int32 CUDA");

    TORCH_CHECK(offsets_flat.device() == trees.device(), "offsets must be on same device as trees");
    TORCH_CHECK(child_indices.device() == trees.device(), "child_indices must be on same device as trees");
    TORCH_CHECK(out_violation_mask.device() == trees.device(), "mask must be on same device as trees");

    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);
    TORCH_CHECK(offsets_flat.size(0) == B * N + 1, "offsets size mismatch");
    TORCH_CHECK(child_indices.numel() == offsets_flat.index({-1}).item<int>(),
                "child_indices length must match offsets[-1]");
    TORCH_CHECK(out_violation_mask.size(0) == B, "mask size mismatch");

    c10::cuda::CUDAGuard guard(trees.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // Launch the validation kernel.
    dim3 grid(B), block(256);
    k_validate_adjacency<<<grid, block, 0, stream>>>(
        trees.data_ptr<float>(), B, N, D,
        offsets_flat.data_ptr<int>(),
        child_indices.data_ptr<int>(),
        out_violation_mask.data_ptr<int>()
    );
    CUDA_CHECK_ERRORS();
}

// -----------------------------------------------------------------------------
// Build adjacency dictionaries for Python synchronization.
// This kernel is designed to be called from Python to get the adjacency information
// in a Python-friendly format.
// -----------------------------------------------------------------------------
__global__ void k_build_adjacency_dicts(
    const float* __restrict__ trees,     // (B,N,D)
    const int*   __restrict__ next_idxs, // (B) next_idx per tree
    int B, int N, int D,
    int* __restrict__ offsets,           // (B*N+1) output offsets 
    int* __restrict__ children           // (total_children) output children indices
){
    const int b = blockIdx.x;
    if (b >= B) return;

    const float* tree = trees + (size_t)b * N * D;
    const int next_idx = next_idxs[b];
    int* tree_offsets = offsets + (size_t)b * N;
    
    // First pass: count children per parent
    for (int p = threadIdx.x; p < N; p += blockDim.x) {
        tree_offsets[p] = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < next_idx; i += blockDim.x){
        const float* nb = node_ptr(tree, D, i);
        int nt = f2i(nb[COL_NODE_TYPE]);
        if (nt == NODE_TYPE_UNUSED) continue;
        
        int parent_idx = f2i(nb[COL_PARENT_IDX]);
        if (parent_idx != -1 && valid_idx(parent_idx, N)) {
            atomicAdd(&tree_offsets[parent_idx], 1);
        }
    }
    __syncthreads();

    // Convert counts to offsets using prefix sum  
    if (threadIdx.x == 0) {
        // Get the starting offset for this tree from the global offsets array
        int global_start = (b == 0) ? 0 : offsets[b * N];
        int current_offset = global_start;
        
        for (int p = 0; p < N; p++) {
            int count = tree_offsets[p];
            tree_offsets[p] = current_offset;  // store start position for this parent
            current_offset += count;
        }
    }
    __syncthreads();

    // Second pass: fill children indices
    extern __shared__ int s_write_cursors[];
    for (int p = threadIdx.x; p < N; p += blockDim.x) {
        s_write_cursors[p] = tree_offsets[p];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < next_idx; i += blockDim.x){
        const float* nb = node_ptr(tree, D, i);
        int nt = f2i(nb[COL_NODE_TYPE]);
        if (nt == NODE_TYPE_UNUSED) continue;
        
        int parent_idx = f2i(nb[COL_PARENT_IDX]);
        if (parent_idx != -1 && valid_idx(parent_idx, N)) {
            int pos = atomicAdd(&s_write_cursors[parent_idx], 1);
            children[pos] = i;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> build_adjacency_dicts_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& next_idxs
){
    TORCH_CHECK(trees.is_cuda() && trees.dtype()==torch::kFloat32 && trees.dim()==3 && trees.is_contiguous(),
                "trees must be (B,N,D) float32 CUDA contiguous");
    TORCH_CHECK(next_idxs.is_cuda() && next_idxs.dtype()==torch::kInt32 && next_idxs.dim()==1,
                "next_idxs must be (B,) int32 CUDA");

    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);
    TORCH_CHECK(next_idxs.size(0) == B, "next_idxs size must match batch size");

    auto i32 = torch::TensorOptions().device(trees.device()).dtype(torch::kInt32);
    
    c10::cuda::CUDAGuard guard(trees.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Use existing CSR building functions for consistency
    long total_children;
    torch::Tensor offsets_flat, overflow_mask;
    std::tie(total_children, offsets_flat, overflow_mask) = count_and_create_offsets_cuda(trees, 1000); // high max_children limit
    
    // Allocate children indices
    auto children = torch::empty({(long)total_children}, i32);
    
    if (total_children > 0) {
        // Use existing fill function
        fill_child_indices_cuda(trees, offsets_flat, children, 1000, false); // no sorting needed for dict building
    }
    
    return {offsets_flat, children};
}

// -----------------------------------------------------------------------------
// Memory-optimized adjacency dict builder using compact format
// -----------------------------------------------------------------------------
__global__ void k_build_adjacency_compact(
    const float* __restrict__ trees,     // (B,N,D)
    const int*   __restrict__ next_idxs, // (B) next_idx per tree
    int B, int N, int D,
    int* __restrict__ parent_counts,     // (B*N) output: child count per parent
    int* __restrict__ parent_starts,     // (B*N) output: start index in children array
    int* __restrict__ children_flat,     // (total_children) output: flattened children
    int* __restrict__ tree_offsets       // (B+1) cumulative children count per tree
){
    const int b = blockIdx.x;
    if (b >= B) return;

    const float* tree = trees + (size_t)b * N * D;
    const int next_idx = next_idxs[b];
    int* counts = parent_counts + (size_t)b * N;
    
    // Step 1: Count children per parent using minimal shared memory
    extern __shared__ int s_temp[];
    const int threads = blockDim.x;
    const int warps = (threads + 31) / 32;
    
    // Initialize counts to zero
    for (int p = threadIdx.x; p < N; p += blockDim.x) {
        counts[p] = 0;
    }
    __syncthreads();

    // Count children with warp-level optimization
    for (int i = threadIdx.x; i < next_idx; i += blockDim.x) {
        const float* nb = node_ptr(tree, D, i);
        int nt = f2i(nb[COL_NODE_TYPE]);
        if (nt == NODE_TYPE_UNUSED) continue;
        
        int parent_idx = f2i(nb[COL_PARENT_IDX]);
        if (parent_idx != -1 && valid_idx(parent_idx, N)) {
            atomicAdd(&counts[parent_idx], 1);
        }
    }
    __syncthreads();

    // Step 2: Compute prefix sum for start positions (single thread per block)
    if (threadIdx.x == 0) {
        int tree_base = (b == 0) ? 0 : tree_offsets[b];
        int cumsum = tree_base;
        
        for (int p = 0; p < N; p++) {
            parent_starts[(size_t)b * N + p] = cumsum;
            cumsum += counts[p];
        }
    }
    __syncthreads();

    // Step 3: Fill children array with coalesced memory access
    int* starts = parent_starts + (size_t)b * N;
    for (int p = threadIdx.x; p < N; p += blockDim.x) {
        s_temp[p] = starts[p]; // Copy start positions to shared memory
    }
    __syncthreads();

    for (int i = threadIdx.x; i < next_idx; i += blockDim.x) {
        const float* nb = node_ptr(tree, D, i);
        int nt = f2i(nb[COL_NODE_TYPE]);
        if (nt == NODE_TYPE_UNUSED) continue;
        
        int parent_idx = f2i(nb[COL_PARENT_IDX]);
        if (parent_idx != -1 && valid_idx(parent_idx, N)) {
            int pos = atomicAdd(&s_temp[parent_idx], 1);
            children_flat[pos] = i;
        }
    }
}

py::list build_adjacency_dicts_optimized_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& next_idxs
){
    TORCH_CHECK(trees.is_cuda() && trees.dtype()==torch::kFloat32 && trees.dim()==3 && trees.is_contiguous(),
                "trees must be (B,N,D) float32 CUDA contiguous");
    TORCH_CHECK(next_idxs.is_cuda() && next_idxs.dtype()==torch::kInt32 && next_idxs.dim()==1,
                "next_idxs must be (B,) int32 CUDA");

    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);
    TORCH_CHECK(next_idxs.size(0) == B, "next_idxs size must match batch size");

    auto i32 = torch::TensorOptions().device(trees.device()).dtype(torch::kInt32);
    
    c10::cuda::CUDAGuard guard(trees.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Use existing efficient functions for memory optimization
    long total_children;
    torch::Tensor offsets_flat, overflow_mask;
    std::tie(total_children, offsets_flat, overflow_mask) = count_and_create_offsets_cuda(trees, 1000);
    
    auto children_flat = torch::empty({(long)total_children}, i32);
    
    if (total_children > 0) {
        fill_child_indices_cuda(trees, offsets_flat, children_flat, 1000, false);
    }
    
    // Convert tensors to CPU for processing
    auto offsets_cpu = offsets_flat.cpu();
    auto children_cpu = children_flat.cpu();
    
    // Build Python list of dictionaries directly in C++ - NO PYTHON LOOPS!
    py::list adjacency_dicts_list;
    
    for (int tree_idx = 0; tree_idx < B; tree_idx++) {
        py::dict tree_dict;
        
        // Extract offsets for this tree 
        int tree_start = tree_idx * N;
        
        for (int parent_idx = 0; parent_idx < N; parent_idx++) {
            int offset_idx = tree_start + parent_idx;
            if (offset_idx + 1 < offsets_cpu.numel()) {
                int start_pos = offsets_cpu[offset_idx].item<int>();
                int end_pos = offsets_cpu[offset_idx + 1].item<int>();
                
                if (end_pos > start_pos) {
                    // Extract children for this parent
                    py::list children_list;
                    for (int i = start_pos; i < end_pos; i++) {
                        children_list.append(children_cpu[i].item<int>());
                    }
                    if (py::len(children_list) > 0) {
                        tree_dict[py::int_(parent_idx)] = children_list;
                    }
                }
            }
        }
        
        adjacency_dicts_list.append(tree_dict);
    }
    
    return adjacency_dicts_list;
}
