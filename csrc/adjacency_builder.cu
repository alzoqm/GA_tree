// csrc/adjacency_builder.cu
#include "adjacency_builder.cuh"
#include "constants.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>

#define CUDA_CHECK_ERRORS() \
  do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  }} while (0)

__device__ __forceinline__ int   f2i(float x){ return __float2int_rn(x); }
__device__ __forceinline__ float i2f(int   x){ return static_cast<float>(x); }
__device__ __forceinline__ bool valid_idx(int idx, int N){ return idx >= 0 && idx < N; }
__device__ __forceinline__ const float* node_ptr(const float* base, int D, int i){
    return base + (size_t)i * D;
}

static inline int next_pow2_int(int x){
    int v = 1; while (v < x) v <<= 1; return (v <= 0) ? 1 : v;
}

static inline int get_shared_mem_budget_bytes(){
    int dev = -1, bytes = 0, got = 0;
    if (cudaGetDevice(&dev) == cudaSuccess){
        if (cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) == cudaSuccess && bytes > 0){
            got = 1;
        } else if (cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlock, dev) == cudaSuccess && bytes > 0){
            got = 1;
        }
    }
    if (!got || bytes <= 0) bytes = 48 * 1024; // conservative default
    return bytes;
}

// -----------------------------------------------------------------------------
// Pass 1: Count children per parent (block = tree) and detect overflow per tree
// -----------------------------------------------------------------------------
__global__ void k_count_children_per_tree(
    const float* __restrict__ trees, // (B,N,D)
    int B, int N, int D,
    int* __restrict__ child_counts   // (B,N)
){
    const int b = blockIdx.x;
    if (b >= B) return;

    const float* tree = trees + (size_t)b * N * D;
    int* cc          = child_counts + (size_t)b * N;

    for (int i = threadIdx.x; i < N; i += blockDim.x) cc[i] = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x){
        const float* nb = node_ptr(tree, D, i);
        int nt = f2i(nb[COL_NODE_TYPE]); if (nt == NODE_TYPE_UNUSED) continue;
        int p  = f2i(nb[COL_PARENT_IDX]);
        if (!valid_idx(p, N)) continue; // -1 is root/ROOT_BRANCH
        atomicAdd(&cc[p], 1);
    }
}

__global__ void k_check_overflow_per_tree(
    const int* __restrict__ child_counts, // (B,N)
    int B, int N, int max_children,
    int* __restrict__ overflow_mask       // (B) int32
){
    const int b = blockIdx.x;
    if (b >= B) return;
    const int* cc = child_counts + (size_t)b * N;

    __shared__ int s_flag;
    if (threadIdx.x == 0) s_flag = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x){
        if (cc[i] > max_children) atomicExch(&s_flag, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) overflow_mask[b] = s_flag;
}

std::tuple<long, torch::Tensor, torch::Tensor> count_and_create_offsets_cuda(
    const torch::Tensor& trees,
    int max_children
){
    TORCH_CHECK(trees.is_cuda() && trees.dtype()==torch::kFloat32 && trees.dim()==3 && trees.is_contiguous(),
                "trees must be (B,N,D) float32 CUDA contiguous");
    TORCH_CHECK(max_children >= 0, "max_children must be >= 0");

    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);
    auto i32 = torch::TensorOptions().device(trees.device()).dtype(torch::kInt32);

    // (B,N) counts
    auto child_counts  = torch::empty({B, N}, i32);
    // (B) overflow mask
    auto overflow_mask = torch::empty({B}, i32);

    c10::cuda::CUDAGuard guard(trees.device());
    auto stream = at::cuda::getCurrentCUDAStream();

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

    // CSR offsets: flatten counts (B*N) -> cumsum
    auto counts_flat = child_counts.reshape({B * N});
    auto offsets     = torch::empty({B * N + 1}, i32);
    offsets.index_put_({0}, 0);
    offsets.slice(0, 1, B * N + 1).copy_(torch::cumsum(counts_flat, 0, torch::kInt32));
    long total_children = offsets.index({-1}).item<long>();
    return {total_children, offsets, overflow_mask};
}

// -----------------------------------------------------------------------------
// Pass 2A: Fill + (optional) sort using shared-cursor tiling (block = tree)
// -----------------------------------------------------------------------------
__global__ void k_fill_sort_shared(
    const float* __restrict__ trees, // (B,N,D)
    int B, int N, int D,
    const int*  __restrict__ off,    // (B*N+1)
    int*        __restrict__ chi,    // (total_children)
    int max_children,
    int parents_tile,                 // parents in shared
    int sort_buf,                     // shared sort capacity (#ints)
    bool sort_children
){
    extern __shared__ int s_mem[];
    int* s_wcursor = s_mem;                  // size = parents_tile
    int* s_sort    = s_mem + parents_tile;   // size = sort_buf
    const int b = blockIdx.x;
    if (b >= B) return;

    const float* tree = trees + (size_t)b * N * D;
    const int INT_INF = std::numeric_limits<int>::max();

    for (int p0 = 0; p0 < N; p0 += parents_tile){
        int tile = parents_tile;
        if (p0 + tile > N) tile = N - p0;
        if (tile <= 0) break;

        // 1) init cursors for this tile
        for (int t = threadIdx.x; t < tile; t += blockDim.x){
            s_wcursor[t] = off[(size_t)b * N + (p0 + t)];
        }
        __syncthreads();

        // 2) fill (scan nodes once)
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

        // 3) sort per parent (sequential by parent; all threads cooperate)
        if (sort_children && sort_buf >= 2){
            for (int lp = 0; lp < tile; ++lp){
                int p     = p0 + lp;
                int start = off[(size_t)b * N + p];
                int end   = off[(size_t)b * N + p + 1];
                int wrote = s_wcursor[lp]; if (wrote > end) wrote = end;
                int deg   = wrote - start;
                if (deg <= 1){ __syncthreads(); continue; }

                int T = 1; while (T < deg && T < sort_buf) T <<= 1;
                if (T < 2) T = 2; if (T > sort_buf) T = sort_buf;

                // load + filter invalid + pad
                for (int i = threadIdx.x; i < T; i += blockDim.x){
                    int v = (i < deg) ? chi[start + i] : INT_INF;
                    if (i < deg){
                        if (!valid_idx(v, N)) v = INT_INF; // defensive
                    }
                    s_sort[i] = v;
                }
                __syncthreads();

                // bitonic
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

                // store
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
// Pass 2B: Fill + (optional) sort using GLOBAL cursors (fallback; block = tree)
// -----------------------------------------------------------------------------
__global__ void k_fill_sort_global(
    const float* __restrict__ trees, // (B,N,D)
    int B, int N, int D,
    const int*  __restrict__ off,    // (B*N+1)
    int*        __restrict__ chi,    // (total_children)
    int*        __restrict__ curs,   // (B*N) global cursors (initialized as off[b*N + p])
    int max_children,
    int sort_buf,                     // shared sort capacity (#ints)
    bool sort_children
){
    extern __shared__ int s_sort[]; // size = sort_buf
    const int b = blockIdx.x;
    if (b >= B) return;

    const float* tree = trees + (size_t)b * N * D;
    const int INT_INF = std::numeric_limits<int>::max();

    // 1) fill (single pass)
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

    // 2) sort per parent (sequential; all threads cooperate)
    if (sort_children && sort_buf >= 2){
        for (int p = 0; p < N; ++p){
            int start = off[(size_t)b * N + p];
            int end   = off[(size_t)b * N + p + 1];
            int deg   = end - start; // global cursor finalized to end; we use [start,end)
            if (deg <= 1){ __syncthreads(); continue; }

            int T = 1; while (T < deg && T < sort_buf) T <<= 1;
            if (T < 2) T = 2; if (T > sort_buf) T = sort_buf;

            // load + filter invalid + pad
            for (int i = threadIdx.x; i < T; i += blockDim.x){
                int v = (i < deg) ? chi[start + i] : INT_INF;
                if (i < deg){
                    if (!valid_idx(v, N)) v = INT_INF;
                }
                s_sort[i] = v;
            }
            __syncthreads();

            // bitonic
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

            // store
            for (int i = threadIdx.x; i < deg; i += blockDim.x){
                chi[start + i] = s_sort[i];
            }
            __syncthreads();
        }
    }
}

void fill_child_indices_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& offsets_flat,
    torch::Tensor&       child_indices,
    int                  max_children,
    bool                 sort_children
){
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

    // Length sanity: chi length must equal offsets[-1]
    TORCH_CHECK(child_indices.numel() == offsets_flat.index({-1}).item<int>(),
                "child_indices length must match offsets[-1]");

    // Shared memory planning
    const int shm_budget_bytes = get_shared_mem_budget_bytes();
    int max_ints = shm_budget_bytes / (int)sizeof(int);
    if (max_ints < 2) max_ints = 2;

    int sort_buf = next_pow2_int(std::max(1, max_children));
    int sort_buf_max = max_ints - 1; // leave at least 1 int for parents_tile
    if (sort_buf > sort_buf_max){
        // Sorting cannot fit; disable deterministically (safe)
        sort_children = false;
        sort_buf = 1;
    }
    int parents_tile = max_ints - sort_buf;
    if (parents_tile < 1) parents_tile = 1;
    if (parents_tile > N) parents_tile = N;

    // Heuristic: if tiling is too small, switch to global-cursor mode
    bool use_global = (parents_tile < (N >> 3)); // threshold = N/8

    c10::cuda::CUDAGuard guard(trees.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(B), block(256);

    if (!use_global){
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
        // Allocate global cursors as a clone of offsets (B*N+1) -> we use only first B*N
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
// Validator
// -----------------------------------------------------------------------------
enum ViolBits {
  V_MIXED_CHILDREN   = 1<<0,
  V_LEAF_NOT_ACTION  = 1<<1,
  V_ACTION_HAS_CHILD = 1<<2,
  V_SINGLE_ACTION_BR = 1<<3,
  V_DEPTH_MISMATCH   = 1<<4,
  V_CHILD_OVERFLOW   = 1<<5,
  V_BAD_PARENT       = 1<<6,
  V_ROOT_BROKEN      = 1<<7,
  V_ROOT_LEAF        = 1<<8
};

__global__ void k_validate_adjacency(
  const float* __restrict__ trees, int B, int N, int D,
  const int*   __restrict__ off,
  const int*   __restrict__ chi,
  int max_children, int max_depth,
  int*         __restrict__ out_mask
){
  const int b = blockIdx.x;
  if (b >= B) return;

  const float* tree = trees + (size_t)b * N * D;
  int local_mask = 0;

  for (int p = threadIdx.x; p < N; p += blockDim.x){
    int s = off[(size_t)b * N + p];
    int e = off[(size_t)b * N + p + 1];
    int deg = e - s;
    if (deg > max_children) local_mask |= V_CHILD_OVERFLOW;

    int act=0, dec=0;
    for (int k = s; k < e; ++k){
      int c = chi[k];
      if (!valid_idx(c, N)){ local_mask |= V_BAD_PARENT; continue; }
      int ct = f2i(node_ptr(tree, D, c)[COL_NODE_TYPE]);
      if (ct == NODE_TYPE_ACTION)        ++act;
      else if (ct == NODE_TYPE_DECISION) ++dec;

      int pd = f2i(node_ptr(tree, D, p)[COL_DEPTH]);
      int cd = f2i(node_ptr(tree, D, c)[COL_DEPTH]);
      if (cd != pd + 1 || cd >= max_depth) local_mask |= V_DEPTH_MISMATCH;
    }

    // Mixed children
    if (act>0 && dec>0) local_mask |= V_MIXED_CHILDREN;
    // Single-action rule (FIXED): if any action children exist, degree must be exactly 1
    if (act>0 && deg!=1) local_mask |= V_SINGLE_ACTION_BR;

    int pt = f2i(node_ptr(tree, D, p)[COL_NODE_TYPE]);
    // ACTION node must have zero children
    if (pt == NODE_TYPE_ACTION && deg>0) local_mask |= V_ACTION_HAS_CHILD;

    // Leaf must be ACTION (exclude true roots)
    int parent = f2i(node_ptr(tree, D, p)[COL_PARENT_IDX]);
    if (deg==0 && pt != NODE_TYPE_ACTION){
        if (parent >= 0) local_mask |= V_LEAF_NOT_ACTION;
    }
    // Root sanity
    if (parent == -1){
        if (pt != NODE_TYPE_ROOT_BRANCH) local_mask |= V_ROOT_BROKEN;
        int depth = f2i(node_ptr(tree, D, p)[COL_DEPTH]);
        if (depth != 0) local_mask |= V_DEPTH_MISMATCH;
        if (deg == 0)   local_mask |= V_ROOT_LEAF; // optional diagnostic
    } else if (parent < -1 || parent >= N){
        local_mask |= V_BAD_PARENT;
    }
    // Depth range
    int depth = f2i(node_ptr(tree, D, p)[COL_DEPTH]);
    if (depth >= max_depth) local_mask |= V_DEPTH_MISMATCH;
  }

  __shared__ int s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  atomicOr(&s, local_mask);
  __syncthreads();
  if (threadIdx.x == 0) out_mask[b] = s;
}

void validate_adjacency_cuda(
    const torch::Tensor& trees,
    const torch::Tensor& offsets_flat,
    const torch::Tensor& child_indices,
    int                  max_children,
    int                  max_depth,
    torch::Tensor        out_violation_mask
){
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

    dim3 grid(B), block(256);
    k_validate_adjacency<<<grid, block, 0, stream>>>(
        trees.data_ptr<float>(), B, N, D,
        offsets_flat.data_ptr<int>(),
        child_indices.data_ptr<int>(),
        max_children, max_depth,
        out_violation_mask.data_ptr<int>()
    );
    CUDA_CHECK_ERRORS();
}