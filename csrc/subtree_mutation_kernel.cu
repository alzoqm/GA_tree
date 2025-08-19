// csrc/subtree_mutation_kernel.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "constants.h"
#include "subtree_mutation_kernel.cuh"

#define CUDA_CHECK_ERRORS() \
  do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  }} while (0)

// ----- Safe float<->int conversion (negative values preserved) -----
__device__ __forceinline__ int   f2i(float x){ return __float2int_rn(x); }
__device__ __forceinline__ float i2f(int   x){ return static_cast<float>(x); }

__device__ __forceinline__ bool valid_node_index(int idx, int N){ return idx >= 0 && idx < N; }

// Simple Linear Congruential Generator for per-thread randomization
struct Lcg {
    unsigned int s;
    __device__ explicit Lcg(unsigned int seed): s(seed) {}
    __device__ __forceinline__ unsigned int nextu(){ s = 1664525u * s + 1013904223u; return s; }
    __device__ __forceinline__ float nextf(){ return float(nextu() & 0x00FFFFFF) / float(0x01000000); }
};

__device__ __forceinline__ float* node_ptr(float* base, int D, int i){ return base + (size_t)i * D; }
__device__ __forceinline__ const float* node_ptr(const float* base, int D, int i){ return base + (size_t)i * D; }

// Climb to root to fetch ROOT_BRANCH type (COL_PARAM_1)
__device__ int get_root_branch_type(const float* tree, int D, int N, int start_idx){
    int cur = start_idx, safety = 0;
    while(valid_node_index(cur, N) && safety < N){
        const float* nb = node_ptr(tree, D, cur);
        int p = f2i(nb[COL_PARENT_IDX]);
        if (p < 0) return f2i(nb[COL_PARAM_1]);
        cur = p; 
        ++safety;
    }
    return -1; // fallback
}

// One block per tree - comprehensive invariant-preserving AddSubtree kernel
__global__ void k_add_subtrees_batch(
    float* __restrict__ trees, int B, int N, int D,
    const int* __restrict__ num_to_grow,
    int max_children, int max_depth, int max_nodes, int max_new_nodes,
    int* __restrict__ out_dec_nodes, int* __restrict__ out_act_nodes, float* __restrict__ out_act_rtype,
    int* __restrict__ bfs_queue, int* __restrict__ scratch, // (B, 2*N)
    int* __restrict__ child_cnt, int* __restrict__ act_cnt, int* __restrict__ dec_cnt,
    int* __restrict__ cand_idx, float* __restrict__ cand_w
){
    const int b = blockIdx.x;
    if (b >= B) return;
    
    float* tree = trees + (size_t)b * N * D;

    // Per-tree slices
    int* open_q   = bfs_queue + (size_t)b * (2 * N);
    int* scr      = scratch   + (size_t)b * (2 * N);
    int* ch       = child_cnt + (size_t)b * N;
    int* ac       = act_cnt   + (size_t)b * N;
    int* dc       = dec_cnt   + (size_t)b * N;
    int* cidx     = cand_idx  + (size_t)b * N;
    float* cw     = cand_w    + (size_t)b * N;
    int* out_dec  = out_dec_nodes + (size_t)b * max_new_nodes;
    int* out_act  = out_act_nodes + (size_t)b * max_new_nodes;
    float* out_rt = out_act_rtype + (size_t)b * max_new_nodes;

    // scratch buffer layout:
    // scr[0] -> candidate_count
    // scr[1] -> freelist_count (raw)
    // scr[2] -> freelist_cursor
    // scr[3..] -> freelist entries (UNUSED node indices), capacity = (2*N - 3)

    // 0) Zero buffers in parallel
    for (int i = threadIdx.x; i < N; i += blockDim.x){ 
        ch[i] = 0; ac[i] = 0; dc[i] = 0; cidx[i] = -1; cw[i] = 0.0f; 
    }
    for (int i = threadIdx.x; i < 2 * N; i += blockDim.x){ 
        open_q[i] = -1; scr[i] = -1; 
    }
    for (int i = threadIdx.x; i < max_new_nodes; i += blockDim.x){ 
        out_dec[i] = -1; out_act[i] = -1; out_rt[i] = -1.0f; 
    }
    __syncthreads();

    // 1) Build counts in parallel
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        const float* nb = node_ptr(tree, D, i);
        int t = f2i(nb[COL_NODE_TYPE]);
        if (t == NODE_TYPE_UNUSED) continue;
        int p = f2i(nb[COL_PARENT_IDX]);
        if (valid_node_index(p, N)){
            atomicAdd(&ch[p], 1);
            if (t == NODE_TYPE_ACTION)   atomicAdd(&ac[p], 1);
            if (t == NODE_TYPE_DECISION) atomicAdd(&dc[p], 1);
        }
    }
    __syncthreads();

    // 2) Build candidate list (parents: ROOT_BRANCH or DECISION) in parallel
    if (threadIdx.x == 0){ scr[0] = 0; } // candidate_count
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x){
        const float* nb = node_ptr(tree, D, i);
        int t = f2i(nb[COL_NODE_TYPE]);
        if (t == NODE_TYPE_UNUSED) continue;
        if (!(t == NODE_TYPE_DECISION || t == NODE_TYPE_ROOT_BRANCH)) continue;

        int d = f2i(nb[COL_DEPTH]);
        if (d >= max_depth - 1) continue; // too deep

        int cc = ch[i];
        if (cc >= max_children) continue; // full capacity

        // exclude parents with action children to avoid mixing
        if (ac[i] > 0) continue;

        float w = (float)(max_children - cc) / (float)(d + 1);
        if (w <= 0.0f) continue;

        int pos = atomicAdd(&scr[0], 1); // candidate_count++
        if (pos < N){
            cidx[pos] = i;
            cw[pos] = w;
        }
    }
    __syncthreads();

    // 3) Build freelist of UNUSED slots (O(1) allocation later)
    if (threadIdx.x == 0){ scr[1] = 0; scr[2] = 0; } // freelist_count(raw)=0, cursor=0
    __syncthreads();

    // Fill freelist entries
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        const float* nb = node_ptr(tree, D, i);
        if (f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED){
            int pos = atomicAdd(&scr[1], 1); // raw count
            // capacity guard; valid payload indices are scr[3 .. 3+cap-1]
            int freelist_cap = 2 * N - 3;
            if (freelist_cap > 0 && pos < freelist_cap){
                scr[3 + pos] = i;
            }
        }
    }
    __syncthreads();

    // If no candidates or zero budget, done
    const int budget = num_to_grow[b];
    if (threadIdx.x == 0){
        const int cand_count = scr[0];
        if (cand_count == 0 || budget <= 0) return;
    }
    __syncthreads();

    // 4) Thread 0: sample initial parent via roulette wheel selection
    int parent_idx = -1;
    if (threadIdx.x == 0){
        Lcg rng(0x9E3779B1u ^ (unsigned)(b * 2654435761u + 12345u));
        const int cand_count = scr[0];

        float sum_w = 0.0f;
        for (int k = 0; k < cand_count && k < N; ++k){ 
            sum_w += cw[k]; 
        }

        if (sum_w > 0.0f){
            float r = rng.nextf() * sum_w;
            float acc = 0.0f;
            for (int k = 0; k < cand_count && k < N; ++k){
                acc += cw[k];
                if (r <= acc){ 
                    parent_idx = cidx[k]; 
                    break; 
                }
            }
        }
        if (parent_idx < 0 && cand_count > 0) parent_idx = cidx[0]; // fallback

        if (valid_node_index(parent_idx, N)) open_q[0] = parent_idx;
    }
    __syncthreads();

    // 5) Serial growth (thread 0) with O(1) freelist allocations and queue guards
    if (threadIdx.x == 0){
        int q_head = 0, q_tail = (open_q[0] >= 0 ? 1 : 0);
        int created = 0;
        int dec_out = 0, act_out = 0;

        Lcg rng(0xC001C0DEu ^ (unsigned)(b * 747796405u + 48271u));

        // Apply freelist cap to avoid OOB
        int freelist_count_raw = scr[1];
        int freelist_cap = 2 * N - 3;
        if (freelist_cap < 0) freelist_cap = 0;
        int freelist_count = freelist_count_raw;
        if (freelist_count > freelist_cap) freelist_count = freelist_cap;
        int free_cursor = scr[2];

        // O(1) slot allocation from freelist
        auto alloc_slot = [&](int& out_idx) -> bool {
            if (free_cursor >= freelist_count) return false;
            int idx = scr[3 + free_cursor++];
            if (!valid_node_index(idx, N)) return false;
            const float* nb = node_ptr(tree, D, idx);
            if (f2i(nb[COL_NODE_TYPE]) != NODE_TYPE_UNUSED) return false;
            out_idx = idx; 
            return true;
        };

        const int q_capacity = 2 * N;
        int max_iters = 4 * (budget > 0 ? budget : 1); // prevent infinite loops
        int it = 0;

        // Growth loop with invariant enforcement
        while (created < budget && q_head < q_tail && it < max_iters){
            ++it;
            int cur_parent = open_q[q_head++];
            if (!valid_node_index(cur_parent, N)) continue;

            const float* pb = node_ptr(tree, D, cur_parent);
            int pd = f2i(pb[COL_DEPTH]);
            bool force_action = (pd + 1 >= max_depth - 1);

            int cur_childs = ch[cur_parent];
            bool choose_action = force_action || (rng.nextf() < 0.5f);

            if (choose_action){
                // ACTION only when parent has 0 children (single-action parent rule)
                if (cur_childs > 0) continue; 

                int slot;
                if (!alloc_slot(slot)) break; // no free slots

                float* nb = node_ptr(tree, D, slot);
                nb[COL_NODE_TYPE]  = i2f(NODE_TYPE_ACTION);
                nb[COL_PARENT_IDX] = i2f(cur_parent);
                nb[COL_DEPTH]      = i2f(pd + 1);

                ch[cur_parent] = 1;  // exactly one child now
                ac[cur_parent] = 1;  // action child

                int rbt = get_root_branch_type(tree, D, N, cur_parent);
                if (act_out < max_new_nodes){ 
                    out_act[act_out] = slot; 
                    out_rt[act_out] = i2f(rbt); 
                    ++act_out; 
                }

                ++created;
                continue; // ACTION is leaf, no further expansion
            }

            // DECISION path: cannot mix and must have capacity
            if (ac[cur_parent] > 0) continue; // no mixing rule
            int can_add = max_children - cur_childs;
            if (can_add <= 0) continue;

            int remaining_budget = budget - created;
            int kmax = (can_add < remaining_budget ? can_add : remaining_budget);
            if (kmax <= 0) continue;

            // Queue capacity guard: must be able to enqueue all new decisions
            int q_room = q_capacity - q_tail;
            if (q_room <= 0) break; // cannot safely create DECISIONs

            if (kmax > q_room) kmax = q_room; // ensure all can be enqueued

            int k = 1;
            if (kmax > 1){
                k = 1 + (int)(rng.nextf() * kmax);
                if (k < 1) k = 1;
                if (k > kmax) k = kmax;
            }

            for (int r = 0; r < k; ++r){
                int slot;
                if (!alloc_slot(slot)) { it = max_iters; break; } // out of free slots

                float* nb = node_ptr(tree, D, slot);
                nb[COL_NODE_TYPE]  = i2f(NODE_TYPE_DECISION);
                nb[COL_PARENT_IDX] = i2f(cur_parent);
                nb[COL_DEPTH]      = i2f(pd + 1);

                ++cur_childs;   
                ch[cur_parent] = cur_childs;
                ++dc[cur_parent];

                if (dec_out < max_new_nodes) out_dec[dec_out++] = slot;

                // enqueue new decision (safe by q_room guard)
                open_q[q_tail++] = slot;

                ++created;
                if (created >= budget) break;
            }
        }

        // Dangling DECISION parents → attach exactly one ACTION (if depth allows)
        // This ensures the leaf rule: all leaves must be ACTION
        while (q_head < q_tail){
            int dp = open_q[q_head++];
            if (!valid_node_index(dp, N)) continue;
            if (ch[dp] == 0){
                const float* pb2 = node_ptr(tree, D, dp);
                int d2 = f2i(pb2[COL_DEPTH]);
                if (d2 + 1 < max_depth){
                    int slot;
                    if (!alloc_slot(slot)) break;

                    float* nb2 = node_ptr(tree, D, slot);
                    nb2[COL_NODE_TYPE]  = i2f(NODE_TYPE_ACTION);
                    nb2[COL_PARENT_IDX] = i2f(dp);
                    nb2[COL_DEPTH]      = i2f(d2 + 1);

                    ch[dp] = 1;
                    ac[dp] = 1;

                    int rbt = get_root_branch_type(tree, D, N, dp);
                    if (act_out < max_new_nodes){ 
                        out_act[act_out] = slot; 
                        out_rt[act_out] = i2f(rbt); 
                        ++act_out; 
                    }
                }
            }
        }

        // write back cursor (not used after kernel, but good practice)
        scr[2] = free_cursor;
    }
}

void add_subtrees_batch_cuda(
    torch::Tensor trees, torch::Tensor num_to_grow,
    int max_children, int max_depth, int max_nodes, int max_new_nodes,
    torch::Tensor new_decision_nodes, torch::Tensor new_action_nodes, torch::Tensor action_root_branch_type,
    torch::Tensor bfs_queue_buffer, torch::Tensor result_indices_buffer,
    torch::Tensor child_count_buffer, torch::Tensor act_cnt_buffer, torch::Tensor dec_cnt_buffer,
    torch::Tensor candidate_indices_buffer, torch::Tensor candidate_weights_buffer
){
    TORCH_CHECK(trees.is_cuda(), "trees must be CUDA");
    TORCH_CHECK(num_to_grow.is_cuda(), "num_to_grow must be CUDA");
    TORCH_CHECK(new_decision_nodes.is_cuda() && new_action_nodes.is_cuda() && action_root_branch_type.is_cuda(),
                "output tensors must be CUDA");
    TORCH_CHECK(bfs_queue_buffer.is_cuda() && result_indices_buffer.is_cuda() &&
                child_count_buffer.is_cuda() && act_cnt_buffer.is_cuda() && dec_cnt_buffer.is_cuda() &&
                candidate_indices_buffer.is_cuda() && candidate_weights_buffer.is_cuda(),
                "work buffers must be CUDA");

    TORCH_CHECK(trees.scalar_type() == torch::kFloat32, "trees must be float32");
    TORCH_CHECK(num_to_grow.scalar_type() == torch::kInt32, "num_to_grow must be int32");
    TORCH_CHECK(new_decision_nodes.scalar_type() == torch::kInt32, "new_decision_nodes must be int32");
    TORCH_CHECK(new_action_nodes.scalar_type() == torch::kInt32, "new_action_nodes must be int32");
    TORCH_CHECK(action_root_branch_type.scalar_type() == torch::kFloat32, "action_root_branch_type must be float32");

    TORCH_CHECK(trees.dim() == 3 && trees.is_contiguous(), "trees must be (B,N,D) contiguous");
    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);
    TORCH_CHECK(N == max_nodes, "max_nodes must equal trees.size(1)");

    auto check_shape = [&](const torch::Tensor& t, int dim0, int dim1, const char* name){
        TORCH_CHECK(t.size(0) == dim0 && t.size(1) == dim1, name);
    };
    check_shape(new_decision_nodes, B, max_new_nodes, "new_decision_nodes shape (B,K)");
    check_shape(new_action_nodes,   B, max_new_nodes, "new_action_nodes shape (B,K)");
    check_shape(action_root_branch_type, B, max_new_nodes, "action_root_branch_type shape (B,K)");
    check_shape(bfs_queue_buffer,   B, 2 * N, "bfs_queue_buffer shape (B,2N)");
    check_shape(result_indices_buffer, B, 2 * N, "result_indices_buffer shape (B,2N)");
    check_shape(child_count_buffer, B, N, "child_count_buffer shape (B,N)");
    check_shape(act_cnt_buffer,     B, N, "act_cnt_buffer shape (B,N)");
    check_shape(dec_cnt_buffer,     B, N, "dec_cnt_buffer shape (B,N)");
    check_shape(candidate_indices_buffer, B, N, "candidate_indices_buffer shape (B,N)");
    check_shape(candidate_weights_buffer, B, N, "candidate_weights_buffer shape (B,N)");

    dim3 grid(B), block(128); // one block per tree, parallel scans within each tree
    k_add_subtrees_batch<<<grid, block>>>(
        trees.data_ptr<float>(), B, N, D,
        num_to_grow.data_ptr<int>(),
        max_children, max_depth, max_nodes, max_new_nodes,
        new_decision_nodes.data_ptr<int>(),
        new_action_nodes.data_ptr<int>(),
        action_root_branch_type.data_ptr<float>(),
        bfs_queue_buffer.data_ptr<int>(),
        result_indices_buffer.data_ptr<int>(),
        child_count_buffer.data_ptr<int>(),
        act_cnt_buffer.data_ptr<int>(),
        dec_cnt_buffer.data_ptr<int>(),
        candidate_indices_buffer.data_ptr<int>(),
        candidate_weights_buffer.data_ptr<float>());
    CUDA_CHECK_ERRORS();
}