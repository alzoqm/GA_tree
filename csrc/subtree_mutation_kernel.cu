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

// =============================================================================
// DELETE-SUBTREE MUTATION: OPTIMIZED CUDA IMPLEMENTATION WITH INVARIANT PRESERVATION
// =============================================================================

// Safe float<->int conversion with bounds checking
__device__ __forceinline__ int   d_f2i(float x){ return __float2int_rn(x); }
__device__ __forceinline__ float d_i2f(int   x){ return static_cast<float>(x); }

__device__ __forceinline__ bool d_valid_idx(int idx, int N){ return idx >= 0 && idx < N; }
__device__ __forceinline__ float* d_node(float* base, int D, int i){ return base + (size_t)i * D; }
__device__ __forceinline__ const float* d_node(const float* base, int D, int i){ return base + (size_t)i * D; }

// Find root index (ROOT_BRANCH or fallback to any node with parent=-1)
__device__ int find_root_index(const float* tree, int N, int D){
    int fallback = -1;
    for (int i = 0; i < N; ++i){
        const float* nb = d_node(tree, D, i);
        int t = d_f2i(nb[COL_NODE_TYPE]);
        if (t == NODE_TYPE_UNUSED) continue;
        int p = d_f2i(nb[COL_PARENT_IDX]);
        if (p < 0){
            if (t == NODE_TYPE_ROOT_BRANCH) return i;
            if (fallback < 0) fallback = i;
        }
    }
    return fallback;
}

// BFS collect subtree into q/res, return count and number of actions in subtree.
// q and res are per-tree buffers of capacity 2*N (allocated in Python).
__device__ void bfs_subtree_collect(
    const float* tree, int N, int D, int r,
    int* q, int* res, int qcap,
    int* out_count, int* out_action_count
){
    *out_count = 0; *out_action_count = 0;
    if (!d_valid_idx(r, N)) return;

    const float* rb = d_node(tree, D, r);
    if (d_f2i(rb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) return;

    int head = 0, tail = 0, outc = 0, actc = 0;
    if (qcap <= 0) { *out_count = 0; *out_action_count = 0; return; }

    q[tail++] = r; res[outc++] = r;
    if (d_f2i(rb[COL_NODE_TYPE]) == NODE_TYPE_ACTION) ++actc;

    while (head < tail && tail < qcap && outc < qcap){
        int cur = q[head++];
        for (int j = 0; j < N && tail < qcap && outc < qcap; ++j){
            const float* nb = d_node(tree, D, j);
            if (d_f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) continue;
            if (d_f2i(nb[COL_PARENT_IDX]) == cur){
                q[tail++] = j;
                res[outc++] = j;
                if (d_f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_ACTION) ++actc;
            }
        }
    }
    *out_count = outc; *out_action_count = actc;
}

// Recompute parent-child counts excluding nodes marked for deletion/repair
__device__ void recompute_counts_excluding_masks(
    const float* tree, int N, int D,
    const int* del_mask, const int* rep_mask,
    int* child_cnt, int* act_cnt, int* dec_cnt
){
    // Phase 1: Zero counts (separate from accumulation to avoid race conditions)
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        child_cnt[i] = 0; act_cnt[i] = 0; dec_cnt[i] = 0;
    }
    __syncthreads();

    // Phase 2: Accumulate counts
    for (int j = threadIdx.x; j < N; j += blockDim.x){
        const float* nb = d_node(tree, D, j);
        if (d_f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) continue;
        if (del_mask[j] || rep_mask[j]) continue;
        int p = d_f2i(nb[COL_PARENT_IDX]);
        if (!d_valid_idx(p, N)) continue;
        atomicAdd(&child_cnt[p], 1);
        int t = d_f2i(nb[COL_NODE_TYPE]);
        if (t == NODE_TYPE_ACTION)   atomicAdd(&act_cnt[p], 1);
        if (t == NODE_TYPE_DECISION) atomicAdd(&dec_cnt[p], 1);
    }
    __syncthreads();
}

// Count children of parent excluding masked nodes
__device__ int parent_child_count_excluding_masks(
    const float* tree, int N, int D, int p,
    const int* del_mask, const int* rep_mask
){
    if (!d_valid_idx(p, N)) return 0;
    int c = 0;
    for (int j = 0; j < N; ++j){
        const float* nb = d_node(tree, D, j);
        if (d_f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) continue;
        if (del_mask[j] || rep_mask[j]) continue;
        if (d_f2i(nb[COL_PARENT_IDX]) == p) ++c;
    }
    return c;
}

// One block per tree - comprehensive invariant-preserving DeleteSubtree kernel
__global__ void k_delete_subtrees_batch(
    float* __restrict__ trees, int B, int N, int D,
    const int* __restrict__ mutate_mask,
    float alpha,
    int ensure_action_left,

    int* __restrict__ child_cnt,
    int* __restrict__ act_cnt,
    int* __restrict__ dec_cnt,

    int* __restrict__ cand_idx,
    float* __restrict__ cand_w,

    int* __restrict__ bfs_q,
    int* __restrict__ bfs_res,

    int* __restrict__ del_mask,
    int* __restrict__ rep_mask,

    int* __restrict__ chosen_roots
){
    const int b = blockIdx.x;
    if (b >= B) return;
    if (mutate_mask[b] == 0) { if (threadIdx.x==0) chosen_roots[b] = -1; return; }

    float* tree = trees + (size_t)b * N * D;

    // Per-tree slices
    int* ch    = child_cnt + (size_t)b * N;
    int* ac    = act_cnt   + (size_t)b * N;
    int* dc    = dec_cnt   + (size_t)b * N;
    int* cidx  = cand_idx  + (size_t)b * N;
    float* cw  = cand_w    + (size_t)b * N;
    int* qbuf  = bfs_q     + (size_t)b * (2 * N);
    int* rbuf  = bfs_res   + (size_t)b * (2 * N);
    int* dmask = del_mask  + (size_t)b * N;
    int* rmask = rep_mask  + (size_t)b * N;

    // Zero masks and candidate arrays
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        dmask[i] = 0; rmask[i] = 0; cidx[i] = -1; cw[i] = 0.0f;
    }
    __syncthreads();

    // === Phase 1: zero counts ===
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        ch[i]=0; ac[i]=0; dc[i]=0;
    }
    __syncthreads();

    // === total_actions (shared) ===
    __shared__ int sh_total_actions;
    if (threadIdx.x == 0) sh_total_actions = 0;
    __syncthreads();

    // === Phase 2: accumulate counts and total_actions ===
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        const float* nb = d_node(tree, D, i);
        int t = d_f2i(nb[COL_NODE_TYPE]);
        if (t == NODE_TYPE_UNUSED) continue;
        int p = d_f2i(nb[COL_PARENT_IDX]);
        if (d_valid_idx(p, N)){
            atomicAdd(&ch[p], 1);
            if (t == NODE_TYPE_ACTION)   atomicAdd(&ac[p], 1);
            if (t == NODE_TYPE_DECISION) atomicAdd(&dc[p], 1);
        }
        if (t == NODE_TYPE_ACTION) atomicAdd(&sh_total_actions, 1);
    }
    __syncthreads();

    int total_actions = 0;
    if (threadIdx.x == 0) total_actions = sh_total_actions;
    __syncthreads();

    // Build candidate list (exclude ROOT_BRANCH and G1 guard)
    int cand_count = 0;
    if (threadIdx.x == 0){
        for (int i = 0; i < N && cand_count < N; ++i){
            const float* nb = d_node(tree, D, i);
            int t = d_f2i(nb[COL_NODE_TYPE]);
            if (t == NODE_TYPE_UNUSED) continue;
            if (t == NODE_TYPE_ROOT_BRANCH) continue;

            // G1: Prevent immediate root orphaning
            int p = d_f2i(nb[COL_PARENT_IDX]);
            bool g1_violate = false;
            if (d_valid_idx(p, N)){
                const float* pb = d_node(tree, D, p);
                int pt = d_f2i(pb[COL_NODE_TYPE]);
                if (pt == NODE_TYPE_ROOT_BRANCH && ch[p] == 1) g1_violate = true;
            }
            if (g1_violate) continue;

            // Calculate subtree size and action count
            int cnt = 0, actc = 0;
            bfs_subtree_collect(tree, N, D, i, qbuf, rbuf, 2*N, &cnt, &actc);

            float w = (cnt > 0 ? powf((float)cnt, alpha) : 0.0f);
            
            // Optional: Ensure at least one ACTION remains if ensure_action_left=True
            if (ensure_action_left && (total_actions - actc) <= 0) w = 0.0f;

            cidx[cand_count] = i;
            cw[cand_count]   = w;
            ++cand_count;
        }
    }
    __syncthreads();

    // Weighted sampling using deterministic LCG seeded by batch index
    int chosen = -1;
    if (threadIdx.x == 0){
        float sumw = 0.0f;
        for (int k = 0; k < cand_count; ++k) sumw += cw[k];

        if (sumw > 0.0f){
            unsigned int s = 0x9E3779B1u ^ (unsigned)(b * 2654435761u + 12345u);
            auto nextf = [&s](){ s = 1664525u * s + 1013904223u; return float(s & 0x00FFFFFF) / float(0x01000000); };
            float r = nextf() * sumw;
            float acc = 0.0f;
            for (int k = 0; k < cand_count; ++k){
                acc += cw[k];
                if (r <= acc){ chosen = cidx[k]; break; }
            }
        }
        chosen_roots[b] = (chosen >= 0 ? chosen : -1);
    }
    __syncthreads();
    if (chosen_roots[b] < 0) return;

    chosen = chosen_roots[b];

    // Mark subtree for deletion
    int cnt = 0, actc = 0;
    bfs_subtree_collect(tree, N, D, chosen, qbuf, rbuf, 2*N, &cnt, &actc);
    for (int i = threadIdx.x; i < cnt; i += blockDim.x){
        int node = rbuf[i];
        if (d_valid_idx(node, N)) dmask[node] = 1;
    }
    __syncthreads();

    // Repair loop with EARLY EXIT to prevent infinite loops
    for (int it = 0; it < N; ++it){
        __shared__ int sh_changed;
        if (threadIdx.x == 0) sh_changed = 0;
        __syncthreads();

        // R2: ACTION must be leaf => remove all children of ACTION nodes
        for (int i = threadIdx.x; i < N; i += blockDim.x){
            const float* nb = d_node(tree, D, i);
            if (d_f2i(nb[COL_NODE_TYPE]) != NODE_TYPE_ACTION) continue;
            if (dmask[i] || rmask[i]) continue;
            for (int j = 0; j < N; ++j){
                const float* cb = d_node(tree, D, j);
                if (d_f2i(cb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) continue;
                if (dmask[j] || rmask[j]) continue;
                if (d_f2i(cb[COL_PARENT_IDX]) == i) { rmask[j] = 1; sh_changed = 1; }
            }
        }
        __syncthreads();

        // Recompute counts excluding masked nodes
        recompute_counts_excluding_masks(tree, N, D, dmask, rmask, ch, ac, dc);

        // R1: If parent has any ACTION child, keep exactly ONE action; delete others and all DECISION children
        for (int i = threadIdx.x; i < N; i += blockDim.x) cidx[i] = 0; // reuse as keep_seen
        __syncthreads();

        for (int j = threadIdx.x; j < N; j += blockDim.x){
            const float* nb = d_node(tree, D, j);
            if (d_f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) continue;
            if (dmask[j] || rmask[j]) continue;
            int p = d_f2i(nb[COL_PARENT_IDX]);
            if (!d_valid_idx(p, N)) continue;
            if (ac[p] > 0){
                int t = d_f2i(nb[COL_NODE_TYPE]);
                if (t == NODE_TYPE_ACTION){
                    if (atomicCAS(&cidx[p], 0, 1) != 0){ rmask[j] = 1; sh_changed = 1; }
                } else {
                    rmask[j] = 1; sh_changed = 1;
                }
            }
        }
        __syncthreads();

        // R3: DECISION with no children => remove (using ch[] cached counts)
        for (int i = threadIdx.x; i < N; i += blockDim.x){
            const float* nb = d_node(tree, D, i);
            if (dmask[i] || rmask[i]) continue;
            if (d_f2i(nb[COL_NODE_TYPE]) != NODE_TYPE_DECISION) continue;
            if (ch[i] == 0){ rmask[i] = 1; sh_changed = 1; }
        }
        __syncthreads();

        // Early exit if no changes in this iteration
        if (sh_changed == 0) break;
    }

    // G2: Ensure root has >=1 child and (optionally) at least one ACTION remains
    int root = find_root_index(tree, N, D);
    bool root_ok = true;
    if (root >= 0){
        int rc = parent_child_count_excluding_masks(tree, N, D, root, dmask, rmask);
        if (rc <= 0) root_ok = false;
    }

    __shared__ int sh_actions_left;
    if (threadIdx.x == 0) sh_actions_left = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x){
        const float* nb = d_node(tree, D, i);
        if (d_f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_ACTION && !(dmask[i] || rmask[i])) {
            atomicOr(&sh_actions_left, 1);
        }
    }
    __syncthreads();

    int g2_fail = 0;
    if (threadIdx.x == 0){
        if (!root_ok) g2_fail = 1;
        else if (ensure_action_left && sh_actions_left == 0) g2_fail = 1;
    }
    __syncthreads();

    // If G2 fails, cancel the mutation by clearing all masks
    if (g2_fail){
        for (int i = threadIdx.x; i < N; i += blockDim.x){ dmask[i] = 0; rmask[i] = 0; }
        if (threadIdx.x == 0) chosen_roots[b] = -1;
        __syncthreads();
        return;
    }
}

// Host wrapper function with comprehensive validation
void delete_subtrees_batch_cuda(
    torch::Tensor trees, torch::Tensor mutate_mask_i32,
    int max_nodes, float alpha, int ensure_action_left,
    torch::Tensor child_count_buffer,
    torch::Tensor act_cnt_buffer,
    torch::Tensor dec_cnt_buffer,
    torch::Tensor candidate_indices_buffer,
    torch::Tensor candidate_weights_buffer,
    torch::Tensor bfs_queue_buffer,
    torch::Tensor result_indices_buffer,
    torch::Tensor deletion_mask_buffer,
    torch::Tensor repair_mask_buffer,
    torch::Tensor chosen_roots_buffer
){
    TORCH_CHECK(trees.is_cuda(), "trees must be CUDA");
    TORCH_CHECK(mutate_mask_i32.is_cuda(), "mutate_mask must be CUDA");
    TORCH_CHECK(trees.scalar_type() == torch::kFloat32, "trees: float32");
    TORCH_CHECK(mutate_mask_i32.scalar_type() == torch::kInt32, "mutate_mask: int32");
    TORCH_CHECK(trees.dim()==3 && trees.is_contiguous(), "trees must be (B,N,D) contiguous");

    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);
    TORCH_CHECK(N == max_nodes, "max_nodes must equal trees.size(1)");

    auto chk = [&](const torch::Tensor& t, int s0, int s1, const char* msg){
        TORCH_CHECK(t.is_cuda(), "buffer must be CUDA: ", msg);
        TORCH_CHECK(t.size(0)==s0 && t.size(1)==s1, msg);
    };
    chk(child_count_buffer,       B, N,  "child_count_buffer (B,N)");
    chk(act_cnt_buffer,           B, N,  "act_cnt_buffer (B,N)");
    chk(dec_cnt_buffer,           B, N,  "dec_cnt_buffer (B,N)");
    chk(candidate_indices_buffer, B, N,  "candidate_indices_buffer (B,N)");
    chk(candidate_weights_buffer, B, N,  "candidate_weights_buffer (B,N)");
    chk(bfs_queue_buffer,         B, 2*N,"bfs_queue_buffer (B,2N)");
    chk(result_indices_buffer,    B, 2*N,"result_indices_buffer (B,2N)");
    chk(deletion_mask_buffer,     B, N,  "deletion_mask_buffer (B,N)");
    chk(repair_mask_buffer,       B, N,  "repair_mask_buffer (B,N)");
    TORCH_CHECK(chosen_roots_buffer.is_cuda() && chosen_roots_buffer.size(0)==B, "chosen_roots_buffer (B,)");

    dim3 grid(B), block(128);
    k_delete_subtrees_batch<<<grid, block>>>(
        trees.data_ptr<float>(), B, N, D,
        mutate_mask_i32.data_ptr<int>(),
        alpha,
        ensure_action_left,
        child_count_buffer.data_ptr<int>(),
        act_cnt_buffer.data_ptr<int>(),
        dec_cnt_buffer.data_ptr<int>(),
        candidate_indices_buffer.data_ptr<int>(),
        candidate_weights_buffer.data_ptr<float>(),
        bfs_queue_buffer.data_ptr<int>(),
        result_indices_buffer.data_ptr<int>(),
        deletion_mask_buffer.data_ptr<int>(),
        repair_mask_buffer.data_ptr<int>(),
        chosen_roots_buffer.data_ptr<int>());
    CUDA_CHECK_ERRORS();
}

// =============================================================================
// CRITICAL REPAIR: CUDA KERNEL TO REPLACE SLOW PYTHON LOOPS
// =============================================================================

// Critical repair kernel: Ensure no root branch is left without children
__global__ void k_critical_repair_batch(
    float* __restrict__ trees, int B, int N, int D
){
    const int b = blockIdx.x;
    if (b >= B) return;
    
    float* tree = trees + (size_t)b * N * D;
    
    // Only thread 0 performs the repair logic for this tree
    if (threadIdx.x == 0) {
        // Check each root branch (indices 0, 1, 2)
        for (int root_idx = 0; root_idx < 3; ++root_idx) {
            const float* root_node = d_node(tree, D, root_idx);
            if (d_f2i(root_node[COL_NODE_TYPE]) != NODE_TYPE_ROOT_BRANCH) {
                continue;
            }
            
            // Count children of this root branch
            bool has_children = false;
            for (int child_idx = 3; child_idx < N; ++child_idx) {
                const float* child_node = d_node(tree, D, child_idx);
                if (d_f2i(child_node[COL_NODE_TYPE]) != NODE_TYPE_UNUSED && 
                    d_f2i(child_node[COL_PARENT_IDX]) == root_idx) {
                    has_children = true;
                    break;
                }
            }
            
            // If no children, add a default ACTION child
            if (!has_children) {
                // Find an available slot
                for (int slot_idx = 3; slot_idx < N; ++slot_idx) {
                    float* slot_node = d_node(tree, D, slot_idx);
                    if (d_f2i(slot_node[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) {
                        slot_node[COL_NODE_TYPE] = d_i2f(NODE_TYPE_ACTION);
                        slot_node[COL_PARENT_IDX] = d_i2f(root_idx);
                        slot_node[COL_DEPTH] = 1.0f;
                        slot_node[COL_PARAM_1] = d_i2f(ACTION_CLOSE_ALL);
                        
                        // Clear other parameters
                        for (int col = 4; col < D; ++col) {
                            slot_node[col] = 0.0f;
                        }
                        break;
                    }
                }
            }
        }
    }
}

// Host wrapper for critical repair
void critical_repair_batch_cuda(torch::Tensor trees) {
    TORCH_CHECK(trees.is_cuda(), "trees must be CUDA");
    TORCH_CHECK(trees.scalar_type() == torch::kFloat32, "trees must be float32");
    TORCH_CHECK(trees.dim() == 3 && trees.is_contiguous(), "trees must be (B,N,D) contiguous");
    
    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);
    
    dim3 grid(B), block(1); // Single thread per tree for simplicity
    k_critical_repair_batch<<<grid, block>>>(
        trees.data_ptr<float>(), B, N, D);
    CUDA_CHECK_ERRORS();
}