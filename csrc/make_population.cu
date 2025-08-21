// csrc/make_population.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "constants.h"
#include "make_population.cuh"

#define CUDA_CHECK_ERRORS() \
  do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  }} while (0)

// Utility functions
__device__ __forceinline__ int   f2i(float x){ return __float2int_rn(x); }
__device__ __forceinline__ float i2f(int   x){ return static_cast<float>(x); }
__device__ __forceinline__ bool valid_idx(int idx, int N){ return idx >= 0 && idx < N; }

__device__ __forceinline__ float* node_ptr(float* base, int D, int i){ return base + (size_t)i * D; }
__device__ __forceinline__ const float* node_ptr(const float* base, int D, int i){ return base + (size_t)i * D; }

// Lightweight LCG for reproducible per-block RNG
struct Lcg {
    unsigned int s;
    __device__ explicit Lcg(unsigned int seed): s(seed) {}
    __device__ __forceinline__ unsigned int nextu(){ s = 1664525u * s + 1013904223u; return s; }
    __device__ __forceinline__ float nextf(){ return float(nextu() & 0x00FFFFFF) / float(0x01000000); }
    __device__ __forceinline__ int randint(int lo, int hi){
        if (hi < lo) return lo;
        unsigned int r = nextu();
        int span = (hi - lo + 1);
        return lo + (int)(r % (unsigned)span);
    }
};

// Scratch buffer indices (per-tree slice)
#define SCR_FC   0  // freelist count (clamped to capacity)
#define SCR_CUR  1  // freelist cursor
#define SCR_CC   2  // candidate count (for diagnostics)

// Helper function to get root branch type from index
__device__ __forceinline__ int root_type_from_index(int r){
    if (r == 0) return ROOT_BRANCH_LONG;
    if (r == 1) return ROOT_BRANCH_HOLD;
    return ROOT_BRANCH_SHORT;
}

// Walk up to find root branch type - prevents array bounds violations
__device__ static int get_root_branch_type_make_pop(const float* tree, int D, int N, int start_idx){
    int cur = start_idx, safety = 0;
    while(valid_idx(cur, N) && safety < N){
        const float* nb = node_ptr(tree, D, cur);
        int p = f2i(nb[COL_PARENT_IDX]);
        if (p < 0) return f2i(nb[COL_PARAM_1]); // PARAM_1 of ROOT_BRANCH holds branch type
        cur = p; 
        ++safety;
    }
    return ROOT_BRANCH_HOLD; // fallback
}

// Select action based on root context with bounds checking
__device__ __forceinline__ int pick_action_for_root(Lcg& rng, int rbt,
                                                    const int* la, int la_n,
                                                    const int* ha, int ha_n,
                                                    const int* sa, int sa_n){
    const int* lst = nullptr; 
    int n = 0;
    if (rbt == ROOT_BRANCH_LONG){ lst = la; n = la_n; }
    else if (rbt == ROOT_BRANCH_HOLD){ lst = ha; n = ha_n; }
    else { lst = sa; n = sa_n; }
    if (n <= 0) return ACTION_CLOSE_ALL; // fallback action
    int k = rng.randint(0, n-1);
    return lst[k];
}

__global__ void k_init_population_batch(
    float* __restrict__ trees, int B, int N, int D,
    const int* __restrict__ total_budget,
    int max_children, int max_depth, int max_nodes,
    int* __restrict__ bfs_q,
    int* __restrict__ scratch,
    int* __restrict__ child_cnt,
    int* __restrict__ act_cnt,
    int* __restrict__ dec_cnt,
    int* __restrict__ cand_idx,
    float* __restrict__ cand_w,
    const int* __restrict__ num_feat_indices, int num_feat_count,
    const float* __restrict__ num_feat_minmax, // (Kn,2) flattened
    const int* __restrict__ bool_feat_indices, int bool_feat_count,
    const int* __restrict__ ff_pairs, int ff_pair_count, // (P,2) flattened
    const int* __restrict__ long_actions,  int long_count,
    const int* __restrict__ hold_actions,  int hold_count,
    const int* __restrict__ short_actions, int short_count
){
    const int b = blockIdx.x;
    if (b >= B) return;

    // Per-tree data pointers
    float* tree = trees + (size_t)b * N * D;
    int*   q    = bfs_q  + (size_t)b * (2 * N);
    int*   scr  = scratch+ (size_t)b * (2 * N);
    int*   ch   = child_cnt + (size_t)b * N;
    int*   ac   = act_cnt   + (size_t)b * N;
    int*   dc   = dec_cnt   + (size_t)b * N;
    int*   cidx = cand_idx  + (size_t)b * N;
    float* cw   = cand_w    + (size_t)b * N;

    // Step 0: Zero all buffers in parallel
    for (int i = threadIdx.x; i < N*D; i += blockDim.x){
        trees[(size_t)b * N * D + i] = 0.0f;
    }
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        ch[i] = 0; ac[i] = 0; dc[i] = 0; cidx[i] = -1; cw[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < 2 * N; i += blockDim.x){
        q[i] = -1; scr[i] = -1;
    }
    __syncthreads();

    // Step 1: Create 3 root-branch nodes at indices 0, 1, 2 (thread 0 only)
    if (threadIdx.x == 0){
        for (int r = 0; r < 3; ++r){
            float* nb = node_ptr(tree, D, r);
            nb[COL_NODE_TYPE]  = i2f(NODE_TYPE_ROOT_BRANCH);
            nb[COL_PARENT_IDX] = i2f(-1);
            nb[COL_DEPTH]      = i2f(0);
            nb[COL_PARAM_1]    = i2f(root_type_from_index(r)); // store branch type
        }
    }
    __syncthreads();

    // Step 2: Build freelist for [3..N-1] in scratch payload (parallel construction)
    if (threadIdx.x == 0){ scr[SCR_FC] = 0; scr[SCR_CUR] = 0; scr[SCR_CC] = 0; }
    __syncthreads();
    
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        if (i < 3) continue; // Skip root indices
        int pos = atomicAdd(&scr[SCR_FC], 1);
        int cap = 2 * N - 3; // Available payload slots
        if (pos < cap){ scr[3 + pos] = i; }
    }
    __syncthreads();
    
    // Clamp freelist count to available capacity
    if (threadIdx.x == 0){
        int fc = scr[SCR_FC];
        int cap = 2 * N - 3;
        if (fc > cap) fc = cap;
        if (fc < 0) fc = 0;
        scr[SCR_FC] = fc;
        scr[SCR_CUR] = 0;
    }
    __syncthreads();

    // Step 3: Seed BFS queue with root indices
    if (threadIdx.x == 0){ 
        q[0] = 0; q[1] = 1; q[2] = 2; 
    }
    __syncthreads();

    // Slot allocation lambda with capacity checking
    auto alloc_slot = [&](int& out_idx) -> bool {
        int cur = atomicAdd(&scr[SCR_CUR], 1);
        int fc = scr[SCR_FC];
        if (cur >= fc) return false;
        int idx = scr[3 + cur];
        if (!valid_idx(idx, N)) return false;
        out_idx = idx;
        return true;
    };

    // Step 4: Main growth loop (thread 0 orchestration)
    if (threadIdx.x == 0){
        int qh = 0, qt = 3, qcap = 2 * N;
        int budget = total_budget[b]; 
        if (budget < 0) budget = 0;

        int frontier = 0; // Count of DECISION leaves needing ACTION attachment
        Lcg rng(0x8F1BBCDCu ^ (unsigned)(b * 2654435761u + 91138233u));
        int max_iters = 4 * (budget > 0 ? budget : 1); // Infinite loop protection

        for (int it = 0; it < max_iters && budget > 0 && qh < qt; ++it){
            // Step 4.1: Build parent candidates from current queue
            int ccnt = 0;
            for (int scan = qh; scan < qt && ccnt < N; ++scan){
                int p = q[scan];
                if (!valid_idx(p, N)) continue;
                const float* pb = node_ptr(tree, D, p);
                int pt = f2i(pb[COL_NODE_TYPE]);
                if (!(pt == NODE_TYPE_ROOT_BRANCH || pt == NODE_TYPE_DECISION)) continue;

                int pd = f2i(pb[COL_DEPTH]);
                if (pd >= max_depth - 1) continue; // No room to grow

                int ccp = ch[p];
                if (ccp >= max_children) continue; // Full capacity

                if (ac[p] > 0) continue; // No mixing: parent with ACTION child forbidden

                // Candidate weight: favor fewer children and shallower depth
                float w = (float)(max_children - ccp) / (float)(pd + 1);
                if (w <= 0.0f) continue;

                cidx[ccnt] = p; 
                cw[ccnt] = w; 
                ++ccnt;
            }
            scr[SCR_CC] = ccnt;
            if (ccnt == 0) break; // No valid candidates

            // Step 4.2: Sample parent by roulette wheel
            float sumw = 0.0f;
            for (int k = 0; k < ccnt; ++k) sumw += cw[k];
            int parent = cidx[0]; // fallback
            if (sumw > 0.0f){
                float r = rng.nextf() * sumw, acc = 0.0f;
                for (int k = 0; k < ccnt; ++k){ 
                    acc += cw[k]; 
                    if (r <= acc){ parent = cidx[k]; break; }
                }
            }

            const float* pb = node_ptr(tree, D, parent);
            int pd = f2i(pb[COL_DEPTH]);
            int pre_cc = ch[parent];
            int pt = f2i(pb[COL_NODE_TYPE]);

            bool force_action = (pd + 1 >= max_depth - 1); // Depth guard
            bool choose_action = force_action || (rng.nextf() < 0.5f);

            // Step 4.3: Check DECISION family availability
            int has_num  = (num_feat_count  > 0) ? 1 : 0;
            int has_ff   = (ff_pair_count   > 0) ? 1 : 0;
            int has_bool = (bool_feat_count > 0) ? 1 : 0;
            int any_decision_family = (has_num | has_ff | has_bool);

            // Step 4.4: Frontier reservation guard
            int free_slots_now = scr[SCR_FC] - scr[SCR_CUR];
            if (!choose_action){
                if (!any_decision_family){
                    choose_action = true; // Force ACTION if no DECISION families
                } else {
                    // Calculate frontier increase if we add a DECISION
                    int inc_frontier = 0;
                    if (pt == NODE_TYPE_DECISION){
                        if (pre_cc == 0) inc_frontier = 0; // Parent leaves frontier, child enters
                        else inc_frontier = 1; // Child enters frontier
                    } else { 
                        inc_frontier = 1; // Root becomes non-frontier, child enters
                    }
                    if (free_slots_now <= (frontier + inc_frontier)){
                        choose_action = true; // Force ACTION to prevent resource starvation
                    }
                }
            }

            // Step 4.5: ACTION path
            if (choose_action){
                if (pre_cc > 0) continue; // Single-action parent rule violation
                int slot; 
                if (!alloc_slot(slot)) break; // Out of slots

                float* nb = node_ptr(tree, D, slot);
                nb[COL_NODE_TYPE]  = i2f(NODE_TYPE_ACTION);
                nb[COL_PARENT_IDX] = i2f(parent);
                nb[COL_DEPTH]      = i2f(pd + 1);

                // Select action based on root context
                int rbt = get_root_branch_type_make_pop(tree, D, N, parent);
                int at = pick_action_for_root(rng, rbt,
                                              long_actions, long_count,
                                              hold_actions, hold_count,
                                              short_actions, short_count);
                nb[COL_PARAM_1] = i2f(at);
                
                // Set action-specific parameters
                if (at == ACTION_NEW_LONG || at == ACTION_NEW_SHORT || at == ACTION_FLIP_POSITION){
                    nb[COL_PARAM_2] = rng.nextf(); // ratio
                    nb[COL_PARAM_3] = i2f(1 + (int)(rng.nextf() * 100.0f)); // leverage
                } else if (at == ACTION_CLOSE_PARTIAL || at == ACTION_ADD_POSITION){
                    nb[COL_PARAM_2] = rng.nextf(); // ratio
                }

                // Update counters
                ch[parent] = 1; 
                ac[parent] = 1;
                
                // Update frontier count
                if (pt == NODE_TYPE_DECISION && pre_cc == 0 && frontier > 0) frontier -= 1;
                budget -= 1;
                continue;
            }

            // Step 4.6: DECISION path
            if (ac[parent] > 0) continue; // No mixing rule
            if (!any_decision_family) continue; // No families available

            int can_add = max_children - pre_cc;
            if (can_add <= 0) continue;

            // Determine how many DECISION children to add
            int kmax = can_add;
            if (kmax > budget) kmax = budget;
            int qroom = qcap - qt;
            if (kmax > qroom) kmax = qroom;
            if (kmax <= 0) continue;

            int k = 1 + (int)(rng.nextf() * (float)kmax);
            if (k > kmax) k = kmax;

            for (int r = 0; r < k; ++r){
                // Re-check frontier reservation for each child
                int fs_now = scr[SCR_FC] - scr[SCR_CUR];
                int inc_frontier = 0;
                if (pt == NODE_TYPE_DECISION){
                    if (pre_cc == 0) inc_frontier = 0;
                    else inc_frontier = 1;
                } else { 
                    inc_frontier = 1;
                }
                if (fs_now <= (frontier + inc_frontier)) break;

                int slot; 
                if (!alloc_slot(slot)) { r = k; break; }
                
                float* nb = node_ptr(tree, D, slot);
                nb[COL_NODE_TYPE]  = i2f(NODE_TYPE_DECISION);
                nb[COL_PARENT_IDX] = i2f(parent);
                nb[COL_DEPTH]      = i2f(pd + 1);

                // Choose comparison type among available families only
                int avail = has_num + has_ff + has_bool;
                int pick = (avail > 0) ? rng.randint(1, avail) : 0; // 1..avail
                int ctype = COMP_TYPE_FEAT_NUM; // default

                if (avail == 0){
                    // Defensive fallback: should not happen due to guard above
                    nb[COL_NODE_TYPE]  = i2f(NODE_TYPE_ACTION);
                    int rbt = get_root_branch_type_make_pop(tree, D, N, parent);
                    int at  = pick_action_for_root(rng, rbt,
                                                   long_actions, long_count,
                                                   hold_actions, hold_count,
                                                   short_actions, short_count);
                    nb[COL_PARAM_1] = i2f(at);
                    if (at == ACTION_NEW_LONG || at == ACTION_NEW_SHORT || at == ACTION_FLIP_POSITION){
                        nb[COL_PARAM_2] = rng.nextf();
                        nb[COL_PARAM_3] = i2f(1 + (int)(rng.nextf() * 100.0f));
                    } else if (at == ACTION_CLOSE_PARTIAL || at == ACTION_ADD_POSITION){
                        nb[COL_PARAM_2] = rng.nextf();
                    }
                    ch[parent] = 1; ac[parent] = 1;
                    if (pt == NODE_TYPE_DECISION && pre_cc == 0 && frontier > 0) frontier -= 1;
                    budget -= 1;
                    break;
                } else {
                    // Select comparison type
                    if (has_num){
                        if (pick == 1) ctype = COMP_TYPE_FEAT_NUM;
                        --pick;
                    }
                    if (pick > 0 && has_ff){
                        if (pick == 1) ctype = COMP_TYPE_FEAT_FEAT;
                        --pick;
                    }
                    if (pick > 0 && has_bool){
                        ctype = COMP_TYPE_FEAT_BOOL;
                    }
                }
                nb[COL_PARAM_3] = i2f(ctype);

                // Set comparison parameters based on type
                if (ctype == COMP_TYPE_FEAT_NUM){
                    int kf = rng.randint(0, num_feat_count - 1);
                    int fidx = num_feat_indices[kf];
                    float fmin = num_feat_minmax[2 * kf + 0];
                    float fmax = num_feat_minmax[2 * kf + 1];
                    float thr = fmin + rng.nextf() * (fmax - fmin);
                    nb[COL_PARAM_1] = i2f(fidx);
                    nb[COL_PARAM_2] = i2f(rng.randint(0,1) ? OP_GTE : OP_LTE);
                    nb[COL_PARAM_4] = thr;
                } else if (ctype == COMP_TYPE_FEAT_FEAT){
                    int kp = rng.randint(0, ff_pair_count - 1);
                    int f1 = ff_pairs[2 * kp + 0];
                    int f2 = ff_pairs[2 * kp + 1];
                    nb[COL_PARAM_1] = i2f(f1);
                    nb[COL_PARAM_2] = i2f(rng.randint(0,1) ? OP_GTE : OP_LTE);
                    nb[COL_PARAM_4] = i2f(f2);
                } else { // FEAT_BOOL
                    int kb = rng.randint(0, bool_feat_count - 1);
                    int fidx = bool_feat_indices[kb];
                    nb[COL_PARAM_1] = i2f(fidx);
                    nb[COL_PARAM_4] = i2f(rng.randint(0,1));
                }

                // Update counters
                int pre_cc_local = pre_cc; 
                pre_cc += 1;
                ch[parent] = pre_cc; 
                dc[parent] += 1;

                // Update frontier
                if (pt == NODE_TYPE_DECISION){
                    if (pre_cc_local == 0 && frontier > 0) frontier -= 1; // parent leaves frontier
                    frontier += 1; // new decision becomes frontier
                } else {
                    frontier += 1; // root case
                }

                q[qt++] = slot; // Add to queue for future expansion
                budget -= 1;
                if (budget <= 0 || qt >= qcap) break;
            }
        }

        // Step 5: Finalize - attach ACTION to every DECISION leaf or convert to ACTION
        for (int qi = 0; qi < qt; ++qi){
            int p = q[qi];
            if (!valid_idx(p, N)) continue;
            if (ch[p] == 0){ // Leaf node
                const float* pb2 = node_ptr(tree, D, p);
                int pt2 = f2i(pb2[COL_NODE_TYPE]);
                if (pt2 != NODE_TYPE_DECISION) continue; // Only process DECISION leaves
                
                int d2 = f2i(pb2[COL_DEPTH]);
                if (d2 + 1 < max_depth){ // Depth allows child
                    int slot; 
                    if (alloc_slot(slot)){
                        float* nb2 = node_ptr(tree, D, slot);
                        nb2[COL_NODE_TYPE]  = i2f(NODE_TYPE_ACTION);
                        nb2[COL_PARENT_IDX] = i2f(p);
                        nb2[COL_DEPTH]      = i2f(d2 + 1);

                        int rbt = get_root_branch_type_make_pop(tree, D, N, p);
                        int at  = pick_action_for_root(rng, rbt,
                                                       long_actions, long_count,
                                                       hold_actions, hold_count,
                                                       short_actions, short_count);
                        nb2[COL_PARAM_1] = i2f(at);
                        
                        if (at == ACTION_NEW_LONG || at == ACTION_NEW_SHORT || at == ACTION_FLIP_POSITION){
                            nb2[COL_PARAM_2] = rng.nextf();
                            nb2[COL_PARAM_3] = i2f(1 + (int)(rng.nextf() * 100.0f));
                        } else if (at == ACTION_CLOSE_PARTIAL || at == ACTION_ADD_POSITION){
                            nb2[COL_PARAM_2] = rng.nextf();
                        }
                        
                        ch[p] = 1; 
                        ac[p] = 1;
                        if (frontier > 0) frontier -= 1;
                    } else {
                        // Out of slots - convert DECISION to ACTION directly
                        float* pb_mut = node_ptr(tree, D, p);
                        pb_mut[COL_NODE_TYPE] = i2f(NODE_TYPE_ACTION);
                        
                        int rbt = get_root_branch_type_make_pop(tree, D, N, p);
                        int at = pick_action_for_root(rng, rbt,
                                                     long_actions, long_count,
                                                     hold_actions, hold_count,
                                                     short_actions, short_count);
                        pb_mut[COL_PARAM_1] = i2f(at);
                        
                        if (at == ACTION_NEW_LONG || at == ACTION_NEW_SHORT || at == ACTION_FLIP_POSITION){
                            pb_mut[COL_PARAM_2] = rng.nextf();
                            pb_mut[COL_PARAM_3] = i2f(1 + (int)(rng.nextf() * 100.0f));
                        } else if (at == ACTION_CLOSE_PARTIAL || at == ACTION_ADD_POSITION){
                            pb_mut[COL_PARAM_2] = rng.nextf();
                        } else {
                            pb_mut[COL_PARAM_2] = i2f(0);
                            pb_mut[COL_PARAM_3] = i2f(0);
                        }
                        pb_mut[COL_PARAM_4] = i2f(0); // Clear decision-specific param
                        
                        if (frontier > 0) frontier -= 1;
                    }
                } else {
                    // Max depth reached - convert DECISION to ACTION directly
                    float* pb_mut = node_ptr(tree, D, p);
                    pb_mut[COL_NODE_TYPE] = i2f(NODE_TYPE_ACTION);
                    
                    int rbt = get_root_branch_type_make_pop(tree, D, N, p);
                    int at = pick_action_for_root(rng, rbt,
                                                 long_actions, long_count,
                                                 hold_actions, hold_count,
                                                 short_actions, short_count);
                    pb_mut[COL_PARAM_1] = i2f(at);
                    
                    if (at == ACTION_NEW_LONG || at == ACTION_NEW_SHORT || at == ACTION_FLIP_POSITION){
                        pb_mut[COL_PARAM_2] = rng.nextf();
                        pb_mut[COL_PARAM_3] = i2f(1 + (int)(rng.nextf() * 100.0f));
                    } else if (at == ACTION_CLOSE_PARTIAL || at == ACTION_ADD_POSITION){
                        pb_mut[COL_PARAM_2] = rng.nextf();
                    } else {
                        pb_mut[COL_PARAM_2] = i2f(0);
                        pb_mut[COL_PARAM_3] = i2f(0);
                    }
                    pb_mut[COL_PARAM_4] = i2f(0); // Clear decision-specific param
                    
                    if (frontier > 0) frontier -= 1;
                }
            }
        }

        // Step 6: Minimal tree guarantee - ensure all root branches have children
        // The validator expects all leaf nodes to be ACTION, so root branches cannot be leaves
        for (int root = 0; root < 3; ++root){
            if (ch[root] == 0){ // Root has no children
                int slot = -1;
                if (alloc_slot(slot)){
                    float* nb = node_ptr(tree, D, slot);
                    nb[COL_NODE_TYPE]  = i2f(NODE_TYPE_ACTION);
                    nb[COL_PARENT_IDX] = i2f(root);
                    nb[COL_DEPTH]      = i2f(1);  // Root depth + 1
                    nb[COL_PARAM_1]    = i2f(ACTION_CLOSE_ALL); // Safe fallback action
                    nb[COL_PARAM_2]    = i2f(0);
                    nb[COL_PARAM_3]    = i2f(0);
                    nb[COL_PARAM_4]    = i2f(0);
                    
                    ch[root] = 1; 
                    ac[root] = 1; // Update counters
                }
            }
        }
    }
}

void init_population_cuda(
    torch::Tensor trees,
    torch::Tensor total_budget,
    int max_children,
    int max_depth,
    int max_nodes,
    torch::Tensor bfs_q,
    torch::Tensor scratch,
    torch::Tensor child_cnt,
    torch::Tensor act_cnt,
    torch::Tensor dec_cnt,
    torch::Tensor cand_idx,
    torch::Tensor cand_w,
    torch::Tensor num_feat_indices,
    torch::Tensor num_feat_minmax,
    torch::Tensor bool_feat_indices,
    torch::Tensor ff_pairs,
    torch::Tensor long_actions,
    torch::Tensor hold_actions,
    torch::Tensor short_actions
){
    // Input validation
    TORCH_CHECK(trees.is_cuda() && trees.dtype() == torch::kFloat32 && trees.dim()==3 && trees.is_contiguous(),
                "trees must be (B,N,D) float32 CUDA contiguous");
    TORCH_CHECK(total_budget.is_cuda() && total_budget.dtype()==torch::kInt32 && total_budget.dim()==1,
                "total_budget (B,) int32 CUDA");
    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);

    // Host-side feasibility checks
    TORCH_CHECK(max_nodes == N, "max_nodes must equal trees.size(1)");
    TORCH_CHECK(max_nodes >= 6, "max_nodes must be >= 6 (need 3 roots + 3 actions for minimal tree)");
    TORCH_CHECK(max_depth >= 2, "max_depth must be >= 2 (root-action minimal tree)");
    TORCH_CHECK(max_children >= 1, "max_children must be >= 1 (otherwise no growth)");
    TORCH_CHECK(D == NODE_INFO_DIM, "Node dimension must match NODE_INFO_DIM");

    // Validate work buffer dimensions
    auto chk2 = [&](const torch::Tensor& t, int d0, int d1, const char* msg){
        TORCH_CHECK(t.is_cuda(), msg);
        TORCH_CHECK(t.size(0)==d0 && t.size(1)==d1, msg);
        TORCH_CHECK(t.dtype()==torch::kInt32, msg);
    };
    chk2(bfs_q,   B, 2*N, "bfs_q must be (B,2N) int32 CUDA");
    chk2(scratch, B, 2*N, "scratch must be (B,2N) int32 CUDA");
    chk2(child_cnt,B, N,  "child_cnt must be (B,N) int32 CUDA");
    chk2(act_cnt,  B, N,  "act_cnt must be (B,N) int32 CUDA");
    chk2(dec_cnt,  B, N,  "dec_cnt must be (B,N) int32 CUDA");
    chk2(cand_idx, B, N,  "cand_idx must be (B,N) int32 CUDA");
    TORCH_CHECK(cand_w.is_cuda() && cand_w.size(0)==B && cand_w.size(1)==N && cand_w.dtype()==torch::kFloat32,
                "cand_w must be (B,N) float32 CUDA");

    // Validate feature tables (allow zero-length)
    TORCH_CHECK(num_feat_indices.is_cuda() && num_feat_indices.dtype()==torch::kInt32 && num_feat_indices.dim()==1,
                "num_feat_indices (Kn,) int32 CUDA");
    TORCH_CHECK(num_feat_minmax.is_cuda() && num_feat_minmax.dtype()==torch::kFloat32 && num_feat_minmax.dim()==2 && num_feat_minmax.size(1)==2,
                "num_feat_minmax (Kn,2) float32 CUDA");
    TORCH_CHECK(num_feat_minmax.size(0) == num_feat_indices.size(0),
                "num_feat_minmax must align with num_feat_indices");

    TORCH_CHECK(bool_feat_indices.is_cuda() && bool_feat_indices.dtype()==torch::kInt32 && bool_feat_indices.dim()==1,
                "bool_feat_indices (Kb,) int32 CUDA");
    TORCH_CHECK(ff_pairs.is_cuda() && ff_pairs.dtype()==torch::kInt32 && ff_pairs.dim()==2 && ff_pairs.size(1)==2,
                "ff_pairs (P,2) int32 CUDA");

    // Validate action lists
    TORCH_CHECK(long_actions.is_cuda() && long_actions.dtype()==torch::kInt32 && long_actions.dim()==1,  "long_actions (La,) int32 CUDA");
    TORCH_CHECK(hold_actions.is_cuda() && hold_actions.dtype()==torch::kInt32 && hold_actions.dim()==1,  "hold_actions (Ha,) int32 CUDA");
    TORCH_CHECK(short_actions.is_cuda() && short_actions.dtype()==torch::kInt32 && short_actions.dim()==1,"short_actions (Sa,) int32 CUDA");

    // Launch kernel: one block per tree
    dim3 grid(B), block(128);
    k_init_population_batch<<<grid, block>>>(
        trees.data_ptr<float>(), B, N, D,
        total_budget.data_ptr<int>(),
        max_children, max_depth, max_nodes,
        bfs_q.data_ptr<int>(),
        scratch.data_ptr<int>(),
        child_cnt.data_ptr<int>(),
        act_cnt.data_ptr<int>(),
        dec_cnt.data_ptr<int>(),
        cand_idx.data_ptr<int>(),
        cand_w.data_ptr<float>(),
        num_feat_indices.data_ptr<int>(), (int)num_feat_indices.numel(),
        num_feat_minmax.data_ptr<float>(),
        bool_feat_indices.data_ptr<int>(), (int)bool_feat_indices.numel(),
        ff_pairs.data_ptr<int>(), (int)ff_pairs.size(0),
        long_actions.data_ptr<int>(),  (int)long_actions.numel(),
        hold_actions.data_ptr<int>(),  (int)hold_actions.numel(),
        short_actions.data_ptr<int>(), (int)short_actions.numel()
    );
    CUDA_CHECK_ERRORS();
}