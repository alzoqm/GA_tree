// csrc/node_mutation_kernel.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <curand_kernel.h>
#include "constants.h"
#include "node_mutation_kernel.cuh"
#include "mutation_utils_kernel.cuh"

#define CUDA_CHECK_ERRORS() \
  do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  }} while (0)

__device__ __forceinline__ int f2i(float x) { return static_cast<int>(x + 0.5f); }
__device__ __forceinline__ float i2f(int x) { return static_cast<float>(x); }
__device__ __forceinline__ bool valid_idx(int x, int max_nodes){ return x>=0 && x<max_nodes; }

struct TreeView {
    float* base;      // pointer to first node
    int max_nodes;
    int node_dim;

    __device__ float* node(int idx) { return base + idx*node_dim; }
    __device__ const float* node_c(int idx) const { return base + idx*node_dim; }
};

// Check if parent has mixed ACTION and DECISION children
__device__ bool parent_has_mixed_types(const TreeView& tv, int parent) {
    bool has_action=false, has_decision=false;
    for (int j=0; j<tv.max_nodes; ++j) {
        const float* nb = tv.node_c(j);
        if (f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) continue;
        if (f2i(nb[COL_PARENT_IDX]) == parent) {
            int t = f2i(nb[COL_NODE_TYPE]);
            if (t == NODE_TYPE_DECISION) {
                has_decision = true;
            } else if (t == NODE_TYPE_ACTION) {
                has_action = true;
            }
        }
    }
    return has_action && has_decision;
}

// Count children of a parent
__device__ int child_count(const TreeView& tv, int parent) {
    int c=0;
    for (int j=0; j<tv.max_nodes; ++j) {
        const float* nb = tv.node_c(j);
        if (f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) continue;
        if (f2i(nb[COL_PARENT_IDX]) == parent) ++c;
    }
    return c;
}

// Check if parent has any ACTION children
__device__ bool parent_has_action_child(const TreeView& tv, int parent) {
    for (int j=0; j<tv.max_nodes; ++j) {
        const float* nb = tv.node_c(j);
        if (f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) continue;
        if (f2i(nb[COL_PARENT_IDX]) == parent) {
            int t = f2i(nb[COL_NODE_TYPE]);
            if (t == NODE_TYPE_ACTION) return true;
        }
    }
    return false;
}

// Find first unused slot
__device__ int first_unused_slot(const TreeView& tv) {
    for (int i=0;i<tv.max_nodes;++i) {
        if (f2i(tv.node_c(i)[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) return i;
    }
    return -1;
}

// Collect subtree nodes into res[], returning count
__device__ int bfs_collect(const TreeView& tv, int root, int* q, int* res, int cap) {
    if (!valid_idx(root, tv.max_nodes)) return 0;
    if (f2i(tv.node_c(root)[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) return 0;
    
    int head=0, tail=0, outc=0;
    if (cap > 0) {
        q[tail++] = root; 
        res[outc++] = root;
    }
    
    while (head<tail && tail<cap && outc<cap) {
        int cur=q[head++];
        for(int j=0;j<tv.max_nodes;++j){
            const float* nb = tv.node_c(j);
            if (f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) continue;
            if (f2i(nb[COL_PARENT_IDX]) == cur) {
                if (tail<cap && outc<cap) { 
                    q[tail++]=j; 
                    res[outc++]=j; 
                }
            }
        }
    }
    return outc;
}

// Get max depth in subtree
__device__ int subtree_max_depth(const TreeView& tv, int root, int* q, int* res, int cap) {
    int cnt = bfs_collect(tv, root, q, res, cap);
    int md = -2147483648;
    for (int i=0;i<cnt;++i) {
        int idx = res[i];
        if (valid_idx(idx, tv.max_nodes)) {
            int d = f2i(tv.node_c(idx)[COL_DEPTH]);
            if (d>md) md=d;
        }
    }
    return md;
}

// Collect valid candidates with depth weighting
__device__ int collect_weighted_candidates(
    const TreeView& tv, 
    int* candidates, 
    float* weights, 
    int max_candidates,
    int max_depth,
    int* q, int* res, int cap
) {
    int count = 0;
    
    for (int j = 0; j < tv.max_nodes && count < max_candidates; ++j) {
        const float* child_node = tv.node_c(j);
        int child_type = f2i(child_node[COL_NODE_TYPE]);
        
        // Skip UNUSED and ROOT_BRANCH nodes
        if (child_type == NODE_TYPE_UNUSED || child_type == NODE_TYPE_ROOT_BRANCH) continue;
        
        int parent = f2i(child_node[COL_PARENT_IDX]);
        if (!valid_idx(parent, tv.max_nodes)) continue;
        
        // Invariant checks
        if (parent_has_mixed_types(tv, parent)) continue;
        
        if (parent_has_action_child(tv, parent)) {
            int cc = child_count(tv, parent);
            if (cc != 1) continue;
        }
        
        // Depth check
        int md = subtree_max_depth(tv, j, q, res, cap);
        if (md + 1 >= max_depth) continue;
        
        // Add to candidates
        candidates[count] = j;
        
        // Weight by inverse of parent depth (prefer shallow insertions)
        const float* parent_node = tv.node_c(parent);
        int parent_depth = f2i(parent_node[COL_DEPTH]);
        weights[count] = 1.0f / (parent_depth + 1.0f);
        
        count++;
    }
    
    return count;
}

// Simple weighted selection using linear search
__device__ int weighted_select(float* weights, int count, float rand_val) {
    if (count == 0) return -1;
    
    // Calculate total weight
    float total_weight = 0.0f;
    for (int i = 0; i < count; ++i) {
        total_weight += weights[i];
    }
    
    if (total_weight <= 0.0f) {
        // Fallback to uniform selection
        return static_cast<int>(rand_val * count) % count;
    }
    
    // Select based on weight
    float target = rand_val * total_weight;
    float cumulative = 0.0f;
    
    for (int i = 0; i < count; ++i) {
        cumulative += weights[i];
        if (target <= cumulative) {
            return i;
        }
    }
    
    return count - 1; // fallback
}

__global__ void k_add_decision_nodes_batch(
    float* __restrict__ trees, int B, int max_nodes, int node_dim,
    const int* __restrict__ num_to_add, int max_depth, int max_add_nodes,
    int* __restrict__ out_new_node_indices,
    int* __restrict__ bfs_queue_buffer,
    int* __restrict__ result_indices_buffer,
    int* __restrict__ old_to_new_map_buffer // reserved; not used in this implementation
){
    int b = blockIdx.x;
    if (b >= B) return;

    TreeView tv{ trees + b*max_nodes*node_dim, max_nodes, node_dim };

    int* q   = bfs_queue_buffer      + b*(2*max_nodes);
    int* res = result_indices_buffer + b*(2*max_nodes);

    int K = num_to_add[b];
    if (K <= 0) return;

    // Initialize random state using global thread ID
    curandState rand_state;
    curand_init(blockIdx.x * blockDim.x + threadIdx.x, 0, 0, &rand_state);

    int inserted = 0;
    const int MAX_CANDIDATES = 64; // Reasonable limit for stack allocation
    int candidates[MAX_CANDIDATES];
    float weights[MAX_CANDIDATES];

    // Try up to max_nodes attempts to find valid candidates
    for (int attempt = 0; attempt < tv.max_nodes && inserted < K; ++attempt) {
        // Collect weighted candidates
        int candidate_count = collect_weighted_candidates(
            tv, candidates, weights, MAX_CANDIDATES, max_depth, q, res, 2*max_nodes
        );
        
        if (candidate_count == 0) break; // No valid candidates
        
        // Select candidate using weighted sampling
        float rand_val = curand_uniform(&rand_state);
        int selected_idx = weighted_select(weights, candidate_count, rand_val);
        
        if (selected_idx < 0 || selected_idx >= candidate_count) continue;
        
        int child = candidates[selected_idx];
        if (!valid_idx(child, tv.max_nodes)) continue;
        
        const float* child_node = tv.node_c(child);
        int parent = f2i(child_node[COL_PARENT_IDX]);
        if (!valid_idx(parent, tv.max_nodes)) continue;

        // Allocate new node slot
        int new_idx = first_unused_slot(tv);
        if (!valid_idx(new_idx, tv.max_nodes)) {
            break; // No space
        }

        float* newn = tv.node(new_idx);
        const float* par = tv.node_c(parent);

        // Initialize NEW_DECISION
        newn[COL_NODE_TYPE]  = i2f(NODE_TYPE_DECISION);
        newn[COL_PARENT_IDX] = i2f(parent);
        newn[COL_DEPTH]      = i2f(f2i(par[COL_DEPTH]) + 1);
        // Params will be filled later in Python

        // Rewire child -> new parent
        float* chm = tv.node(child);
        chm[COL_PARENT_IDX] = i2f(new_idx);

        // Depth += 1 for entire subtree of child
        int cnt2 = bfs_collect(tv, child, q, res, 2*max_nodes);
        for (int i=0;i<cnt2;++i) {
            int idx = res[i];
            if (valid_idx(idx, tv.max_nodes)) {
                float* nb = tv.node(idx);
                nb[COL_DEPTH] = nb[COL_DEPTH] + i2f(1);
            }
        }

        // Record output index
        if (inserted < max_add_nodes) {
            out_new_node_indices[b * max_add_nodes + inserted] = new_idx;
        }

        ++inserted;
        if (inserted >= K) break;
    }
}

void add_decision_nodes_batch_cuda(
    torch::Tensor trees,
    torch::Tensor num_to_add,
    int max_depth,
    int max_nodes,
    int max_add_nodes,
    torch::Tensor out_new_node_indices,
    torch::Tensor bfs_queue_buffer,
    torch::Tensor result_indices_buffer,
    torch::Tensor old_to_new_map_buffer
){
    TORCH_CHECK(trees.is_cuda() && num_to_add.is_cuda(), "CUDA expected");
    TORCH_CHECK(out_new_node_indices.is_cuda(), "out_new_node_indices must be CUDA");
    TORCH_CHECK(bfs_queue_buffer.is_cuda() && result_indices_buffer.is_cuda() && old_to_new_map_buffer.is_cuda(), "CUDA buffers expected");

    TORCH_CHECK(trees.scalar_type() == torch::kFloat32, "trees must be float32");
    TORCH_CHECK(num_to_add.scalar_type() == torch::kInt32, "num_to_add must be int32");
    TORCH_CHECK(out_new_node_indices.scalar_type() == torch::kInt32, "out_new_node_indices must be int32");
    TORCH_CHECK(bfs_queue_buffer.scalar_type() == torch::kInt32, "bfs_queue_buffer must be int32");
    TORCH_CHECK(result_indices_buffer.scalar_type() == torch::kInt32, "result_indices_buffer must be int32");
    TORCH_CHECK(old_to_new_map_buffer.scalar_type() == torch::kInt32, "old_to_new_map_buffer must be int32");

    TORCH_CHECK(trees.dim() == 3, "trees must be (B,max_nodes,node_dim)");
    TORCH_CHECK(trees.is_contiguous(), "trees must be contiguous");
    TORCH_CHECK(out_new_node_indices.is_contiguous(), "out_new_node_indices must be contiguous");
    
    int B = trees.size(0);
    int node_dim = trees.size(2);
    
    TORCH_CHECK(out_new_node_indices.size(0) == B, "out_new_node_indices batch size mismatch");
    TORCH_CHECK(out_new_node_indices.size(1) == max_add_nodes, "out_new_node_indices max_add_nodes mismatch");

    // Initialize outputs with -1
    out_new_node_indices.fill_(-1);

    dim3 grid(B), block(1);
    k_add_decision_nodes_batch<<<grid, block>>>(
        trees.data_ptr<float>(), B, max_nodes, node_dim,
        num_to_add.data_ptr<int>(), max_depth, max_add_nodes,
        out_new_node_indices.data_ptr<int>(),
        bfs_queue_buffer.data_ptr<int>(),
        result_indices_buffer.data_ptr<int>(),
        old_to_new_map_buffer.data_ptr<int>());

    CUDA_CHECK_ERRORS();
}

// --------------------------- DELETE NODE IMPLEMENTATION ---------------------------

// Utility: zero buffers
__device__ inline void zero_int_buffer(int* buf, int n) {
    for (int i = 0; i < n; ++i) buf[i] = 0;
}

__device__ inline void fill_int_buffer(int* buf, int n, int v) {
    for (int i = 0; i < n; ++i) buf[i] = v;
}

__device__ inline void fill_float_buffer(float* buf, int n, float v) {
    for (int i = 0; i < n; ++i) buf[i] = v;
}

// Build child/type counts
__device__ void build_child_and_type_counts(
    const TreeView& tv,
    int* child_cnt,  // size N
    int* act_cnt,    // size N
    int* dec_cnt     // size N
){
    zero_int_buffer(child_cnt, tv.max_nodes);
    zero_int_buffer(act_cnt, tv.max_nodes);
    zero_int_buffer(dec_cnt, tv.max_nodes);

    for (int j = 0; j < tv.max_nodes; ++j) {
        const float* nb = tv.node_c(j);
        int nt = f2i(nb[COL_NODE_TYPE]);
        if (nt == NODE_TYPE_UNUSED) continue;

        int p = f2i(nb[COL_PARENT_IDX]);
        if (!valid_idx(p, tv.max_nodes)) continue;

        // increment counts for this parent
        child_cnt[p] += 1;
        if (nt == NODE_TYPE_ACTION) {
            act_cnt[p] += 1;
        } else if (nt == NODE_TYPE_DECISION) {
            dec_cnt[p] += 1;
        }
    }
}

// Candidate collection
__device__ int collect_delete_candidates(
    const TreeView& tv,
    const int* child_cnt,     // N
    const int* act_cnt,       // N
    const int* dec_cnt,       // N
    int max_children,
    int* cand_idx_out,        // N
    float* cand_w_out         // N
){
    int count = 0;

    for (int u = 0; u < tv.max_nodes; ++u) {
        const float* un = tv.node_c(u);
        int ut = f2i(un[COL_NODE_TYPE]);
        if (ut != NODE_TYPE_DECISION) continue;           // only DECISION nodes deletable

        int p = f2i(un[COL_PARENT_IDX]);
        if (!valid_idx(p, tv.max_nodes)) continue;        // must have a parent

        const float* pn = tv.node_c(p);
        int pt = f2i(pn[COL_NODE_TYPE]);

        int u_childs = child_cnt[u];
        int p_childs = child_cnt[p];
        int u_act    = act_cnt[u];
        int u_dec    = dec_cnt[u];
        int p_act    = act_cnt[p];

        // Guard 1: no orphan parent / no non-ACTION leaf creation
        if (u_childs == 0 && p_childs == 1) {
            // deleting u would make p a leaf; p must be ACTION, but it's ROOT/DECISION
            continue;
        }

        // Guard 2a: prevent action/decision mixing after deletion
        if (u_dec > 0 && p_act > 0) {
            // moving decision-children under a parent that already has action-children -> mix
            continue;
        }
        // Guard 2b: if u has an action child, parent must end with exactly one child
        if (u_act > 0 && p_childs > 1) {
            // parent would have an action child but >1 children -> violates finality
            continue;
        }

        // Guard 3: max_children after reattach
        int future_children = p_childs - 1 + u_childs;
        if (future_children > max_children) continue;

        // Guard 4: root orphan handled by Guard 1 (p_childs==1 & u_childs==0)

        // Passed all guards — add candidate
        cand_idx_out[count] = u;
        // Score: prefer deleting hubs (more children)
        cand_w_out[count]   = static_cast<float>(u_childs) + 1e-6f;
        ++count;
    }

    return count;
}

// Weighted sample from candidates
__device__ int weighted_pick(
    const float* weights, int n, float rnd01
){
    if (n <= 0) return -1;
    float total = 0.0f;
    for (int i = 0; i < n; ++i) total += weights[i];
    if (total <= 0.0f) {
        // uniform fallback
        int k = static_cast<int>(rnd01 * n);
        if (k >= n) k = n - 1;
        return k;
    }
    float target = rnd01 * total;
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += weights[i];
        if (target <= acc) return i;
    }
    return n - 1;
}

// Rewire + depth update
__device__ void rewire_children_and_lift_depth(
    TreeView& tv,
    int u,      // node to delete
    int p,      // its parent
    int* q,     // (2N) BFS queue
    int* res    // (2N) BFS output
){
    // For every child c of u: set parent[c] = p, and depth(subtree(c)) -= 1
    for (int j = 0; j < tv.max_nodes; ++j) {
        float* nb = tv.node(j);
        int nt = f2i(nb[COL_NODE_TYPE]);
        if (nt == NODE_TYPE_UNUSED) continue;
        if (f2i(nb[COL_PARENT_IDX]) == u) {
            // reattach c -> p
            nb[COL_PARENT_IDX] = i2f(p);

            // BFS subtree of c, decrement depths by 1
            int cnt = bfs_collect(tv, j, q, res, 2 * tv.max_nodes);
            for (int k = 0; k < cnt; ++k) {
                int idx = res[k];
                if (!valid_idx(idx, tv.max_nodes)) continue;
                float* rb = tv.node(idx);
                rb[COL_DEPTH] = rb[COL_DEPTH] - 1.0f;
            }
        }
    }

    // Finally, clear u
    float* un = tv.node(u);
    for (int c = 0; c < tv.node_dim; ++c) un[c] = 0.0f;
    un[COL_NODE_TYPE]  = i2f(NODE_TYPE_UNUSED);
    un[COL_PARENT_IDX] = i2f(-1);
    un[COL_DEPTH]      = i2f(0);
}

// Main kernel per batch item
__global__ void k_delete_nodes_batch(
    float* __restrict__ trees, int B, int max_nodes, int node_dim,
    const int* __restrict__ num_to_delete,
    int max_children, int max_depth, int max_delete_nodes,
    int* __restrict__ deleted_nodes,            // (B, K)
    int* __restrict__ bfs_queue_buffer,         // (B, 2N)
    int* __restrict__ result_indices_buffer,    // (B, 2N)
    int* __restrict__ child_count_buffer,       // (B, N)
    int* __restrict__ act_cnt_buffer,           // (B, N)
    int* __restrict__ dec_cnt_buffer,           // (B, N)
    int* __restrict__ candidate_indices_buffer, // (B, N)
    float* __restrict__ candidate_weights_buffer// (B, N)
){
    int b = blockIdx.x;
    if (b >= B) return;

    // Pointers to this tree's slices
    TreeView tv{ trees + b*max_nodes*node_dim, max_nodes, node_dim };

    int* q    = bfs_queue_buffer         + b*(2*max_nodes);
    int* res  = result_indices_buffer    + b*(2*max_nodes);
    int* ch   = child_count_buffer       + b*max_nodes;
    int* ac   = act_cnt_buffer           + b*max_nodes;
    int* dc   = dec_cnt_buffer           + b*max_nodes;
    int* cidx = candidate_indices_buffer + b*max_nodes;
    float* cw = candidate_weights_buffer + b*max_nodes;

    // Initialize outputs to -1 (for this batch row)
    for (int t = 0; t < max_delete_nodes; ++t) {
        deleted_nodes[b*max_delete_nodes + t] = -1;
    }

    int K = num_to_delete[b];
    if (K <= 0) return;

    // curand state per block/thread
    curandState rng;
    curand_init(0xC0FFEEu + b, 0, 0, &rng);

    int deleted = 0;
    // Try up to K deletions
    for (int step = 0; step < K; ++step) {
        // Recompute counts (cheap O(N))
        build_child_and_type_counts(tv, ch, ac, dc);

        // Collect valid candidates under invariants
        int cand_n = collect_delete_candidates(tv, ch, ac, dc, max_children, cidx, cw);
        if (cand_n <= 0) break;

        // Weighted sampling (no replacement by recomputing next round)
        int pick_pos = weighted_pick(cw, cand_n, curand_uniform(&rng));
        if (pick_pos < 0 || pick_pos >= cand_n) break;

        int u = cidx[pick_pos];
        if (!valid_idx(u, tv.max_nodes)) break;

        const float* un = tv.node_c(u);
        int p = f2i(un[COL_PARENT_IDX]);
        if (!valid_idx(p, tv.max_nodes)) break;

        // Rewire and lift subtree depths
        rewire_children_and_lift_depth(tv, u, p, q, res);

        // Record deletion
        deleted_nodes[b*max_delete_nodes + deleted] = u;
        ++deleted;
        if (deleted >= K) break;
    }
}

// Host wrapper
void delete_nodes_batch_cuda(
    torch::Tensor trees,
    torch::Tensor num_to_delete,
    int max_children,
    int /*max_depth*/,   // kept for symmetry; not needed by current guards
    int max_nodes,
    int max_delete_nodes,
    torch::Tensor deleted_nodes,
    torch::Tensor bfs_queue_buffer,
    torch::Tensor result_indices_buffer,
    torch::Tensor child_count_buffer,
    torch::Tensor act_cnt_buffer,
    torch::Tensor dec_cnt_buffer,
    torch::Tensor candidate_indices_buffer,
    torch::Tensor candidate_weights_buffer
){
    TORCH_CHECK(trees.is_cuda(), "trees must be CUDA");
    TORCH_CHECK(num_to_delete.is_cuda(), "num_to_delete must be CUDA");
    TORCH_CHECK(deleted_nodes.is_cuda(), "deleted_nodes must be CUDA");
    TORCH_CHECK(bfs_queue_buffer.is_cuda() && result_indices_buffer.is_cuda(), "BFS buffers must be CUDA");
    TORCH_CHECK(child_count_buffer.is_cuda() && act_cnt_buffer.is_cuda() && dec_cnt_buffer.is_cuda(), "count buffers must be CUDA");
    TORCH_CHECK(candidate_indices_buffer.is_cuda() && candidate_weights_buffer.is_cuda(), "candidate buffers must be CUDA");

    TORCH_CHECK(trees.scalar_type() == torch::kFloat32, "trees must be float32");
    TORCH_CHECK(num_to_delete.scalar_type() == torch::kInt32, "num_to_delete must be int32");
    TORCH_CHECK(deleted_nodes.scalar_type() == torch::kInt32, "deleted_nodes must be int32");
    TORCH_CHECK(bfs_queue_buffer.scalar_type() == torch::kInt32, "bfs_queue_buffer must be int32");
    TORCH_CHECK(result_indices_buffer.scalar_type() == torch::kInt32, "result_indices_buffer must be int32");
    TORCH_CHECK(child_count_buffer.scalar_type() == torch::kInt32, "child_count_buffer must be int32");
    TORCH_CHECK(act_cnt_buffer.scalar_type() == torch::kInt32, "act_cnt_buffer must be int32");
    TORCH_CHECK(dec_cnt_buffer.scalar_type() == torch::kInt32, "dec_cnt_buffer must be int32");
    TORCH_CHECK(candidate_indices_buffer.scalar_type() == torch::kInt32, "candidate_indices_buffer must be int32");
    TORCH_CHECK(candidate_weights_buffer.scalar_type() == torch::kFloat32, "candidate_weights_buffer must be float32");

    TORCH_CHECK(trees.dim() == 3, "trees must be (B, N, D)");
    TORCH_CHECK(trees.is_contiguous(), "trees must be contiguous");

    int B = trees.size(0);
    int N = trees.size(1);
    int D = trees.size(2);

    TORCH_CHECK(N == max_nodes, "max_nodes must match trees.size(1)");
    TORCH_CHECK(deleted_nodes.size(0) == B && deleted_nodes.size(1) == max_delete_nodes, "deleted_nodes shape (B, K)");
    TORCH_CHECK(bfs_queue_buffer.size(0) == B && bfs_queue_buffer.size(1) == 2 * max_nodes, "bfs_queue_buffer shape (B, 2N)");
    TORCH_CHECK(result_indices_buffer.size(0) == B && result_indices_buffer.size(1) == 2 * max_nodes, "result_indices_buffer shape (B, 2N)");
    TORCH_CHECK(child_count_buffer.size(0) == B && child_count_buffer.size(1) == max_nodes, "child_count_buffer shape (B, N)");
    TORCH_CHECK(act_cnt_buffer.size(0) == B && act_cnt_buffer.size(1) == max_nodes, "act_cnt_buffer shape (B, N)");
    TORCH_CHECK(dec_cnt_buffer.size(0) == B && dec_cnt_buffer.size(1) == max_nodes, "dec_cnt_buffer shape (B, N)");
    TORCH_CHECK(candidate_indices_buffer.size(0) == B && candidate_indices_buffer.size(1) == max_nodes, "candidate_indices_buffer shape (B, N)");
    TORCH_CHECK(candidate_weights_buffer.size(0) == B && candidate_weights_buffer.size(1) == max_nodes, "candidate_weights_buffer shape (B, N)");

    // Initialize outputs to -1 safely on device
    deleted_nodes.fill_(-1);

    dim3 grid(B), block(1);
    k_delete_nodes_batch<<<grid, block>>>(
        trees.data_ptr<float>(), B, N, D,
        num_to_delete.data_ptr<int>(),
        max_children, /*max_depth*/0, max_delete_nodes,
        deleted_nodes.data_ptr<int>(),
        bfs_queue_buffer.data_ptr<int>(),
        result_indices_buffer.data_ptr<int>(),
        child_count_buffer.data_ptr<int>(),
        act_cnt_buffer.data_ptr<int>(),
        dec_cnt_buffer.data_ptr<int>(),
        candidate_indices_buffer.data_ptr<int>(),
        candidate_weights_buffer.data_ptr<float>());

    CUDA_CHECK_ERRORS();
}