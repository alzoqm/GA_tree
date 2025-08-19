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