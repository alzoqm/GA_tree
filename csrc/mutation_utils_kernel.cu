// csrc/mutation_utils_kernel.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "constants.h"
#include "mutation_utils_kernel.cuh"

#define CUDA_CHECK_ERRORS() \
  do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  }} while (0)

// Helper: read/write float-encoded ints safely
__device__ __forceinline__ int f2i(float x) { return static_cast<int>(x + 0.5f); }
__device__ __forceinline__ float i2f(int x) { return static_cast<float>(x); }

__device__ __forceinline__ bool valid_node_index(int idx, int max_nodes) {
    return idx >= 0 && idx < max_nodes;
}

// BFS over subtree rooted at r, collecting into result[]; returns count
__device__ int bfs_collect_subtree(
    const float* tree, int max_nodes, int node_dim,
    int r,
    int* q, int* res, int qcap
){
    if (!valid_node_index(r, max_nodes)) return 0;
    // skip UNUSED roots
    const float* root = tree + r*node_dim;
    if (f2i(root[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) return 0;

    int head = 0, tail = 0, outc = 0;
    if (qcap > 0) {
        q[tail++] = r;
        res[outc++] = r;
    }

    while (head < tail && tail < qcap && outc < qcap) {
        int cur = q[head++];
        // scan all nodes to find children (parent == cur)
        for (int j = 0; j < max_nodes; ++j) {
            const float* nb = tree + j*node_dim;
            if (f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) continue;
            if (f2i(nb[COL_PARENT_IDX]) == cur) {
                if (tail < qcap && outc < qcap) {
                    q[tail++] = j;
                    res[outc++] = j;
                }
            }
        }
    }
    return outc;
}

__global__ void k_find_subtree_nodes_batch(
    const float* __restrict__ trees, int B, int max_nodes, int node_dim,
    const int* __restrict__ root_indices,
    int* __restrict__ bfs_q,            // (B, 2*max_nodes)
    int* __restrict__ bfs_res,          // (B, 2*max_nodes)
    int* __restrict__ out_counts
){
    int b = blockIdx.x;
    if (b >= B) return;
    int r = root_indices[b];
    if (r < 0 || r >= max_nodes) { 
        out_counts[b] = 0; 
        return; 
    }

    const float* tree = trees + b*max_nodes*node_dim;
    int* q   = bfs_q  + b*(2*max_nodes);
    int* res = bfs_res+ b*(2*max_nodes);

    int cnt = bfs_collect_subtree(tree, max_nodes, node_dim, r, q, res, 2*max_nodes);
    out_counts[b] = cnt;
}

void find_subtree_nodes_batch_cuda(
    const torch::Tensor& trees, const torch::Tensor& root_indices, int max_nodes,
    torch::Tensor bfs_queue_buffer, torch::Tensor result_indices_buffer, torch::Tensor out_counts
){
    TORCH_CHECK(trees.is_cuda() && root_indices.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(bfs_queue_buffer.is_cuda() && result_indices_buffer.is_cuda() && out_counts.is_cuda(), "buffers must be CUDA");
    TORCH_CHECK(trees.scalar_type() == torch::kFloat32, "trees must be float32");
    TORCH_CHECK(root_indices.scalar_type() == torch::kInt32, "root_indices must be int32");
    TORCH_CHECK(bfs_queue_buffer.scalar_type() == torch::kInt32, "bfs queue must be int32");
    TORCH_CHECK(result_indices_buffer.scalar_type() == torch::kInt32, "bfs result must be int32");
    TORCH_CHECK(out_counts.scalar_type() == torch::kInt32, "out_counts must be int32");
    TORCH_CHECK(trees.dim() == 3, "trees must be (B,max_nodes,node_dim)");
    TORCH_CHECK(trees.is_contiguous(), "trees must be contiguous");

    int B = trees.size(0);
    int node_dim = trees.size(2);

    dim3 grid(B), block(1);
    k_find_subtree_nodes_batch<<<grid, block>>>(
        trees.data_ptr<float>(), B, max_nodes, node_dim,
        root_indices.data_ptr<int>(),
        bfs_queue_buffer.data_ptr<int>(),
        result_indices_buffer.data_ptr<int>(),
        out_counts.data_ptr<int>());
    CUDA_CHECK_ERRORS();
}

// Update depth += delta for each subtree root in batch
__global__ void k_update_subtree_depth_batch(
    float* __restrict__ trees, int B, int max_nodes, int node_dim,
    const int* __restrict__ root_indices,
    int delta,
    int* __restrict__ bfs_q,
    int* __restrict__ bfs_res
){
    int b = blockIdx.x;
    if (b >= B) return;
    int r = root_indices[b];
    if (r < 0 || r >= max_nodes) return;

    float* tree = trees + b*max_nodes*node_dim;
    int* q   = bfs_q  + b*(2*max_nodes);
    int* res = bfs_res+ b*(2*max_nodes);

    int cnt = bfs_collect_subtree(tree, max_nodes, node_dim, r, q, res, 2*max_nodes);
    for (int i = 0; i < cnt; ++i) {
        int idx = res[i];
        if (valid_node_index(idx, max_nodes)) {
            float* nb = tree + idx*node_dim;
            nb[COL_DEPTH] = nb[COL_DEPTH] + i2f(delta);
        }
    }
}

void update_subtree_depth_batch_cuda(
    torch::Tensor trees, const torch::Tensor& root_indices, int delta, int max_nodes,
    torch::Tensor bfs_queue_buffer, torch::Tensor result_indices_buffer
){
    TORCH_CHECK(trees.is_cuda() && root_indices.is_cuda(), "CUDA expected");
    TORCH_CHECK(bfs_queue_buffer.is_cuda() && result_indices_buffer.is_cuda(), "CUDA buffers expected");
    TORCH_CHECK(trees.scalar_type() == torch::kFloat32, "trees must be float32");
    TORCH_CHECK(root_indices.scalar_type() == torch::kInt32, "root_indices int32");
    TORCH_CHECK(trees.is_contiguous(), "trees must be contiguous");

    int B = trees.size(0);
    int node_dim = trees.size(2);

    dim3 grid(B), block(1);
    k_update_subtree_depth_batch<<<grid, block>>>(
        trees.data_ptr<float>(), B, max_nodes, node_dim,
        root_indices.data_ptr<int>(), delta,
        bfs_queue_buffer.data_ptr<int>(),
        result_indices_buffer.data_ptr<int>());
    CUDA_CHECK_ERRORS();
}

// Get max depth inside subtree
__global__ void k_get_subtree_max_depth_batch(
    const float* __restrict__ trees, int B, int max_nodes, int node_dim,
    const int* __restrict__ root_indices,
    int* __restrict__ bfs_q,
    int* __restrict__ bfs_res,
    int* __restrict__ out_max_depths
){
    int b = blockIdx.x;
    if (b >= B) return;
    int r = root_indices[b];
    if (r < 0 || r >= max_nodes) { 
        out_max_depths[b] = -2147483648; 
        return; 
    }

    const float* tree = trees + b*max_nodes*node_dim;
    int* q   = bfs_q  + b*(2*max_nodes);
    int* res = bfs_res+ b*(2*max_nodes);

    int cnt = bfs_collect_subtree(tree, max_nodes, node_dim, r, q, res, 2*max_nodes);
    int md = -2147483648;
    for (int i = 0; i < cnt; ++i) {
        int idx = res[i];
        if (valid_node_index(idx, max_nodes)) {
            const float* nb = tree + idx*node_dim;
            int d = f2i(nb[COL_DEPTH]);
            if (d > md) md = d;
        }
    }
    out_max_depths[b] = md;
}

void get_subtree_max_depth_batch_cuda(
    const torch::Tensor& trees, const torch::Tensor& root_indices, int max_nodes,
    torch::Tensor bfs_queue_buffer, torch::Tensor result_indices_buffer, torch::Tensor out_max_depths
){
    TORCH_CHECK(trees.is_cuda() && root_indices.is_cuda(), "CUDA expected");
    TORCH_CHECK(bfs_queue_buffer.is_cuda() && result_indices_buffer.is_cuda() && out_max_depths.is_cuda(), "CUDA buffers expected");
    TORCH_CHECK(trees.scalar_type() == torch::kFloat32, "trees must be float32");
    TORCH_CHECK(root_indices.scalar_type() == torch::kInt32, "root_indices int32");
    TORCH_CHECK(out_max_depths.scalar_type() == torch::kInt32, "out_max_depths int32");
    TORCH_CHECK(trees.is_contiguous(), "trees must be contiguous");

    int B = trees.size(0);
    int node_dim = trees.size(2);

    dim3 grid(B), block(1);
    k_get_subtree_max_depth_batch<<<grid, block>>>(
        trees.data_ptr<float>(), B, max_nodes, node_dim,
        root_indices.data_ptr<int>(),
        bfs_queue_buffer.data_ptr<int>(),
        result_indices_buffer.data_ptr<int>(),
        out_max_depths.data_ptr<int>());
    CUDA_CHECK_ERRORS();
}

// Find first `count` UNUSED slots per tree
__global__ void k_find_empty_slots_batch(
    const float* __restrict__ trees, int B, int max_nodes, int node_dim,
    int count,
    int* __restrict__ out_indices // (B,count)
){
    int b = blockIdx.x;
    if (b >= B) return;

    const float* tree = trees + b*max_nodes*node_dim;
    int found = 0;
    
    // Initialize all outputs to -1
    for (int j = 0; j < count; ++j) {
        out_indices[b*count + j] = -1;
    }
    
    // Find UNUSED slots
    for (int i = 0; i < max_nodes && found < count; ++i) {
        const float* nb = tree + i*node_dim;
        if (f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) {
            out_indices[b*count + found] = i;
            ++found;
        }
    }
}

void find_empty_slots_batch_cuda(
    const torch::Tensor& trees, int count, int max_nodes, torch::Tensor out_indices
){
    TORCH_CHECK(trees.is_cuda() && out_indices.is_cuda(), "CUDA expected");
    TORCH_CHECK(trees.scalar_type() == torch::kFloat32, "trees float32");
    TORCH_CHECK(out_indices.scalar_type() == torch::kInt32, "out_indices int32");
    TORCH_CHECK(trees.is_contiguous() && out_indices.is_contiguous(), "tensors must be contiguous");

    int B = trees.size(0);
    int node_dim = trees.size(2);
    TORCH_CHECK(out_indices.size(0) == B && out_indices.size(1) == count, "out_indices shape (B,count)");

    dim3 grid(B), block(1);
    k_find_empty_slots_batch<<<grid, block>>>(
        trees.data_ptr<float>(), B, max_nodes, node_dim,
        count,
        out_indices.data_ptr<int>());
    CUDA_CHECK_ERRORS();
}