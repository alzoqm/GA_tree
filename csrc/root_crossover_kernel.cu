// csrc/root_crossover_kernel.cu
// This file contains the CUDA implementation of the root branch crossover operations.

#include "root_crossover_kernel.cuh"
#include "crossover_utils.cuh"
#include "constants.h"
#include <curand_kernel.h>

// ==============================================================================
//       Kernel 3: Batch Copy Kernel for RootBranchCrossover
// ==============================================================================
__global__ void copy_branches_kernel(
    float* child_batch_ptr,
    const float* p1_batch_ptr,
    const float* p2_batch_ptr,
    const int* donor_map_ptr,
    int* bfs_queue_buffer_ptr,
    int* result_indices_buffer_ptr,
    int* old_to_new_map_buffer_ptr,
    int batch_size,
    int max_nodes)
{
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Initialize non-root nodes of the child buffer to UNUSED to prevent residual nodes.
    float* child_ptr_init = child_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    for (int i = 3; i < max_nodes; ++i) {
        float* nd = child_ptr_init + i * NODE_INFO_DIM;
        nd[COL_NODE_TYPE] = NODE_TYPE_UNUSED;
    }

    // --- 1. Set up thread-specific pointers and buffers ---
    float* child_ptr = child_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    const float* p1_ptr = p1_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    const float* p2_ptr = p2_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    
    int* my_queue = bfs_queue_buffer_ptr + batch_idx * max_nodes;
    int* my_results = result_indices_buffer_ptr + batch_idx * max_nodes;
    int* my_old_to_new_map = old_to_new_map_buffer_ptr + batch_idx * max_nodes;

    int child_next_idx = 3; 

    // --- 2. Sequentially copy the three branches (LONG, HOLD, SHORT) ---
    for (int b_idx = 0; b_idx < 3; ++b_idx) {
        int donor_choice = donor_map_ptr[batch_idx * 3 + b_idx];
        const float* donor_ptr = (donor_choice == 0) ? p1_ptr : p2_ptr;

        for (int donor_node_idx = 0; donor_node_idx < max_nodes; ++donor_node_idx) {
            if ((int)donor_ptr[donor_node_idx * NODE_INFO_DIM + COL_PARENT_IDX] == b_idx) {
                int subtree_root_idx = donor_node_idx;
                
                int subtree_size = find_subtree_nodes_device(donor_ptr, subtree_root_idx, my_queue, my_results, max_nodes);

                if (child_next_idx + subtree_size > max_nodes) {
                    continue; 
                }

                for(int i=0; i<max_nodes; ++i) my_old_to_new_map[i] = -1;
                for (int k = 0; k < subtree_size; ++k) {
                    my_old_to_new_map[my_results[k]] = child_next_idx + k;
                }

                float dest_parent_depth = child_ptr[b_idx * NODE_INFO_DIM + COL_DEPTH];
                float source_root_depth = donor_ptr[subtree_root_idx * NODE_INFO_DIM + COL_DEPTH];
                float depth_offset = (dest_parent_depth + 1) - source_root_depth;

                for (int k = 0; k < subtree_size; ++k) {
                    int old_idx = my_results[k];
                    int new_idx = my_old_to_new_map[old_idx];
                    
                    const float* src_node_data = donor_ptr + old_idx * NODE_INFO_DIM;
                    float* dest_node_data = child_ptr + new_idx * NODE_INFO_DIM;

                    for(int d=0; d<NODE_INFO_DIM; ++d) dest_node_data[d] = src_node_data[d];

                    dest_node_data[COL_DEPTH] += depth_offset;
                    
                    int old_parent_idx = (int)src_node_data[COL_PARENT_IDX];
                    if (old_idx == subtree_root_idx) {
                        dest_node_data[COL_PARENT_IDX] = (float)b_idx;
                    } else {
                        dest_node_data[COL_PARENT_IDX] = (float)my_old_to_new_map[old_parent_idx];
                    }
                }
                child_next_idx += subtree_size;
            }
        }
    }
}

// ==============================================================================
//                 Repair After RootBranch Crossover Kernel
// ==============================================================================

__device__ __forceinline__ int   rb_f2i(float x){ return __float2int_rn(x); }
__device__ __forceinline__ float rb_i2f(int   x){ return static_cast<float>(x); }
__device__ __forceinline__ bool  rb_valid_idx(int idx, int N){ return idx >= 0 && idx < N; }

__device__ __forceinline__ float* rb_node(float* base, int D, int i){ return base + (size_t)i * D; }
__device__ __forceinline__ const float* rb_node(const float* base, int D, int i){ return base + (size_t)i * D; }

__device__ int rb_bfs_collect_subtree(
    const float* tree, int N, int D, int root,
    int* q, int* res, int cap
){
    if (!rb_valid_idx(root, N)) return 0;
    const float* nb = rb_node(tree, D, root);
    if (rb_f2i(nb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) return 0;
    int head=0, tail=0, outc=0;
    if (cap > 0){ q[tail++] = root; res[outc++] = root; }
    while (head < tail && tail < cap && outc < cap){
        int cur = q[head++];
        for (int j = 0; j < N && tail < cap && outc < cap; ++j){
            const float* cb = rb_node(tree, D, j);
            if (rb_f2i(cb[COL_NODE_TYPE]) == NODE_TYPE_UNUSED) continue;
            if (rb_f2i(cb[COL_PARENT_IDX]) == cur){
                q[tail++] = j;
                res[outc++] = j;
            }
        }
    }
    return outc;
}

__device__ void rb_clear_subtree(
    float* tree, int N, int D, int root,
    int* q, int* res, int cap
){
    int cnt = rb_bfs_collect_subtree(tree, N, D, root, q, res, cap);
    for (int i = 0; i < cnt; ++i){
        int idx = res[i];
        if (!rb_valid_idx(idx, N)) continue;
        float* nd = rb_node(tree, D, idx);
        for (int k = 0; k < D; ++k) nd[k] = 0.0f;
        nd[COL_NODE_TYPE]  = rb_i2f(NODE_TYPE_UNUSED);
        nd[COL_PARENT_IDX] = rb_i2f(-1);
        nd[COL_DEPTH]      = 0.0f;
    }
}

__global__ void repair_after_root_branch_kernel(
    float* __restrict__ trees, int B, int N, int D,
    int* __restrict__ child_cnt, int* __restrict__ act_cnt, int* __restrict__ dec_cnt,
    int* __restrict__ bfs_q, int* __restrict__ bfs_res
){
    int b = blockIdx.x;
    if (b >= B) return;

    float* tree = trees + (size_t)b * N * D;
    int* ch = child_cnt + (size_t)b * N;
    int* ac = act_cnt   + (size_t)b * N;
    int* dc = dec_cnt   + (size_t)b * N;
    int* q  = bfs_q     + (size_t)b * (2 * N);
    int* rs = bfs_res   + (size_t)b * (2 * N);

    for (int i = threadIdx.x; i < N; i += blockDim.x){ ch[i] = 0; ac[i] = 0; dc[i] = 0; }
    __syncthreads();

    for (int j = threadIdx.x; j < N; j += blockDim.x){
        const float* nb = rb_node(tree, D, j);
        int t = rb_f2i(nb[COL_NODE_TYPE]);
        if (t == NODE_TYPE_UNUSED) continue;
        int p = rb_f2i(nb[COL_PARENT_IDX]);
        if (!rb_valid_idx(p, N)) continue;
        atomicAdd(&ch[p], 1);
        if (t == NODE_TYPE_ACTION)   atomicAdd(&ac[p], 1);
        if (t == NODE_TYPE_DECISION) atomicAdd(&dc[p], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0){
        for (int r = 0; r < 3; ++r){
            for (int j = 3; j < N; ++j){
                float* nd = rb_node(tree, D, j);
                if (rb_f2i(nd[COL_NODE_TYPE]) != NODE_TYPE_UNUSED && 
                    rb_f2i(nd[COL_PARENT_IDX]) == r){
                    rb_clear_subtree(tree, N, D, j, q, rs, 2*N);
                }
            }
            
            int action_slot = -1;
            for (int i = 3; i < N; ++i){
                float* nd = rb_node(tree, D, i);
                if (rb_f2i(nd[COL_NODE_TYPE]) == NODE_TYPE_UNUSED){
                    action_slot = i;
                    break;
                }
            }
            
            if (action_slot >= 0){
                float* nd = rb_node(tree, D, action_slot);
                nd[COL_NODE_TYPE]  = rb_i2f(NODE_TYPE_ACTION);
                nd[COL_PARENT_IDX] = rb_i2f(r);
                nd[COL_DEPTH]      = 1.0f;
                nd[COL_PARAM_1]    = rb_i2f(ACTION_CLOSE_ALL);
                for (int c = 4; c < D; ++c) nd[c] = 0.0f;
            }
        }
    }
    __syncthreads();
}

// ==============================================================================
//                Parent Pool Sampling Root Branch Crossover Kernel
// ==============================================================================

__global__ void root_branch_sampling_crossover_kernel(
    float* child_batch_ptr,
    const float* parent_pool_ptr,
    int batch_size,
    int pool_size,
    int max_nodes,
    int max_retries,
    int* bfs_queue_buffer_ptr,
    int* result_indices_buffer_ptr,
    int* old_to_new_map_buffer_ptr
)
{
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    float* child_ptr = child_batch_ptr + batch_idx * max_nodes * NODE_INFO_DIM;
    int* my_queue = bfs_queue_buffer_ptr + batch_idx * max_nodes;
    int* my_results = result_indices_buffer_ptr + batch_idx * max_nodes;
    int* my_old_to_new_map = old_to_new_map_buffer_ptr + batch_idx * max_nodes;

    for (int i = 3; i < max_nodes; ++i) {
        float* nd = child_ptr + i * NODE_INFO_DIM;
        nd[COL_NODE_TYPE] = NODE_TYPE_UNUSED;
        nd[COL_PARENT_IDX] = -1;
        nd[COL_DEPTH] = 0;
        for (int c = 3; c < NODE_INFO_DIM; ++c) nd[c] = 0.0f;
    }

    curandState state;
    curand_init(batch_idx, 0, 0, &state);

    int child_next_idx = 3;
    bool success = false;

    for (int retry = 0; retry < max_retries && !success; ++retry) {
        child_next_idx = 3;
        for (int i = 3; i < max_nodes; ++i) {
            child_ptr[i * NODE_INFO_DIM + COL_NODE_TYPE] = NODE_TYPE_UNUSED;
        }

        int branch_parents[3];
        int total_estimated_size = 3;
        bool size_ok = true;
        
        if (batch_idx == 0) {
        }

        for (int branch = 0; branch < 3; ++branch) {
            branch_parents[branch] = (int)(curand_uniform(&state) * pool_size);
            const float* parent_ptr = parent_pool_ptr + branch_parents[branch] * max_nodes * NODE_INFO_DIM;
            int branch_child_count = 0;
            for (int node_idx = 0; node_idx < max_nodes; ++node_idx) {
                int node_parent_idx = (int)parent_ptr[node_idx * NODE_INFO_DIM + COL_PARENT_IDX];
                if (node_parent_idx == branch) {
                    int subtree_size = find_subtree_nodes_device(parent_ptr, node_idx, my_queue, my_results, max_nodes);
                    total_estimated_size += subtree_size;
                    branch_child_count++;
                }
            }
        }

        if (total_estimated_size > max_nodes) {
            size_ok = false;
            if (batch_idx == 0) {
            }
            continue;
        }

        if (size_ok) {
            bool copy_success = true;
            for (int branch = 0; branch < 3 && copy_success; ++branch) {
                const float* donor_ptr = parent_pool_ptr + branch_parents[branch] * max_nodes * NODE_INFO_DIM;
                for (int donor_node_idx = 0; donor_node_idx < max_nodes && copy_success; ++donor_node_idx) {
                    if ((int)donor_ptr[donor_node_idx * NODE_INFO_DIM + COL_PARENT_IDX] == branch) {
                        int subtree_size = find_subtree_nodes_device(donor_ptr, donor_node_idx, my_queue, my_results, max_nodes);
                        if (child_next_idx + subtree_size > max_nodes) {
                            copy_success = false;
                            break;
                        }

                        for(int i = 0; i < max_nodes; ++i) my_old_to_new_map[i] = -1;
                        for (int k = 0; k < subtree_size; ++k) {
                            my_old_to_new_map[my_results[k]] = child_next_idx + k;
                        }

                        float dest_parent_depth = child_ptr[branch * NODE_INFO_DIM + COL_DEPTH];
                        float source_root_depth = donor_ptr[donor_node_idx * NODE_INFO_DIM + COL_DEPTH];
                        float depth_offset = (dest_parent_depth + 1) - source_root_depth;

                        for (int k = 0; k < subtree_size; ++k) {
                            int old_idx = my_results[k];
                            int new_idx = my_old_to_new_map[old_idx];
                            const float* src_node_data = donor_ptr + old_idx * NODE_INFO_DIM;
                            float* dest_node_data = child_ptr + new_idx * NODE_INFO_DIM;

                            for(int d = 0; d < NODE_INFO_DIM; ++d) dest_node_data[d] = src_node_data[d];

                            dest_node_data[COL_DEPTH] += depth_offset;
                            
                            int old_parent_idx = (int)src_node_data[COL_PARENT_IDX];
                            if (old_idx == donor_node_idx) {
                                dest_node_data[COL_PARENT_IDX] = (float)branch;
                            } else {
                                dest_node_data[COL_PARENT_IDX] = (float)my_old_to_new_map[old_parent_idx];
                            }
                        }
                        child_next_idx += subtree_size;
                    }
                }
            }
            if (copy_success) {
                success = true;
            }
        }
    }

    if (!success) {
        child_next_idx = 3;
        for (int branch = 0; branch < 3; ++branch) {
            if (child_next_idx < max_nodes) {
                float* action_node = child_ptr + child_next_idx * NODE_INFO_DIM;
                action_node[COL_NODE_TYPE] = (float)NODE_TYPE_ACTION;
                action_node[COL_PARENT_IDX] = (float)branch;
                action_node[COL_DEPTH] = 1.0f;
                action_node[COL_PARAM_1] = (float)ACTION_CLOSE_ALL;
                for (int c = 4; c < NODE_INFO_DIM; ++c) action_node[c] = 0.0f;
                child_next_idx++;
            } else {
            }
        }
    }
}

// ==============================================================================
//                       C++ Wrapper Functions (Kernel Launchers)
// ==============================================================================

void copy_branches_batch_cuda(
    torch::Tensor& child_batch,
    const torch::Tensor& p1_batch,
    const torch::Tensor& p2_batch,
    const torch::Tensor& donor_map,
    torch::Tensor& bfs_queue_buffer,
    torch::Tensor& result_indices_buffer,
    torch::Tensor& old_to_new_map_buffer
)
{
    const int batch_size = child_batch.size(0);
    if (batch_size == 0) return;
    const int max_nodes = child_batch.size(1);

    copy_branches_kernel<<<batch_size, 1>>>(
        child_batch.data_ptr<float>(),
        p1_batch.data_ptr<float>(),
        p2_batch.data_ptr<float>(),
        donor_map.data_ptr<int>(),
        bfs_queue_buffer.data_ptr<int>(),
        result_indices_buffer.data_ptr<int>(),
        old_to_new_map_buffer.data_ptr<int>(),
        batch_size,
        max_nodes
    );
    cudaDeviceSynchronize();
}

void repair_after_root_branch_cuda(
    torch::Tensor& trees,
    torch::Tensor& child_count_buffer,
    torch::Tensor& act_cnt_buffer,
    torch::Tensor& dec_cnt_buffer,
    torch::Tensor& bfs_queue_buffer,
    torch::Tensor& result_indices_buffer
){
    TORCH_CHECK(trees.is_cuda(), "trees must be CUDA");
    TORCH_CHECK(child_count_buffer.is_cuda() && act_cnt_buffer.is_cuda() && dec_cnt_buffer.is_cuda(), "count buffers must be CUDA");
    TORCH_CHECK(bfs_queue_buffer.is_cuda() && result_indices_buffer.is_cuda(), "BFS buffers must be CUDA");
    TORCH_CHECK(trees.scalar_type() == torch::kFloat32, "trees must be float32");
    TORCH_CHECK(child_count_buffer.scalar_type() == torch::kInt32, "child_count_buffer int32");
    TORCH_CHECK(act_cnt_buffer.scalar_type() == torch::kInt32, "act_cnt_buffer int32");
    TORCH_CHECK(dec_cnt_buffer.scalar_type() == torch::kInt32, "dec_cnt_buffer int32");
    TORCH_CHECK(bfs_queue_buffer.scalar_type() == torch::kInt32 && result_indices_buffer.scalar_type() == torch::kInt32, "BFS buffers int32");
    TORCH_CHECK(trees.dim()==3 && trees.is_contiguous(), "trees must be (B,N,D) contiguous");

    const int B = trees.size(0), N = trees.size(1), D = trees.size(2);
    TORCH_CHECK(child_count_buffer.size(0)==B && child_count_buffer.size(1)==N, "child_count_buffer shape");
    TORCH_CHECK(act_cnt_buffer.size(0)==B && act_cnt_buffer.size(1)==N, "act_cnt_buffer shape");
    TORCH_CHECK(dec_cnt_buffer.size(0)==B && dec_cnt_buffer.size(1)==N, "dec_cnt_buffer shape");
    TORCH_CHECK(bfs_queue_buffer.size(0)==B && bfs_queue_buffer.size(1)>=2*N, "bfs_queue_buffer shape (B,>=2N)");
    TORCH_CHECK(result_indices_buffer.size(0)==B && result_indices_buffer.size(1)>=2*N, "result_indices_buffer shape (B,>=2N)");

    dim3 grid(B), block(128);
    repair_after_root_branch_kernel<<<grid, block>>>(
        trees.data_ptr<float>(), B, N, D,
        child_count_buffer.data_ptr<int>(),
        act_cnt_buffer.data_ptr<int>(),
        dec_cnt_buffer.data_ptr<int>(),
        bfs_queue_buffer.data_ptr<int>(),
        result_indices_buffer.data_ptr<int>()
    );

    cudaDeviceSynchronize();
}

void root_branch_sampling_crossover_cuda(
    torch::Tensor& child_batch,
    const torch::Tensor& parent_pool,
    int max_nodes,
    int max_retries,
    torch::Tensor& bfs_queue_buffer,
    torch::Tensor& result_indices_buffer,
    torch::Tensor& old_to_new_map_buffer
)
{
    const int batch_size = child_batch.size(0);
    const int pool_size = parent_pool.size(0);
    if (batch_size == 0 || pool_size == 0) return;

    TORCH_CHECK(child_batch.is_cuda(), "child_batch must be CUDA");
    TORCH_CHECK(parent_pool.is_cuda(), "parent_pool must be CUDA");
    TORCH_CHECK(child_batch.scalar_type() == torch::kFloat32, "child_batch must be float32");
    TORCH_CHECK(parent_pool.scalar_type() == torch::kFloat32, "parent_pool must be float32");
    TORCH_CHECK(child_batch.dim() == 3 && child_batch.is_contiguous(), "child_batch must be (B,N,D) contiguous");
    TORCH_CHECK(parent_pool.dim() == 3 && parent_pool.is_contiguous(), "parent_pool must be (P,N,D) contiguous");
    
    TORCH_CHECK(bfs_queue_buffer.size(0) == batch_size && bfs_queue_buffer.size(1) >= max_nodes, 
                "bfs_queue_buffer shape mismatch");
    TORCH_CHECK(result_indices_buffer.size(0) == batch_size && result_indices_buffer.size(1) >= max_nodes,
                "result_indices_buffer shape mismatch");
    TORCH_CHECK(old_to_new_map_buffer.size(0) == batch_size && old_to_new_map_buffer.size(1) >= max_nodes,
                "old_to_new_map_buffer shape mismatch");

    root_branch_sampling_crossover_kernel<<<batch_size, 1>>>(
        child_batch.data_ptr<float>(),
        parent_pool.data_ptr<float>(),
        batch_size,
        pool_size,
        max_nodes,
        max_retries,
        bfs_queue_buffer.data_ptr<int>(),
        result_indices_buffer.data_ptr<int>(),
        old_to_new_map_buffer.data_ptr<int>()
    );
    cudaDeviceSynchronize();
}
