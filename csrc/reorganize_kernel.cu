// csrc/reorganize_kernel.cu
// This file contains the CUDA implementation for reorganizing the population by
// removing unused nodes and compacting the trees.

#include "reorganize_kernel.cuh"
#include "constants.h"
#include <algorithm> // for std::min

// Kernel 1: Computes the number of active nodes and remaps indices.
__global__ void compute_remap_indices_kernel(
    const float* population_ptr,
    int* active_counts_per_tree_ptr,
    int* old_gid_to_new_gid_map_ptr,
    int pop_size,
    int max_nodes)
{
    const int tree_idx = blockIdx.x;
    if (tree_idx >= pop_size) return;
    
    extern __shared__ int local_scan_buffer[];
    
    // Identify active nodes.
    for (int i = threadIdx.x; i < max_nodes; i += blockDim.x) {
        const int old_gid = tree_idx * max_nodes + i;
        local_scan_buffer[i] = (population_ptr[old_gid * NODE_INFO_DIM + COL_NODE_TYPE] != NODE_TYPE_UNUSED) ? 1 : 0;
    }
    __syncthreads();

    // Perform a Blelloch scan to compute the new indices.
    for (int offset = 1; offset < max_nodes; offset *= 2) {
        __syncthreads();
        int temp = 0;
        if (threadIdx.x >= offset) {
            temp = local_scan_buffer[threadIdx.x - offset];
        }
        __syncthreads();
        if (threadIdx.x >= offset) {
            local_scan_buffer[threadIdx.x] += temp;
        }
    }
    __syncthreads();
    
    // Create the remapping from old to new indices.
    for (int i = threadIdx.x; i < max_nodes; i += blockDim.x) {
        const int old_gid = tree_idx * max_nodes + i;
        if (population_ptr[old_gid * NODE_INFO_DIM + COL_NODE_TYPE] != NODE_TYPE_UNUSED) {
            int new_local_idx = local_scan_buffer[i] - 1;
            old_gid_to_new_gid_map_ptr[old_gid] = tree_idx * max_nodes + new_local_idx;
        } else {
            old_gid_to_new_gid_map_ptr[old_gid] = -1;
        }
    }
    
    if (threadIdx.x == 0 && max_nodes > 0) {
        active_counts_per_tree_ptr[tree_idx] = local_scan_buffer[max_nodes - 1];
    }
}

// Kernel 2: Reorganizes the nodes and updates parent indices.
__global__ void reorganize_and_update_kernel(
    const float* original_pop_ptr,
    const int* old_gid_to_new_gid_map_ptr,
    float* new_pop_ptr,
    int pop_size,
    int max_nodes)
{
    const int old_gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (old_gid >= pop_size * max_nodes) return;

    if (original_pop_ptr[old_gid * NODE_INFO_DIM + COL_NODE_TYPE] != NODE_TYPE_UNUSED) {
        const int new_gid = old_gid_to_new_gid_map_ptr[old_gid];
        if (new_gid == -1) return;
        
        // Copy node data to the new location.
        for (int i = 0; i < NODE_INFO_DIM; ++i) {
            new_pop_ptr[new_gid * NODE_INFO_DIM + i] = original_pop_ptr[old_gid * NODE_INFO_DIM + i];
        }

        // Update the parent index.
        int old_parent_local_idx = (int)original_pop_ptr[old_gid * NODE_INFO_DIM + COL_PARENT_IDX];
        if (old_parent_local_idx != -1) {
            int tree_idx = old_gid / max_nodes;
            int old_parent_gid = tree_idx * max_nodes + old_parent_local_idx;
            int new_parent_gid = old_gid_to_new_gid_map_ptr[old_parent_gid];
            
            if (new_parent_gid != -1) {
                int new_parent_local_idx = new_parent_gid % max_nodes;
                new_pop_ptr[new_gid * NODE_INFO_DIM + COL_PARENT_IDX] = (float)new_parent_local_idx;
            } else {
                 new_pop_ptr[new_gid * NODE_INFO_DIM + COL_PARENT_IDX] = -1.0f;
            }
        }
    }
}

// C++ wrapper function to launch the reorganization kernels.
void reorganize_population_cuda(torch::Tensor population_tensor) {
    TORCH_CHECK(population_tensor.is_cuda(), "Population tensor must be on a CUDA device for reorganize.");
    const int pop_size = population_tensor.size(0);
    const int max_nodes = population_tensor.size(1);
    const int total_nodes = pop_size * max_nodes;
    if (total_nodes == 0) return;

    auto int_options = torch::TensorOptions().device(population_tensor.device()).dtype(torch::kInt32);
    torch::Tensor active_counts_per_tree = torch::zeros({pop_size}, int_options);
    torch::Tensor old_gid_to_new_gid_map = torch::empty({total_nodes}, int_options);
    
    const int threads_per_block_scan = std::min(max_nodes, 1024);
    compute_remap_indices_kernel<<<pop_size, threads_per_block_scan, max_nodes * sizeof(int)>>>(
        population_tensor.data_ptr<float>(), active_counts_per_tree.data_ptr<int>(),
        old_gid_to_new_gid_map.data_ptr<int>(), pop_size, max_nodes
    );

    torch::Tensor new_population = torch::full_like(population_tensor, 0.0f);
    new_population.select(2, COL_NODE_TYPE).fill_(NODE_TYPE_UNUSED);

    const int threads_per_block_reorg = 256;
    const int num_blocks_reorg = (total_nodes + threads_per_block_reorg - 1) / threads_per_block_reorg;
    reorganize_and_update_kernel<<<num_blocks_reorg, threads_per_block_reorg>>>(
        population_tensor.data_ptr<float>(), old_gid_to_new_gid_map.data_ptr<int>(),
        new_population.data_ptr<float>(), pop_size, max_nodes
    );

    population_tensor.copy_(new_population);
}

// A version of the reorganization function that uses pre-allocated arrays.
void reorganize_population_with_arrays_cuda(
    torch::Tensor population_tensor,
    torch::Tensor active_counts_per_tree,
    torch::Tensor old_gid_to_new_gid_map) {
    
    TORCH_CHECK(population_tensor.is_cuda(), "Population tensor must be on a CUDA device for reorganize.");
    TORCH_CHECK(active_counts_per_tree.is_cuda(), "active_counts_per_tree tensor must be on a CUDA device.");
    TORCH_CHECK(old_gid_to_new_gid_map.is_cuda(), "old_gid_to_new_gid_map tensor must be on a CUDA device.");
    TORCH_CHECK(population_tensor.device() == active_counts_per_tree.device(), "Tensors must be on the same device");
    TORCH_CHECK(population_tensor.device() == old_gid_to_new_gid_map.device(), "Tensors must be on the same device");
    
    const int pop_size = population_tensor.size(0);
    const int max_nodes = population_tensor.size(1);
    const int total_nodes = pop_size * max_nodes;
    if (total_nodes == 0) return;

    TORCH_CHECK(active_counts_per_tree.size(0) == pop_size, "active_counts_per_tree size mismatch");
    TORCH_CHECK(old_gid_to_new_gid_map.size(0) == total_nodes, "old_gid_to_new_gid_map size mismatch");
    
    const int threads_per_block_scan = std::min(max_nodes, 1024);
    compute_remap_indices_kernel<<<pop_size, threads_per_block_scan, max_nodes * sizeof(int)>>>(
        population_tensor.data_ptr<float>(), 
        active_counts_per_tree.data_ptr<int>(),
        old_gid_to_new_gid_map.data_ptr<int>(), 
        pop_size, 
        max_nodes
    );

    cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA synchronization failed in compute_remap_indices_kernel: " + std::string(cudaGetErrorString(err)));

    torch::Tensor new_population = torch::full_like(population_tensor, 0.0f);
    new_population.select(2, COL_NODE_TYPE).fill_(NODE_TYPE_UNUSED);

    const int threads_per_block_reorg = 256;
    const int num_blocks_reorg = (total_nodes + threads_per_block_reorg - 1) / threads_per_block_reorg;
    reorganize_and_update_kernel<<<num_blocks_reorg, threads_per_block_reorg>>>(
        population_tensor.data_ptr<float>(), 
        old_gid_to_new_gid_map.data_ptr<int>(),
        new_population.data_ptr<float>(), 
        pop_size, 
        max_nodes
    );

    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA synchronization failed in reorganize_and_update_kernel: " + std::string(cudaGetErrorString(err)));

    population_tensor.copy_(new_population);
}