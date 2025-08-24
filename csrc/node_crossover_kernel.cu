// csrc/node_crossover_kernel.cu
#include "node_crossover_kernel.cuh"
#include "constants.h"
#include <curand_kernel.h>

// ==============================================================================
//           커널 1: 컨텍스트 마스크 생성 (Contextual Mask Generation)
// ==============================================================================
__global__ void get_contextual_mask_kernel(
    const float* trees_ptr,
    bool* output_mask_ptr,
    int batch_size,
    int max_nodes,
    int node_type_target,
    int branch_type_target)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_nodes = batch_size * max_nodes;
    if (gid >= total_nodes) return;

    const int batch_idx = gid / max_nodes;
    const int node_idx = gid % max_nodes;

    const float* node_data = trees_ptr + gid * NODE_INFO_DIM;
    
    // 1. 노드 타입이 일치하는지 확인
    if ((int)node_data[COL_NODE_TYPE] == node_type_target) {
        
        // 2. 루트까지 부모를 거슬러 올라가기
        int current_idx = node_idx;
        int parent_idx = (int)node_data[COL_PARENT_IDX];

        while (parent_idx != -1) {
            current_idx = parent_idx;
            parent_idx = (int)trees_ptr[(batch_idx * max_nodes + current_idx) * NODE_INFO_DIM + COL_PARENT_IDX];
        }

        // 3. 루트 분기 타입이 일치하는지 확인
        const float* root_data = trees_ptr + (batch_idx * max_nodes + current_idx) * NODE_INFO_DIM;
        if ((int)root_data[COL_PARAM_1] == branch_type_target) {
            output_mask_ptr[gid] = true;
        }
    }
}

// ==============================================================================
//         커널 2: 노드 파라미터 교환 (Node Parameter Swap)
// ==============================================================================
__global__ void swap_node_params_kernel(
    float* c1_ptr,
    float* c2_ptr,
    const bool* p1_mask_ptr,
    const bool* p2_mask_ptr,
    int batch_size,
    int max_nodes)
{
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // --- 1. 교환 가능한 노드 수 계산 ---
    int p1_count = 0;
    int p2_count = 0;
    for (int i = 0; i < max_nodes; ++i) {
        if (p1_mask_ptr[batch_idx * max_nodes + i]) p1_count++;
        if (p2_mask_ptr[batch_idx * max_nodes + i]) p2_count++;
    }

    int max_swaps = min(p1_count, p2_count);
    if (max_swaps == 0) return;

    // --- 2. 교환할 개수 'k' 결정 ---
    curandState state;
    curand_init(batch_idx, 0, 0, &state);
    int k_upper = max(1, max_swaps / 2);
    int k = (int)(curand_uniform(&state) * k_upper) + 1;
    k = min(k, max_swaps);

    // --- 3. 교환 수행 (단일 스레드 내 루프) ---
    for (int i = 0; i < k; ++i) {
        // p1에서 랜덤 인덱스 선택
        int p1_rand_n = (int)(curand_uniform(&state) * p1_count);
        int p1_swap_idx = -1;
        int current_n = 0;
        for (int j = 0; j < max_nodes; ++j) {
            if (p1_mask_ptr[batch_idx * max_nodes + j]) {
                if (current_n == p1_rand_n) {
                    p1_swap_idx = j;
                    break;
                }
                current_n++;
            }
        }

        // p2에서 랜덤 인덱스 선택
        int p2_rand_n = (int)(curand_uniform(&state) * p2_count);
        int p2_swap_idx = -1;
        current_n = 0;
        for (int j = 0; j < max_nodes; ++j) {
            if (p2_mask_ptr[batch_idx * max_nodes + j]) {
                if (current_n == p2_rand_n) {
                    p2_swap_idx = j;
                    break;
                }
                current_n++;
            }
        }

        if (p1_swap_idx != -1 && p2_swap_idx != -1) {
            // 파라미터(COL_PARAM_1 ~ 끝) 교환
            for (int param_col = COL_PARAM_1; param_col < NODE_INFO_DIM; ++param_col) {
                float* p1_addr = c1_ptr + (batch_idx * max_nodes + p1_swap_idx) * NODE_INFO_DIM + param_col;
                float* p2_addr = c2_ptr + (batch_idx * max_nodes + p2_swap_idx) * NODE_INFO_DIM + param_col;
                
                float temp = *p1_addr;
                *p1_addr = *p2_addr;
                *p2_addr = temp;
            }
        }
    }
}

// ==============================================================================
//                       C++ 래퍼 함수 (커널 런처)
// ==============================================================================
void get_contextual_mask_cuda(const torch::Tensor& trees, torch::Tensor& output_mask, int node_type, int branch_type) {
    const int batch_size = trees.size(0);
    const int max_nodes = trees.size(1);
    const int total_nodes = batch_size * max_nodes;
    if (total_nodes == 0) return;

    const int threads = 256;
    const int blocks = (total_nodes + threads - 1) / threads;
    
    get_contextual_mask_kernel<<<blocks, threads>>>(
        trees.data_ptr<float>(),
        output_mask.data_ptr<bool>(),
        batch_size, max_nodes, node_type, branch_type);
    cudaDeviceSynchronize();
}

void swap_node_params_cuda(torch::Tensor& c1, torch::Tensor& c2, const torch::Tensor& p1_mask, const torch::Tensor& p2_mask) {
    const int batch_size = c1.size(0);
    if (batch_size == 0) return;
    const int max_nodes = c1.size(1);

    swap_node_params_kernel<<<batch_size, 1>>>(
        c1.data_ptr<float>(),
        c2.data_ptr<float>(),
        p1_mask.data_ptr<bool>(),
        p2_mask.data_ptr<bool>(),
        batch_size, max_nodes);
    cudaDeviceSynchronize();
}