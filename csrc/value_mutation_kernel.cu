// /csrc/value_mutation_kernel.cu (신규 파일 - 전체 구현)
#include "value_mutation_kernel.cuh"
#include "constants.h"
#include <curand_kernel.h>
#include <cmath> // for fabsf

// ==============================================================================
//                      CUDA 커널 및 디바이스 함수
// ==============================================================================

// 디바이스 함수: float 범위 내로 값을 제한 (clamp)
__device__ float clamp(float val, float min_val, float max_val) {
    return fmaxf(min_val, fminf(val, max_val));
}

// 메인 변이 커널
__global__ void value_mutation_kernel(
    float* population_ptr,
    bool is_reinitialize,
    float mutation_prob,
    float noise_ratio,
    int leverage_change,
    const int* feature_num_indices_ptr, int num_feat_num,
    const float* feature_min_vals_ptr,
    const float* feature_max_vals_ptr,
    const int* feature_comparison_indices_ptr, int num_feat_comp,
    const int* feature_bool_indices_ptr, int num_feat_bool,
    int pop_size,
    int max_nodes)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_nodes = pop_size * max_nodes;

    if (gid >= total_nodes) return;

    // --- 1. 난수 생성기 초기화 ---
    curandState state;
    curand_init(gid, 0, 0, &state);

    // --- 2. 노드 데이터 포인터 설정 및 기본 검사 ---
    float* node_data = population_ptr + gid * NODE_INFO_DIM;
    const int node_type = (int)node_data[COL_NODE_TYPE];
    
    // 변이 대상 노드 타입이 아니면 종료 (Decision 또는 Action 노드만 대상)
    if (node_type != NODE_TYPE_DECISION && node_type != NODE_TYPE_ACTION) {
        return;
    }

    // 확률에 따라 변이 실행 여부 결정
    if (curand_uniform(&state) >= mutation_prob) {
        return;
    }

    // --- 3. 노드 타입에 따른 변이 실행 ---
    if (node_type == NODE_TYPE_DECISION) {
        // --- Decision Node 변이 ---
        int comp_type = (int)node_data[COL_PARAM_3];

        if (is_reinitialize) { // Reinitialize Mutation
            // 새로운 비교 타입 랜덤 선택
            int new_comp_type_choice = curand_uniform(&state) * 3;
            if (new_comp_type_choice == 0 && num_feat_num > 0) comp_type = COMP_TYPE_FEAT_NUM;
            else if (new_comp_type_choice == 1 && num_feat_comp > 0) comp_type = COMP_TYPE_FEAT_FEAT;
            else if (new_comp_type_choice == 2 && num_feat_bool > 0) comp_type = COMP_TYPE_FEAT_BOOL;
            node_data[COL_PARAM_3] = (float)comp_type;
        }

        if (comp_type == COMP_TYPE_FEAT_NUM) {
            if (is_reinitialize) {
                int rand_idx = curand_uniform(&state) * num_feat_num;
                node_data[COL_PARAM_1] = (float)feature_num_indices_ptr[rand_idx];
                float min_val = feature_min_vals_ptr[rand_idx];
                float max_val = feature_max_vals_ptr[rand_idx];
                node_data[COL_PARAM_4] = curand_uniform(&state) * (max_val - min_val) + min_val;
            } else { // Noise Mutation
                int current_feat_idx = (int)node_data[COL_PARAM_1];
                for (int i = 0; i < num_feat_num; ++i) {
                    if (feature_num_indices_ptr[i] == current_feat_idx) {
                        float min_val = feature_min_vals_ptr[i];
                        float max_val = feature_max_vals_ptr[i];
                        float range = max_val - min_val;
                        float noise = range * noise_ratio * (curand_uniform(&state) * 2.0f - 1.0f);
                        node_data[COL_PARAM_4] = clamp(node_data[COL_PARAM_4] + noise, min_val, max_val);
                        break;
                    }
                }
            }
            node_data[COL_PARAM_2] = (curand_uniform(&state) < 0.5f) ? (float)OP_GTE : (float)OP_LTE;

        } else if (comp_type == COMP_TYPE_FEAT_FEAT) {
            if (is_reinitialize && num_feat_comp > 0) {
                 // 이 부분은 복잡하여 CPU에서 처리하는 것이 더 안정적일 수 있으나,
                 // 여기서는 간단히 랜덤 인덱스만 교체합니다.
                node_data[COL_PARAM_1] = (float)feature_comparison_indices_ptr[(int)(curand_uniform(&state) * num_feat_comp)];
                node_data[COL_PARAM_4] = (float)feature_comparison_indices_ptr[(int)(curand_uniform(&state) * num_feat_comp)];
            }
            node_data[COL_PARAM_2] = (curand_uniform(&state) < 0.5f) ? (float)OP_GTE : (float)OP_LTE;

        } else if (comp_type == COMP_TYPE_FEAT_BOOL) {
            if (is_reinitialize && num_feat_bool > 0) {
                node_data[COL_PARAM_1] = (float)feature_bool_indices_ptr[(int)(curand_uniform(&state) * num_feat_bool)];
            }
            node_data[COL_PARAM_4] = (curand_uniform(&state) < 0.5f) ? 0.0f : 1.0f;
        }

    } else if (node_type == NODE_TYPE_ACTION) {
        // --- Action Node 변이 ---
        int action_type = (int)node_data[COL_PARAM_1];
        if (is_reinitialize) {
            // 재초기화는 CPU 로직이 복잡하므로 여기서는 단순 값 변경만 수행
            // (이 부분은 필요 시 Python의 `_create_random_action_params` 로직을 CUDA로 포팅해야 함)
            action_type = 1 + (int)(curand_uniform(&state) * 6);
            node_data[COL_PARAM_1] = (float)action_type;
        }

        if (action_type == ACTION_NEW_LONG || action_type == ACTION_NEW_SHORT || action_type == ACTION_FLIP_POSITION) {
            // Param2: 진입/플립 비율 (0.0 ~ 1.0)
            float old_ratio = node_data[COL_PARAM_2];
            float noise_ratio_val = noise_ratio * (curand_uniform(&state) * 2.0f - 1.0f);
            node_data[COL_PARAM_2] = clamp(old_ratio + noise_ratio_val, 0.0f, 1.0f);
            
            // Param3: 레버리지 (1 ~ 100)
            int old_lev = (int)node_data[COL_PARAM_3];
            int lev_change = (int)(curand_uniform(&state) * (2 * leverage_change + 1)) - leverage_change;
            node_data[COL_PARAM_3] = (float)clamp(old_lev + lev_change, 1, 100);

        } else if (action_type == ACTION_CLOSE_PARTIAL || action_type == ACTION_ADD_POSITION) {
            // Param2: 청산/추가 비율 (0.0 ~ 1.0)
            float old_ratio = node_data[COL_PARAM_2];
            float noise_ratio_val = noise_ratio * (curand_uniform(&state) * 2.0f - 1.0f);
            node_data[COL_PARAM_2] = clamp(old_ratio + noise_ratio_val, 0.0f, 1.0f);
        }
    }
}


// ==============================================================================
//                         C++ 래퍼 함수 (커널 런처)
// ==============================================================================
void _launch_mutation_kernel_cpp(
    torch::Tensor& population,
    bool is_reinitialize,
    float mutation_prob,
    float noise_ratio,
    int leverage_change,
    torch::Tensor& feature_num_indices,
    torch::Tensor& feature_min_vals,
    torch::Tensor& feature_max_vals,
    torch::Tensor& feature_comparison_indices,
    torch::Tensor& feature_bool_indices)
{
    TORCH_CHECK(population.is_cuda(), "Population tensor must be on a CUDA device for mutation.");
    TORCH_CHECK(population.is_contiguous(), "Population tensor must be contiguous.");

    const int pop_size = population.size(0);
    const int max_nodes = population.size(1);
    const int total_nodes = pop_size * max_nodes;

    if (total_nodes == 0) return;

    const int threads_per_block = 256;
    const int num_blocks = (total_nodes + threads_per_block - 1) / threads_per_block;

    value_mutation_kernel<<<num_blocks, threads_per_block>>>(
        population.data_ptr<float>(),
        is_reinitialize,
        mutation_prob,
        noise_ratio,
        leverage_change,
        feature_num_indices.data_ptr<int>(), feature_num_indices.size(0),
        feature_min_vals.data_ptr<float>(),
        feature_max_vals.data_ptr<float>(),
        feature_comparison_indices.data_ptr<int>(), feature_comparison_indices.size(0),
        feature_bool_indices.data_ptr<int>(), feature_bool_indices.size(0),
        pop_size,
        max_nodes
    );
}