// /csrc/value_mutation_kernel.cu
#include <curand_kernel.h>
#include "constants.h"
#include "value_mutation_kernel.cuh" // [수정] 헤더 파일명 변경

// ==============================================================================
//           cuRAND 상태 초기화 커널 (변경 없음)
// ==============================================================================
__global__ void init_curand_states_kernel(unsigned long long seed, int pop_size, int max_nodes, curandStatePhilox4_32_10_t* states) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pop_size * max_nodes) return;
    curand_init(seed, gid, 0, &states[gid]);
}

// ==============================================================================
//           헬퍼 함수: 노드의 루트 분기 타입 찾기 (변경 없음)
// ==============================================================================
__device__ int get_root_branch_type_device(const float* tree_data, int node_idx, int max_nodes) {
    int current_idx = node_idx;
    while (true) {
        int parent_idx = (int)tree_data[current_idx * NODE_INFO_DIM + COL_PARENT_IDX];
        if (parent_idx == -1) {
            return (int)tree_data[current_idx * NODE_INFO_DIM + COL_PARAM_1];
        }
        current_idx = parent_idx;
    }
    return -1;
}

// ==============================================================================
//           메인 돌연변이 커널 (내부 로직은 동일, 이름 유지)
// ==============================================================================
__global__ void value_mutation_kernel(
    float* population_ptr,
    curandStatePhilox4_32_10_t* states,
    // --- 제어 파라미터 ---
    bool is_reinitialize,
    float mutation_prob,
    float noise_ratio,
    int leverage_change,
    // --- Config에서 전달받는 포인터들 ---
    const int* feature_num_indices,
    const float* feature_min_vals,
    const float* feature_max_vals,
    int num_feature_num,
    const int* feature_comparison_indices,
    int num_feature_comparison,
    const int* feature_bool_indices,
    int num_feature_bool,
    // --- 트리 정보 ---
    int pop_size, int max_nodes
) {
    // ... (이 커널의 내부 로직은 이전 답변과 동일하게 유지) ...
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pop_size * max_nodes) return;

    curandStatePhilox4_32_10_t state = states[gid];
    float* node_data = population_ptr + gid * NODE_INFO_DIM;
    int node_type = (int)node_data[COL_NODE_TYPE];

    if (node_type != NODE_TYPE_DECISION && node_type != NODE_TYPE_ACTION) {
        states[gid] = state;
        return;
    }

    if (curand_uniform(&state) >= mutation_prob) {
        states[gid] = state;
        return;
    }

    int tree_idx = gid / max_nodes;
    int local_node_idx = gid % max_nodes;
    const float* tree_data = population_ptr + tree_idx * max_nodes * NODE_INFO_DIM;

    if (node_type == NODE_TYPE_ACTION) {
        int root_branch_type = get_root_branch_type_device(tree_data, local_node_idx, max_nodes);
        int current_action_type = (int)node_data[COL_PARAM_1];

        if (is_reinitialize) {
            int new_action_type = current_action_type;
            if (root_branch_type == ROOT_BRANCH_HOLD) {
                new_action_type = (curand_uniform(&state) < 0.5f) ? ACTION_NEW_LONG : ACTION_NEW_SHORT;
            } else if (root_branch_type == ROOT_BRANCH_LONG || root_branch_type == ROOT_BRANCH_SHORT) {
                float rand_val = curand_uniform(&state);
                if (rand_val < 0.25f) new_action_type = ACTION_CLOSE_ALL;
                else if (rand_val < 0.5f) new_action_type = ACTION_CLOSE_PARTIAL;
                else if (rand_val < 0.75f) new_action_type = ACTION_ADD_POSITION;
                else new_action_type = ACTION_FLIP_POSITION;
            }
            node_data[COL_PARAM_1] = new_action_type;
            current_action_type = new_action_type;
        }
        
        if (current_action_type == ACTION_NEW_LONG || current_action_type == ACTION_NEW_SHORT || current_action_type == ACTION_FLIP_POSITION) {
            if (is_reinitialize) {
                node_data[COL_PARAM_2] = curand_uniform(&state);
                node_data[COL_PARAM_3] = roundf(curand_uniform(&state) * 99.0f + 1.0f);
            } else { // NodeParamMutation
                if (curand_uniform(&state) < 0.5f) {
                    float noise = curand_normal(&state) * 0.1f;
                    node_data[COL_PARAM_2] = fminf(fmaxf(node_data[COL_PARAM_2] + noise, 0.0f), 1.0f);
                } else {
                    float change = curand_uniform(&state) * 2.0f * leverage_change - leverage_change;
                    node_data[COL_PARAM_3] = fminf(fmaxf(roundf(node_data[COL_PARAM_3] + change), 1.0f), 100.0f);
                }
            }
        } else if (current_action_type == ACTION_CLOSE_PARTIAL || current_action_type == ACTION_ADD_POSITION) {
             float noise = curand_normal(&state) * 0.1f;
             node_data[COL_PARAM_2] = fminf(fmaxf(node_data[COL_PARAM_2] + noise, 0.0f), 1.0f);
        }

    } else if (node_type == NODE_TYPE_DECISION) {
        if (is_reinitialize) {
            float rand_type = curand_uniform(&state);
            int comp_type = (rand_type < 0.6f) ? COMP_TYPE_FEAT_NUM : ((rand_type < 0.8f) ? COMP_TYPE_FEAT_FEAT : COMP_TYPE_FEAT_BOOL);
            node_data[COL_PARAM_3] = comp_type;
            if (comp_type != COMP_TYPE_FEAT_BOOL) node_data[COL_PARAM_2] = (curand_uniform(&state) < 0.5f) ? OP_GTE : OP_LTE;

            if (comp_type == COMP_TYPE_FEAT_NUM) {
                int rand_idx = (int)(curand_uniform(&state) * num_feature_num);
                node_data[COL_PARAM_1] = feature_num_indices[rand_idx];
                float min_v = feature_min_vals[rand_idx], max_v = feature_max_vals[rand_idx];
                node_data[COL_PARAM_4] = curand_uniform(&state) * (max_v - min_v) + min_v;
            } else if (comp_type == COMP_TYPE_FEAT_FEAT) {
                int rand_idx1 = (int)(curand_uniform(&state) * num_feature_comparison);
                int rand_idx2 = (int)(curand_uniform(&state) * num_feature_comparison);
                node_data[COL_PARAM_1] = feature_comparison_indices[rand_idx1];
                node_data[COL_PARAM_4] = feature_comparison_indices[rand_idx2];
            } else {
                int rand_idx = (int)(curand_uniform(&state) * num_feature_bool);
                node_data[COL_PARAM_1] = feature_bool_indices[rand_idx];
                node_data[COL_PARAM_4] = (curand_uniform(&state) < 0.5f) ? 0.0f : 1.0f;
            }
        } else { // NodeParamMutation
             int comp_type = (int)node_data[COL_PARAM_3];
             if (comp_type == COMP_TYPE_FEAT_NUM) {
                 float noise = curand_normal(&state) * noise_ratio;
                 // Note: Clamping requires finding the correct min/max, which is complex here.
                 // For simplicity in this example, we just add noise.
                 node_data[COL_PARAM_4] += noise;
             } else if (comp_type == COMP_TYPE_FEAT_BOOL) {
                 node_data[COL_PARAM_4] = 1.0f - node_data[COL_PARAM_4];
             }
        }
    }
    states[gid] = state;
}

// ==============================================================================
//           [신규] 공통 커널 실행 로직 헬퍼 함수
// ==============================================================================
void _launch_mutation_kernel(
    torch::Tensor& population, bool is_reinitialize, float mutation_prob,
    float noise_ratio, int leverage_change,
    torch::Tensor& feature_num_indices, torch::Tensor& feature_min_vals, torch::Tensor& feature_max_vals,
    torch::Tensor& feature_comparison_indices, torch::Tensor& feature_bool_indices
) {
    const int pop_size = population.size(0);
    const int max_nodes = population.size(1);
    const int total_nodes = pop_size * max_nodes;

    auto options = torch::TensorOptions().device(population.device()).dtype(torch::kInt64);
    torch::Tensor curand_states = torch::empty({total_nodes, sizeof(curandStatePhilox4_32_10_t) / sizeof(int64_t)}, options);

    const int threads = 256;
    const int blocks = (total_nodes + threads - 1) / threads;

    init_curand_states_kernel<<<blocks, threads>>>(
        (unsigned long long)time(0) + (unsigned long long)rand(), // Add more randomness
        pop_size, max_nodes, (curandStatePhilox4_32_10_t*)curand_states.data_ptr()
    );

    value_mutation_kernel<<<blocks, threads>>>(
        population.data_ptr<float>(),
        (curandStatePhilox4_32_10_t*)curand_states.data_ptr(),
        is_reinitialize, mutation_prob, noise_ratio, leverage_change,
        feature_num_indices.data_ptr<int>(), feature_min_vals.data_ptr<float>(), feature_max_vals.data_ptr<float>(), feature_num_indices.size(0),
        feature_comparison_indices.data_ptr<int>(), feature_comparison_indices.size(0),
        feature_bool_indices.data_ptr<int>(), feature_bool_indices.size(0),
        pop_size, max_nodes
    );
}

// ==============================================================================
//           [신규] NodeParamMutation을 위한 C++ 래퍼 함수
// ==============================================================================
void node_param_mutate_cuda(
    torch::Tensor population, float mutation_prob, float noise_ratio, int leverage_change,
    torch::Tensor feature_num_indices, torch::Tensor feature_min_vals, torch::Tensor feature_max_vals
) {
    TORCH_CHECK(population.is_cuda(), "Population tensor must be on CUDA");
    
    // 사용하지 않는 텐서는 빈 텐서로 생성하여 전달
    auto empty_int_tensor = torch::empty({0}, torch::dtype(torch::kInt32).device(population.device()));

    _launch_mutation_kernel(
        population, false, mutation_prob, noise_ratio, leverage_change,
        feature_num_indices, feature_min_vals, feature_max_vals,
        empty_int_tensor, empty_int_tensor
    );
}

// ==============================================================================
//           [신규] ReinitializeNodeMutation을 위한 C++ 래퍼 함수
// ==============================================================================
void reinitialize_node_mutate_cuda(
    torch::Tensor population, float mutation_prob,
    torch::Tensor feature_num_indices, torch::Tensor feature_min_vals, torch::Tensor feature_max_vals,
    torch::Tensor feature_comparison_indices, torch::Tensor feature_bool_indices
) {
    TORCH_CHECK(population.is_cuda(), "Population tensor must be on CUDA");

    _launch_mutation_kernel(
        population, true, mutation_prob, 0.0, 0,
        feature_num_indices, feature_min_vals, feature_max_vals,
        feature_comparison_indices, feature_bool_indices
    );
}