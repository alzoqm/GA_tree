// /csrc/value_mutation_kernel.cu
#include <curand_kernel.h>
#include "constants.h"
#include "value_mutation_kernel.cuh"

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
                // --- [수정 시작] 클램핑 로직 추가 ---
                int feat_idx = (int)node_data[COL_PARAM_1];
                
                // config에서 해당 피처의 min/max 값을 찾기 위해 인덱스(offset)를 찾습니다.
                int offset = -1;
                for (int i = 0; i < num_feature_num; ++i) {
                    if (feature_num_indices[i] == feat_idx) {
                        offset = i;
                        break;
                    }
                }
                
                if (offset != -1) {
                    float min_v = feature_min_vals[offset];
                    float max_v = feature_max_vals[offset];
                    float current_val = node_data[COL_PARAM_4];
                    float noise_range = (max_v - min_v) * noise_ratio;
                    float noise = (curand_uniform(&state) * 2.0f - 1.0f) * noise_range;
                    
                    // 값을 더한 후, fminf와 fmaxf로 유효 범위 내로 클램핑합니다.
                    node_data[COL_PARAM_4] = fminf(max_v, fmaxf(min_v, current_val + noise));
                }
                // --- [수정 끝] ---
             } else if (comp_type == COMP_TYPE_FEAT_BOOL) {
                 node_data[COL_PARAM_4] = 1.0f - node_data[COL_PARAM_4];
             }
        }
    }
    states[gid] = state;
}

// [수정] C++ 래퍼 함수들(_launch_mutation_kernel, node_param_mutate_cuda, reinitialize_node_mutate_cuda)을
// predict.cpp로 이동시켰으므로 이 파일에서는 삭제합니다.