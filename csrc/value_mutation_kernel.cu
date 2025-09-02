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

        if (is_reinitialize) { // Reinitialize Mutation (fully on GPU)
            // Select new comparison type among available groups
            int choice = (int)(curand_uniform(&state) * 3.0f);
            if (choice == 0 && num_feat_num > 0) comp_type = COMP_TYPE_FEAT_NUM;
            else if (choice == 1 && num_feat_comp > 0) comp_type = COMP_TYPE_FEAT_FEAT;
            else if (choice == 2 && num_feat_bool > 0) comp_type = COMP_TYPE_FEAT_BOOL;
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
                // Pick two feature indices from the comparison set (GPU-side)
                int idx1 = (int)(curand_uniform(&state) * num_feat_comp);
                int idx2 = (int)(curand_uniform(&state) * num_feat_comp);
                if (idx1 < 0) idx1 = 0; if (idx1 >= num_feat_comp) idx1 = num_feat_comp - 1;
                if (idx2 < 0) idx2 = 0; if (idx2 >= num_feat_comp) idx2 = num_feat_comp - 1;
                node_data[COL_PARAM_1] = (float)feature_comparison_indices_ptr[idx1];
                node_data[COL_PARAM_4] = (float)feature_comparison_indices_ptr[idx2];
            }
            node_data[COL_PARAM_2] = (curand_uniform(&state) < 0.5f) ? (float)OP_GTE : (float)OP_LTE;

        } else if (comp_type == COMP_TYPE_FEAT_BOOL) {
            if (is_reinitialize && num_feat_bool > 0) {
                node_data[COL_PARAM_1] = (float)feature_bool_indices_ptr[(int)(curand_uniform(&state) * num_feat_bool)];
            }
            node_data[COL_PARAM_4] = (curand_uniform(&state) < 0.5f) ? 0.0f : 1.0f;
        }

    } else if (node_type == NODE_TYPE_ACTION) {
        // --- Action Node mutation/reinitialize ---
        int action_type = (int)node_data[COL_PARAM_1];
        if (is_reinitialize) {
            // Fully ported _create_random_action_params logic
            // Derive tree-local indices
            // gid = global node index over entire population
            // Compute tree and local node index to access parents within the same tree
            const int tree_idx = gid / max_nodes;
            const int local_idx = gid % max_nodes;
            float* tree_base = population_ptr + (size_t)tree_idx * max_nodes * NODE_INFO_DIM;

            // Find root branch type by climbing from the parent of this node
            int parent_idx = (int)node_data[COL_PARENT_IDX];
            int cur = parent_idx;
            int safety = 0;
            int root_branch_type = -1;
            while (cur >= 0 && cur < max_nodes && safety < max_nodes) {
                const float* nb = tree_base + (size_t)cur * NODE_INFO_DIM;
                int p = (int)nb[COL_PARENT_IDX];
                if (p == -1) {
                    root_branch_type = (int)nb[COL_PARAM_1];
                    break;
                }
                cur = p;
                ++safety;
            }

            // Choose possible actions based on root branch type
            // HOLD: NEW_LONG, NEW_SHORT
            // LONG/SHORT: CLOSE_ALL, CLOSE_PARTIAL, ADD_POSITION, FLIP_POSITION
            int candidates[4];
            int cand_n = 0;
            if (root_branch_type == 1 /*ROOT_BRANCH_HOLD*/ || root_branch_type == ROOT_BRANCH_HOLD) {
                candidates[cand_n++] = ACTION_NEW_LONG;
                candidates[cand_n++] = ACTION_NEW_SHORT;
            } else if (root_branch_type == 0 /*ROOT_BRANCH_LONG*/ || root_branch_type == ROOT_BRANCH_LONG ||
                       root_branch_type == 2 /*ROOT_BRANCH_SHORT*/ || root_branch_type == ROOT_BRANCH_SHORT) {
                candidates[cand_n++] = ACTION_CLOSE_ALL;
                candidates[cand_n++] = ACTION_CLOSE_PARTIAL;
                candidates[cand_n++] = ACTION_ADD_POSITION;
                candidates[cand_n++] = ACTION_FLIP_POSITION;
            }
            if (cand_n == 0) {
                // Fallback: ensure valid action exists
                candidates[cand_n++] = ACTION_CLOSE_ALL;
            }
            int pick = (int)(curand_uniform(&state) * cand_n);
            if (pick < 0) pick = 0; if (pick >= cand_n) pick = cand_n - 1;
            action_type = candidates[pick];
            node_data[COL_PARAM_1] = (float)action_type;

            // Zero out unused params by default
            // Keep COL_PARAM_2 and COL_PARAM_3 updated below as needed
            // COL_PARAM_4 not used for actions here
            // Initialize to 0 safely
            // Note: NODE_INFO_DIM >= 7 by constants
            node_data[COL_PARAM_2] = 0.0f;
            node_data[COL_PARAM_3] = 0.0f;
            node_data[COL_PARAM_4] = 0.0f;
        }

        if (action_type == ACTION_NEW_LONG || action_type == ACTION_NEW_SHORT || action_type == ACTION_FLIP_POSITION) {
            // Param2: 진입/플립 비율 (0.0 ~ 1.0)
            float old_ratio = node_data[COL_PARAM_2];
            float noise_ratio_val = noise_ratio * (curand_uniform(&state) * 2.0f - 1.0f);
            node_data[COL_PARAM_2] = is_reinitialize ? curand_uniform(&state) : clamp(old_ratio + noise_ratio_val, 0.0f, 1.0f);
            
            // Param3: 레버리지 (1 ~ 100)
            int old_lev = (int)node_data[COL_PARAM_3];
            int lev_change = (int)(curand_uniform(&state) * (2 * leverage_change + 1)) - leverage_change;
            if (is_reinitialize) {
                node_data[COL_PARAM_3] = (float)(1 + (int)(curand_uniform(&state) * 100.0f));
            } else {
                node_data[COL_PARAM_3] = (float)clamp(old_lev + lev_change, 1, 100);
            }

        } else if (action_type == ACTION_CLOSE_PARTIAL || action_type == ACTION_ADD_POSITION) {
            // Param2: 청산/추가 비율 (0.0 ~ 1.0)
            float old_ratio = node_data[COL_PARAM_2];
            float noise_ratio_val = noise_ratio * (curand_uniform(&state) * 2.0f - 1.0f);
            node_data[COL_PARAM_2] = is_reinitialize ? curand_uniform(&state) : clamp(old_ratio + noise_ratio_val, 0.0f, 1.0f);
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
