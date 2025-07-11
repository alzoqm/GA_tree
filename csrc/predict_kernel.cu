// csrc/predict_kernel.cu
#include <cuda_runtime.h>
#include "constants.h"

// --- Device-level Helper Function ---
// Python의 _evaluate_node와 동일한 로직
__device__ bool evaluate_node_device(
    const float* node_data,
    const float* feature_values) {

    int op = static_cast<int>(node_data[COL_PARAM_2]);
    int comp_type = static_cast<int>(node_data[COL_PARAM_3]);
    int feat1_idx = static_cast<int>(node_data[COL_PARAM_1]);

    float val1 = feature_values[feat1_idx];
    float val2;

    if (comp_type == COMP_TYPE_FEAT_NUM) {
        val2 = node_data[COL_PARAM_4];
    } else { // COMP_TYPE_FEAT_FEAT
        int feat2_idx = static_cast<int>(node_data[COL_PARAM_4]);
        val2 = feature_values[feat2_idx];
    }

    switch(op) {
        case OP_GT: return val1 > val2;
        case OP_LT: return val1 < val2;
        case OP_EQ: return val1 == val2;
    }
    return false;
}


// --- Main CUDA Kernel ---
// Python의 predict 메소드와 동일한 로직
__global__ void predict_kernel(
    const float* population_ptr,
    const float* features_ptr,
    const long* positions_ptr,
    float* results_ptr,
    int pop_size,
    int max_nodes,
    int num_features) {

    // 1. Thread-to-Tree Mapping
    const int tree_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tree_idx >= pop_size) {
        return;
    }

    // 2. Setup pointers for the current tree
    const float* tree_data = population_ptr + tree_idx * max_nodes * NODE_INFO_DIM;
    const float* features = features_ptr + tree_idx * num_features;
    float* result_out = results_ptr + tree_idx * 3; // 3 = pos, size, leverage

    // 3. Find the starting node (root branch)
    long start_pos_type = positions_ptr[tree_idx];
    int start_node_idx = -1;
    for (int i = 0; i < 3; ++i) { // Root branches are always at indices 0, 1, 2
        const float* node = tree_data + i * NODE_INFO_DIM;
        if (static_cast<int>(node[COL_NODE_TYPE]) == NODE_TYPE_ROOT_BRANCH &&
            static_cast<int>(node[COL_PARAM_1]) == start_pos_type) {
            start_node_idx = i;
            break;
        }
    }

    if (start_node_idx == -1) {
        // Should not happen for a valid tree. Default to HOLD.
        result_out[0] = POS_TYPE_SHORT; // Represent HOLD as action with 0 size
        result_out[1] = 0.0f;
        result_out[2] = 0.0f;
        return;
    }

    // 4. Iterative DFS using a local stack
    constexpr int STACK_SIZE = 128; // Max tree depth is much smaller
    int node_stack[STACK_SIZE];
    int stack_top = -1;

    node_stack[++stack_top] = start_node_idx; // Push start node

    bool found_action = false;

    while (stack_top >= 0 && !found_action) {
        int current_node_idx = node_stack[stack_top--]; // Pop

        constexpr int MAX_CHILDREN_PER_NODE = 64; // A safe upper bound
        int successful_children[MAX_CHILDREN_PER_NODE];
        int success_count = 0;

        // Find all children of the current node
        for (int child_idx = 0; child_idx < max_nodes; ++child_idx) {
            const float* child_node_data = tree_data + child_idx * NODE_INFO_DIM;
            if (static_cast<int>(child_node_data[COL_PARENT_IDX]) == current_node_idx) {
                int child_node_type = static_cast<int>(child_node_data[COL_NODE_TYPE]);

                if (child_node_type == NODE_TYPE_ACTION) {
                    // Action found, this is the final result.
                    result_out[0] = child_node_data[COL_PARAM_1]; // Position type
                    result_out[1] = child_node_data[COL_PARAM_2]; // Size
                    result_out[2] = child_node_data[COL_PARAM_3]; // Leverage
                    found_action = true;
                    break; // Exit the for-loop over children
                }
                else if (child_node_type == NODE_TYPE_DECISION) {
                    if (evaluate_node_device(child_node_data, features)) {
                        if (success_count < MAX_CHILDREN_PER_NODE) {
                           successful_children[success_count++] = child_idx;
                        }
                    }
                }
            }
        }

        if (found_action) {
            break; // Exit the while-loop
        }

        // Push successful children to the stack in reverse order for correct DFS
        for (int i = success_count - 1; i >= 0; --i) {
            if (stack_top < STACK_SIZE - 1) {
                node_stack[++stack_top] = successful_children[i];
            }
        }
    }

    // 5. If no action was found after traversing, default to HOLD
    if (!found_action) {
        result_out[0] = POS_TYPE_SHORT;
        result_out[1] = 0.0f;
        result_out[2] = 0.0f;
    }
}


// --- Kernel Launcher ---
void launch_predict_kernel(
    const float* population_ptr,
    const float* features_ptr,
    const long* positions_ptr,
    float* results_ptr,
    int pop_size,
    int max_nodes,
    int num_features) {

    if (pop_size == 0) return;

    const int threads_per_block = 256;
    const int num_blocks = (pop_size + threads_per_block - 1) / threads_per_block;

    predict_kernel<<<num_blocks, threads_per_block>>>(
        population_ptr,
        features_ptr,
        positions_ptr,
        results_ptr,
        pop_size,
        max_nodes,
        num_features
    );
}