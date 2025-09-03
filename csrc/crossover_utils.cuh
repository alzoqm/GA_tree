#ifndef CROSSOVER_UTILS_CUH
#define CROSSOVER_UTILS_CUH

#include "crossover_kernel.cuh"
#include "constants.h"

// ==============================================================================
//                       BFS Helper Function for Subtree Collection
// ==============================================================================

/**
 * @brief Finds all nodes in a subtree starting from a given root using Breadth-First Search (BFS).
 *
 * This device function is executed by a single thread and is designed to be called from a CUDA kernel.
 * It traverses a tree structure represented by a flat tensor and collects the indices of all nodes
 * belonging to the subtree rooted at `root_idx`.
 *
 * @param tree_ptr Pointer to the beginning of the tree data in global memory.
 * @param root_idx The index of the root node of the subtree to be collected.
 * @param queue_buffer A thread-local buffer to be used as a queue for the BFS.
 * @param result_buffer A thread-local buffer to store the indices of the nodes in the subtree.
 * @param max_nodes The maximum number of nodes in the tree, used for bounds checking.
 * @return The number of nodes found in the subtree.
 */
__device__ inline int find_subtree_nodes_device(
    const float* tree_ptr,
    int root_idx,
    int* queue_buffer,      // Thread-local BFS queue
    int* result_buffer,     // Thread-local result array
    int max_nodes
) {
    // Basic validation to ensure the root index is within bounds and the node is in use.
    if (root_idx < 0 || root_idx >= max_nodes) return 0;
    if ((int)tree_ptr[root_idx * NODE_INFO_DIM + COL_NODE_TYPE] == NODE_TYPE_UNUSED) return 0;

    int queue_front = 0;
    int queue_back = 0;
    int result_count = 0;

    // Initialize BFS: add the root node to the queue.
    queue_buffer[queue_back++] = root_idx;

    // Process the queue until it is empty or the result buffer is full.
    while (queue_front < queue_back && result_count < max_nodes) {
        int current_idx = queue_buffer[queue_front++];
        result_buffer[result_count++] = current_idx;

        // Find all children of the current node and add them to the queue.
        // This is a simple linear scan, which is acceptable for small N, but for larger N,
        // an adjacency list would be more efficient.
        for (int i = 0; i < max_nodes; i++) {
            if ((int)tree_ptr[i * NODE_INFO_DIM + COL_PARENT_IDX] == current_idx &&
                (int)tree_ptr[i * NODE_INFO_DIM + COL_NODE_TYPE] != NODE_TYPE_UNUSED) {
                if (queue_back < max_nodes) {
                    queue_buffer[queue_back++] = i;
                }
            }
        }
    }

    return result_count;
}

#endif // CROSSOVER_UTILS_CUH