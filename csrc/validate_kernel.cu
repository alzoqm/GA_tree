// csrc/validate_kernel.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include <string>
#include <sstream>

#include "constants.h"
#include "validate_kernel.cuh"

// Error bitmask definitions
// 1 << 0: root branches invalid (count/parent/depth/branch types)
// 1 << 1: invalid parent or depth relation
// 1 << 2: leaf not action, or action has children
// 1 << 3: mixed child types under a parent (action+decision)
// 1 << 4: parent with action child has more than one child
// 1 << 5: invalid node type encountered

__global__ void validate_trees_kernel(
    const float* __restrict__ trees,
    int B,
    int max_nodes,
    int node_dim,
    int* __restrict__ out_error_mask,
    int* __restrict__ out_error_node
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float* base = trees + b * max_nodes * node_dim;
    int error_mask = 0;
    int error_node = -1;

    // 1) Validate root branches: exactly three, depth==0, parent==-1, unique branch types {0,1,2}
    int root_count = 0;
    int branch_seen[3] = {0, 0, 0};

    // Precompute parents/depths/types for convenience
    // Also count total active nodes (optional, not strictly used)
    for (int i = 0; i < max_nodes; ++i) {
        const float* node = base + i * node_dim;
        int t = (int)node[COL_NODE_TYPE];
        if (t == NODE_TYPE_UNUSED) continue;
        if (t < NODE_TYPE_UNUSED || t > NODE_TYPE_ACTION) {
            error_mask |= (1 << 5);
            error_node = (error_node == -1 ? i : error_node);
        }
        int parent = (int)node[COL_PARENT_IDX];
        int depth  = (int)node[COL_DEPTH];
        if (t == NODE_TYPE_ROOT_BRANCH) {
            if (parent != -1 || depth != 0) {
                error_mask |= (1 << 0);
                error_node = (error_node == -1 ? i : error_node);
            }
            root_count += 1;
            int branch_type = (int)node[COL_PARAM_1];
            if (branch_type >= 0 && branch_type <= 2) {
                branch_seen[branch_type] += 1;
            } else {
                error_mask |= (1 << 0);
                error_node = (error_node == -1 ? i : error_node);
            }
        } else {
            // Non-root with parent == -1 is invalid
            if (parent == -1) {
                error_mask |= (1 << 1);
                error_node = (error_node == -1 ? i : error_node);
            }
        }
    }

    if (!(root_count == 3 && branch_seen[0] == 1 && branch_seen[1] == 1 && branch_seen[2] == 1)) {
        error_mask |= (1 << 0);
    }

    // 2) Validate depth relation and gather child stats (O(N^2) scan per tree)
    for (int p = 0; p < max_nodes; ++p) {
        const float* pnode = base + p * node_dim;
        int ptype = (int)pnode[COL_NODE_TYPE];
        if (ptype == NODE_TYPE_UNUSED) continue;

        int pdepth = (int)pnode[COL_DEPTH];
        int pchild_count = 0;
        int action_children = 0;
        int decision_children = 0;

        for (int i = 0; i < max_nodes; ++i) {
            if (i == p) continue;
            const float* cnode = base + i * node_dim;
            int ctype = (int)cnode[COL_NODE_TYPE];
            if (ctype == NODE_TYPE_UNUSED) continue;
            int parent = (int)cnode[COL_PARENT_IDX];
            if (parent == p) {
                // Check parent-depth relation for child
                int cdepth = (int)cnode[COL_DEPTH];
                if (cdepth != pdepth + 1) {
                    error_mask |= (1 << 1);
                    error_node = (error_node == -1 ? i : error_node);
                }
                pchild_count += 1;
                if (ctype == NODE_TYPE_ACTION) action_children += 1;
                else if (ctype == NODE_TYPE_DECISION) decision_children += 1;
            }
        }

        // Leaf nodes must be ACTION (root branches should not be leaves)
        if (pchild_count == 0) {
            if (ptype != NODE_TYPE_ACTION) {
                // A root branch or decision node without children is invalid
                error_mask |= (1 << 2);
                error_node = (error_node == -1 ? p : error_node);
            }
        } else {
            // Non-leaf action is invalid
            if (ptype == NODE_TYPE_ACTION) {
                error_mask |= (1 << 2);
                error_node = (error_node == -1 ? p : error_node);
            }
        }

        // Mixed children types under the same parent not allowed
        if (action_children > 0 && decision_children > 0) {
            error_mask |= (1 << 3);
            error_node = (error_node == -1 ? p : error_node);
        }

        // If a parent has an ACTION child, it must have exactly one child
        if (action_children > 0 && pchild_count != 1) {
            error_mask |= (1 << 4);
            error_node = (error_node == -1 ? p : error_node);
        }
    }

    out_error_mask[b] = error_mask;
    out_error_node[b] = error_node;
}

void validate_trees_or_throw_cuda(const torch::Tensor& trees) {
    TORCH_CHECK(trees.is_cuda(), "Trees tensor must be on CUDA");
    TORCH_CHECK(trees.is_contiguous(), "Trees tensor must be contiguous");
    TORCH_CHECK(trees.dim() == 3, "Trees tensor must be 3D: (B, N, D)");
    TORCH_CHECK(trees.size(2) == NODE_INFO_DIM, "Node dim must match NODE_INFO_DIM");

    int B = trees.size(0);
    int N = trees.size(1);
    int D = trees.size(2);
    if (B == 0) return;

    auto options_i32 = torch::TensorOptions().device(trees.device()).dtype(torch::kInt32);
    torch::Tensor error_mask = torch::zeros({B}, options_i32);
    torch::Tensor error_node = torch::full({B}, -1, options_i32);

    int threads = 128;
    int blocks = (B + threads - 1) / threads;
    validate_trees_kernel<<<blocks, threads>>>(
        trees.data_ptr<float>(), B, N, D,
        error_mask.data_ptr<int>(),
        error_node.data_ptr<int>()
    );
    cudaDeviceSynchronize();

    // Copy results to CPU
    auto h_mask = error_mask.cpu();
    auto h_node = error_node.cpu();

    // Collect invalid indices and format message
    std::vector<int> bad_indices;
    bad_indices.reserve(B);
    std::ostringstream oss;
    bool any_bad = false;

    for (int b = 0; b < B; ++b) {
        int m = h_mask[b].item<int>();
        if (m != 0) {
            any_bad = true;
            bad_indices.push_back(b);
        }
    }

    if (any_bad) {
        // Build detailed message
        oss << "Tree structure validation failed. Invalid tree indices: [";
        for (size_t i = 0; i < bad_indices.size(); ++i) {
            oss << bad_indices[i] << (i + 1 < bad_indices.size() ? ", " : "]\n");
        }
        oss << "Error details per tree (bitmask -> node):\n";
        for (size_t i = 0; i < bad_indices.size(); ++i) {
            int idx = bad_indices[i];
            int m = h_mask[idx].item<int>();
            int n = h_node[idx].item<int>();
            oss << "  index " << idx << ": mask=0x" << std::hex << m << std::dec << ", node=" << n << "\n";
        }
        oss << "Bit meanings: 1:roots, 2:parent/depth, 4:leaf/action, 8:mixed-children, 16:action-child>1, 32:invalid-type";

        // Throw with formatted message (prints indices and raises error)
        TORCH_CHECK(false, oss.str());
    }
}

