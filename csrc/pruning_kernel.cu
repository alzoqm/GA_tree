#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

#include "constants.h"
#include "pruning_kernel.cuh"

// Helpers
__device__ __forceinline__ int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }

// Compute per-node branch type by climbing to root (0:LONG,1:HOLD,2:SHORT), or -1 if unknown.
__device__ int branch_of_node(const float* base, int node_idx, int max_nodes) {
    int cur = node_idx;
    int safe = 0;
    while (cur >= 0 && cur < max_nodes && safe++ < max_nodes) {
        int parent = (int)base[cur * NODE_INFO_DIM + COL_PARENT_IDX];
        int ntype  = (int)base[cur * NODE_INFO_DIM + COL_NODE_TYPE];
        if (ntype == NODE_TYPE_ROOT_BRANCH) {
            int b = (int)base[cur * NODE_INFO_DIM + COL_PARAM_1];
            return (b >= 0 && b <= 2) ? b : -1;
        }
        if (parent == -1) return -1;
        cur = parent;
    }
    return -1;
}

// Kernel: For each tree, determine prune candidates for ACTION nodes and mark them UNUSED
// while protecting top M_min actions per branch by score.
__global__ void prune_mark_kernel(
    float* trees,
    const int B,
    const int N,
    const int D,
    const int total_steps,
    const int k_min,
    const float tau_use,
    const float tau_ineff,
    const int k_perf_min,
    const float tau_wr,
    const int k_harm,
    const float delta_harm,
    const int m_min,
    const float w_use,
    const float w_ineff,
    const float w_pnl,
    const float w_wr,
    const int* attempt,
    const int* apply,
    const int* noop,
    const int* stall,
    const int* win,
    const float* pnl_sum,
    const float* exposure_sum
) {
    const int b = blockIdx.x; // one block per tree
    if (b >= B) return;

    const int t0 = b * N * D;
    float* base = trees + t0;

    // Shared storage for thresholds and k_keep
    __shared__ int   k_keep;
    __shared__ float thr_long;
    __shared__ float thr_hold;
    __shared__ float thr_short;

    if (threadIdx.x == 0) {
        // Compute per-branch top-K scores sequentially to avoid race conditions
        k_keep   = m_min < 1 ? 1 : (m_min > 8 ? 8 : m_min);
        float top_long[8];
        float top_hold[8];
        float top_short[8];
        for (int i = 0; i < 8; ++i) top_long[i] = top_hold[i] = top_short[i] = -1e30f;

        for (int idx = 0; idx < N; ++idx) {
            int ntype = (int)base[idx * D + COL_NODE_TYPE];
            if (ntype != NODE_TYPE_ACTION) continue;

            int a = attempt[b * N + idx];
            int ap = apply[b * N + idx];
            int no = noop[b * N + idx];
            int st = stall[b * N + idx];
            int wi = win[b * N + idx];
            float pnl = pnl_sum[b * N + idx];
            float exp = exposure_sum[b * N + idx];
            float eff_use = total_steps > 0 ? (float)ap / (float)total_steps : 0.0f;
            float ineff   = a > 0 ? (float)(no + st) / (float)a : 0.0f;
            float wr      = ap > 0 ? (float)wi / (float)ap : 0.0f;
            float ppe     = (exp > 1e-12f) ? (pnl / exp) : 0.0f;
            float score = w_use * eff_use - w_ineff * ineff + w_pnl * ppe + w_wr * wr;
            int br = branch_of_node(base, idx, N);
            if (br == 0) {
                for (int k = 0; k < k_keep; ++k) {
                    if (score > top_long[k]) { for (int s = k_keep - 1; s > k; --s) top_long[s] = top_long[s-1]; top_long[k] = score; break; }
                }
            } else if (br == 1) {
                for (int k = 0; k < k_keep; ++k) {
                    if (score > top_hold[k]) { for (int s = k_keep - 1; s > k; --s) top_hold[s] = top_hold[s-1]; top_hold[k] = score; break; }
                }
            } else if (br == 2) {
                for (int k = 0; k < k_keep; ++k) {
                    if (score > top_short[k]) { for (int s = k_keep - 1; s > k; --s) top_short[s] = top_short[s-1]; top_short[k] = score; break; }
                }
            }
        }
        thr_long  = top_long[k_keep - 1];
        thr_hold  = top_hold[k_keep - 1];
        thr_short = top_short[k_keep - 1];
    }
    __syncthreads();

    // Second pass: apply candidate rule and mark UNUSED if below branch thresholds
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        int idx = i;
        int ntype = (int)base[idx * D + COL_NODE_TYPE];
        if (ntype != NODE_TYPE_ACTION) continue;

        int a = attempt[b * N + idx];
        int ap = apply[b * N + idx];
        int no = noop[b * N + idx];
        int st = stall[b * N + idx];
        int wi = win[b * N + idx];
        float pnl = pnl_sum[b * N + idx];
        float exp = exposure_sum[b * N + idx];
        float eff_use = total_steps > 0 ? (float)ap / (float)total_steps : 0.0f;
        float ineff   = a > 0 ? (float)(no + st) / (float)a : 0.0f;
        float wr      = ap > 0 ? (float)wi / (float)ap : 0.0f;
        float ppe     = (exp > 1e-12f) ? (pnl / exp) : 0.0f;

        bool cond_use  = (ap < k_min) || (eff_use < tau_use) || (ineff > tau_ineff);
        // Evaluate performance only when we have enough samples
        bool has_samples = (ap >= k_perf_min);
        bool cond_perf = has_samples && ((ppe <= 0.0f) || (wr <= tau_wr));
        bool cond_harm = (ap >= k_harm) && (ppe <= -delta_harm);
        bool candidate = (cond_use && cond_perf) || cond_harm;
        if (!candidate) continue;

        float score = w_use * eff_use - w_ineff * ineff + w_pnl * ppe + w_wr * wr;
        int br = branch_of_node(base, idx, N);
        float keep_thr = (br == 0 ? thr_long : (br == 1 ? thr_hold : thr_short));
        if (score < keep_thr) {
            base[idx * D + COL_NODE_TYPE] = (float)NODE_TYPE_UNUSED;
        }
    }
}

// After ACTION marking, perform a backward cleanup: any DECISION with no ACTION descendant becomes UNUSED.
__global__ void cleanup_decisions_kernel(
    float* trees, int B, int N, int D
) {
    const int b = blockIdx.x;
    if (b >= B) return;

    float* base = trees + b * N * D;

    // Iteratively remove DECISION nodes with no children until convergence or max N rounds.
    for (int round = 0; round < N; ++round) {
        __shared__ int any_change;
        if (threadIdx.x == 0) any_change = 0;
        __syncthreads();

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            int t = (int)base[i * D + COL_NODE_TYPE];
            if (t == NODE_TYPE_UNUSED) continue;
            if (t == NODE_TYPE_ACTION) {
                // Leaf ok
                continue;
            } else if (t == NODE_TYPE_DECISION || t == NODE_TYPE_ROOT_BRANCH) {
                // Check children types; if none remain (all UNUSED), mark UNUSED except root branches
                int child_count = 0;
                int action_child = 0;
                for (int j = 0; j < N; ++j) {
                    if (j == i) continue;
                    int pt = (int)base[j * D + COL_NODE_TYPE];
                    if (pt == NODE_TYPE_UNUSED) continue;
                    int p = (int)base[j * D + COL_PARENT_IDX];
                    if (p == i) {
                        child_count++;
                        if (pt == NODE_TYPE_ACTION) action_child++;
                    }
                }
                if (child_count == 0) {
                    if (t == NODE_TYPE_DECISION) { base[i * D + COL_NODE_TYPE] = (float)NODE_TYPE_UNUSED; atomicOr(&any_change, 1); }
                }
            }
        }
        __syncthreads();
        if (any_change == 0) break;
    }
}

void prune_mark_and_cleanup_cuda(
    torch::Tensor population_tensor,
    torch::Tensor node_attempt_count,
    torch::Tensor node_apply_count,
    torch::Tensor node_noop_count,
    torch::Tensor node_stall_count,
    torch::Tensor node_win_count,
    torch::Tensor node_realized_pnl_sum,
    torch::Tensor node_exposure_sum,
    PruneThresholds thr
) {
    TORCH_CHECK(population_tensor.is_cuda(), "population_tensor must be CUDA");
    TORCH_CHECK(population_tensor.is_contiguous(), "population_tensor must be contiguous");
    TORCH_CHECK(node_attempt_count.is_cuda() && node_apply_count.is_cuda(), "stats must be CUDA");
    TORCH_CHECK(node_noop_count.is_cuda() && node_stall_count.is_cuda(), "stats must be CUDA");
    TORCH_CHECK(node_win_count.is_cuda() && node_realized_pnl_sum.is_cuda(), "stats must be CUDA");
    TORCH_CHECK(node_exposure_sum.is_cuda(), "stats must be CUDA");

    const int B = population_tensor.size(0);
    const int N = population_tensor.size(1);
    const int D = population_tensor.size(2);
    if (B == 0 || N == 0) return;

    const dim3 blocks(B);
    const dim3 threads( min(256, N) );
    size_t shmem = 3 * 8 * sizeof(float); // 3 branches x top-K (<=8)

    prune_mark_kernel<<<blocks, threads, shmem>>>(
        population_tensor.data_ptr<float>(), B, N, D,
        thr.total_steps,
        thr.k_min, thr.tau_use, thr.tau_ineff,
        thr.k_perf_min, thr.tau_wr,
        thr.k_harm, thr.delta_harm,
        thr.m_min,
        thr.w_use, thr.w_ineff, thr.w_pnl, thr.w_wr,
        node_attempt_count.data_ptr<int>(),
        node_apply_count.data_ptr<int>(),
        node_noop_count.data_ptr<int>(),
        node_stall_count.data_ptr<int>(),
        node_win_count.data_ptr<int>(),
        node_realized_pnl_sum.data_ptr<float>(),
        node_exposure_sum.data_ptr<float>()
    );
    cudaDeviceSynchronize();

    cleanup_decisions_kernel<<<blocks, threads>>>(
        population_tensor.data_ptr<float>(), B, N, D
    );
    cudaDeviceSynchronize();
}
