// csrc/ppts_selection.cu - High-Performance PPTS Selection Implementation
#include "ppts_selection.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK_ERRORS() \
  do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  }} while (0)

// Performance optimization constants
#define MAX_TOURNAMENT_SIZE 64        // Compile-time upper bound for k
#define MAX_OBJECTIVES 16            // Compile-time upper bound for M  
#define WARP_SIZE 32
#define MAX_SHARED_MEMORY (48 * 1024) // 48KB shared memory budget

// Utility functions matching existing codebase style
__device__ __forceinline__ int   f2i(float x) { return __float2int_rn(x); }
__device__ __forceinline__ float i2f(int x) { return static_cast<float>(x); }
__device__ __forceinline__ bool  valid_idx(int idx, int N) { return idx >= 0 && idx < N; }

// Fast integer ceil(log2(x)) for tournament size computation
__device__ __forceinline__ int ceil_log2(int x) {
    return x <= 1 ? 1 : 32 - __clz(x - 1);
}

// Warp-level reduction for computing minimum across warp
__device__ __forceinline__ float warp_reduce_min(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Warp-level reduction for computing median (approximated via sorting network)
__device__ __forceinline__ float warp_median_approx(float val, int valid_count) {
    // Simple approximation: return value at median position
    // For small k, this provides sufficient approximation for MAD computation
    float vals[MAX_TOURNAMENT_SIZE];
    vals[threadIdx.x] = val;
    __syncwarp();
    
    // Bubble sort for small arrays (k << 64)
    for (int i = 0; i < valid_count - 1; i++) {
        for (int j = 0; j < valid_count - i - 1; j++) {
            if (vals[j] > vals[j + 1]) {
                float temp = vals[j];
                vals[j] = vals[j + 1];
                vals[j + 1] = temp;
            }
        }
    }
    return vals[valid_count / 2];
}

/**
 * Fused high-performance PPTS tournament kernel
 * 
 * Each block processes one tournament with the following fused operations:
 * 1. Load tournament objectives to shared memory
 * 2. Compute Pareto dominance relationships  
 * 3. Apply ε-lexicase tie-breaking if needed
 * 4. Select final winner with random tie-breaking
 * 
 * Shared memory layout:
 * - objectives: [k, M] tournament objectives
 * - dominance_mask: [k] non-dominated flags
 * - survivors: [k] ε-lexicase survivors  
 * - temp_storage: temporary arrays for reductions
 */
__global__ void k_ppts_fused_tournaments(
    const float* __restrict__ objectives,     // [N, M] population objectives
    int64_t* __restrict__ winner_indices,    // [B] output winner indices  
    const int32_t* __restrict__ tournament_indices, // [B, k] tournament participants
    int N,                                   // Population size
    int M,                                   // Number of objectives  
    int k,                                   // Tournament size
    int64_t rng_seed                        // Random seed
) {
    const int b = blockIdx.x;               // Tournament index
    const int tid = threadIdx.x;            // Thread within tournament
    
    // Shared memory allocation - maximize usage for performance
    extern __shared__ char shared_mem[];
    float* s_objectives = (float*)shared_mem;                    // [k, M]
    bool* s_dominance_mask = (bool*)(s_objectives + k * M);     // [k]
    bool* s_survivors = (bool*)(s_dominance_mask + k);          // [k] 
    float* s_temp = (float*)(s_survivors + k);                  // [M] temp space
    
    // Initialize curand state for this tournament
    curandState rng_state;
    if (tid == 0) {
        curand_init(rng_seed + b, 0, 0, &rng_state);
    }
    __syncthreads();
    
    // Load tournament objectives to shared memory (coalesced access)
    if (tid < k) {
        const int32_t participant = tournament_indices[b * k + tid];
        if (valid_idx(participant, N)) {
            for (int m = 0; m < M; m++) {
                s_objectives[tid * M + m] = objectives[participant * M + m];
            }
        } else {
            // Invalid participant - set to worst possible values
            for (int m = 0; m < M; m++) {
                s_objectives[tid * M + m] = INFINITY;
            }
        }
    }
    __syncthreads();
    
    // Phase 1: Compute Pareto dominance relationships
    if (tid < k) {
        s_dominance_mask[tid] = true; // Assume non-dominated initially
        
        // Check if this individual is dominated by any other
        for (int j = 0; j < k; j++) {
            if (j == tid) continue;
            
            bool j_dominates_i = true;
            bool j_strictly_better = false;
            
            // Check dominance: j dominates i if j <= i in all objectives AND j < i in at least one
            for (int m = 0; m < M; m++) {
                float i_val = s_objectives[tid * M + m];
                float j_val = s_objectives[j * M + m];
                
                if (j_val > i_val) {
                    j_dominates_i = false;
                    break;
                }
                if (j_val < i_val) {
                    j_strictly_better = true;
                }
            }
            
            if (j_dominates_i && j_strictly_better) {
                s_dominance_mask[tid] = false;
                break;
            }
        }
    }
    __syncthreads();
    
    // Count non-dominated individuals
    __shared__ int non_dominated_count;
    if (tid == 0) {
        non_dominated_count = 0;
        for (int i = 0; i < k; i++) {
            if (s_dominance_mask[i]) non_dominated_count++;
        }
    }
    __syncthreads();
    
    // Handle simple cases
    if (non_dominated_count == 0) {
        // No non-dominated individuals - random selection
        if (tid == 0) {
            int winner_pos = curand(&rng_state) % k;
            winner_indices[b] = tournament_indices[b * k + winner_pos];
        }
        return;
    }
    
    if (non_dominated_count == 1) {
        // Exactly one non-dominated individual
        if (tid == 0) {
            for (int i = 0; i < k; i++) {
                if (s_dominance_mask[i]) {
                    winner_indices[b] = tournament_indices[b * k + i];
                    break;
                }
            }
        }
        return;
    }
    
    // Phase 2: ε-lexicase tie-breaking for multiple non-dominated individuals
    
    // Initialize survivors as non-dominated individuals
    if (tid < k) {
        s_survivors[tid] = s_dominance_mask[tid];
    }
    __syncthreads();
    
    // Compute epsilon values using MAD (Median Absolute Deviation) per objective
    __shared__ float s_epsilon[MAX_OBJECTIVES];
    
    for (int m = 0; m < M; m++) {
        if (tid == 0) {
            // Collect non-dominated values for this objective
            float values[MAX_TOURNAMENT_SIZE];
            int count = 0;
            for (int i = 0; i < k; i++) {
                if (s_dominance_mask[i]) {
                    values[count++] = s_objectives[i * M + m];
                }
            }
            
            if (count <= 1) {
                s_epsilon[m] = 0.0f;
            } else {
                // Compute median
                for (int i = 0; i < count - 1; i++) {
                    for (int j = i + 1; j < count; j++) {
                        if (values[i] > values[j]) {
                            float temp = values[i];
                            values[i] = values[j];
                            values[j] = temp;
                        }
                    }
                }
                float median = values[count / 2];
                
                // Compute MAD
                for (int i = 0; i < count; i++) {
                    values[i] = fabsf(values[i] - median);
                }
                for (int i = 0; i < count - 1; i++) {
                    for (int j = i + 1; j < count; j++) {
                        if (values[i] > values[j]) {
                            float temp = values[i];
                            values[i] = values[j];
                            values[j] = temp;
                        }
                    }
                }
                s_epsilon[m] = values[count / 2]; // MAD as epsilon
            }
        }
    }
    __syncthreads();
    
    // Random objective ordering for lexicase
    __shared__ int obj_order[MAX_OBJECTIVES];
    if (tid == 0) {
        // Initialize objective order
        for (int m = 0; m < M; m++) {
            obj_order[m] = m;
        }
        // Fisher-Yates shuffle
        for (int i = M - 1; i > 0; i--) {
            int j = curand(&rng_state) % (i + 1);
            int temp = obj_order[i];
            obj_order[i] = obj_order[j];
            obj_order[j] = temp;
        }
    }
    __syncthreads();
    
    // Apply ε-lexicase filtering
    for (int obj_idx = 0; obj_idx < M; obj_idx++) {
        __shared__ int survivor_count;
        __shared__ float min_obj_val;
        
        int current_obj = obj_order[obj_idx];
        
        // Count current survivors and find minimum value
        if (tid == 0) {
            survivor_count = 0;
            min_obj_val = INFINITY;
            for (int i = 0; i < k; i++) {
                if (s_survivors[i]) {
                    survivor_count++;
                    min_obj_val = fminf(min_obj_val, s_objectives[i * M + current_obj]);
                }
            }
        }
        __syncthreads();
        
        if (survivor_count <= 1) break; // Done if one or fewer survivors
        
        // Filter survivors within epsilon threshold
        float threshold = min_obj_val + s_epsilon[current_obj];
        if (tid < k && s_survivors[tid]) {
            if (s_objectives[tid * M + current_obj] > threshold) {
                s_survivors[tid] = false;
            }
        }
        __syncthreads();
    }
    
    // Final random selection among remaining survivors
    if (tid == 0) {
        int survivor_indices[MAX_TOURNAMENT_SIZE];
        int survivor_count = 0;
        
        for (int i = 0; i < k; i++) {
            if (s_survivors[i]) {
                survivor_indices[survivor_count++] = i;
            }
        }
        
        if (survivor_count > 0) {
            int chosen_idx = survivor_indices[curand(&rng_state) % survivor_count];
            winner_indices[b] = tournament_indices[b * k + chosen_idx];
        } else {
            // Fallback: random among non-dominated
            int non_dom_indices[MAX_TOURNAMENT_SIZE];
            int non_dom_count = 0;
            for (int i = 0; i < k; i++) {
                if (s_dominance_mask[i]) {
                    non_dom_indices[non_dom_count++] = i;
                }
            }
            int chosen_idx = non_dom_indices[curand(&rng_state) % non_dom_count];
            winner_indices[b] = tournament_indices[b * k + chosen_idx];
        }
    }
}

/**
 * Tournament index sampling kernel for pre-generating tournament participants
 */
__global__ void k_sample_tournament_indices(
    int32_t* __restrict__ tournament_indices,  // [B, k] output indices
    int N,                                     // Population size
    int k,                                     // Tournament size  
    int num_tournaments,                       // Number of tournaments
    int64_t rng_seed                          // Random seed
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_tournaments) return;
    
    curandState rng_state;
    curand_init(rng_seed, b, 0, &rng_state);
    
    // Sample k participants for this tournament
    for (int i = 0; i < k; i++) {
        tournament_indices[b * k + i] = curand(&rng_state) % N;
    }
}

/**
 * Host launcher for fused PPTS tournaments
 */
void launch_ppts_tournaments_kernel(
    const float* objectives,
    int64_t* winner_indices,
    const int32_t* tournament_indices,
    int N, int M, int num_tournaments, int k,
    int64_t rng_seed,
    cudaStream_t stream
) {
    // Validate kernel parameters
    if (k > MAX_TOURNAMENT_SIZE || M > MAX_OBJECTIVES) {
        printf("PPTS kernel error: k=%d > MAX_TOURNAMENT_SIZE=%d or M=%d > MAX_OBJECTIVES=%d\n", 
               k, MAX_TOURNAMENT_SIZE, M, MAX_OBJECTIVES);
        return;
    }
    
    // Calculate shared memory requirements
    size_t shared_mem_size = 
        k * M * sizeof(float) +      // s_objectives
        k * sizeof(bool) +           // s_dominance_mask  
        k * sizeof(bool) +           // s_survivors
        M * sizeof(float);           // s_temp
        
    if (shared_mem_size > MAX_SHARED_MEMORY) {
        printf("PPTS kernel error: shared memory requirement %zu exceeds limit %d\n", 
               shared_mem_size, MAX_SHARED_MEMORY);
        return;
    }
    
    // Launch configuration: one block per tournament
    dim3 grid(num_tournaments);
    dim3 block(std::min(k, 256)); // Use up to 256 threads per block
    
    k_ppts_fused_tournaments<<<grid, block, shared_mem_size, stream>>>(
        objectives, winner_indices, tournament_indices, N, M, k, rng_seed
    );
    
    CUDA_CHECK_ERRORS();
}

/**
 * Compute optimal chunk size based on available GPU memory
 */
int compute_optimal_chunk_size(int N, int M, int k, int num_tournaments) {
    // Query available GPU memory
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    
    // Estimate memory usage per tournament
    size_t mem_per_tournament = 
        k * M * sizeof(float) +           // objectives in shared mem
        k * sizeof(int32_t) +            // tournament indices
        sizeof(int64_t);                 // winner index
        
    // Use 80% of available memory to leave headroom
    size_t usable_memory = (free_bytes * 4) / 5;
    int max_chunk_size = static_cast<int>(usable_memory / mem_per_tournament);
    
    // Clamp to reasonable bounds
    max_chunk_size = std::max(1024, std::min(max_chunk_size, 65536));
    
    return std::min(max_chunk_size, num_tournaments);
}

/**
 * Elite selection implementation with memory optimization
 */
torch::Tensor ppts_select_elites_cuda(
    const torch::Tensor& objectives,
    int64_t num_elites,
    int64_t chunk_size,
    int64_t rng_seed
) {
    // Input validation
    TORCH_CHECK(objectives.is_cuda() && objectives.dtype() == torch::kFloat32,
                "objectives must be float32 CUDA tensor");
    TORCH_CHECK(objectives.dim() == 2, "objectives must be [N, M] tensor");
    TORCH_CHECK(objectives.is_contiguous(), "objectives must be contiguous");
    
    const int N = objectives.size(0);
    const int M = objectives.size(1);
    
    if (num_elites <= 0) {
        return torch::empty({0}, torch::TensorOptions().device(objectives.device()).dtype(torch::kInt64));
    }
    if (num_elites >= N) {
        return torch::arange(N, torch::TensorOptions().device(objectives.device()).dtype(torch::kInt64));
    }
    
    // Tournament size: larger for elites (higher selection pressure)
    const int k = std::min(static_cast<int>(std::max(2, static_cast<int>(std::ceil(std::log2(N))) * 2)), N);
    
    // Oversample to ensure uniqueness after deduplication
    const int num_tournaments = num_elites * 3;
    
    // Compute chunk size if not specified
    if (chunk_size <= 0) {
        chunk_size = compute_optimal_chunk_size(N, M, k, num_tournaments);
    }
    
    auto device = objectives.device();
    auto int64_opts = torch::TensorOptions().device(device).dtype(torch::kInt64);
    auto int32_opts = torch::TensorOptions().device(device).dtype(torch::kInt32);
    
    // Pre-allocate output tensors
    auto all_winners = torch::empty({num_tournaments}, int64_opts);
    
    // Process tournaments in chunks for memory efficiency
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    for (int start = 0; start < num_tournaments; start += chunk_size) {
        const int end = std::min(start + static_cast<int>(chunk_size), num_tournaments);
        const int current_chunk_size = end - start;
        
        // Sample tournament indices
        auto tournament_indices = torch::empty({current_chunk_size, k}, int32_opts);
        
        dim3 sample_grid((current_chunk_size + 255) / 256);
        dim3 sample_block(256);
        
        k_sample_tournament_indices<<<sample_grid, sample_block, 0, stream>>>(
            tournament_indices.data_ptr<int32_t>(), N, k, current_chunk_size, rng_seed + start
        );
        
        // Run tournaments
        launch_ppts_tournaments_kernel(
            objectives.data_ptr<float>(),
            all_winners.data_ptr<int64_t>() + start,
            tournament_indices.data_ptr<int32_t>(),
            N, M, current_chunk_size, k, rng_seed + start, stream
        );
    }
    
    // Deduplicate winners using sort + compact approach
    auto sort_result = torch::sort(all_winners);
    auto sorted_winners = std::get<0>(sort_result);
    auto unique_mask = torch::ones_like(sorted_winners, torch::TensorOptions().dtype(torch::kBool));
    if (sorted_winners.size(0) > 1) {
        auto diff = sorted_winners.slice(0, 1) != sorted_winners.slice(0, 0, -1);
        unique_mask.slice(0, 1) = diff;
    }
    auto unique_winners = sorted_winners.masked_select(unique_mask);
    
    if (unique_winners.size(0) >= num_elites) {
        return unique_winners.slice(0, 0, num_elites);
    }
    
    // If insufficient unique winners, run additional tournaments
    const int needed = num_elites - unique_winners.size(0);
    const int additional_tournaments = needed * 2;
    
    auto additional_winners = torch::empty({additional_tournaments}, int64_opts);
    
    for (int start = 0; start < additional_tournaments; start += chunk_size) {
        const int end = std::min(start + static_cast<int>(chunk_size), additional_tournaments);
        const int current_chunk_size = end - start;
        
        auto tournament_indices = torch::empty({current_chunk_size, k}, int32_opts);
        
        dim3 sample_grid((current_chunk_size + 255) / 256);
        dim3 sample_block(256);
        
        k_sample_tournament_indices<<<sample_grid, sample_block, 0, stream>>>(
            tournament_indices.data_ptr<int32_t>(), N, k, current_chunk_size, 
            rng_seed + num_tournaments + start
        );
        
        launch_ppts_tournaments_kernel(
            objectives.data_ptr<float>(),
            additional_winners.data_ptr<int64_t>() + start,
            tournament_indices.data_ptr<int32_t>(),
            N, M, current_chunk_size, k, rng_seed + num_tournaments + start, stream
        );
    }
    
    // Combine and deduplicate
    auto combined = torch::cat({unique_winners, additional_winners});
    auto combined_sort_result = torch::sort(combined);
    auto sorted_combined = std::get<0>(combined_sort_result);
    auto combined_unique_mask = torch::ones_like(sorted_combined, torch::TensorOptions().dtype(torch::kBool));
    if (sorted_combined.size(0) > 1) {
        auto diff = sorted_combined.slice(0, 1) != sorted_combined.slice(0, 0, -1);
        combined_unique_mask.slice(0, 1) = diff;
    }
    auto final_unique = sorted_combined.masked_select(combined_unique_mask);
    
    if (final_unique.size(0) >= num_elites) {
        return final_unique.slice(0, 0, num_elites);
    }
    
    // Fill remaining deterministically if still insufficient
    auto remaining = torch::arange(N, int64_opts);
    auto selected_mask = torch::zeros({N}, torch::TensorOptions().device(device).dtype(torch::kBool));
    selected_mask.index_put_({final_unique}, true);
    auto available = remaining.masked_select(~selected_mask);
    
    const int still_needed = num_elites - final_unique.size(0);
    if (available.size(0) > 0) {
        auto filler = available.slice(0, 0, std::min(still_needed, static_cast<int>(available.size(0))));
        return torch::cat({final_unique, filler}).slice(0, 0, num_elites);
    }
    
    return final_unique;
}

/**
 * Parent selection implementation (allows duplicates)
 */
torch::Tensor ppts_pick_parents_cuda(
    const torch::Tensor& objectives,
    int64_t num_parents,
    int64_t chunk_size,
    int64_t rng_seed
) {
    // Input validation
    TORCH_CHECK(objectives.is_cuda() && objectives.dtype() == torch::kFloat32,
                "objectives must be float32 CUDA tensor");
    TORCH_CHECK(objectives.dim() == 2, "objectives must be [N, M] tensor");
    TORCH_CHECK(objectives.is_contiguous(), "objectives must be contiguous");
    
    const int N = objectives.size(0);
    const int M = objectives.size(1);
    
    if (num_parents <= 0) {
        return torch::empty({0}, torch::TensorOptions().device(objectives.device()).dtype(torch::kInt64));
    }
    
    // Tournament size: standard size for parent selection
    const int k = std::min(static_cast<int>(std::max(2, static_cast<int>(std::ceil(std::log2(N))))), N);
    
    // Compute chunk size if not specified
    if (chunk_size <= 0) {
        chunk_size = compute_optimal_chunk_size(N, M, k, num_parents);
    }
    
    auto device = objectives.device();
    auto int64_opts = torch::TensorOptions().device(device).dtype(torch::kInt64);
    auto int32_opts = torch::TensorOptions().device(device).dtype(torch::kInt32);
    
    auto parent_indices = torch::empty({num_parents}, int64_opts);
    
    // Process tournaments in chunks
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    for (int start = 0; start < num_parents; start += chunk_size) {
        const int end = std::min(start + static_cast<int>(chunk_size), static_cast<int>(num_parents));
        const int current_chunk_size = end - start;
        
        // Sample tournament indices
        auto tournament_indices = torch::empty({current_chunk_size, k}, int32_opts);
        
        dim3 sample_grid((current_chunk_size + 255) / 256);
        dim3 sample_block(256);
        
        k_sample_tournament_indices<<<sample_grid, sample_block, 0, stream>>>(
            tournament_indices.data_ptr<int32_t>(), N, k, current_chunk_size, rng_seed + start
        );
        
        // Run tournaments
        launch_ppts_tournaments_kernel(
            objectives.data_ptr<float>(),
            parent_indices.data_ptr<int64_t>() + start,
            tournament_indices.data_ptr<int32_t>(),
            N, M, current_chunk_size, k, rng_seed + start, stream
        );
    }
    
    return parent_indices;
}