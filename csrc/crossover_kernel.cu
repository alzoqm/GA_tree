// csrc/crossover_kernel.cu
#include "crossover_kernel.cuh"

// This main crossover kernel file now includes specialized implementations
// All CUDA kernel implementations have been moved to their respective specialized files:
// - node_crossover_kernel.cu: Contains node parameter swapping and contextual masking
// - subtree_crossover_kernel.cu: Contains subtree crossover operations  
// - root_crossover_kernel.cu: Contains root branch crossover and tree repair operations
