// csrc/crossover_kernel.cu
// This file serves as the main entry point for the crossover kernels.
// It includes the specialized crossover implementations from other files.

#include "crossover_kernel.cuh"

// All CUDA kernel implementations for crossover have been modularized into
// the following files for better organization and maintainability:
//
// - node_crossover_kernel.cu: Contains kernels for node parameter swapping
//   and contextual masking, which are fundamental operations for various
//   crossover types.
//
// - subtree_crossover_kernel.cu: Implements the logic for subtree crossover,
//   a key operator for exchanging genetic material between trees.
//
// - root_crossover_kernel.cu: Handles the specifics of root branch crossover
//   and the necessary repair operations to maintain tree validity after
//   the crossover.
