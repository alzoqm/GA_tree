// csrc/crossover_kernel.cuh
// This header file serves as a central point for including all crossover-related kernel headers.
// By including this single file, other parts of the codebase can get access to all
// the necessary crossover function declarations.

#pragma once
#include <torch/extension.h>

// Include specialized crossover headers. This modular approach helps in keeping
// the codebase organized and easy to maintain.
#include "node_crossover_kernel.cuh"
#include "subtree_crossover_kernel.cuh"
#include "root_crossover_kernel.cuh"
