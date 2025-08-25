// csrc/crossover_kernel.cuh
#pragma once
#include <torch/extension.h>

// Include specialized crossover headers
#include "node_crossover_kernel.cuh"
#include "subtree_crossover_kernel.cuh"
#include "root_crossover_kernel.cuh"
