// csrc/predict_kernel.cuh
// This header file declares the CUDA function for performing predictions with a GA-Tree.

#pragma once

#include <torch/extension.h>

/**
 * @brief Performs GPU-accelerated prediction for a batch of GA-Trees with safety guarantees.
 *
 * This function performs batch prediction on a population of GA-Trees. It is optimized for
 * performance and includes several safety features to prevent common issues such as illegal
 * memory access and infinite loops.
 *
 * Key Features:
 * - **Safe Memory Access**: Prevents illegal memory access even with malformed inputs.
 * - **Infinite Loop Protection**: Uses a visited bitmap and step limits to prevent infinite loops.
 * - **CSR Validation**: Includes proper validation and bounds checking for the CSR data structure.
 * - **Scalability**: Designed to be scalable to hundreds of thousands of trees.
 *
 * @param trees The population tensor (B, N, D).
 * @param features The shared feature vector (F,).
 * @param positions The root branch types for each tree (B,).
 * @param offsets The CSR row offsets for each tree (B, N+1).
 * @param children The CSR column indices for each tree (B, Emax).
 * @param results The output tensor for the prediction results (B, 4).
 * @param bfs_q A buffer for the BFS queue for each tree (B, N).
 * @param visited A buffer for the visited flags for each tree (B, N).
 */
void predict_cuda(
    torch::Tensor trees,     // (B,N,D) float32, CUDA, contiguous
    torch::Tensor features,  // (F,)     float32, CUDA
    torch::Tensor positions, // (B,)     int32,  CUDA
    torch::Tensor offsets,   // (B,N+1)  int32,  CUDA
    torch::Tensor children,  // (B,Emax) int32,  CUDA
    torch::Tensor results,   // (B,4)    float32, CUDA
    torch::Tensor bfs_q,     // (B,N)    int32, CUDA
    torch::Tensor visited    // (B,N)    int32, CUDA
);