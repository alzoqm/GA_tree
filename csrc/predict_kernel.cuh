// csrc/predict_kernel.cuh - Optimized GA-Tree Prediction Kernel
#pragma once

#include <torch/extension.h>

/**
 * GPU-accelerated prediction for GA-Tree populations with safety guarantees
 * 
 * This function performs batch prediction on GA-Tree populations with the following guarantees:
 * - No illegal memory access even with malformed inputs
 * - No infinite loops through visited bitmap and step limits
 * - Proper CSR validation and bounds checking
 * - Scalable to hundreds of thousands of trees
 * 
 * Architecture: One block per tree for optimal scalability
 * Traversal: BFS from root branch to first reachable ACTION node
 * 
 * @param trees       (B,N,D) float32, CUDA, contiguous - population tensor
 * @param features    (F,) float32, CUDA - shared feature vector
 * @param positions   (B,) int32, CUDA - root branch types per tree
 * @param offsets     (B,N+1) int32, CUDA - CSR row offsets per tree
 * @param children    (B,Emax) int32, CUDA - CSR column indices per tree
 * @param results     (B,4) float32, CUDA - output [action, p2, p3, p4]
 * @param bfs_q       (B,N) int32, CUDA - BFS queue buffer per tree
 * @param visited     (B,N) int32, CUDA - visited flags per tree (0/1)
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