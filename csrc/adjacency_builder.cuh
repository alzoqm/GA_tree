// csrc/adjacency_builder.cuh (수정됨)
#pragma once

#include <torch/extension.h>
#include <vector>
#include <utility> // for std::pair

// 1단계: 각 노드의 자식 수를 세고, 총 자식 수와 오프셋 배열을 반환하는 함수
std::pair<long, torch::Tensor> count_and_create_offsets_cuda(
    const torch::Tensor& population_tensor
);

// 2단계: 사전에 할당된 child_indices 텐서의 내용을 채우는 함수
void fill_child_indices_cuda(
    const torch::Tensor& population_tensor,
    const torch::Tensor& offset_array,
    torch::Tensor& child_indices // 입력이자 출력 (in-place modification)
);