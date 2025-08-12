// csrc/adjacency_builder.cuh (수정된 파일)
#pragma once

#include <torch/extension.h>
#include <vector>
#include <utility> // for std::pair

// 1단계: 각 노드의 자식 수를 세고, 총 자식 수와 오프셋 배열을 반환하는 함수
std::pair<long, torch::Tensor> count_and_create_offsets_cuda(
    const torch::Tensor& population_tensor
);

// 2단계: 사전에 할당된 child_indices 텐서의 내용을 채우고 정렬하는 함수
// [수정] max_children 파라미터 추가
void fill_child_indices_cuda(
    const torch::Tensor& population_tensor,
    const torch::Tensor& offset_array,
    torch::Tensor& child_indices, // 입력이자 출력 (in-place modification)
    int max_children
);

// [수정] 자식 리스트를 정렬하는 커널을 호출하는 런처 함수 선언. max_children 추가
void launch_sort_children_kernel(
    const int* offset_ptr,
    int* child_indices_ptr,
    int pop_size,
    int max_nodes,
    int max_children
);