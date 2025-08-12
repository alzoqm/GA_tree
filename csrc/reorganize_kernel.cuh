// /csrc/reorganize_kernel.cuh (신규 파일)
#pragma once

#include <torch/extension.h>

/**
 * @brief GPU에서 집단 텐서의 단편화를 제거합니다.
 * 
 * 이 함수는 스트림 압축(Stream Compaction) 알고리즘을 사용하여
 * NODE_TYPE_UNUSED로 표시된 노드들을 제거하고, 활성 노드들만 텐서의
 * 앞부분으로 모읍니다. 이 과정에서 부모-자식 관계가 깨지지 않도록
 * COL_PARENT_IDX도 올바르게 업데이트합니다.
 * 
 * @param population_tensor 재구성할 집단 텐서. 
 *                          Shape: (pop_size, max_nodes, node_dim).
 *                          이 텐서는 in-place로 수정됩니다.
 */
void reorganize_population_cuda(torch::Tensor population_tensor);