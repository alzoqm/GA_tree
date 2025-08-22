# training/predictor.py (수정된 전체 코드)

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from models.model import GATreePop
from models.constants import ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT

try:
    import gatree_cuda
except ImportError:
    print("="*60)
    print(">>> 경고: 'gatree_cuda' 모듈을 찾을 수 없습니다.")
    print(">>> C++/CUDA 코드를 먼저 컴파일해야 합니다.")
    print(">>> 프로젝트 루트에서 다음 명령을 실행하세요:")
    print(">>> python setup.py build_ext --inplace")
    print("="*60)
    gatree_cuda = None

POSITION_TO_INT_MAP: Dict[str, int] = {
    'LONG': ROOT_BRANCH_LONG,
    'HOLD': ROOT_BRANCH_HOLD,
    'SHORT': ROOT_BRANCH_SHORT,
}

def build_adjacency_list_cuda(population: GATreePop,
                              sort_children: bool = True,
                              validate: bool = False,
                              strict_overflow: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build CSR adjacency (offsets, children) for the whole population on GPU.
    Returns properly formatted CSR with (B, N+1) offsets for the optimized predict kernel.
    - Optionally sorts children per parent (determinism)
    - Optionally validates invariants (bitmask per tree)
    - Optionally fails early if any parent count > max_children (strict_overflow=True)
    """
    if gatree_cuda is None:
        raise RuntimeError("gatree_cuda not loaded; build the extension first.")
    if not population.initialized:
        raise RuntimeError("Population is not initialized. Call make_population() first.")

    trees = population.population_tensor.to('cuda')
    if not trees.is_contiguous():
        trees = trees.contiguous()

    # Step 1: counts -> offsets (+ overflow mask)
    total_children, flat_offsets, overflow_mask = gatree_cuda.count_and_create_offsets(
        trees, population.max_children
    )

    if strict_overflow:
        bad = (overflow_mask != 0).nonzero(as_tuple=False).flatten()
        if bad.numel() > 0:
            raise RuntimeError(
                f"[Adjacency] Count overflow: {bad.numel()} trees exceeded max_children "
                f"(e.g., first few: {bad[:8].tolist()}). "
                f"Rebuild/repair before CSR fill."
            )

    # Step 2: indices
    indices = torch.empty(int(total_children), dtype=torch.int32, device=trees.device)
    if indices.numel() > 0:
        gatree_cuda.fill_child_indices(trees, flat_offsets, indices,
                                       population.max_children, sort_children)

    # Length sanity check
    if int(flat_offsets[-1].item()) != int(indices.numel()):
        raise RuntimeError("CSR length mismatch: offsets[-1] != child_indices.numel()")

    # Step 3: Convert to per-tree CSR format for optimized predict kernel
    B, N = population.pop_size, population.max_nodes
    
    # Create proper per-tree offsets tensor (B, N+1)
    per_tree_offsets = torch.zeros((B, N + 1), dtype=torch.int32, device=trees.device)
    
    # Fill per-tree offsets from flat offsets
    for tree_idx in range(B):
        tree_start = tree_idx * (N + 1)
        tree_end = tree_start + (N + 1)
        if tree_end <= flat_offsets.size(0):
            per_tree_offsets[tree_idx] = flat_offsets[tree_start:tree_end]
        else:
            # Handle case where flat_offsets doesn't have enough elements
            available = flat_offsets.size(0) - tree_start
            if available > 0:
                per_tree_offsets[tree_idx, :available] = flat_offsets[tree_start:tree_start + available]
    
    # Calculate max edges per tree for proper children tensor sizing
    edge_counts = per_tree_offsets[:, N]  # Last column contains total edge count per tree
    max_edges = edge_counts.max().item()
    print(f'max_edges: {max_edges}')
    # Reshape children to (B, Emax) format
    if max_edges > 0:
        # Pad children to ensure each tree has the same number of allocated edges
        per_tree_children = torch.zeros((B, max_edges), dtype=torch.int32, device=trees.device)
        
        # Fill children data per tree
        for tree_idx in range(B):
            start_idx = tree_idx * N if tree_idx == 0 else per_tree_offsets[tree_idx - 1, N].item()
            end_idx = per_tree_offsets[tree_idx, N].item()
            tree_edge_count = end_idx - start_idx
            
            if tree_edge_count > 0:
                per_tree_children[tree_idx, :tree_edge_count] = indices[start_idx:end_idx]
    else:
        per_tree_children = torch.zeros((B, 1), dtype=torch.int32, device=trees.device)

    if validate:
        mask = torch.empty(population.pop_size, dtype=torch.int32, device=trees.device)
        gatree_cuda.validate_adjacency(trees, flat_offsets, indices,
                                       population.max_children, population.max_depth, mask)
        bad = (mask != 0).nonzero(as_tuple=False).flatten()
        if bad.numel() > 0:
            raise RuntimeError(
                f"[Adjacency] Validation failed for {bad.numel()} trees "
                f"(first few: {bad[:8].tolist()}). "
                f"Mask bits: MIXED(1), LEAF!=ACT(2), ACT_HAS_CHILD(4), SINGLE_ACT(8), "
                f"DEPTH(16), OVERFLOW(32), BAD_PARENT(64), ROOT(128), ROOT_LEAF(256)."
            )

    # Always validate trees after CUDA kernels
    try:
        gatree_cuda.validate_trees(trees.contiguous())
    except Exception as e:
        raise RuntimeError(f"validate_trees failed after building adjacency: {e}")

    torch.cuda.synchronize()
    return per_tree_offsets, per_tree_children


def predict_population_cuda(
    population: GATreePop,
    feature_values: pd.Series,
    current_positions: List[str],
    adj_offsets: torch.Tensor,
    adj_indices: torch.Tensor,
    device: str = 'cuda'
) -> torch.Tensor | None:
    """
    사전 생성된 인접 리스트를 이용하여 전체 GATree 집단의 예측을 병렬로 수행합니다.
    
    새 CUDA 커널 인터페이스와 호환되는 최적화된 예측 함수:
    - BFS queue와 visited 버퍼를 별도로 할당하여 메모리 안전성 보장
    - CSR (Compressed Sparse Row) 형식의 adjacency 데이터 사용
    - 각 트리당 하나의 블록으로 확장성 최적화
    """
    if gatree_cuda is None:
        print("오류: gatree_cuda 모듈이 로드되지 않아 예측을 수행할 수 없습니다.")
        return None
        
    if not population.initialized:
        raise RuntimeError("Population 객체가 초기화되지 않았습니다. make_population()을 먼저 호출하세요.")

    if len(current_positions) != population.pop_size:
        raise ValueError(f"current_positions의 길이({len(current_positions)})가 "
                         f"population의 크기({population.pop_size})와 일치하지 않습니다.")

    population_tensor = population.population_tensor.to(device)
    
    # Memory validation checks
    if not population_tensor.is_contiguous():
        population_tensor = population_tensor.contiguous()
    
    # Validate tensor dimensions
    expected_shape = (population.pop_size, population.max_nodes, population_tensor.shape[2])
    if population_tensor.shape != expected_shape:
        raise ValueError(f"Population tensor shape {population_tensor.shape} doesn't match expected {expected_shape}")
    
    ordered_features = feature_values.reindex(population.all_features).values
    numeric_features = ordered_features.astype(np.float32)
    features_tensor = torch.tensor(numeric_features, dtype=torch.float32, device=device)
    positions_int = [POSITION_TO_INT_MAP[pos] for pos in current_positions]
    positions_tensor = torch.tensor(positions_int, dtype=torch.int32, device=device)
    results_tensor = torch.zeros((population.pop_size, 4), dtype=torch.float32, device=device)
    bfs_queue_buffer = torch.zeros(
        (population.pop_size, population.max_nodes), 
        dtype=torch.int32, 
        device=device
    )
    visited_buffer = torch.zeros(
        (population.pop_size, population.max_nodes), 
        dtype=torch.int32, 
        device=device
    )
    gatree_cuda.predict(
        population_tensor,
        features_tensor,
        positions_tensor,
        adj_offsets,
        adj_indices,
        results_tensor,
        bfs_queue_buffer,
        visited_buffer
    )
    # Ensure CUDA operations complete before returning
    torch.cuda.synchronize()
    
    # Validate trees after CUDA predict
    try:
        gatree_cuda.validate_trees(population_tensor.contiguous())
    except Exception as e:
        raise RuntimeError(f"validate_trees failed after predict: {e}")


    return results_tensor

# # =======================================================
# # ---           이 함수들을 테스트하기 위한 예제         ---
# # =======================================================
# if __name__ == '__main__':
#     # 이 스크립트를 직접 실행하면 새로운 2단계 예측 파이프라인의 동작을 테스트합니다.
    
#     if gatree_cuda is None:
#         print("\n테스트를 건너뜁니다.")
#     else:
#         print("===== [개선된] CUDA 예측 파이프라인 테스트 시작 =====")
#         # 1. 테스트용 GATreePop 객체 생성
#         POP_SIZE = 10
#         MAX_NODES = 512
#         MAX_DEPTH = 10
#         MAX_CHILDREN = 5

#         from models.model import FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL
        
#         print(f"\n1. {POP_SIZE}개체로 구성된 GATreePop 생성 중...")
#         population = GATreePop(
#             pop_size=POP_SIZE,
#             max_nodes=MAX_NODES,
#             max_depth=MAX_DEPTH,
#             max_children=MAX_CHILDREN,
#             feature_num=FEATURE_NUM,
#             feature_comparison_map=FEATURE_COMPARISON_MAP,
#             feature_bool=FEATURE_BOOL
#         )
#         population.make_population()
#         print("   GATreePop 생성 완료.")

#         # 2. [신규] 1단계: 인접 리스트 생성
#         print("\n2. `build_adjacency_list_cuda` 함수 호출 (준비 단계)...")
#         try:
#             if not torch.cuda.is_available():
#                  raise SystemExit("오류: CUDA를 사용할 수 있는 GPU가 없습니다. 테스트를 중단합니다.")
            
#             adj_offsets, adj_indices = build_adjacency_list_cuda(population)
            
#             print("   인접 리스트 생성 완료.")
#             print(f"   - 반환된 Offset 텐서 Shape: {adj_offsets.shape}")
#             print(f"   - 반환된 Child Indices 텐서 Shape: {adj_indices.shape}")

#         except Exception as e:
#             print(f"\n인접 리스트 생성 중 오류 발생: {e}")
#             exit()

#         # 3. 예측에 사용할 가상 데이터 생성
#         print("\n3. 예측에 사용할 가상 데이터 생성 중...")
#         all_feature_names = population.all_features
#         dummy_feature_data = {name: torch.randn(1).item() * 10 for name in all_feature_names}
#         feature_values = pd.Series(dummy_feature_data)
#         current_positions = [random.choice(['LONG', 'HOLD', 'SHORT']) for _ in range(POP_SIZE)]
#         print(f"   - 생성된 피처 수: {len(feature_values)}")

#         # 4. [수정] 2단계: 예측 함수 호출
#         print("\n4. `predict_population_cuda` 함수 호출 (실행 단계)...")
#         try:
#             action_results = predict_population_cuda(
#                 population=population,
#                 feature_values=feature_values,
#                 current_positions=current_positions,
#                 adj_offsets=adj_offsets,
#                 adj_indices=adj_indices,
#                 device='cuda'
#             )
            
#             print("   함수 실행 완료.")

#             # 5. 결과 확인
#             print("\n5. 결과 확인...")
#             print(f"   - 반환된 텐서의 Shape: {action_results.shape}")
            
#             print("\n--- 예측 결과 (일부) ---")
#             print("Tree | Position | Action Type | Param 2 | Param 3 | Param 4")
#             print("----------------------------------------------------------------")
#             ACTION_TYPE_MAP[ACTION_NOT_FOUND] = 'NOT_FOUND'

#             for i in range(min(POP_SIZE, 5)):
#                 pos = current_positions[i]
#                 res = action_results[i]
#                 action_type = int(res[0].item())
#                 action_name = ACTION_TYPE_MAP.get(action_type, 'UNKNOWN')
                
#                 print(f"{i:4d} | {pos:<8s} | {action_name:<11s} ({action_type:d}) | {res[1]:7.4f} | {res[2]:7.1f} | {res[3]:7.4f}")

#             print("\n===== 테스트 성공 =====")

#         except Exception as e:
#             print(f"\n예측 실행 중 오류 발생: {e}")
