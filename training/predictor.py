# training/predictor.py (수정된 전체 코드)

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from models.model import GATreePop
from models.constants import ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT

try:
    import gatree_cuda_compat as gatree_cuda
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
    Returns properly formatted CSR with (B, N+1) offsets and (B, Emax) children for the optimized predict kernel.
    """
    if gatree_cuda is None:
        raise RuntimeError("gatree_cuda not loaded; build the extension first.")
    if not population.initialized:
        raise RuntimeError("Population is not initialized. Call make_population() first.")

    trees = population.population_tensor.to('cuda')
    if not trees.is_contiguous():
        trees = trees.contiguous()

    # Step 1: Count children and produce a single flat CSR offsets array
    # Note: offsets are laid out as (B*N+1), with tree b occupying [b*N : (b+1)*N],
    # and its end sentinel at index (b+1)*N.
    # Count children and build CSR offsets. When strict_overflow is False,
    # pass a very large sentinel to avoid spurious overflow flags; we will
    # size buffers from the observed max degree instead.
    count_limit = 512 if strict_overflow else (1 << 29)
    total_children, flat_offsets, overflow_mask = gatree_cuda.count_and_create_offsets(
        trees, int(count_limit)
    )

    if strict_overflow:
        bad = (overflow_mask != 0).nonzero(as_tuple=False).flatten()
        if bad.numel() > 0:
            raise RuntimeError(
                f"[Adjacency] Count overflow: {bad.numel()} trees exceeded max_children "
                f"(e.g., first few: {bad[:8].tolist()}). "
                f"Rebuild/repair before CSR fill."
            )

    # Step 2: Fill the flat child indices array
    indices = torch.empty(int(total_children), dtype=torch.int32, device=trees.device)
    if indices.numel() > 0:
        # Choose a safe max_children for the fill kernel:
        # - If strict_overflow is False, allow arbitrarily large branching by using the
        #   observed maximum per-parent degree so sorting buffers are sized safely.
        # - Otherwise, keep the conservative 512 default.
        if not strict_overflow:
            # Compute max degree across all parents from the CSR offsets
            degs = flat_offsets[1:] - flat_offsets[:-1]
            max_deg = int(degs.max().item()) if degs.numel() > 0 else 0
            max_children_for_fill = max(1, max_deg)
        else:
            max_children_for_fill = 512

        gatree_cuda.fill_child_indices(
            trees, flat_offsets, indices, max_children_for_fill, sort_children
        )

    # Sanity: flat arrays must agree in total length
    if int(flat_offsets[-1].item()) != int(indices.numel()):
        raise RuntimeError("CSR length mismatch: offsets[-1] != child_indices.numel()")

    # --- Reconstruct per-tree CSR for predict kernel ---
    B, N = population.pop_size, population.max_nodes

    # 1) Per-tree total edge counts using boundary sentinels at [0, N, 2N, ...]
    tree_boundaries = flat_offsets[::N]           # shape: (B+1,)
    edge_counts = tree_boundaries[1:] - tree_boundaries[:-1]  # shape: (B,)
    max_edges = int(edge_counts.max().item()) if edge_counts.numel() > 0 else 0

    # 2) Allocate outputs
    per_tree_offsets = torch.zeros((B, N + 1), dtype=torch.int32, device=trees.device)
    per_tree_children = torch.zeros((B, max_edges), dtype=torch.int32, device=trees.device) if max_edges > 0 else torch.zeros((B, 0), dtype=torch.int32, device=trees.device)

    # 3) Populate per-tree CSR
    for b in range(B):
        # Flat offsets slice for this tree is [b*N : (b+1)*N] plus end sentinel at (b+1)*N
        start_off = flat_offsets[b * N]
        tree_offs = flat_offsets[b * N : (b + 1) * N + 1]
        # Make offsets relative to the start of this tree so they index into per_tree_children[b]
        per_tree_offsets[b, :] = tree_offs - start_off

        # Fill children segment for this tree
        num_edges = int(per_tree_offsets[b, N].item())
        if num_edges > 0:
            s = int(start_off.item())
            e = s + num_edges
            per_tree_children[b, :num_edges] = indices[s:e]


    if validate:
        mask = torch.empty(population.pop_size, dtype=torch.int32, device=trees.device)
        # Validator expects flat CSR (offsets_flat, indices)
        gatree_cuda.validate_adjacency(trees, flat_offsets, indices, mask)
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
    feature_values: pd.Series | None,
    current_positions: List[str] | torch.Tensor,
    adj_offsets: torch.Tensor,
    adj_indices: torch.Tensor,
    device: str = 'cuda',
    *,
    # Optional preallocated buffers for reuse
    results: torch.Tensor | None = None,
    bfs: torch.Tensor | None = None,
    visited: torch.Tensor | None = None,
    positions_tensor: torch.Tensor | None = None,
    features_tensor: torch.Tensor | None = None,
    validate: bool = False,
    sync: bool = False,
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

    # Validate current_positions (list or tensor)
    if isinstance(current_positions, list):
        if len(current_positions) != population.pop_size:
            raise ValueError(
                f"current_positions의 길이({len(current_positions)})가 population의 크기({population.pop_size})와 일치하지 않습니다.")
    else:
        if current_positions.dim() != 1 or current_positions.numel() != population.pop_size:
            raise ValueError("current_positions tensor must be shape (B,)")

    # Avoid per-call device copy if already on device
    population_tensor = population.population_tensor
    if str(population_tensor.device) != str(device):
        population_tensor = population_tensor.to(device)
    
    # Memory validation checks
    if not population_tensor.is_contiguous():
        population_tensor = population_tensor.contiguous()
    
    # Validate tensor dimensions
    expected_shape = (population.pop_size, population.max_nodes, population_tensor.shape[2])
    if population_tensor.shape != expected_shape:
        raise ValueError(f"Population tensor shape {population_tensor.shape} doesn't match expected {expected_shape}")
    
    # Prepare or reuse buffers
    B, N = population.pop_size, population.max_nodes

    # Features
    if features_tensor is None:
        if feature_values is None:
            raise ValueError("Either feature_values or features_tensor must be provided")
        ordered = feature_values.reindex(population.all_features).values.astype(np.float32)
        features_tensor = torch.as_tensor(ordered, dtype=torch.float32, device=device)
    else:
        if features_tensor.dtype != torch.float32:
            features_tensor = features_tensor.to(dtype=torch.float32)
        if str(features_tensor.device) != str(device):
            features_tensor = features_tensor.to(device)

    # Positions
    if positions_tensor is None:
        if isinstance(current_positions, list):
            positions_int = [POSITION_TO_INT_MAP[p] for p in current_positions]
            positions_tensor = torch.tensor(positions_int, dtype=torch.int32, device=device)
        else:
            positions_tensor = current_positions.to(device=device, dtype=torch.int32)
    else:
        if str(positions_tensor.device) != str(device) or positions_tensor.dtype != torch.int32:
            positions_tensor = positions_tensor.to(device=device, dtype=torch.int32)

    # Results
    if results is None:
        results_tensor = torch.empty((B, 4), dtype=torch.float32, device=device)
    else:
        results_tensor = results
        if results_tensor.shape != (B, 4):
            raise ValueError("results buffer must be shape (B,4)")
        if results_tensor.dtype != torch.float32:
            raise ValueError("results buffer must be float32")

    # BFS / visited
    if bfs is None:
        bfs_queue_buffer = torch.empty((B, N), dtype=torch.int32, device=device)
    else:
        bfs_queue_buffer = bfs
        if bfs_queue_buffer.shape != (B, N):
            raise ValueError("bfs buffer must be shape (B,N)")
    if visited is None:
        visited_buffer = torch.zeros((B, N), dtype=torch.int32, device=device)
    else:
        visited_buffer = visited
        if visited_buffer.shape != (B, N):
            raise ValueError("visited buffer must be shape (B,N)")
        visited_buffer.zero_()
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
    if sync:
        torch.cuda.synchronize()
    if validate:
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
