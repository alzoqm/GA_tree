# training/predictor.py (수정된 전체 코드)

import torch
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

# [수정] 2단계 통신을 사용하여 인접 리스트를 생성하는 함수
def build_adjacency_list_cuda(population: GATreePop) -> Tuple[torch.Tensor, torch.Tensor] | None:
    """
    GPU를 사용하여 GATree 집단의 인접 리스트(CSR 형식)를 병렬로 생성합니다.
    내부적으로 2단계 통신을 통해 C++에서의 텐서 생성을 최소화합니다.
    """
    if gatree_cuda is None:
        print("오류: gatree_cuda 모듈이 로드되지 않아 인접 리스트를 생성할 수 없습니다.")
        return None, None

    if not population.initialized:
        raise RuntimeError("Population 객체가 초기화되지 않았습니다. make_population()을 먼저 호출하세요.")

    population_tensor_cuda = population.population_tensor.to('cuda')
    
    # --- 1단계: 총 자식 수와 오프셋 배열 계산 ---
    total_children_count, offset_array = gatree_cuda.count_and_create_offsets(
        population_tensor_cuda
    )

    # --- 2단계: Python에서 child_indices 텐서 할당 후 내용 채우기 ---
    # `torch.empty`를 사용하여 GPU에 필요한 크기만큼의 메모리만 할당
    child_indices = torch.empty(
        total_children_count, 
        dtype=torch.int32, 
        device='cuda'
    )

    if total_children_count > 0:
        gatree_cuda.fill_child_indices(
            population_tensor_cuda,
            offset_array,
            child_indices # In-place로 내용이 채워짐
        )
    
    return offset_array, child_indices


# predict_population_cuda 함수는 이전 답변과 동일하게 유지됩니다.
# (호출 인터페이스 변경 없음)
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
    ordered_features = feature_values.reindex(population.all_features).values
    features_tensor = torch.tensor(ordered_features, dtype=torch.float32, device=device)
    positions_int = [POSITION_TO_INT_MAP[pos] for pos in current_positions]
    positions_tensor = torch.tensor(positions_int, dtype=torch.int64, device=device)
    next_indices = population.return_next_idx()
    next_indices_tensor = torch.tensor(next_indices, dtype=torch.int32, device=device)
    results_tensor = torch.zeros((population.pop_size, 4), dtype=torch.float32, device=device)
    bfs_queue_buffer = torch.zeros(
        (population.pop_size, population.max_nodes), 
        dtype=torch.int32, 
        device=device
    )
    
    gatree_cuda.predict(
        population_tensor,
        features_tensor,
        positions_tensor,
        next_indices_tensor,
        adj_offsets,
        adj_indices,
        results_tensor,
        bfs_queue_buffer
    )

    return results_tensor

# =======================================================
# ---           이 함수들을 테스트하기 위한 예제         ---
# =======================================================
if __name__ == '__main__':
    # 이 스크립트를 직접 실행하면 새로운 2단계 예측 파이프라인의 동작을 테스트합니다.
    
    if gatree_cuda is None:
        print("\n테스트를 건너뜁니다.")
    else:
        print("===== [개선된] CUDA 예측 파이프라인 테스트 시작 =====")
        # 1. 테스트용 GATreePop 객체 생성
        POP_SIZE = 10
        MAX_NODES = 512
        MAX_DEPTH = 10
        MAX_CHILDREN = 5

        from models.model import FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL
        
        print(f"\n1. {POP_SIZE}개체로 구성된 GATreePop 생성 중...")
        population = GATreePop(
            pop_size=POP_SIZE,
            max_nodes=MAX_NODES,
            max_depth=MAX_DEPTH,
            max_children=MAX_CHILDREN,
            feature_num=FEATURE_NUM,
            feature_comparison_map=FEATURE_COMPARISON_MAP,
            feature_bool=FEATURE_BOOL
        )
        population.make_population()
        print("   GATreePop 생성 완료.")

        # 2. [신규] 1단계: 인접 리스트 생성
        print("\n2. `build_adjacency_list_cuda` 함수 호출 (준비 단계)...")
        try:
            if not torch.cuda.is_available():
                 raise SystemExit("오류: CUDA를 사용할 수 있는 GPU가 없습니다. 테스트를 중단합니다.")
            
            adj_offsets, adj_indices = build_adjacency_list_cuda(population)
            
            print("   인접 리스트 생성 완료.")
            print(f"   - 반환된 Offset 텐서 Shape: {adj_offsets.shape}")
            print(f"   - 반환된 Child Indices 텐서 Shape: {adj_indices.shape}")

        except Exception as e:
            print(f"\n인접 리스트 생성 중 오류 발생: {e}")
            exit()

        # 3. 예측에 사용할 가상 데이터 생성
        print("\n3. 예측에 사용할 가상 데이터 생성 중...")
        all_feature_names = population.all_features
        dummy_feature_data = {name: torch.randn(1).item() * 10 for name in all_feature_names}
        feature_values = pd.Series(dummy_feature_data)
        current_positions = [random.choice(['LONG', 'HOLD', 'SHORT']) for _ in range(POP_SIZE)]
        print(f"   - 생성된 피처 수: {len(feature_values)}")

        # 4. [수정] 2단계: 예측 함수 호출
        print("\n4. `predict_population_cuda` 함수 호출 (실행 단계)...")
        try:
            action_results = predict_population_cuda(
                population=population,
                feature_values=feature_values,
                current_positions=current_positions,
                adj_offsets=adj_offsets,
                adj_indices=adj_indices,
                device='cuda'
            )
            
            print("   함수 실행 완료.")

            # 5. 결과 확인
            print("\n5. 결과 확인...")
            print(f"   - 반환된 텐서의 Shape: {action_results.shape}")
            
            print("\n--- 예측 결과 (일부) ---")
            print("Tree | Position | Action Type | Param 2 | Param 3 | Param 4")
            print("----------------------------------------------------------------")
            ACTION_TYPE_MAP[ACTION_NOT_FOUND] = 'NOT_FOUND'

            for i in range(min(POP_SIZE, 5)):
                pos = current_positions[i]
                res = action_results[i]
                action_type = int(res[0].item())
                action_name = ACTION_TYPE_MAP.get(action_type, 'UNKNOWN')
                
                print(f"{i:4d} | {pos:<8s} | {action_name:<11s} ({action_type:d}) | {res[1]:7.4f} | {res[2]:7.1f} | {res[3]:7.4f}")

            print("\n===== 테스트 성공 =====")

        except Exception as e:
            print(f"\n예측 실행 중 오류 발생: {e}")