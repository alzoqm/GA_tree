# training/predictor.py

import torch
import pandas as pd
from typing import List, Dict
import random # 테스트 코드용

# 프로젝트 구조에 따라 model, gatree_cuda 모듈을 임포트합니다.
# 이 파일이 프로젝트 루트에서 실행된다고 가정합니다.
from models.model import GATreePop
# [수정] 상수는 constants 모듈에서 직접 임포트
from models.constants import ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT

# setup.py를 통해 빌드된 CUDA 확장 모듈을 임포트합니다.
# 이 모듈이 존재하지 않으면 ImportError가 발생합니다.
try:
    import gatree_cuda
except ImportError:
    print("="*60)
    print(">>> 경고: 'gatree_cuda' 모듈을 찾을 수 없습니다.")
    print(">>> C++/CUDA 코드를 먼저 컴파일해야 합니다.")
    print(">>> 프로젝트 루트에서 다음 명령을 실행하세요:")
    print(">>> python setup.py build_ext --inplace")
    print("="*60)
    # 학습 환경에서는 필수적이므로 예외를 발생시킬 수 있습니다.
    # raise ImportError("gatree_cuda extension not found. Please compile it first.")
    gatree_cuda = None

# 포지션 문자열을 C++/CUDA에서 사용하는 정수 상수로 변환하기 위한 맵
POSITION_TO_INT_MAP: Dict[str, int] = {
    'LONG': ROOT_BRANCH_LONG,
    'HOLD': ROOT_BRANCH_HOLD,
    'SHORT': ROOT_BRANCH_SHORT,
}

def predict_population_cuda(
    population: GATreePop,
    feature_values: pd.Series,
    current_positions: List[str],
    device: str = 'cuda'
) -> torch.Tensor | None:
    """
    GPU를 사용하여 전체 GATree 집단의 예측을 병렬로 수행합니다.

    이 함수는 GA의 적합도 평가 단계에서 각 개체의 행동을 빠르게 결정하기 위해
    컴파일된 C++/CUDA 확장 모듈을 호출합니다.

    Args:
        population (GATreePop): 예측을 수행할 GATree 집단 객체.
        feature_values (pd.Series): 단일 타임스텝에 대한 모든 피처 값을 담은 Series.
                                   (인덱스: 피처 이름, 값: 피처 값)
        current_positions (List[str]): 집단의 각 트리에 대한 현재 포지션.
                                      (e.g., ['LONG', 'HOLD', 'SHORT', ...])
        device (str): 연산을 수행할 장치 (기본값: 'cuda').

    Returns:
        torch.Tensor | None: 각 트리의 예측 결과를 담은 2D 텐서.
                             Shape: (pop_size, 4)
                             Columns: [action_type, param_2, param_3, param_4]
                             CUDA 모듈이 없으면 None을 반환합니다.
    """
    # 0. 모듈 가용성 및 입력 유효성 검사
    if gatree_cuda is None:
        print("오류: gatree_cuda 모듈이 로드되지 않아 예측을 수행할 수 없습니다.")
        return None
        
    if not population.initialized:
        raise RuntimeError("Population 객체가 초기화되지 않았습니다. make_population()을 먼저 호출하세요.")

    if len(current_positions) != population.pop_size:
        raise ValueError(f"current_positions의 길이({len(current_positions)})가 "
                         f"population의 크기({population.pop_size})와 일치하지 않습니다.")

    # 1. CUDA 커널에 전달할 텐서 준비
    
    # 1-1. population_tensor: (pop_size, max_nodes, 7), float32
    population_tensor = population.population_tensor.to(device)

    # 1-2. features_tensor: (num_features,), float32
    ordered_features = feature_values.reindex(population.all_features).values
    features_tensor = torch.tensor(ordered_features, dtype=torch.float32, device=device)

    # 1-3. positions_tensor: (pop_size,), int64 (long)
    positions_int = [POSITION_TO_INT_MAP[pos] for pos in current_positions]
    positions_tensor = torch.tensor(positions_int, dtype=torch.int64, device=device)

    # 1-4. next_indices_tensor: (pop_size,), int32
    next_indices = population.return_next_idx()
    next_indices_tensor = torch.tensor(next_indices, dtype=torch.int32, device=device)
    
    # 1-5. results_tensor: (pop_size, 4), float32
    results_tensor = torch.zeros((population.pop_size, 4), dtype=torch.float32, device=device)

    # [신규] 1-6. BFS 큐 버퍼 텐서 생성
    # 각 트리가 최대 max_nodes만큼의 큐 공간을 가질 수 있도록 버퍼를 할당합니다.
    bfs_queue_buffer = torch.zeros(
        (population.pop_size, population.max_nodes), 
        dtype=torch.int32, 
        device=device
    )
    
    # 2. 컴파일된 CUDA 확장 모듈 함수 호출
    # [수정] bfs_queue_buffer를 새로운 인자로 추가하여 호출
    gatree_cuda.predict(
        population_tensor,
        features_tensor,
        positions_tensor,
        next_indices_tensor,
        results_tensor,
        bfs_queue_buffer  # [신규] 추가된 인자
    )

    # 3. 결과 반환
    return results_tensor

# =======================================================
# ---           이 함수를 테스트하기 위한 예제         ---
# =======================================================
if __name__ == '__main__':
    # 이 스크립트를 직접 실행하면 함수의 사용법을 보여주고 정상 동작을 테스트합니다.
    
    if gatree_cuda is None:
        print("\n테스트를 건너뜁니다.")
    else:
        print("===== CUDA 예측 함수 테스트 시작 =====")
        # 1. 테스트용 GATreePop 객체 생성
        POP_SIZE = 10
        MAX_NODES = 512
        MAX_DEPTH = 10
        MAX_CHILDREN = 5

        # model.py에 정의된 기본 피처 설정 사용
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

        # 2. 테스트용 가상 데이터 생성
        print("\n2. 예측에 사용할 가상 데이터 생성 중...")
        # 2-1. 피처 값 (pandas.Series)
        all_feature_names = population.all_features
        dummy_feature_data = {name: torch.randn(1).item() * 10 for name in all_feature_names}
        feature_values = pd.Series(dummy_feature_data)
        print(f"   - 생성된 피처 수: {len(feature_values)}")

        # 2-2. 현재 포지션 (List[str])
        current_positions = [random.choice(['LONG', 'HOLD', 'SHORT']) for _ in range(POP_SIZE)]
        print(f"   - 생성된 포지션 정보 (첫 5개): {current_positions[:5]}")

        # 3. CUDA 예측 함수 호출
        print("\n3. `predict_population_cuda` 함수 호출...")
        try:
            # GPU가 사용 가능한지 확인
            if not torch.cuda.is_available():
                 raise SystemExit("오류: CUDA를 사용할 수 있는 GPU가 없습니다. 테스트를 중단합니다.")
            
            # 함수 실행
            action_results = predict_population_cuda(
                population=population,
                feature_values=feature_values,
                current_positions=current_positions,
                device='cuda'
            )
            
            print("   함수 실행 완료.")

            # 4. 결과 확인
            print("\n4. 결과 확인...")
            print(f"   - 반환된 텐서의 Shape: {action_results.shape}")
            print(f"   - 반환된 텐서의 Device: {action_results.device}")
            print(f"   - 반환된 텐서의 Dtype: {action_results.dtype}")
            
            print("\n--- 예측 결과 (일부) ---")
            print("Tree | Position | Action Type | Param 2 | Param 3 | Param 4")
            print("----------------------------------------------------------------")
            # 결과 해석을 위한 Action Type -> 이름 맵
            from models.constants import ACTION_TYPE_MAP, ACTION_NOT_FOUND
            ACTION_TYPE_MAP[ACTION_NOT_FOUND] = 'NOT_FOUND' # 0번 추가

            for i in range(min(POP_SIZE, 5)):
                pos = current_positions[i]
                res = action_results[i]
                action_type = int(res[0].item())
                action_name = ACTION_TYPE_MAP.get(action_type, 'UNKNOWN')
                
                print(f"{i:4d} | {pos:<8s} | {action_name:<11s} ({action_type:d}) | {res[1]:7.4f} | {res[2]:7.1f} | {res[3]:7.4f}")

            print("\n===== 테스트 성공 =====")

        except Exception as e:
            print(f"\n테스트 중 오류 발생: {e}")