# verify_predictions.py (최종 수정본 2)

import torch
import pandas as pd
import random
import sys
import time
import os

# --- 1. 프로젝트 모듈 임포트 ---
# 이 스크립트는 프로젝트 루트 디렉터리에서 실행되어야 합니다.
try:
    from models.model import GATreePop
    from models.model import FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL, get_all_features
    from models.constants import ACTION_NOT_FOUND, ACTION_TYPE_MAP
    from training.predictor import build_adjacency_list_cuda, predict_population_cuda
except ImportError as e:
    print(f"FATAL: 모듈을 임포트할 수 없습니다. 스크립트를 프로젝트 루트에서 실행하고 있는지 확인하세요. 에러: {e}")
    sys.exit(1)

# --- 2. 검증 환경 설정 ---
POP_SIZE = 100
MAX_NODES = 512
MAX_DEPTH = 15
MAX_CHILDREN = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_test_population() -> GATreePop:
    """테스트를 위한 GATreePop 객체를 생성하고 초기화합니다."""
    print("--- 1. 테스트용 GATreePop 생성 중... ---")
    
    all_features = get_all_features(FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL)
    
    population = GATreePop(
        pop_size=POP_SIZE,
        max_nodes=MAX_NODES,
        max_depth=MAX_DEPTH,
        max_children=MAX_CHILDREN,
        feature_num=FEATURE_NUM,
        feature_comparison_map=FEATURE_COMPARISON_MAP,
        feature_bool=FEATURE_BOOL,
        all_features=all_features
    )
    population.make_population()
    print(f"{POP_SIZE}개의 트리로 구성된 집단 생성 완료.\n")
    return population

def generate_dummy_inputs(population: GATreePop) -> tuple[pd.Series, list[str]]:
    """예측에 사용할 가상의 피처 값과 현재 포지션을 생성합니다."""
    print("--- 2. 예측에 사용할 가상 입력 데이터 생성 중... ---")
    
    all_feature_names = population.all_features
    
    # [수정] bool 타입 대신 float(1.0, 0.0)을 사용하여 데이터 타입 문제를 해결합니다.
    dummy_feature_data = {
        name: random.uniform(-100, 100) if name not in population.feature_bool else random.choice([1.0, 0.0])
        for name in all_feature_names
    }
    feature_values = pd.Series(dummy_feature_data)
    
    # 생성된 시리즈의 dtype 확인 (디버깅용)
    # print(f"생성된 feature_values의 dtype: {feature_values.dtype}") # float64가 되어야 함
    
    current_positions = [random.choice(['LONG', 'HOLD', 'SHORT']) for _ in range(population.pop_size)]
    
    print(f"랜덤 피처 {len(feature_values)}개 및 포지션 {len(current_positions)}개 생성 완료.\n")
    return feature_values, current_positions

def run_python_predictions(population: GATreePop, features: pd.Series, positions: list[str]) -> torch.Tensor:
    """순수 Python으로 구현된 GATree.predict를 사용하여 결과를 계산합니다."""
    print("--- 3. Python 기반 예측 실행 중... ---")
    
    python_results = []
    
    start_time = time.time()
    for i, tree in enumerate(population.population):
        current_pos = positions[i]
        result_tuple = tree.predict(features, current_pos)
        
        action_type = result_tuple[0] if result_tuple[0] is not None else ACTION_NOT_FOUND
        
        python_results.append((action_type, *result_tuple[1:]))
        
    end_time = time.time()
    print(f"Python 예측 완료. 소요 시간: {end_time - start_time:.4f} 초")

    return torch.tensor(python_results, dtype=torch.float32)

def run_cuda_predictions(population: GATreePop, features: pd.Series, positions: list[str]) -> torch.Tensor | None:
    """CUDA 커널을 사용하여 결과를 병렬로 계산합니다."""
    if DEVICE != 'cuda':
        print("경고: CUDA를 사용할 수 없어 CUDA 예측을 건너뜁니다.")
        return None
        
    print("--- 4. CUDA 기반 예측 실행 중... ---")
    
    start_time = time.time()
    
    adj_offsets, adj_indices = build_adjacency_list_cuda(population)
    if adj_offsets is None:
        print("오류: CUDA에서 인접 리스트 생성에 실패했습니다.")
        return None
    
    results_tensor = predict_population_cuda(
        population=population,
        feature_values=features,
        current_positions=positions,
        adj_offsets=adj_offsets,
        adj_indices=adj_indices,
        device=DEVICE
    )
    if results_tensor is None:
        print("오류: CUDA 예측 실행에 실패했습니다.")
        return None
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"CUDA 예측 완료. 소요 시간: {end_time - start_time:.4f} 초")
    
    return results_tensor.cpu()

def verify():
    """메인 검증 로직을 실행합니다."""
    if DEVICE == 'cpu':
        print("="*60)
        print("경고: CUDA 사용 가능 GPU를 찾을 수 없습니다. 검증을 중단합니다.")
        print("="*60)
        return

    population = create_test_population()
    features, positions = generate_dummy_inputs(population)
    
    python_results = run_python_predictions(population, features, positions)
    cuda_results = run_cuda_predictions(population, features, positions)
    
    if cuda_results is None:
        return
        
    print("\n--- 5. 결과 비교 ---")
    
    are_identical = torch.allclose(python_results, cuda_results, atol=1e-6)
    
    if are_identical:
        print("\n========================================")
        print("✅ 성공: Python과 CUDA 예측 결과가 모두 일치합니다!")
        print("========================================")
    else:
        print("\n===================================================")
        print("❌ 실패: Python과 CUDA 예측 결과 간에 불일치가 발견되었습니다.")
        print("===================================================")
        
        diff_mask = ~torch.isclose(python_results, cuda_results, atol=1e-6)
        mismatch_rows_mask = diff_mask.any(dim=1)
        mismatch_indices = mismatch_rows_mask.nonzero(as_tuple=True)[0]
        
        print(f"\n총 {POP_SIZE}개 중 {len(mismatch_indices)}개의 개체에서 불일치 발견:")
        
        for idx in mismatch_indices[:10]: 
            i = idx.item()
            py_res = python_results[i]
            cu_res = cuda_results[i]
            
            py_action_name = ACTION_TYPE_MAP.get(int(py_res[0].item()), 'UNKNOWN')
            cu_action_name = ACTION_TYPE_MAP.get(int(cu_res[0].item()), 'UNKNOWN')
            
            print(f"\n--- 불일치 인덱스: {i} (Position: {positions[i]}) ---")
            print(f"  - Python 결과: "
                  f"Action={py_action_name}({py_res[0]:.0f}), P2={py_res[1]:.6f}, P3={py_res[2]:.6f}, P4={py_res[3]:.6f}")
            print(f"  - CUDA   결과: "
                  f"Action={cu_action_name}({cu_res[0]:.0f}), P2={cu_res[1]:.6f}, P3={cu_res[2]:.6f}, P4={cu_res[3]:.6f}")
            
            param_diff = []
            if not torch.isclose(py_res[0], cu_res[0], atol=1e-6): param_diff.append("ActionType (P1)")
            if not torch.isclose(py_res[1], cu_res[1], atol=1e-6): param_diff.append("Param2")
            if not torch.isclose(py_res[2], cu_res[2], atol=1e-6): param_diff.append("Param3")
            if not torch.isclose(py_res[3], cu_res[3], atol=1e-6): param_diff.append("Param4")
            print(f"  - 다른 파라미터: {', '.join(param_diff)}")

if __name__ == "__main__":
    verify()
