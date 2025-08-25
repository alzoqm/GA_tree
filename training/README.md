# training/README.md

GA 트리 집단의 예측을 CUDA로 가속하고, 거래 시뮬레이션 환경에서 피트니스를 계산합니다.

## 파일 구성

- `predictor.py`: GPU 인접 리스트 생성 및 배치 예측
  - `build_adjacency_list_cuda(population, sort_children=True, validate=False, strict_overflow=True)`:
    - `population.population_tensor`로부터 `gatree_cuda.count_and_create_offsets`, `fill_child_indices`를 이용해 CSR 인접 리스트를 GPU에서 생성
    - 최적화된 예측 커널에 맞춘 per‑tree CSR 반환: `per_tree_offsets(B, N+1)`, `per_tree_children(B, Emax)`
    - 자식 수 초과/구조 검증 옵션 제공
  - `predict_population_cuda(population, feature_values, current_positions, adj_offsets, adj_indices, device='cuda')`:
    - `population.all_features` 순서에 맞춰 피처를 정렬하고, 현재 포지션(LONG/HOLD/SHORT)을 인코딩하여 `gatree_cuda.predict` 호출
    - 트리당 `(4,)` 결과 텐서 반환: `[action_type, param2, param3, param4]`
  - 비고: CUDA 디바이스와 컴파일된 확장 모듈(`gatree_cuda_compat`)이 필요하며, 모듈 미존재 시 경고 출력

- `trading_env.py`: 벡터화된 백테스트/시뮬레이션 환경
  - `TradingEnvironment`:
    - 염색체별 상태: 포지션/진입가/레버리지/진입비율/추가진입/보유기간/수익누적/드로다운/복리 등
    - 펀딩 수수료(`_apply_funding_fees`): 0/8/16시 적용
    - 청산 체크(`_check_liquidation`): 현재 캔들의 고가/저가를 사용해 유지증거금 대비 마진 평가
    - 액션 실행(`_execute_actions`): 예측된 액션을 다음 캔들 시가(`next_open_price`) 기준으로 체결, 슬리피지/테이커 수수료 적용. 신규 진입/전부/부분 청산/추가 진입/포지션 플립 지원
    - `step(market_data, next_open_price, predicted_actions)`: 한 바 진행(펀딩/청산 후 액션 실행). 룩어헤드 바이어스 방지를 위해 다음 바 시가로 체결
    - `get_final_metrics(minimum_date=40)`: 평균 수익, PF(상한 처리), 승률, 최대 낙폭, 누적 복리 값 집계
  - 피트니스 유틸:
    - `calculate_fitness(metrics, weights)`: 지표를 방향성에 맞춰 min‑max 정규화 후 가중합, 유효하지 않은 샘플 패널티 적용
    - `fitness_fn(...)`: 시간 구동 루프에서 예측→환경 step을 수행하고 최종 지표 반환
    - `generation_valid(...)`: 세대별 평가 루프. 매 세대 인접 리스트를 생성하고 피트니스 계산/체크포인트 관리

## 참고 사항

- `python setup.py build_ext --inplace`로 `gatree_cuda_compat`를 컴파일해야 하며 CUDA GPU가 필요합니다.
- 예측 시 피처 순서는 학습 시와 동일해야 합니다(`population.all_features`).
- 체결 가격은 다음 캔들의 시가를 사용하며, 슬리피지/수수료는 설정으로 조절 가능합니다.

## 예시

### 1) 확장 모듈 컴파일

```bash
python setup.py build_ext --inplace
```

### 2) 인접 리스트 생성 + 배치 예측

```python
import pandas as pd
from training.predictor import build_adjacency_list_cuda, predict_population_cuda

# 이미 초기화된 population (models 참고)
adj_offsets, adj_indices = build_adjacency_list_cuda(population, validate=True)

# 더미 피처 값과 포지션 (실사용 시 실제 피처/포지션 사용)
feature_values = pd.Series({k: 0.0 for k in population.all_features})
current_positions = ['HOLD'] * population.pop_size

actions = predict_population_cuda(
    population=population,
    feature_values=feature_values,
    current_positions=current_positions,
    adj_offsets=adj_offsets,
    adj_indices=adj_indices,
    device='cuda'
)
print(actions.shape)  # (B, 4)
```

### 3) 피트니스 계산 루프

```python
from training.predictor import fitness_fn

# 데이터프레임 `data`는 인덱스가 DatetimeIndex이며, `population.all_features` 컬럼이 포함되어야 함
evaluation_config = {
    'simulation_env': {
        'taker_fee_rate': 0.0004,
        'maintenance_margin_rate': 0.005,
        'fixed_slippage_rate': 0.0005,
        'min_additional_enter_ratio': 0.05,
        'max_additional_entries': 3,
    },
    'fitness_function': {
        'weights': [0.35, 0.25, 0.2, 0.1, 0.1],
        'minimum_trades': 40,
    }
}

adj_offsets, adj_indices = build_adjacency_list_cuda(population)
metrics = fitness_fn(
    population=population,
    data=data,
    all_feature_names=population.all_features,
    adj_offsets=adj_offsets,
    adj_indices=adj_indices,
    start_data_cnt=100,
    stop_data_cnt=len(data)-1,
    device='cuda',
    evaluation_config=evaluation_config,
)
print(metrics.shape)  # (B, 5): mean, PF, winrate, maxDD, compound
```
