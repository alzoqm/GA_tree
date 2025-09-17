# /training/README.md

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
  - 청산 체크(`_check_liquidation`): 라스트가 아닌 마크 프라이스 근사(보수적 프록시: 바의 저가/고가)로 유지증거금 대비 마진 평가. 청산 시 초기증거금 손실 + 청산 정리 수수료(`liq_clearance_fee_rate`) 반영. Lot 원장 및 노드 집계가 활성화된 경우 CUDA 커널 `check_liquidation_with_attrib`에서 청산 손익을 Lot 노출 비율로 액션 노드에 분배하여 `node_realized_pnl_sum`/`node_exposure_sum`/`node_win_count`를 갱신(비활성화 시 기존 `check_liquidation_batch` 사용).
    - 액션 실행(`_execute_actions`): 예측된 액션을 다음 캔들 시가(`next_open_price`) 기준으로 체결, 슬리피지/테이커 수수료 적용. 신규 진입/전부/부분 청산/추가 진입/포지션 플립 지원
    - `step(market_data, next_open_price, predicted_actions)`: 한 바 진행(펀딩/청산 후 액션 실행). 룩어헤드 바이어스 방지를 위해 다음 바 시가로 체결. 에쿼티는 마크‑투‑마켓(MtM)으로 계산(typical price: (H+L+C)/3 프록시)
    - `get_final_metrics(minimum_date=40)`: 평균 수익, PF(상한 처리), 이벤트 기반 승률, 최대 낙폭, 누적 복리 값, 트레이드 기반 승률(신규) 집계
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
        'use_real_mark_price': False,
        'equity_mtm_basis': 'mark_proxy',
        'liq_clearance_fee_rate': 0.003,
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
print(metrics.shape)  # (B, 6): mean, PF, winrate(event), maxDD, compound, winrate(trade)
```

## TradingEnvironment 규칙 상세

- 체결 가격 및 수수료/슬리피지:
  - 체결 기준은 항상 다음 캔들의 시가 `next_open_price`.
  - 매수체결가: `buy_exec_price = next_open_price * (1 + fixed_slippage_rate)`.
  - 매도체결가: `sell_exec_price = next_open_price * (1 - fixed_slippage_rate)`.
- 테이커 수수료: `fee = position_size_ratio * leverage * taker_fee_rate`.
  - 청산 정리 수수료: `liq_fee = (enter_ratio * leverage) * liq_clearance_fee_rate` (청산 발생 시 추가 차감)

- 포지션/상태 변수:
  - `pos_list`: 0=HOLD, 1=SHORT, 2=LONG.
  - `price_list`: 평균 진입가(없으면 -1.0).
  - `leverage_ratio`: 현재 레버리지(없으면 -1).
  - `enter_ratio`: 현재 투입 비율(없으면 -1.0). 전체 자본 대비 비율로 0~1 범위.
  - `additional_count`: 피라미딩(추가 진입) 횟수.

- 최소 제약(기본값은 `experiment_config.yaml`의 `evaluation.simulation_env`로부터 주입):
  - `min_entry_ratio`: 0.05 (5%). 신규 진입 및 플립 시 최소 진입 비율로 적용.
  - `max_entry_ratio`: 1.0 (100%). 신규 진입 및 플립 시 최대 진입 비율 상한.
  - `min_leverage`: 2. 신규 진입 및 플립 시 레버리지 하한.
  - `max_leverage`: 100. 신규 진입 및 플립 시 레버리지 상한.
  - `min_additional_enter_ratio`: 0.05 (5%). 추가 진입의 최소 단위 및 남은 자본 용량 판정에 사용.

- 신규 진입(ACTION_NEW_LONG / ACTION_NEW_SHORT):
  - 입력 비율은 `[min_entry_ratio, max_entry_ratio]`로 클램프.
  - 레버리지는 `[min_leverage, max_leverage]` 범위로 클램프.
  - 수수료 차감 후 포지션, 평균가, 레버리지, 진입 비율 초기화.

- 포지션 플립(ACTION_FLIP_POSITION):
  - 기존 포지션을 청산(PnL 및 수수료 정산) 후, 동일 바에서 즉시 반대 포지션으로 진입.
  - 새 진입 비율은 `[min_entry_ratio, max_entry_ratio]`로, 레버리지는 `[min_leverage, max_leverage]`로 강제.
  - 청산 수수료(기존 비율/레버리지) + 신규 진입 수수료(새 비율/레버리지) 모두 반영.

- 전부 청산(ACTION_CLOSE_ALL):
  - LONG: `((sell_exec_price - entry_price) / entry_price) * leverage * enter_ratio - fee`.
  - SHORT: `((entry_price - buy_exec_price) / entry_price) * leverage * enter_ratio - fee`.
  - 상태값 초기화(포지션 0, 가격/레버리지/비율 초기화).

- 부분 청산(ACTION_CLOSE_PARTIAL):
  - 요청 청산 비율을 `ratio`라 할 때, 부분 청산 후 남을 비율은 `enter_ratio * (1 - ratio)`.
  - 남을 비율이 `<= min_entry_ratio(=5%)`이면, 1단계 판단으로 전부 청산으로 대체하여 한 번에 처리.
  - 그 외에는 부분 청산만 수행: LONG/SHORT 각각 위의 전부 청산 식에서 `enter_ratio` 대신 `closed_margin = enter_ratio * ratio` 사용 후 `enter_ratio -= closed_margin`.

- 추가 진입(ACTION_ADD_POSITION):
  - 남은 자본 용량 `remaining = max(0, 1.0 - enter_ratio)`가 `>= min_additional_enter_ratio(=5%)`일 때만 수행.
  - 실제 추가 비율은 `actual = min(max(requested, min_additional_enter_ratio), remaining)`.
  - 수수료 차감 후, 새 평균가: `new_avg = (old_price * old_ratio + exec_price * actual) / (old_ratio + actual + 1e-9)`.
  - `enter_ratio += actual`, `additional_count += 1`.

- 펀딩 수수료(_apply_funding_fees):
  - 0/8/16시 타임스탬프 경계에서 적용.
  - `funding_pnl = funding_rate * leverage`를 기반으로, LONG은 비용(차감), SHORT은 수익(가산)으로 `profit`에 반영하며 규모는 `enter_ratio`에 비례.

- 청산 체크(_check_liquidation):
  - 마크 프라이스 근사(보수적: 저가/고가 경계)로 강제 청산 여부 평가.
  - 청산 발생 시 초기증거금 전액 손실에 더해 청산 정리 수수료 반영. 트레이드 기반 집계는 해당 시점을 1회 종료로 기록.

- 에쿼티/드로다운:
  - 에쿼티 = 누적 실현 PnL + 미실현 PnL(MtM, typical price 기반)으로 계산하여 드로다운 갱신.
  - 복리 값은 실현손익 이벤트를 기준으로 갱신(옵션에 따라 대체 가능).

- 승률 집계 체계:
  - 이벤트 기반: 스텝별 실현 이벤트 기준(부분청산 포함)으로 승/패 집계.
  - 트레이드 기반(신규): 비보유→보유 시 트레이드 시작, 부분청산은 누적만, 전부청산/플립/청산 시점에 1회만 승패 판정.

- 기타:
  - 모든 계산은 텐서 연산으로 배치(염색체 단위) 처리.
  - 드로다운/복리/수익 집계는 `get_final_metrics` 내에서 계산되며, 유효 거래 수(`minimum_trades`) 미달 샘플은 패널티 적용.
