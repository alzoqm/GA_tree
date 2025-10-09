# GP Feature Pipeline — Usage Guide

이 문서는 GP 기반 단일(또는 조합) 특성 생성 파이프라인(gp_pipeline)을 빠르게 실행하고 재개(resume)하는 방법을 정리합니다.

## 빠른 시작
- 기본 실행
  - `python -m gp_pipeline.cli --config Config/gp_feature_pipeline.yaml`
- 체크포인트에서 재개
  - `python -m gp_pipeline.cli --config Config/gp_feature_pipeline.yaml --resume`

권장: 초기에는 `population.size`와 `generations`를 작게 설정하고 동작을 확인하세요. 이후 점진적으로 늘리면 시간을 절약할 수 있습니다.

## 요구 사항
- Python 패키지(최소): `deap`, `numpy`, `pandas`, `tqdm`, `pyyaml`
- 선택(백엔드/데모): `numexpr`(권장), `polars`
- 설치 예시
  - `pip install deap numpy pandas tqdm pyyaml numexpr`

## 데이터 & 설정 파일
- 설정 파일 경로: `--config`로 지정(예: `Config/gp_feature_pipeline.yaml`)
- 주요 데이터 필드
  - `data.path`: 데이터 파일 경로 (`.csv/.txt/.parquet/.pkl` 지원)
  - `data.feature_columns`: 특성 컬럼 목록
  - `data.target_column`: 타깃 컬럼. 값을 비우면 자동 생성(비율/위험 조정된 다중 지평 타깃)
  - `data.max_target_corr`: 타깃과 상관 높은 특성 제거 임계값(선택)
- 경로 해석 주의: `output.artifacts_dir`와 같은 경로들은 “설정 파일이 위치한 디렉터리” 기준으로 해석됩니다.

## 실행 옵션 (CLI)
- `--config PATH`: 설정 파일 경로
- `--resume`: 최근 체크포인트에서 이어서 실행

예시
- 새로 시작: `python -m gp_pipeline.cli --config Config/gp_feature_pipeline.yaml`
- 재개: `python -m gp_pipeline.cli --config Config/gp_feature_pipeline.yaml --resume`

## 구성(설정) 하이라이트
- `feature.mode`: `numeric` 또는 `boolean`
- `timeframe_primitives`
  - `enabled: true`이면 시계열 윈도우 연산자(EMA, Z-score 등) 활성화
  - 주의: 타임프레임 연산을 쓰면 백엔드 후보에 반드시 `numpy`가 포함되어야 합니다(`backend.primary` 또는 `fallbacks`)
- `population`
  - `size`, `generations`, `max_depth`, `mutation_prob`, `crossover_prob`, `seed` 등
- `backend`
  - `primary`: `numexpr` 권장, `fallbacks`에 `numpy` 포함 권장(타임프레임 사용 시 필수)
  - `low_fidelity_ratio`: 전체 데이터 일부만 샘플링하여 저충실도 평가(가속)
  - `full_eval_top_k`: 상위 k%는 전체 데이터로 재평가
- `validator`
  - `enabled: true` 시 통계적 유의성/강건성 지표 평가 수행
- `output`
  - `artifacts_dir`: 결과 저장 디렉터리(기본 `results/gp_pipeline`)
  - `save_best`, `save_population`: 최적/개체군 저장 옵션

## 체크포인트 & 재개
- 저장 위치: `<artifacts_dir>/checkpoints/gp/gp_gen_XXXX.json`
- 저장 시점: 각 세대의 재생산(offspring) 후 저장 → 재개 시 “다음 세대”부터 평가 시작
- 포함 정보: 다음 세대 인덱스, population(문자열 표현), run_timestamp, RNG 상태(python/numpy/subset_rng), 누적 history, best_records, metric_goals
- 재개 방법: `--resume` 플래그 사용 시 최신 체크포인트 자동 탐색/복원
- 강제 새 시작: 해당 체크포인트 디렉터리(`checkpoints/gp/`)를 삭제하거나 `--resume` 없이 실행
- 주의: 설정/프리미티브 구성이 크게 바뀌면 복원이 실패할 수 있으며, 이 경우 자동으로 새 실행으로 폴백됩니다.

## 출력 아티팩트
- 세대별 요약: `<artifacts_dir>/generations_YYYYMMDD_HHMMSS/generation_0000.json` 등
- 집계 결과(러너 종료 시 저장)
  - `hof_<timestamp>.json`: Hall of Fame(임계 통과 개체 중 상위)
  - `history_<timestamp>.json`: 세대별 최고/평균 메트릭 요약
  - `best_<timestamp>.json`: 임계 통과 후 집계 점수 기준 상위 개체 스냅샷
  - `population_<timestamp>.json`: 옵션(`output.save_population: true`)일 때 저장

## 팁
- 빠른 실험: `backend.low_fidelity_ratio`를 0.2–0.4로 설정해 빠르게 후보 선별 후 상위만 전체 평가
- 재현성: `population.seed`를 고정하면 동일한 초기 개체군을 생성합니다. 재개 시 RNG 상태까지 함께 복원합니다.
- 불안정 값 처리: 숫자형 모드에서 NaN/Inf는 0으로 대체, float32 범위 밖은 클리핑하여 오버플로 방지

## 문제 해결
- 데이터 컬럼 누락: `data.feature_columns`와 실제 파일 헤더가 일치하는지 확인
- 타임프레임 사용 오류: `backend` 후보에 `numpy` 포함 필요
- 경로 문제: 설정 파일 기준 상대 경로이므로, 설정 파일 위치 변경 시 경로도 함께 점검
- 성능/메모리: `population.size`, `max_depth`, `generations`를 줄여 단계적으로 확장

문의/개선 제안은 리포지토리 이슈에 남겨주세요. 즐거운 연구 되세요!
