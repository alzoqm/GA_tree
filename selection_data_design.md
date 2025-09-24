# selection_data.py: 고급 피처 선택 설계안 v2 (피드백 반영)

본 문서는 기존 설계(버전 v1)를 “실전 투입” 기준으로 업그레이드한 v2 설계서다. 두 개의 피드백 보고서에서 제안한 사항을 반영하여, 계산 현실성(Phased rollout), 시계열 일관 검정과 크로스피팅, 다중가설 결합·FDR 통제 규칙, Knockoffs의 시계열 적응형 변종, 다양성 보장 목적함수 명세, 희소 불리언 처리, 재현성/감사 메타데이터, CLI/설정 스키마를 강화했다.

## 0. 변경 요약(핵심)
- 단계적 도입 로드맵: MVP→Stage 2→Stage 3로 점진 실행·위험 관리.
- Cross-fitting 강화: 스크리닝·선택·캘리브레이션을 out-of-fold로 분리, 선택-후-검정 편향 완화.
- 시계열 일관 검정: HSIC는 블록/와일드 부트스트랩, dCor는 고차 독립성 t-검정·블록 순열 지원.
- 다중가설·의존 p-값: Empirical Brown/Simes 결합 + BH/BY 2단계 규칙을 명문화.
- Knockoffs 고도화: Copula/Nonparanormal + Robust/Approx + TSKI(시간의존) + 그룹/멀티레이어 옵션과 폴백 체인.
- 다양성: DPP 커널 정의 고정 및 대안(시설위치 서브모듈러, 1−1/e 보장) 추가.
- 불리언 이벤트: Fisher/G-test, OR/RR·Cliff’s δ, 블록 부트스트랩 CI.
- VIF 재정의: 절대 컷 대신 부분상관/부분R²/조건수·OOS 영향 병기.
- 재현성: 메타해시, git 커밋, seed/환경/패키지 버전, dry-run, 원자적 쓰기 확대.
- 설정: `selection_pipeline` 섹션 신설 및 단일 진실 공급원(SSOT)로 CPCV 공유.

## 1. 목표와 범위(동일)
- 목표: 전처리 산출물(최종 피처 프레임, `model_config_path`)로부터 학습 직전 “최종 피처 집합”을 엄격히 선정하고, 결과를 ① `model_config_path`에 반영, ② `Config/feature_classification_active.yaml`에 스냅샷으로 기록.
- 비범위: 학습 루프/모델 구조 변경 없음. 본 문서는 선택 파이프라인·스키마만 규정.

## 2. 상위 아키텍처 및 I/O(요약)
- 실행 위치: `preprocess_data.py` 이후, `main.py` 이전.
- 입력: 최종 피처 프레임(pkl/csv), `model_config_path`, `Config/experiment_config.yaml`, `Config/feature_classification_active.yaml`.
- 출력: 축소된 `model_config_path`, 강화된 `feature_classification_active.yaml(selection_snapshot)`, 선택 리포트·아티팩트.

## 3. 단계적 도입 로드맵(Phased Implementation)
- 1단계(MVP, 즉시 가동):
  - 무결성 게이트(결측/상수/희소/중복/완전상관 제거),
  - 랭크 상관 기반 1차 스크리닝(Spearman/Kendall),
  - 부분상관/그래픽컬 라쏘 요약으로 대표 피처 선택,
  - 최종 VIF·부분R² 확인 + 로버스트 분위(0.5–99.5%) 범위 캘리브레이션.
- 2단계(신뢰도 강화):
  - 안정 선택(Subsamping L1; PFER 상한 기반 튜닝) 또는 Model‑X Knockoffs(q 제어),
  - 크로스피팅으로 선택-후-검정 편향 완화,
  - 다양성(서브모듈러 facility-location) 우선 도입.
- 3단계(고성능·비선형 확장):
  - HSIC(랜덤 프로세스/블록 부트스트랩)·dCor t‑검정,
  - TSKI/Robust Knockoffs, DPP로 다양성 고도화,
  - 대규모 최적화(Gram 캐시, 블록화, 저랭크 근사, GPU 가속).

## 4. CPCV + Cross‑fitting 규범
- Purge/Embargo: `training_strategy.max_hold_period_bars`를 절대 바 단위로 해석, 타임프레임 환산 규칙 명시(실제 구현 시 로그에 공식 기록).
- 모든 통계량은 “train 구간에서만” 추정. 집계는 out‑of‑fold.
- 각 CPCV train fold 내부에서 2‑way split(또는 K‑fold)로 스크리닝/선택/캘리브레이션의 데이터를 분리하여 double‑dipping 방지.
- 최소 샘플 제약: `selection_pipeline.min_bars_per_split` 이상이어야 검정/선택 수행.

## 5. 스크리닝(시계열 일관 검정)
- 타깃: 포워드 로그수익 r_{t→t+h}, h∈H(예: {1,3,12,48}·5m).
- 검정:
  - HSIC: RBF 커널, 블록/와일드 부트스트랩으로 p‑값; 대용량 시 Block‑HSIC Lasso.
  - dCor: 고차 독립성 t‑검정 + 블록/서큘러 순열 옵션.
  - 순위계수/단조성: Spearman ρ; 효과크기(Cliff’s δ) 병기.
- 결합·보정: 동일 피처의 (h×fold) p‑값을 Empirical Brown(의존 강) 또는 Simes(의존 약)로 결합 → 피처 간 BH/BY로 FDR 제어.

## 6. 선택기(Selection)
- Model‑X Knockoffs:
  - 생성모형: Copula/Nonparanormal, Robust/Approx Knockoffs, TSKI(시계열) 옵션화.
  - 통계 W: LCD/SMX 등 선택 가능, Knockoff/Knockoff+ 임계 지원.
  - 그룹/멀티레이어: 지표 패밀리별 그룹 선택 허용.
  - 폴백 체인: TSKI→Robust→Copula→안정 선택.
- 안정 선택(Subsamping L1):
  - PFER 상한 우선 정의 → (평균 선택수 q, cut‑off) 역산 튜닝.
  - 고상관 클러스터 보완(클러스터 안정 선택) 옵션.

## 7. 다양성(De‑redundancy)
- DPP: PSD 유사도 커널 K = exp(−λ·(1−|partial‑corr|)), λ 기본값/탐색범위 문서화.
- 대안: 시설위치 서브모듈러 + 그리디(1−1/e 보장), 패밀리 상한(m개) 제약 지원.

## 8. 불리언 피처 특화
- 희소 이벤트: 최소 발생 `n_min` 및 인터어라이벌 CV/버스티니스 자동 임계 추정.
- 검정: 표본 충분 시 G‑test/LRT, 희소 시 Fisher exact. 효과크기 OR/RR(95% CI), 연속 반응 차이는 Cliff’s δ.
- 모든 유의성은 블록 부트스트랩으로 보정. 패밀리는 그룹 선택으로 단순화.

## 9. 공선성·범위 캘리브레이션
- 공선성 판단: VIF는 참고로만; 부분상관/부분R²/조건수 및 OOS 성능 영향과 함께 의사결정.
- 범위: 로버스트 분위(기본 0.5–99.5%)에 MAD 기반 마진을 자동 가산(폴드 간 변동이 큰 피처에 더 크게).

## 10. 재현성·감사 메타데이터
- 스냅샷 메타해시: 데이터 경로, 행수, 열 목록, CPCV seed/파라미터, 타임프레임, 모델 설정 해시를 결합한 SHA‑256.
- git 커밋 해시, 난수 seed, CPU/GPU/스레드, 주요 패키지 버전(freeze) 기록.
- 원자적 쓰기(tmp→move)와 `.bak‑TIMESTAMP` 백업; `--dry-run` 모드로 사전 검토.

## 11. 성능·가속 계획
- 프로파일링: 단계/기능별 시간·메모리 계측과 Top‑N 병목 리포트.
- 최적화: Gram 캐시, 블록화, 저랭크 근사, 병렬화.
- GPU/가속: CuPy/PyTorch로 대규모 행렬 및 커널 연산을 부분 가속.

## 12. 설정 스키마(신설) — experiment_config.yaml
```yaml
selection_pipeline:
  enabled: true
  artifact_dir: "dataset/selection_artifacts"
  min_bars_per_split: 5000
  dry_run: false
  stage1_screening:
    H: [1, 3, 12, 48]
    hsic: {kernel: rbf, bootstrap: block, alpha: 0.05}
    dcor: {test: t, permute: block, threshold: 0.1}
    combine: {method: empirical_brown}   # or "simes"
    fdr: {method: by}                    # "bh" | "by"
  stage2_selection:
    knockoffs: {type: copula, q: 0.1, W: lcd, group: true, tski: {enabled: false}}
    stability_selection: {enabled: true, PFER: 1.0, subsamples: 200, cutoff: 0.75}
    cross_fit: {enabled: true, inner_splits: 2}
  stage3_diversity:
    method: dpp                          # "dpp" | "facility_location"
    k: 128
    dpp_kernel: {lambda: 1.0}
    family_cap: {enabled: true, max_per_family: 3}
  stage4_calibration:
    vif_threshold: 10.0
    robust_quantiles: [0.005, 0.995]
    mad_margin: {enabled: true, factor: 1.4826}
```

## 13. YAML 스냅샷 규약(강화) — feature_classification_active.yaml
```yaml
selection_snapshot:
  created_at: "2025-09-23T12:34:56Z"
  git_commit: "<hash>"
  data_meta_hash: "sha256:<...>"
  env: {seed: 42, device: "cuda:0", threads: 8, packages: {numpy: "2.2.6", pandas: "2.3.1", torch: "2.8.0"}}
  method:
    cv: {mode: CPCV, folds: 10, test_folds: 2, purge_bars: null, embargo_bars: null}
    screening: {hsic: {kernel: rbf, bootstrap: block}, dcor: {test: t, permute: block}, effect_size: {cliffs_delta: true}}
    combine: {within_feature: empirical_brown, fdr: by}
    selector: {knockoffs: {type: tski, W: lcd, q: 0.1, group: true}, stability: {PFER: 1.0, subsamples: 200, cutoff: 0.75}}
    diversity: {obj: facility_location, k: 128, family_cap: {max_per_family: 3}}
    calibration: {quantiles: [0.5, 99.5], mad_margin: 1.4826}
  active_feature_whitelist:
    - RSI_14_5m
    - ATR_7_5m
    - Dojis_1h
  feature_num_overrides:
    RSI_14_5m: [2.0, 98.0]
    ATR_7_5m: [0.0, 180.0]
  feature_bool_whitelist:
    - Dojis_1h
  feature_comparison_whitelist:
    - Close_5m
    - EMA_20_5m
  notes: "Cross‑fitting; block bootstrap; BY at 5%; TSKI if available"
  fallback:
    chain: [tski_knockoff, robust_knockoff, copula_knockoff, stability_selection]
    reason: "tski_not_available"
```

## 14. CLI/동작 순서(갱신)
- 엔트리: `python selection_data.py --config Config/experiment_config.yaml \
  --screening {hsic_rp,dcor_t,spearman} --combine {empirical_brown,simes} \
  --fdr {bh,by,knockoff,knockoff_plus} --knockoff {copula,robust,tski,group} \
  --diversity {dpp,facility_location} --k 128 --dry-run`
- 순서: 설정 로드 → 데이터 로드 → CPCV 분할 생성 → Cross‑fitting 하 스크리닝→선택→다양성→캘리브레이션 → `model_config_path` 갱신(백업) → YAML 스냅샷 append → 아티팩트 저장.

## 15. 산출물·리포트(확장)
- FDR 실측 곡선(BH/BY/Knockoff/+) 비교, 선택 안정성(시드/윈도우/재실행 Jaccard·표준편차), 다양성 단계 전후 네트워크/커버리지, 불리언 OR/RR/δ 및 CI.
- 실패 폴백 로그: 폴백 사유·증거(진단 수치) 명시.

## 16. 검증·위험·폴백
- 회귀 검증: 선택 반복 자카드≥0.6, 실측 FDR≤q+ε, 최종 K 다양성 지표 개선.
- 위험: 선택‑후‑검정, i.i.d. 순열 오용, 과소다양성/패밀리 편중.
- 완화: 크로스피팅 강제, 블록/랜덤 프로세스 부트스트랩, 서브모듈러 보장·패밀리 상한.

---
본 v2 설계는 시계열 안전성·비선형 의존성·다중가설·선택 안정성·다양성·재현성의 요건을 한층 강화했다. 구현은 §3의 단계별 로드맵을 따르며, 설정은 §12의 `selection_pipeline`로 중앙집중화한다. 스냅샷과 메타데이터 규약(§13)으로 실험 재현 가능성을 최대화한다.

## 17. 구현 청사진(How-to): 모듈 구성과 API 설계

### 17.1 파일/모듈 레이아웃(초안)
- `selection_data.py` (엔트리/CLI): 파이프라인 orchestration, 설정 로드, 산출물 저장.
- `selection/loader.py`: 데이터/설정 로딩·경로 해석(`final_features_[pkl|csv]`, `model_config_path`).
- `selection/splits.py`: CPCV 스케줄러 래퍼 및 inner split 유틸(기존 `training.cpcv` 재사용).
- `selection/screening.py`: HSIC/dCor/Spearman 계산, 블록/와일드 부트스트랩, p‑값 결합.
- `selection/selector.py`: Knockoffs( Copula/Robust/TSKI ), 안정 선택(PFER 기반) 구현.
- `selection/diversity.py`: DPP 및 facility‑location 그리디.
- `selection/calibration.py`: 범위 로버스트 분위 + MAD 마진, VIF/부분R²/조건수 평가.
- `selection/snapshot.py`: YAML 스냅샷 append, 원자적 쓰기/백업, 메타해시·환경 기록.
- `selection/report.py`: JSON/PNG 아티팩트 생성(FDR 곡선, 네트워크 등).
- `selection/utils.py`: 공통 수학/통계 유틸(부분상관, Gram 캐시, GPU 전환 등).

프로젝트 규모를 고려하면 모듈 분리는 단계적 도입 가능. MVP에서는 `selection_data.py` 단일 파일에 핵심 절차를 두고, Stage2+에서 모듈화 권장.

### 17.2 핵심 API 시그니처(제안)
```python
# selection/loader.py
def load_experiment_config(path: str) -> dict: ...
def resolve_paths(cfg: dict) -> dict:  # returns {final_pkl, final_csv, model_cfg, fc_active_yaml, artifacts_dir}
def load_final_features(paths: dict) -> pd.DataFrame: ...  # prefers pickle, index=Open time
def load_model_config(paths: dict) -> dict: ...

# selection/splits.py
from training.cpcv import CPCVScheduler, CPCVSplit
def make_cpcv_scheduler(total_len: int, cfg: dict, train_cfg: dict) -> CPCVScheduler: ...
def inner_time_split(train_idx: np.ndarray, ratio: float=0.5) -> tuple[np.ndarray, np.ndarray]: ...

# selection/screening.py
class ScreeningResult(TypedDict):
    pvals: dict[str, float]          # within‑feature combined p
    effects: dict[str, float]        # e.g., |Cliff's δ| or |ρ|
    details: dict[str, Any]          # per (h,fold) stats for report

def screening_out_of_fold(df: pd.DataFrame, target_h: list[int], splits: list[CPCVSplit], methods: dict, min_bars: int) -> ScreeningResult: ...

# selection/selector.py
class SelectionResult(TypedDict):
    selected: set[str]
    stability: dict[str, float]      # selection prob or knockoff freq
    diagnostics: dict[str, Any]

def run_knockoffs(df: pd.DataFrame, candidates: list[str], target: np.ndarray, cfg: dict, train_idx: np.ndarray) -> SelectionResult: ...
def run_stability_selection(df: pd.DataFrame, candidates: list[str], target: np.ndarray, cfg: dict, train_idx: np.ndarray) -> SelectionResult: ...

# selection/diversity.py
def pick_diverse_subset(candidates: list[str], sim: np.ndarray, method: str, k: int, family_cap: dict|None) -> list[str]: ...

# selection/calibration.py
class CalibrationResult(TypedDict):
    feature_num: dict[str, tuple[float, float]]
    feature_bool: list[str]
    feature_comparison_map: dict[str, list[str]]
    diagnostics: dict[str, Any]

def calibrate_ranges(df: pd.DataFrame, features: list[str], percentiles: tuple[float,float], mad_cfg: dict) -> CalibrationResult: ...
def assess_collinearity(df: pd.DataFrame, features: list[str], cfg: dict) -> dict[str, Any]: ...  # partial R², κ, VIF

# selection/snapshot.py
def append_selection_snapshot(fc_active_yaml: str, metadata: dict, final_set: CalibrationResult) -> None: ...
def atomic_write_yaml(path: str, data: dict, backup: bool=True) -> None: ...

# selection/report.py
def save_json(path: str, payload: dict) -> None;  def save_png(path: str, fig) -> None
```

### 17.3 데이터 계약(Data Contracts)
- 입력 DF: `Open time` 인덱스, 열은 모든 후보 피처(+OHLCV). 불리언은 {0,1} 또는 bool.
- 타깃: 각 h에 대해 `y^{(h)}` 시계열(로그수익) on‑the‑fly 생성.
- 출력 모델 설정: 기존 스키마 유지(`feature_num`, `feature_comparison_map`, `feature_bool`) — 선택 결과로 축소·갱신.
- 스냅샷: §13 규약의 필드 보장 + 최종 화이트리스트(타임프레임 접미사 포함).

### 17.4 스테이지별 구현 절차(체크리스트)
- Stage1 Screening
  1) CPCV 스케줄러 생성 → 각 유효 split마다 train_idx 확보.
  2) 각 h∈H, 각 split에서: HSIC/dCor/Spearman·Cliff’s δ 계산(블록/와일드 부트스트랩 사용), p_hf 기록.
  3) 동일 피처 내 p‑값 결합(Empirical Brown 기본, 대안 Simes), 효과크기 병기.
  4) FDR(BY)로 통제하여 1차 후보군 C₁ 도출.
- Stage2 Selector
  5) Cross‑fit: train_idx를 A,B로 절단 후 A에서 학습→B 평가, B에서 학습→A 평가.
  6) Knockoffs(q) 또는 안정 선택(PFER) 실행, 선택 확률/안정성 스코어 산출.
  7) 안정성 임계 초과 항목으로 후보군 C₂ 갱신. 실패 시 폴백 체인 적용.
- Stage3 Diversity
  8) 부분상관/HSIC 유사도 행렬 계산 → DPP 또는 facility‑location 그리디로 K개 선정.
  9) 패밀리 상한 규칙 적용(접두 그룹별 m개 제한).
- Stage4 Calibration
  10) CPCV train fold 분포에서 로버스트 분위(q_low–q_high) 추정 + MAD 마진 가산.
  11) 공선성 평가(부분R²/조건수/VIF) 후 영향 큰 항목 그리디 제거(로그 남김).
  12) 최종 `feature_num/feature_bool/feature_comparison_map` 구성.
- Persist
  13) `model_config_path` 백업 후 갱신(원자적 쓰기). `feature_classification_active.yaml`에 스냅샷 append.
  14) 아티팩트(JSON/PNG) 저장, 성공 로그·실패 폴백 로그 기록.

### 17.5 비교형 피처 맵 구성 규칙
- 입력 `model_config.feature_comparison_map`에서 LHS/RHS 모두 “선택된 피처”에 교차 필터링.
- LHS마다 RHS는 자기 자신 제외, 동일 패밀리 과밀 시 상한 적용.
- 빈 RHS가 되면 해당 LHS는 제외.

### 17.6 원자적 쓰기와 백업 규칙
- `tmp = path + .tmp-<pid>-<ts>`에 먼저 저장 → `fsync` → `rename`으로 교체.
- 동일 디렉토리에 `.bak-YYYYmmdd-HHMMSS` 백업 파일 생성(최대 N 보관 정책 옵션화 가능).

### 17.7 오류 처리/폴백 시나리오
- 샘플 부족(`min_bars_per_split` 미달): 해당 split/지평 제외, 경고 집계.
- Knockoffs 분포 추정 실패: Robust→Copula 순으로 자동 폴백.
- DPP 커널 비PSD: λ 조정 후 실패 시 facility‑location로 폴백.
- 스냅샷 쓰기 실패: 롤백 후 경고, 보고서에 실패 원인 기록.

### 17.8 성능 최적화 체크리스트(구현 관점)
- Gram 행렬 캐시: 동일 피처 서브셋·커널 폭 σ 재사용.
- 블록 처리: HSIC의 O(n²) 메모리 부담 완화.
- 벡터화/배치: 다수 h·split 동시 처리, NumPy/CuPy로 배치 연산.
- 병렬화: `parallel_chunks`로 split 배치 병렬 평가.

## 18. 통합 포인트와 실행 예
- 통합: `preprocess_data.py` 완료 후 `python selection_data.py --config Config/experiment_config.yaml` 실행. 완료 시 `dataset/model_test_config.yaml`가 축소/갱신되어 `main.py`가 그대로 소비.
- 실행 예:
  - BY+Knockoffs+facility‑location: `--fdr by --knockoff copula --diversity facility_location --k 128`
  - TSKI+Cross‑fit 3분할: `--knockoff tski --inner-splits 3`
  - Dry‑run: `--dry-run`(아티팩트만 생성, 파일 미변경)

---

## 부록 A. 기호·데이터 정의(Notation)
- 시점 t, 타임프레임 f ∈ F(예: {5m, 30m, 1h, 4h, 1d}).
- 피처 i ∈ {1..p}, 관측치 수 n. 피처 행렬 X ∈ R^{n×p}, 타깃은 h∈H 지평의 포워드 로그수익 y^{(h)}_t = log(C_{t+h}/C_t).
- CPCV 분할: 폴드 k=1..K, 테스트 폴드 수 K_test; purge, embargo 바 수는 각각 P, E.
- 학습/검정 인덱스: (I_k^train, I_k^test), cross‑fit 내부 분할 a∈{A,B}.

## 부록 B. CPCV + Cross‑fitting 절차(정의/의사코드)
의사코드(요약):
```
Input: X, {y^{(h)}}_{h∈H}, CPCV(K, K_test, P, E), inner_splits=2
For each outer fold k:
  Split train indices I_k^train into A,B (time‑ordered, disjoint)
  # Screening
  S_A = Screen(X[A], y[A]; methods=HSIC/dCor/Spearman)
  S_B = Screen(X[B], y[B]; methods=HSIC/dCor/Spearman)
  Combine S = out‑of‑fold aggregation of S_A,S_B  # no double dipping
  # Selector (Knockoffs or Stability Selection), trained only on A, scored on B (and swap):
  Sel_A = Select(X[A], y[A]) ; Eval on B (importance/stability)
  Sel_B = Select(X[B], y[B]) ; Eval on A
  # Diversity & Calibration use only train folds, aggregated out‑of‑fold
  D_k  = Diversity(Sel_A ∪ Sel_B, sim=partial_corr/HSIC)
  Cal_k = CalibrateRanges(D_k, quantiles=[q_low,q_high], MAD_margin)
Aggregate across outer folds: intersect/union with stability score; apply FDR rule (BY/BH) to within‑feature combined p‑values
Return final feature set, numeric ranges, boolean list, comparison map
```
주의: 모든 통계량·학습은 반드시 I_k^train에서 수행하고, 리포트 집계는 out‑of‑fold(반대 절반)로 수치화한다.

## 부록 C. 스크리닝 통계의 수학적 정의와 기본값
- HSIC(비편향 추정치):
  - 커널 k(x,x′)=exp(−||x−x′||²/(2σ_x²)), l(y,y′)=exp(−||y−y′||²/(2σ_y²)).
  - σ는 median heuristic(각 데이터셋의 pairwise 거리 중앙값) 기본, 또는 Silverman 규칙.
  - 통계량: HSIC_u = (1/(n(n−3))) Tr(ḰĹ) (수식은 구현 시 참고 문헌에 맞춰 선택).
  - p‑값: 블록 부트스트랩(BBS) 또는 와일드 부트스트랩(WBS). 블록 길이 L은 데이터구동형(§성능 참고) 선택.
- dCor t‑검정:
  - distance covariance 기반의 Studentized 통계 사용(고차 독립성). 순열은 블록/서큘러.
- Spearman ρ, Kendall τ:
  - 동점(tie) 보정 포함. 유의성은 블록 부트스트랩으로 재평가.
- 효과크기 병기(권장 cut‑off): Cliff’s δ ∈ [−1,1], |δ|<0.147(작음), <0.33(중간), <0.474(큼). 미미한 효과는 제외.
- 하이퍼 기본값(제안): HSIC alpha=0.05, dCor threshold=0.1, combine=Empirical Brown(의존 강) 또는 Simes(의존 약), FDR=BY.

## 부록 D. p‑값 결합 및 다중가설 제어 규칙
- 동일 피처 i의 모든 (h×fold) p‑값 집합 P_i에 대해 결합 p_i^∗ 계산:
  - Empirical Brown: Z = w^T Σ^{-1} w, 자유도 적응형으로 조정(의존 반영). 또는
  - Simes: p_i^∗ = min_j (m·p_{(j)}/j).
- 피처 간 FDR 통제: BH(독립/양의의존) 또는 BY(일반 의존). BY는 보수적이나 시계열·상관 구조에서 안전.
- 문서 규칙: 기본 BY, 보고서에 BH와 비교 그래프 병기.

## 부록 E. Knockoffs(시계열 적응형) 상세
1) 분포 추정: Nonparanormal(SKEPTIC) 또는 Copula‑Gaussian로 Σ 추정. 필요 시 robust covariance.
2) Knockoff 생성: equi‑correlated 혹은 SDP 근사로 X̃ 생성(시간의존 시 TSKI 사용 — AR/VAR 구조 반영).
3) 선택 통계 W: LCD(regularization path 진입 임계) 차이, 또는 signed max correlation 등.
4) 임계 선택: Knockoff or Knockoff+ 규칙으로 T 찾기 → {i: W_i ≥ T} 선택.
5) 그룹/멀티레이어: 패밀리별 그룹화하여 그룹 단위 선택(해석 용이). 계층적 그룹도 허용.
6) 폴백 체인: TSKI→Robust→Copula→StabilitySelection.
7) 리포트: 실측 FDR(시뮬레이션 기반 추정), 선택 수 표준편차, 윈도우/시드 Jaccard.

## 부록 F. 안정 선택(Stability Selection) 튜닝
- 목표오류(PFER) 상한을 먼저 정하고, (평균 선택수 q, cut‑off π*)를 역산.
- 고전적 근사 관계(참고식): PFER ≲ (q^2)/(2p(π*−0.5)) (문맥에 따라 상수·정의가 다르므로 구현 시 문헌 값을 사용).
- 절차: subsamples B회(기본 200), 각 회 L1경로에서 선택빈도 π̂_i 계산 → {i: π̂_i ≥ π*}.
- 클러스터 안정 선택: 고상관 군집에서 대표 손실 최소화를 위해 가중 합집합/평활화 적용.

## 부록 G. 다양성 보장 수식
- DPP: 유사도 s_ij=|partial_corr(i,j)| → 커널 K_ij = exp(−λ(1−s_ij)), λ∈[0.5,2] 권장. PSD 확인 후 k‑DPP 혹은 MAP 근사.
- 시설위치 서브모듈러: F(S)=∑_{u∈U} max_{v∈S} sim(u,v); 그리디 선택은 1−1/e 보장. `family_cap`은 패널티/제약으로 구현.

## 부록 H. 불리언 피처 통계(희소 이벤트)
- 빈도·버스티니스: 최소 발생 n_min(데이터구동형), 인터어라이벌 CV·버스티니스 지표로 잡음 필터.
- 검정: 표본 충분 시 G‑test/LRT, 희소 시 Fisher exact.
- 효과크기: 오즈비 OR=(a/c)/(b/d), 상대위험도 RR=(a/(a+b))/(c/(c+d)), 로그스케일 CI.
- 연속 반응 차이: Cliff’s δ 및 블록 부트스트랩 CI.

## 부록 I. 공선성 및 범위 캘리브레이션 알고리즘
- 공선성 절차: (1) 부분상관/부분R² 순위화, (2) 조건수 κ(X_S), (3) 보조로 VIF, (4) 영향 큰 피처부터 그리디 제거(각 제거의 OOS 영향 기록).
- 범위 캘리브레이션: fold‑내 분포에서 q_low–q_high 분위 추정(기본 0.5–99.5%) 후 MAD 기반 마진 m=c·MAD(잔차) 가산(c≈1.4826), 최종 범위 [q_low−m, q_high+m] 클램프.

## 부록 J. 복잡도·성능 가이드
- HSIC O(n²) → 블록/저랭크 근사, Gram 캐시, GPU.
- Knockoffs: Σ추정 O(p²n)+생성 O(p³) 근사; 그룹/TSKI는 추가 비용.
- 서브모듈러 그리디 O(pk) 유사; DPP MAP는 커널 분해 병목.
- 프로파일링 리포트: 단계별 시간/메모리, 최적화 전/후 비교 저장.

## 부록 K. 구현 체크리스트(QA)
- 분할 무결성: Purge/Embargo가 학습/검정 교차에 미치지 않는지 단위 테스트.
- 크로스피팅: 스크리닝/선택/캘리브레이션 데이터 분리 검증.
- 통계 일관성: i.i.d. 순열 금지, 블록/와일드만 허용.
- 재현성: git 해시·seed·패키지 버전 스냅샷 포함 검증.
- 스냅샷 원자적 쓰기 및 .bak 생성 여부.

## 부록 L. selection_data.py 파이프라인 의사코드(상세)
```
parse_args(); cfg = load_experiment_config()
paths = resolve_paths(cfg); df, model_cfg = load_final_dataframe_and_modelcfg(paths)
splits = make_CPCV_splits(cfg.cpcv, cfg.training_strategy.max_hold_period_bars)
for fold in splits:
  A,B = inner_time_split(fold.train, ratio=0.5)
  S_A = screening_stats(df[A], methods=cfg.selection_pipeline.stage1_screening)
  S_B = screening_stats(df[B], methods=cfg.selection_pipeline.stage1_screening)
  S   = combine_pvalues_out_of_fold(S_A, S_B, rule=cfg.selection_pipeline.stage1_screening.combine)
  Sel_A = selector_fit(df[A], cfg.selection_pipeline.stage2_selection)
  Sel_B = selector_fit(df[B], cfg.selection_pipeline.stage2_selection)
  Eval_B = selector_eval(Sel_A, df[B]); Eval_A = selector_eval(Sel_B, df[A])
  Cand = aggregate_candidates(Eval_A, Eval_B, stability=cfg.selection_pipeline.stage2_selection)
  Div  = diversity_pick(Cand, method=cfg.selection_pipeline.stage3_diversity)
  Cal  = calibrate_ranges(Div, df[fold.train], cfg.selection_pipeline.stage4_calibration)
Final = aggregate_over_folds(Div, Cal, fdr=cfg.selection_pipeline.stage1_screening.fdr)
write_model_config(paths.model_config_path, Final, backup=True)
append_selection_snapshot(paths.feature_classification_active_yaml, metadata(cfg, paths, Final))
write_artifacts(report_json/png, logs)
```

## 부록 M. 산출물 스펙(자세)
- JSON: screening(결합 규칙·BY/BH 곡선·효과크기), selection(FDR 추정·안정성), diversity(커버리지/네트워크), calibration(분위·MAD·최종 범위), stability(자카드/표준편차).
- PNG: 네트워크(전/후), 커버리지 곡선, FDR 곡선.

## 부록 N. 설정 필드 상세(§12 확장)
- `min_bars_per_split`: 검정·선택 최소 샘플; 미달 시 해당 단계 skip 및 폴백 경고 기록.
- `cross_fit.inner_splits`: 2(권장) 또는 3; 시간 순서를 유지하는 절단(split)만 허용.
- `dpp_kernel.lambda`: 0.5–2.0 탐색; 데이터 기반 초기화(평균 유사도 목표) 옵션.
- `family_cap.max_per_family`: 패턴/계열별 상한; ‘RSI_*’, ‘Doji_*’ 등 접두 기반 그룹핑 사용.

## 부록 O. 스냅샷 메타해시 정의(예시)
`sha256( path(data) || nrows || sorted(columns) || seed || cpcv_params || timeframe_set || hash(model_config_baseline) )`
