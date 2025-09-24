# selection_data.py: Advanced Feature Selection Design v2 (Feedback Incorporated)

This document is the v2 design specification, upgrading the previous design (version v1) for "real-world deployment". It incorporates suggestions from two feedback reports, strengthening computational realism (Phased rollout), time-series consistency testing and cross-fitting, multiple hypothesis combination & FDR control rules, time-series adaptive variants of Knockoffs, diversity-ensuring objective function specifications, sparse boolean processing, reproducibility/audit metadata, and CLI/configuration schema.

## 0. Summary of Changes (Core)
- **Phased Rollout Roadmap:** Gradual execution and risk management from MVP → Stage 2 → Stage 3.
- **Enhanced Cross-fitting:** Separates screening, selection, and calibration into out-of-fold processes to mitigate post-selection inference bias.
- **Time-Series Consistency Testing:** HSIC supports block/wild bootstrap; dCor supports higher-order independence t-tests and block permutation.
- **Multiple Hypotheses & Dependent p-values:** Formalized the use of Empirical Brown/Simes for combination + a two-stage BH/BY rule for control.
- **Advanced Knockoffs:** Options for Copula/Nonparanormal + Robust/Approx + TSKI (time-dependent) + group/multi-layer options and a fallback chain.
- **Diversity:** Fixed the DPP kernel definition and added an alternative (facility-location submodular, with a 1−1/e guarantee).
- **Boolean Events:** Fisher's/G-test, OR/RR·Cliff’s δ, and block bootstrap CIs.
- **Redefined VIF:** Instead of an absolute cut-off, it will be considered alongside partial correlation, partial R², condition number, and OOS impact.
- **Reproducibility:** Expanded use of meta-hashes, git commits, seed/environment/package versions, dry-runs, and atomic writes.
- **Configuration:** A new `selection_pipeline` section is introduced, sharing CPCV as the Single Source of Truth (SSOT).

## 1. Goals and Scope (Unchanged)
- **Goal:** To rigorously select the "final feature set" just before training from pre-processing outputs (the final feature frame, `model_config_path`), and ① reflect the results in `model_config_path`, and ② record a snapshot in `Config/feature_classification_active.yaml`.
- **Out of Scope:** No changes to the training loop or model architecture. This document only defines the selection pipeline and schema.

## 2. High-Level Architecture and I/O (Summary)
- **Execution Point:** After `preprocess_data.py`, before `main.py`.
- **Inputs:** Final feature frame (pkl/csv), `model_config_path`, `Config/experiment_config.yaml`, `Config/feature_classification_active.yaml`.
- **Outputs:** A reduced `model_config_path`, an enhanced `feature_classification_active.yaml (selection_snapshot)`, and selection reports/artifacts.

## 3. Phased Implementation Roadmap
- **Stage 1 (MVP, Immediate Go-live):**
  - Integrity gates (remove missing, constant, sparse, duplicate, fully correlated features).
  - Primary screening based on rank correlation (Spearman/Kendall).
  - Representative feature selection using partial correlation / graphical Lasso summary.
  - Final VIF & partial R² check + robust quantile (0.5–99.5%) range calibration.
- **Stage 2 (Confidence Enhancement):**
  - Stability Selection (Subsampling L1; tuned based on PFER upper bound) or Model-X Knockoffs (q-control).
  - Mitigate post-selection inference bias with cross-fitting.
  - Prioritize introduction of diversity (submodular facility-location).
- **Stage 3 (High-Performance & Non-linear Expansion):**
  - HSIC (random process/block bootstrap) & dCor t-test.
  - TSKI/Robust Knockoffs, advanced diversity with DPP.
  - Large-scale optimization (Gram caching, blocking, low-rank approximation, GPU acceleration).

## 4. CPCV + Cross-fitting Norms
- **Purge/Embargo:** `training_strategy.max_hold_period_bars` is interpreted in absolute bar units, with the timeframe conversion rule explicitly stated (the formula will be logged during actual implementation).
- All statistics are estimated "only on the train set". Aggregation is out-of-fold.
- Within each CPCV train fold, use a 2-way split (or K-fold) to separate data for screening/selection/calibration to prevent double-dipping.
- **Minimum Sample Constraint:** Tests/selections are only performed if `selection_pipeline.min_bars_per_split` is met.

## 5. Screening (Time-Series Consistency Testing)
- **Target:** Forward log-return r_{t→t+h}, where h∈H (e.g., {1,3,12,48}·5m).
- **Tests:**
  - **HSIC:** RBF kernel, p-value via block/wild bootstrap; Block-HSIC Lasso for large datasets.
  - **dCor:** Higher-order independence t-test + block/circular permutation options.
  - **Rank Coeff./Monotonicity:** Spearman ρ; report effect size (Cliff’s δ) alongside.
- **Combination & Correction:** Combine (h×fold) p-values for the same feature using Empirical Brown (strong dependence) or Simes (weak dependence) → Control FDR across features with BH/BY.

## 6. Selectors
- **Model-X Knockoffs:**
  - **Generative Model:** Options for Copula/Nonparanormal, Robust/Approx Knockoffs, TSKI (time-series).
  - **Statistic W:** Selectable from LCD/SMX, etc., supporting Knockoff/Knockoff+ thresholds.
  - **Group/Multi-layer:** Allows group selection for families of indicators.
  - **Fallback Chain:** TSKI → Robust → Copula → Stability Selection.
- **Stability Selection (Subsampling L1):**
  - Define PFER upper bound first → inversely tune (average selected count q, cut-off).
  - Option to complement with cluster stability selection for highly correlated clusters.

## 7. Diversity (De-redundancy)
- **DPP:** PSD similarity kernel K = exp(−λ·(1−|partial-corr|)), with the default λ and its search range documented.
- **Alternative:** Facility-location submodular + greedy (1−1/e guarantee), supports family caps (m items).

## 8. Boolean Feature Specialization
- **Sparse Events:** Automatically estimate thresholds for minimum occurrences `n_min` and inter-arrival CV/burstiness.
- **Tests:** G-test/LRT if sample is sufficient, Fisher's exact test if sparse. Effect sizes OR/RR (95% CI), Cliff’s δ for differences in continuous responses.
- All significances are corrected with block bootstrap. Families are simplified through group selection.

## 9. Collinearity & Range Calibration
- **Collinearity Assessment:** VIF is for reference only; decisions are made together with partial correlation, partial R², condition number, and OOS performance impact.
- **Range:** Automatically add a MAD-based margin to robust quantiles (default 0.5–99.5%). The margin is larger for features with high variance across folds.

## 10. Reproducibility & Audit Metadata
- **Snapshot Meta-hash:** An SHA-256 hash combining data path, row count, column list, CPCV seed/parameters, timeframe, and model settings hash.
- Record git commit hash, random seed, CPU/GPU/threads, and major package versions (frozen).
- Atomic writes (tmp→move) and `.bak-TIMESTAMP` backups; pre-flight checks with `--dry-run` mode.

## 11. Performance & Acceleration Plan
- **Profiling:** Instrument time and memory per stage/function and report Top-N bottlenecks.
- **Optimization:** Gram caching, blocking, low-rank approximation, parallelization.
- **GPU/Acceleration:** Partially accelerate large-scale matrix and kernel operations with CuPy/PyTorch.

## 12. Configuration Schema (New) — experiment_config.yaml
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

## 13. YAML Snapshot Convention (Enhanced) — feature_classification_active.yaml
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
  notes: "Cross-fitting; block bootstrap; BY at 5%; TSKI if available"
  fallback:
    chain: [tski_knockoff, robust_knockoff, copula_knockoff, stability_selection]
    reason: "tski_not_available"
```

## 14. CLI / Operational Sequence (Updated)
- **Entrypoint:** `python selection_data.py --config Config/experiment_config.yaml \
  --screening {hsic_rp,dcor_t,spearman} --combine {empirical_brown,simes} \
  --fdr {bh,by,knockoff,knockoff_plus} --knockoff {copula,robust,tski,group} \
  --diversity {dpp,facility_location} --k 128 --dry-run`
- **Sequence:** Load config → Load data → Create CPCV splits → Screening under Cross-fitting → Selection → Diversity → Calibration → Update `model_config_path` (with backup) → Append YAML snapshot → Save artifacts.

## 15. Deliverables & Reports (Expanded)
- Empirical FDR curves (BH/BY/Knockoff/+) comparison, selection stability (Jaccard/std dev across seeds/windows/reruns), diversity pre/post network/coverage, boolean OR/RR/δ and CIs.
- Fallback logs: Explicitly state the reason and evidence (diagnostic values) for fallbacks.

## 16. Validation, Risks, & Fallbacks
- **Regression Tests:** Repetitive selection Jaccard ≥ 0.6, empirical FDR ≤ q+ε, improvement in final K diversity metrics.
- **Risks:** Post-selection inference, misuse of i.i.d. permutations, under-diversification/family bias.
- **Mitigation:** Enforce cross-fitting, block/random process bootstrap, submodular guarantees & family caps.

---
This v2 design further strengthens requirements for time-series safety, non-linear dependencies, multiple hypotheses, selection stability, diversity, and reproducibility. Implementation will follow the phased roadmap in §3, and configuration is centralized in `selection_pipeline` as per §12. Experiment reproducibility is maximized through the snapshot and metadata conventions (§13).

## 17. Implementation Blueprint (How-to): Module Structure and API Design

### 17.1 File/Module Layout (Draft)
- `selection_data.py` (Entrypoint/CLI): Pipeline orchestration, config loading, artifact saving.
- `selection/loader.py`: Data/config loading and path resolution (`final_features_[pkl|csv]`, `model_config_path`).
- `selection/splits.py`: CPCV scheduler wrapper and inner split utility (reuses existing `training.cpcv`).
- `selection/screening.py`: HSIC/dCor/Spearman calculation, block/wild bootstrap, p-value combination.
- `selection/selector.py`: Knockoffs (Copula/Robust/TSKI), Stability Selection (PFER-based) implementation.
- `selection/diversity.py`: DPP and facility-location greedy algorithm.
- `selection/calibration.py`: Robust quantile range + MAD margin, VIF/partial R²/condition number evaluation.
- `selection/snapshot.py`: YAML snapshot appending, atomic writes/backups, meta-hash & environment recording.
- `selection/report.py`: JSON/PNG artifact generation (FDR curves, networks, etc.).
- `selection/utils.py`: Common math/stats utilities (partial correlation, Gram caching, GPU switching, etc.).

Considering the project scale, module separation can be phased. For MVP, the core logic can reside in a single `selection_data.py` file, with modularization recommended from Stage 2+.

### 17.2 Core API Signatures (Proposed)
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
    pvals: dict[str, float]          # within-feature combined p
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

### 17.3 Data Contracts
- **Input DF:** `Open time` index, columns are all candidate features (+OHLCV). Booleans are {0,1} or bool.
- **Target:** `y^{(h)}` time-series (log return) generated on-the-fly for each h.
- **Output Model Config:** Maintain existing schema (`feature_num`, `feature_comparison_map`, `feature_bool`) — reduced and updated with selection results.
- **Snapshot:** Guarantee fields from §13 convention + final whitelist (including timeframe suffix).

### 17.4 Stage-by-Stage Implementation Procedure (Checklist)
- **Stage 1 Screening**
  1) Create CPCV scheduler → obtain train_idx for each valid split.
  2) For each h∈H and each split: Calculate HSIC/dCor/Spearman·Cliff’s δ (using block/wild bootstrap), record p_hf.
  3) Combine p-values within the same feature (Empirical Brown default, Simes alternative), and report effect sizes.
  4) Derive primary candidate set C₁ by controlling FDR (BY).
- **Stage 2 Selector**
  5) Cross-fit: Split train_idx into A,B. Train on A → evaluate on B; train on B → evaluate on A.
  6) Run Knockoffs(q) or Stability Selection(PFER), compute selection probabilities/stability scores.
  7) Update candidate set C₂ with items exceeding the stability threshold. Apply fallback chain on failure.
- **Stage 3 Diversity**
  8) Calculate partial correlation/HSIC similarity matrix → select K items with DPP or facility-location greedy.
  9) Apply family cap rule (limit to m items per prefix group).
- **Stage 4 Calibration**
  10) Estimate robust quantiles (q_low–q_high) from CPCV train fold distributions + add MAD margin.
  11) Assess collinearity (partial R²/condition number/VIF) and greedily remove high-impact items (leaving logs).
  12) Construct final `feature_num/feature_bool/feature_comparison_map`.
- **Persist**
  13) Backup and update `model_config_path` (atomic write). Append a snapshot to `feature_classification_active.yaml`.
  14) Save artifacts (JSON/PNG), record success logs and failure fallback logs.

### 17.5 Comparative Feature Map Construction Rules
- Filter the input `model_config.feature_comparison_map` by cross-referencing both LHS/RHS against the "selected features".
- For each LHS, the RHS excludes itself. If a family is overrepresented, apply a cap.
- If an LHS has an empty RHS, that LHS is excluded.

### 17.6 Atomic Write and Backup Rules
- First save to `tmp = path + .tmp-<pid>-<ts>` → `fsync` → replace with `rename`.
- Create a backup file `.bak-YYYYmmdd-HHMMSS` in the same directory (option to configure max N backups).

### 17.7 Error Handling / Fallback Scenarios
- **Insufficient Samples** (below `min_bars_per_split`): Exclude that split/horizon, aggregate warnings.
- **Knockoffs Distribution Estimation Failure:** Automatically fall back from Robust→Copula.
- **DPP Kernel Not PSD:** Adjust λ, then fall back to facility-location on failure.
- **Snapshot Write Failure:** Roll back and warn, record failure reason in the report.

### 17.8 Performance Optimization Checklist (Implementation View)
- **Gram Matrix Caching:** Reuse for the same feature subset and kernel width σ.
- **Block Processing:** Mitigate the O(n²) memory burden of HSIC.
- **Vectorization/Batching:** Process multiple h·splits simultaneously, use NumPy/CuPy for batch operations.
- **Parallelization:** Evaluate split batches in parallel with `parallel_chunks`.

## 18. Integration Points and Execution Examples
- **Integration:** After `preprocess_data.py` completes, run `python selection_data.py --config Config/experiment_config.yaml`. Upon completion, `dataset/model_test_config.yaml` is reduced/updated for direct consumption by `main.py`.
- **Execution Examples:**
  - BY+Knockoffs+facility-location: `--fdr by --knockoff copula --diversity facility_location --k 128`
  - TSKI+Cross-fit 3-split: `--knockoff tski --inner-splits 3`
  - Dry-run: `--dry-run` (only generate artifacts, no files are modified)

---

## Appendix A. Notation & Data Definitions
- Time point t, timeframe f ∈ F (e.g., {5m, 30m, 1h, 4h, 1d}).
- Feature i ∈ {1..p}, number of observations n. Feature matrix X ∈ R^{n×p}, target is forward log-return over horizon h∈H, y^{(h)}_t = log(C_{t+h}/C_t).
- CPCV split: fold k=1..K, number of test folds K_test; purge and embargo bars are P and E, respectively.
- Train/test indices: (I_k^train, I_k^test), inner cross-fit splits a∈{A,B}.

## Appendix B. CPCV + Cross-fitting Procedure (Definition/Pseudocode)
Pseudocode (Summary):
```
Input: X, {y^{(h)}}_{h∈H}, CPCV(K, K_test, P, E), inner_splits=2
For each outer fold k:
  Split train indices I_k^train into A,B (time-ordered, disjoint)
  # Screening
  S_A = Screen(X[A], y[A]; methods=HSIC/dCor/Spearman)
  S_B = Screen(X[B], y[B]; methods=HSIC/dCor/Spearman)
  Combine S = out-of-fold aggregation of S_A,S_B  # no double dipping
  # Selector (Knockoffs or Stability Selection), trained only on A, scored on B (and swap):
  Sel_A = Select(X[A], y[A]) ; Eval on B (importance/stability)
  Sel_B = Select(X[B], y[B]) ; Eval on A
  # Diversity & Calibration use only train folds, aggregated out-of-fold
  D_k  = Diversity(Sel_A ∪ Sel_B, sim=partial_corr/HSIC)
  Cal_k = CalibrateRanges(D_k, quantiles=[q_low,q_high], MAD_margin)
Aggregate across outer folds: intersect/union with stability score; apply FDR rule (BY/BH) to within-feature combined p-values
Return final feature set, numeric ranges, boolean list, comparison map
```
Note: All statistics and training must be performed on I_k^train, and report aggregation is quantified out-of-fold (on the other half).

## Appendix C. Mathematical Definitions and Defaults for Screening Statistics
- **HSIC (unbiased estimator):**
  - Kernels: k(x,x′)=exp(−||x−x′||²/(2σ_x²)), l(y,y′)=exp(−||y−y′||²/(2σ_y²)).
  - σ defaults to the median heuristic (median of pairwise distances for each dataset), or Silverman's rule.
  - Statistic: HSIC_u = (1/(n(n−3))) Tr(ḰĹ) (formula to be chosen based on reference literature during implementation).
  - p-value: Block Bootstrap (BBS) or Wild Bootstrap (WBS). Block length L is data-driven (see §Performance).
- **dCor t-test:**
  - Uses a Studentized statistic based on distance covariance (higher-order independence). Permutations are block/circular.
- **Spearman ρ, Kendall τ:**
  - Includes tie correction. Significance is re-evaluated with block bootstrap.
- **Report Effect Sizes (recommended cut-offs):** Cliff’s δ ∈ [−1,1], |δ|<0.147(small), <0.33(medium), <0.474(large). Exclude negligible effects.
- **Hyperparameter Defaults (proposed):** HSIC alpha=0.05, dCor threshold=0.1, combine=Empirical Brown (strong dependence) or Simes (weak dependence), FDR=BY.

## Appendix D. p-value Combination and Multiple Hypothesis Control Rules
- For the set of all (h×fold) p-values P_i for a feature i, calculate a combined p-value p_i^∗:
  - **Empirical Brown:** Z = w^T Σ^{-1} w, with adaptively adjusted degrees of freedom (reflects dependence). Or
  - **Simes:** p_i^∗ = min_j (m·p_{(j)}/j).
- **Inter-feature FDR Control:** BH (for independence/positive dependence) or BY (for general dependence). BY is more conservative but safer for time-series and correlated structures.
- **Document Rule:** Default to BY, include a comparison graph with BH in the report.

## Appendix E. Knockoffs (Time-Series Adaptive) Details
1) **Distribution Estimation:** Estimate Σ with Nonparanormal (SKEPTIC) or Copula-Gaussian. Use robust covariance if necessary.
2) **Knockoff Generation:** Generate X̃ using equi-correlated or SDP approximation (use TSKI for time-dependence — reflects AR/VAR structure).
3) **Selection Statistic W:** Difference in regularization path entry thresholds (LCD), or signed max correlation, etc.
4) **Threshold Selection:** Find T using the Knockoff or Knockoff+ rule → select {i: W_i ≥ T}.
5) **Group/Multi-layer:** Group by families for group-level selection (improves interpretability). Hierarchical groups are also allowed.
6) **Fallback Chain:** TSKI→Robust→Copula→StabilitySelection.
7) **Report:** Empirical FDR (estimated via simulation), standard deviation of selected count, window/seed Jaccard score.

## Appendix F. Stability Selection Tuning
- First, set a target error (PFER) upper bound, then inversely solve for (average selected count q, cut-off π*).
- Classic approximation relationship (reference only): PFER ≲ (q^2)/(2p(π*−0.5)) (constants and definitions vary, use literature values during implementation).
- **Procedure:** For B subsamples (default 200), calculate selection frequency π̂_i from each L1 path → select {i: π̂_i ≥ π*}.
- **Cluster Stability Selection:** Apply weighted union/smoothing to minimize representative loss in highly correlated clusters.

## Appendix G. Diversity Guarantee Formulas
- **DPP:** Similarity s_ij=|partial_corr(i,j)| → Kernel K_ij = exp(−λ(1−s_ij)), λ∈[0.5,2] recommended. After checking for PSD, use k-DPP or MAP approximation.
- **Facility-location Submodular:** F(S)=∑_{u∈U} max_{v∈S} sim(u,v); greedy selection has a 1−1/e guarantee. Implement `family_cap` as a penalty/constraint.

## Appendix H. Boolean Feature Statistics (Sparse Events)
- **Frequency & Burstiness:** Filter noise with a minimum occurrence n_min (data-driven) and inter-arrival CV/burstiness metrics.
- **Tests:** G-test/LRT for sufficient samples, Fisher's exact test for sparse data.
- **Effect Sizes:** Odds Ratio OR=(a/c)/(b/d), Relative Risk RR=(a/(a+b))/(c/(c+d)), log-scale CIs.
- **Difference in Continuous Response:** Cliff’s δ and block bootstrap CIs.

## Appendix I. Collinearity and Range Calibration Algorithms
- **Collinearity Procedure:** (1) Rank by partial correlation / partial R², (2) Calculate condition number κ(X_S), (3) Use VIF as a supplement, (4) Greedily remove high-impact features (logging the OOS impact of each removal).
- **Range Calibration:** Estimate q_low–q_high quantiles from in-fold distributions (default 0.5–99.5%), then add a MAD-based margin m=c·MAD(residuals) (c≈1.4826), clamping to the final range [q_low−m, q_high+m].

## Appendix J. Complexity & Performance Guide
- HSIC O(n²) → Mitigate with block/low-rank approximation, Gram caching, GPU.
- Knockoffs: Σ estimation is O(p²n) + generation is approx. O(p³); Group/TSKI add costs.
- Submodular greedy is O(pk)-like; DPP MAP is bottlenecked by kernel decomposition.
- Profiling Report: Store per-stage time/memory, and pre/post optimization comparisons.

## Appendix K. Implementation Checklist (QA)
- **Split Integrity:** Unit test that Purge/Embargo does not leak between train/test.
- **Cross-fitting:** Verify data separation for screening/selection/calibration.
- **Statistical Consistency:** Prohibit i.i.d. permutations, allow only block/wild.
- **Reproducibility:** Verify that git hash, seed, and package versions are included in snapshots.
- **Atomic Writes:** Check if snapshot atomic writes and .bak creation are working.

## Appendix L. selection_data.py Pipeline Pseudocode (Detailed)
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

## Appendix M. Artifact Specification (Detailed)
- **JSON:** screening (combination rule, BY/BH curves, effect sizes), selection (FDR estimation, stability), diversity (coverage/network), calibration (quantiles, MAD, final ranges), stability (Jaccard/std dev).
- **PNG:** Network graphs (before/after), coverage curves, FDR curves.

## Appendix N. Configuration Field Details (§12 Expanded)
- `min_bars_per_split`: Minimum samples for testing/selection; if not met, skip the step for that split and log a fallback warning.
- `cross_fit.inner_splits`: 2 (recommended) or 3; only time-ordered splits are allowed.
- `dpp_kernel.lambda`: Search in [0.5, 2.0]; option for data-driven initialization (target mean similarity).
- `family_cap.max_per_family`: Cap per pattern/series; uses prefix-based grouping like ‘RSI_*’, ‘Doji_*’.

## Appendix O. Snapshot Meta-hash Definition (Example)
`sha256( path(data) || nrows || sorted(columns) || seed || cpcv_params || timeframe_set || hash(model_config_baseline) )`