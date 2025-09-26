# selection_target_change.md — Targetless Selection Plan (Analysis & Proposal)

This document provides a detailed analysis of `selection_data_design.md(v2)` and the currently implemented code (modules under `selection/`, `selection_data.py`). It proposes how to replace the target-dependent methodology with an unsupervised methodology to operate "without specifying a target column," and what changes are needed in the code and configuration. This phase involves only writing the report; no code modifications will be performed.

**Background**
- The current pipeline is based on a supervised (target-required) design: Stage 1 (Correlation/Dependence-based initial screening) → Stage 2 (Stability Selection/Knockoffs) → Stage 3 (Diversity) → Stage 4 (Range Calibration and Multicollinearity Cleanup).
- Requirement: The pipeline must operate in a mode that does not specify a target. This means replacing target-required methodologies with "unsupervised" methodologies of a similar purpose.

**Target Dependency Points (Summary)**
- **Stage 1 Screening (feature vs. target tests)**
  - File: `selection/screening.py`
  - Function: `screening_out_of_fold(df, feature_columns, target_column, ...)`
  - Internal Usage: All tests like Spearman/Kendall/HSIC/dCor use `y=target`.
- **Stage 2 Selection (Supervised)**
  - File: `selection/selector.py`
  - Functions: `stability_selection(...)` (Lasso-based), `modelx_knockoffs(...)` (Knockoffs statistics), both require a target.
  - Pipeline Connection: `_run_stage2_selection` in `selection/pipeline.py:744` calls these functions.
  - Auxiliary Diagnostics: `_holdout_effects` in `selection/pipeline.py:969` calculates feature vs. target correlation in the holdout set.
- **Stage 4 Calibration – Boolean Statistics (effect relative to events)**
  - File: `selection/boolean_stats.py`
  - Function: `compute_boolean_feature_stats(df, target, features, ...)` (OR/RR/G-test/Fisher/Cliff’s δ, etc., all require a target).
- **Target Usage in Representative/Multicollinearity Cleanup**
  - File: `selection/selector.py`
  - Function: `enforce_vif_and_partial_r2(...)` uses `_partial_r2(target, features, feature_name)`.
- **Target Column Enforcement in Pipeline**
  - File: `selection/pipeline.py:343-345` (raises an exception if `target_column` is not in the frame).

**Overall Replacement Strategy (Targetless Mode)**
- **Core Idea:** Shift the focus from "relationship with the target" to the "temporal structure/stability/redundancy of the features themselves."
  - **Stage 1:** Change from (feature ↔ target) independence tests to (feature_t ↔ feature_{t+lag}) self-dependence (autocorrelation/auto-dependence) tests.
  - **Stage 2:** Replace Supervised Stability/Knockoffs selection with "Unsupervised Stability Selection" (based on the frequency of representative features selected from bootstrap subsamples).
  - **Stage 4:** Replace Partial R² (vs. target) with "Self-R² (the degree to which a feature can be explained by the other features)" or omit it. VIF is maintained (as it's targetless).
  - **Boolean Stats:** Remove target-relative effects and replace with descriptive statistics (frequency/burstiness/co-occurrence) as an option, or disable them.

**Replacement Mapping (Targetless Alternatives)**
- **Screening (Target-required) → Self-dependence Screening (Targetless)**
  - Spearman/Kendall → Auto-Spearman/Auto-Kendall: Test `corr(feature(t), feature(t+lag))`.
  - HSIC → Auto-HSIC: `HSIC(feature(t), feature(t+lag))` + block bootstrap for p-value.
  - dCor → Auto-dCor: `dCor(feature(t), feature(t+lag))` + block permutation for p-value.
  - Combination/Correction remains the same: Simes/Empirical Brown/Fisher combination → BY/BH FDR application.
- **Stage 2 Selection (Target-based) → Unsupervised Stability Selection**
  - **Existing:** L1/Knockoffs/holdout target effects.
  - **Alternative:** For each row subsample iteration, run `greedy_representative_selection` (controls redundancy based on partial correlation) → Aggregate feature selection frequency π̂_i → Select features above a cutoff (e.g., 0.6).
  - Knockoffs will be disabled (no similar unsupervised statistic exists).
- **Stage 4 Partial R² (target) → Self-R² or Omission**
  - **Self-R² Definition:** For each feature x_i, linearly predict it using the other features X_{-i} and calculate R²(x_i ~ X_{-i}). A lower value indicates higher "uniqueness."
  - Used as a reference metric for reporting/selection alongside VIF. Threshold-based removal is optional.
- **Boolean Stats (vs. target) → Descriptive Stats (Optional)**
  - Only record metrics like frequency/sparsity/burstiness (interval CV)/co-occurrence rate. Recommended to be disabled by default.

**File-by-File Change Proposals (Specifics)**
- **selection/pipeline.py**
  - Add a supervision mode flag to `StageOneConfig`: `supervision: "unsupervised" | "supervised"` (default: supervised). Alternatively, automatically enable targetless mode if `target_column: null`.
  - Remove the target enforcement check in `run(...)` for targetless mode: branch `selection/pipeline.py:343-345`.
  - Support targetless branching for Screening calls: Pass `auto_*` methods to the existing `screening_out_of_fold(...)` so it can operate without a target.
  - Stage 2 branching:
    - Targetless: Call `unsupervised_stability_selection(...)` (a proposed new function, see below). Ignore/log Knockoffs.
    - Existing `_holdout_effects` should not be used or should be replaced with an "effect stability" metric (optional).
  - Boolean Stats: Guard the call at `selection/pipeline.py:475-485` to only execute if `stage4_calibration.boolean_stats.enabled` is true and a target exists.
- **selection/screening.py**
  - Extend the existing function: `screening_out_of_fold(df, feature_columns, target_column, ...)` should, when `auto_*` methods are included in `methods`, replace `y` with `shift(x, lag)` for each feature and use the same test paths (Spearman/Kendall/HSIC/dCor).
    - Example config key: `methods.auto_spearman: {enabled: true, lags: [1,3,12,48]}`
    - The effect size should be the maximum absolute value across lags, and the p-value should be combined using a rule (Simes/Empirical Brown).
    - Reuse existing HSIC/dCor implementations for block bootstrap/permutation.
  - The output contract remains the same (`ScreeningResult`).
- **selection/selector.py**
  - **New Function:** `unsupervised_stability_selection(frame, features, *, subsamples, sample_fraction, cutoff, corr_cap, max_features, partial_alpha, random_state)`
    - For each subsample, run `greedy_representative_selection(...)` → Aggregate selection frequencies π̂_i → Apply cutoff.
    - Reuse the existing `StabilitySelectionOutcome` schema for diagnostics (frequency, support count, etc.).
  - Extend `enforce_vif_and_partial_r2(...)`: Add a `use_target_for_partial: bool=False` argument.
    - When `False`, implement an additional function to calculate `self_r2(feature | others)` instead of `_partial_r2` and record it in `diagnostics.partial_r2` (either keep the field name or add a new key like `partial_r2_self`).
- **selection/boolean_stats.py**
  - Guard the call to `compute_boolean_feature_stats(...)` on the pipeline side. Do not execute it in targetless mode.
  - Optional: A separate `compute_boolean_feature_descriptives(...)` (for frequency/burstiness/co-occurrence) could be added.
- **selection_data.py**
  - No changes needed (the pipeline will branch based on the mode). The CLI remains the same.

**Configuration Change Proposal (Example YAML)**
- File: `Config/experiment_config.yaml`
- Purpose: Enable targetless mode, use `auto_*` screening, disable Knockoffs, and use targetless Stage 2 stability selection.

```yaml
selection_pipeline:
  enabled: true
  dry_run: false
  supervision: unsupervised     # Or target_column: null
  target_column: null           # Signal for targetless mode
  artifact_dir: "results/selection_artifacts"
  min_bars_per_split: 256

  stage1_screening:
    methods:
      auto_spearman: {enabled: true, lags: [1, 3, 12, 48], cliffs_delta: false}
      auto_hsic:     {enabled: false, bootstrap: {samples: 64, block_size: 64, method: block}}
      auto_dcor:     {enabled: false, permutations: 64, block_size: 64, method: block}
    combine: {method: simes, alpha: 0.1, fdr: by}
    enforce_significance: true
    effect_min: 0.0

  stage2_selection:
    corr_cap: 0.9
    max_features: 64
    inner_ratio: 0.5
    stability_selection:
      enabled: true
      subsamples: 48
      sample_fraction: 0.6
      cutoff: 0.6
      max_features: 32
      # L1-related params like alpha_grid, max_iter, tolerance are ignored (targetless)
    knockoffs:
      enabled: false  # Disabled in targetless mode

  stage3_diversity:
    enabled: true
    method: facility_location
    top_k: 32
    family_cap: {max_per_family: 2}

  stage4_calibration:
    vif_threshold: 12.0
    robust_quantiles: [0.01, 0.99]
    mad_margin: {factor: 1.4826}
    boolean_stats: {enabled: false}
```

**Selection (Representativeness) Logic Maintenance/Enhancement Points**
- `greedy_representative_selection(...)` and Stage 3 `diversity` are inherently targetless (based on partial correlation similarity). They should be maintained and use the Stage 1 effect scores (targetless self-dependence effect) as weights.
- The current implementation in `selection/diversity.py`, which sets weights to a uniform 1.0 if all effect scores are zero, is also safe for targetless mode.

**Migration Impact**
- Much of the test code depends on target generation (e.g., `test_selection_stage1.py`, `test_selection_stage2.py`). When switching to targetless, separate "unsupervised" test cases should be added, or existing tests should be branched.
- Existing Knockoffs-related results/snapshot fields may be disabled/empty in targetless mode (keep the snapshot schema but leave the values empty).

**Verification Plan**
- **Functional Verification**
  - Verify that the pipeline runs without exceptions in targetless mode and that artifacts (`artifact_dir` JSON, model config updates, snapshots) are generated.
  - When only `auto_*` methods are enabled in Stage 1, confirm that significant features are found (pass FDR).
  - In Stage 2 targetless stability selection, check the distribution of selection frequencies and the behavior of the `cutoff`.
- **Statistical Sanity Checks**
  - Features with self-dependence (e.g., AR integrated signals) should have high effect scores/significant p-values.
  - Pure random noise features should have a low FDR pass rate.
- **Regression Testing**
  - Measure the Jaccard stability of selection results (from repeated runs) on the same data and seed.

**Risks and Limitations**
- The targetless criteria do not guarantee "direct relevance to a predictive target." Instead, they use "self-structure/stability/non-redundancy" as a measure, so the correlation with final model performance is domain-dependent.
- With the removal of Knockoffs, the explicit FDR (q) guarantee is lost (BY/BH is only applied at the Screening level).
- Auto-HSIC/dCor are computationally expensive (recommend reducing the number of samples, adjusting block size, or using only Auto-Spearman if needed).

**Summary: What to Replace and How (Key Points)**
- **Stage 1:** Spearman/Kendall/HSIC/dCor (target-required) → Auto-Spearman/Auto-Kendall/Auto-HSIC/Auto-dCor (targetless self-dependence).
- **Stage 2:** Stability Selection/Knockoffs (target-required) → Unsupervised Stability (bootstrap + representative selection frequency).
- **Stage 4:** Partial R² (target) → Self-R² or Omission (VIF is kept). Boolean effect stats → Disabled or descriptive stats.
- **Pipeline:** Allow `target_column` to be unspecified + implement call branching + guard boolean stats calls.

**Reference: Code Locations (Key Line Guide)**
- `selection/pipeline.py:343-345` – Target column existence enforcement.
- `selection/pipeline.py:366-380` – Screening call site.
- `selection/pipeline.py:744-851` – Stage 2 stability selection and holdout effects.
- `selection/pipeline.py:475-485` – Boolean statistics call site.
- `selection/selector.py:544-571` – Partial R² (target) calculation.
- `selection/screening.py:428-499` – Screening OOF implementation (current target-based version).

By implementing the above proposals, features can be selected based on the three axes of "informativeness (self-dependence) + non-redundancy (partial correlation/diversity) + stability (frequency)" even without specifying a target. It is recommended to apply the configuration (YAML) changes first to verify the targetless pipeline path before starting code modifications.