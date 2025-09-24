# Selection Pipeline Usage Guide

This document explains how to run and customize the feature selection pipeline implemented inside the `selection` package. The guide covers configuration knobs, stage-by-stage behaviors, command line usage, and tips for running the pipeline in different environments.

## 1. Overview

The selection pipeline transforms a prepared feature dataframe into a production-ready feature whitelist, numeric range calibration, and audit snapshot. It follows a phased roadmap:

1. **Integrity & Screening (Stage 1)** – Cleans the feature matrix, evaluates dependency tests (Spearman, HSIC, dCor, etc.), and applies multiple-hypothesis correction.
2. **Selector (Stage 2)** – Optionally runs stability selection (subsampled L1) to remove brittle features while respecting correlation caps.
3. **Diversity (Stage 3)** – Promotes complementary signals using facility-location or DPP-based objectives with family capping.
4. **Calibration** – Produces robust numeric bounds, boolean whitelists, and comparison maps before writing artifacts.

The pipeline can operate in dry-run mode (only in-memory) or persist updates to the model config and selection snapshot files.

## 2. Prerequisites

- A preprocessed feature dataframe saved as CSV or Pickle (`final_features_path` / `final_features_path_pkl`).
- A baseline model config YAML (`model_config_path`) using the schema consumed by the training pipeline.
- (Optional) `Config/feature_classification_active.yaml` for snapshot history.
- Experiment configuration YAML (e.g., `Config/experiment_config.yaml`) describing CPCV, training strategy, environment metadata, and default selection settings.

All paths are resolved relative to the directory that contains the experiment config unless absolute paths are supplied.

## 3. Key Modules

| Module | Responsibility |
| --- | --- |
| `selection/loader.py` | Resolves file paths, loads configs and feature data. |
| `selection/pipeline.py` | High-level orchestrator implementing the staged roadmap. |
| `selection/integrity.py` | Applies missing/constant/duplicate/correlation gates. |
| `selection/screening.py` | Runs dependency tests (Spearman, HSIC, dCor) and FDR control. |
| `selection/selector.py` | Stability selection and collinearity enforcement. |
| `selection/diversity.py` | Facility-location and DPP diversity routines. |
| `selection/calibration.py` | Range calibration, MAD margins, VIF checks. |
| `selection/snapshot.py` | Atomic snapshot writes and metadata hashing. |

## 4. Configuration

The `selection_pipeline` section of your experiment config drives runtime behavior. Important keys include:

```yaml
selection_pipeline:
  enabled: true
  dry_run: false                # set true to skip disk writes
  target_column: "target"
  artifact_dir: "results/selection_artifacts"
  min_bars_per_split: 256

  integrity:
    max_missing_ratio: 0.4
    min_unique_fraction: 0.005
    max_zero_fraction: 0.995
    correlation_threshold: 0.999

  stage1_screening:
    random_state: 123
    methods:
      spearman: {enabled: true, cliffs_delta: true}
      hsic:
        enabled: true
        bootstrap: {samples: 48, block_size: 64, method: block}
      dcor:
        enabled: true
        permutations: 48
        block_size: 64
    combine: {method: simes, alpha: 0.1, fdr: by}

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
      alpha_grid: [0.8, 0.4, 0.1]
      random_state: 21
      max_iter: 200
      tolerance: 1e-4

  stage3_diversity:
    enabled: true
    method: dpp                  # or "facility_location"
    top_k: 32
    family_cap: {max_per_family: 2}
    lambda: 1.0                  # DPP kernel sharpness
    jitter: 1e-6                 # PSD stabilization

  stage4_calibration:
    vif_threshold: 12.0
    robust_quantiles: [0.01, 0.99]
    mad_margin: {factor: 1.4826}

  feature_whitelist: []
```

**Notes**
- `stage1_screening.methods` accepts any combination of supported tests. Disabled methods are skipped. Each method can customize bootstrap/permutation counts and block sizes for time-series safety.
- `combine` parameters choose how per-horizon p-values are aggregated (`simes`, `empirical_brown`, etc.) and which FDR rule to apply (`bh`, `by`).
- Stage 2 runs only when `stability_selection.enabled` is true; otherwise the pipeline uses Stage 1 candidates directly.
- Stage 3 supports `dpp` (k-DPP MAP approximation) or `facility_location`. DPP automatically falls back to facility-location if the kernel becomes ill-conditioned.
- Calibration inherits Stage 1’s final feature set and enforces VIF/partial R² checks before calculating robust ranges.

## 5. Command Line Execution

The CLI entrypoint is `selection_data.py`:

```bash
python selection_data.py \
  --config Config/experiment_config.yaml \
  --dry-run          # optional flag to prevent file writes
```

Behavior:
- Loads the experiment config and resolves paths via `selection.loader`.
- Instantiates `SelectionPipeline` with the merged configuration.
- Executes the pipeline; if `dry_run` is omitted and the config allows writing, the script updates the model config, appends a snapshot entry, and emits JSON artifacts inside `artifact_dir`.

### Example (Dry Run)

```bash
python selection_data.py --config Config/experiment_config.yaml --dry-run
```

### Example (Full Write)

```bash
python selection_data.py --config Config/experiment_config.yaml
```

Ensure that `selection_pipeline.enabled` is set to `true` in the configuration; otherwise the script exits without running.

## 6. Programmatic Usage

The pipeline can also be executed from Python code, which is useful for notebooks or parameter sweeps:

```python
from pathlib import Path
from selection.loader import load_experiment_config, load_final_features, load_model_config, resolve_paths
from selection.pipeline import SelectionPipeline

cfg_path = Path("Config/experiment_config.yaml")
config = load_experiment_config(cfg_path)
selection_cfg = config["selection_pipeline"]
paths = resolve_paths(config, cfg_path.parent)
frame = load_final_features(paths)
model_cfg = load_model_config(paths.model_config_path)

pipeline = SelectionPipeline(config=config, selection_cfg=selection_cfg, paths=paths, dry_run=False)
artifacts = pipeline.run(frame, model_cfg)

print("Selected", len(artifacts.selected_features), "features")
```

`PipelineArtifacts` exposes:
- `integrity`, `screening`, `stage2`, `diversity`, `selection`, `calibration` diagnostics.
- `selected_features`: final list after calibration/whitelists.
- `updated_model_config`: ready-to-write configuration dictionary.

## 7. Outputs

When `dry_run=False`, the pipeline writes:

- **Model Config** – Updated YAML with `feature_num`, `feature_bool`, and filtered `feature_comparison_map` entries.
- **Selection Snapshot** – Appended to `feature_classification_active.yaml` with metadata (git hash, data meta hash, method settings, whitelists).
- **Artifacts JSON** – Stored under `artifact_dir` (timestamped). Contains screening statistics, selection diagnostics, and calibration summaries.

With `dry_run=True`, these payloads are still produced in-memory inside `PipelineArtifacts` but not persisted.

## 8. Tips & Troubleshooting

- **Missing Target Column** – Ensure the feature dataframe includes `selection_pipeline.target_column`. Adjust the config if your label uses a different name.
- **No Valid Splits** – Decrease `min_bars_per_split`, relax CPCV `min_trades_per_split`, or reduce `training_strategy.max_hold_period_bars` for small datasets.
- **YAML with Python Tuples** – Convert tuple ranges to simple lists (`[low, high]`) if you plan to reload the YAML with `yaml.safe_load`.
- **Time-Series Safety** – Increase bootstrap/permutation block sizes proportionally to the expected dependence horizon. HSIC and dCor functions honour `block_size` and `method` settings.
- **DPP Stability** – Use `lambda` in `[0.5, 2.0]` and keep `jitter` around `1e-6` to maintain positive definiteness. The pipeline automatically falls back when necessary.
- **Performance** – For large datasets, consider lowering Stage 3 `top_k`, reducing HSIC bootstrap counts, or enabling GPU acceleration in future extensions (hooks are left in `selection/utils.py`).

## 9. Related Tests

Pytest suites verify the staged behavior:
- `test/test_selection_stage1.py` – Integrity gates and Stage 1 flow.
- `test/test_selection_stage2.py` – Stability selection and facility-location integration.
- `test/test_selection_stage3.py` – HSIC/dCor screening and DPP diversity logic.

Run with:

```bash
pytest test/test_selection_stage*.py
```

Use these tests as templates when customizing configs or adding new statistical modules.

---
For further architectural details, refer to `selection_data_design.md`, which captures the v2 design specification and phased rollout plan.
