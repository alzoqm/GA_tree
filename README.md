# GA Tree Trading Framework

A GPU-accelerated research framework for evolving cryptocurrency futures strategies as interpretable decision trees. The system combines massive-scale genetic search, walk-forward and combinatorial cross-validation workflows, multi-objective fitness, and statistical robustness diagnostics to surface resilient trading policies.

> **Heads-up**: This repository is under active development. Expect rapid iteration, occasional breaking changes, and incomplete documentation for newly landed features. Pin a specific commit before running extended experiments, and re-run preprocessing/training after pulling updates to avoid config or checkpoint mismatches.

## Architecture At A Glance
| Path / Module                 | Role                                                                                                                                   |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `Config/`                    | Central experiment and feature definitions (`experiment_config.yaml`, `feature_config.yaml`, backup variants).                         |
| `data/`                      | Binance data ingestion, deterministic funding/mark estimation, feature generation helpers.                                             |
| `preprocess_data.py`         | End-to-end data pipeline orchestration driven by Config settings.                                                                      |
| `models/`                    | `GATree` / `GATreePop` tensor representations, feature registries, GPU-friendly tree utilities.                                        |
| `csrc/`                      | C++/CUDA sources for prediction, evolution, and simulator kernels compiled via `setup.py`.                                             |
| `gatree_cuda_compat/`        | Python bindings exposing compiled CUDA kernels; required at runtime for GPU execution.                                                 |
| `evolution/`                 | Selection (PPTS), crossover (subtree, node, root-branch), mutation (node/subtree/parameter) operators, unified `Evolution` loop.       |
| `training/`                  | Simulation, trading environment, multi-objective fitness, WFO/CPCV coordinators, generation runner, caching, pruning.                 |
| `fitness/`                   | CPCV objective aggregation and distributional statistics.                                                                              |
| `analysis/`                  | Robustness probes (PBO/DSR), W&B logger, tree/feature visualisation, post-run analytics tooling.                                      |
| `utils/`                     | Checkpoint manager with resume/cleanup utilities.                                                                                      |
| `selection/`, `selection_data.py` | Optional feature selection CLI and pipeline stages (screening, stability, calibration).                                      |
| `test/`                      | Pytest-based regression suites for selection stages and CUDA trading environment kernels.                                              |

## End-to-End Flow
```
                           +-----------------------------+
                           | Config/experiment_config    |
                           +-----------------------------+
                                         |
                                         v
+--------------------+      +---------------------------+       +-----------------------------+
| Binance / local IO |<-----| preprocess_data.py        |-----> | final_features.(csv|pkl)    |
+--------------------+      +---------------------------+       +-----------------------------+
                                         |
                                         v
                           +-----------------------------+
                           | GATreePop initialization    |
                           | (population tensor on GPU)  |
                           +-----------------------------+
                                         |
                                         v
                    +-------------------------------------------+
                    | Evolution engine (selection/crossover/    |
                    | mutation with CUDA kernels)               |
                    +-------------------------------------------+
                                 |
                 +---------------+----------------+
                 |                                |
                 v                                v
+-------------------------------+   +-------------------------------+
| WalkForwardOptimizer (WFO)    |   | CPCVManager (CPCV scheduler)  |
+-------------------------------+   +-------------------------------+
                 |                                |
                 +---------------+----------------+
                                 v
                      +------------------------------+
                      | Fitness simulator & metrics  |
                      | (IS/OOS objectives, pruning) |
                      +------------------------------+
                                 |
                      +----------+-----------+
                      | Robustness diagnostics |
                      | (PBO, DSR, bootstrap)  |
                      +----------+-----------+
                                 |
                      +------------------------------+
                      | Artifacts, checkpoints, W&B   |
                      | logging & analysis exports    |
                      +------------------------------+
```

## Features
- **Data & Feature Engineering**: Deterministic download/merge of spot, futures, and funding data with optional proxy estimation, memory optimisations, and YAML-driven multi-timeframe feature generation (`preprocess_data.py`, `data/`).
- **GPU-native Tree Representation**: `GATree`/`GATreePop` store trees in fixed-shape tensors with adjacency, depth, and branch metadata enabling batched CUDA kernels and validation hooks (`gatree_cuda.validate_trees`).
- **Genetic Operators**: PPTS multi-objective selection, weighted crossover chains (subtree, root-branch, node), structural and parametric mutation suites with device-aware limits and context-sensitive behaviour.
- **Training Strategies**: Switchable Walk-Forward Optimisation and Combinatorial Purged Cross-Validation. WFO supports sliding/expanding windows, embargo, warmup, and OOS aggregation. CPCV includes scheduler-driven splits, reuse/caching, and parallel combination processing.
- **Holdout Support**: Set `cpcv.holdout` in `Config/experiment_config.yaml` to reserve the trailing portion of data as an untouched holdout window. The configured bars (or ratio) are excluded from CPCV folds and only used for the final evaluation that gets exported with `cpcv_results.json`.
- **Multi-Objective Fitness**: `training/multi_objective_fitness` evaluates configurable metric sets (return, drawdown, PSR, Sharpe, win-rate, hit-rate) with min/max orientation, clamping, and Pareto ranking.
- **Trading Environment Simulator**: CUDA-backed environment modelling leverage, slippage (ATR-relative), taker fees, liquidation, and root-branch balance penalties; supports node-level attributions for pruning.
- **Checkpoint & Resume**: `utils/checkpoint_manager.CheckpointManager` manages main/cpcv/wfo checkpoints, config/data fingerprint validation, tensor offloading, cleanup policies, and resume metadata (including W&B run IDs).
- **Generation Caching & Pruning**: CPCV cache deduplicates segment simulations across combinations; optional `PopulationPruner` applies node-level attrition before aggregation.
- **Robustness Analytics**: `analysis/robustness` derives heuristic PBO, deflated Sharpe, block-bootstrap CI, and SPA placeholders; results logged per candidate with notes for threshold breaches.
- **Weights & Biases Integration**: Optional experiment tracking, artifact logging, per-generation tables, configurable metric monitoring (`analysis/wandb_logger`).
- **Feature Selection Pipeline**: Multi-stage screening, stability selection, diversity control, and calibration via `selection_data.py`; outputs curated feature configs for downstream GA runs.
- **Visualisation & Reporting**: `analysis/analyze_best.py`, `analysis/trees.py`, and `analysis/visualize_topk_html.py` export feature usage, interaction heatmaps, HTML tree renders, and permutation importance.
- **CUDA Extension**: `setup.py` builds unified kernels powering prediction, crossover/mutation, and simulator loops; CPU fallbacks are limited, so GPU build is strongly recommended.
- **Testing**: Pytest suite (`test/`) covers selection stages and CUDA environment equity accounting; CUDA-marked tests require compiled extensions and a GPU device.

## Usage
### Prerequisites
- Python 3.9 or later.
- NVIDIA GPU with a compatible CUDA toolkit (11.x or newer recommended).
- C++17-capable compiler toolchain.
- TA-Lib C library (feature generators rely on TA-Lib indicators).
- Optional: Binance API key/secret for historical data downloads; Weights & Biases account for experiment logging.

### 1. Clone & Environment Setup
```bash
git clone <repo-url>
cd GA_tree
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Build CUDA Extension
```bash
python setup.py build_ext --inplace
```
If you skip this step `main.py` will abort after reporting `gatree_cuda` is missing.

### 3. Configure Experiment
Edit `Config/experiment_config.yaml`:
- `environment`: device, random seed, output directory, data directory.
- `data`: symbol, interval, history depth, output filenames, feature config path.
- `ga_core`: population size/shape, generation count, early stopping.
- `operators`: PPTS settings, crossover weights/modes, mutation probabilities and limits.
- `training_strategy`: choose `wfo` or `cpcv`; specify hold-period for purge/embargo defaults.
- `walk_forward` or `cpcv`: window geometry, parallel chunks, reuse policy, min trades.
- `evaluation`: trading simulator fees/slippage/leverage, IS/OOS objective sets, aggregation rules, pruning config.
- `feature_classification` & optional `selection_pipeline`: maintain feature taxonomy and selection parameters.
- `wandb`, `resumption`, `robustness_filter`, `fitness_objectives`: enable logging, auto-resume, diagnostics.

### 4. Provide Secrets (optional)
Create `.env` with API keys when `data/data_download.py` needs authenticated endpoints and when W&B logging is enabled:
```
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
WANDB_API_KEY=...
```

### 5. Preprocess Data
```bash
python preprocess_data.py --config Config/experiment_config.yaml
```
Generates `final_features.csv/.pkl` and `model_config.yaml` under `environment.data_dir`, injecting funding/mark proxies, memory optimisation, and skipping regeneration when assets already exist and are non-empty.

### 6. (Optional) Run Feature Selection Stages
```bash
python selection_data.py --config Config/experiment_config.yaml
```
Writes curated feature sets and calibration artefacts to `selection_pipeline.artifact_dir`; honour `--dry-run` to validate without persisting.

### 7. Launch Training
```bash
python main.py                       # fresh run
python main.py --resume              # resume using latest checkpoint
python main.py --clear-checkpoints   # wipe checkpoints before starting
python main.py --config path/to/alt_config.yaml  # custom config
```
Pipeline stages: environment setup, dataset validation/loading (prefers pickle), GPU population build, evolution loop, WFO or CPCV orchestration, evaluation, robustness, artifact export. Checkpoints reside in `<output_dir>/checkpoints/` and retain W&B run IDs when enabled.

### 8. Inspect Outputs
Key artefacts in `environment.output_dir`:
- `best_population.pth` / `best_population_topK.pth`: elite chromosomes (torch tensors).
- `wfo_results.json` or `cpcv_results.json`: aggregated metrics, objective names/directions, scheduler metadata.
- `cpcv_cache/` (optional): generation-level simulation caches.
- `checkpoints/`: generation snapshots (population tensors, objectives, node stats).
- W&B logs/artifacts (when enabled) synced to the configured project.

### 9. Post-run Analytics
```bash
python analysis/analyze_best.py --config Config/experiment_config.yaml
python analysis/visualize_topk_html.py --config Config/experiment_config.yaml
```
Exports feature usage charts, interaction heatmaps, HTML visualisations of top strategies, and raw JSON/CSV tables under `<output_dir>/analysis/`.

### 10. Run Tests
```bash
pytest
```
Use `pytest -m "not cuda"` when running on machines without a GPU or compiled kernels.

## Configuration Highlights & Tips
- **Auto Resume**: `resumption.auto_resume` combined with `--resume` ensures recovery after interruptions. Stale checkpoints are pruned via `cleanup_old_checkpoints` / `keep_last_n_generations`.
- **Robustness Filters**: `robustness_filter` thresholds can exclude fragile candidates before aggregation; results feed into `analysis/robustness` for reporting.
- **Aggregation Metrics**: `fitness_objectives.cpcv_objectives` defines CPCV-specific distributions (mean, median, IQR, hit-rate). Add names/minimize flags to align W&B logging and result exports.
- **Caching**: Enable `cpcv.evaluation_cache_dir` to reuse simulation segments across generations, reducing duplicated GPU workloads.
- **Population Pruning**: Activate `evaluation.pruning.enabled` to leverage node statistics and maintain compact trees.
- **WFO Aggregation**: `evaluation.oos_aggregate` supports percentile/consistency methods and optional deflated Sharpe reporting (`include_dsr`).

## Troubleshooting
- **Missing CUDA Module**: Re-run `python setup.py build_ext --inplace`; verify compiler and CUDA toolkit paths.
- **Data Mismatch on Resume**: Checkpoint manager warns if `final_features` changed size or timestamp; regenerate data or clear checkpoints when configs diverge.
- **Insufficient Trades**: WFO/CPCV skip windows/splits failing `min_trades_per_window` / `min_trades_per_split`; adjust parameters or expand dataset.
- **W&B Optional**: Disable by setting `wandb.enabled: false` or `mode: disabled` when offline.

This README reflects the current codebase; update `Config/` values to suit your market, horizon, and experiment scale.
