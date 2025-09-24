# CPCV Learning Mode Design

## 1. Goals & Scope
- Introduce a CPCV (Combinatorial Purged Cross-Validation) learning mode that is more robust than the existing Walk-Forward Optimization (WFO) without removing WFO.
- Keep the PPTS selection operator and the current multi-objective evaluation infrastructure, extending the objective vector with robustness-focused distributional statistics (median/IQR/CVaR/turnover/complexity) derived from CPCV splits.
- Provide a post-selection robustness filter (PBO, DSR, SPA/RC, bootstrap, MinTRL) to control selection bias and quantify confidence before the final lockbox check.
- Allow the new mode to be chosen through configuration so experiments can switch between `wfo` and `cpcv` without code changes.

## 2. High-Level Architecture
```
┌──────────────────────────────┐
│  experiment_config.yaml      │
│  training_strategy.mode = X  │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ StrategyOrchestrator         │
│  - WFOManager (existing)     │
│  - CPCVManager (new)         │
└─────────────┬────────────────┘
              │
      ┌───────┴────────┐
      ▼                ▼
┌──────────────┐  ┌──────────────┐
│ WFO windows  │  │ CPCV folds   │
│ (rolling IS) │  │ (purge+emb)  │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
    GA evolution with PPTS using
    MultiObjectiveFitnessEvaluator
       │                 │
       ▼                 ▼
Distributional metrics aggregation
       │                 │
       ▼                 ▼
Robustness filter: PBO → DSR → SPA/RC → Block bootstrap → MinTRL
       │
       ▼
Lockbox OOT validation & ranking
```

`StrategyOrchestrator` is a thin wrapper invoked from `main.py`. It reads `training_strategy.mode` and delegates to the existing WFO pipeline or the new CPCV pipeline. No code is removed; WFO remains untouched.

## 3. Configuration Additions
Append to `experiment_config.yaml` (names chosen to avoid collisions):
```yaml
training_strategy:
  mode: "cpcv"          # "wfo" (default, backward compatible) | "cpcv"
  max_hold_period_bars: 1440   # Used for purge/embargo length (example: 5 days @ 5m)

cpcv:
  folds: 10                    # F in CPCV
  test_folds: 2                # k (number of simultaneous OOS folds)
  shuffle_seed: 42             # Deterministic combination ordering
  purge_bars: null             # Defaults to max_hold_period_bars
  embargo_bars: null           # Defaults to max_hold_period_bars
  min_trades_per_split: 32     # Skip split if insufficient trading activity
  max_combinations: null       # Optional cap; else C(F,k)
  parallel_chunks: 4           # Number of fold-combo batches evolved per generation
  reuse_population: true       # Share population state across combos within a generation
  evaluation_cache_dir: "results/cpcv_cache"

robustness_filter:
  pbo:
    enabled: true
    min_splits: 12
    cscv_k: 2
  dsr:
    enabled: true
    sr_benchmark: 0.0
  spa:
    enabled: true
    bootstrap_samples: 5000
    studentized: true
  block_bootstrap:
    enabled: true
    method: "stationary"
    block_length: 50
    samples: 5000
  mintrl:
    enabled: true
    confidence: 0.95

fitness_objectives:
  # Unified list used by CPTS for multi-objective selection
  cpcv_objectives:
    names: [
      "median_oos_return",
      "median_oos_sharpe",
      "iqr_oos_sharpe",
      "cvar95_oos_return",
      "oos_turnover",
      "tree_complexity",
      "median_oos_psr",
      "median_oos_profit_factor",
      "iqr_oos_drawdown",
      "is_mean_return",
      "is_max_drawdown",
      "is_psr",
      "is_root_balance_penalty"
    ]
    minimize: [
      false,  # median_oos_return (maximize → negate before PPTS)
      false,  # median_oos_sharpe
      true,   # iqr_oos_sharpe
      true,   # cvar95_oos_return
      true,   # oos_turnover
      true,   # tree_complexity
      false,  # median_oos_psr
      false,  # median_oos_profit_factor
      true,   # iqr_oos_drawdown
      false,  # is_mean_return (existing objective)
      true,   # is_max_drawdown (existing)
      false,  # is_psr (existing)
      true    # is_root_balance_penalty (existing)
    ]
    normalization: "minmax"
```

Key points:
- `mode` allows runtime selection. Default remains `wfo` to preserve legacy behaviour.
- Fold-level purge and embargo lengths fall back to the maximum holding period defined in training strategy.
- Existing `evaluation.objectives`, `is_objectives`, and `oos_objectives` stay intact; the new `cpcv_objectives` combine distributional OOS metrics with current IS objectives so PPTS sees a richer objective tensor.

## 4. CPCV Splitter with Purge & Embargo
Module: `training/cpcv.py`

### Data structures
```python
@dataclass
class CPCVSplit:
    combo_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    embargo_idx: np.ndarray

class CPCVScheduler:
    def __init__(self, config, total_length):
        ...
    def iter_combinations(self) -> Iterable[CPCVSplit]:
        ...
```

### Algorithm
1. Partition the timeline into `folds` equal-length, chronological blocks.
2. Enumerate combinations of `test_folds` (k). Optionally truncate at `max_combinations`.
3. For each combination:
   - Test indices: union of fold indices in the combination.
   - Purge: remove samples whose labels overlap with test indices by `purge_bars` from the training set.
   - Embargo: drop `embargo_bars` immediately after the last test index.
   - Return `train_idx`, `test_idx`, `embargo_idx` arrays (all 1D, sorted).
4. Skip splits where remaining train length or test length is < `min_trades_per_split` or where embargo exceeds available data.
5. Provide metadata (combo id, relative position) for caching and logging.

All indices operate in bar-space (consistent with existing runner APIs). The scheduler exposes helper methods:
- `get_combo_pairs()` to allow batching combos per generation (respecting `parallel_chunks`).
- `max_hold_period_bars` is used both for purge and embargo by default to match the longest strategy action horizon.

## 5. Evolution Workflow in CPCV Mode
Module: `training/cpcv_manager.py` (mirrors `training/wfo.py`).

### Per Generation Workflow
```
for combo_batch in scheduler.iter_batches():
    # 1. Build/refresh shared population on GPU
    population = evolution_engine.population
    for combo in combo_batch:
        # 2. Evaluate IS metrics on CPCV train slice
        is_objectives = evaluator.evaluate_population(... train_idx ...)
        # 3. Run PPTS selection with IS objectives (existing logic)
        parents = selection.pick_parents(is_objectives)
        # 4. Create offspring, mutate, etc. (existing evolution loop)
        # 5. Collect OOS time series for each individual on combo.test_idx
        oos_metrics, oos_returns = collect_metrics(...)
        archive.store(combo_id, individual_id, is_objectives, oos_metrics, oos_returns)
    # 6. Synchronize elites across combos if reuse_population is true
    # 7. Proceed to next generation
```

- Evaluation uses `training.runner.generation_training` and `generation_test`, now parameterized by generic index ranges instead of fixed WFO windows.
- `CPCVArchive` stores per-individual metrics for every combination, including raw OOS return series (needed for CVaR and bootstrap).
- PPTS continues to operate on the objective tensor. For each combo we assemble an objective matrix with columns ordered according to `fitness_objectives.cpcv_objectives`.

### Objective Assembly
1. **Per combo**
   - Compute existing instantaneous metrics (mean_return, max_drawdown, psr, root_balance_penalty, etc.) using existing evaluation code.
   - Store OOS returns and trade logs for turnover calculations.
2. **Across combos**
   - Use `fitness/distributional_metrics.py` (new helper) to aggregate distributional statistics:
     - `median_oos_return`, `median_oos_sharpe`, `median_oos_psr`, `median_oos_profit_factor`
     - `iqr_oos_sharpe`, `iqr_oos_drawdown`
     - `cvar95_oos_return` using empirical returns (left tail)
     - `oos_turnover` from trade metadata (mean trades per unit time).
   - Combine with best available in-sample metrics from the last generation (`is_mean_return`, etc.).
3. **Orientation**
   - Convert maximize metrics to minimization by negation before feeding to PPTS (`objectives_matrix[:, maximize] *= -1`).
   - Clamp invalid individuals with ±1e9 sentinel values (existing guard rails).

The resulting matrix has shape `[population_size, len(cpcv_objectives)]` and is passed unchanged to PPTS for both elite selection and parent sampling.

## 6. Robustness Filter Pipeline
After evolution stabilizes, take the Pareto front (or top-N by crowding distance) and run the following sequential checks. Implement in `analysis/robustness.py`:

1. **PBO via CSCV**
   - Reuse CPCV splits but permute combinations according to Bailey et al. to compute the Probability of Backtest Overfitting.
   - Require at least `pbo.min_splits` valid permutations; otherwise mark candidate as indeterminate.
   - Discard candidates with `PBO > 0.2` (configurable).
2. **Deflated Sharpe Ratio (DSR)**
   - Use existing `analysis.stat_tests.deflated_sharpe_ratio` with `n_trials` equal to the number of evaluated strategies.
   - Require `DSR >= 0.0` (or > 0.05 for 95% confidence).
3. **White’s Reality Check / Hansen’s SPA**
   - Implement bootstrapped SPA with stationary blocks (shared with bootstrap config) using OOS return series across combos.
   - Compare against baseline strategies (buy-and-hold, moving-average, etc.) either precomputed or loaded from `analysis/baselines.py` (add new helper).
   - Keep candidates whose SPA p-value < 0.05.
4. **MinTRL / PSR-based Track Record**
   - Calculate minimum track record length required for Sharpe significance given the number of trials; rule out shorter histories.
5. **Block Bootstrap Confidence Intervals**
   - Generate lower/upper bounds for Sharpe, CAGR, and MaxDD using stationary or circular bootstrap; require lower Sharpe bound > 0.

Candidates passing all checks are marked “robust”. Store their metrics, PBO, DSR, SPA p-values, and bootstrap intervals in a summary table for logging.

## 7. Robustness Scoring & Ranking
Create `analysis/robustness_scoring.py`:
- Inputs: candidate ID, aggregated OOS metrics, statistical test outputs, complexity metrics, turnover.
- Compute normalized sub-scores (0–1) for each pillar:
  - **Stability**: median Sharpe, IQR Sharpe, positive split ratio.
  - **Significance**: DSR, PSR, SPA/RC pass flag.
  - **Overfitting Risk**: PBO, MinTRL satisfaction.
  - **Risk**: CVaR, MaxDD.
  - **Complexity**: tree node count & depth (penalize larger trees).
  - **Tradability**: turnover, net return after costs.
- Combine with configurable weights (default equal). Present final ranking along with raw metrics.

The top-ranked robust candidate proceeds to the **lockbox** evaluation (final OOT period untouched by CPCV). Store lockbox performance separately for audit.

## 8. Integration Points
1. **`main.py`**
   - Read `training_strategy.mode`.
   - Instantiate `WalkForwardOptimizer` if `wfo`, or `CPCVManager` otherwise.
   - Pass shared dependencies (population, evaluation configs, checkpoint manager, wandb logger).
2. **Checkpointing**
   - Mirror the WFO checkpoint layout: save CPCV archive snapshots (per combo) and resume state.
   - Cache OOS time series in `evaluation_cache_dir` keyed by `combo_id` and generation to avoid recomputation.
3. **Logging**
   - Extend `wandb_logger` to log distributional metrics histogram, PBO/DSR trajectories, and bootstrap CI bands.
   - Emit summary tables (top-N Pareto front, robustness filter results) to `analysis/analyze_best.py` for post-hoc review.
4. **Testing Hooks**
   - Unit tests for `CPCVScheduler` (purge/embargo invariants, combination counts, determinism).
   - Property tests for `distributional_metrics` (median/IQR/quantile correctness).
   - Integration tests running a tiny dataset with `folds=4, test_folds=1` to ensure end-to-end compatibility.

## 9. Implementation Roadmap
1. **Infrastructure**
   - Add config sections and data classes (`pydantic` or manual validation) for `training_strategy`, `cpcv`, and `robustness_filter`.
   - Implement `StrategyOrchestrator` to keep `main.py` slim.
2. **Data Splitting**
   - Build `CPCVScheduler` with purge/embargo logic and tests.
3. **Evolution Loop**
   - Create `CPCVManager` replicating WFO’s interface (run, resume, archive).
   - Develop `CPCVArchive` & `distributional_metrics` utilities.
4. **Fitness Expansion**
   - Wire aggregated objectives into `MultiObjectiveFitnessEvaluator` (ensure normalization handles new columns).
   - Update PPTS calls to expect the expanded objective tensor.
5. **Robustness Analytics**
   - Implement PBO, SPA, bootstrap routines under `analysis/` with reusable helpers.
   - Provide CLI or notebook integration for manual inspection.
6. **Lockbox Evaluation**
   - Add `evaluation.lockbox` config block (dates or index range) and extend `analysis/analyze_best.py` to load lockbox performance.
7. **Documentation & Examples**
   - Update `README.md` and WandB docs with instructions for running `mode=cpcv`.
   - Provide sample notebook showing the full CPCV-to-lockbox workflow.

## 10. Backward Compatibility & Migration Notes
- WFO remains default and unchanged; no migration needed for legacy experiments.
- Existing checkpoints continue to load because `training_strategy.mode` defaults to `wfo` when absent.
- New fitness objectives are only activated in CPCV mode; WFO continues to use `is_objectives` and `oos_objectives` as before.
- Statistical filter modules are optional: if disabled via config, the pipeline still produces CPCV metrics, but candidates are flagged as "untested" for downstream consumers.

---

With this design, the project gains a CPCV-based learning mode that enforces information leak prevention, evaluates consistency across numerous OOS paths, leverages PPTS for multi-objective selection, and filters survivors using rigorous selection-bias corrections before final deployment.