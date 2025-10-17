# /gp_pipeline/runner.py
"""Orchestration for the GP feature generation pipeline."""

from __future__ import annotations

import json
import logging
import operator
import os
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import count
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from deap import gp, tools
from deap.tools.emo import assignCrowdingDist, sortNondominated
from tqdm.auto import tqdm

from data.target import compute_target

from .config import GPConfig, GPConfigError
from .evaluation import BACKENDS, EvaluationCache, choose_backend
from .ir import IRNode, tree_to_ir
from .primitives import build_toolbox
from .timeframe_ops import (
    TimeframeOperatorError,
    build_timeframe_operator_specs,
    register_timeframe_operators,
)
from .validators import run_boolean_validator, run_numeric_validator
from .checkpoint import GPPipelineCheckpointManager


logger = logging.getLogger(__name__)


PathLike = os.PathLike[str] | str


@dataclass(slots=True)
class IndividualRecord:
    index: int
    expression: str
    binary_expression: Optional[str] = None
    feature_type: str = "numeric"
    metrics: Dict[str, float] = None  # type: ignore[assignment]
    validator_objectives: Dict[str, float] = None  # type: ignore[assignment]
    validator_goals: Dict[str, str] = None  # type: ignore[assignment]
    signature: Optional[Tuple[int, int]] = None
    node_count: int = 0
    tree_height: int = 0
    expression_length: int = 0


@dataclass(slots=True)
class GenerationSummary:
    generation: int
    best_metrics: Dict[str, Optional[float]]
    mean_metrics: Dict[str, Optional[float]]
    best_expression: str


@dataclass(slots=True)
class SignatureEntry:
    signature: Tuple[int, int]
    individual: gp.PrimitiveTree
    complexity: Tuple[int, int, int]


class GPFeatureTrainer:
    def __init__(self, cfg: GPConfig, *, resume: bool = False) -> None:
        self.cfg = cfg
        self.resume = bool(resume)
        self._seed(cfg.population.seed)
        self.df_full, feature_columns, dropped_features = self._load_dataframe(cfg)
        if cfg.data.target_column is None:
            raise GPConfigError(
                "Target column could not be resolved from config and automatic generation"
            )
        self.feature_columns = feature_columns
        self.target_column = cfg.data.target_column
        self.feature_mode = cfg.feature.mode.lower()
        if self.feature_mode not in {"numeric", "boolean"}:
            raise ValueError("feature.mode must be 'numeric' or 'boolean'")

        self.mode_fitness = cfg.fitness.for_mode(self.feature_mode)
        self.mode_metrics = list(self.mode_fitness.metrics)
        self.mode_weights = list(self.mode_fitness.weights)
        self.mode_maximize = [bool(flag) for flag in self.mode_fitness.maximize]
        self.mode_thresholds = dict(self.mode_fitness.thresholds)
        self.metric_default_goals = {
            name: maximize for name, maximize in zip(self.mode_metrics, self.mode_maximize)
        }

        creator_weights = [
            weight if maximize else -weight
            for weight, maximize in zip(self.mode_weights, self.mode_maximize)
        ]

        try:
            timeframe_specs = build_timeframe_operator_specs(cfg.timeframe_primitives, self.df_full)
        except TimeframeOperatorError as exc:
            raise GPConfigError(str(exc)) from exc
        register_timeframe_operators(timeframe_specs)
        self.timeframe_specs = timeframe_specs
        extra_primitives = [(name, spec.arity) for name, spec in timeframe_specs.items()]

        self.toolbox = build_toolbox(
            self.feature_columns,
            cfg.population,
            creator_weights,
            extra_primitives=extra_primitives or None,
        )

        backend_candidates = (cfg.backend.primary,) + tuple(cfg.backend.fallbacks)
        if timeframe_specs:
            if "numpy" not in backend_candidates:
                raise GPConfigError(
                    "Timeframe primitives require the numpy backend; add 'numpy' to backend.primary or fallbacks."
                )
            self.backend_name = "numpy"
        else:
            self.backend_name = choose_backend(cfg.backend.primary, cfg.backend.fallbacks)
        self.backend_fn = BACKENDS[self.backend_name]
        self.low_ratio = max(0.0, min(cfg.backend.low_fidelity_ratio, 1.0))
        self.full_eval_top_k = max(0.0, min(cfg.backend.full_eval_top_k, 1.0))
        self.df_subset = self._build_subset()
        self.cache_full = EvaluationCache() if cfg.backend.cache_subtrees else None
        self.cache_subset = EvaluationCache() if cfg.backend.cache_subtrees and self.df_subset is not None else None
        self.history: List[GenerationSummary] = []
        self.best_records: Dict[str, IndividualRecord] = {}
        self._best_signature_records: Dict[Tuple[int, int], Tuple[IndividualRecord, Tuple[int, int, int]]] = {}
        self._subset_signatures: Dict[Tuple[int, int], SignatureEntry] = {}
        self._full_signatures: Dict[Tuple[int, int], SignatureEntry] = {}
        self.output_dir = os.fspath(self._ensure_output_dir())
        self.run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.generation_output_dir = os.path.join(
            self.output_dir,
            f"generations_{self.run_timestamp}",
        )
        os.makedirs(self.generation_output_dir, exist_ok=True)
        self._ckpt = GPPipelineCheckpointManager(self.output_dir)
        self.metric_goals: Dict[str, bool] = {}
        if dropped_features:
            logger.warning(
                "Dropped %d feature(s) highly correlated with target %s: %s",
                len(dropped_features),
                self.target_column,
                ", ".join(dropped_features),
            )
        self._feature_counter = count(1)

        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register(
            "mutate",
            gp.mutUniform,
            expr=self.toolbox.expr,
            pset=self.toolbox.pset,
        )
        self.toolbox.register(
            "select",
            tools.selTournament,
            tournsize=cfg.population.tournament_size,
        )
        height_limit = gp.staticLimit(
            key=operator.attrgetter("height"),
            max_value=cfg.population.max_depth,
        )
        size_limit = gp.staticLimit(
            key=len,
            max_value=cfg.population.max_depth * 2,
        )
        self.toolbox.decorate("mate", height_limit)
        self.toolbox.decorate("mate", size_limit)
        self.toolbox.decorate("mutate", height_limit)
        self.toolbox.decorate("mutate", size_limit)

    @staticmethod
    def _sanitize_mapping_for_json(mapping: Dict[str, object]) -> Dict[str, object]:
        """Create a JSON-safe copy of a flat mapping.

        Converts NaN/Inf/-Inf float-like values to None to avoid emitting
        non-standard JSON numbers. Other values are passed through unchanged.
        """
        clean: Dict[str, object] = {}
        for k, v in mapping.items():
            try:
                fv = float(v)  # type: ignore[arg-type]
            except Exception:
                clean[k] = v
                continue
            if not np.isfinite(fv):
                clean[k] = None
            else:
                clean[k] = fv
        return clean

    def _seed(self, seed: Optional[int]) -> None:
        # Initialize the generator used for low-fidelity sampling first so it always exists.
        self._subset_rng = np.random.default_rng(seed)
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)

    def _load_dataframe(self, cfg: GPConfig) -> Tuple[pd.DataFrame, List[str], List[str]]:
        feature_columns = [str(col) for col in cfg.data.feature_columns]
        target_column = cfg.data.target_column

        if cfg.data.usecols:
            usecols = [col for col in cfg.data.usecols if col is not None]
        else:
            usecols = list(feature_columns)
            if target_column:
                usecols.append(target_column)
        path = cfg.data.path
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path, columns=usecols)
        elif suffix in {".csv", ".txt"}:
            df = pd.read_csv(path, usecols=usecols)
        elif suffix in {".pkl", ".pickle"}:
            df = pd.read_pickle(path)
            df = df[usecols]
        else:
            raise ValueError(f"Unsupported data extension: {suffix}")
        df = df.reset_index(drop=True)

        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required feature columns in dataset: {missing_features}")

        if target_column and target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        if target_column is None:
            print('make target')
            column_name, target_series = compute_target(
                df,
                mode="multi_horizon_risk_adjusted",
                params={},
            )
            if column_name != target_series.name:
                target_series = target_series.rename(column_name)
            if not target_series.index.equals(df.index):
                target_series = target_series.reindex(df.index)
            df[column_name] = target_series
            cfg.data.target_column = column_name
            target_column = column_name
            logger.info(
                "Generated target column '%s' using multi_horizon_risk_adjusted mode",
                column_name,
            )

        dropped_features: List[str] = []
        corr_threshold = cfg.data.max_target_corr
        if corr_threshold is not None and target_column is not None:
            if not 0.0 <= corr_threshold <= 1.0:
                raise GPConfigError(
                    "data.max_target_corr must be between 0 and 1 inclusive"
                )
            numeric_cols = df[[c for c in feature_columns + [target_column] if c in df.columns]].select_dtypes(include="number")
            target_series = numeric_cols.get(target_column)
            if target_series is not None:
                candidate_cols = [col for col in feature_columns if col in numeric_cols.columns]
                if candidate_cols:
                    correlations = numeric_cols[candidate_cols].corrwith(target_series).abs()
                    dropped_features = [
                        col
                        for col, value in correlations.items()
                        if np.isfinite(value) and value >= corr_threshold
                    ]
                    if dropped_features:
                        feature_columns = [col for col in feature_columns if col not in dropped_features]
                        df = df.drop(columns=dropped_features)

        if not feature_columns:
            raise GPConfigError("No feature columns remain after applying correlation filter")

        return df, feature_columns, dropped_features

    def _build_subset(self) -> Optional[pd.DataFrame]:
        if self.low_ratio <= 0.0 or self.low_ratio >= 1.0:
            return None
        n = len(self.df_full)
        subset_n = max(128, int(n * self.low_ratio))
        indices = np.sort(self._subset_rng.choice(n, size=subset_n, replace=False))
        df_sub = self.df_full.iloc[indices].reset_index(drop=True)
        return df_sub

    def run(self) -> Dict[str, object]:
        pop_cfg = self.cfg.population
        # Default: fresh initial population
        population = self.toolbox.population(n=pop_cfg.size)
        hall_of_fame = tools.HallOfFame(self.cfg.output.save_best)
        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"]

        # Resume support
        start_gen = 0
        if self.resume:
            latest = self._ckpt.find_latest()
            if latest is not None:
                payload: Dict[str, object]
                try:
                    payload = self._ckpt.load(latest)
                except Exception as exc:
                    raise RuntimeError(f"Failed to load GP checkpoint '{latest}': {exc}") from exc

                pop_str = payload.get("population_str", [])
                if not (isinstance(pop_str, list) and pop_str):
                    raise RuntimeError(
                        f"Checkpoint '{latest.name}' does not contain a population. "
                        "Cannot resume; remove --resume to start a fresh run."
                    )

                checkpoint_features = payload.get("feature_columns")
                if checkpoint_features:
                    ckpt_features = [str(name) for name in checkpoint_features]
                    current_features = [str(name) for name in self.feature_columns]
                    if set(ckpt_features) != set(current_features):
                        missing = sorted(set(ckpt_features) - set(current_features))
                        extra = sorted(set(current_features) - set(ckpt_features))
                        details: List[str] = []
                        if missing:
                            details.append(f"missing in config: {', '.join(missing)}")
                        if extra:
                            details.append(f"not present in checkpoint: {', '.join(extra)}")
                        detail_msg = "; ".join(details) if details else "feature set mismatch"
                        raise RuntimeError(
                            f"Checkpoint '{latest.name}' uses feature_columns incompatible with the current "
                            f"configuration ({detail_msg}). Ensure the config matches the checkpoint or rerun without --resume."
                        )

                try:
                    population = self._ckpt.restore_population(pop_str, self.toolbox.pset)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to reconstruct population from checkpoint '{latest.name}': {exc}"
                    ) from exc

                if len(population) != pop_cfg.size:
                    logger.warning(
                        "Restored population size %d differs from configured size %d; proceeding with checkpoint population.",
                        len(population),
                        pop_cfg.size,
                    )

                # Restore bookkeeping
                start_gen = int(payload.get("generation_next", 0))
                rt = payload.get("run_timestamp")
                if isinstance(rt, str) and rt:
                    self.run_timestamp = rt
                    self.generation_output_dir = os.path.join(
                        self.output_dir,
                        f"generations_{self.run_timestamp}",
                    )
                    os.makedirs(self.generation_output_dir, exist_ok=True)
                # History and best records
                hist = payload.get("history", [])
                if isinstance(hist, list):
                    self.history = [GenerationSummary(**h) for h in hist]  # type: ignore[arg-type]
                best_list = payload.get("best_records", [])
                if isinstance(best_list, list):
                    # Rehydrate as mapping by expression
                    self.best_records = {}
                    self._best_signature_records = {}
                    for entry in best_list:
                        try:
                            rec = IndividualRecord(**entry)  # type: ignore[arg-type]
                            if rec.signature is not None:
                                sig_tuple = tuple(rec.signature)
                                rec.signature = sig_tuple
                                self._best_signature_records[sig_tuple] = (
                                    rec,
                                    self._record_complexity(rec),
                                )
                            self.best_records[rec.expression] = rec
                        except Exception:
                            continue
                mg = payload.get("metric_goals")
                if isinstance(mg, dict):
                    self.metric_goals.update({str(k): bool(v) for k, v in mg.items()})
                # RNG
                rng_payload = payload.get("rng")
                if isinstance(rng_payload, dict):
                    self._ckpt.restore_rng(rng_payload, self._subset_rng)
                logger.info("Resuming GP pipeline from checkpoint '%s' (generation %d)", latest.name, start_gen)

        for gen in range(start_gen, pop_cfg.generations):
            eval_count = self._evaluate_population(population, generation=gen)
            eligible = [ind for ind in population if self._passes_thresholds(ind)]
            if eligible:
                hall_of_fame.update(eligible)
            logbook.record(gen=gen, evals=eval_count)
            self._log_generation(population, gen)

            if gen == pop_cfg.generations - 1:
                break

            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            for idx in range(1, len(offspring), 2):
                if random.random() < pop_cfg.crossover_prob:
                    first = offspring[idx - 1]
                    second = offspring[idx]
                    new_first, new_second = self.toolbox.mate(first, second)
                    # Replace references so static limits from decorators take effect.
                    offspring[idx - 1], offspring[idx] = new_first, new_second
                    # Invalidate cached representations/signatures after structural changes
                    self._invalidate_individual_cache(offspring[idx - 1])
                    self._invalidate_individual_cache(offspring[idx])
                    del offspring[idx - 1].fitness.values
                    del offspring[idx].fitness.values

            for idx, mutant in enumerate(offspring):
                if random.random() < pop_cfg.mutation_prob:
                    (new_mutant,) = self.toolbox.mutate(mutant)
                    offspring[idx] = new_mutant
                    # Invalidate cached representations/signatures after structural changes
                    self._invalidate_individual_cache(offspring[idx])
                    del offspring[idx].fitness.values

            population[:] = offspring

            # Save checkpoint so we can resume at the start of the next generation
            try:
                self._ckpt.save(
                    generation_next=gen + 1,
                    run_timestamp=self.run_timestamp,
                    population=population,
                    feature_columns=self.feature_columns,
                    subset_rng=self._subset_rng,
                    best_records=[asdict(rec) for rec in self.best_records.values()],
                    history=[asdict(summary) for summary in self.history],
                    metric_goals=self.metric_goals,
                )
            except Exception:  # pragma: no cover - defensive
                # Non-fatal
                pass

        artifacts = self._save_artifacts(hall_of_fame, population)
        best_list = sorted(
            self.best_records.values(),
            key=lambda rec: self._aggregate_for_ranking(rec.metrics),
            reverse=True,
        )
        return {
            "hall_of_fame": [self._individual_to_record(ind) for ind in hall_of_fame],
            "logbook": logbook,
            "artifacts": artifacts,
            "history": self.history,
            "eligible_best": best_list,
        }

    def _invalidate_individual_cache(self, individual: gp.PrimitiveTree) -> None:
        """Clear cached expression/signature state after genetic edits.

        Cloning and genetic operators copy attributes like `_cached_expression` and
        previously computed signatures. If not cleared after structural changes,
        saved generation files can show repeated or mismatched expressions.
        """
        for attr in ("_cached_expression", "subset_signature", "full_signature", "is_duplicate"):
            if hasattr(individual, attr):
                try:
                    delattr(individual, attr)
                except Exception:
                    pass

    def _evaluate_population(self, population: Sequence[gp.PrimitiveTree], generation: int) -> int:
        evaluations = 0
        provisional: List[
            Tuple[gp.PrimitiveTree, Dict[str, float], Dict[str, str], IRNode, np.ndarray]
        ] = []
        progress = tqdm(
            population,
            total=len(population),
            desc=f"Gen {generation + 1} validations",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for individual in progress:
            if individual.fitness.valid:
                continue
            ir = tree_to_ir(individual, self.feature_columns)
            data_frame = self.df_subset if self.df_subset is not None else self.df_full
            cache = self.cache_subset if self.df_subset is not None else self.cache_full
            values = self.backend_fn(ir, data_frame, cache)
            using_subset = self.df_subset is not None and data_frame is self.df_subset
            signature = self._vector_signature(values)
            registry = self._subset_signatures if using_subset else self._full_signatures
            should_eval, replaced = self._register_signature(registry, signature, individual)
            if replaced is not None and replaced is not individual:
                self._mark_duplicate(replaced)
            if not should_eval:
                if using_subset:
                    individual.subset_signature = signature
                else:
                    individual.full_signature = signature
                self._mark_duplicate(individual)
                continue
            if using_subset:
                individual.subset_signature = signature
            else:
                individual.full_signature = signature
            individual.is_duplicate = False  # type: ignore[attr-defined]
            validation = self._run_validator(data_frame, values)
            metrics = validation.fitness_objectives
            goals = validation.fitness_goals
            self._assign_fitness(individual, metrics, goals)
            self._register_validator_payload(individual, validation)
            self._maybe_register_best(individual)
            if using_subset:
                provisional.append((individual, metrics, goals, ir, values))
            evaluations += 1

        progress.close()

        if self.df_subset is not None and self.full_eval_top_k > 0 and provisional:
            filtered = [
                entry for entry in provisional if not getattr(entry[0], "is_duplicate", False)
            ]
            if not filtered:
                return evaluations
            k = max(1, int(len(filtered) * self.full_eval_top_k))
            entry_map: Dict[int, Tuple[gp.PrimitiveTree, Any, Any, Any, Any]] = {}
            individuals: List[gp.PrimitiveTree] = []
            for entry in filtered:
                individual = entry[0]
                key = id(individual)
                if key in entry_map:
                    continue
                entry_map[key] = entry
                individuals.append(individual)
            fronts = sortNondominated(individuals, len(individuals), first_front_only=False)

            selected_inds: List[gp.PrimitiveTree] = []
            for front in fronts:
                assignCrowdingDist(front)
                front_sorted = sorted(
                    front,
                    key=lambda ind: getattr(ind.fitness, "crowding_dist", float("inf")),
                    reverse=True,
                )
                for ind in front_sorted:
                    if len(selected_inds) >= k:
                        break
                    if ind not in selected_inds:
                        selected_inds.append(ind)
                if len(selected_inds) >= k:
                    break

            if len(selected_inds) < k:
                # Fallback: fill up with remaining individuals preserving iteration order.
                for ind in individuals:
                    if len(selected_inds) >= k:
                        break
                    if ind not in selected_inds:
                        selected_inds.append(ind)

            top = []
            for ind in selected_inds[:k]:
                mapped = entry_map.get(id(ind))
                if mapped is not None:
                    top.append(mapped)
            for individual, _, _, ir, _ in top:
                if getattr(individual, "is_duplicate", False):
                    continue
                values_full = self.backend_fn(ir, self.df_full, self.cache_full)
                signature_full = self._vector_signature(values_full)
                should_eval_full, replaced_full = self._register_signature(
                    self._full_signatures, signature_full, individual
                )
                if replaced_full is not None and replaced_full is not individual:
                    self._mark_duplicate(replaced_full)
                if not should_eval_full:
                    individual.full_signature = signature_full
                    self._mark_duplicate(individual)
                    continue
                individual.full_signature = signature_full
                individual.is_duplicate = False  # type: ignore[attr-defined]
                validation_full = self._run_validator(self.df_full, values_full)
                metrics_full = validation_full.fitness_objectives
                goals_full = validation_full.fitness_goals
                self._assign_fitness(individual, metrics_full, goals_full)
                self._register_validator_payload(individual, validation_full)
                self._maybe_register_best(individual)
                evaluations += 1
        return evaluations

    def _assign_fitness(
        self,
        individual: gp.PrimitiveTree,
        objectives: Dict[str, float],
        goals: Dict[str, str],
    ) -> None:
        objectives = dict(objectives)
        goals = dict(goals)
        selected: Dict[str, float] = {}
        values: List[float] = []
        for idx, name in enumerate(self.mode_metrics):
            maximize = self._determine_goal(name, goals)
            if name not in goals:
                goals[name] = "maximize" if maximize else "minimize"
            val = objectives.get(name)
            if val is None:
                val = float("-inf") if maximize else float("inf")
                objectives[name] = val
            selected[name] = val
            if not maximize:
                val = -val
            weight = float(self.mode_weights[idx])
            values.append(val * weight)
        individual.fitness.values = tuple(values)
        individual.metrics = selected  # type: ignore[attr-defined]
        individual.validator_metrics = objectives  # type: ignore[attr-defined]
        individual.validator_goals = goals  # type: ignore[attr-defined]
        individual.feature_mode = self.feature_mode  # type: ignore[attr-defined]

    def _determine_goal(self, name: str, goals: Dict[str, str]) -> bool:
        if name in self.metric_goals:
            return self.metric_goals[name]
        if name in goals:
            maximize = goals[name].lower() != "minimize"
        else:
            maximize = self.metric_default_goals.get(name, True)
        self.metric_goals[name] = maximize
        return maximize

    def _run_validator(self, df: pd.DataFrame, values: np.ndarray):
        feature_name = f"__gp_feature_{next(self._feature_counter)}__"
        df_temp = df.copy()
        if self.feature_mode == "numeric":
            cleaned_values = np.asarray(values, dtype=np.float64)
            if not np.isfinite(cleaned_values).all():
                finite_mask = np.isfinite(cleaned_values)
                if finite_mask.any():
                    cleaned_values = cleaned_values.copy()
                    cleaned_values[~finite_mask] = 0.0
                else:
                    cleaned_values = np.zeros_like(cleaned_values, dtype=np.float64)
            cleaned_values = self._clip_to_float32_bounds(cleaned_values)
            df_temp[feature_name] = cleaned_values.astype(np.float32, copy=False)
            result = run_numeric_validator(
                df_temp,
                feature_name,
                target=self.target_column,
                kwargs=self.cfg.validator.numeric_kwargs,
            )
        else:
            bool_values = (values > 0).astype(np.int8)
            df_temp[feature_name] = bool_values
            result = run_boolean_validator(
                df_temp,
                feature_name,
                target=self.target_column,
                kwargs=self.cfg.validator.boolean_kwargs,
            )
        return result

    def _register_validator_payload(self, individual: gp.PrimitiveTree, validation) -> None:
        individual.validator_result = validation  # type: ignore[attr-defined]

    def _aggregate_for_ranking(self, metrics: Dict[str, float]) -> float:
        score = 0.0
        for idx, name in enumerate(self.mode_metrics):
            if name not in metrics:
                continue
            val = metrics[name]
            maximize = self.metric_goals.get(name, self.metric_default_goals.get(name, True))
            if not maximize:
                val = -val
            weight = float(self.mode_weights[idx])
            score += val * weight
        return score

    def _passes_thresholds(self, individual: gp.PrimitiveTree) -> bool:
        metrics = getattr(individual, "metrics", {})
        if not metrics:
            return False
        thresholds = self.mode_thresholds
        for name in self.mode_metrics:
            if name not in thresholds:
                continue
            value = metrics.get(name)
            if value is None:
                return False
            maximize = self.metric_goals.get(name, self.metric_default_goals.get(name, True))
            threshold = thresholds[name]
            if maximize:
                if value < threshold:
                    return False
            else:
                if value > threshold:
                    return False

        # For boolean features, enforce exposure bounds as a hard constraint.
        if self.feature_mode == "boolean":
            constraints = getattr(getattr(individual, "validator_result", None), "constraints", {})
            try:
                ok = bool(constraints.get("exposure_rate_within_bounds", True))
            except Exception:
                ok = True
            if not ok:
                return False
        return True

    def _maybe_register_best(self, individual: gp.PrimitiveTree) -> None:
        metrics = getattr(individual, "metrics", {})
        if not self._passes_thresholds(individual):
            return
        if getattr(individual, "is_duplicate", False):
            return
        signature = getattr(individual, "full_signature", None)
        if signature is None:
            signature = getattr(individual, "subset_signature", None)
        if signature is None:
            return
        record = self._individual_to_record(individual)
        signature = tuple(signature)
        record.signature = signature
        complexity = self._individual_complexity(individual)
        existing_entry = self._best_signature_records.get(signature)
        if existing_entry is not None:
            existing_record, existing_complexity = existing_entry
            if complexity < existing_complexity:
                self.best_records.pop(existing_record.expression, None)
            elif complexity == existing_complexity:
                existing_score = self._aggregate_for_ranking(existing_record.metrics)
                new_score = self._aggregate_for_ranking(record.metrics)
                if new_score < existing_score:
                    return
                self.best_records.pop(existing_record.expression, None)
            else:
                return
        self._best_signature_records[signature] = (record, complexity)
        self.best_records[record.expression] = record

    def _log_generation(self, population: Sequence[gp.PrimitiveTree], generation: int) -> None:
        metric_names = list(self.mode_metrics)
        metrics_matrix = np.array([
            [getattr(ind, "metrics", {}).get(name, float("nan")) for name in metric_names]
            for ind in population
        ], dtype=float)
        scores = [self._aggregate_for_ranking(getattr(ind, "metrics", {})) for ind in population]
        best_idx = int(np.argmax(scores)) if scores else 0
        best_individual = population[best_idx]

        best_metrics: Dict[str, Optional[float]] = {}
        mean_metrics: Dict[str, Optional[float]] = {}
        for idx, name in enumerate(metric_names):
            col = metrics_matrix[:, idx]
            col_finite = col[np.isfinite(col)]
            best_val = metrics_matrix[best_idx, idx]
            best_metrics[name] = float(best_val) if np.isfinite(best_val) else None
            mean_metrics[name] = (
                float(col_finite.mean()) if col_finite.size else None
            )

        summary = GenerationSummary(
            generation=generation,
            best_metrics=best_metrics,
            mean_metrics=mean_metrics,
            best_expression=self._expression_from_individual(best_individual),
        )
        self.history.append(summary)
        self._persist_generation(population, summary, best_individual)

    def _persist_generation(
        self,
        population: Sequence[gp.PrimitiveTree],
        summary: GenerationSummary,
        best_individual: gp.PrimitiveTree,
    ) -> None:
        filepath = os.path.join(
            self.generation_output_dir,
            f"generation_{summary.generation:04d}.json",
        )
        try:
            best_record = asdict(self._individual_to_record(best_individual))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to serialize best individual for generation %s: %s",
                summary.generation,
                exc,
            )
            best_record = {
                "index": id(best_individual),
                "expression": None,
                "feature_type": getattr(best_individual, "feature_mode", self.feature_mode),
                "metrics": self._sanitize_mapping_for_json(dict(getattr(best_individual, "metrics", {}))),
                "validator_objectives": self._sanitize_mapping_for_json(
                    dict(getattr(best_individual, "validator_metrics", {}))
                ),
                "validator_goals": getattr(best_individual, "validator_goals", {}),
                "error": str(exc),
            }

        population_records: List[Dict[str, object]] = []
        for individual in population:
            try:
                population_records.append(asdict(self._individual_to_record(individual)))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to serialize individual %s in generation %s: %s",
                    id(individual),
                    summary.generation,
                    exc,
                )
                population_records.append(
                    {
                        "index": id(individual),
                        "expression": None,
                        "feature_type": getattr(individual, "feature_mode", self.feature_mode),
                        "metrics": self._sanitize_mapping_for_json(dict(getattr(individual, "metrics", {}))),
                        "validator_objectives": self._sanitize_mapping_for_json(
                            dict(getattr(individual, "validator_metrics", {}))
                        ),
                        "validator_goals": getattr(individual, "validator_goals", {}),
                        "error": str(exc),
                    }
                )

        payload = {
            "summary": asdict(summary),
            "best": best_record,
            "population": population_records,
            "written_at": datetime.utcnow().isoformat() + "Z",
        }

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _expression_from_individual(self, individual: gp.PrimitiveTree) -> str:
        ir = tree_to_ir(individual, self.feature_columns)
        expr = self._expression_from_ir(ir)
        return expr

    def _expression_from_ir(self, node: IRNode) -> str:
        if node.kind == "var":
            return node.value
        if node.kind == "const":
            return f"{node.value:.4f}"
        args = [self._expression_from_ir(child) for child in node.args]
        if node.value in {"add", "sub", "mul", "div"}:
            symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[node.value]
            return f"({args[0]} {symbol} {args[1]})"
        if node.value in {"min", "max"}:
            return f"{node.value}({', '.join(args)})"
        if node.value == "abs":
            return f"abs({args[0]})"
        if node.value == "log1p":
            return f"log1p({args[0]})"
        if node.value == "clip":
            return f"clip({args[0]}, {args[1]}, {args[2]})"
        return f"{node.value}({', '.join(args)})"

    def _individual_to_record(self, individual: gp.PrimitiveTree) -> IndividualRecord:
        expr = getattr(individual, "_cached_expression", None)
        if expr is None:
            expr = self._expression_from_individual(individual)
            setattr(individual, "_cached_expression", expr)
        metrics_raw = getattr(individual, "metrics", {})
        validator_objectives_raw = getattr(individual, "validator_metrics", {})
        metrics = self._sanitize_mapping_for_json(dict(metrics_raw))
        validator_objectives = self._sanitize_mapping_for_json(dict(validator_objectives_raw))
        validator_goals = getattr(individual, "validator_goals", {})
        feature_type = getattr(individual, "feature_mode", self.feature_mode)
        signature = getattr(individual, "full_signature", None)
        if signature is None:
            signature = getattr(individual, "subset_signature", None)
        node_count, tree_height, expr_len = self._individual_complexity(individual)
        binary_expr = None
        if feature_type == "boolean":
            binary_expr = f"({expr}) > 0"
        return IndividualRecord(
            index=id(individual),
            expression=expr,
            binary_expression=binary_expr,
            feature_type=feature_type,
            metrics=metrics,
            validator_objectives=validator_objectives,
            validator_goals=validator_goals,
            signature=signature,
            node_count=node_count,
            tree_height=tree_height,
            expression_length=expr_len,
        )

    @staticmethod
    def _record_complexity(record: IndividualRecord) -> Tuple[int, int, int]:
        node_count = record.node_count or 10**9
        tree_height = record.tree_height or 10**9
        expr_len = record.expression_length or len(record.expression)
        return (node_count, tree_height, expr_len)

    def _individual_complexity(self, individual: gp.PrimitiveTree) -> Tuple[int, int, int]:
        node_count = len(individual)
        tree_height = int(getattr(individual, "height", 0))
        expr = getattr(individual, "_cached_expression", None)
        if expr is None:
            expr = self._expression_from_individual(individual)
            setattr(individual, "_cached_expression", expr)
        expression_length = len(expr)
        return (node_count, tree_height, expression_length)

    def _vector_signature(self, values: np.ndarray) -> Tuple[int, int]:
        if self.feature_mode == "numeric":
            arr = np.asarray(values, dtype=np.float64)
            finite_mask = np.isfinite(arr)
            nonfinite_codes = np.zeros(arr.shape, dtype=np.int8)
            if not finite_mask.all():
                posinf = np.isposinf(arr)
                neginf = np.isneginf(arr)
                nanmask = np.isnan(arr)
                if posinf.any():
                    nonfinite_codes[posinf] = 1
                if neginf.any():
                    nonfinite_codes[neginf] = 2
                if nanmask.any():
                    nonfinite_codes[nanmask] = 3
                arr = arr.copy()
                arr[~finite_mask] = 0.0
            arr = self._clip_to_float32_bounds(arr).astype(np.float32, copy=False)
            hash_input = pd.DataFrame(
                {"value": arr, "nonfinite": nonfinite_codes},
                copy=False,
            )
        else:
            arr = (np.asarray(values, dtype=np.float64) > 0).astype(np.int8, copy=False)
            hash_input = pd.DataFrame({"value": arr}, copy=False)
        hashed = pd.util.hash_pandas_object(hash_input, index=False)
        if hashed.empty:
            return (0, 0)
        hash_values = hashed.to_numpy(dtype=np.uint64, copy=False)
        signature_sum = int(hash_values.sum(dtype=np.uint64))
        signature_xor = int(np.bitwise_xor.reduce(hash_values))
        return (signature_sum, signature_xor)

    def _register_signature(
        self,
        registry: Dict[Tuple[int, int], SignatureEntry],
        signature: Tuple[int, int],
        individual: gp.PrimitiveTree,
    ) -> Tuple[bool, Optional[gp.PrimitiveTree]]:
        complexity = self._individual_complexity(individual)
        entry = registry.get(signature)
        if entry is None or entry.individual is individual:
            registry[signature] = SignatureEntry(signature, individual, complexity)
            return True, None
        if complexity < entry.complexity:
            previous = entry.individual
            registry[signature] = SignatureEntry(signature, individual, complexity)
            return True, previous
        # fall back to keep existing representative
        return False, None

    def _remove_best_record(self, individual: gp.PrimitiveTree) -> None:
        signature = getattr(individual, "full_signature", None)
        if signature is None:
            signature = getattr(individual, "subset_signature", None)
        if signature is None:
            return
        entry = self._best_signature_records.get(signature)
        if entry is None:
            return
        record, _ = entry
        if record.index != id(individual):
            return
        self._best_signature_records.pop(signature, None)
        self.best_records.pop(record.expression, None)

    def _mark_duplicate(self, individual: gp.PrimitiveTree) -> None:
        self._remove_best_record(individual)
        self._assign_fitness(individual, {}, {})
        individual.is_duplicate = True  # type: ignore[attr-defined]

    def _ensure_output_dir(self) -> PathLike:
        # Prefer per-mode directories when configured; otherwise, fall back to the default.
        if self.feature_mode == "numeric" and getattr(self.cfg.output, "artifacts_dir_numeric", None):
            path = self.cfg.output.artifacts_dir_numeric  # type: ignore[assignment]
        elif self.feature_mode == "boolean" and getattr(self.cfg.output, "artifacts_dir_boolean", None):
            path = self.cfg.output.artifacts_dir_boolean  # type: ignore[assignment]
        else:
            path = self.cfg.output.artifacts_dir
        os.makedirs(path, exist_ok=True)
        return path

    def _save_artifacts(self, hall_of_fame: tools.HallOfFame, population: Sequence[gp.PrimitiveTree]):
        output_dir = self.output_dir
        timestamp = self.run_timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        hof_entries = [asdict(self._individual_to_record(ind)) for ind in hall_of_fame]
        hof_path = os.path.join(output_dir, f"hof_{timestamp}.json")
        with open(hof_path, "w", encoding="utf-8") as fh:
            json.dump(hof_entries, fh, indent=2)
        history_path = os.path.join(output_dir, f"history_{timestamp}.json")
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(summary) for summary in self.history], fh, indent=2)
        population_path = None
        qualifying_population = [
            self._individual_to_record(ind)
            for ind in population
            if self._passes_thresholds(ind)
        ]
        if self.cfg.output.save_population and qualifying_population:
            population_path = os.path.join(output_dir, f"population_{timestamp}.json")
            with open(population_path, "w", encoding="utf-8") as fh:
                json.dump([asdict(rec) for rec in qualifying_population], fh, indent=2)

        best_sorted = sorted(
            self.best_records.values(),
            key=lambda rec: self._aggregate_for_ranking(rec.metrics),
            reverse=True,
        )
        best_path = os.path.join(output_dir, f"best_{timestamp}.json")
        with open(best_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(rec) for rec in best_sorted], fh, indent=2)
        return {
            "hall_of_fame": hof_path,
            "history": history_path,
            "population": population_path,
            "best": best_path,
            "generations": self.generation_output_dir,
        }

    @staticmethod
    def _clip_to_float32_bounds(values: np.ndarray) -> np.ndarray:
        """Clamp values to the finite range representable by float32 to avoid overflow casts."""

        float32_max = np.finfo(np.float32).max
        overflow_mask = np.abs(values) > float32_max
        if not overflow_mask.any():
            return values
        clipped = values.copy()
        clipped[overflow_mask] = np.sign(clipped[overflow_mask]) * float32_max
        return clipped
