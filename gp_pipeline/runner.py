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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from deap import gp, tools
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


logger = logging.getLogger(__name__)


PathLike = os.PathLike[str] | str


@dataclass(slots=True)
class IndividualRecord:
    index: int
    expression: str
    feature_type: str
    metrics: Dict[str, float]
    validator_objectives: Dict[str, float]
    validator_goals: Dict[str, str]


@dataclass(slots=True)
class GenerationSummary:
    generation: int
    best_metrics: Dict[str, Optional[float]]
    mean_metrics: Dict[str, Optional[float]]
    best_expression: str


class GPFeatureTrainer:
    def __init__(self, cfg: GPConfig) -> None:
        self.cfg = cfg
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
        self.output_dir = os.fspath(self._ensure_output_dir())
        self.run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.generation_output_dir = os.path.join(
            self.output_dir,
            f"generations_{self.run_timestamp}",
        )
        os.makedirs(self.generation_output_dir, exist_ok=True)
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
        population = self.toolbox.population(n=pop_cfg.size)
        hall_of_fame = tools.HallOfFame(self.cfg.output.save_best)
        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"]

        for gen in range(pop_cfg.generations):
            eval_count = self._evaluate_population(population, generation=gen)
            eligible = [
                ind
                for ind in population
                if self._passes_thresholds(getattr(ind, "metrics", {}))
            ]
            if eligible:
                hall_of_fame.update(eligible)
            logbook.record(gen=gen, evals=eval_count)
            self._log_generation(population, gen)

            if gen == pop_cfg.generations - 1:
                break

            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < pop_cfg.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < pop_cfg.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            population[:] = offspring

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
            validation = self._run_validator(data_frame, values)
            metrics = validation.fitness_objectives
            goals = validation.fitness_goals
            self._assign_fitness(individual, metrics, goals)
            self._register_validator_payload(individual, validation)
            self._maybe_register_best(individual)
            provisional.append((individual, metrics, goals, ir, values))
            evaluations += 1

        progress.close()

        if self.df_subset is not None and self.full_eval_top_k > 0 and provisional:
            k = max(1, int(len(provisional) * self.full_eval_top_k))
            ranked = sorted(
                provisional,
                key=lambda x: self._aggregate_for_ranking(x[1]),
                reverse=True,
            )
            top = ranked[:k]
            for individual, _, _, ir, _ in top:
                values_full = self.backend_fn(ir, self.df_full, self.cache_full)
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

    def _passes_thresholds(self, metrics: Dict[str, float]) -> bool:
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
        return True

    def _maybe_register_best(self, individual: gp.PrimitiveTree) -> None:
        metrics = getattr(individual, "metrics", {})
        if not self._passes_thresholds(metrics):
            return
        record = self._individual_to_record(individual)
        existing = self.best_records.get(record.expression)
        if existing is None or self._aggregate_for_ranking(record.metrics) > self._aggregate_for_ranking(existing.metrics):
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
                "metrics": getattr(best_individual, "metrics", {}),
                "validator_objectives": getattr(best_individual, "validator_metrics", {}),
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
                        "metrics": getattr(individual, "metrics", {}),
                        "validator_objectives": getattr(individual, "validator_metrics", {}),
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
        expr = self._expression_from_individual(individual)
        metrics = getattr(individual, "metrics", {})
        validator_objectives = getattr(individual, "validator_metrics", {})
        validator_goals = getattr(individual, "validator_goals", {})
        feature_type = getattr(individual, "feature_mode", self.feature_mode)
        return IndividualRecord(
            index=id(individual),
            expression=expr,
            feature_type=feature_type,
            metrics=metrics,
            validator_objectives=validator_objectives,
            validator_goals=validator_goals,
        )

    def _ensure_output_dir(self) -> PathLike:
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
            if self._passes_thresholds(getattr(ind, "metrics", {}))
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
