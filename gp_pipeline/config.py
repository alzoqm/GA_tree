"""Configuration models for the GP feature pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml


@dataclass(slots=True)
class PrimitiveConfig:
    name: str
    arity: int
    backend: str
    params: Dict[str, float | int] = field(default_factory=dict)


@dataclass(slots=True)
class BackendConfig:
    primary: str = "numexpr"
    fallbacks: Sequence[str] = field(default_factory=lambda: ("numpy",))
    cache_subtrees: bool = True
    low_fidelity_ratio: float = 0.25
    full_eval_top_k: float = 0.1


@dataclass(slots=True)
class PopulationConfig:
    size: int = 64
    generations: int = 10
    min_depth: int = 1
    max_depth: int = 4
    tournament_size: int = 5
    elitism: int = 2
    mutation_prob: float = 0.2
    crossover_prob: float = 0.8
    seed: Optional[int] = 42


@dataclass(slots=True)
class ModeFitnessConfig:
    metrics: List[str]
    weights: List[float]
    maximize: List[bool]
    thresholds: Dict[str, float]


def _default_numeric_fitness() -> ModeFitnessConfig:
    metrics = [
        "neg_log_pvalue",
        "effect_size",
        "cliffs_delta",
        "significance_margin",
        "mean_method_effect",
        "effect_stability",
        "pvalue_consistency",
        "min_pvalue",
    ]
    weights = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0]
    maximize = [True, True, True, True, True, True, True, False]
    thresholds = {
        "neg_log_pvalue": 1.0,
        "effect_size": 0.0,
        "cliffs_delta": 0.0,
        "significance_margin": 0.0,
        "mean_method_effect": 0.0,
        "effect_stability": 0.5,
        "pvalue_consistency": 0.5,
        "min_pvalue": 0.05,
    }
    return ModeFitnessConfig(metrics=metrics, weights=weights, maximize=maximize, thresholds=thresholds)


def _default_boolean_fitness() -> ModeFitnessConfig:
    metrics = [
        "neg_log_pvalue",
        "effect_strength",
        "effect_stability",
        "pvalue_consistency",
        "significance_margin",
        "prop_diff_abs",
        "odds_ratio_abs",
        "cliffs_delta_abs",
        "mutual_information",
        "min_pvalue",
    ]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0]
    maximize = [True, True, True, True, True, True, True, True, True, False]
    thresholds = {
        "neg_log_pvalue": 1.0,
        "effect_strength": 0.0,
        "effect_stability": 0.5,
        "pvalue_consistency": 0.5,
        "significance_margin": 0.0,
        "prop_diff_abs": 0.0,
        "odds_ratio_abs": 1.0,
        "cliffs_delta_abs": 0.0,
        "mutual_information": 0.0,
        "min_pvalue": 0.05,
    }
    return ModeFitnessConfig(metrics=metrics, weights=weights, maximize=maximize, thresholds=thresholds)


@dataclass(slots=True)
class FitnessConfig:
    numeric: ModeFitnessConfig = field(default_factory=_default_numeric_fitness)
    boolean: ModeFitnessConfig = field(default_factory=_default_boolean_fitness)

    def for_mode(self, mode: str) -> ModeFitnessConfig:
        mode_lc = mode.lower()
        if mode_lc == "numeric":
            return self.numeric
        if mode_lc == "boolean":
            return self.boolean
        raise GPConfigError(f"Unknown feature mode '{mode}'")


@dataclass(slots=True)
class ValidatorConfig:
    enabled: bool = False
    numeric_kwargs: Dict[str, object] = field(default_factory=dict)
    boolean_kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class DataConfig:
    path: Path
    target_column: str
    feature_columns: Sequence[str]
    usecols: Optional[Sequence[str]] = None


@dataclass(slots=True)
class OutputConfig:
    artifacts_dir: Path = Path("results/gp_pipeline")
    save_best: int = 5
    save_population: bool = False


@dataclass(slots=True)
class FeatureConfig:
    mode: str = "numeric"


@dataclass(slots=True)
class GPConfig:
    data: DataConfig
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def resolve(self, base_path: Optional[Path] = None) -> "GPConfig":
        base = base_path or Path.cwd()
        self.data.path = (base / self.data.path).resolve()
        self.output.artifacts_dir = (base / self.output.artifacts_dir).resolve()
        return self


class GPConfigError(RuntimeError):
    """Raised when the configuration file is invalid."""


def _ensure_lengths(cfg: GPConfig) -> None:
    for mode_name, mode_cfg in ("numeric", cfg.fitness.numeric), ("boolean", cfg.fitness.boolean):
        metrics = len(mode_cfg.metrics)
        if len(mode_cfg.weights) != metrics:
            raise GPConfigError(
                f"fitness.{mode_name}.weights must match metrics length ({metrics})"
            )
        if len(mode_cfg.maximize) != metrics:
            raise GPConfigError(
                f"fitness.{mode_name}.maximize must match metrics length ({metrics})"
            )
        for metric in mode_cfg.metrics:
            if metric not in mode_cfg.thresholds:
                raise GPConfigError(
                    f"Missing threshold for metric '{metric}' in fitness.{mode_name}.thresholds"
                )


def load_gp_config(path: str | Path) -> GPConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    fitness_raw = data.get("fitness", {}) or {}

    def _merge_mode(raw_cfg: Optional[Dict[str, object]], base: ModeFitnessConfig) -> ModeFitnessConfig:
        raw_cfg = raw_cfg or {}
        metrics = list(raw_cfg.get("metrics", base.metrics))
        weights = [float(w) for w in raw_cfg.get("weights", base.weights)]
        maximize = [bool(x) for x in raw_cfg.get("maximize", base.maximize)]
        thresholds = dict(base.thresholds)
        thresholds.update(raw_cfg.get("thresholds", {}))
        return ModeFitnessConfig(metrics=metrics, weights=weights, maximize=maximize, thresholds={str(k): float(v) for k, v in thresholds.items()})

    numeric_cfg = _merge_mode(fitness_raw.get("numeric"), _default_numeric_fitness())
    boolean_cfg = _merge_mode(fitness_raw.get("boolean"), _default_boolean_fitness())

    try:
        cfg = GPConfig(
            data=DataConfig(
                path=Path(data["data"]["path"]),
                target_column=data["data"]["target_column"],
                feature_columns=tuple(data["data"]["feature_columns"]),
                usecols=tuple(data["data"].get("usecols", [])) or None,
            ),
            feature=FeatureConfig(**data.get("feature", {})),
            population=PopulationConfig(**data.get("population", {})),
            backend=BackendConfig(**data.get("backend", {})),
            fitness=FitnessConfig(numeric=numeric_cfg, boolean=boolean_cfg),
            validator=ValidatorConfig(**data.get("validator", {})),
            output=OutputConfig(**data.get("output", {})),
        )
    except KeyError as err:
        raise GPConfigError(f"Missing configuration key: {err}") from err
    cfg.resolve(Path(path).parent)
    _ensure_lengths(cfg)
    return cfg
