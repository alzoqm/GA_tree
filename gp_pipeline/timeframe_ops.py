# /gp_pipeline/timeframe_ops.py
"""Utilities for registering and executing multi-timeframe primitives."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .config import TimeframePrimitiveConfig

SAFE_EPS = 1e-6


@dataclass(frozen=True)
class TimeframeOperatorSpec:
    name: str
    operator: str
    allow_partial: bool
    lag_fill_strategy: str
    arity: int
    window: int | None = None
    timeframe_label: str | None = None
    timeframe_minutes: float | None = None
    min_periods: int | None = None
    dynamic: bool = False
    min_window: int = 1
    max_window: int = 720
    param_kind: str = "bars"
    base_interval_minutes: float = 1.0


class TimeframeOperatorError(RuntimeError):
    """Raised when timeframe primitive configuration is invalid."""


_TIMEFRAME_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)([smhdw]?)$")
_UNIT_TO_MINUTES = {
    "": 1.0,
    "s": 1.0 / 60.0,
    "m": 1.0,
    "h": 60.0,
    "d": 1440.0,
    "w": 10080.0,
}

TIMEFRAME_OPERATORS: Dict[str, TimeframeOperatorSpec] = {}


def register_timeframe_operators(specs: Mapping[str, TimeframeOperatorSpec]) -> None:
    """Register the active timeframe operators for downstream evaluation."""
    TIMEFRAME_OPERATORS.clear()
    TIMEFRAME_OPERATORS.update(specs)


def is_timeframe_operator(name: str) -> bool:
    return name in TIMEFRAME_OPERATORS


def _infer_base_interval_minutes(df: pd.DataFrame, timestamp_column: str | None) -> float | None:
    candidates: Iterable[str] = ()
    if timestamp_column and timestamp_column in df.columns:
        candidates = (timestamp_column,)
    else:
        candidates = tuple(
            col
            for col in df.columns
            if df[col].dtype.kind == "M" or "time" in col.lower()
        )
    for column in candidates:
        series = pd.to_datetime(df[column], errors="coerce")
        if series.isna().all():
            continue
        ordered = series.dropna().sort_values()
        diffs = ordered.diff().dropna()
        if diffs.empty:
            continue
        median_delta = diffs.median()
        minutes = median_delta.total_seconds() / 60.0
        if minutes > 0:
            return minutes
    return None


def _minutes_to_window(minutes: float, base_interval: float) -> int:
    ratio = minutes / base_interval
    window = int(round(ratio))
    if window < 1:
        window = 1
    # Dynamic parameters can come from floating point math, so tolerate small
    # rounding mismatches (e.g. 65614 minutes on a 5 minute base interval).
    tolerance = max(base_interval * 0.5, 1e-6)
    if abs(minutes - window * base_interval) > tolerance:
        raise TimeframeOperatorError(
            f"Timeframe of {minutes} minutes is incompatible with base interval {base_interval}"
        )
    return window


def _parse_timeframe_value(
    value: Union[str, int],
    base_interval: float,
    integers_as_rows: bool,
) -> Tuple[str, int, float]:
    if isinstance(value, int):
        if integers_as_rows:
            window = max(1, int(value))
            minutes = window * base_interval
            label = f"{window}b"
            return label, window, minutes
        minutes = float(value)
        window = _minutes_to_window(minutes, base_interval)
        label = f"{int(minutes)}m"
        return label, window, minutes

    raw = value.strip().lower()
    if raw.startswith("bars:"):
        count = int(raw.split(":", 1)[1])
        window = max(1, count)
        minutes = window * base_interval
        label = f"{window}b"
        return label, window, minutes
    match = _TIMEFRAME_PATTERN.match(raw)
    if not match:
        raise TimeframeOperatorError(f"Unsupported timeframe token: {value}")
    magnitude = float(match.group(1))
    unit = match.group(2)
    if unit not in _UNIT_TO_MINUTES:
        raise TimeframeOperatorError(f"Unsupported timeframe unit in token: {value}")
    minutes = magnitude * _UNIT_TO_MINUTES[unit]
    window = _minutes_to_window(minutes, base_interval)
    label = value.strip()
    return label, window, minutes


SUPPORTED_OPERATORS = {
    "high",
    "low",
    "max",
    "min",
    "mean",
    "median",
    "std",
    "var",
    "sum",
    "range",
    "ema",
    "lag",
    "diff",
    "pct_change",
    "zscore",
    "skew",
    "kurtosis",
}


def build_timeframe_operator_specs(
    cfg: TimeframePrimitiveConfig,
    df: pd.DataFrame,
) -> Dict[str, TimeframeOperatorSpec]:
    use_static = cfg.enabled and (
        bool(cfg.timeframes) or bool(cfg.custom_timeframes) or bool(cfg.extra_operators)
    )
    use_dynamic = bool(cfg.dynamic_operators)
    if not use_static and not use_dynamic:
        return {}

    needs_interval = use_static or any(
        str(entry.get("param_kind", "bars")).lower() != "bars" for entry in cfg.dynamic_operators
    )
    base_interval = cfg.base_interval_minutes
    if needs_interval and (base_interval is None or base_interval <= 0):
        base_interval = _infer_base_interval_minutes(df, cfg.timestamp_column)
    if needs_interval and (base_interval is None or base_interval <= 0):
        raise TimeframeOperatorError(
            "Unable to determine base interval minutes for timeframe primitives. "
            "Specify timeframe_primitives.base_interval_minutes or provide a valid timestamp column."
        )
    if base_interval is None or base_interval <= 0:
        base_interval = 1.0

    specs: Dict[str, TimeframeOperatorSpec] = {}

    if use_static:
        operators = list(dict.fromkeys(tuple(cfg.operators) + tuple(cfg.extra_operators)))
        unsupported = [op for op in operators if op not in SUPPORTED_OPERATORS]
        if unsupported:
            raise TimeframeOperatorError(f"Unsupported timeframe operators requested: {unsupported}")

        min_periods_default = cfg.min_periods
        timeframe_values: Sequence[Union[str, int]] = tuple(cfg.timeframes) + tuple(cfg.custom_timeframes)
        seen: set[str] = set()
        for token in timeframe_values:
            label, window, minutes = _parse_timeframe_value(token, base_interval, cfg.integers_as_rows)
            if label in seen:
                continue
            seen.add(label)
            min_periods = (
                max(1, min(window, min_periods_default)) if min_periods_default else window
            )
            if cfg.allow_partial_windows and min_periods_default is None:
                min_periods = 1
            for operator in operators:
                name = f"{label}_{operator}"
                if name in specs:
                    raise TimeframeOperatorError(
                        f"Duplicate timeframe operator name detected: {name}"
                    )
                spec = TimeframeOperatorSpec(
                    name=name,
                    operator=operator,
                    allow_partial=cfg.allow_partial_windows,
                    lag_fill_strategy=cfg.lag_fill_strategy.lower(),
                    arity=1,
                    window=window,
                    timeframe_label=label,
                    timeframe_minutes=minutes,
                    min_periods=min_periods,
                    dynamic=False,
                    base_interval_minutes=base_interval,
                )
                specs[name] = spec

    if use_dynamic:
        for entry in cfg.dynamic_operators:
            if not isinstance(entry, dict):
                raise TimeframeOperatorError(
                    "Each dynamic operator must be a mapping with configuration entries"
                )
            name = str(entry.get("name"))
            operator = str(entry.get("operator", "mean")).lower()
            if not name:
                raise TimeframeOperatorError("Dynamic operator is missing a 'name' field")
            if operator not in SUPPORTED_OPERATORS:
                raise TimeframeOperatorError(
                    f"Unsupported dynamic timeframe operator '{operator}' for '{name}'"
                )
            if name in specs:
                raise TimeframeOperatorError(
                    f"Dynamic operator name '{name}' conflicts with existing static operator"
                )
            param_kind = str(entry.get("param_kind", "bars")).lower()
            if param_kind not in {"bars", "minutes"}:
                raise TimeframeOperatorError(
                    f"Dynamic operator '{name}' has unsupported param_kind '{param_kind}'"
                )
            min_window = max(1, int(entry.get("min_window", 1)))
            max_window = int(entry.get("max_window", 720))
            if max_window < min_window:
                raise TimeframeOperatorError(
                    f"Dynamic operator '{name}' requires max_window >= min_window"
                )
            min_periods = entry.get("min_periods")
            min_periods = int(min_periods) if min_periods is not None else None
            allow_partial = bool(entry.get("allow_partial", cfg.allow_partial_windows))
            lag_strategy = str(entry.get("lag_fill_strategy", cfg.lag_fill_strategy)).lower()
            spec = TimeframeOperatorSpec(
                name=name,
                operator=operator,
                allow_partial=allow_partial,
                lag_fill_strategy=lag_strategy,
                arity=2,
                window=None,
                timeframe_label=None,
                timeframe_minutes=None,
                min_periods=min_periods,
                dynamic=True,
                min_window=min_window,
                max_window=max_window,
                param_kind=param_kind,
                base_interval_minutes=base_interval,
            )
            specs[name] = spec

    return specs


def _rolling(series: pd.Series, window: int, min_periods: int, func: str) -> pd.Series:
    if func == "max":
        return series.rolling(window=window, min_periods=min_periods).max()
    if func == "min":
        return series.rolling(window=window, min_periods=min_periods).min()
    if func == "mean":
        return series.rolling(window=window, min_periods=min_periods).mean()
    if func == "median":
        return series.rolling(window=window, min_periods=min_periods).median()
    if func == "std":
        return series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    if func == "var":
        return series.rolling(window=window, min_periods=min_periods).var(ddof=0)
    if func == "sum":
        return series.rolling(window=window, min_periods=min_periods).sum()
    if func == "skew":
        return series.rolling(window=window, min_periods=min_periods).skew()
    if func == "kurtosis":
        return series.rolling(window=window, min_periods=min_periods).kurt()
    raise TimeframeOperatorError(f"Unsupported rolling function: {func}")


def _finalize_result(
    operator: str,
    base_series: pd.Series,
    result: pd.Series,
    allow_partial: bool,
    lag_fill_strategy: str,
) -> pd.Series:
    if operator == "lag":
        strategy = lag_fill_strategy
        if strategy == "ffill":
            result = result.ffill()
        elif strategy == "bfill":
            result = result.bfill()
        elif strategy == "zero":
            result = result.fillna(0.0)
        else:  # edge
            if not base_series.dropna().empty:
                edge = base_series.dropna().iloc[0]
            else:
                edge = 0.0
            result = result.fillna(edge)
        return result
    if operator in {"diff", "pct_change", "zscore"}:
        result = result.replace([np.inf, -np.inf], 0.0)
        return result.fillna(0.0)
    if operator in {"skew", "kurtosis"}:
        return result.fillna(0.0)
    if allow_partial:
        # Only use past information to impute partial windows to avoid future leaks.
        forward_filled = result.ffill()
        return forward_filled.fillna(0.0)
    return result.fillna(0.0)


def _scalar_from_array(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return float("nan")
    return float(flat[0])


def _resolve_dynamic_window(spec: TimeframeOperatorSpec, arg: np.ndarray) -> int:
    raw_value = _scalar_from_array(arg)
    if np.isnan(raw_value):
        raw_value = float(spec.min_window)
    if spec.param_kind == "minutes":
        minutes = max(raw_value, spec.min_window * spec.base_interval_minutes)
        window = _minutes_to_window(minutes, spec.base_interval_minutes)
    else:
        window = int(round(raw_value))
    window = max(spec.min_window, window)
    window = min(spec.max_window, window)
    if window < 1:
        window = 1
    return window


def _resolve_min_periods(spec: TimeframeOperatorSpec, window: int) -> int:
    if spec.min_periods is not None:
        return max(1, min(window, spec.min_periods))
    if spec.allow_partial:
        return 1
    return window


def apply_timeframe_operator(name: str, args: Sequence[np.ndarray]) -> np.ndarray:
    spec = TIMEFRAME_OPERATORS.get(name)
    if spec is None:
        raise TimeframeOperatorError(f"Timeframe operator '{name}' is not registered")
    if len(args) != spec.arity:
        raise TimeframeOperatorError(
            f"Operator '{name}' expects {spec.arity} argument(s) but received {len(args)}"
        )
    source = args[0]
    series = pd.Series(np.asarray(source, dtype=np.float64))
    if spec.dynamic:
        window = _resolve_dynamic_window(spec, args[1])
    else:
        if spec.window is None:
            raise TimeframeOperatorError(
                f"Operator '{name}' is missing a configured window size"
            )
        window = spec.window
    min_periods = _resolve_min_periods(spec, window)

    op = spec.operator
    if op in {"high", "max"}:
        result = _rolling(series, window, min_periods, "max")
    elif op in {"low", "min"}:
        result = _rolling(series, window, min_periods, "min")
    elif op == "mean":
        result = _rolling(series, window, min_periods, "mean")
    elif op == "median":
        result = _rolling(series, window, min_periods, "median")
    elif op == "std":
        result = _rolling(series, window, min_periods, "std")
    elif op == "var":
        result = _rolling(series, window, min_periods, "var")
    elif op == "sum":
        result = _rolling(series, window, min_periods, "sum")
    elif op == "range":
        max_series = _rolling(series, window, min_periods, "max")
        min_series = _rolling(series, window, min_periods, "min")
        result = max_series - min_series
    elif op == "ema":
        span = max(1, window)
        result = series.ewm(span=span, adjust=False, min_periods=min_periods).mean()
    elif op == "lag":
        result = series.shift(window)
    elif op == "diff":
        result = series.diff(periods=window)
    elif op == "pct_change":
        result = series.pct_change(periods=window)
    elif op == "zscore":
        mean_series = _rolling(series, window, min_periods, "mean")
        std_series = _rolling(series, window, min_periods, "std")
        std_safe = std_series.mask(std_series.abs() < SAFE_EPS, np.nan)
        result = (series - mean_series) / std_safe
    elif op == "skew":
        result = _rolling(series, window, min_periods, "skew")
    elif op == "kurtosis":
        result = _rolling(series, window, min_periods, "kurtosis")
    else:
        raise TimeframeOperatorError(f"Operator '{op}' does not have an implementation")

    result = _finalize_result(op, series, result, spec.allow_partial, spec.lag_fill_strategy)
    return result.to_numpy(dtype=np.float32)
