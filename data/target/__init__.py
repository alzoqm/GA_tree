# /data/target/__init__.py

"""Target generation registry for supervised selection modes."""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Tuple

import pandas as pd

TargetBuilder = Callable[[pd.DataFrame, Mapping[str, object]], Tuple[str, pd.Series]]

_REGISTRY: Dict[str, TargetBuilder] = {}


def register(name: str) -> Callable[[TargetBuilder], TargetBuilder]:
    """Decorator to register a target builder under ``name``.

    Parameters
    ----------
    name:
        Unique identifier for the target generation mode.
    """

    def _decorator(func: TargetBuilder) -> TargetBuilder:
        if name in _REGISTRY:
            raise ValueError(f"Target mode '{name}' is already registered")
        _REGISTRY[name] = func
        return func

    return _decorator


def available_modes() -> Tuple[str, ...]:
    """Return the currently registered target mode identifiers."""

    return tuple(sorted(_REGISTRY))


def compute_target(
    frame: pd.DataFrame,
    *,
    mode: str,
    params: Mapping[str, object] | None = None,
) -> Tuple[str, pd.Series]:
    """Generate a supervised target series according to ``mode``.

    Parameters
    ----------
    frame:
        Feature dataframe aligned by index.
    mode:
        Registered target mode identifier.
    params:
        Optional mapping of mode-specific configuration values.

    Returns
    -------
    tuple[str, pandas.Series]
        A tuple containing ``(column_name, target_series)``. The target series is
        reindexed to match ``frame`` if needed.
    """

    if mode not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown target mode '{mode}'. Available: {available}")

    builder = _REGISTRY[mode]
    column_name, series = builder(frame, params or {})
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"Target builder for mode '{mode}' must return a pandas Series, "
            f"got {type(series)!r}"
        )
    if not series.index.equals(frame.index):
        series = series.reindex(frame.index)
    return column_name, series


# Register built-in modes
try:  # noqa: SIM105 - defensive import guard
    from . import multi_horizon_risk_adjusted  # noqa: F401
except Exception:  # pragma: no cover - optional dependency handling
    pass

