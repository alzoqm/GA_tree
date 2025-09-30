# screening.py/target/multi_horizon_risk_adjusted.py
"""Multi-horizon risk-adjusted return target generation."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import register

_DEFAULT_HORIZONS: tuple[int, ...] = (1, 12, 72, 288)
_DEFAULT_OUTPUT = "target_multi_horizon_risk_adjusted"


def _as_float_series(frame: pd.DataFrame, column: str | None) -> pd.Series | None:
    if column is None or column not in frame.columns:
        return None
    series = frame[column]
    if not np.issubdtype(series.dtype, np.number):
        try:
            series = series.astype(float)
        except ValueError:  # pragma: no cover - defensive
            return None
    return series.astype(float)


def _resolve_price_series(
    frame: pd.DataFrame,
    primary: str | None,
    fallback: str | None,
) -> pd.Series | None:
    series = _as_float_series(frame, primary)
    if series is not None:
        return series
    return _as_float_series(frame, fallback)


def _resolve_weights(horizons: Sequence[int], cfg: object) -> np.ndarray:
    horizons_arr = np.asarray(horizons, dtype=float)
    if horizons_arr.size == 0:
        raise ValueError("At least one horizon must be provided for target generation")

    if isinstance(cfg, Mapping):
        mode = str(cfg.get("type", "exponential")).lower()
        if mode == "uniform":
            weights = np.ones_like(horizons_arr)
        elif mode == "exponential":
            lam = float(cfg.get("lambda", 0.01))
            weights = np.exp(-lam * horizons_arr)
        else:
            raise ValueError(f"Unsupported weight type '{mode}'")
    elif isinstance(cfg, Sequence) and not isinstance(cfg, (str, bytes)):
        weights = np.asarray(cfg, dtype=float)
        if weights.size != horizons_arr.size:
            raise ValueError(
                "Explicit weight list must match the number of horizons: "
                f"got {weights.size} weights for {horizons_arr.size} horizons"
            )
    else:
        # Default exponential decay
        weights = np.exp(-0.01 * horizons_arr)

    if not np.isfinite(weights).all():
        raise ValueError("Weights contain non-finite values")
    if np.all(weights <= 0.0):
        weights = np.ones_like(horizons_arr)

    weights = np.clip(weights, a_min=0.0, a_max=None)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value")
    return weights / total


@register("multi_horizon_risk_adjusted")
def build_target(
    frame: pd.DataFrame,
    params: Mapping[str, object],
):
    """Compute a multi-horizon risk-adjusted return target.

    Parameters
    ----------
    frame:
        Input dataframe containing price/volatility columns.
    params:
        Optional mapping to override defaults. Recognised keys:

        ``horizons`` (Sequence[int]):
            Forward horizons (bars) to evaluate. Default ``(1, 12, 72, 288)``.
        ``price_in_column`` (str):
            Column used for entry price (shifted by -1). Defaults to ``markPrice_est``.
        ``price_out_column`` (str):
            Column for exit price (shifted by -H). Defaults to ``price_in_column``.
        ``open_column``/``close_column`` (str):
            Fallback columns when preferred prices are missing. Defaults ``Open``/``Close``.
        ``atr_column`` (str):
            Average true range (percent) column used for volatility scaling. Default ``ATR_7_5m``.
        ``atr_scale`` (float):
            Divides the ATR column to convert percentage-style values to fractions. Default ``100.0``.
        ``volatility_floor`` (float):
            Minimum volatility used to avoid division-by-zero. Default ``1e-4``.
        ``weights`` (mapping or sequence):
            Weight specification. Supports ``{"type": "uniform"}``,
            ``{"type": "exponential", "lambda": 0.01}``, or explicit lists.
        ``winsor_limits`` (Sequence[float]):
            Lower/upper quantiles for winsorisation. Default ``(0.005, 0.995)``.
        ``output_column`` (str):
            Name of the generated column. Default
            ``target_multi_horizon_risk_adjusted``.
    """

    horizons_cfg = params.get("horizons", _DEFAULT_HORIZONS)
    horizons = tuple(int(h) for h in horizons_cfg)
    if any(h <= 0 for h in horizons):
        raise ValueError("Horizon values must be positive integers")

    price_in_col = params.get("price_in_column", "markPrice_est")
    price_out_col = params.get("price_out_column") or price_in_col
    open_col = params.get("open_column", "Open")
    close_col = params.get("close_column", "Close")
    atr_col = params.get("atr_column", "ATR_7_5m")
    atr_scale = float(params.get("atr_scale", 100.0))
    vol_floor = float(params.get("volatility_floor", 1e-4))
    winsor_limits = params.get("winsor_limits", (0.005, 0.995))
    output_column = str(params.get("output_column", _DEFAULT_OUTPUT))

    entry_prices = _resolve_price_series(frame, price_in_col, open_col)
    if entry_prices is None:
        raise KeyError(
            "Unable to resolve entry price series. Checked columns: "
            f"{price_in_col!r}, {open_col!r}"
        )
    exit_prices_base = _resolve_price_series(frame, price_out_col, close_col)
    if exit_prices_base is None:
        raise KeyError(
            "Unable to resolve exit price series. Checked columns: "
            f"{price_out_col!r}, {close_col!r}"
        )

    atr_series = _as_float_series(frame, atr_col)
    if atr_series is None:
        close_series = _resolve_price_series(frame, close_col, None)
        if close_series is None:
            raise KeyError(
                "Volatility column could not be resolved; provide either an ATR column "
                "or a numeric close column for fallback computation"
            )
        log_returns = np.log(close_series).diff()
        window = max(288, max(horizons) * 2)
        atr_series = log_returns.rolling(window=window, min_periods=window // 2).std()
    volatility = atr_series.astype(float) / max(atr_scale, 1e-12)
    volatility = volatility.reindex(frame.index)

    weights = _resolve_weights(horizons, params.get("weights"))
    aggregated = pd.Series(np.nan, index=frame.index, dtype=float)

    base_entry = entry_prices.shift(-1)
    for h, w in zip(horizons, weights):
        exit_prices = exit_prices_base.shift(-h)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_ret = np.log(exit_prices) - np.log(base_entry)
        log_ret = log_ret.where(np.isfinite(log_ret))

        vol_scale = np.sqrt(float(h)) * volatility
        vol_scale = vol_scale.where(vol_scale > 0.0, other=np.nan)
        vol_scale = vol_scale.clip(lower=vol_floor)
        adj_return = log_ret / vol_scale
        aggregated = aggregated.add(w * adj_return, fill_value=0.0)

    if winsor_limits and isinstance(winsor_limits, Sequence) and len(winsor_limits) == 2:
        low_q, high_q = sorted(float(q) for q in winsor_limits)
        if 0.0 <= low_q < high_q <= 1.0:
            quantiles = aggregated.quantile([low_q, high_q], interpolation="linear")
            lower, upper = quantiles.iloc[0], quantiles.iloc[-1]
            aggregated = aggregated.clip(lower=lower, upper=upper)

    aggregated = aggregated.rename(output_column)
    return output_column, aggregated
