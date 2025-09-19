# /analysis/robustness.py
"""Utility helpers to run robustness checks on candidate strategies.

These routines provide lightweight approximations of the statistics described in
Bailey & López de Prado (PBO/DSR) and White/Hansen (Reality Check / SPA).
The implementations here prioritise integration scaffolding; they deliver
heuristic estimates so the pipeline can surface potential red flags without
blocking execution when full statistical tooling is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.stat_tests import deflated_sharpe_ratio


@dataclass
class RobustnessReport:
    """Aggregate robustness diagnostics for a single candidate."""

    pbo: Optional[float] = None
    dsr: Optional[float] = None
    spa_pvalue: Optional[float] = None
    min_trl_ok: Optional[bool] = None
    bootstrap_ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _flatten_returns(returns_stack: Sequence[np.ndarray]) -> np.ndarray:
    non_empty = [arr for arr in returns_stack if isinstance(arr, np.ndarray) and arr.size > 0]
    if not non_empty:
        return np.zeros((0, 0), dtype=np.float32)
    shapes = {arr.shape[1] for arr in non_empty}
    if len(shapes) != 1:
        raise ValueError("Inconsistent population size in returns stack.")
    return np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in non_empty], axis=0)


def _compute_naive_pbo(raw_oos: np.ndarray, metric_idx: int) -> np.ndarray:
    """Crude proxy for Probability of Backtest Overfitting.

    We count how often each candidate is the top performer across combinations
    based on the selected metric (default: mean return). A candidate that rarely
    ranks first is assigned a higher pseudo-PBO, signalling instability.
    """
    combos, pop_size, _ = raw_oos.shape
    if combos == 0:
        return np.ones((pop_size,), dtype=np.float32)
    metric_slice = raw_oos[:, :, metric_idx]
    winners = np.argmax(metric_slice, axis=1)
    counts = np.bincount(winners, minlength=pop_size).astype(np.float32)
    freq = counts / float(combos)
    return 1.0 - freq  # Higher => greater overfitting risk


def _compute_dsr(candidate_returns: np.ndarray, config: dict) -> Optional[float]:
    if candidate_returns.size < 3:
        return None
    sr_benchmark = float(config.get('sr_benchmark', 0.0))
    sr_hat = float(np.mean(candidate_returns))
    std = float(np.std(candidate_returns, ddof=1))
    if std <= 1e-12:
        return None
    sr_hat /= std
    n_obs = candidate_returns.size
    return deflated_sharpe_ratio(
        sr_hat=sr_hat,
        n_obs=n_obs,
        skew=float(np.mean(((candidate_returns - candidate_returns.mean()) / (std + 1e-12)) ** 3)),
        kurtosis=float(np.mean(((candidate_returns - candidate_returns.mean()) / (std + 1e-12)) ** 4)),
        n_trials=int(config.get('n_trials', 1)),
        rho=float(config.get('rho', 0.0)),
        sr_benchmark=sr_benchmark,
    )


def _bootstrap_ci(candidate_returns: np.ndarray, block_length: int, samples: int, alpha: float) -> Tuple[float, float]:
    if candidate_returns.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(314159)
    sample_size = candidate_returns.size
    block_length = max(1, min(block_length, sample_size))
    num_blocks = int(np.ceil(sample_size / block_length))
    estimates = []
    for _ in range(max(samples, 1)):
        max_start = sample_size - block_length
        if max_start <= 0:
            indices = np.zeros(num_blocks, dtype=int)
        else:
            indices = rng.integers(0, max_start + 1, size=num_blocks)
        boot = np.concatenate([
            candidate_returns[idx: idx + block_length]
            for idx in indices
        ])[:sample_size]
        estimates.append(float(np.mean(boot)))
    lower = float(np.percentile(estimates, (1 - alpha) / 2 * 100))
    upper = float(np.percentile(estimates, (1 + alpha) / 2 * 100))
    return lower, upper


def run_robustness_checks(
    raw_returns: Optional[Sequence[np.ndarray]],
    raw_oos_metrics: Optional[np.ndarray],
    oos_metric_names: Sequence[str],
    config: dict,
    minimize_flags: Optional[Sequence[bool]] = None,
    metrics_are_minimized: bool = True,
) -> List[RobustnessReport]:
    """Compute robustness diagnostics for each candidate.

    Parameters
    ----------
    raw_returns : list of arrays
        Step-level return matrices per combination [steps, population].
    raw_oos_metrics : np.ndarray
        OOS aggregated metrics per combination [combos, population, metrics].
    oos_metric_names : list[str]
        Names corresponding to the last axis in raw_oos_metrics.
    config : dict
        Robustness configuration block from experiment_config.yaml.
    minimize_flags : sequence of bool, optional
        Minimization flags aligned with ``raw_oos_metrics`` columns. Used to
        recover natural (maximize) orientation when metrics arrive in
        minimization form.
    metrics_are_minimized : bool, default True
        Set to True when ``raw_oos_metrics`` is provided in minimization space.
    """
    if isinstance(raw_oos_metrics, np.ndarray) and metrics_are_minimized and minimize_flags:
        oos_metrics_nat = raw_oos_metrics.copy()
        max_idx = min(oos_metrics_nat.shape[2], len(minimize_flags))
        for idx in range(max_idx):
            if not minimize_flags[idx]:
                oos_metrics_nat[..., idx] = -oos_metrics_nat[..., idx]
    else:
        oos_metrics_nat = raw_oos_metrics

    pop_size = raw_returns[0].shape[1] if raw_returns and raw_returns[0].ndim == 2 else (
        oos_metrics_nat.shape[1] if isinstance(oos_metrics_nat, np.ndarray) else 0
    )
    if pop_size == 0:
        return []

    reports: List[RobustnessReport] = [RobustnessReport() for _ in range(pop_size)]

    returns_matrix = _flatten_returns(raw_returns) if raw_returns else np.zeros((0, pop_size))

    # --- PBO (heuristic) ---
    pbo_cfg = config.get('pbo', {}) if isinstance(config, dict) else {}
    if pbo_cfg.get('enabled', False) and isinstance(oos_metrics_nat, np.ndarray):
        metric_name = 'mean_return'
        metric_idx = oos_metric_names.index(metric_name) if metric_name in oos_metric_names else 0
        combos = oos_metrics_nat.shape[0]
        min_splits = int(pbo_cfg.get('min_splits', 0))
        if combos < min_splits:
            for report in reports:
                report.notes.append(f'PBO skipped (need >= {min_splits} splits, have {combos}).')
        else:
            pbo_scores = _compute_naive_pbo(oos_metrics_nat, metric_idx)
            threshold = pbo_cfg.get('threshold')
            for idx, score in enumerate(pbo_scores):
                reports[idx].pbo = float(score)
                if threshold is not None and float(score) > float(threshold):
                    reports[idx].notes.append(f'PBO {score:.3f} exceeds threshold {threshold}.')
    else:
        for report in reports:
            report.notes.append('PBO disabled or unavailable')

    # --- DSR ---
    dsr_cfg = dict(config.get('dsr', {})) if isinstance(config, dict) else {}
    if dsr_cfg.get('enabled', False) and returns_matrix.size > 0:
        if dsr_cfg.get('n_trials') in (None, 0):
            dsr_cfg['n_trials'] = pop_size
        min_value = dsr_cfg.get('min_value')
        for idx in range(pop_size):
            candidate_returns = returns_matrix[:, idx]
            dsr_value = _compute_dsr(candidate_returns, dsr_cfg)
            reports[idx].dsr = None if dsr_value is None else float(dsr_value)
            if dsr_value is None:
                reports[idx].notes.append('DSR unavailable (insufficient variance or samples).')
            elif min_value is not None and float(dsr_value) < float(min_value):
                reports[idx].notes.append(
                    f'DSR {float(dsr_value):.3f} below minimum {min_value}.'
                )
    else:
        for report in reports:
            report.notes.append('DSR disabled or insufficient data')

    # --- SPA / Reality Check (placeholder) ---
    if config.get('spa', {}).get('enabled', False):
        for report in reports:
            report.spa_pvalue = None
            report.notes.append('SPA placeholder – detailed implementation pending')

    # --- Block bootstrap confidence intervals ---
    if config.get('block_bootstrap', {}).get('enabled', False) and returns_matrix.size > 0:
        bb_cfg = config.get('block_bootstrap', {})
        block_length = int(bb_cfg.get('block_length', 50))
        samples = int(bb_cfg.get('samples', 1000))
        alpha = float(bb_cfg.get('confidence', 0.95))
        for idx in range(pop_size):
            lower, upper = _bootstrap_ci(returns_matrix[:, idx], block_length, samples, alpha)
            reports[idx].bootstrap_ci['mean_return'] = (lower, upper)
    else:
        for report in reports:
            report.notes.append('Bootstrap disabled or insufficient data')

    # --- Min Track Record Length (heuristic) ---
    if config.get('mintrl', {}).get('enabled', False) and returns_matrix.size > 0:
        confidence = float(config.get('mintrl', {}).get('confidence', 0.95))
        required_length = int(np.ceil(1.0 / (1.0 - confidence)))
        actual_length = returns_matrix.shape[0]
        ok = actual_length >= required_length
        for report in reports:
            report.min_trl_ok = ok
            if not ok:
                report.notes.append(f'MinTRL heuristic failed ({actual_length} < {required_length})')
    else:
        for report in reports:
            report.notes.append('MinTRL check disabled or insufficient data')

    return reports
