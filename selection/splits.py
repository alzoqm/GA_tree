# /selection/splits.py
"""Cross-fitting split helpers built on top of ``training.cpcv`` utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from training.cpcv import CPCVScheduler, CPCVSplit


@dataclass(frozen=True)
class CrossFitSplit:
    """Stores the inner A/B split indices for a CPCV train fold."""

    outer: CPCVSplit
    inner_train_a: np.ndarray
    inner_train_b: np.ndarray


def make_cpcv_scheduler(total_len: int, cfg: dict, train_cfg: dict) -> CPCVScheduler:
    folds = cfg.get("folds", 10)
    test_folds = cfg.get("test_folds", 2)
    purge = cfg.get("purge_bars")
    embargo = cfg.get("embargo_bars")
    max_combinations = cfg.get("max_combinations")
    min_bars = cfg.get("min_bars_per_split", train_cfg.get("min_trades_per_split", 64))
    shuffle_seed = cfg.get("shuffle_seed")
    parallel_chunks = cfg.get("parallel_chunks")

    purge = purge if purge is not None else train_cfg.get("max_hold_period_bars")
    embargo = embargo if embargo is not None else train_cfg.get("max_hold_period_bars")

    return CPCVScheduler(
        total_length=total_len,
        folds=folds,
        test_folds=test_folds,
        purge_bars=purge,
        embargo_bars=embargo,
        shuffle_seed=shuffle_seed,
        max_combinations=max_combinations,
        min_bars_per_split=min_bars,
        parallel_chunks=parallel_chunks,
    )


def inner_time_split(train_idx: np.ndarray, ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    if train_idx.size == 0:
        return train_idx, train_idx
    ratio = float(np.clip(ratio, 0.05, 0.95))
    cutoff = int(round(train_idx.size * ratio))
    cutoff = max(1, min(train_idx.size - 1, cutoff))
    sorted_idx = np.sort(train_idx.astype(int, copy=False))
    split_a = sorted_idx[:cutoff]
    split_b = sorted_idx[cutoff:]
    return split_a, split_b


def build_cross_fit_splits(
    scheduler: CPCVScheduler,
    inner_ratio: float = 0.5,
    require_valid: bool = True,
) -> List[CrossFitSplit]:
    splits: List[CrossFitSplit] = []
    iterator: Iterable[CPCVSplit]
    iterator = scheduler.iter_valid_splits() if require_valid else scheduler.splits()
    for outer in iterator:
        if outer.train_indices.size == 0:
            continue
        split_a, split_b = inner_time_split(outer.train_indices, ratio=inner_ratio)
        if split_a.size == 0 or split_b.size == 0:
            continue
        splits.append(CrossFitSplit(outer=outer, inner_train_a=split_a, inner_train_b=split_b))
    return splits
