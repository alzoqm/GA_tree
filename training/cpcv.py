# /training/cpcv.py
"""CPCV split utilities and scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CPCVSplit:
    """Container describing a single CPCV combination."""

    combo_id: int
    test_folds: Tuple[int, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray
    purge_indices: np.ndarray
    embargo_indices: np.ndarray

    def train_segments(self) -> List[Tuple[int, int]]:
        return _indices_to_segments(self.train_indices)

    def test_segments(self) -> List[Tuple[int, int]]:
        return _indices_to_segments(self.test_indices)


def _indices_to_segments(indices: np.ndarray) -> List[Tuple[int, int]]:
    if indices.size == 0:
        return []
    idx_sorted = np.sort(indices.astype(int, copy=False))
    segments: List[Tuple[int, int]] = []
    start = int(idx_sorted[0])
    prev = int(idx_sorted[0])
    for idx in idx_sorted[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        segments.append((start, prev + 1))
        start = idx
        prev = idx
    segments.append((start, prev + 1))
    return segments


class CPCVScheduler:
    """Generate CPCV splits with purge and embargo application."""

    def __init__(
        self,
        total_length: int,
        folds: int,
        test_folds: int,
        purge_bars: Optional[int],
        embargo_bars: Optional[int],
        shuffle_seed: Optional[int] = None,
        max_combinations: Optional[int] = None,
        min_bars_per_split: int = 32,
        parallel_chunks: Optional[int] = None,
    ) -> None:
        if folds <= 1:
            raise ValueError("folds must be > 1 for CPCV")
        if not (1 <= test_folds < folds):
            raise ValueError("test_folds must be in [1, folds-1]")
        self.total_length = int(total_length)
        self.folds = int(folds)
        self.test_folds = int(test_folds)
        self.purge_bars = max(0, int(purge_bars)) if purge_bars is not None else 0
        self.embargo_bars = max(0, int(embargo_bars)) if embargo_bars is not None else 0
        self.shuffle_seed = shuffle_seed
        self.max_combinations = max_combinations if (max_combinations is None or max_combinations > 0) else None
        self.min_bars_per_split = max(2, int(min_bars_per_split))
        self.parallel_chunks = max(1, int(parallel_chunks)) if parallel_chunks else 1

        self.fold_boundaries = self._build_fold_boundaries(total_length, folds)
        self._combos = list(combinations(range(folds), self.test_folds))
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(self._combos)
        if self.max_combinations is not None:
            self._combos = self._combos[: self.max_combinations]

    @staticmethod
    def _build_fold_boundaries(total_length: int, folds: int) -> List[Tuple[int, int]]:
        fold_sizes = [total_length // folds] * folds
        remainder = total_length % folds
        for i in range(remainder):
            fold_sizes[i] += 1
        boundaries: List[Tuple[int, int]] = []
        cursor = 0
        for size in fold_sizes:
            start = cursor
            stop = min(total_length, cursor + size)
            boundaries.append((start, stop))
            cursor = stop
        return boundaries

    def __len__(self) -> int:
        return len(self._combos)

    def splits(self) -> Iterator[CPCVSplit]:
        for combo_id, fold_tuple in enumerate(self._combos):
            yield self._build_split(combo_id, fold_tuple)

    def _build_split(self, combo_id: int, fold_tuple: Tuple[int, ...]) -> CPCVSplit:
        total = self.total_length
        test_mask = np.zeros(total, dtype=bool)
        purge_mask = np.zeros(total, dtype=bool)
        embargo_mask = np.zeros(total, dtype=bool)

        for fold_idx in fold_tuple:
            start, stop = self.fold_boundaries[fold_idx]
            if stop - start < 2:
                continue
            test_mask[start:stop] = True

            if self.purge_bars > 0:
                purge_start = max(0, start - self.purge_bars)
                purge_mask[purge_start:start] = True
            if self.embargo_bars > 0:
                embargo_start = stop
                embargo_stop = min(total, stop + self.embargo_bars)
                embargo_mask[embargo_start:embargo_stop] = True

        train_mask = ~(test_mask | purge_mask | embargo_mask)

        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        purge_idx = np.flatnonzero(purge_mask & ~test_mask)
        embargo_idx = np.flatnonzero(embargo_mask & ~test_mask)

        if train_idx.size < self.min_bars_per_split or test_idx.size < self.min_bars_per_split:
            return CPCVSplit(
                combo_id=combo_id,
                test_folds=tuple(sorted(fold_tuple)),
                train_indices=np.array([], dtype=int),
                test_indices=np.array([], dtype=int),
                purge_indices=purge_idx,
                embargo_indices=embargo_idx,
            )

        return CPCVSplit(
            combo_id=combo_id,
            test_folds=tuple(sorted(fold_tuple)),
            train_indices=train_idx,
            test_indices=test_idx,
            purge_indices=purge_idx,
            embargo_indices=embargo_idx,
        )

    def enumerate(self) -> List[CPCVSplit]:
        return [self._build_split(i, combo) for i, combo in enumerate(self._combos)]

    def iter_valid_splits(self) -> Iterator[CPCVSplit]:
        for split in self.splits():
            if split.train_indices.size >= self.min_bars_per_split and split.test_indices.size >= self.min_bars_per_split:
                yield split

    def iter_split_batches(
        self,
        batch_size: Optional[int] = None,
        require_valid: bool = True,
    ) -> Iterator[List[CPCVSplit]]:
        """Yield CPCV splits grouped into batches.

        Args:
            batch_size: Desired batch size. When ``None`` or <= 1, yields single splits.
            require_valid: When True, only include splits meeting ``min_bars_per_split``.
        """
        if batch_size is None or batch_size <= 1:
            for split in self.iter_valid_splits() if require_valid else self.splits():
                yield [split]
            return

        current: List[CPCVSplit] = []
        iterator: Iterable[CPCVSplit]
        iterator = self.iter_valid_splits() if require_valid else self.splits()
        for split in iterator:
            current.append(split)
            if len(current) >= batch_size:
                yield current
                current = []
        if current:
            yield current
