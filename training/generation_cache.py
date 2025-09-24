# /training/generation_cache.py
"""Utilities for generation-level CPCV simulation caching."""

from __future__ import annotations

import hashlib
import logging
import shutil
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from training.cpcv import CPCVScheduler, CPCVSplit


@dataclass(frozen=True)
class SegmentKey:
    """Deterministic identifier for a cached simulation segment."""

    kind: str
    start_idx: int
    stop_idx: int
    purge_before: int
    embargo_after: int
    population_hash: str
    feature_hash: str

    def digest(self) -> str:
        hasher = hashlib.blake2b(digest_size=20)
        hasher.update(self.kind.encode('utf-8'))
        hasher.update(int(self.start_idx).to_bytes(8, 'big', signed=True))
        hasher.update(int(self.stop_idx).to_bytes(8, 'big', signed=True))
        hasher.update(int(self.purge_before).to_bytes(4, 'big', signed=True))
        hasher.update(int(self.embargo_after).to_bytes(4, 'big', signed=True))
        hasher.update(self.population_hash.encode('utf-8'))
        hasher.update(self.feature_hash.encode('utf-8'))
        return hasher.hexdigest()


@dataclass
class SegmentDescriptor:
    """Metadata describing a contiguous segment emitted by the scheduler."""

    key: SegmentKey
    start: int
    stop: int
    kind: str
    combo_id: int
    order: int


@dataclass
class SegmentRecord:
    """Cached payload container for a unique segment."""

    descriptor: SegmentDescriptor
    ref_count: int = 0
    returns: Optional[np.ndarray] = None
    env_state: Optional[Dict[str, object]] = None
    node_stats: Optional[Dict[str, object]] = None
    loaded: bool = False
    file_path: Optional[Path] = None

    def payload(self) -> Tuple[np.ndarray, Dict[str, object], Optional[Dict[str, object]]]:
        if not self.loaded or self.returns is None or self.env_state is None:
            raise RuntimeError("Segment payload requested before simulation results were loaded.")
        return self.returns, self.env_state, self.node_stats


@dataclass
class ComboPlan:
    """Ordered segment plan for a specific CPCV combination."""

    combo_id: int
    train_segments: List[SegmentDescriptor] = field(default_factory=list)
    test_segments: List[SegmentDescriptor] = field(default_factory=list)


class GenerationSimulationCache:
    """Manage unique CPCV segment simulations within a generation."""

    def __init__(
        self,
        *,
        population_hash: str,
        feature_hash: str,
        generation_id: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.population_hash = population_hash
        self.feature_hash = feature_hash
        self.generation_id = generation_id
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._segments: Dict[SegmentKey, SegmentRecord] = {}
        self._combos: Dict[int, ComboPlan] = {}
        self._manifest: Dict[str, Dict[str, object]] = {}

        self.base_dir: Optional[Path] = None
        self.segment_dir: Optional[Path] = None
        self.manifest_path: Optional[Path] = None

        if self.cache_dir is not None:
            label = str(generation_id) if generation_id is not None else f"adhoc_{population_hash[:10]}"
            safe_label = label.replace('/', '_')
            self.base_dir = self.cache_dir / f"generation_{safe_label}"
            self.segment_dir = self.base_dir / "segments"
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.segment_dir.mkdir(parents=True, exist_ok=True)
            self.manifest_path = self.base_dir / "manifest.json"
            self._manifest = self._load_manifest()
            self._cleanup_stale_generations()

    def _cleanup_stale_generations(self) -> None:
        """Remove cache directories older than the previous generation."""
        if self.cache_dir is None:
            return
        try:
            current_idx = int(self.generation_id)
        except (TypeError, ValueError):
            return
        keep_cutoff = current_idx - 1
        prefix = 'generation_'
        for entry in self.cache_dir.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]
            try:
                gen_idx = int(suffix)
            except ValueError:
                continue
            if gen_idx < keep_cutoff:
                try:
                    shutil.rmtree(entry)
                except Exception as exc:
                    logging.warning("Failed to remove stale CPCV cache directory %s: %s", entry, exc)

    @staticmethod
    def _count_adjacent(indices: Sequence[int], start: int, step: int) -> int:
        if isinstance(indices, np.ndarray):
            if indices.size == 0:
                return 0
            index_set = set(int(i) for i in indices.ravel().tolist())
        else:
            seq = list(indices)
            if len(seq) == 0:
                return 0
            index_set = set(int(i) for i in seq)
        count = 0
        cursor = start
        while cursor in index_set:
            count += 1
            cursor += step
        return count

    def _build_descriptor(
        self,
        *,
        split: CPCVSplit,
        start: int,
        stop: int,
        kind: str,
        order: int,
    ) -> SegmentDescriptor:
        purge_before = 0
        embargo_after = 0
        if kind == 'train':
            if split.purge_indices.size:
                purge_before = self._count_adjacent(split.purge_indices, start - 1, step=-1)
            if split.embargo_indices.size:
                embargo_after = self._count_adjacent(split.embargo_indices, stop, step=1)

        key = SegmentKey(
            kind=kind,
            start_idx=start,
            stop_idx=stop,
            purge_before=purge_before,
            embargo_after=embargo_after,
            population_hash=self.population_hash,
            feature_hash=self.feature_hash,
        )
        descriptor = SegmentDescriptor(
            key=key,
            start=start,
            stop=stop,
            kind=kind,
            combo_id=split.combo_id,
            order=order,
        )
        return descriptor

    def prepare(
        self,
        scheduler: CPCVScheduler,
        *,
        min_bars: int,
    ) -> List[ComboPlan]:
        plans: List[ComboPlan] = []
        for split in scheduler.splits():
            if split.train_indices.size < min_bars or split.test_indices.size < min_bars:
                continue

            plan = ComboPlan(combo_id=split.combo_id)

            for order, (start, stop) in enumerate(split.train_segments()):
                if (stop - start) < 2:
                    continue
                descriptor = self._build_descriptor(split=split, start=start, stop=stop, kind='train', order=order)
                plan.train_segments.append(descriptor)
                record = self._segments.get(descriptor.key)
                digest = descriptor.key.digest()
                if record is None:
                    record = SegmentRecord(descriptor=descriptor, ref_count=1)
                    manifest_entry = self._manifest.get(digest)
                    if manifest_entry is not None and self.base_dir is not None:
                        path_str = manifest_entry.get('path')
                        if isinstance(path_str, str):
                            file_path = self.base_dir / path_str
                            if file_path.exists():
                                record.file_path = file_path
                    self._segments[descriptor.key] = record
                else:
                    record.ref_count += 1

            for order, (start, stop) in enumerate(split.test_segments()):
                if (stop - start) < 2:
                    continue
                descriptor = self._build_descriptor(split=split, start=start, stop=stop, kind='test', order=order)
                plan.test_segments.append(descriptor)
                record = self._segments.get(descriptor.key)
                digest = descriptor.key.digest()
                if record is None:
                    record = SegmentRecord(descriptor=descriptor, ref_count=1)
                    manifest_entry = self._manifest.get(digest)
                    if manifest_entry is not None and self.base_dir is not None:
                        path_str = manifest_entry.get('path')
                        if isinstance(path_str, str):
                            file_path = self.base_dir / path_str
                            if file_path.exists():
                                record.file_path = file_path
                    self._segments[descriptor.key] = record
                else:
                    record.ref_count += 1

            if plan.test_segments:
                plans.append(plan)
                self._combos[plan.combo_id] = plan
        return plans

    @property
    def unique_segments(self) -> Iterable[SegmentRecord]:
        def _sort_key(item: SegmentKey) -> Tuple[int, int, int, int, int]:
            kind_rank = 0 if item.kind == 'train' else 1
            return (
                kind_rank,
                item.start_idx,
                item.stop_idx,
                item.purge_before,
                item.embargo_after,
            )

        keys = sorted(self._segments.keys(), key=_sort_key)
        for key in keys:
            yield self._segments[key]

    def register_payload(
        self,
        key: SegmentKey,
        returns: np.ndarray,
        env_state: Dict[str, object],
        node_stats: Optional[Dict[str, object]],
    ) -> None:
        record = self._segments.get(key)
        if record is None:
            raise KeyError("Segment key not known to cache")
        record.returns = returns.astype(np.float32, copy=False)
        record.env_state = env_state
        record.node_stats = node_stats
        record.loaded = True
        if self.segment_dir is not None:
            digest = key.digest()
            file_path = self.segment_dir / f"segment_{digest}.npz"
            payload = {
                'returns': record.returns,
                'env_state': np.array([env_state], dtype=object),
                'node_stats': np.array([node_stats], dtype=object),
            }
            np.savez(file_path, **payload)
            record.file_path = file_path
            self._manifest[digest] = {
                'path': str(file_path.relative_to(self.base_dir)) if self.base_dir is not None else str(file_path),
                'kind': record.descriptor.kind,
                'start': record.descriptor.start,
                'stop': record.descriptor.stop,
                'purge_before': key.purge_before,
                'embargo_after': key.embargo_after,
                'population_hash': key.population_hash,
                'feature_hash': key.feature_hash,
            }
            self._flush_manifest()

    def get_payload(
        self,
        key: SegmentKey,
    ) -> Tuple[np.ndarray, Dict[str, object], Optional[Dict[str, object]]]:
        record = self._segments.get(key)
        if record is None:
            raise KeyError("Segment key not available")
        if not record.loaded and record.file_path is not None:
            self._load_from_disk(record)
        return record.payload()

    def release(self, key: SegmentKey) -> None:
        record = self._segments.get(key)
        if record is None:
            raise KeyError("Segment key not available")
        record.ref_count = max(0, record.ref_count - 1)
        if record.ref_count == 0:
            record.returns = None
            record.env_state = None
            record.node_stats = None
            record.loaded = False

    def get_plan(self, combo_id: int) -> ComboPlan:
        plan = self._combos.get(combo_id)
        if plan is None:
            raise KeyError(f"Combination {combo_id} not prepared")
        return plan

    def _load_manifest(self) -> Dict[str, Dict[str, object]]:
        if self.manifest_path is None or not self.manifest_path.exists():
            return {}
        try:
            with self.manifest_path.open('r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception:
            return {}
        if isinstance(data, dict):
            segments = data.get('segments')
            if isinstance(segments, dict):
                return segments
        return {}

    def _flush_manifest(self) -> None:
        if self.manifest_path is None:
            return
        manifest_payload = {'segments': self._manifest}
        try:
            with self.manifest_path.open('w', encoding='utf-8') as fh:
                json.dump(manifest_payload, fh, indent=2)
        except Exception:
            pass

    def _load_from_disk(self, record: SegmentRecord) -> None:
        if record.file_path is None or not record.file_path.exists():
            raise RuntimeError("Segment payload missing on disk")
        try:
            with np.load(record.file_path, allow_pickle=True) as npz:
                returns = np.asarray(npz['returns'], dtype=np.float32)
                env_state_raw = npz['env_state']
                node_stats_raw = npz['node_stats'] if 'node_stats' in npz else None

                env_state: Dict[str, object]
                if isinstance(env_state_raw, np.ndarray):
                    if env_state_raw.size == 0:
                        env_state = {}
                    else:
                        env_obj = env_state_raw.reshape(-1)[0]
                        if isinstance(env_obj, np.ndarray) and env_obj.shape == ():
                            env_obj = env_obj.item()
                        env_state = env_obj if isinstance(env_obj, dict) else {}
                elif isinstance(env_state_raw, dict):
                    env_state = env_state_raw
                else:
                    env_state = {}

                if isinstance(node_stats_raw, np.ndarray):
                    if node_stats_raw.size == 0:
                        node_stats = None
                    else:
                        node_obj = node_stats_raw.reshape(-1)[0]
                        if isinstance(node_obj, np.ndarray) and node_obj.shape == ():
                            node_obj = node_obj.item()
                        node_stats = node_obj if isinstance(node_obj, dict) else None
                else:
                    node_stats = node_stats_raw if isinstance(node_stats_raw, dict) else None
        except Exception as exc:
            raise RuntimeError(f"Failed to load cached segment payload: {exc}") from exc

        record.returns = returns
        record.env_state = env_state
        record.node_stats = node_stats
        record.loaded = True
