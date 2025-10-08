# /gp_pipeline/checkpoint.py
"""Checkpoint utilities for GP feature pipeline resume.

This module provides a lightweight, JSON-based checkpoint format that
captures the generation index, population representation, run timestamp,
and RNG states needed to resume training from the last completed generation.
"""

from __future__ import annotations

import json
import os
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from deap import gp

from .ir import IRNode, tree_to_ir


class GPPipelineCheckpointManager:
    """Manages GP pipeline checkpoints stored under artifacts_dir/checkpoints/gp.

    Each checkpoint file is named `gp_gen_XXXX.json` and stores:
      - generation_next: int (the next generation index to evaluate)
      - run_timestamp: str (to keep artifacts grouped consistently)
      - population_str: List[str] (DEAP string representation of individuals)
      - best_records: List[dict] (IndividualRecord as dict)
      - history: List[dict] (GenerationSummary as dict)
      - metric_goals: Dict[str, bool]
      - rng: python, numpy, subset_rng states
    """

    def __init__(self, artifacts_dir: os.PathLike[str] | str) -> None:
        self.base_dir = Path(artifacts_dir)
        self.ckpt_dir = self.base_dir / "checkpoints" / "gp"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, generation_next: int) -> Path:
        return self.ckpt_dir / f"gp_gen_{generation_next:04d}.json"

    def find_latest(self) -> Optional[Path]:
        if not self.ckpt_dir.exists():
            return None
        latest_idx = -1
        latest_path: Optional[Path] = None
        for path in self.ckpt_dir.glob("gp_gen_*.json"):
            name = path.stem
            try:
                idx = int(name.split("_")[-1])
            except ValueError:
                continue
            if idx > latest_idx:
                latest_idx = idx
                latest_path = path
        return latest_path

    @staticmethod
    def _serialize_rng_states(subset_rng: np.random.Generator | None) -> Dict[str, object]:
        py_state = random.getstate()
        # numpy global RNG state (legacy RandomState API)
        name, state_arr, pos, has_gauss, cached_gauss = np.random.get_state()

        payload: Dict[str, object] = {
            "python": pickle.dumps(py_state).hex(),
            "numpy": {
                "name": str(name),
                "state": state_arr.tolist(),
                "pos": int(pos),
                "has_gauss": int(has_gauss),
                "cached_gauss": float(cached_gauss),
            },
        }
        if subset_rng is not None:
            try:
                payload["subset_rng"] = subset_rng.bit_generator.state
            except Exception:
                payload["subset_rng"] = None
        else:
            payload["subset_rng"] = None
        return payload

    @staticmethod
    def _restore_rng_states(payload: Dict[str, object], subset_rng: np.random.Generator | None) -> None:
        try:
            py_state_hex = payload.get("python")  # type: ignore[assignment]
            if isinstance(py_state_hex, str):
                random.setstate(pickle.loads(bytes.fromhex(py_state_hex)))
        except Exception:
            pass
        try:
            np_payload = payload.get("numpy")  # type: ignore[assignment]
            if isinstance(np_payload, dict) and "state" in np_payload:
                name = np_payload.get("name", "MT19937")
                pos = int(np_payload.get("pos", 0))
                has_gauss = int(np_payload.get("has_gauss", 0))
                cached_gauss = float(np_payload.get("cached_gauss", 0.0))
                state_arr = np.array(np_payload["state"], dtype=np.uint32)
                np.random.set_state((name, state_arr, pos, has_gauss, cached_gauss))
        except Exception:
            pass
        try:
            subset_state = payload.get("subset_rng")  # type: ignore[assignment]
            if subset_rng is not None and isinstance(subset_state, dict):
                subset_rng.bit_generator.state = subset_state
        except Exception:
            pass

    def save(
        self,
        *,
        generation_next: int,
        run_timestamp: str,
        population: Sequence[gp.PrimitiveTree],
        feature_columns: Sequence[str],
        subset_rng: np.random.Generator | None,
        best_records: Sequence[object],
        history: Sequence[object],
        metric_goals: Dict[str, bool],
    ) -> Path:
        # Use DEAP string representation to ensure we can reconstruct with from_string
        population_str = [str(ind) for ind in population]
        rng_payload = self._serialize_rng_states(subset_rng)

        payload = {
            "version": 1,
            "generation_next": int(generation_next),
            "run_timestamp": str(run_timestamp),
            "population_str": population_str,
            "feature_columns": list(map(str, feature_columns)),
            "best_records": list(best_records),
            "history": list(history),
            "metric_goals": dict(metric_goals),
            "rng": rng_payload,
        }
        path = self._checkpoint_path(generation_next)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path

    def load(self, path: Path) -> Dict[str, object]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def restore_population(self, population_str: Sequence[str], pset: gp.PrimitiveSet) -> List[gp.PrimitiveTree]:
        # Rebuild individuals from their string representation
        out: List[gp.PrimitiveTree] = []
        for s in population_str:
            ind = gp.PrimitiveTree.from_string(s, pset)
            out.append(ind)
        return out

    def restore_rng(self, payload: Dict[str, object], subset_rng: np.random.Generator | None) -> None:
        self._restore_rng_states(payload, subset_rng)
