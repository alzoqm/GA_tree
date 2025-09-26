# /selection/snapshot.py
"""Snapshot helpers for reproducibility."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence

import yaml


@dataclass(slots=True)
class SnapshotPayload:
    created_at: str
    git_commit: str
    data_meta_hash: str
    env: Dict[str, object]
    method: Dict[str, object]
    feature_num_overrides: Dict[str, tuple[float, float]]
    feature_bool_whitelist: list[str]
    feature_comparison_whitelist: list[str]


def compute_data_meta_hash(*, data_path: Path | None, df_shape: tuple[int, int], config_digest: str, extra: Mapping[str, object]) -> str:
    payload = {
        "data_path": str(data_path) if data_path else "",
        "rows": df_shape[0],
        "cols": df_shape[1],
        "config_digest": config_digest,
        "extra": extra,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def atomic_write_yaml(path: Path, data: Mapping[str, object], backup: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=False)
    if backup and path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_suffix(path.suffix + f".bak-{stamp}")
        path.replace(backup_path)
    os.replace(tmp_path, path)


def append_selection_snapshot(path: Path, snapshot: Mapping[str, object]) -> None:
    path = Path(path)
    existing: Dict[str, object]
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            existing = yaml.safe_load(fh) or {}
    else:
        existing = {}

    history = existing.setdefault("selection_snapshot_history", [])
    history.append(snapshot)
    existing["selection_snapshot"] = snapshot

    atomic_write_yaml(path, existing, backup=True)


def build_snapshot_payload(
    *,
    env: Mapping[str, object],
    method: Mapping[str, object],
    feature_num: Mapping[str, tuple[float, float]],
    feature_bool: Mapping[str, bool] | Sequence[str],
    feature_comparison: Sequence[str] | Mapping[str, Sequence[str]],
    git_commit: str,
    data_meta_hash: str,
) -> Dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    if isinstance(feature_bool, dict):
        bool_list = [name for name, flag in feature_bool.items() if flag]
    else:
        bool_list = list(feature_bool)

    if isinstance(feature_comparison, Mapping):
        comparison_whitelist = sorted(set(feature_comparison.keys()))
    else:
        comparison_whitelist = list(feature_comparison)

    whitelist = sorted(set(feature_num.keys()) | set(bool_list))

    payload = {
        "created_at": now,
        "git_commit": git_commit,
        "data_meta_hash": data_meta_hash,
        "env": dict(env),
        "method": dict(method),
        "active_feature_whitelist": whitelist,
        "feature_num_overrides": {k: list(v) for k, v in feature_num.items()},
        "feature_bool_whitelist": bool_list,
        "feature_comparison_whitelist": comparison_whitelist,
    }
    return payload
