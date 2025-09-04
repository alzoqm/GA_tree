from __future__ import annotations

from typing import Dict, List, Tuple
import os
import numpy as np


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def plot_bar(data: Dict[str, float], title: str, out_path: str, top_n: int | None = 30):
    plt = _safe_import_matplotlib()
    if plt is None:
        return False
    items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        items = items[:top_n]
    labels, values = zip(*items) if items else ([], [])
    plt.figure(figsize=(10, max(4, len(labels) * 0.25)))
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return True


def plot_heatmap(mat: np.ndarray, labels: List[str], title: str, out_path: str,
                 vmax: float | None = None):
    plt = _safe_import_matplotlib()
    if plt is None:
        return False
    plt.figure(figsize=(8, 6))
    vmax = vmax or (np.percentile(mat, 99) if mat.size else 1.0)
    im = plt.imshow(mat, cmap='viridis', vmin=0, vmax=max(vmax, 1e-9))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return True


def plot_line(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, out_path: str):
    plt = _safe_import_matplotlib()
    if plt is None:
        return False
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return True

