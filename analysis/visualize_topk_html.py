#!/usr/bin/env python3
"""
Top‑K Model HTML Visualization (Latest Generation)

Code analysis and rationale (compact):

- Where are the "latest results"? During training, each generation snapshot is
  saved under `<output_dir>/checkpoints/wf_window_{i}/generation_{g}.json` with
  tensors saved alongside (population, best_chromosomes, best_objectives, etc.).
  We detect the latest window/generation by scanning that directory and picking
  the highest generation number found. This avoids relying on main_checkpoint
  fields that may lag or be minimal in some runs.

- How do we know which objectives and directions apply? The generation JSON
  contains `objectives_config` (names + minimize flags) used to select elites.
  The saved `best_objectives_gen_*.pth` values are in minimization space:
  metrics that we conceptually maximize (e.g., return, psr) are stored as
  negated values so that lower is better consistently. For human display we
  flip the sign back when `minimize == False`.

- How do we rank "best" among elites? The file contains top elites already,
  but ordering isn’t guaranteed. We therefore compute a simple, reproducible
  ranking using the existing MultiObjectiveFitnessEvaluator’s
  `calculate_weighted_fitness` on the minimization‑space objectives (equal
  weights if unspecified). This respects objective directions and performs an
  internal minmax normalization before forming a weighted sum. Higher fitness
  is better.

- How do we reconstruct and visualize models? We build GATree objects directly
  from the elite chromosomes. The generation JSON stores the exact
  feature_num/feature_comparison_map/feature_bool used during training, so the
  feature index space matches the chromosomes. We pass these into GATree and
  invoke its `visualize_graph`, which renders an HTML (pyvis). We then create a
  small index HTML with a metrics leaderboard and links to each model’s graph.

Assumptions and fallbacks:
- PyVis and NetworkX are present (requirements.txt includes pyvis, networkx).
- We default to window 0, scanning its generations. Extendable to multiple
  windows if needed.
- If `best_*` files are missing for the latest generation, we raise a clear
  error message.

Usage:
  python tools/visualize_topk_html.py --k 10 \
      --exp-config Config/experiment_config.yaml \
      --output-dir results/exp_default

This writes:
  - <output-dir>/visualizations/topk/index.html          (leaderboard)
  - <output-dir>/visualizations/topk/model_001.html ...  (per-model graph)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

# Ensure repository root is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.model import GATree, get_all_features
from models.constants import (
    NODE_INFO_DIM,
    COL_NODE_TYPE,
    NODE_TYPE_UNUSED,
)
from training.multi_objective_fitness import MultiObjectiveFitnessEvaluator


def load_experiment_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_latest_generation_dir(output_dir: Path, window_idx: int = 0) -> Tuple[Path, int]:
    """Return (window_dir, latest_generation_number).

    We scan `output_dir/checkpoints/wf_window_{window_idx}` and pick the
    highest `generation_*.json` number.
    """
    ckpt_root = output_dir / "checkpoints"
    window_dir = ckpt_root / f"wf_window_{window_idx}"
    if not window_dir.exists():
        raise FileNotFoundError(f"Checkpoint window directory not found: {window_dir}")

    gens: List[int] = []
    for p in window_dir.glob("generation_*.json"):
        try:
            gens.append(int(p.stem.split("_")[1]))
        except Exception:
            continue
    if not gens:
        raise FileNotFoundError(f"No generation_*.json found in {window_dir}")

    latest_gen = max(gens)
    return window_dir, latest_gen


def load_generation_meta(window_dir: Path, gen: int) -> Dict:
    gen_file = window_dir / f"generation_{gen}.json"
    with open(gen_file, "r") as f:
        return json.load(f)


def load_best_sets(window_dir: Path, gen: int) -> Tuple[torch.Tensor, np.ndarray]:
    chrom_path = window_dir / f"best_chromosomes_gen_{gen}.pth"
    obj_path = window_dir / f"best_objectives_gen_{gen}.pth"
    if not chrom_path.exists() or not obj_path.exists():
        raise FileNotFoundError(
            f"Missing best files for generation {gen}: {chrom_path.name}, {obj_path.name}"
        )
    best_chromosomes: torch.Tensor = torch.load(chrom_path, weights_only=False)
    best_objectives: np.ndarray = torch.load(obj_path, weights_only=False)
    if not isinstance(best_chromosomes, torch.Tensor) or best_chromosomes.ndim != 3:
        raise ValueError("best_chromosomes has unexpected structure")
    if not isinstance(best_objectives, np.ndarray) or best_objectives.ndim != 2:
        raise ValueError("best_objectives has unexpected structure")
    if best_chromosomes.shape[0] != best_objectives.shape[0]:
        raise ValueError(
            f"Mismatched counts: {best_chromosomes.shape[0]} chromosomes vs {best_objectives.shape[0]} objectives"
        )
    return best_chromosomes.cpu(), best_objectives


def rank_topk_indices(best_objectives: np.ndarray, objectives_config: Dict, k: int) -> List[int]:
    """Rank elites by a consistent scalar fitness and return top‑k indices.

    We use MultiObjectiveFitnessEvaluator.calculate_weighted_fitness over the
    minimization‑space objectives provided in `best_objectives`.
    """
    evaluator = MultiObjectiveFitnessEvaluator(objectives_config)
    fitness = evaluator.calculate_weighted_fitness(best_objectives)
    # Higher fitness is better
    order = np.argsort(-fitness)
    return order[:k].tolist()


def format_metric(value: float, minimize: bool) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "NaN"
    # stored as minimization values; flip for display if maximize
    disp = value if minimize else -value
    # pretty format
    return f"{disp:.6g}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def render_tree_html(
    chrom: torch.Tensor,
    viz_dir: Path,
    rank_1based: int,
    meta: Dict,
) -> Path:
    """Build a GATree from a chromosome and save pyvis HTML.

    We use the feature_* from generation meta to ensure the feature index space
    matches exactly what the chromosome expects.
    """
    pop_state = meta.get("population_state", {})
    max_nodes = int(pop_state.get("max_nodes"))
    max_depth = int(pop_state.get("max_depth"))
    max_children = int(pop_state.get("max_children"))
    feature_num = pop_state.get("feature_num", {})
    feature_map = pop_state.get("feature_comparison_map", {})
    feature_bool = pop_state.get("feature_bool", [])

    # Compute the all_features list exactly as the model expects
    all_features = get_all_features(feature_num, feature_map, feature_bool)

    # Prepare tree instance and load state (this builds adjacency for visualization)
    tree = GATree(
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_children=max_children,
        feature_num=feature_num,
        feature_comparison_map=feature_map,
        feature_bool=feature_bool,
        all_features=all_features,
    )

    used_count = int((chrom[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED).sum().item())
    state = {
        "data": chrom,
        "next_idx": used_count,
        "max_nodes": max_nodes,
        "max_depth": max_depth,
        "max_children": max_children,
        "feature_num": feature_num,
        "feature_comparison_map": feature_map,
        "feature_bool": feature_bool,
        "all_features": all_features,
    }
    tree.load(state)  # uses provided state and builds adjacency

    out_file = viz_dir / f"model_{rank_1based:03d}.html"
    # Do not auto‑open browser during batch generation
    tree.visualize_graph(file=str(out_file), open_browser=False)
    return out_file


def write_index_html(
    out_path: Path,
    links: List[Tuple[int, Path, Dict[str, str]]],
    obj_names: List[str],
) -> None:
    """Write a simple leaderboard HTML with metric table and links."""
    rows = []
    # Header
    header_cols = ["Rank", "Graph"] + obj_names
    rows.append("<tr>" + "".join(f"<th>{h}</th>" for h in header_cols) + "</tr>")
    # Rows
    for rank, file_path, m in links:
        link = f"<a href='{file_path.name}' target='_blank'>open</a>"
        cols = [str(rank), link] + [m.get(name, "") for name in obj_names]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cols) + "</tr>")

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Top‑K Models Leaderboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    .note {{ color: #555; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f8f8f8; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
  <base href="." />
  <!-- The per-model HTML files are separate pyvis pages opened via link. -->
  <!-- Metrics shown here are in natural orientation (maximize/minimize as expected). -->
  <!-- Fitness ranking used equal weights across normalized objectives. -->
  <!-- Source: tools/visualize_topk_html.py -->
  
</head>
<body>
  <h1>Top‑K Models (Latest Generation)</h1>
  <div class="note">Click "open" to view each model's interactive graph in a new tab.</div>
  <table>
    {''.join(rows)}
  </table>
  <p style="margin-top:12px;color:#777;font-size:12px;">Note: Values reflect latest generation elites; maximize metrics were sign-corrected for readability.</p>
  <p style="color:#777;font-size:12px;">Ranking: equal-weight normalized sum via MultiObjectiveFitnessEvaluator.</p>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Visualize Top‑K models from the latest generation as HTML")
    ap.add_argument("--k", type=int, default=10, help="Number of top models to render")
    ap.add_argument(
        "--exp-config",
        type=str,
        default="Config/experiment_config.yaml",
        help="Path to the experiment configuration YAML file",
    )
    ap.add_argument("--output-dir", type=str, default='/root/workspace/GA_tree/results', help="Override output_dir; defaults to config.environment.output_dir")
    ap.add_argument("--window", type=int, default=0, help="Window index to inspect (default: 0)")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.exp_config))
    output_dir = Path(args.output_dir or cfg.get("environment", {}).get("output_dir", "results/exp_default")).resolve()

    window_dir, latest_gen = find_latest_generation_dir(output_dir, window_idx=args.window)
    gen_meta = load_generation_meta(window_dir, latest_gen)

    # Pull objective names and directions as used for the elite set
    objectives_cfg = gen_meta.get("objectives_config") or {}
    obj_names: List[str] = list(objectives_cfg.get("objectives", []))
    minimize_flags: List[bool] = list(objectives_cfg.get("minimize", []))
    if not obj_names:
        # Fallback to names used in generation_stats, if present
        stats = gen_meta.get("generation_stats", {})
        obj_names = list(stats.get("objective_names", []))
        # Try to infer minimize flags from generation_stats if needed
        minimize_flags = list(stats.get("minimize_objectives", [])) or [False] * len(obj_names)
        objectives_cfg = {"objectives": obj_names, "minimize": minimize_flags}

    best_chromosomes, best_objectives = load_best_sets(window_dir, latest_gen)
    if best_chromosomes.shape[0] == 0:
        raise RuntimeError("No elites found in best_chromosomes for latest generation")

    k = max(1, min(args.k, best_chromosomes.shape[0]))
    top_indices = rank_topk_indices(best_objectives, objectives_cfg, k)

    # Prepare visualization directory
    viz_dir = output_dir / "visualizations" / "topk"
    ensure_dir(viz_dir)

    # Generate per‑model graphs and collect table rows
    link_entries: List[Tuple[int, Path, Dict[str, str]]] = []
    for rank, idx in enumerate(top_indices, start=1):
        chrom = best_chromosomes[idx]
        graph_path = render_tree_html(chrom, viz_dir, rank_1based=rank, meta=gen_meta)

        # Build display metrics (flip sign for maximize)
        metric_row: Dict[str, str] = {}
        for j, name in enumerate(obj_names):
            minimize = bool(minimize_flags[j]) if j < len(minimize_flags) else False
            metric_row[name] = format_metric(float(best_objectives[idx, j]), minimize=minimize)

        link_entries.append((rank, graph_path, metric_row))

    # Write leaderboard index
    index_path = viz_dir / "index.html"
    write_index_html(index_path, link_entries, obj_names)

    print(f"[OK] Wrote leaderboard: {index_path}")
    for _, p, _ in link_entries:
        print(f" - {p}")


if __name__ == "__main__":
    main()
