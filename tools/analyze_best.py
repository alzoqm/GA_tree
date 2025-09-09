#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from typing import List

import yaml
import numpy as np
import pandas as pd

import torch

from models.model import GATreePop
from analysis import trees as trees_analysis
from analysis import plots as plots
from analysis.importance import quantile_binned_curve, permutation_importance
from fitness.simulator import FitnessSimulator
from training.wfo import WalkForwardOptimizer

import warnings
warnings.filterwarnings("ignore")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_population(path: str) -> GATreePop:
    pop = GATreePop(1, 1, 1, 1, {}, {}, [], [])  # placeholder; load() will reinit
    pop.load(path)
    return pop


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description='Analyze best GA_tree models')
    ap.add_argument('-c', '--config', default='experiment_config.yaml', help='Path to experiment config')
    ap.add_argument('--outdir', default=None, help='Override results output dir')
    ap.add_argument('--topk', type=int, default=None, help='Limit number of individuals to analyze')
    ap.add_argument('--quantiles', type=int, default=10, help='Quantile bins for response curves')
    ap.add_argument('--max_features', type=int, default=10, help='Max features for response/importance')
    args = ap.parse_args()

    cfg = load_config(args.config)
    env = cfg['environment']
    data_cfg = cfg['data']
    eval_cfg = cfg['evaluation']
    wfo_cfg = cfg.get('walk_forward', {'enabled': False})

    results_dir = args.outdir or env['output_dir']
    out_dir = os.path.join(results_dir, 'analysis')
    ensure_dir(out_dir)

    # Prefer Top-K if present
    topk_path = os.path.join(results_dir, 'best_population_topK.pth')
    best_path = os.path.join(results_dir, 'best_population.pth')
    pop_path = topk_path if os.path.exists(topk_path) else best_path

    population = load_population(pop_path)
    all_features: List[str] = population.all_features

    if args.topk is not None and args.topk < population.pop_size:
        # Trim to first K for analysis
        k = args.topk
        trimmed = GATreePop(k, population.max_nodes, population.max_depth, population.max_children,
                            population.feature_num, population.feature_comparison_map,
                            population.feature_bool, population.all_features)
        trimmed.population_tensor.copy_(population.population_tensor[:k])
        trimmed.set_next_idx()
        trimmed.initialized = True
        population = trimmed

    # 1) Tree-structure analyses
    usage = trees_analysis.feature_usage_counts(population.population_tensor, all_features)
    mix = trees_analysis.comparison_type_mix(population.population_tensor)
    mat, labels = trees_analysis.feature_interaction_matrix(population.population_tensor, all_features)
    dists = trees_analysis.threshold_distributions(population.population_tensor, all_features)

    # Save JSON summaries
    with open(os.path.join(out_dir, 'feature_usage.json'), 'w') as f:
        json.dump(usage, f, indent=2)
    with open(os.path.join(out_dir, 'comparison_type_mix.json'), 'w') as f:
        json.dump(mix, f, indent=2)
    np.save(os.path.join(out_dir, 'feature_interaction.npy'), mat)
    with open(os.path.join(out_dir, 'threshold_distributions.json'), 'w') as f:
        json.dump({k: v[:5000] for k, v in dists.items()}, f, indent=2)  # cap for size

    # Plots (best-effort if matplotlib is present)
    plots.plot_bar(usage, 'Feature Usage Frequency', os.path.join(out_dir, 'feature_usage.png'))
    plots.plot_bar(mix, 'Comparison Type Mix', os.path.join(out_dir, 'comparison_type_mix.png'))
    plots.plot_heatmap(mat, labels, 'Feature Interaction (FEAT-FEAT)', os.path.join(out_dir, 'feature_interaction.png'))

    # 2) Quantile-binned response curves and permutation importance (lightweight)
    #    Use last WFO window OOS by default; if WFO disabled, use last 20% of data.
    final_features_path = os.path.join(env['data_dir'], data_cfg['final_features_path'])
    df = pd.read_csv(final_features_path, parse_dates=['Open time']).set_index('Open time')
    total_len = len(df)

    if wfo_cfg.get('enabled', False):
        from training.wfo import generate_wfo_windows
        windows = generate_wfo_windows(
            total_length=total_len,
            train_bars=wfo_cfg['train_bars'],
            oos_bars=wfo_cfg['oos_bars'],
            step_bars=wfo_cfg['step_bars'],
            warmup_bars=wfo_cfg.get('warmup_bars', 0),
            embargo_bars=wfo_cfg.get('embargo_bars', 0),
            scheme=wfo_cfg.get('scheme', 'sliding'),
        )
        if not windows:
            start, stop = int(0.8 * total_len), total_len
        else:
            _, _, start, stop = windows[-1]
    else:
        start, stop = int(0.8 * total_len), total_len

    # Select features for deeper analysis: top-M by usage
    sorted_usage = sorted(usage.items(), key=lambda x: x[1], reverse=True)
    focus_features = [k for k, _ in sorted_usage[:args.max_features] if k in df.columns]

    # Build evaluation config for OOS objectives (ensure mean_return is first)
    oos_obj_cfg = eval_cfg.get('oos_objectives', eval_cfg.get('objectives', {}))
    # Ensure mean_return is present in first position for baseline importance metric
    obj_names = oos_obj_cfg['objectives']
    if 'mean_return' in obj_names:
        pass
    else:
        obj_names = ['mean_return'] + obj_names
    oos_cfg = {**eval_cfg, 'objectives': {**oos_obj_cfg, 'objectives': obj_names}}

    # Quantile curves using strategy step-returns from simulator traces
    try:
        sim = FitnessSimulator()
        # Request returns history
        _, ret_stack = sim.simulate(population=population,
                                    data=df,
                                    all_feature_names=all_features,
                                    start_data_cnt=start,
                                    stop_data_cnt=stop,
                                    device=env['device'],
                                    evaluation_config=oos_cfg,
                                    return_returns=True)
        for fname in focus_features:
            feat_series = df[fname].iloc[start:stop]
            xs, ys = quantile_binned_curve(feat_series, ret_stack, q=args.quantiles)
            if xs.size and ys.size:
                plots.plot_line(xs.tolist(), ys.tolist(), f'Quantile Response (strategy): {fname}',
                                'feature (quantile center)', 'avg step return',
                                os.path.join(out_dir, f'quantile_curve_{fname}.png'))
    except Exception:
        pass

    # Permutation importance (Top-M features, using mean_return degradation)
    try:
        imp = permutation_importance(population=population,
                                     data=df,
                                     start=start,
                                     stop=stop,
                                     device=env['device'],
                                     evaluation_config=oos_cfg,
                                     feature_names=all_features,
                                     base_metric_index=0,
                                     features_to_permute=focus_features)
        # Filter to focus features for readability
        imp_focus = {k: imp[k] for k in focus_features if k in imp}
        with open(os.path.join(out_dir, 'permutation_importance.json'), 'w') as f:
            json.dump(imp_focus, f, indent=2)
        plots.plot_bar(imp_focus, 'Permutation Importance (Δ mean_return)', os.path.join(out_dir, 'permutation_importance.png'))
    except Exception:
        pass

    print(f"Analysis complete. Outputs -> {out_dir}")


if __name__ == '__main__':
    main()
