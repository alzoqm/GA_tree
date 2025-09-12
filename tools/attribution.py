#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import pandas as pd

# Ensure project root is on sys.path when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.replayer import compute_shapley_for_all, aggregate_node_contributions


def main():
    ap = argparse.ArgumentParser(description="Compute per-action-node Shapley attributions from logs.")
    ap.add_argument('--data', required=True, help='Path to preprocessed features CSV (same as training).')
    ap.add_argument('--config', default='experiment_config.yaml', help='Experiment YAML to read env.simulation_env config.')
    ap.add_argument('--results-dir', default='results', help='Directory containing events.csv and windows.csv, and to write outputs.')
    ap.add_argument('--device', default='cuda', help='Device for replay (cuda or cpu).')
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    if 'Open time' in df.columns:
        df.set_index('Open time', inplace=True)
    # Use numeric positional index for step lookup
    df.reset_index(drop=True, inplace=True)

    # Load env config
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg.get('evaluation', {}).get('simulation_env', {})
    if not env_cfg:
        raise RuntimeError('simulation_env not found in experiment_config.yaml')

    # Load logs
    events_path = os.path.join(args.results_dir, 'events.csv')
    windows_path = os.path.join(args.results_dir, 'windows.csv')
    if not (os.path.exists(events_path) and os.path.exists(windows_path)):
        raise FileNotFoundError('events.csv or windows.csv not found in results directory')
    events_df = pd.read_csv(events_path)
    windows_df = pd.read_csv(windows_path)

    # Compute Shapley per event and aggregate by node
    shapley_df = compute_shapley_for_all(df, env_cfg, events_df, windows_df, device=args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    shapley_out = os.path.join(args.results_dir, 'event_shapley.csv')
    shapley_df.to_csv(shapley_out, index=False)

    agg_df = aggregate_node_contributions(shapley_df)
    agg_out = os.path.join(args.results_dir, 'action_node_attribution.csv')
    agg_df.to_csv(agg_out, index=False)
    print(f"Saved event-level Shapley -> {shapley_out}")
    print(f"Saved node-level attribution -> {agg_out}")


if __name__ == '__main__':
    main()
