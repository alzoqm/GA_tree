"""CLI entrypoint for the feature selection pipeline (Stage 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

from selection.loader import (
    load_experiment_config,
    load_final_features,
    load_model_config,
    resolve_paths,
)
from selection.pipeline import SelectionPipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the feature selection pipeline (Stage 1)")
    parser.add_argument("--config", default="Config/experiment_config.yaml", help="Experiment config path")
    parser.add_argument("--dry-run", action="store_true", help="Skip filesystem writes")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    experiment_cfg = load_experiment_config(args.config)
    selection_cfg = experiment_cfg.get("selection_pipeline", {})
    if not selection_cfg or not selection_cfg.get("enabled", True):
        raise SystemExit("selection_pipeline.enabled is False; nothing to do")

    config_dir = Path(args.config).parent
    paths = resolve_paths(experiment_cfg, config_dir)

    frame = load_final_features(paths)
    model_cfg = load_model_config(paths.model_config_path)

    pipeline = SelectionPipeline(
        config=experiment_cfg,
        selection_cfg=selection_cfg,
        paths=paths,
        dry_run=args.dry_run,
    )
    artifacts = pipeline.run(frame, model_cfg)

    summary = (
        f"Selected {len(artifacts.selected_features)} features; "
        f"{len(artifacts.calibration.feature_num)} numeric ranges; "
        f"{len(artifacts.calibration.feature_bool)} boolean features"
    )
    print(summary)


if __name__ == "__main__":
    main()
