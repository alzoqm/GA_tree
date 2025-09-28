# /selection_data.py
"""CLI entrypoint for the feature selection pipeline (Stages 1 & 2)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from selection.loader import (
    load_experiment_config,
    load_final_features,
    load_model_config,
    resolve_paths,
)
from selection.pipeline import SelectionPipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the feature selection pipeline (Stages 1 & 2)")
    parser.add_argument("--config", default="Config/experiment_config.yaml", help="Experiment config path")
    parser.add_argument("--dry-run", action="store_true", help="Skip filesystem writes")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to write logs in addition to stderr",
    )
    return parser.parse_args()


def _configure_logging(log_level: str, log_file: Optional[str]) -> None:
    try:
        numeric_level = int(log_level)
    except (TypeError, ValueError):
        numeric_level = getattr(logging, str(log_level).upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("Loading experiment config from %s", args.config)
    experiment_cfg = load_experiment_config(args.config)
    selection_cfg = experiment_cfg.get("selection_pipeline", {})
    if not selection_cfg or not selection_cfg.get("enabled", True):
        raise SystemExit("selection_pipeline.enabled is False; nothing to do")

    config_dir = Path(args.config).parent
    logger.debug("Resolving selection paths using config dir %s", config_dir)
    paths = resolve_paths(experiment_cfg, config_dir)

    logger.info("Loading final feature frame from %s", paths.final_features_path())
    frame = load_final_features(paths)
    logger.info("Loaded frame with shape rows=%d cols=%d", frame.shape[0], frame.shape[1])
    logger.debug("Loading model config from %s", paths.model_config_path)
    model_cfg = load_model_config(paths.model_config_path)

    pipeline = SelectionPipeline(
        config=experiment_cfg,
        selection_cfg=selection_cfg,
        paths=paths,
        dry_run=args.dry_run,
    )
    logger.info("Starting selection pipeline (dry_run=%s)", args.dry_run)
    artifacts = pipeline.run(frame, model_cfg)

    summary = (
        f"Selected {len(artifacts.selected_features)} features; "
        f"{len(artifacts.calibration.feature_num)} numeric ranges; "
        f"{len(artifacts.calibration.feature_bool)} boolean features"
    )
    logger.info(summary)


if __name__ == "__main__":
    main()
