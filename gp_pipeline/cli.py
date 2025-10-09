# /gp_pipeline/cli.py
"""Command line interface for the GP feature pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import GPConfig, GPConfigError, load_gp_config
from .runner import GPFeatureTrainer



import warnings
warnings.filterwarnings("ignore")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GP feature generation pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Config/gp_feature_pipeline.yaml"),
        help="Path to the GP pipeline configuration file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint if available",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_gp_config(args.config)
    except GPConfigError as exc:
        parser.error(str(exc))
    trainer = GPFeatureTrainer(cfg, resume=bool(args.resume))
    result = trainer.run()
    print(f"Backend used: {trainer.backend_name}")
    print(f"Artifacts saved to: {result['artifacts']}")


if __name__ == "__main__":  # pragma: no cover
    main()
