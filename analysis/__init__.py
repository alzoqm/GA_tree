# /analysis/__init__.py
"""Analysis utilities for GA_tree.

Includes tree-structure summaries, behavior analyses, and lightweight
importance/response-curve tools used outside the training loop.

This package now also contains the former `tools` utilities (merged),
including `wandb_logger` and analysis scripts.
"""

# Re-export merged tools for convenient imports
try:
    from . import wandb_logger as wandb_logger  # noqa: F401
except Exception:
    pass
