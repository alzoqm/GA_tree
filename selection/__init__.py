# /selection/init.py
"""Feature selection pipeline (Stages 1 & 2 implementation).

This package follows the roadmap in ``selection_data_design.md`` by
implementing the integrity/screening/calibration flow from Stage 1 and
the confidence-enhancing selection/diversity additions from Stage 2.
"""

from .pipeline import SelectionPipeline, run_pipeline  # noqa: F401
