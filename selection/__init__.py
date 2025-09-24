"""Feature selection pipeline (Stage 1 implementation).

This package implements the Stage 1 pipeline described in
``selection_data_design.md``.  It focuses on integrity screening,
rationalized rank-based screening, representative selection, and
calibration with reproducible snapshotting.
"""

from .pipeline import SelectionPipeline, run_pipeline  # noqa: F401
