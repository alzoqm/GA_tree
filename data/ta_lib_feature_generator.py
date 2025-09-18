"""Compatibility wrapper for TA-Lib feature generation.

The implementation has been split across the ``data.features`` subpackage.
Importing this module continues to expose the previous API surface.
"""

from data.features import *  # noqa: F401,F403
from data.features import __all__ as _features_all

__all__ = _features_all

# Clean up re-export helper to avoid leaking into consumers.
del _features_all
