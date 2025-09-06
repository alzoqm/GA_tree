# /training/__init__.py
"""
Narrow exports to avoid leaking legacy/compat symbols.
Consumers should import training APIs explicitly from their modules.
"""

# Explicit re-exports (no wildcards)
from .trading_env import TradingEnvironment
from .predictor import (
    build_adjacency_list_cuda,
    predict_population_cuda,
    predict_population_cpu,
    predict_population,
)

__all__ = [
    'TradingEnvironment',
    'build_adjacency_list_cuda',
    'predict_population_cuda',
    'predict_population_cpu',
    'predict_population',
]
