# /training/early_stopping.py
"""Utilities for monitoring early stopping conditions during GA training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class EarlyStoppingConfig:
    """Configuration options for early stopping."""

    metric: str
    stat: str = "mean"
    minimize: Optional[bool] = None
    patience: int = 5
    min_delta: float = 0.0
    warmup_generations: int = 0

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "EarlyStoppingConfig":
        if "metric" not in cfg:
            raise ValueError("early_stopping.metric must be provided when enabled")
        return cls(
            metric=str(cfg.get("metric")),
            stat=str(cfg.get("stat", "mean")),
            minimize=cfg.get("minimize"),
            patience=int(cfg.get("patience", 5)),
            min_delta=float(cfg.get("min_delta", 0.0)),
            warmup_generations=int(cfg.get("warmup_generations", 0)),
        )


@dataclass
class EarlyStoppingMonitor:
    """Tracks improvements of a metric and signals when to stop training."""

    config: EarlyStoppingConfig
    best_value: Optional[float] = None
    best_generation: Optional[int] = None
    num_bad_epochs: int = 0
    minimize: Optional[bool] = field(default=None, init=False)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best_value": self.best_value,
            "best_generation": self.best_generation,
            "num_bad_epochs": self.num_bad_epochs,
            "minimize": self.minimize,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best_value = state.get("best_value")
        self.best_generation = state.get("best_generation")
        self.num_bad_epochs = int(state.get("num_bad_epochs", 0))
        self.minimize = state.get("minimize")

    @property
    def metric(self) -> str:
        return self.config.metric

    @property
    def stat(self) -> str:
        return self.config.stat

    @property
    def warmup_generations(self) -> int:
        return self.config.warmup_generations

    def register_direction(self, minimize: bool) -> None:
        if self.minimize is None:
            self.minimize = bool(minimize)

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        assert self.minimize is not None, "minimize direction must be set before comparison"
        delta = value - self.best_value
        if self.minimize:
            return delta <= -self.config.min_delta
        return delta >= self.config.min_delta

    def update(self, *, value: Optional[float], generation: int, minimize: Optional[bool] = None) -> bool:
        """Update monitor with the latest value.

        Returns True if training should stop.
        """
        if value is None:
            # Missing metric counts as no improvement but should not crash
            self.num_bad_epochs += 1
            return self.num_bad_epochs > self.config.patience

        if minimize is not None:
            self.register_direction(minimize)
        elif self.minimize is None:
            raise ValueError("EarlyStoppingMonitor requires a minimize direction before comparisons")

        if generation < self.config.warmup_generations:
            # Ignore updates before warmup completes
            return False

        if self.minimize is None:
            raise ValueError("EarlyStoppingMonitor direction not set before update")

        if self._is_better(value):
            self.best_value = value
            self.best_generation = generation
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs > self.config.patience

    def describe(self) -> str:
        if self.best_value is None:
            return "No improvements recorded"
        direction = "min" if self.minimize else "max"
        return (f"best {self.metric}/{self.stat} = {self.best_value:.6f} "
                f"at generation {self.best_generation} ({direction}-oriented)")


def build_early_stopping_monitor(config_dict: Optional[Dict[str, Any]]) -> Optional[EarlyStoppingMonitor]:
    """Factory helper that returns a configured monitor or None."""
    if not config_dict:
        return None
    if not bool(config_dict.get("enabled", False)):
        return None
    cfg = EarlyStoppingConfig.from_dict(config_dict)
    monitor = EarlyStoppingMonitor(cfg)
    monitor.minimize = cfg.minimize
    return monitor
