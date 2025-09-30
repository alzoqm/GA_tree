# /gp_pipeline/validators.py
"""Wrappers around selection validators used for GP fitness evaluation."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

try:
    from .boolean_feature_validator import validate_boolean_feature
    from .single_feature_validator import validate_numeric_feature
except ImportError:  # pragma: no cover
    validate_boolean_feature = None
    validate_numeric_feature = None


class ValidatorUnavailable(RuntimeError):
    """Raised when selection validators are not available."""


def _prepare_kwargs(
    base_kwargs: Optional[Dict[str, object]],
    *,
    default_supervision: str,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"verbose": False, "supervision": default_supervision}
    if base_kwargs:
        kwargs.update(base_kwargs)
    return kwargs


def run_numeric_validator(
    df: pd.DataFrame,
    feature: str,
    *,
    target: Optional[str],
    kwargs: Optional[Dict[str, object]] = None,
):
    if validate_numeric_feature is None:
        raise ValidatorUnavailable("numeric feature validator is not installed")
    supervision = "supervised" if target is not None else "unsupervised"
    effective_kwargs = _prepare_kwargs(kwargs, default_supervision=supervision)
    return validate_numeric_feature(df, feature=feature, target=target, **effective_kwargs)


def run_boolean_validator(
    df: pd.DataFrame,
    feature: str,
    *,
    target: Optional[str],
    kwargs: Optional[Dict[str, object]] = None,
):
    if validate_boolean_feature is None:
        raise ValidatorUnavailable("boolean feature validator is not installed")
    supervision = "supervised" if target is not None else "unsupervised"
    effective_kwargs = _prepare_kwargs(kwargs, default_supervision=supervision)
    return validate_boolean_feature(df, feature=feature, target=target, **effective_kwargs)
