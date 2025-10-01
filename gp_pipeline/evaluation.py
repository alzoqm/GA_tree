# /gp_pipeline/evaluation.py
"""Vectorized execution backends for GP feature IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # optional dependencies
    import numexpr as ne
except ImportError:  # pragma: no cover
    ne = None

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

from .ir import IRNode, hash_ir
from .timeframe_ops import apply_timeframe_operator, is_timeframe_operator


@dataclass(slots=True)
class EvaluationCache:
    arrays: MutableMapping[Tuple, np.ndarray] = field(default_factory=dict)
    polars_expr: MutableMapping[Tuple, "pl.Expr"] = field(default_factory=dict)  # type: ignore[name-defined]

    def get_array(self, key: Tuple) -> Optional[np.ndarray]:
        return self.arrays.get(key)

    def set_array(self, key: Tuple, value: np.ndarray) -> None:
        self.arrays[key] = value

    def get_polars(self, key: Tuple):  # type: ignore[override]
        return self.polars_expr.get(key)

    def set_polars(self, key: Tuple, expr) -> None:  # type: ignore[override]
        self.polars_expr[key] = expr


class EvaluatorError(RuntimeError):
    pass


SAFE_EPS = 1e-6


def evaluate_numpy(node: IRNode, df: pd.DataFrame, cache: Optional[EvaluationCache] = None) -> np.ndarray:
    key = hash_ir(node)
    if cache is not None:
        cached = cache.get_array(key)
        if cached is not None:
            return cached

    if node.kind == "var":
        arr = df[node.value].to_numpy(dtype=np.float32, copy=False)
    elif node.kind == "const":
        arr = np.full(len(df), float(node.value), dtype=np.float32)
    elif node.kind == "op":
        args = [evaluate_numpy(child, df, cache) for child in node.args]
        arr = _apply_numpy(node.value, args)
    else:
        raise EvaluatorError(f"Unknown node kind: {node.kind}")
    if cache is not None:
        cache.set_array(key, arr)
    return arr


NUMPY_OPERATORS: Mapping[str, callable] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": lambda a, b: np.divide(
        a,
        np.where(
            np.abs(b) < SAFE_EPS,
            np.where(b >= 0, SAFE_EPS, -SAFE_EPS),
            b,
        ),
    ),
    "min": np.minimum,
    "max": np.maximum,
    "abs": np.abs,
    "log1p": np.log1p,
    "clip": lambda x, lo, hi: np.clip(x, lo, hi),
}


def _apply_numpy(op: str, args: Sequence[np.ndarray]) -> np.ndarray:
    if op in NUMPY_OPERATORS:
        return NUMPY_OPERATORS[op](*args)
    if is_timeframe_operator(op):
        return apply_timeframe_operator(op, args)
    raise EvaluatorError(f"Unsupported numpy operator: {op}")


NUMEXPR_RENDER = {
    "add": lambda a, b: f"(({a}) + ({b}))",
    "sub": lambda a, b: f"(({a}) - ({b}))",
    "mul": lambda a, b: f"(({a}) * ({b}))",
    "div": lambda a, b: f"(({a}) / ({b}))",
    "min": lambda a, b: f"where(({a}) < ({b}), ({a}), ({b}))",
    "max": lambda a, b: f"where(({a}) > ({b}), ({a}), ({b}))",
    "abs": lambda x: f"abs({x})",
    "log1p": lambda x: f"log1p({x})",
    "clip": lambda x, lo, hi: (
        f"where(({x}) < ({lo}), ({lo}), where(({x}) > ({hi}), ({hi}), ({x})))"
    ),
}


def ir_to_numexpr(node: IRNode) -> str:
    if node.kind == "var":
        return node.value
    if node.kind == "const":
        return repr(float(node.value))
    if is_timeframe_operator(node.value):
        raise EvaluatorError(
            f"Operator '{node.value}' is not supported by the numexpr backend"
        )
    renderer = NUMEXPR_RENDER.get(node.value)
    if renderer is None:
        raise EvaluatorError(f"Operator not supported for numexpr: {node.value}")
    inner = [ir_to_numexpr(child) for child in node.args]
    return renderer(*inner)


def evaluate_numexpr(node: IRNode, df: pd.DataFrame, cache: Optional[EvaluationCache] = None) -> np.ndarray:
    if ne is None:
        raise EvaluatorError("numexpr is not available but was requested")
    key = hash_ir(node)
    if cache is not None:
        cached = cache.get_array(key)
        if cached is not None:
            return cached
    if node.kind == "op" and is_timeframe_operator(node.value):
        arr = _apply_numpy(node.value, [evaluate_numexpr(child, df, cache) for child in node.args])
        if cache is not None:
            cache.set_array(key, arr)
        return arr
    # Fallback to numpy evaluation if any descendant requires timeframe operators.
    if node.kind == "op" and any(is_timeframe_operator(child.value) for child in _collect_ops(node)):
        arr = evaluate_numpy(node, df, cache)
        if cache is not None:
            cache.set_array(key, arr)
        return arr
    expr = ir_to_numexpr(node)
    local_dict = {
        col: df[col].to_numpy(dtype=np.float32, copy=False)
        for col in _collect_vars(node)
    }
    arr = ne.evaluate(expr, local_dict=local_dict)
    if np.isscalar(arr):
        arr = np.full(len(df), float(arr), dtype=np.float32)
    if cache is not None:
        cache.set_array(key, arr)
    return arr


def _collect_vars(node: IRNode) -> Sequence[str]:
    if node.kind == "var":
        return [node.value]
    vars_found: set[str] = set()
    for child in node.args:
        vars_found.update(_collect_vars(child))
    return sorted(vars_found)


def _collect_ops(node: IRNode) -> Sequence[IRNode]:
    if node.kind != "op":
        return []
    nodes = [node]
    for child in node.args:
        nodes.extend(_collect_ops(child))
    return nodes


def ir_to_polars(node: IRNode):  # type: ignore[return-type]
    if pl is None:
        raise EvaluatorError("polars is not available but was requested")
    if node.kind == "var":
        return pl.col(node.value)
    if node.kind == "const":
        return pl.lit(float(node.value))
    builder = POLARS_OPERATORS.get(node.value)
    if builder is None:
        raise EvaluatorError(f"Operator not supported for polars: {node.value}")
    args = [ir_to_polars(child) for child in node.args]
    return builder(*args)


def evaluate_polars(node: IRNode, df: pd.DataFrame, cache: Optional[EvaluationCache] = None) -> np.ndarray:
    if pl is None:
        raise EvaluatorError("polars is not available but was requested")
    key = hash_ir(node)
    if cache is not None:
        cached = cache.get_array(key)
        if cached is not None:
            return cached
    if node.kind == "op" and is_timeframe_operator(node.value):
        args = [evaluate_polars(child, df, cache) for child in node.args]
        arr = apply_timeframe_operator(node.value, args)
        if cache is not None:
            cache.set_array(key, arr)
        return arr
    expr = ir_to_polars(node).alias("gp_feature")
    lazy = pl.from_pandas(df).lazy()
    result = lazy.with_columns(expr).collect(streaming=True)
    arr = result["gp_feature"].to_numpy()
    if cache is not None:
        cache.set_array(key, arr)
    return arr


def _pl_clip(expr, lo, hi):  # pragma: no cover - trivial wrapper
    return (
        pl.when(expr < lo)
        .then(lo)
        .when(expr > hi)
        .then(hi)
        .otherwise(expr)
    )


POLARS_OPERATORS = {}
if pl is not None:  # pragma: no branch
    POLARS_OPERATORS = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "min": lambda a, b: pl.min_horizontal(a, b),
        "max": lambda a, b: pl.max_horizontal(a, b),
        "abs": lambda x: x.abs(),
        "log1p": lambda x: x.log1p(),
        "clip": _pl_clip,
    }


BACKENDS = {
    "numpy": evaluate_numpy,
    "numexpr": evaluate_numexpr,
    "polars": evaluate_polars,
}


def choose_backend(primary: str, fallbacks: Sequence[str]) -> str:
    if primary in BACKENDS:
        if primary == "numexpr" and ne is None:
            pass
        elif primary == "polars" and pl is None:
            pass
        else:
            return primary
    for candidate in fallbacks:
        if candidate not in BACKENDS:
            continue
        if candidate == "numexpr" and ne is None:
            continue
        if candidate == "polars" and pl is None:
            continue
        return candidate
    raise EvaluatorError(
        f"No available backend among primary={primary} and fallbacks={fallbacks}"
    )
