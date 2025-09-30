"""Demonstration script for the GP single-feature pipeline described in
"docs/gp_feature_gp_report.md".

It assembles a minimal genetic programming run with DEAP, lowers the
resulting trees to an intermediate representation, and shows how to
execute the generated feature on top of pandas/NumExpr or Polars. The
output can optionally be passed into the provided numeric/boolean
validators so the full loop ("generate → evaluate → validate") is visible
in one place.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

try:
    import numexpr as ne
except ImportError:  # pragma: no cover - optional dependency for the demo
    ne = None

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency for the demo
    pl = None

try:
    from gp_pipeline.boolean_feature_validator import validate_boolean_feature
    from gp_pipeline.single_feature_validator import validate_numeric_feature
except ImportError:  # pragma: no cover - allow running without validators
    validate_boolean_feature = None
    validate_numeric_feature = None


@dataclass(frozen=True)
class IRNode:
    kind: str  # "var", "const", "op"
    value: object
    args: Tuple["IRNode", ...] = ()


NUMEXPR_RENDER: Dict[str, callable] = {
    "add": lambda a, b: f"({a}) + ({b})",
    "sub": lambda a, b: f"({a}) - ({b})",
    "mul": lambda a, b: f"({a}) * ({b})",
    "div": lambda a, b: f"({a}) / ({b})",
    "min": lambda a, b: f"where(({a}) < ({b}), ({a}), ({b}))",
    "max": lambda a, b: f"where(({a}) > ({b}), ({a}), ({b}))",
    "abs": lambda x: f"abs({x})",
    "log1p": lambda x: f"log1p({x})",
    "clip": lambda x, lo, hi: (
        f"where(({x}) < ({lo}), ({lo}),"
        f" where(({x}) > ({hi}), ({hi}), ({x})))"
    ),
}


def _pl_clip(expr: "pl.Expr", lo: "pl.Expr", hi: "pl.Expr") -> "pl.Expr":
    return (
        pl.when(expr < lo)
        .then(lo)
        .when(expr > hi)
        .then(hi)
        .otherwise(expr)
    )


POLARS_RENDER: Dict[str, callable] = {}
if pl is not None:  # pragma: no branch - avoid referencing polars when missing
    POLARS_RENDER = {
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


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    epsilon = 1e-6
    guard = np.where(np.abs(b) < epsilon, np.where(b >= 0.0, epsilon, -epsilon), b)
    return np.divide(a, guard)


SAFE_NUMPY: Dict[str, callable] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": _safe_divide,
    "min": np.minimum,
    "max": np.maximum,
    "abs": np.abs,
    "log1p": np.log1p,
    "clip": lambda x, lo, hi: np.clip(x, lo, hi),
}


def _random_constant() -> float:
    return float(np.random.uniform(-2.0, 2.0))


class GPPipeline:
    def __init__(self, feature_columns: Sequence[str]) -> None:
        self.feature_columns = list(feature_columns)
        self.pset = self._build_primitive_set()
        self.toolbox = self._build_toolbox()

    def _build_primitive_set(self) -> gp.PrimitiveSet:
        pset = gp.PrimitiveSet("MAIN", arity=0)
        for col in self.feature_columns:
            pset.addTerminal(col, name=f"COL_{col}")
        pset.addPrimitive(lambda a, b: ("op", "add", a, b), 2, name="add")
        pset.addPrimitive(lambda a, b: ("op", "sub", a, b), 2, name="sub")
        pset.addPrimitive(lambda a, b: ("op", "mul", a, b), 2, name="mul")
        pset.addPrimitive(lambda a, b: ("op", "div", a, b), 2, name="div")
        pset.addPrimitive(lambda a, b: ("op", "min", a, b), 2, name="min")
        pset.addPrimitive(lambda a, b: ("op", "max", a, b), 2, name="max")
        pset.addPrimitive(lambda x: ("op", "abs", x), 1, name="abs")
        pset.addPrimitive(lambda x: ("op", "log1p", x), 1, name="log1p")
        pset.addPrimitive(
            lambda x, lo, hi: ("op", "clip", x, lo, hi),
            3,
            name="clip",
        )
        pset.addEphemeralConstant("const", _random_constant)
        return pset

    def _build_toolbox(self) -> base.Toolbox:
        if "FitnessSingleObjective" not in creator.__dict__:
            creator.create("FitnessSingleObjective", base.Fitness, weights=(1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessSingleObjective)
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox

    def compile_individual(self, individual: gp.PrimitiveTree) -> IRNode:
        stack: List[IRNode] = []
        for node in reversed(individual):
            if isinstance(node, gp.Terminal):
                value = node.value
                if isinstance(value, str):
                    if value.startswith("COL_"):
                        col = value[4:]
                        if col not in self.feature_columns:
                            raise ValueError(f"Unknown column token: {value}")
                        stack.append(IRNode("var", col))
                        continue
                    stack.append(IRNode("const", float(value)))
                    continue
                stack.append(IRNode("const", float(value)))
            else:
                args = [stack.pop() for _ in range(node.arity)]
                args.reverse()
                stack.append(IRNode("op", node.name, tuple(args)))
        if len(stack) != 1:
            raise ValueError("Primitive tree did not collapse to a single root")
        return stack[0]

    def ir_to_numexpr(self, node: IRNode) -> str:
        if node.kind == "var":
            return node.value
        if node.kind == "const":
            return repr(node.value)
        renderer = NUMEXPR_RENDER[node.value]
        rendered_args = [self.ir_to_numexpr(child) for child in node.args]
        return renderer(*rendered_args)

    def ir_to_polars(self, node: IRNode) -> "pl.Expr":
        if pl is None:
            raise RuntimeError("Polars is not available")
        if node.kind == "var":
            return pl.col(node.value)
        if node.kind == "const":
            return pl.lit(node.value)
        renderer = POLARS_RENDER[node.value]
        rendered_args = [self.ir_to_polars(child) for child in node.args]
        return renderer(*rendered_args)

    def evaluate_numpy(self, node: IRNode, df: pd.DataFrame) -> np.ndarray:
        if node.kind == "var":
            return df[node.value].to_numpy(dtype=np.float32, copy=False)
        if node.kind == "const":
            return np.full(len(df), float(node.value), dtype=np.float32)
        op = SAFE_NUMPY[node.value]
        args = [self.evaluate_numpy(child, df) for child in node.args]
        return op(*args)

    def evaluate_numexpr(self, node: IRNode, df: pd.DataFrame) -> np.ndarray:
        if ne is None:
            raise RuntimeError("NumExpr backend requested but numexpr is not installed")
        expr = self.ir_to_numexpr(node)
        local_dict = {col: df[col].to_numpy(dtype=np.float32, copy=False) for col in self.collect_variables(node)}
        return ne.evaluate(expr, local_dict=local_dict)

    def evaluate_polars(self, node: IRNode, df: pd.DataFrame) -> np.ndarray:
        if pl is None:
            raise RuntimeError("Polars backend requested but polars is not installed")
        expr = self.ir_to_polars(node).alias("gp_feature")
        lazy = pl.from_pandas(df).lazy()
        result = lazy.with_columns(expr).collect()
        return result["gp_feature"].to_numpy()

    def collect_variables(self, node: IRNode) -> Set[str]:
        if node.kind == "var":
            return {node.value}
        vars_found: Set[str] = set()
        for child in node.args:
            vars_found.update(self.collect_variables(child))
        return vars_found


def build_sample_frame(n_rows: int = 1024, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time_index = np.arange(n_rows, dtype=np.float32)
    price = np.cumsum(rng.normal(0.0, 1.0, n_rows)).astype(np.float32)
    volume = rng.normal(1000.0, 120.0, n_rows).astype(np.float32)
    spread = np.sin(time_index / 15.0) + rng.normal(0.0, 0.05, n_rows)
    target = np.roll(price, -1) - price
    df = pd.DataFrame(
        {
            "price": price,
            "volume": volume,
            "spread": spread.astype(np.float32),
            "target": target.astype(np.float32),
        }
    )
    df.iloc[-1, df.columns.get_loc("target")] = 0.0
    return df


def feature_to_boolean(values: np.ndarray) -> np.ndarray:
    return (values > 0.0).astype(np.int8)


def compute_demo_metrics(values: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    vals = values.astype(np.float64)
    tgt = target.astype(np.float64)
    epsilon = 1e-9
    # Pearson correlation
    pearson = float(np.corrcoef(vals, tgt)[0, 1])
    # Spearman correlation via rank transform
    ranks_val = pd.Series(vals).rank(method="average").to_numpy()
    ranks_tgt = pd.Series(tgt).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(ranks_val, ranks_tgt)[0, 1])
    # Sharpe-like signal to noise ratio
    sharpe_like = float(vals.mean() / (vals.std(ddof=1) + epsilon))
    # Hit ratio (positives share)
    hit_ratio = float(np.mean(vals > 0.0))
    # Lag-1 autocorrelation as stability proxy
    autocorr = float(np.corrcoef(vals[:-1], vals[1:])[0, 1]) if len(vals) > 1 else 0.0
    return {
        "pearson": pearson,
        "spearman": spearman,
        "sharpe_like": sharpe_like,
        "hit_ratio": hit_ratio,
        "autocorr_lag1": autocorr,
    }


def run_validators(df: pd.DataFrame, numeric: str, boolean: str, target: str) -> None:
    if validate_numeric_feature is None or validate_boolean_feature is None:
        print("validators unavailable; using built-in demo fitness metrics instead")
    target_values = df[target].to_numpy(dtype=np.float32, copy=False)
    metric_values = compute_demo_metrics(df[numeric].to_numpy(), target_values)
    print(f"   demo fitness metrics: {metric_values}")
    if validate_numeric_feature is None or validate_boolean_feature is None:
        return
    print("→ Running numeric validator (lightweight settings)...")
    numeric_res = validate_numeric_feature(
        df,
        feature=numeric,
        target=target,
        supervision="supervised",
        combine_method="simes",
        min_bars_per_split=32,
        screening_methods={"spearman": {"enabled": True, "cliffs_delta": False}},
        verbose=False,
    )
    print(f"   fitness objectives: {numeric_res.fitness_objectives}")
    print("→ Running boolean validator (numeric > 0 threshold)...")
    boolean_res = validate_boolean_feature(
        df,
        feature=boolean,
        target=target,
        supervision="supervised",
        combine_method="simes",
        min_bars_per_split=32,
        screening_methods={"spearman": {"enabled": True, "cliffs_delta": False}},
        verbose=False,
    )
    print(f"   fitness objectives: {boolean_res.fitness_objectives}")


def print_feature_summary(idx: int, ir: IRNode, pipeline: GPPipeline) -> None:
    expr_str = pipeline.ir_to_numexpr(ir)
    print(f"[Individual {idx}] IR→NumExpr: {expr_str}")
    if pl is not None:
        polars_expr = pipeline.ir_to_polars(ir)
        print(f"             IR→Polars: {polars_expr}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the GP feature pipeline demo")
    parser.add_argument(
        "--backend",
        choices=["numpy", "numexpr", "polars"],
        default="numpy",
        help="Vector engine used to materialize features",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=3,
        help="Number of GP individuals to generate in the demo run",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1024,
        help="Synthetic dataset length (number of time steps)",
    )
    parser.add_argument(
        "--run-validators",
        action="store_true",
        help="Execute numeric/boolean validators after feature creation",
    )
    args = parser.parse_args(argv)

    df = build_sample_frame(n_rows=args.rows)
    pipeline = GPPipeline(feature_columns=["price", "volume", "spread"])
    population = pipeline.toolbox.population(n=args.population)

    backend_eval = {
        "numpy": pipeline.evaluate_numpy,
        "numexpr": pipeline.evaluate_numexpr,
        "polars": pipeline.evaluate_polars,
    }[args.backend]

    for idx, individual in enumerate(population, start=1):
        ir = pipeline.compile_individual(individual)
        print_feature_summary(idx, ir, pipeline)
        values = backend_eval(ir, df).astype(np.float32)
        feature_name = f"gp_feat_{idx}"
        df[feature_name] = values
        bool_name = f"{feature_name}__gt0"
        df[bool_name] = feature_to_boolean(values)
        print(
            f"             numeric stats → mean={values.mean():.3f}, std={values.std(ddof=1):.3f},"
            f" positives={(df[bool_name] == 1).sum()} / {len(df)}"
        )
        if args.run_validators:
            run_validators(df, feature_name, bool_name, target="target")
        print()

    print("Demo complete – generated columns are appended to the DataFrame.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
