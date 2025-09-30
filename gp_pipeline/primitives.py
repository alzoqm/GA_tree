# /gp_pipeline/primitives.py
"""Primitive set construction for the GP pipeline."""

from __future__ import annotations

from functools import partial
from typing import Sequence

import numpy as np
from deap import base, creator, gp, tools

from .config import PopulationConfig
from .ir import IRNode


SAFE_EPS = 1e-6


def _rand_const(low: float = -2.0, high: float = 2.0) -> float:
    return float(np.random.uniform(low, high))


def build_primitive_set(feature_columns: Sequence[str]) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", arity=0)
    for col in feature_columns:
        pset.addTerminal(col, name=f"COL_{col}")
    pset.addPrimitive(lambda a, b: ("add", a, b), 2, name="add")
    pset.addPrimitive(lambda a, b: ("sub", a, b), 2, name="sub")
    pset.addPrimitive(lambda a, b: ("mul", a, b), 2, name="mul")
    pset.addPrimitive(lambda a, b: ("div", a, b), 2, name="div")
    pset.addPrimitive(lambda a, b: ("min", a, b), 2, name="min")
    pset.addPrimitive(lambda a, b: ("max", a, b), 2, name="max")
    pset.addPrimitive(lambda x: ("abs", x), 1, name="abs")
    pset.addPrimitive(lambda x: ("log1p", x), 1, name="log1p")
    pset.addPrimitive(lambda x, lo, hi: ("clip", x, lo, hi), 3, name="clip")
    pset.addEphemeralConstant("const", _rand_const)
    return pset


def ensure_creators(num_metrics: int, weights: Sequence[float]):
    if "GPFitness" not in creator.__dict__:
        creator.create("GPFitness", base.Fitness, weights=tuple(weights))
    if "GPIndividual" not in creator.__dict__:
        creator.create("GPIndividual", gp.PrimitiveTree, fitness=creator.GPFitness)


def build_toolbox(feature_columns: Sequence[str], pop_cfg: PopulationConfig, weights: Sequence[float]):
    pset = build_primitive_set(feature_columns)
    ensure_creators(len(weights), weights)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=pop_cfg.min_depth, max_=pop_cfg.max_depth)
    toolbox.register("individual", tools.initIterate, creator.GPIndividual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.pset = pset
    return toolbox
