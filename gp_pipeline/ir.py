"""Intermediate representation utilities for GP trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from deap import gp


@dataclass(frozen=True)
class IRNode:
    kind: str
    value: str | float
    args: Tuple["IRNode", ...] = ()

    def children(self) -> Sequence["IRNode"]:
        return self.args

    def is_leaf(self) -> bool:
        return not self.args


def tree_to_ir(individual: gp.PrimitiveTree, feature_columns: Sequence[str]) -> IRNode:
    stack: List[IRNode] = []
    for node in reversed(individual):
        if isinstance(node, gp.Terminal):
            value = node.value
            if isinstance(value, str):
                if value.startswith("COL_"):
                    col = value[4:]
                    if col not in feature_columns:
                        raise ValueError(f"Unknown feature column token: {value}")
                    stack.append(IRNode("var", col))
                else:
                    stack.append(IRNode("const", float(value)))
            else:
                stack.append(IRNode("const", float(value)))
        else:
            args = [stack.pop() for _ in range(node.arity)]
            args.reverse()
            stack.append(IRNode("op", node.name, tuple(args)))
    if len(stack) != 1:
        raise ValueError("Primitive tree conversion failed to produce a single root")
    return stack[0]


def iter_nodes(node: IRNode) -> Iterable[IRNode]:
    yield node
    for child in node.args:
        yield from iter_nodes(child)


def hash_ir(node: IRNode) -> Tuple:
    return (
        node.kind,
        node.value,
        tuple(hash_ir(child) for child in node.args),
    )
