"""Registry of operation converters.

Converters register by bare op name.  Captured targets arrive in several
costumes -- the standard operator callables report the C implementation
module (``_operator``), TensorPlay function targets may carry an empty
module, and method calls carry the method name itself -- so dispatch runs on
the resolved name alone.  The table is small and every entry was written
against one known operation; a name collision from some other module would
need its own entry and a module check, which is why nothing is mapped
implicitly.

The ``impl`` package populates the registry through import side effects, the
way the interpreter pulls in its converter surface.
"""

from __future__ import annotations

import operator
from typing import Any, Callable

__all__ = ["CONVERTERS", "UnsupportedOperator", "register_converter", "target_name"]


class UnsupportedOperator(Exception):
    """The interpreter reached an operation with no registered converter."""


CONVERTERS: dict[str, Callable[..., Any]] = {}


def register_converter(
    *names: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Bind one converter to one or more op names."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        for name in names:
            CONVERTERS[name] = fn
        return fn

    return deco


_OPERATOR_TARGETS = {
    operator.add: "add",
    operator.sub: "sub",
    operator.mul: "mul",
    operator.truediv: "truediv",
    operator.floordiv: "floordiv",
    operator.pow: "pow",
    operator.neg: "neg",
    operator.and_: "and_",
    operator.or_: "or_",
    operator.eq: "eq",
    operator.lt: "lt",
    operator.gt: "gt",
}


def _bare_op_name(name: str) -> str:
    """Strip namespace and overload decoration from a qualified op name.

    Op overloads report ``namespace::op.overload``; the converter table is
    keyed on the bare operation name so a functional call and its op-overload
    form share one converter.
    """

    if "::" in name:
        name = name.split("::", 1)[1]
    if "." in name:
        name = name.split(".", 1)[0]
    return name


def target_name(node: Any) -> str:
    """Bare op name for a captured call node, or ``""`` when unknown."""

    if node.op == "call_method":
        return str(node.target)
    target = node.target
    if target in _OPERATOR_TARGETS:
        return _OPERATOR_TARGETS[target]
    # Op-overload targets expose the qualified name through ``name()``; the
    # ``namespace`` attribute separates them from plain callables, whose
    # ``__name__`` already is the bare op name.
    if getattr(target, "namespace", None) is not None:
        name_attr = getattr(target, "name", None)
        if callable(name_attr):
            return _bare_op_name(str(name_attr()))
    return _bare_op_name(getattr(target, "__name__", ""))
