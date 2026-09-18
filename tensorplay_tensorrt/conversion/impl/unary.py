"""Unary pointwise converters."""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import get_trt_tensor


def _unary(op: Any) -> Any:
    def convert(
        ctx: ConversionContext,
        target: Any,
        args: Any,
        kwargs: Any,
        name: str,
    ) -> Any:
        if not args:
            raise ValueError(f"unary operation {name!r} expects one operand")
        x = get_trt_tensor(ctx, args[0], ctx.unique(f"{name}_x"))
        layer = ctx.net.add_unary(x, op)
        layer.name = name
        return layer.get_output(0)

    return convert


def _register(trt: Any) -> None:
    register_converter("neg")(_unary(trt.UnaryOperation.NEG))
    register_converter("exp")(_unary(trt.UnaryOperation.EXP))
    register_converter("log")(_unary(trt.UnaryOperation.LOG))
    register_converter("sqrt")(_unary(trt.UnaryOperation.SQRT))
    register_converter("abs")(_unary(trt.UnaryOperation.ABS))
    register_converter("floor")(_unary(trt.UnaryOperation.FLOOR))
    register_converter("sin")(_unary(trt.UnaryOperation.SIN))
    register_converter("cos")(_unary(trt.UnaryOperation.COS))
    register_converter("sign")(_unary(trt.UnaryOperation.SIGN))
