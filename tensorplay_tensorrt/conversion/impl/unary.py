"""Unary pointwise converters.

One ``add_unary`` layer per operation, guarded by the runtime enum so a
build against an older TensorRT simply skips the entries it lacks.  rsqrt
has no enum member; it composes reciprocal over square root.
"""

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


def _rsqrt(trt: Any) -> Any:
    def convert(
        ctx: ConversionContext,
        target: Any,
        args: Any,
        kwargs: Any,
        name: str,
    ) -> Any:
        if not args:
            raise ValueError("rsqrt expects one operand")
        x = get_trt_tensor(ctx, args[0], ctx.unique(f"{name}_x"))
        root = ctx.net.add_unary(x, trt.UnaryOperation.SQRT)
        root.name = ctx.unique(f"{name}_sqrt")
        root = root.get_output(0)
        layer = ctx.net.add_unary(root, trt.UnaryOperation.RECIP)
        layer.name = name
        return layer.get_output(0)

    return convert


def _register(trt: Any) -> None:
    unary_op = trt.UnaryOperation
    for name, member in (
        ("neg", "NEG"),
        ("exp", "EXP"),
        ("log", "LOG"),
        ("sqrt", "SQRT"),
        ("abs", "ABS"),
        ("floor", "FLOOR"),
        ("sin", "SIN"),
        ("cos", "COS"),
        ("sign", "SIGN"),
        ("ceil", "CEIL"),
        ("round", "ROUND"),
        ("reciprocal", "RECIP"),
        ("tan", "TAN"),
        ("sinh", "SINH"),
        ("cosh", "COSH"),
        ("asin", "ASIN"),
        ("acos", "ACOS"),
        ("atan", "ATAN"),
        ("asinh", "ASINH"),
        ("acosh", "ACOSH"),
        ("atanh", "ATANH"),
        ("erf", "ERF"),
        ("logical_not", "NOT"),
        ("isnan", "ISNAN"),
        ("isinf", "ISINF"),
    ):
        op = getattr(unary_op, member, None)
        if op is not None:
            register_converter(name)(_unary(op))

    register_converter("rsqrt")(_rsqrt(trt))
