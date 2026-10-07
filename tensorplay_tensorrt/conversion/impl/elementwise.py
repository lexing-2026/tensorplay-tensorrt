"""Binary elementwise converters.

Every binary converter follows the same three moves: coerce both operands to
ITensors (a literal is typed to its tensor sibling), rank-align them, and add
one elementwise layer.  Only after the ranks agree does TensorRT apply its
NumPy-style broadcasting.

Comparison gaps compose from the members the runtime exposes: ``ne`` negates
an equality test, ``ge`` and ``le`` OR the strict comparison with equality,
mirroring how the semantics decompose.  ``clamp_min``/``clamp_max`` are
max/min against a scalar bound.
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import broadcast, get_trt_tensor


def _binary(op: Any) -> Any:
    def convert(
        ctx: ConversionContext,
        target: Any,
        args: Any,
        kwargs: Any,
        name: str,
    ) -> Any:
        if len(args) != 2:
            raise ValueError(f"elementwise operation {name!r} expects two operands")
        lhs, rhs = args
        if isinstance(lhs, (int, float, bool)) and isinstance(rhs, (int, float, bool)):
            raise ValueError(f"elementwise operation {name!r} has no tensor operand")
        if isinstance(lhs, (int, float, bool)):
            lhs_t = get_trt_tensor(ctx, lhs, ctx.unique(f"{name}_lhs"), dtype=rhs.dtype)
            rhs_t = get_trt_tensor(ctx, rhs, ctx.unique(f"{name}_rhs"))
        elif isinstance(rhs, (int, float, bool)):
            lhs_t = get_trt_tensor(ctx, lhs, ctx.unique(f"{name}_lhs"))
            rhs_t = get_trt_tensor(ctx, rhs, ctx.unique(f"{name}_rhs"), dtype=lhs.dtype)
        else:
            lhs_t = get_trt_tensor(ctx, lhs, ctx.unique(f"{name}_lhs"))
            rhs_t = get_trt_tensor(ctx, rhs, ctx.unique(f"{name}_rhs"))
        lhs_t, rhs_t = broadcast(ctx, lhs_t, rhs_t, f"{name}_lhs", f"{name}_rhs")
        layer = ctx.net.add_elementwise(lhs_t, rhs_t, op)
        layer.name = name
        return layer.get_output(0)

    return convert


def _pointwise(ctx: ConversionContext, name: str, op: Any, lhs: Any,
               rhs: Any) -> Any:
    lhs, rhs = broadcast(ctx, lhs, rhs, f"{name}_lhs", f"{name}_rhs")
    layer = ctx.net.add_elementwise(lhs, rhs, op)
    layer.name = name
    return layer.get_output(0)


def _register(trt: Any) -> None:
    ew_op = trt.ElementWiseOperation
    register_converter("add")(_binary(ew_op.SUM))
    register_converter("sub")(_binary(ew_op.SUB))
    register_converter("mul")(_binary(ew_op.PROD))
    register_converter("truediv", "div")(_binary(ew_op.DIV))
    register_converter("floordiv")(_binary(ew_op.FLOOR_DIV))
    register_converter("pow")(_binary(ew_op.POW))
    register_converter("and_")(_binary(ew_op.AND))
    register_converter("or_")(_binary(ew_op.OR))
    register_converter("eq")(_binary(ew_op.EQUAL))
    register_converter("lt")(_binary(ew_op.LESS))
    register_converter("gt")(_binary(ew_op.GREATER))

    def _not(ctx: ConversionContext, name: str, tensor: Any) -> Any:
        layer = ctx.net.add_unary(tensor, trt.UnaryOperation.NOT)
        layer.name = name
        return layer.get_output(0)

    def _ne() -> Any:
        def convert(ctx: ConversionContext, target: Any, args: Any,
                    kwargs: Any, name: str) -> Any:
            equal = _binary(ew_op.EQUAL)
            equal_out = equal(ctx, target, args, kwargs, ctx.unique(f"{name}_eq"))
            return _not(ctx, name, equal_out)

        return convert

    def _ordered(strict_op: Any) -> Any:
        """ge as (a > b) | (a == b); le as (a < b) | (a == b)."""

        def convert(ctx: ConversionContext, target: Any, args: Any,
                    kwargs: Any, name: str) -> Any:
            strict = _binary(strict_op)
            strict_out = strict(ctx, target, args, kwargs,
                                ctx.unique(f"{name}_strict"))
            equal = _binary(ew_op.EQUAL)
            equal_out = equal(ctx, target, args, kwargs, ctx.unique(f"{name}_eq"))
            return _pointwise(ctx, name, ew_op.OR, strict_out, equal_out)

        return convert

    register_converter("ne")(_ne())
    register_converter("ge")(_ordered(ew_op.GREATER))
    register_converter("le")(_ordered(ew_op.LESS))

    def _bound_extremum(op: Any) -> Any:
        def convert(ctx: ConversionContext, target: Any, args: Any,
                    kwargs: Any, name: str) -> Any:
            if len(args) != 2:
                raise ValueError(f"{name!r} expects a tensor and a bound")
            tensor = get_trt_tensor(ctx, args[0], ctx.unique(f"{name}_x"))
            bound = get_trt_tensor(ctx, args[1], ctx.unique(f"{name}_bound"),
                                   dtype=tensor.dtype)
            return _pointwise(ctx, name, op, tensor, bound)

        return convert

    register_converter("clamp_min")(_bound_extremum(ew_op.MAX))
    register_converter("clamp_max")(_bound_extremum(ew_op.MIN))
