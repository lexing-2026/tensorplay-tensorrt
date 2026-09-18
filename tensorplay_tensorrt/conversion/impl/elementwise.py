"""Binary elementwise converters.

Every binary converter follows the same three moves: coerce both operands to
ITensors (a literal is typed to its tensor sibling), rank-align them, and add
one elementwise layer.  Only after the ranks agree does TensorRT apply its
NumPy-style broadcasting.
"""

from __future__ import annotations

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


def _register(trt: Any) -> None:
    register_converter(
        "add",
    )(_binary(trt.ElementWiseOperation.SUM))
    register_converter("sub")(_binary(trt.ElementWiseOperation.SUB))
    register_converter("mul")(_binary(trt.ElementWiseOperation.PROD))
    register_converter("truediv", "div")(_binary(trt.ElementWiseOperation.DIV))
    register_converter("floordiv")(_binary(trt.ElementWiseOperation.FLOOR_DIV))
    register_converter("pow")(_binary(trt.ElementWiseOperation.POW))
    register_converter("max")(_binary(trt.ElementWiseOperation.MAX))
    register_converter("min")(_binary(trt.ElementWiseOperation.MIN))
    register_converter("and_")(_binary(trt.ElementWiseOperation.AND))
    register_converter("or_")(_binary(trt.ElementWiseOperation.OR))
    register_converter("eq")(_binary(trt.ElementWiseOperation.EQUAL))
    register_converter("lt")(_binary(trt.ElementWiseOperation.LESS))
    register_converter("gt")(_binary(trt.ElementWiseOperation.GREATER))
