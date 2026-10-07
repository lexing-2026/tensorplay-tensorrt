"""Linear-layer converter.

``linear(x, w, b)`` multiplies by the transposed weight matrix and adds the
bias.  A 1-D bias is rank-aligned with the matmul output so the elementwise
add broadcasts it without materializing the full shape.
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import broadcast, get_trt_tensor
from .matmul import matrix_multiply


def linear(
    ctx: ConversionContext,
    name: str,
    input: Any,
    weight: Any,
    bias: Any,
) -> Any:
    trt = ctx.trt
    out = matrix_multiply(
        ctx,
        ctx.unique(f"{name}_matmul"),
        input,
        weight,
        lhs_op=trt.MatrixOperation.NONE,
        rhs_op=trt.MatrixOperation.TRANSPOSE,
    )
    if bias is not None:
        out, bias_t = broadcast(
            ctx, out, bias, ctx.unique(f"{name}_out"), ctx.unique(f"{name}_bias")
        )
        layer = ctx.net.add_elementwise(
            out, bias_t, trt.ElementWiseOperation.SUM
        )
        layer.name = ctx.unique(f"{name}_add_bias")
        return layer.get_output(0)
    return out


def _register(trt: Any) -> None:
    @register_converter("linear")
    def convert_linear(ctx: ConversionContext, target: Any, args: Any,
                       kwargs: Any, name: str) -> Any:
        if len(args) != 3:
            raise ValueError(f"linear {name!r} expects input, weight and bias")
        input, weight, bias = args
        weight = get_trt_tensor(ctx, weight, ctx.unique(f"{name}_weight"))
        if bias is not None:
            bias = get_trt_tensor(
                ctx, bias, ctx.unique(f"{name}_bias"), dtype=weight.dtype
            )
        return linear(ctx, name, input, weight, bias)
