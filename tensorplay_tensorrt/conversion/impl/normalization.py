"""Normalization converters.

Eval-mode batch norm folds into a per-channel scale and shift computed from
the running statistics; every operand stays a runtime tensor.  Layer norm is
built from reductions over the trailing normalized axes: mean, centered
variance, and the reciprocal standard deviation, each one layer wide.
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import (
    broadcast,
    get_trt_tensor,
    reshape_static,
    static_shape,
    UnsupportedOperand,
)


def _channel_aligned(
    ctx: ConversionContext,
    tensor: Any,
    name: str,
    input_rank: int,
) -> Any:
    """Reshape a (C,) parameter so it broadcasts along the channel axis."""

    shape = static_shape(tensor)
    if len(shape) != 1:
        raise UnsupportedOperand(
            f"batch-norm parameter must be 1-D, got shape {shape}"
        )
    aligned = (1, shape[0]) + (1,) * (input_rank - 2)
    return reshape_static(ctx, tensor, name, aligned)


def _fold_scale_shift(
    ctx: ConversionContext,
    name: str,
    input: Any,
    weight: Any,
    bias: Any,
    mean: Any,
    var: Any,
    eps: float,
) -> Any:
    """y = x * scale + shift with scale = w / sqrt(var + eps)."""

    trt = ctx.trt
    rank = len(static_shape(input))
    ew = lambda op, a, b, stem: _elementwise(ctx, op, a, b, ctx.unique(f"{name}_{stem}"))

    eps_t = get_trt_tensor(ctx, eps, ctx.unique(f"{name}_eps"), dtype=input.dtype)
    adjusted_var = ew(trt.ElementWiseOperation.SUM, var, eps_t, "adjusted_var")
    std = _unary(ctx, trt.UnaryOperation.SQRT, adjusted_var,
                 ctx.unique(f"{name}_std"))
    scale = ew(trt.ElementWiseOperation.DIV, weight, std, "scale")
    scaled_mean = ew(trt.ElementWiseOperation.PROD, mean, scale, "scaled_mean")
    shift = ew(trt.ElementWiseOperation.SUB, bias, scaled_mean, "shift")
    scale = _channel_aligned(ctx, scale, ctx.unique(f"{name}_scale_align"), rank)
    shift = _channel_aligned(ctx, shift, ctx.unique(f"{name}_shift_align"), rank)
    out = ew(trt.ElementWiseOperation.PROD, input, scale, "scaled")
    return ew(trt.ElementWiseOperation.SUM, out, shift, "shifted")


def _elementwise(ctx: ConversionContext, op: Any, a: Any, b: Any, name: str) -> Any:
    a, b = broadcast(ctx, a, b, name + "_a", name + "_b")
    layer = ctx.net.add_elementwise(a, b, op)
    layer.name = name
    return layer.get_output(0)


def _unary(ctx: ConversionContext, op: Any, tensor: Any, name: str) -> Any:
    layer = ctx.net.add_unary(tensor, op)
    layer.name = name
    return layer.get_output(0)


def _register(trt: Any) -> None:
    @register_converter("batch_norm")
    def convert_batch_norm(ctx: ConversionContext, target: Any, args: Any,
                           kwargs: Any, name: str) -> Any:
        if len(args) < 8:
            raise ValueError("batch_norm expects input, weight, bias, mean, "
                             "var, training, momentum and eps")
        input, weight, bias, mean, var, training = args[:6]
        eps = float(args[7]) if len(args) > 7 else 1e-5
        if bool(training):
            raise UnsupportedOperand(
                "batch norm is lowered in eval mode only"
            )
        weight = get_trt_tensor(ctx, weight, ctx.unique(f"{name}_weight"),
                                dtype=input.dtype)
        bias = get_trt_tensor(ctx, bias, ctx.unique(f"{name}_bias"),
                              dtype=input.dtype)
        mean = get_trt_tensor(ctx, mean, ctx.unique(f"{name}_mean"),
                              dtype=input.dtype)
        var = get_trt_tensor(ctx, var, ctx.unique(f"{name}_var"),
                             dtype=input.dtype)
        return _fold_scale_shift(ctx, name, input, weight, bias, mean, var, eps)

    @register_converter("layer_norm")
    def convert_layer_norm(ctx: ConversionContext, target: Any, args: Any,
                           kwargs: Any, name: str) -> Any:
        if len(args) < 2:
            raise ValueError("layer_norm expects input and normalized_shape")
        input, normalized_shape = args[0], args[1]
        weight = args[2] if len(args) > 2 else None
        bias = args[3] if len(args) > 3 else None
        eps = float(args[4]) if len(args) > 4 else 1e-5
        shape = static_shape(input)
        nd = len(tuple(int(dim) for dim in normalized_shape))
        if nd > len(shape) or tuple(shape[len(shape) - nd:]) != tuple(
            int(dim) for dim in normalized_shape
        ):
            raise UnsupportedOperand(
                "normalized_shape must match the trailing input dims"
            )
        axes = 0
        for offset in range(nd):
            axes |= 1 << (len(shape) - 1 - offset)

        mean = ctx.net.add_reduce(input, trt.ReduceOperation.AVG, axes, True)
        mean.name = ctx.unique(f"{name}_mean")
        mean = mean.get_output(0)
        diff = _elementwise(ctx, trt.ElementWiseOperation.SUB, input, mean,
                            ctx.unique(f"{name}_center"))
        squared = _elementwise(ctx, trt.ElementWiseOperation.PROD, diff, diff,
                               ctx.unique(f"{name}_squared"))
        var = ctx.net.add_reduce(squared, trt.ReduceOperation.AVG, axes, True)
        var.name = ctx.unique(f"{name}_var")
        var = var.get_output(0)
        eps_t = get_trt_tensor(ctx, eps, ctx.unique(f"{name}_eps"),
                               dtype=input.dtype)
        var = _elementwise(ctx, trt.ElementWiseOperation.SUM, var, eps_t,
                           ctx.unique(f"{name}_var_eps"))
        invstd = _unary(ctx, trt.UnaryOperation.SQRT, var,
                        ctx.unique(f"{name}_std"))
        invstd = _unary(ctx, trt.UnaryOperation.RECIP, invstd,
                        ctx.unique(f"{name}_invstd"))
        out = _elementwise(ctx, trt.ElementWiseOperation.PROD, diff, invstd,
                           ctx.unique(f"{name}_normalized"))
        if weight is not None:
            weight = get_trt_tensor(ctx, weight, ctx.unique(f"{name}_weight"),
                                    dtype=input.dtype)
            aligned = (1,) * (len(shape) - nd) + tuple(
                int(dim) for dim in normalized_shape
            )
            weight = reshape_static(ctx, weight,
                                    ctx.unique(f"{name}_weight_align"), aligned)
            out = _elementwise(ctx, trt.ElementWiseOperation.PROD, out, weight,
                               ctx.unique(f"{name}_weighted"))
        if bias is not None:
            bias = get_trt_tensor(ctx, bias, ctx.unique(f"{name}_bias"),
                                  dtype=input.dtype)
            aligned = (1,) * (len(shape) - nd) + tuple(
                int(dim) for dim in normalized_shape
            )
            bias = reshape_static(ctx, bias, ctx.unique(f"{name}_bias_align"),
                                  aligned)
            out = _elementwise(ctx, trt.ElementWiseOperation.SUM, out, bias,
                               ctx.unique(f"{name}_biased"))
        return out
