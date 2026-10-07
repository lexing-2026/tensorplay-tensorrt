"""Pooling converters.

Max and average pooling map onto one ``add_pooling_nd`` layer.  Adaptive
pooling is covered for the two cases that need no interpolation: global
output (one value per channel, through a reduce layer) and output equal to
the input extent (identity).
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import (
    static_shape,
    UnsupportedOperand,
)


def _spatial_tuple(value: Any, count: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * count
    if isinstance(value, (list, tuple)):
        values = tuple(int(item) for item in value)
        if len(values) == 1:
            values = values * count
        if len(values) != count:
            raise UnsupportedOperand(f"expected {count} values, got {values}")
        return values
    if value is None:
        return (0,) * count
    raise UnsupportedOperand(f"expected int or sequence, got {value!r}")


def _pooling(
    ctx: ConversionContext,
    name: str,
    input: Any,
    pooling_type: Any,
    kernel_size: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    ceil_mode: Any,
) -> Any:
    """Add one pooling layer; returns the layer for attribute adjustments."""

    rank = len(static_shape(input))
    num_spatial = rank - 2
    if num_spatial < 1:
        raise UnsupportedOperand(f"pooling needs at least 3 dims, got rank {rank}")
    kernel = _spatial_tuple(kernel_size, num_spatial)
    stride_tuple = _spatial_tuple(stride, num_spatial) if stride else kernel
    padding_tuple = _spatial_tuple(padding, num_spatial)
    if dilation is not None:
        if isinstance(dilation, (list, tuple)):
            dilation_vals = tuple(int(dim) for dim in dilation)
        else:
            dilation_vals = (int(dilation),)
        if any(dim != 1 for dim in dilation_vals):
            raise UnsupportedOperand("pooled dilation is not supported")
    if ceil_mode is not None and bool(ceil_mode):
        raise UnsupportedOperand("ceil_mode is not supported")

    pool = ctx.net.add_pooling_nd(input, pooling_type, kernel)
    pool.name = name
    pool.stride_nd = stride_tuple
    pool.padding_nd = padding_tuple
    return pool


def _adaptive_global(
    ctx: ConversionContext,
    name: str,
    input: Any,
    pooling_type: Any,
    output_size: Any,
) -> Any:
    shape = static_shape(input)
    num_spatial = len(shape) - 2
    requested = tuple(int(dim) for dim in output_size)
    if requested == tuple(shape[2:]):
        return input
    if requested != (1,) * num_spatial:
        raise UnsupportedOperand(
            "adaptive pooling is covered for global output and identity only"
        )
    axes = 0
    for dim in range(num_spatial):
        axes |= 1 << (len(shape) - 1 - dim)
    reduce = ctx.net.add_reduce(input, pooling_type, axes, True)
    reduce.name = name
    return reduce.get_output(0)


def _register(trt: Any) -> None:
    @register_converter("max_pool1d", "max_pool2d", "max_pool3d")
    def convert_max_pool(ctx: ConversionContext, target: Any, args: Any,
                         kwargs: Any, name: str) -> Any:
        if len(args) < 5:
            raise ValueError("max_pool expects input, kernel, stride, "
                             "padding and dilation")
        return _pooling(
            ctx, name, args[0], trt.PoolingType.MAX,
            args[1], args[2], args[3], args[4],
            args[5] if len(args) > 5 else None,
        ).get_output(0)

    @register_converter("avg_pool1d", "avg_pool2d", "avg_pool3d")
    def convert_avg_pool(ctx: ConversionContext, target: Any, args: Any,
                         kwargs: Any, name: str) -> Any:
        if len(args) < 5:
            raise ValueError("avg_pool expects input, kernel, stride, "
                             "padding and dilation")
        pool = _pooling(
            ctx, name, args[0], trt.PoolingType.AVERAGE,
            args[1], args[2], args[3], args[4],
            args[5] if len(args) > 5 else None,
        )
        # count_include_pad arrives after ceil_mode; by default the padded
        # zeros count toward the average, which is the layer's behavior too.
        count_include_pad = args[6] if len(args) > 6 else True
        pool.average_count_excludes_padding = not bool(count_include_pad)
        return pool.get_output(0)

    @register_converter("adaptive_avg_pool1d", "adaptive_avg_pool2d",
                        "adaptive_avg_pool3d")
    def convert_adaptive_avg(ctx: ConversionContext, target: Any, args: Any,
                             kwargs: Any, name: str) -> Any:
        return _adaptive_global(ctx, name, args[0], trt.ReduceOperation.AVG,
                                args[1])

    @register_converter("adaptive_max_pool1d", "adaptive_max_pool2d",
                        "adaptive_max_pool3d")
    def convert_adaptive_max(ctx: ConversionContext, target: Any, args: Any,
                             kwargs: Any, name: str) -> Any:
        return _adaptive_global(ctx, name, args[0], trt.ReduceOperation.MAX,
                                args[1])
