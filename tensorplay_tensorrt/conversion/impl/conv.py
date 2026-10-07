"""Convolution converters.

The kernel and bias ride as runtime tensors: a convolution layer takes its
weights through ``set_input`` the same way any other layer does, so trained
parameters stay ordinary network inputs and an engine never bakes values in.
1-D input is lifted to 2-D, computed, and squeezed back.
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import (
    get_trt_tensor,
    reshape_static,
    static_shape,
    UnsupportedOperand,
)


def _tuple_of(value: Any, count: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * count
    if isinstance(value, (list, tuple)):
        values = tuple(int(item) for item in value)
        if len(values) == 1:
            values = values * count
        if len(values) != count:
            raise UnsupportedOperand(
                f"expected {count} values, got {values}"
            )
        return values
    raise UnsupportedOperand(f"expected int or sequence, got {value!r}")


def convolution(
    ctx: ConversionContext,
    name: str,
    input: Any,
    weight: Any,
    bias: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
    num_spatial: int,
) -> Any:
    trt = ctx.trt
    weight = get_trt_tensor(ctx, weight, ctx.unique(f"{name}_weight"))
    weight_shape = static_shape(weight)
    if len(weight_shape) != num_spatial + 2:
        raise UnsupportedOperand(
            f"kernel rank {len(weight_shape)} does not match a "
            f"{num_spatial}-D convolution"
        )
    num_output_maps = weight_shape[0]
    kernel_shape = weight_shape[2:]

    stride = _tuple_of(stride, num_spatial)
    padding = _tuple_of(padding, num_spatial)
    dilation = _tuple_of(dilation, num_spatial)
    if groups is not None and int(groups) != 1:
        raise UnsupportedOperand("grouped convolution is not supported")

    conv = ctx.net.add_convolution_nd(
        input, num_output_maps, kernel_shape, trt.Weights(), trt.Weights()
    )
    conv.name = name
    if weight.dtype != input.dtype:
        weight = cast_layer(ctx, weight, ctx.unique(f"{name}_weight_cast"),
                            input.dtype)
    conv.set_input(1, weight)
    if bias is not None:
        bias = get_trt_tensor(ctx, bias, ctx.unique(f"{name}_bias"),
                              dtype=input.dtype)
        if bias.dtype != input.dtype:
            bias = cast_layer(ctx, bias, ctx.unique(f"{name}_bias_cast"),
                              input.dtype)
        conv.set_input(2, bias)
    conv.stride_nd = stride
    conv.padding_nd = padding
    conv.dilation_nd = dilation
    return conv.get_output(0)


def cast_layer(ctx: ConversionContext, tensor: Any, name: str, dtype: Any) -> Any:
    layer = ctx.net.add_cast(tensor, dtype)
    layer.name = name
    return layer.get_output(0)


def _register(trt: Any) -> None:
    @register_converter("conv1d")
    def convert_conv1d(ctx: ConversionContext, target: Any, args: Any,
                       kwargs: Any, name: str) -> Any:
        input, weight, bias, stride, padding, dilation, groups = _conv_args(args)
        lifted = _unsqueeze_last(ctx, input, ctx.unique(f"{name}_lift"))
        out = convolution(ctx, name, lifted, weight, bias, stride, padding,
                          dilation, groups, num_spatial=2)
        # The lifted axis rides through the 2-D computation untouched.
        return _squeeze_last(ctx, out, ctx.unique(f"{name}_drop"))

    @register_converter("conv2d")
    def convert_conv2d(ctx: ConversionContext, target: Any, args: Any,
                       kwargs: Any, name: str) -> Any:
        input, weight, bias, stride, padding, dilation, groups = _conv_args(args)
        return convolution(ctx, name, input, weight, bias, stride, padding,
                           dilation, groups, num_spatial=2)

    @register_converter("conv3d")
    def convert_conv3d(ctx: ConversionContext, target: Any, args: Any,
                       kwargs: Any, name: str) -> Any:
        input, weight, bias, stride, padding, dilation, groups = _conv_args(args)
        return convolution(ctx, name, input, weight, bias, stride, padding,
                           dilation, groups, num_spatial=3)


def _conv_args(args: Any) -> Any:
    if len(args) < 7:
        raise ValueError("convolution expects input, weight, bias, stride, "
                         "padding, dilation and groups")
    input, weight, bias, stride, padding, dilation, groups = args[:7]
    return input, weight, bias, stride, padding, dilation, groups


def _unsqueeze_last(ctx: ConversionContext, tensor: Any, name: str) -> Any:
    shape = static_shape(tensor) + (1,)
    return reshape_static(ctx, tensor, name, shape)


def _squeeze_last(ctx: ConversionContext, tensor: Any, name: str) -> Any:
    shape = static_shape(tensor)
    return reshape_static(ctx, tensor, name, shape[:-1])
