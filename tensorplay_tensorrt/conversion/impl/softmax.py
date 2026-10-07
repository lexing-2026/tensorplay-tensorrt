"""Softmax converters.

TRT normalizes along one axis per layer, so a softmax over ``dim`` is the
axis itself; log softmax chains a log layer after it.
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import positive_dims


def _softmax(ctx: ConversionContext, name: str, input: Any, dim: Any) -> Any:
    (axis,) = positive_dims(dim, len(input.shape))
    layer = ctx.net.add_softmax(input)
    layer.name = name
    layer.axes = 1 << axis
    return layer.get_output(0)


def _register(trt: Any) -> None:
    @register_converter("softmax")
    def convert_softmax(ctx: ConversionContext, target: Any, args: Any,
                        kwargs: Any, name: str) -> Any:
        if not args:
            raise ValueError(f"softmax {name!r} expects one operand")
        dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
        if dim is None or (isinstance(dim, str) and dim == "undefined"):
            dim = -1
        return _softmax(ctx, name, args[0], dim)

    @register_converter("log_softmax")
    def convert_log_softmax(ctx: ConversionContext, target: Any, args: Any,
                            kwargs: Any, name: str) -> Any:
        if not args:
            raise ValueError(f"log_softmax {name!r} expects one operand")
        dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
        if dim is None or (isinstance(dim, str) and dim == "undefined"):
            dim = -1
        out = _softmax(ctx, ctx.unique(f"{name}_softmax"), args[0], dim)
        layer = ctx.net.add_unary(out, trt.UnaryOperation.LOG)
        layer.name = ctx.unique(f"{name}_log")
        return layer.get_output(0)
