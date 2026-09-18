"""Activation converters."""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import get_trt_tensor


def _activation(op: Any) -> Any:
    def convert(
        ctx: ConversionContext,
        target: Any,
        args: Any,
        kwargs: Any,
        name: str,
    ) -> Any:
        if not args:
            raise ValueError(f"activation {name!r} expects one operand")
        x = get_trt_tensor(ctx, args[0], ctx.unique(f"{name}_x"))
        layer = ctx.net.add_activation(x, op)
        layer.name = name
        return layer.get_output(0)

    return convert


def _register(trt: Any) -> None:
    register_converter("relu")(_activation(trt.ActivationType.RELU))
    register_converter("tanh")(_activation(trt.ActivationType.TANH))
    register_converter("sigmoid")(_activation(trt.ActivationType.SIGMOID))
