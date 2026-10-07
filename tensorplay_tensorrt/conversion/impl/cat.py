"""Concatenation converter."""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import (
    get_trt_tensor,
    positive_dims,
)


def _register(trt: Any) -> None:
    @register_converter("cat", "concat")
    def convert_cat(ctx: ConversionContext, target: Any, args: Any,
                    kwargs: Any, name: str) -> Any:
        if len(args) < 2:
            raise ValueError("cat expects a tensor list and a dimension")
        tensors, dim = args[0], args[1]
        if not isinstance(tensors, (list, tuple)) or not tensors:
            raise ValueError(f"cat {name!r} needs a non-empty tensor list")
        inputs = [
            get_trt_tensor(ctx, tensor, ctx.unique(f"{name}_in{index}"))
            for index, tensor in enumerate(tensors)
        ]
        (axis,) = positive_dims(dim, len(inputs[0].shape))
        layer = ctx.net.add_concatenation(inputs)
        layer.name = name
        layer.axis = axis
        return layer.get_output(0)
