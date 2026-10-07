"""Conditional-selection converters.

``where`` lowers to one select layer after rank-aligning the three operands.
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import broadcast, get_trt_tensor


def _register(trt: Any) -> None:
    @register_converter("where")
    def convert_where(ctx: ConversionContext, target: Any, args: Any,
                      kwargs: Any, name: str) -> Any:
        if len(args) != 3:
            raise ValueError(f"where {name!r} expects condition and two operands")
        condition, lhs, rhs = args
        condition = get_trt_tensor(ctx, condition, ctx.unique(f"{name}_cond"))
        lhs = get_trt_tensor(ctx, lhs, ctx.unique(f"{name}_lhs"))
        rhs = get_trt_tensor(ctx, rhs, ctx.unique(f"{name}_rhs"))
        rank = max(len(condition.shape), len(lhs.shape), len(rhs.shape))
        condition, lhs, rhs = _rank_align_three(
            ctx, condition, lhs, rhs, name, rank
        )
        layer = ctx.net.add_select(condition, lhs, rhs)
        layer.name = name
        return layer.get_output(0)


def _rank_align_three(ctx: ConversionContext, condition: Any, lhs: Any,
                      rhs: Any, name: str, rank: int) -> Any:
    from ..converter_utils import prepend_ones

    def pad(tensor: Any, stem: str) -> Any:
        missing = rank - len(tensor.shape)
        if missing > 0:
            return prepend_ones(ctx, tensor, ctx.unique(f"{name}_{stem}"),
                                missing)
        return tensor

    return pad(condition, "cond"), pad(lhs, "lhs"), pad(rhs, "rhs")
