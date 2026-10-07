"""Axis-reordering converters.

Transpose and permute ride the shuffle layer's dedicated permutation input;
``t`` is the 2-D case of permute.
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import (
    positive_dims,
    shuffle_permute,
    UnsupportedOperand,
)


def _transposed(shape_rank: int, dim0: int, dim1: int) -> tuple[int, ...]:
    (first,) = positive_dims(dim0, shape_rank)
    (second,) = positive_dims(dim1, shape_rank)
    permutation = list(range(shape_rank))
    permutation[first], permutation[second] = permutation[second], permutation[first]
    return tuple(permutation)


def _register(trt: Any) -> None:
    @register_converter("transpose")
    def convert_transpose(ctx: ConversionContext, target: Any, args: Any,
                          kwargs: Any, name: str) -> Any:
        if len(args) < 3:
            raise ValueError("transpose expects input and two dims")
        rank = len(args[0].shape)
        return shuffle_permute(
            ctx, args[0], name,
            _transposed(rank, int(args[1]), int(args[2])),
        )

    @register_converter("permute")
    def convert_permute(ctx: ConversionContext, target: Any, args: Any,
                        kwargs: Any, name: str) -> Any:
        if len(args) < 2:
            raise ValueError("permute expects input and a permutation")
        input = args[0]
        rest = args[1:]
        if len(rest) == 1 and isinstance(rest[0], (list, tuple)):
            dims = list(int(dim) for dim in rest[0])
        else:
            dims = list(int(dim) for dim in rest)
        if sorted(positive_dims(dims, len(input.shape))) != list(
            range(len(input.shape))
        ):
            raise UnsupportedOperand(
                "permute needs every axis exactly once"
            )
        return shuffle_permute(ctx, input, name, tuple(dims))

    @register_converter("t")
    def convert_t(ctx: ConversionContext, target: Any, args: Any,
                  kwargs: Any, name: str) -> Any:
        if not args:
            raise ValueError("t expects an input")
        rank = len(args[0].shape)
        if rank != 2:
            raise UnsupportedOperand("t is defined for 2-D tensors")
        return shuffle_permute(ctx, args[0], name, (1, 0))
