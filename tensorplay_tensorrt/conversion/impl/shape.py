"""Shape-changing converters.

Reshape-style changes go through a shuffle layer's reshape input, which also
performs size-1 dimension broadcasting for ``expand``.  ``-1`` infers a
dimension and ``0`` copies one, matching TensorRT's reshape semantics.
Identity-shaped operations (contiguous, clone, detach) return their operand;
inference-mode dropout and like-shaped constant fills are covered too.
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import (
    reshape_static,
    static_shape,
    UnsupportedOperand,
)


def _reshape_from_args(ctx: ConversionContext, name: str, input: Any,
                       rest: Any) -> Any:
    """Collect shape arguments: one sequence or scalars spread over args."""

    if len(rest) == 1 and isinstance(rest[0], (list, tuple)):
        dims = tuple(int(dim) for dim in rest[0])
    else:
        dims = tuple(int(dim) for dim in rest)
    if not dims:
        raise UnsupportedOperand(f"{name!r} received an empty target shape")
    return reshape_static(ctx, input, name, dims)


def _flatten_shape(shape: tuple[int, ...], start: int, end: int) -> tuple[int, ...]:
    rank = len(shape)
    if start < 0:
        start += rank
    if end < 0:
        end += rank
    if not 0 <= start < rank or not 0 <= end < rank or end < start:
        raise UnsupportedOperand(
            f"flatten range [{start}, {end}] invalid for rank {rank}"
        )
    collapsed = 1
    for dim in shape[start:end + 1]:
        collapsed *= dim
    return shape[:start] + (collapsed,) + shape[end + 1:]


def _register(trt: Any) -> None:
    @register_converter("view", "reshape")
    def convert_view(ctx: ConversionContext, target: Any, args: Any,
                     kwargs: Any, name: str) -> Any:
        if len(args) < 2:
            raise ValueError(f"{name!r} expects input and target shape")
        return _reshape_from_args(ctx, name, args[0], args[1:])

    @register_converter("flatten")
    def convert_flatten(ctx: ConversionContext, target: Any, args: Any,
                        kwargs: Any, name: str) -> Any:
        if not args:
            raise ValueError("flatten expects an input")
        input = args[0]
        shape = static_shape(input)
        start = args[1] if len(args) > 1 else kwargs.get("start_dim", 0)
        end = args[2] if len(args) > 2 else kwargs.get("end_dim", -1)
        return reshape_static(ctx, input, name,
                              _flatten_shape(shape, int(start), int(end)))

    @register_converter("squeeze")
    def convert_squeeze(ctx: ConversionContext, target: Any, args: Any,
                        kwargs: Any, name: str) -> Any:
        if not args:
            raise ValueError("squeeze expects an input")
        input = args[0]
        shape = static_shape(input)
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        if dim is None:
            drop = tuple(index for index, size in enumerate(shape) if size == 1)
        else:
            drop = (int(dim),)
        if not drop:
            return input
        kept = [
            size for index, size in enumerate(shape)
            if index not in drop
        ] or [1]
        return reshape_static(ctx, input, name, tuple(kept))

    @register_converter("unsqueeze")
    def convert_unsqueeze(ctx: ConversionContext, target: Any, args: Any,
                          kwargs: Any, name: str) -> Any:
        if len(args) < 2:
            raise ValueError("unsqueeze expects input and dim")
        input = args[0]
        shape = list(static_shape(input))
        dim = int(args[1])
        if dim < 0:
            dim += len(shape) + 1
        if not 0 <= dim <= len(shape):
            raise UnsupportedOperand(f"unsqueeze dim {args[1]} out of range")
        shape.insert(dim, 1)
        return reshape_static(ctx, input, name, tuple(shape))

    @register_converter("expand")
    def convert_expand(ctx: ConversionContext, target: Any, args: Any,
                       kwargs: Any, name: str) -> Any:
        if len(args) < 2:
            raise ValueError("expand expects input and target sizes")
        input = args[0]
        rest = args[1:]
        if len(rest) == 1 and isinstance(rest[0], (list, tuple)):
            sizes = list(int(size) for size in rest[0])
        else:
            sizes = list(int(size) for size in rest)
        shape = list(static_shape(input))
        # Eager right-aligns the existing dims: a shorter size list keeps the
        # leading input dims, a longer one prepends new broadcast dims.
        rank = len(sizes)
        if rank < len(shape):
            sizes = list(shape[: len(shape) - rank]) + sizes
            rank = len(sizes)
        offset = rank - len(shape)
        for index, size in enumerate(sizes):
            existing = shape[index - offset] if index >= offset else None
            if size == -1:
                sizes[index] = existing if existing is not None else 1
            elif existing is None:
                if size < 1:
                    raise UnsupportedOperand(
                        f"cannot expand a new leading dim to {size}"
                    )
            elif size != existing and existing != 1:
                raise UnsupportedOperand(
                    f"cannot expand dim {index} from {existing} to {size}"
                )
        rank = len(sizes)
        if len(shape) < rank:
            input = reshape_static(
                ctx, input, ctx.unique(f"{name}_align"),
                (1,) * (rank - len(shape)) + tuple(shape),
            )
        # A slice layer with stride 0 broadcasts a size-1 dimension to the
        # requested extent, which is exactly a static expand.
        aligned = list(static_shape(input))
        strides = [0 if aligned[index] == 1 else 1 for index in range(rank)]
        layer = ctx.net.add_slice(input, [0] * rank, sizes, strides)
        layer.name = name
        return layer.get_output(0)

    @register_converter("contiguous", "clone", "detach", "dropout")
    def convert_identity(ctx: ConversionContext, target: Any, args: Any,
                         kwargs: Any, name: str) -> Any:
        if not args:
            raise ValueError(f"{name!r} expects an input")
        # An identity shuffle emits a fresh tensor with the same shape, so a
        # network input is never also the network output.  The region was
        # traced under inference discipline, so dropout has no mask to
        # materialize.
        layer = ctx.net.add_shuffle(args[0])
        layer.name = name
        return layer.get_output(0)

    @register_converter("zeros_like", "ones_like")
    def convert_like_fill(ctx: ConversionContext, target: Any, args: Any,
                          kwargs: Any, name: str) -> Any:
        if not args:
            raise ValueError(f"{name!r} expects an input")
        import numpy as np

        shape = static_shape(args[0])
        value = 0.0 if name == "zeros_like" else 1.0
        np_dtype = np.float32
        if getattr(args[0], "dtype", None) is not None:
            dtype_name = str(args[0].dtype).rsplit(".", 1)[-1]
            np_dtype = getattr(np, dtype_name, np.float32)
        weights = ctx.trt.Weights(np.full(shape, value, dtype=np_dtype))
        layer = ctx.net.add_constant(shape, weights)
        layer.name = name
        return layer.get_output(0)
