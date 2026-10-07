"""Basic-indexing converter for captured ``getitem`` nodes.

Python indexing over static shapes: per dimension an ``add_slice`` window
(start, size, stride), an ``int`` index selects one position and drops the
axis, ``None`` inserts an axis, ``Ellipsis`` expands to full slices.  Steps
must be positive; advanced (tensor) indexing is out of scope and leaves the
region uncompiled.
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


def _normalize_index(index: Any, rank: int) -> list[Any]:
    """Expand one index expression to one entry per input axis."""

    if not isinstance(index, tuple):
        index = (index,)
    ellipsis_at = None
    for position, entry in enumerate(index):
        if entry is Ellipsis or (isinstance(entry, slice)
                                 and entry == Ellipsis):
            ellipsis_at = position
            break

    def _consumes(entry: Any) -> bool:
        return entry is not None and not (
            entry is Ellipsis or (isinstance(entry, slice) and entry == Ellipsis)
        )

    consumed = sum(1 for entry in index if _consumes(entry))
    fill = rank - consumed
    if fill < 0:
        raise UnsupportedOperand("too many indices for the tensor rank")
    if ellipsis_at is not None:
        index = (
            index[:ellipsis_at]
            + (slice(None),) * fill
            + index[ellipsis_at + 1:]
        )
    else:
        index = index + (slice(None),) * fill
    return list(index)


def _slice_window(entry: Any, size: int) -> tuple[int, int, int]:
    start, stop, step = entry.start, entry.stop, entry.step
    if step is None:
        step = 1
    if step <= 0:
        raise UnsupportedOperand("negative or zero slicing steps are not supported")
    start = 0 if start is None else int(start)
    stop = size if stop is None else int(stop)
    if start < 0:
        start = max(start + size, 0)
    else:
        start = min(start, size)
    if stop < 0:
        stop = max(stop + size, 0)
    else:
        stop = min(stop, size)
    if stop < start:
        stop = start
    length = (stop - start + step - 1) // step
    return start, length, step


def _register(trt: Any) -> None:
    @register_converter("getitem")
    def convert_getitem(ctx: ConversionContext, target: Any, args: Any,
                        kwargs: Any, name: str) -> Any:
        if len(args) != 2:
            raise ValueError(f"getitem {name!r} expects a tensor and an index")
        input, index = args
        # A converter may have produced several tensors (max/min with dim);
        # eager unpacks them with integer indexing, so pick the element.
        if isinstance(input, (list, tuple)):
            if not isinstance(index, int):
                raise UnsupportedOperand(
                    "multi-output tensors support integer indexing only"
                )
            return input[index]
        if isinstance(index, (list, tuple)) and any(
            isinstance(entry, (list, tuple)) for entry in index
        ):
            raise UnsupportedOperand("advanced indexing is not supported")
        shape = static_shape(input)
        rank = len(shape)
        entries = _normalize_index(index, rank)

        starts: list[int] = []
        sizes: list[int] = []
        strides: list[int] = []
        result_shape: list[int] = []

        cursor = 0
        for entry in entries:
            if entry is None:
                result_shape.append(1)
                continue
            if isinstance(entry, int):
                position = entry if entry >= 0 else entry + shape[cursor]
                if not 0 <= position < shape[cursor]:
                    raise UnsupportedOperand(
                        f"index {entry} out of range for dim of size "
                        f"{shape[cursor]}"
                    )
                starts.append(position)
                sizes.append(1)
                strides.append(1)
                # An integer index contributes no result axis.
                cursor += 1
                continue
            if isinstance(entry, slice):
                start, length, step = _slice_window(entry, shape[cursor])
                starts.append(start)
                sizes.append(length)
                strides.append(step)
                result_shape.append(length)
                cursor += 1
                continue
            raise UnsupportedOperand(
                f"index entries of type {type(entry).__name__} are not "
                "supported"
            )

        if not starts:
            # Every axis was dropped: keep one element for the slice layer.
            starts, sizes, strides = [0], [1], [1]

        layer = ctx.net.add_slice(input, starts, sizes, strides)
        layer.name = name
        out = layer.get_output(0)

        if tuple(result_shape) != static_shape(out):
            out = reshape_static(ctx, out, ctx.unique(f"{name}_adjust"),
                                 tuple(result_shape) or (1,))
        return out
