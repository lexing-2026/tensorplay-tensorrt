"""Reduction converters.

Sum, mean, product, max and min over one or more axes through a single
reduce layer.  A dropped (keepdim=False) axis is squeezed afterwards, which
needs the build-time shapes this backend pins.

``x.max(dim=...)``/``x.min(dim=...)`` produce a (values, indices) pair in
eager, so the method-call form lowers to a top-k layer of one element, which
emits both halves natively; the indices come back as int64 to match the
eager dtype.  ``x.max()`` without an axis reduces to a scalar tensor.
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import (
    axes_mask,
    positive_dims,
    reshape_static,
    static_shape,
)


def _reduce(
    ctx: ConversionContext,
    name: str,
    input: Any,
    op: Any,
    dim: Any,
    keepdim: Any,
) -> Any:
    rank = len(input.shape)
    if dim is None or (isinstance(dim, (tuple, list)) and len(dim) == 0):
        dims = list(range(rank))
    else:
        dims = positive_dims(dim, rank)
    keepdim = bool(keepdim)
    layer = ctx.net.add_reduce(input, op, axes_mask(dims), keepdim)
    layer.name = name
    return layer.get_output(0)


def _extremum_with_indices(
    ctx: ConversionContext,
    name: str,
    input: Any,
    topk_op: Any,
    dim: Any,
    keepdim: bool,
) -> tuple[Any, Any]:
    """(values, indices) through a one-element top-k along ``dim``."""

    from ..converter_utils import cast_tensor

    shape = static_shape(input)
    rank = len(shape)
    (axis,) = positive_dims(dim, rank)
    tensor = input
    if rank == 1:
        # Top-k needs rank >= 2; lift to a column and reduce rows.
        tensor = reshape_static(ctx, input, ctx.unique(f"{name}_lift"), (shape[0], 1))
        axis = 0

    layer = ctx.net.add_topk(tensor, topk_op, 1, 1 << axis)
    layer.name = name
    values = layer.get_output(0)
    indices = layer.get_output(1)
    indices = cast_tensor(ctx, indices, ctx.unique(f"{name}_indices_i64"),
                          "int64")
    if not keepdim:
        values = reshape_static(ctx, values, ctx.unique(f"{name}_values_squeeze"),
                                static_shape(values)[:-1])
        indices = reshape_static(ctx, indices,
                                 ctx.unique(f"{name}_indices_squeeze"),
                                 static_shape(indices)[:-1])
    if rank == 1:
        values = reshape_static(ctx, values, ctx.unique(f"{name}_values_1d"), (1,))
        indices = reshape_static(ctx, indices, ctx.unique(f"{name}_indices_1d"), (1,))
    return values, indices


def _register(trt: Any) -> None:
    def _bound(op: Any, positional_dim: bool) -> Any:
        """One converter closure per operation.

        Method calls carry ``dim``/``keepdim`` as keywords; function calls
        pass them positionally after the operand.
        """

        def convert(ctx: ConversionContext, target: Any, args: Any,
                    kwargs: Any, name: str) -> Any:
            if not args:
                raise ValueError(f"reduction {name!r} expects one operand")
            input = args[0]
            if positional_dim:
                dim = args[1] if len(args) > 1 else None
                keepdim = args[2] if len(args) > 2 else False
            else:
                dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
                keepdim = kwargs.get("keepdim",
                                     args[2] if len(args) > 2 else False)
            return _reduce(ctx, name, input, op, dim, keepdim)

        return convert

    def _extremum(op: Any, topk_op: Any, binary_op: Any) -> Any:
        """max/min under both call costumes.

        A method call reduces: without ``dim`` a scalar tensor, with ``dim``
        a (values, indices) pair.  A function call (two operands) is binary
        elementwise.  The captured target separates the two: a method name
        is a string, a function target is not.
        """

        from .elementwise import _binary

        elementwise_convert = _binary(binary_op)

        def convert(ctx: ConversionContext, target: Any, args: Any,
                    kwargs: Any, name: str) -> Any:
            if not isinstance(target, str):
                return elementwise_convert(ctx, target, args, kwargs, name)
            input = args[0]
            dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
            keepdim = bool(kwargs.get("keepdim",
                                      args[2] if len(args) > 2 else False))
            if dim is None:
                return _reduce(ctx, name, input, op, None, False)
            return _extremum_with_indices(ctx, name, input, topk_op, dim,
                                          keepdim)

        return convert

    register_converter("sum")(_bound(trt.ReduceOperation.SUM, positional_dim=False))
    register_converter("mean")(_bound(trt.ReduceOperation.AVG, positional_dim=False))
    register_converter("prod")(_bound(trt.ReduceOperation.PROD, positional_dim=False))
    register_converter("max")(_extremum(trt.ReduceOperation.MAX,
                                        trt.TopKOperation.MAX,
                                        trt.ElementWiseOperation.MAX))
    register_converter("min")(_extremum(trt.ReduceOperation.MIN,
                                        trt.TopKOperation.MIN,
                                        trt.ElementWiseOperation.MIN))

    register_converter("amax")(_bound(trt.ReduceOperation.MAX, positional_dim=True))
    register_converter("amin")(_bound(trt.ReduceOperation.MIN, positional_dim=True))
