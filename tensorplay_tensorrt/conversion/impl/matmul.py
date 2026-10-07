"""Matrix-multiply converters.

A matrix product is one ``add_matrix_multiply`` layer once both operands are
rank-aligned: TRT batches over equal-rank leading dimensions and broadcasts
size-1 batch dimensions, so only rank differences need shuffles.  A rank-1
operand is a vector product; it is reshaped to rank 2 first so the build-time
shapes stay expressible, then the result is reshaped back.
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import (
    get_trt_tensor,
    reshape_static,
    static_shape,
)


def matrix_multiply(
    ctx: ConversionContext,
    name: str,
    lhs: Any,
    rhs: Any,
    lhs_op: Any = None,
    rhs_op: Any = None,
) -> Any:
    """One matmul layer with rank alignment, reusable by linear/gemm users."""

    trt = ctx.trt
    lhs = get_trt_tensor(ctx, lhs, ctx.unique(f"{name}_lhs"))
    rhs = get_trt_tensor(ctx, rhs, ctx.unique(f"{name}_rhs"))
    lhs_op = lhs_op if lhs_op is not None else trt.MatrixOperation.NONE
    rhs_op = rhs_op if rhs_op is not None else trt.MatrixOperation.NONE

    lhs_shape = static_shape(lhs)
    rhs_shape = static_shape(rhs)
    # VECTOR consumes the one free axis, so the batching rank loses a dim.
    lhs_batch = len(lhs_shape) - (1 if lhs_op == trt.MatrixOperation.VECTOR else 0)
    rhs_batch = len(rhs_shape) - (1 if rhs_op == trt.MatrixOperation.VECTOR else 0)
    if lhs_batch > rhs_batch:
        rhs = prepend_ones_static(ctx, rhs, ctx.unique(f"{name}_rhs_batch"),
                                  lhs_batch - rhs_batch)
    elif rhs_batch > lhs_batch:
        lhs = prepend_ones_static(ctx, lhs, ctx.unique(f"{name}_lhs_batch"),
                                  rhs_batch - lhs_batch)

    layer = ctx.net.add_matrix_multiply(lhs, lhs_op, rhs, rhs_op)
    layer.name = name
    return layer.get_output(0)


def prepend_ones_static(
    ctx: ConversionContext,
    tensor: Any,
    name: str,
    count: int,
) -> Any:
    if count <= 0:
        return tensor
    shape = (1,) * count + static_shape(tensor)
    return reshape_static(ctx, tensor, name, shape)


def _register(trt: Any) -> None:
    @register_converter("matmul", "mm")
    def convert_matmul(ctx: ConversionContext, target: Any, args: Any,
                       kwargs: Any, name: str) -> Any:
        if len(args) != 2:
            raise ValueError(f"matmul {name!r} expects two operands")
        lhs, rhs = args
        lhs = get_trt_tensor(ctx, lhs, ctx.unique(f"{name}_lhs"))
        rhs = get_trt_tensor(ctx, rhs, ctx.unique(f"{name}_rhs"))
        lhs_shape = static_shape(lhs)
        rhs_shape = static_shape(rhs)

        # Rank-1 operands: reshape to (1, n) or (n, 1), multiply, reshape the
        # product back to the vector rank the caller expects.
        if len(lhs_shape) == 1 or len(rhs_shape) == 1:
            lhs_vec = len(lhs_shape) == 1
            rhs_vec = len(rhs_shape) == 1
            if lhs_vec and rhs_vec:
                a2 = reshape_static(ctx, lhs, ctx.unique(f"{name}_lhs2"), (1, lhs_shape[0]))
                b2 = reshape_static(ctx, rhs, ctx.unique(f"{name}_rhs2"), (rhs_shape[0], 1))
                out = matrix_multiply(ctx, name, a2, b2)
                return reshape_static(ctx, out, ctx.unique(f"{name}_out"), (1,))
            if lhs_vec:
                a2 = reshape_static(ctx, lhs, ctx.unique(f"{name}_lhs2"), (1, lhs_shape[0]))
                out = matrix_multiply(ctx, name, a2, rhs)
                # (1, n) @ (..., n, m) -> (..., 1, m); drop the inserted row.
                out_shape = static_shape(out)
                return reshape_static(
                    ctx, out, ctx.unique(f"{name}_out"),
                    out_shape[:-2] + (out_shape[-1],),
                )
            b2 = reshape_static(ctx, rhs, ctx.unique(f"{name}_rhs2"),
                                (rhs_shape[0], 1))
            out = matrix_multiply(ctx, name, lhs, b2)
            # (..., n, m) @ (m, 1) -> (..., n, 1); drop the inserted column.
            out_shape = static_shape(out)
            return reshape_static(
                ctx, out, ctx.unique(f"{name}_out"),
                out_shape[:-1],
            )

        return matrix_multiply(ctx, name, lhs, rhs)
