"""Shared helpers for operation converters.

Every converter receives already-interpreted operands, but an operand may be
an ITensor (a value a producer layer emitted) or a literal that rode through
the frontend (a Python number).  :func:`get_trt_tensor` is the one door every
converter uses to turn either into an ITensor, and :func:`broadcast` is how
binary converters rank-align their operands before adding the layer.
"""

from __future__ import annotations

from typing import Any

from ._conversion_context import ConversionContext

__all__ = [
    "UnsupportedOperand",
    "broadcast",
    "create_constant",
    "get_trt_tensor",
    "prepend_ones",
    "trt_dtype",
]


class UnsupportedOperand(Exception):
    """An operand the converter layer cannot freeze or pass through."""


# TensorPlay dtype name -> TensorRT DataType member name.  float64 has no
# TensorRT equivalent and folds into float32.
_DTYPE_NAME_TO_TRT = {
    "float16": "HALF",
    "float32": "FLOAT",
    "float64": "FLOAT",
    "int8": "INT8",
    "int32": "INT32",
    "int64": "INT64",
    "bool": "BOOL",
}

_NUMPY_DTYPES = {
    "bool": "bool_",
    "int8": "int8",
    "int32": "int32",
    "int64": "int64",
    "float16": "float16",
    "float32": "float32",
}


def _dtype_name(dtype: Any) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def trt_dtype(trt: Any, dtype: Any) -> Any:
    """TensorPlay ``DType`` to the matching ``trt.DataType`` member."""

    trt_name = _DTYPE_NAME_TO_TRT.get(_dtype_name(dtype))
    if trt_name is None:
        raise ValueError(f"unsupported dtype for TensorRT: {dtype}")
    return getattr(trt.DataType, trt_name)


def create_constant(
    ctx: ConversionContext,
    value: Any,
    name: str,
    dtype: Any = None,
) -> Any:
    """Freeze ``value`` as a constant layer producing one ITensor.

    Scalars become shape-``(1,)`` constants -- elementwise layers broadcast a
    size-1 trailing dimension against the sibling operand.  ``dtype`` pins
    the element type; a Python ``int`` feeding a float tensor must arrive as
    that tensor's float, never as integer arithmetic.
    """

    import numpy as np

    if isinstance(value, np.ndarray):
        np_dtype = value.dtype
        weights = ctx.trt.Weights(np.ascontiguousarray(value))
        layer = ctx.net.add_constant(tuple(value.shape), weights)
        layer.name = name
        return layer.get_output(0)

    if dtype is not None:
        np_name = _NUMPY_DTYPES.get(_dtype_name(dtype))
        if np_name is None:
            np_dtype = np.float32
        else:
            np_dtype = getattr(np, np_name)
    elif isinstance(value, bool):
        np_dtype = np.bool_
    elif isinstance(value, int):
        np_dtype = np.int64
    else:
        np_dtype = np.float32

    if isinstance(value, (list, tuple)):
        array = np.array(value, dtype=np_dtype)
        weights = ctx.trt.Weights(array)
        layer = ctx.net.add_constant(tuple(array.shape), weights)
    else:
        weights = ctx.trt.Weights(np.array([value], dtype=np_dtype))
        layer = ctx.net.add_constant((1,), weights)
    layer.name = name
    return layer.get_output(0)


def get_trt_tensor(
    ctx: ConversionContext,
    value: Any,
    name: str,
    dtype: Any = None,
) -> Any:
    """Coerce one operand into an ITensor.

    An ITensor passes through; anything else (a Python number, an array) is
    frozen as a constant layer.  ``dtype`` comes from the sibling tensor
    operand when the call site has one.
    """

    if isinstance(value, (int, float, bool)):
        return create_constant(ctx, value, name, dtype)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return value
    raise UnsupportedOperand(f"cannot convert operand {value!r} to a TensorRT tensor")


def prepend_ones(ctx: ConversionContext, tensor: Any, name: str, count: int) -> Any:
    """Prepend ``count`` size-1 dimensions via a shuffle layer."""

    import numpy as np

    shape = (1,) * count + tuple(int(dim) for dim in tensor.shape)
    weights = ctx.trt.Weights(np.array(shape, dtype=np.int64))
    constant = ctx.net.add_constant((len(shape),), weights).get_output(0)
    layer = ctx.net.add_shuffle(tensor)
    layer.set_input(1, constant)
    layer.name = name
    return layer.get_output(0)


def broadcast(
    ctx: ConversionContext,
    a: Any,
    b: Any,
    a_name: str,
    b_name: str,
):
    """Rank-align two ITensors by prepending size-1 dimensions.

    Elementwise layers broadcast per NumPy rules only once the ranks agree;
    a scalar constant has rank 1 here and needs no further work.
    """

    diff = len(tuple(a.shape)) - len(tuple(b.shape))
    if diff > 0:
        b = prepend_ones(ctx, b, f"{b_name}_broadcast", diff)
    elif diff < 0:
        a = prepend_ones(ctx, a, f"{a_name}_broadcast", -diff)
    return a, b
