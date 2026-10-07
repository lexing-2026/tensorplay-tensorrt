"""Element-type converters.

Method casts (``float``, ``half``, ``int`` ...) and ``to`` land on one cast
layer each.  TensorPlay's ``double`` maps to TensorRT's single float: the
engine has no float64 path, so the region computes in float32 from the cast
onward.
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import cast_tensor, UnsupportedOperand

_CAST_METHOD_DTYPES = {
    "float": "float32",
    "double": "float32",
    "half": "float16",
    "int": "int32",
    "long": "int64",
    "bool": "bool",
}

# Stems the runtime dtype map can translate; anything else a ``to`` receives
# (a device, a memory format) is not an element-type request.
_KNOWN_DTYPE_STEMS = {
    "float16", "float32", "float64", "int8", "int32", "int64", "bool",
}


def _dtype_stem(value: Any) -> str:
    if isinstance(value, str):
        return value.rsplit(".", 1)[-1]
    return str(value).rsplit(".", 1)[-1]


def _register(trt: Any) -> None:
    @register_converter("to")
    def convert_to(ctx: ConversionContext, target: Any, args: Any,
                   kwargs: Any, name: str) -> Any:
        if not args:
            raise ValueError("to expects an input")
        input = args[0]
        if len(args) == 1:
            return input  # a device-only .to() changes nothing for TRT
        stem = _dtype_stem(args[1])
        if stem not in _KNOWN_DTYPE_STEMS:
            return input
        return cast_tensor(ctx, input, name, stem)

    for method, dtype_name in _CAST_METHOD_DTYPES.items():
        def _make_cast(dtype_name: str) -> Any:
            def convert(ctx: ConversionContext, target: Any, args: Any,
                        kwargs: Any, name: str) -> Any:
                if not args:
                    raise ValueError(f"{name!r} expects an input")
                return cast_tensor(ctx, args[0], name, dtype_name)

            return convert

        register_converter(method)(_make_cast(dtype_name))
