"""Converter surface.  Importing this package registers every converter.

Each module exposes ``_register(trt)``; the tensorrt module is imported once
here and handed to all of them, so adding a converter domain is one import
line.
"""

from __future__ import annotations

from .._trt import _trt as _trt_module


def _register_all() -> None:
    from . import (  # noqa: F401
        activation,
        cast,
        cat,
        condition,
        conv,
        elementwise,
        linear,
        matmul,
        normalization,
        permutation,
        pool,
        reduce,
        shape,
        slice,
        softmax,
        unary,
    )

    trt = _trt_module()
    elementwise._register(trt)
    unary._register(trt)
    activation._register(trt)
    matmul._register(trt)
    linear._register(trt)
    conv._register(trt)
    pool._register(trt)
    normalization._register(trt)
    softmax._register(trt)
    reduce._register(trt)
    shape._register(trt)
    permutation._register(trt)
    cat._register(trt)
    slice._register(trt)
    cast._register(trt)
    condition._register(trt)


_register_all()
