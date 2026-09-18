"""Converter surface.  Importing this package registers every converter.

Each module exposes ``_register(trt)``; the tensorrt module is imported once
here and handed to all of them, so adding a converter domain is one import
line.
"""

from __future__ import annotations

from .._trt import _trt as _trt_module


def _register_all() -> None:
    from . import activation, elementwise, unary  # noqa: F401

    trt = _trt_module()
    elementwise._register(trt)
    unary._register(trt)
    activation._register(trt)


_register_all()
