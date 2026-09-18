"""Single import point for the tensorrt bindings."""

from __future__ import annotations

from typing import Any

_MODULE: Any = None


def _trt() -> Any:
    global _MODULE
    if _MODULE is None:
        import tensorrt

        _MODULE = tensorrt
    return _MODULE
