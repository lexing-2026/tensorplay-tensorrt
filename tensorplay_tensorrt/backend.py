"""The ``tensorrt`` backend, selected by ``tensorplay.compile(backend="tensorrt")``.

The backend drives NVIDIA TensorRT: the canonical graph is translated into
TRT layers by the conversion interpreter, built into a serialized engine
(content-addressed on disk), and executed with raw device pointers on
TensorPlay's current stream.

An engine build that fails leaves the region on the interpreter instead of
failing the whole compile, unless ``pass_through_build_failures`` is set --
a build failure is a coverage gap, not a user error.  Only inference regions
lower here; training regions are wrapped ahead-of-time by the compiler
frontend (forward through this backend, backward as traced).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .runtime import build_engine
from .runtime.runner import Runner
from .settings import CompilationSettings

__all__ = ["backend"]

log = logging.getLogger(__name__)


def _backend(
    graph_module: Any,
    example_inputs: list[Any],
    **kwargs: Any,
) -> Callable[..., Any]:
    try:
        import tensorrt  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The tensorrt backend requires TensorRT. Install it with "
            "`pip install tensorrt`, then retry."
        ) from exc

    settings = CompilationSettings.from_kwargs(kwargs)

    try:
        plan = build_engine(graph_module, example_inputs, settings)
    except Exception:
        if settings.pass_through_build_failures:
            raise
        log.warning(
            "tensorrt engine build failed on this region; "
            "running the region uncompiled",
            exc_info=True,
        )
        return graph_module

    return Runner(plan, settings)


try:
    from tensorplay.compiler import BackendCapabilities, declares_capabilities

    backend = declares_capabilities(
        BackendCapabilities(
            inference_only=True,
            handles_training=False,
            optional_deps=("tensorrt",),
        )
    )(_backend)
except ImportError:
    # A core without the capability contract cannot read the declaration;
    # the backend stays importable and the contract gap surfaces when the
    # compiler resolves backends by name.
    backend = _backend
