"""Compilation settings for the TensorRT backend.

The settings mirror what an engine build actually consumes: arithmetic
precision, workspace ceiling, and where serialized engines live.  Anything
the caller does not pin down is decided by the builder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["CompilationSettings", "default_cache_dir"]


def default_cache_dir() -> str:
    return os.environ.get(
        "TENSORPLAY_TENSORRT_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "tensorplay-tensorrt"),
    )


@dataclass(frozen=True)
class CompilationSettings:
    """Immutable knobs for one engine build."""

    #: ``"fp32"`` or ``"fp16"``.  fp16 lets the builder pick half-precision
    #: tactics; accuracy-sensitive layers keep their own precision rules.
    precision: str = "fp32"
    #: Workspace pool ceiling in bytes; ``None`` leaves the builder default.
    workspace_bytes: int | None = None
    #: Directory for serialized engines.  ``None`` disables the disk cache.
    cache_dir: str | None = field(default_factory=default_cache_dir)
    #: Raise on engine-build failure instead of running the region uncompiled.
    pass_through_build_failures: bool = False

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any]) -> "CompilationSettings":
        """Build settings from ``compile(..., backend="tensorrt", options=...)``.

        Accepts either an ``options`` mapping or flat keyword spelling; the
        two spellings may not mix for the same key, and unknown keys are
        rejected so a typo never silently changes nothing.
        """

        options_keys = {
            "precision",
            "workspace_bytes",
            "cache_dir",
            "pass_through_build_failures",
        }
        known = options_keys | {"options"}
        unknown = (set(kwargs) - known) | (
            set(kwargs.get("options") or ()) - options_keys
        )
        if unknown:
            raise TypeError(
                f"tensorrt backend received unsupported option(s): {sorted(unknown)}"
            )
        options = dict(kwargs.get("options") or {})
        payload: dict[str, Any] = {}
        for name in options_keys:
            in_options = name in options
            in_kwargs = name in kwargs
            if in_options and in_kwargs:
                raise TypeError(f"tensorrt backend option {name!r} given twice")
            if in_options:
                payload[name] = options.pop(name)
            elif in_kwargs:
                payload[name] = kwargs[name]
        return cls(**payload)


#: Historical spelling kept as an alias.
Settings = CompilationSettings
