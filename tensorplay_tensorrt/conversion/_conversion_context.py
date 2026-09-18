"""State shared by every converter during one graph translation."""

from __future__ import annotations

from typing import Any

__all__ = ["ConversionContext"]


class ConversionContext:
    """The network under construction plus naming and settings state.

    One context exists per translation.  Converters receive it as their first
    argument and add layers to ``ctx.net``; ``unique`` mints layer names that
    cannot collide, because the same operation may appear many times in one
    region.
    """

    def __init__(self, net: Any, settings: Any, trt: Any) -> None:
        self.net = net
        self.settings = settings
        self.trt = trt
        self.counter = 0
        self.current_node_name: str | None = None

    def unique(self, stem: str) -> str:
        self.counter += 1
        return f"{stem}_{self.counter}"
