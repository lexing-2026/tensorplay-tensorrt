"""Activation converters.

One ``add_activation`` layer per operation, with the operation's parameter
carried in alpha/beta.  Activations TensorRT has no enum for (silu) compose
from a sigmoid multiply; clamp is a max-then-min chain so missing bounds
need no layers.
"""

from __future__ import annotations

from typing import Any

from .._converter_registry import register_converter
from .._conversion_context import ConversionContext
from ..converter_utils import broadcast, get_trt_tensor


def _activation(op: Any) -> Any:
    def convert(
        ctx: ConversionContext,
        target: Any,
        args: Any,
        kwargs: Any,
        name: str,
    ) -> Any:
        if not args:
            raise ValueError(f"activation {name!r} expects one operand")
        x = get_trt_tensor(ctx, args[0], ctx.unique(f"{name}_x"))
        layer = ctx.net.add_activation(x, op)
        layer.name = name
        return layer.get_output(0)

    return convert


def _parameterized(op: Any, alpha: int, beta: int | None = None) -> Any:
    def convert(
        ctx: ConversionContext,
        target: Any,
        args: Any,
        kwargs: Any,
        name: str,
    ) -> Any:
        if not args:
            raise ValueError(f"activation {name!r} expects one operand")
        x = get_trt_tensor(ctx, args[0], ctx.unique(f"{name}_x"))
        layer = ctx.net.add_activation(x, op)
        layer.name = name
        if alpha >= 0:
            layer.alpha = float(args[alpha])
        if beta is not None:
            layer.beta = float(args[beta]) if len(args) > beta else 0.0
        return layer.get_output(0)

    return convert


def _pointwise(ctx: ConversionContext, name: str, op: Any, lhs: Any,
               rhs: Any) -> Any:
    lhs, rhs = broadcast(ctx, lhs, rhs, f"{name}_lhs", f"{name}_rhs")
    layer = ctx.net.add_elementwise(lhs, rhs, op)
    layer.name = name
    return layer.get_output(0)


def _register(trt: Any) -> None:
    register_converter("relu")(_activation(trt.ActivationType.RELU))
    register_converter("tanh")(_activation(trt.ActivationType.TANH))
    register_converter("sigmoid")(_activation(trt.ActivationType.SIGMOID))

    register_converter("gelu")(
        _gelu(trt.ActivationType.GELU_ERF, trt.ActivationType.GELU_TANH)
    )
    register_converter("elu")(_parameterized(trt.ActivationType.ELU, alpha=1))
    register_converter("leaky_relu")(
        _parameterized(trt.ActivationType.LEAKY_RELU, alpha=1)
    )
    register_converter("selu")(_activation(trt.ActivationType.SELU))
    register_converter("softplus")(
        _parameterized(trt.ActivationType.SOFTPLUS, alpha=1)
    )
    register_converter("hardtanh")(
        _parameterized(trt.ActivationType.CLIP, alpha=1, beta=2)
    )
    register_converter("relu6")(_clip_relu6(trt))
    register_converter("silu")(_silu(trt))

    # clamp: max with the lower bound, then min with the upper one; a missing
    # bound needs no layer.
    register_converter("clamp")(_clamp(trt))


def _gelu(erf_op: Any, tanh_op: Any) -> Any:
    def convert(
        ctx: ConversionContext,
        target: Any,
        args: Any,
        kwargs: Any,
        name: str,
    ) -> Any:
        if not args:
            raise ValueError(f"activation {name!r} expects one operand")
        x = get_trt_tensor(ctx, args[0], ctx.unique(f"{name}_x"))
        approximate = args[1] if len(args) > 1 else kwargs.get("approximate", "none")
        op = tanh_op if approximate == "tanh" else erf_op
        layer = ctx.net.add_activation(x, op)
        layer.name = name
        return layer.get_output(0)

    return convert


def _clip_relu6(trt: Any) -> Any:
    def convert(
        ctx: ConversionContext,
        target: Any,
        args: Any,
        kwargs: Any,
        name: str,
    ) -> Any:
        if not args:
            raise ValueError("relu6 expects one operand")
        x = get_trt_tensor(ctx, args[0], ctx.unique(f"{name}_x"))
        layer = ctx.net.add_activation(x, trt.ActivationType.CLIP)
        layer.alpha = 0.0
        layer.beta = 6.0
        layer.name = name
        return layer.get_output(0)

    return convert


def _silu(trt: Any) -> Any:
    def convert(
        ctx: ConversionContext,
        target: Any,
        args: Any,
        kwargs: Any,
        name: str,
    ) -> Any:
        if not args:
            raise ValueError("silu expects one operand")
        x = get_trt_tensor(ctx, args[0], ctx.unique(f"{name}_x"))
        gate = ctx.net.add_activation(x, trt.ActivationType.SIGMOID)
        gate.name = ctx.unique(f"{name}_gate")
        gate = gate.get_output(0)
        return _pointwise(ctx, name, trt.ElementWiseOperation.PROD, x, gate)

    return convert


def _clamp(trt: Any) -> Any:
    def convert(
        ctx: ConversionContext,
        target: Any,
        args: Any,
        kwargs: Any,
        name: str,
    ) -> Any:
        if not args:
            raise ValueError("clamp expects an input")
        out = get_trt_tensor(ctx, args[0], ctx.unique(f"{name}_x"))
        rest = args[1:]
        low = rest[0] if len(rest) > 0 else kwargs.get("min")
        high = rest[1] if len(rest) > 1 else kwargs.get("max")
        if low is not None:
            low_t = get_trt_tensor(ctx, low, ctx.unique(f"{name}_low"),
                                   dtype=out.dtype)
            out = _pointwise(ctx, ctx.unique(f"{name}_max"),
                             trt.ElementWiseOperation.MAX, out, low_t)
        if high is not None:
            high_t = get_trt_tensor(ctx, high, ctx.unique(f"{name}_high"),
                                    dtype=out.dtype)
            out = _pointwise(ctx, ctx.unique(f"{name}_min"),
                             trt.ElementWiseOperation.MIN, out, high_t)
        return out

    return convert
