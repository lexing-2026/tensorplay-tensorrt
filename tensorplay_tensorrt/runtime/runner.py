"""Engine execution for the TensorRT backend.

One :class:`Runner` wraps a deserialized engine.  Buffers are addressed by
raw device pointer: a CUDA input keeps its device memory end to end, a host
input is staged onto the device, and outputs are allocated fresh on every
call so no result can be served from a previous one.  Launches land on
TensorPlay's current stream so capture windows and stream discipline hold.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Runner"]


class Runner:
    """Executable wrapper around one built engine."""

    def __init__(self, plan: Any, settings: Any) -> None:
        import tensorplay

        self._plan = plan
        self._settings = settings
        self._context = plan.engine.create_execution_context()
        self._tp = tensorplay
        self._name = "tensorrt"
        # Visible through the compiled region's cache so callers (and tests)
        # can tell a TensorRT product from an interpreter fallback.
        self._tensorplay_codegen = "tensorrt"

    def _on_cuda(self, value: Any) -> bool:
        device = getattr(value, "device", None)
        return device is not None and device.is_cuda()

    def _prepare_input(self, value: Any) -> Any:
        """The tensor whose pointer feeds one engine input.

        The engine consumes device memory: a device tensor is used where it
        lies (made contiguous first, since the address is handed over raw); a
        host tensor is staged onto the device for the call.
        """

        tensor = value
        if not self._on_cuda(tensor):
            tensor = tensor.to("cuda")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor

    def __call__(self, *args: Any) -> Any:
        plan = self._plan
        context = self._context

        if len(args) != len(plan.input_names):
            raise TypeError(
                f"{self._name} region expects {len(plan.input_names)} "
                f"input(s), got {len(args)}"
            )

        all_host = all(not self._on_cuda(value) for value in args)

        staged: list[Any] = []
        for name, value in zip(plan.input_names, args):
            tensor = self._prepare_input(value)
            staged.append(tensor)
            context.set_tensor_address(name, tensor.data_ptr())

        outputs = []
        for name in plan.output_names:
            out = self._tp.empty(
                plan.io_shapes[name],
                dtype=plan.io_dtypes[name],
                device="cuda",
            )
            context.set_tensor_address(name, out.data_ptr())
            outputs.append(out)

        stream = self._tp.cuda.current_stream()
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT engine execution failed")

        # The engine consumed the staged host buffers asynchronously; the
        # Python objects must outlive the call, which they do here.  A region
        # that lived on the host returns to the host; otherwise results stay
        # on the device.
        if all_host:
            outputs = [out.cpu() for out in outputs]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
