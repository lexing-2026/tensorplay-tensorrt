"""Engine build, serialization, and the content-addressed disk cache.

The interpreter turns the captured graph into a network; the builder turns
the network into a serialized engine.  Engines are keyed by a hash of the
graph structure, the sample shapes, and the settings, so a rebuild only
happens when something that could change the binary changed.  Writes land in
a temporary file first and are renamed into place, so a crash mid-build
never poisons the cache with a partial engine.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Callable

from ..conversion import Interpreter
from ..conversion._trt import _trt

__all__ = ["EnginePlan", "build_engine"]

log = logging.getLogger(__name__)


def _graph_key(graph_module: Any, example_inputs: list[Any], settings: Any) -> str:
    trt = _trt()
    digest = hashlib.sha256()
    digest.update(str(graph_module.graph).encode())
    for sample in example_inputs:
        digest.update(f"|{tuple(int(d) for d in sample.shape)}:{sample.dtype}".encode())
    digest.update(f"|precision={settings.precision}".encode())
    digest.update(f"|workspace={settings.workspace_bytes}".encode())
    digest.update(f"|trt={trt.__version__}".encode())
    return digest.hexdigest()


def _engine_bytes(network: Any, settings: Any, trt: Any) -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    config = builder.create_builder_config()
    if settings.workspace_bytes is not None:
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(settings.workspace_bytes)
        )
    if settings.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            log.warning("fp16 requested but the platform has no fast fp16 path")
        config.set_flag(trt.BuilderFlag.FP16)

    # Static shapes in this release: one profile pinned to the sample shapes.
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(tensor.shape)
        profile.set_shape(tensor.name, shape, shape, shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(serialized)


def _load_or_build(settings: Any, key: str, build: Callable[[], bytes]) -> bytes:
    if settings.cache_dir is None:
        return build()
    os.makedirs(settings.cache_dir, exist_ok=True)
    path = os.path.join(settings.cache_dir, f"{key}.engine")
    if os.path.exists(path):
        log.debug("tensorrt engine cache hit: %s", path)
        with open(path, "rb") as handle:
            return handle.read()
    blob = build()
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(blob)
    os.replace(tmp, path)
    return blob


class EnginePlan:
    """A deserialized engine plus the metadata execution needs."""

    def __init__(
        self,
        engine: Any,
        input_names: list[str],
        output_names: list[str],
        io_dtypes: dict[str, Any],
        io_shapes: dict[str, tuple[int, ...]],
    ) -> None:
        self.engine = engine
        self.input_names = input_names
        self.output_names = output_names
        self.io_dtypes = io_dtypes
        self.io_shapes = io_shapes


def build_engine(
    graph_module: Any,
    example_inputs: list[Any],
    settings: Any,
) -> EnginePlan:
    """Translate the graph, build (or load), and deserialize the engine."""

    # Importing the impl package registers every converter.
    from ..conversion import impl  # noqa: F401
    from tensorplay import bool as tp_bool, float16, float32, int32, int64

    trt = _trt()
    dtype_map = {
        trt.DataType.FLOAT: float32,
        trt.DataType.HALF: float16,
        trt.DataType.INT32: int32,
        trt.DataType.INT64: int64,
        trt.DataType.BOOL: tp_bool,
    }

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    Interpreter(network, list(example_inputs), settings, trt).run(graph_module)

    key = _graph_key(graph_module, example_inputs, settings)
    blob = _load_or_build(settings, key, lambda: _engine_bytes(network, settings, trt))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(blob)
    if engine is None:
        raise RuntimeError("TensorRT engine deserialization failed")

    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    input_names = [
        name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    ]
    output_names = [
        name
        for name in names
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
    ]
    io_dtypes = {
        name: dtype_map[engine.get_tensor_dtype(name)]
        for name in input_names + output_names
    }
    io_shapes = {
        name: tuple(engine.get_tensor_shape(name))
        for name in input_names + output_names
    }
    return EnginePlan(
        engine=engine,
        input_names=input_names,
        output_names=output_names,
        io_dtypes=io_dtypes,
        io_shapes=io_shapes,
    )
