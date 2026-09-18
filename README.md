# tensorplay-tensorrt

NVIDIA TensorRT compiler backend for [TensorPlay](https://github.com/lexing-2026/tensorplay).

Installing this package registers a backend under the
`tensorplay_compiler_backends` entry-point group, so the name `tensorrt`
becomes available to the TensorPlay compiler:

```python
import tensorplay as tp

model = ...                                  # an inference region
optimized = tp.compile(model, backend="tensorrt")
out = optimized(x)                           # engines build on first use
```

## How it lowers

The canonical captured graph is translated into TensorRT layers directly by
a node-by-node interpreter: each supported operation has one converter
registered by name, constants are frozen with explicit dtypes (a Python
number feeding a float tensor arrives as that tensor's float), and binary
operands are rank-aligned before the layer. An operation with no converter
leaves the region uncompiled rather than failing the compile. Engines are
content-addressed on disk (graph + shapes + settings + TensorRT build), so a
rebuild only happens when something that could change the binary changed.
Execution addresses buffers by raw device pointer: CUDA inputs stay in
device memory end to end, host regions stage through the device and return
host results, and launches land on TensorPlay's current stream.

## Scope

- Inference regions only. Training regions are wrapped ahead-of-time by the
  compiler frontend: forward through this backend, backward as traced.
- Static shapes: the optimization profile pins the sample shapes given at
  compile time; a different shape runs uncompiled at runtime.
- A failed engine build runs the region uncompiled instead of failing the
  compile, unless `pass_through_build_failures=True` is passed through
  `options`.

## Options

Passed through `tp.compile(..., backend="tensorrt", options={...})`:

| Option | Default | Meaning |
| --- | --- | --- |
| `precision` | `"fp32"` | `"fp32"` or `"fp16"` tactic selection |
| `workspace_bytes` | builder default | workspace pool ceiling |
| `cache_dir` | `~/.cache/tensorplay-tensorrt` | serialized-engine directory; `None` disables the disk cache |
| `pass_through_build_failures` | `False` | raise on engine-build failure instead of falling back |

## Install

```bash
pip install tensorplay tensorrt
pip install tensorplay-tensorrt
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

Engine and numeric tests need a CUDA device and the `tensorrt` package and
skip when either is absent.

## License

Apache-2.0.
