"""TensorRT compiler backend for TensorPlay.

Registering through the ``tensorplay_compiler_backends`` entry-point group
makes the backend discoverable by name: after installing this package,
``tensorplay.compile(model, backend="tensorrt")`` drives NVIDIA TensorRT.
"""

from .backend import backend
from .settings import CompilationSettings

__all__ = ["backend", "CompilationSettings"]
__version__ = "0.1.0"
