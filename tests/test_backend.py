"""Tests for the tensorplay-tensorrt backend.

Structure tests (registration, capabilities, settings parsing, fallback on
build failure) run everywhere.  Numeric and engine tests need a GPU plus the
``tensorrt`` package and skip when either is absent.
"""

import pytest

import tensorplay as tp
from tensorplay.compiler import (
    get_backend_capabilities,
    list_backends,
)


def _has_trt():
    try:
        import tensorrt  # noqa: F401

        return True
    except ImportError:
        return False


gpu_and_trt = pytest.mark.skipif(
    not (_has_trt() and tp.cuda.is_available()),
    reason="needs tensorrt and a CUDA device",
)


# --- structure ---------------------------------------------------------------


def test_backend_registered_through_entry_point():
    from importlib.metadata import entry_points

    eps = entry_points(group="tensorplay_compiler_backends")
    names = {ep.name for ep in eps}
    assert "tensorrt" in names
    assert "tensorrt" in list_backends(include_unavailable=True)


def test_capabilities_declare_inference_only_and_dep():
    caps = get_backend_capabilities("tensorrt")
    assert caps is not None
    assert caps.inference_only is True
    assert caps.handles_training is False
    assert "tensorrt" in caps.optional_deps


def test_settings_reject_unknown_options():
    from tensorplay_tensorrt.settings import Settings

    with pytest.raises(TypeError):
        Settings.from_kwargs({"options": {"precission": "fp16"}})
    with pytest.raises(TypeError):
        Settings.from_kwargs({"bogus": 1})
    settings = Settings.from_kwargs({"options": {"precision": "fp16"}})
    assert settings.precision == "fp16"


def test_missing_package_error_names_the_install():
    from tensorplay_tensorrt import backend as backend_fn

    if _has_trt():
        pytest.skip("tensorrt installed; error path unexercised")
    with pytest.raises(RuntimeError, match="pip install tensorrt"):
        backend_fn(None, [])


# --- end-to-end --------------------------------------------------------------


def _model(a, b):
    return tp.sigmoid(tp.tanh(a + b) * (a * b)) - a


def _assert_tensorrt_used(compiled) -> None:
    """The compile product must hold the engine runner, not the fallback."""

    tags = {
        getattr(entry, "_tensorplay_codegen", None)
        for entry in compiled._tensorplay_cache.values()
    }
    assert tags == {"tensorrt"}, f"expected tensorrt lowering, got {tags}"


@gpu_and_trt
def test_pointwise_chain_matches_eager_on_cuda():
    compiled = tp.compile(_model, backend="tensorrt")
    x = tp.randn(16, 16, device="cuda")
    y = tp.randn(16, 16, device="cuda")
    got = compiled(x, y)
    _assert_tensorrt_used(compiled)
    want = _model(x, y)
    assert tp.allclose(got, want, atol=1e-4, rtol=1e-4)


@gpu_and_trt
def test_host_region_runs_and_returns_host_results():
    def model(a):
        return tp.relu(a) * 2.0 + 1.0

    compiled = tp.compile(model, backend="tensorrt")
    x = tp.rand(32)
    got = compiled(x)
    _assert_tensorrt_used(compiled)
    want = model(x)
    assert tp.allclose(got, want, atol=1e-5, rtol=1e-5)


@gpu_and_trt
def test_engine_cache_reuses_serialized_engine(tmp_path):
    def model(a):
        return tp.relu(a) + 0.5

    first = tp.compile(model, backend="tensorrt", options={"cache_dir": str(tmp_path)})
    x = tp.rand(8, device="cuda")
    first(x)
    _assert_tensorrt_used(first)
    engines = list(tmp_path.glob("*.engine"))
    assert len(engines) == 1

    second = tp.compile(model, backend="tensorrt", options={"cache_dir": str(tmp_path)})
    assert tp.allclose(second(x), model(x), atol=1e-5, rtol=1e-5)
    _assert_tensorrt_used(second)
    assert len(list(tmp_path.glob("*.engine"))) == 1


@gpu_and_trt
def test_unsupported_region_falls_back_to_interpreter(monkeypatch, tmp_path):
    import importlib

    # The package attribute ``backend`` is the backend function itself; the
    # module must be fetched through the import system to patch its globals.
    backend_mod = importlib.import_module("tensorplay_tensorrt.backend")

    def model(a):
        return tp.exp(a)

    x = tp.randn(8, device="cuda")
    eager = model(x)

    def broken_build(*args, **kwargs):
        raise RuntimeError("no engine for you")

    monkeypatch.setattr(backend_mod, "build_engine", broken_build)
    compiled = tp.compile(
        model,
        backend="tensorrt",
        options={"cache_dir": str(tmp_path)},
    )
    assert tp.allclose(compiled(x), eager, atol=1e-5, rtol=1e-5)
