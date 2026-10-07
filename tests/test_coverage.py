"""Numerical coverage tests for the tensorrt converter surface.

Every converter added for the coverage expansion gets at least one eager
alignment check here.  Each test compiles with an isolated cache directory so
a stale engine can never satisfy the build, and asserts the product really is
a TensorRT lowering (``_assert_tensorrt_used``) rather than an interpreter
fallback.  Grouping follows the converter domains: pointwise/cast, matmul and
linear, conv+bn+pool, softmax, reduce (including the max/min values+indices
pair), shape and indexing, where/clamp, layer_norm, fp16, and multi-output
regions.
"""

import pytest

import tensorplay as tp


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


def _assert_tensorrt_used(compiled) -> None:
    """The compile product must hold the engine runner, not the fallback."""

    tags = {
        getattr(entry, "_tensorplay_codegen", None)
        for entry in compiled._tensorplay_cache.values()
    }
    assert tags == {"tensorrt"}, f"expected tensorrt lowering, got {tags}"


def _run(model, *inputs, tmp_path, precision="fp32", atol=1e-4, rtol=1e-4):
    """Compile ``model`` with tensorrt, run on ``inputs``, check against eager."""

    compiled = tp.compile(
        model,
        backend="tensorrt",
        options={"cache_dir": str(tmp_path), "precision": precision},
    )
    got = compiled(*inputs)
    _assert_tensorrt_used(compiled)
    want = model(*inputs)
    assert tp.allclose(got, want, atol=atol, rtol=rtol), (
        f"tensorrt output {got} diverges from eager {want}"
    )
    return got


# --- pointwise / activation ------------------------------------------------


@gpu_and_trt
def test_activation_family_matches_eager(tmp_path):
    x = tp.randn(4, 8, device="cuda")

    _run(lambda a: tp.relu(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.gelu(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.gelu(a, approximate="tanh"), x, tmp_path=tmp_path)
    _run(lambda a: tp.silu(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.elu(a, alpha=0.5), x, tmp_path=tmp_path)
    _run(lambda a: tp.leaky_relu(a, negative_slope=0.1), x, tmp_path=tmp_path)
    _run(lambda a: tp.selu(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.softplus(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.hardtanh(a, -0.5, 0.5), x, tmp_path=tmp_path)
    _run(lambda a: tp.clamp(tp.relu6(a), 0, 6), x, tmp_path=tmp_path)
    _run(lambda a: tp.clamp(a, -0.5, 0.5), x, tmp_path=tmp_path)
    _run(lambda a: tp.clamp_min(a, -0.5), x, tmp_path=tmp_path)
    _run(lambda a: tp.clamp_max(a, 0.5), x, tmp_path=tmp_path)


@gpu_and_trt
def test_unary_family_matches_eager(tmp_path):
    x = tp.randn(4, 8, device="cuda")
    positive = tp.abs(x) + 1.0

    _run(lambda a: tp.abs(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.rsqrt(a), positive, tmp_path=tmp_path)
    _run(lambda a: tp.erf(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.sin(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.cos(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.ceil(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.round(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.reciprocal(a), positive, tmp_path=tmp_path)
    _run(lambda a: tp.sign(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.floor(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.sqrt(a), positive, tmp_path=tmp_path)
    _run(lambda a: tp.exp(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.log(a), positive, tmp_path=tmp_path)
    _run(lambda a: -a, x, tmp_path=tmp_path)
    _run(lambda a: tp.isnan(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.isinf(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.logical_not(a > 0), x, tmp_path=tmp_path)


@gpu_and_trt
def test_binary_and_comparison_match_eager(tmp_path):
    x = tp.randn(4, 8, device="cuda")
    y = tp.randn(4, 8, device="cuda")

    _run(lambda a, b: a + b, x, y, tmp_path=tmp_path)
    _run(lambda a, b: a - b, x, y, tmp_path=tmp_path)
    _run(lambda a, b: a * b, x, y, tmp_path=tmp_path)
    _run(lambda a, b: a / b, x, y, tmp_path=tmp_path)
    _run(lambda a: tp.pow(tp.abs(a) + 1.0, 2.0), x, tmp_path=tmp_path)
    _run(lambda a, b: a == b, x, y, tmp_path=tmp_path)
    _run(lambda a, b: a != b, x, y, tmp_path=tmp_path)
    _run(lambda a, b: a < b, x, y, tmp_path=tmp_path)
    _run(lambda a, b: a > b, x, y, tmp_path=tmp_path)
    _run(lambda a, b: a <= b, x, y, tmp_path=tmp_path)
    _run(lambda a, b: a >= b, x, y, tmp_path=tmp_path)


# --- matmul / linear -------------------------------------------------------


@gpu_and_trt
def test_matmul_forms_match_eager(tmp_path):
    a = tp.randn(3, 4, device="cuda")
    b = tp.randn(4, 5, device="cuda")
    v = tp.randn(4, device="cuda")
    batched_a = tp.randn(2, 3, 4, device="cuda")
    batched_b = tp.randn(2, 4, 5, device="cuda")

    _run(lambda x, y: tp.matmul(x, y), a, b, tmp_path=tmp_path)
    _run(lambda x, y: tp.matmul(x, y), v, b, tmp_path=tmp_path)
    _run(lambda x, y: tp.matmul(x, y), a, v, tmp_path=tmp_path)
    _run(lambda x, y: tp.matmul(x, y), v, v, tmp_path=tmp_path)
    _run(lambda x, y: tp.matmul(x, y), batched_a, batched_b, tmp_path=tmp_path)


@gpu_and_trt
def test_linear_matches_eager(tmp_path):
    x = tp.randn(3, 4, device="cuda")
    w = tp.randn(5, 4, device="cuda")
    bias = tp.randn(5, device="cuda")

    got = _run(lambda a, wt, b: tp.linear(a, wt, b), x, w, bias, tmp_path=tmp_path)
    assert tuple(got.shape) == (3, 5)


# --- conv + bn + pool ------------------------------------------------------


@gpu_and_trt
def test_conv_bn_pool_resnet_block_matches_eager(tmp_path):
    class Block(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = tp.nn.Conv2d(3, 8, 3, padding=1)
            self.bn = tp.nn.BatchNorm2d(8)
            self.fc = tp.nn.Linear(8, 4)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = tp.relu(x)
            x = tp.max_pool2d(x, 2)
            x = tp.adaptive_avg_pool2d(x, (1, 1))
            x = tp.flatten(x, 1)
            return self.fc(x)

    model = Block().cuda().eval()
    x = tp.randn(2, 3, 8, 8, device="cuda")
    compiled = tp.compile(
        model,
        backend="tensorrt",
        options={"cache_dir": str(tmp_path)},
    )
    with tp.no_grad():
        got = compiled(x)
        _assert_tensorrt_used(compiled)
        want = model(x)
    assert tp.allclose(got, want, atol=1e-3, rtol=1e-3)


# --- softmax ---------------------------------------------------------------


@gpu_and_trt
def test_softmax_and_log_softmax_match_eager(tmp_path):
    x = tp.randn(4, 8, device="cuda")

    _run(lambda a: tp.softmax(a, dim=-1), x, tmp_path=tmp_path)
    _run(lambda a: tp.log_softmax(a, dim=1), x, tmp_path=tmp_path)


# --- reduce ----------------------------------------------------------------


@gpu_and_trt
def test_reductions_match_eager(tmp_path):
    x = tp.randn(4, 8, device="cuda")

    _run(lambda a: a.sum(dim=1), x, tmp_path=tmp_path)
    _run(lambda a: a.sum(dim=1, keepdim=True), x, tmp_path=tmp_path)
    _run(lambda a: a.mean(dim=1), x, tmp_path=tmp_path)
    _run(lambda a: a.prod(dim=1), tp.abs(x) + 1.0, tmp_path=tmp_path)
    _run(lambda a: tp.amax(a, dim=1), x, tmp_path=tmp_path)
    _run(lambda a: tp.amin(a, dim=1), x, tmp_path=tmp_path)
    _run(lambda a: tp.amax(a, dim=1, keepdim=True), x, tmp_path=tmp_path)
    _run(lambda a: a.max(), x, tmp_path=tmp_path)
    _run(lambda a: a.min(), x, tmp_path=tmp_path)


@gpu_and_trt
def test_max_min_dim_return_values_and_indices(tmp_path):
    x = tp.randn(4, 8, device="cuda")

    got = _run(lambda a: a.max(dim=1)[0], x, tmp_path=tmp_path)
    assert tuple(got.shape) == (4,)

    got_idx = _run(lambda a: a.max(dim=1)[1], x, tmp_path=tmp_path)
    assert got_idx.dtype == tp.int64

    got = _run(lambda a: a.min(dim=1)[0], x, tmp_path=tmp_path)
    assert tuple(got.shape) == (4,)

    got = _run(lambda a: a.max(dim=1, keepdim=True)[0], x, tmp_path=tmp_path)
    assert tuple(got.shape) == (4, 1)


# --- shape / permutation / cat / indexing ----------------------------------


@gpu_and_trt
def test_shape_ops_match_eager(tmp_path):
    x = tp.randn(4, 8, device="cuda")

    _run(lambda a: a.view(4, 2, 4), x, tmp_path=tmp_path)
    _run(lambda a: tp.reshape(a, (4, 2, 4)), x, tmp_path=tmp_path)
    _run(lambda a: tp.flatten(a, 0, 1), x, tmp_path=tmp_path)
    _run(lambda a: a.unsqueeze(1).squeeze(1), x, tmp_path=tmp_path)
    _run(lambda a: a.unsqueeze(1), x, tmp_path=tmp_path)
    _run(lambda a: a.unsqueeze(0).expand(3, 4, 8), x, tmp_path=tmp_path)
    _run(lambda a: a.expand(3, 4, 8), x, tmp_path=tmp_path)
    _run(lambda a: a.permute(1, 0), x, tmp_path=tmp_path)
    _run(lambda a: a.transpose(0, 1), x, tmp_path=tmp_path)
    _run(lambda a: a.t(), x, tmp_path=tmp_path)
    _run(lambda a: a.contiguous(), x, tmp_path=tmp_path)
    _run(lambda a: a.clone(), x, tmp_path=tmp_path)
    _run(lambda a: a.detach(), x, tmp_path=tmp_path)
    _run(lambda a: tp.dropout(a, 0.5, train=False), x, tmp_path=tmp_path)
    _run(lambda a: tp.zeros_like(a), x, tmp_path=tmp_path)
    _run(lambda a: tp.ones_like(a), x, tmp_path=tmp_path)


@gpu_and_trt
def test_cat_matches_eager(tmp_path):
    a = tp.randn(2, 4, device="cuda")
    b = tp.randn(3, 4, device="cuda")
    c = tp.randn(5, 4, device="cuda")

    got = _run(lambda x, y, z: tp.cat([x, y, z], dim=0), a, b, c, tmp_path=tmp_path)
    assert tuple(got.shape) == (10, 4)


@gpu_and_trt
def test_basic_indexing_matches_eager(tmp_path):
    x = tp.randn(4, 8, device="cuda")

    _run(lambda a: a[1], x, tmp_path=tmp_path)
    _run(lambda a: a[1:3], x, tmp_path=tmp_path)
    _run(lambda a: a[None], x, tmp_path=tmp_path)
    _run(lambda a: a[:, None], x, tmp_path=tmp_path)
    _run(lambda a: a[..., 1], x, tmp_path=tmp_path)
    _run(lambda a: a[..., None], x, tmp_path=tmp_path)
    _run(lambda a: a[-1], x, tmp_path=tmp_path)
    _run(lambda a: a[::2], x, tmp_path=tmp_path)


# --- where / layer_norm / cast ----------------------------------------------


@gpu_and_trt
def test_where_matches_eager(tmp_path):
    cond = tp.rand(4, 8, device="cuda") > 0.5
    x = tp.randn(4, 8, device="cuda")

    _run(lambda c, a, b: tp.where(c, a, b), cond, x, -x, tmp_path=tmp_path)


@gpu_and_trt
def test_layer_norm_matches_eager(tmp_path):
    x = tp.randn(4, 8, device="cuda")
    w = tp.randn(8, device="cuda")
    b = tp.randn(8, device="cuda")

    _run(lambda a, wt, bs: tp.layer_norm(a, (8,), wt, bs, 1e-5), x, w, b, tmp_path=tmp_path)


@gpu_and_trt
def test_cast_methods_match_eager(tmp_path):
    x = tp.randn(4, 8, device="cuda")

    _run(lambda a: a.float(), x, tmp_path=tmp_path)
    _run(lambda a: a.half().float(), x, tmp_path=tmp_path)
    _run(lambda a: a.int().float(), x, tmp_path=tmp_path)
    _run(lambda a: a.long().float(), x, tmp_path=tmp_path)
    _run(lambda a: (a > 0).bool().float(), x, tmp_path=tmp_path)
    _run(lambda a: a.to(tp.float64).float(), x, tmp_path=tmp_path)


# --- fp16 ------------------------------------------------------------------


@gpu_and_trt
def test_fp16_chain_matches_eager(tmp_path):
    x = tp.randn(4, 8, device="cuda")

    _run(
        lambda a: tp.relu(a + 1.0) * 2.0,
        x,
        tmp_path=tmp_path,
        precision="fp16",
        atol=1e-2,
        rtol=1e-2,
    )


# --- multi-output region ----------------------------------------------------


@gpu_and_trt
def test_multi_output_region_returns_both_tensors(tmp_path):
    def model(a):
        return a + 1.0, a * 2.0

    x = tp.randn(4, 8, device="cuda")
    compiled = tp.compile(
        model,
        backend="tensorrt",
        options={"cache_dir": str(tmp_path)},
    )
    got = compiled(x)
    _assert_tensorrt_used(compiled)
    want = model(x)
    assert len(got) == 2
    for g, w in zip(got, want):
        assert tp.allclose(g, w, atol=1e-4, rtol=1e-4)