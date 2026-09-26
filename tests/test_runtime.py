"""The per-run swaps in runtime.py, checked without a card.

The one that matters for memory is ``fused_attention``: it sends grouped K/V to
cuDNN as they are, or widens them where cuDNN turns the call down, so the
acoustic attention lands in a kernel that streams instead of building the whole
matrix. Whether that kernel is actually chosen is a property of the machine and
was measured on one; what is checked here is everything that is not -- which
calls go where, that a refusal and only a refusal is widened, that the widening
is the same attention, that the call reaches upstream unchanged otherwise, and
that the module is put back.
"""

from __future__ import annotations

import types

import pytest

from yue2_comfy import runtime


class Fake:
    """Shape, rank and device: all ``_grouped`` and the widening ever read."""

    def __init__(self, shape, device="cuda"):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.device = types.SimpleNamespace(type=device)
        self.repeats = None

    def repeat_interleave(self, groups, dim=1):
        wider = Fake((self.shape[0], self.shape[1] * groups) + self.shape[2:],
                     self.device.type)
        wider.repeats = (groups, dim)
        return wider


def trio(heads=16, kv_heads=8, tokens=4, dim=128, device="cuda"):
    return (Fake((tokens, heads, dim), device),
            Fake((tokens, kv_heads, dim), device),
            Fake((tokens, kv_heads, dim), device))


def test_the_grouped_cuda_call_is_the_one_that_is_widened():
    assert runtime._grouped(*trio())


def test_nothing_else_is_touched():
    """Every other shape goes to upstream exactly as it arrived.

    A call this pack does not recognize is upstream's to reject, with upstream's
    own message about it.
    """
    assert not runtime._grouped(*trio(device="cpu"))
    assert not runtime._grouped(*trio(heads=8))
    assert not runtime._grouped(*trio(heads=12, kv_heads=8))
    assert not runtime._grouped(*trio(kv_heads=0))
    q, k, v = trio()
    assert not runtime._grouped(q, k, Fake((4, 4, 128)))
    assert not runtime._grouped(Fake((1, 4, 16, 128)), k, v)
    assert not runtime._grouped(object(), k, v)


REFUSAL = "No available kernel. Aborting execution."


def swapped(nar, recorder):
    """nar.attention replaced by ``recorder`` for a with-block, put back after."""

    class Swap:
        def __enter__(self):
            self.original = nar.attention
            nar.attention = recorder

        def __exit__(self, *exc):
            nar.attention = self.original
            return False

    return Swap()


def test_the_grouped_call_asks_cudnn_first_with_the_key_heads_as_they_are(monkeypatch):
    """No copy, and cuDNN alone allowed, so a refusal raises instead of landing in math."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch.nn.attention")
    nar = pytest.importorskip("yue2_comfy.vendor.yue2.nar")
    monkeypatch.setattr(runtime, "_REFUSED", set())
    seen = {}

    def recorder(q, k, v, **kwargs):
        seen.update(q=q, k=k, v=v, kwargs=kwargs, cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
                    efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                    math=torch.backends.cuda.math_sdp_enabled())
        return "attended"

    q, k, v = trio()
    with swapped(nar, recorder):
        with runtime.fused_attention():
            assert nar.attention is not recorder
            assert nar.attention(q, k, v, causal=True, backend="sdpa") == "attended"

    assert seen["q"] is q and seen["k"] is k and seen["v"] is v
    assert seen["kwargs"] == {"causal": True, "backend": "sdpa"}
    assert seen["cudnn"] and not seen["efficient"] and not seen["math"]
    assert not runtime._REFUSED


def test_a_refused_call_repeats_only_the_key_heads(monkeypatch):
    """Where cuDNN says no, K and V gain the group; Q and the keyword arguments arrive untouched."""
    nar = pytest.importorskip("yue2_comfy.vendor.yue2.nar")
    monkeypatch.setattr(runtime, "_REFUSED", set())
    seen = {}

    def recorder(q, k, v, **kwargs):
        if k.shape[1] == 8:
            raise RuntimeError(REFUSAL)
        seen.update(q=q, k=k, v=v, kwargs=kwargs)
        return "attended"

    q, k, v = trio()
    with swapped(nar, recorder):
        with runtime.fused_attention():
            assert nar.attention(q, k, v, causal=True, backend="sdpa") == "attended"

    assert seen["q"] is q
    assert seen["k"].shape == (4, 16, 128) and seen["k"].repeats == (2, 1)
    assert seen["v"].shape == (4, 16, 128) and seen["v"].repeats == (2, 1)
    assert seen["kwargs"] == {"causal": True, "backend": "sdpa"}
    assert len(runtime._REFUSED) == 1


def test_a_refusal_is_remembered_for_that_shape_only(monkeypatch):
    """The same call is not offered to cuDNN twice; a call of another shape still is."""
    pytest.importorskip("torch.nn.attention")
    nar = pytest.importorskip("yue2_comfy.vendor.yue2.nar")
    monkeypatch.setattr(runtime, "_REFUSED", set())
    offered = []

    def recorder(q, k, v, **kwargs):
        if k.shape[1] == 8:
            offered.append(q.shape[0])
            raise RuntimeError(REFUSAL)
        return "attended"

    with swapped(nar, recorder):
        with runtime.fused_attention():
            for tokens in (4, 4, 6, 4, 6):
                nar.attention(*trio(tokens=tokens), causal=False)
    assert offered == [4, 6]


def test_other_errors_are_not_taken_for_a_refusal(monkeypatch):
    """Running out of memory is a RuntimeError too; widening would only ask for more."""
    pytest.importorskip("torch.nn.attention")
    nar = pytest.importorskip("yue2_comfy.vendor.yue2.nar")
    monkeypatch.setattr(runtime, "_REFUSED", set())
    shapes = []

    def recorder(q, k, v, **kwargs):
        shapes.append(k.shape)
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with swapped(nar, recorder):
        with runtime.fused_attention():
            with pytest.raises(RuntimeError, match="out of memory"):
                nar.attention(*trio())
    assert shapes == [(4, 8, 128)]
    assert not runtime._REFUSED


def test_the_ungrouped_call_reaches_upstream_as_it_is():
    nar = pytest.importorskip("yue2_comfy.vendor.yue2.nar")
    seen = {}

    def recorder(q, k, v, **kwargs):
        seen.update(k=k, v=v)

    original = nar.attention
    nar.attention = recorder
    try:
        q, k, v = trio(device="cpu")
        with runtime.fused_attention():
            nar.attention(q, k, v)
    finally:
        nar.attention = original

    assert seen["k"] is k and seen["v"] is v


def test_upstream_is_put_back_even_when_the_call_raises():
    nar = pytest.importorskip("yue2_comfy.vendor.yue2.nar")
    original = nar.attention
    with pytest.raises(ZeroDivisionError):
        with runtime.fused_attention():
            assert nar.attention is not original
            raise ZeroDivisionError
    assert nar.attention is original


def test_upstream_still_rejects_what_it_rejected_before():
    """The widening must not swallow a shape error and turn it into a wrong song."""
    torch = pytest.importorskip("torch")
    nar = pytest.importorskip("yue2_comfy.vendor.yue2.nar")
    q = torch.zeros(4, 16, 8)
    bad = torch.zeros(4, 3, 8)
    with runtime.fused_attention():
        with pytest.raises(ValueError):
            nar.attention(q, bad, bad)


def test_widening_the_key_heads_is_the_same_attention():
    """The reason this is a memory change and not a quality one.

    Repeating K/V is what ``enable_gqa`` does internally, so the answer is the
    same up to the order the products are summed in. The song it makes is not
    byte-identical with the one the math kernel made, which is a different
    statement and is recorded in the changelog.
    """
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    q = torch.randn(1, 16, 12, 32, dtype=torch.float32)
    k = torch.randn(1, 8, 12, 32, dtype=torch.float32)
    v = torch.randn_like(k)
    grouped = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    widened = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1))
    assert torch.allclose(grouped, widened, atol=1e-6)


def test_the_backend_names_still_exist_on_this_torch():
    """A name that moves must fail loudly here, not silently cost 10 GiB."""
    attention = pytest.importorskip("torch.nn.attention")
    for name in runtime.FUSED_BACKENDS + (runtime.GROUPED_BACKEND,):
        assert hasattr(attention.SDPBackend, name), name


def test_the_ar_records_into_a_guarded_graph(monkeypatch):
    """Upstream's plain graph is swapped for one that records with CAPTURE_MODE; a cleared one stays cleared."""
    pytest.importorskip("torch")
    cuda_graph = pytest.importorskip("yue2_comfy.vendor.yue2.cuda_graph")

    class Upstream:
        def __init__(self, *args, attention_backend="auto", **kwargs):
            self.attention_backend, self.device = attention_backend, "cpu"
            self.graph = None

        def _capture(self):
            self.graph = "a plain CUDAGraph"

        def close(self):
            self.graph = None

    made = []

    def guarded():
        made.append(object())
        return made[-1]

    monkeypatch.setattr(cuda_graph, "GraphAR", Upstream)
    monkeypatch.setattr(runtime, "guarded_graph", guarded)
    with runtime.pinned_attention("sdpa"):
        graph = cuda_graph.GraphAR(object(), [[1]], 4)
        assert graph.graph is None and made == []
        graph._capture()
        assert graph.graph is made[0]
        graph.close()
        assert graph.graph is None
    assert cuda_graph.GraphAR is Upstream


def test_a_guarded_graph_takes_what_a_newer_torch_passes(monkeypatch):
    """Issue #8: torch 2.13 added ``check_input_liveness`` to the call; it goes through, the mode is still ours."""
    torch = pytest.importorskip("torch")
    calls = []

    class Newer:
        def capture_begin(self, pool=None, capture_error_mode="global", check_input_liveness=False):
            calls.append((pool, capture_error_mode, check_input_liveness))

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Newer)
    runtime._guarded_class.cache_clear()
    try:
        graph = runtime.guarded_graph()
        graph.capture_begin(capture_error_mode="global", check_input_liveness=True)
        graph.capture_begin("pool", capture_error_mode="relaxed")
    finally:
        runtime._guarded_class.cache_clear()
    mode = runtime.CAPTURE_MODE
    assert calls == [(None, mode, True), ("pool", mode, False)]


def test_another_thread_may_call_cuda_while_a_guarded_graph_records():
    """The crash of 2026-09-25: another thread's CUDA call in the middle of a capture.

    That user's was a memory query under ComfyUI's cudaMallocAsync allocator;
    here, where the allocator is torch's own, a fresh cudaMalloc stands in for
    it. Under torch's default mode the same call fails in that thread and spoils
    the capture with "operation failed due to a previous error during capture".
    """
    import threading

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA card")
    x = torch.ones(8, device="cuda")
    ready, done = threading.Event(), threading.Event()
    errors = []

    def other():
        ready.wait(10)
        try:
            torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
            torch.cuda.memory_stats()
        except Exception as error:
            errors.append(error)
        done.set()

    thread = threading.Thread(target=other)
    thread.start()
    graph = runtime.guarded_graph()
    with torch.cuda.graph(graph):
        y = x * 2
        ready.set()
        done.wait(10)
        z = y + 1
    graph.replay()
    torch.cuda.synchronize()
    thread.join()
    assert errors == []
    assert z.tolist() == [3.0] * 8
