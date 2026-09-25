"""The token loops' attention engines: which one a setting names, where its answer goes, and that it is right.

Speed and whether a seed repeats were measured on a card (see ``attention.py``);
what is checked here is the wiring -- an old 'cudnn' still runs, only the decode
step's own call is answered, everything is put back -- and, where there is a
card, that 'fast' answers what the math kernel answers.
"""

from __future__ import annotations

import sys
import types

import pytest

from yue2_comfy import attention
from yue2_comfy.constants import ATTENTION_CHOICES, DEFAULT_OPTIONS, LEGACY_ATTENTION


def test_the_default_still_sings_every_seed_as_before():
    assert DEFAULT_OPTIONS["attention_backend"] == "sdpa"
    assert ATTENTION_CHOICES == ("sdpa", "fast", "flash")


def test_an_old_cudnn_is_read_as_fast():
    assert LEGACY_ATTENTION == {"cudnn": "fast"}
    assert attention.resolve("cudnn") == "fast"
    assert attention.resolve("flash") == "flash"
    assert attention.resolve(None) == "sdpa"


def test_a_workflow_saved_with_cudnn_passes_validation():
    """ComfyUI would stop it at the queue with "Value not in list" otherwise."""
    from yue2_comfy.nodes import YuE2Options

    for value in ATTENTION_CHOICES + ("cudnn",):
        assert YuE2Options.VALIDATE_INPUTS(attention_backend=value) is True
    assert YuE2Options.VALIDATE_INPUTS() is True
    refused = YuE2Options.VALIDATE_INPUTS(attention_backend="xformers")
    assert isinstance(refused, str) and "xformers" in refused


def test_sdpa_needs_no_engine_and_a_stranger_is_refused():
    assert attention.engine_for("sdpa", "cuda") is None
    with pytest.raises(ValueError, match="sdpa, fast or flash"):
        attention.engine_for("math", "cuda")


def test_flash_without_the_package_says_what_to_do(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    with pytest.raises(ValueError) as refused:
        attention.engine_for("flash", "cuda")
    assert "flash-attn" in str(refused.value) and "'fast'" in str(refused.value)


def test_fast_on_a_torch_without_lengths_says_what_to_do(monkeypatch):
    monkeypatch.setattr(attention, "_efficient_takes_lengths", lambda: False)
    with pytest.raises(ValueError, match="'sdpa'"):
        attention.engine_for("fast", "cuda")


def test_below_ampere_fast_and_flash_point_to_sdpa(monkeypatch):
    """torch's efficient kernel has no bfloat16 build below sm80, and flash-attn does not run there."""
    monkeypatch.setattr(attention, "_efficient_takes_lengths", lambda: True)
    monkeypatch.setattr(attention, "_major", lambda device: 7)
    with pytest.raises(ValueError) as refused:
        attention.engine_for("fast", "cuda")
    assert "RTX 30" in str(refused.value) and "'sdpa'" in str(refused.value)
    with pytest.raises(ValueError, match="RTX 30"):
        attention.engine_for("fast", "cuda", "torch.bfloat16")
    assert attention.engine_for("fast", "cuda", "torch.float16").name == "fast"
    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace(flash_attn_with_kvcache=None))
    with pytest.raises(ValueError) as refused:
        attention.engine_for("flash", "cuda")
    assert "RTX 30" in str(refused.value) and "'sdpa'" in str(refused.value)
    assert "'fast'" not in str(refused.value)


def test_from_ampere_on_fast_runs_and_no_card_is_not_refused(monkeypatch):
    monkeypatch.setattr(attention, "_efficient_takes_lengths", lambda: True)
    for major in (8, 12, None):
        monkeypatch.setattr(attention, "_major", lambda device, major=major: major)
        assert attention.engine_for("fast", "cuda").name == "fast"


def test_the_card_is_asked_only_of_cuda():
    assert attention._major("cpu") is None
    assert attention._major("no such device") is None


class Shape:
    def __init__(self, *shape):
        self.shape = shape
        self.ndim = len(shape)


def test_only_the_decode_step_call_reaches_the_engine():
    """One query a branch over the whole cache, while a step is decoding; anything else goes to SDPA."""
    seen = []

    class Real:
        def scaled_dot_product_attention(self, *args, **kwargs):
            seen.append("sdpa")
            return "sdpa"

        def linear(self, *args):
            return "linear"

    class Engine:
        def attend(self, graph, query, key, value):
            seen.append(("engine", graph))
            return "engine"

    functional = attention.Functional(Real(), Engine())
    graph = types.SimpleNamespace(capacity=64)
    step = (Shape(2, 16, 1, 128), Shape(2, 8, 64, 128), Shape(2, 8, 64, 128))
    assert functional.linear() == "linear"
    assert functional.scaled_dot_product_attention(*step, attn_mask=None) == "sdpa"
    functional.decoding = graph
    assert functional.scaled_dot_product_attention(*step, attn_mask=None, enable_gqa=True) == "engine"
    prefill = (Shape(2, 16, 5, 128), Shape(2, 8, 64, 128), Shape(2, 8, 64, 128))
    assert functional.scaled_dot_product_attention(*prefill) == "sdpa"
    other = (Shape(2, 16, 1, 128), Shape(2, 8, 32, 128), Shape(2, 8, 32, 128))
    assert functional.scaled_dot_product_attention(*other) == "sdpa"
    assert seen == ["sdpa", ("engine", graph), "sdpa", "sdpa"]


def test_the_step_is_answered_by_the_engine_and_everything_is_put_back(monkeypatch):
    """Upstream's step runs as written; its one attention call goes to the engine; the module is restored."""
    pytest.importorskip("torch")
    cuda_graph = pytest.importorskip("yue2_comfy.vendor.yue2.cuda_graph")
    from yue2_comfy import runtime

    calls = []

    class Upstream:
        def __init__(self, model, prefixes, max_tokens, *, capture=True, attention_backend="auto",
                     fuse_projections=False):
            calls.append(("init", attention_backend))
            self.model, self.device, self.capacity, self.branches = model, "cpu", 64, 1
            self.attention_backend = attention_backend

        def _decode(self):
            return cuda_graph.F.scaled_dot_product_attention(
                Shape(1, 16, 1, 128), Shape(1, 8, 64, 128), Shape(1, 8, 64, 128), attn_mask=None)

    class Engine:
        name = "fast"

        def prepare(self, graph):
            calls.append("prepare")

        def begin(self, graph):
            calls.append("begin")

        def attend(self, graph, query, key, value):
            calls.append("attend")
            return "answered"

    monkeypatch.setattr(cuda_graph, "GraphAR", Upstream)
    monkeypatch.setattr(attention, "engine_for",
                        lambda name, device, dtype=None: Engine() if name == "fast" else None)
    functional = cuda_graph.F
    with runtime.pinned_attention("cudnn"):
        graph = cuda_graph.GraphAR(object(), [[1]], 4)
        assert graph.attention_backend == "fast"
        assert graph._decode() == "answered"
    assert calls == [("init", "sdpa"), "prepare", "begin", "attend"]
    assert cuda_graph.GraphAR is Upstream and cuda_graph.F is functional


def test_sdpa_leaves_the_step_alone(monkeypatch):
    pytest.importorskip("torch")
    cuda_graph = pytest.importorskip("yue2_comfy.vendor.yue2.cuda_graph")
    from yue2_comfy import runtime

    class Upstream:
        def __init__(self, *args, attention_backend="auto", **kwargs):
            self.attention_backend, self.device = attention_backend, "cpu"

        def _decode(self):
            return "upstream"

    monkeypatch.setattr(cuda_graph, "GraphAR", Upstream)
    functional = cuda_graph.F
    with runtime.pinned_attention("sdpa"):
        graph = cuda_graph.GraphAR(object(), [[1]], 4)
        assert cuda_graph.F is functional
        assert graph._decode() == "upstream" and graph.attention_backend == "sdpa"


def test_a_step_writes_into_the_buffers_made_before_the_capture():
    """No tensor made or dropped around a step: the async allocator warns about both inside a capture."""
    torch = pytest.importorskip("torch")
    config = types.SimpleNamespace(num_attention_heads=16, num_key_value_heads=8)
    graph = types.SimpleNamespace(capacity=160, branches=2, device=torch.device("cpu"),
                                  model=types.SimpleNamespace(config=config),
                                  positions=torch.tensor([9, 0]))
    split, flash = attention.Split(), attention.Flash(kernel=None)
    split.prepare(graph)
    flash.prepare(graph)
    held = (graph.split_seen, graph.split_empty, graph.flash_filled)
    for positions in ([9, 0], [25, 11]):
        graph.positions = torch.tensor(positions)
        split.begin(graph)
        flash.begin(graph)
        now = (graph.split_seen, graph.split_empty, graph.flash_filled)
        assert all(a is b for a, b in zip(now, held))
    assert graph.split_seen.view(2, attention.PARTS)[:, :3].tolist() == [[10, 10, 6], [10, 2, 1]]
    empty = graph.split_empty.view(2, attention.PARTS)
    assert empty[0, :3].tolist() == [0.0, 0.0, 0.0] and empty[0, 3] == float("-inf")
    assert empty[1, :2].tolist() == [0.0, 0.0] and empty[1, 2] == float("-inf")
    assert graph.flash_filled.tolist() == [26, 12]


def test_fast_answers_what_the_math_kernel_answers():
    """On a card: two branches at different fill, every piece boundary crossed, against float32."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not attention._efficient_takes_lengths():
        pytest.skip("needs a CUDA card and a torch whose efficient kernel takes lengths")
    import torch.nn.functional as F

    torch.manual_seed(3)
    branches, heads, kv_heads, dim, capacity = 2, 16, 8, 128, 1000
    config = types.SimpleNamespace(num_attention_heads=heads, num_key_value_heads=kv_heads)
    graph = types.SimpleNamespace(capacity=capacity, branches=branches, device=torch.device("cuda"),
                                  model=types.SimpleNamespace(config=config),
                                  positions=torch.tensor([700, 45], device="cuda"))
    engine = attention.Split()
    engine.prepare(graph)
    keys = torch.randn(branches, capacity, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    query = torch.randn(branches, 1, heads, dim, device="cuda", dtype=torch.bfloat16)
    engine.begin(graph)
    got = engine.attend(graph, query.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2))
    assert got.shape == (branches, heads, 1, dim)
    for branch, position in enumerate(graph.positions.tolist()):
        seen_k = keys[branch, :position + 1].float().transpose(0, 1).repeat_interleave(heads // kv_heads, 0)
        seen_v = values[branch, :position + 1].float().transpose(0, 1).repeat_interleave(heads // kv_heads, 0)
        want = F.scaled_dot_product_attention(query[branch].float().transpose(0, 1), seen_k, seen_v)
        error = (got[branch].float() - want).norm() / want.norm()
        assert error < 1e-2, (branch, float(error))
