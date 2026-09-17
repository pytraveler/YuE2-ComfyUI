"""The per-run swaps in runtime.py, checked without a card.

The one that matters for memory is ``fused_attention``: it widens grouped K/V
so the acoustic attention lands in a kernel that streams instead of building
the whole matrix. Whether that kernel is actually chosen is a property of the
machine and was measured on one; what is checked here is everything that is not
-- which calls get widened, that the widening is the same attention, that the
call reaches upstream unchanged otherwise, and that the module is put back.
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


def test_the_widened_call_repeats_only_the_key_heads():
    """K and V gain the group; Q and the keyword arguments arrive untouched."""
    nar = pytest.importorskip("yue2_comfy.vendor.yue2.nar")
    seen = {}

    def recorder(q, k, v, **kwargs):
        seen.update(q=q, k=k, v=v, kwargs=kwargs)
        return "attended"

    original = nar.attention
    nar.attention = recorder
    try:
        q, k, v = trio()
        with runtime.fused_attention():
            assert nar.attention is not recorder
            assert nar.attention(q, k, v, causal=True, backend="sdpa") == "attended"
    finally:
        nar.attention = original

    assert seen["q"] is q
    assert seen["k"].shape == (4, 16, 128) and seen["k"].repeats == (2, 1)
    assert seen["v"].shape == (4, 16, 128) and seen["v"].repeats == (2, 1)
    assert seen["kwargs"] == {"causal": True, "backend": "sdpa"}


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
    for name in runtime.FUSED_BACKENDS:
        assert hasattr(attention.SDPBackend, name), name
