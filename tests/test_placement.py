"""Which half of the model goes where, checked without a card.

The moves are a pure function of the mode, the stage, where each half is and
how much memory is free, so every case a run can meet is checked here with
numbers instead of a GPU. The one thing that needs the real model -- that the
three groups cover all of it -- is checked against the vendored architecture on
the meta device, and skipped where torch is not installed.
"""

from __future__ import annotations

import itertools

import pytest

from yue2_comfy import constants
from yue2_comfy import placement as p

GIB = 1024 ** 3
SIZES = {p.AR: int(4.03 * GIB), p.NAR: int(2.63 * GIB)}
PLENTY = 100 * GIB


def where(ar, nar):
    return {p.AR: ar, p.NAR: nar}


def test_the_modes_are_the_choices_the_node_offers():
    assert tuple(constants.OFFLOAD_CHOICES) == p.MODES
    assert constants.DEFAULT_OPTIONS["offload"] == p.AUTO


def test_on_keeps_only_the_half_the_stage_needs():
    assert p.moves(p.ON, p.AR, where(p.CPU, p.CARD), SIZES, PLENTY, 0) == [
        (p.NAR, p.CPU), (p.AR, p.CARD)]
    assert p.moves(p.ON, p.NAR, where(p.CARD, p.CPU), SIZES, PLENTY, 0) == [
        (p.AR, p.CPU), (p.NAR, p.CARD)]


def test_on_clears_the_card_for_the_decode():
    assert p.moves(p.ON, None, where(p.CARD, p.CARD), SIZES, PLENTY, 0) == [
        (p.AR, p.CPU), (p.NAR, p.CPU)]


def test_off_brings_both_halves_whatever_is_free():
    assert p.moves(p.OFF, p.AR, where(p.CPU, p.CPU), SIZES, 0, 0) == [
        (p.AR, p.CARD), (p.NAR, p.CARD)]
    assert p.moves(p.OFF, p.NAR, where(p.CARD, p.CARD), SIZES, 0, 0) == []


def test_off_still_clears_the_card_for_a_decode_that_would_not_fit():
    """What the pack did before offload existed, and still does."""
    assert p.moves(p.OFF, None, where(p.CARD, p.CARD), SIZES, PLENTY, p.DECODE_BYTES) == []
    assert p.moves(p.OFF, None, where(p.CARD, p.CARD), SIZES, GIB, p.DECODE_BYTES) == [
        (p.AR, p.CPU), (p.NAR, p.CPU)]


def test_auto_leaves_a_card_with_room_alone():
    assert p.moves(p.AUTO, p.NAR, where(p.CARD, p.CARD), SIZES, PLENTY, 8 * GIB) == []
    assert p.moves(p.AUTO, p.NAR, where(p.CARD, p.CPU), SIZES, PLENTY, 8 * GIB) == [
        (p.NAR, p.CARD)]


def test_auto_moves_the_other_half_off_when_the_stage_would_not_fit():
    assert p.moves(p.AUTO, p.NAR, where(p.CARD, p.CPU), SIZES, 9 * GIB, 8 * GIB) == [
        (p.AR, p.CPU), (p.NAR, p.CARD)]


def test_auto_counts_the_half_it_has_to_bring():
    """Room for the work alone is not room: the half arrives before the work does."""
    work = 6 * GIB
    free = work + p.MARGIN + GIB
    assert p.moves(p.AUTO, p.NAR, where(p.CARD, p.CARD), SIZES, free, work) == []
    assert p.moves(p.AUTO, p.NAR, where(p.CARD, p.CPU), SIZES, free, work) == [
        (p.AR, p.CPU), (p.NAR, p.CARD)]


def test_auto_never_brings_a_half_back_for_its_own_sake():
    assert p.moves(p.AUTO, p.AR, where(p.CARD, p.CPU), SIZES, PLENTY, 0) == []
    assert p.moves(p.AUTO, None, where(p.CPU, p.CPU), SIZES, PLENTY, 0) == []


def test_memory_that_cannot_be_read_counts_as_room():
    assert p.moves(p.AUTO, p.AR, where(p.CPU, p.CARD), SIZES, None, 50 * GIB) == [
        (p.AR, p.CARD)]


def test_a_half_interrupted_mid_move_is_finished_rather_than_trusted():
    assert p.moves(p.ON, p.AR, where(p.MIXED, p.MIXED), SIZES, PLENTY, 0) == [
        (p.NAR, p.CPU), (p.AR, p.CARD)]


def test_an_unknown_mode_is_taken_as_auto():
    assert p.moves("sometimes", p.NAR, where(p.CARD, p.CPU), SIZES, 9 * GIB, 8 * GIB) == \
        p.moves(p.AUTO, p.NAR, where(p.CARD, p.CPU), SIZES, 9 * GIB, 8 * GIB)


def test_a_move_off_the_card_always_comes_before_a_move_onto_it():
    """Otherwise both halves sit on the card for a moment: the peak this exists to avoid.

    Checked for every case, together with two promises: the half a stage needs
    ends up on the card, and 'on' leaves the other one off it.
    """
    places = (p.CARD, p.CPU, p.MIXED)
    for mode, required, ar, nar, free in itertools.product(
            p.MODES, (p.AR, p.NAR, None), places, places, (None, 0, PLENTY)):
        case = (mode, required, ar, nar, free)
        planned = p.moves(mode, required, where(ar, nar), SIZES, free, GIB)
        kinds = [place for _half, place in planned]
        assert kinds == sorted(kinds, key=lambda place: place != p.CPU), case
        final = dict(where(ar, nar))
        final.update(dict(planned))
        if required is not None:
            assert final[required] == p.CARD, case
        if mode == p.ON:
            for half in p.HALVES:
                if half != required:
                    assert final[half] == p.CPU, case


def test_the_cache_estimate_is_the_cache_graph_decode_allocates():
    """GraphAR keeps two tensors a layer of [branches, capacity, kv_heads, head_dim]."""
    capacity, branches = 8800, 2
    per_tensor = branches * capacity * p.KV_HEADS * p.HEAD_DIM * p.ELEMENT_BYTES
    assert p.kv_bytes(capacity, branches) == 2 * p.LAYERS * per_tensor


def test_stand_in_models_are_left_alone():
    """The stage tests run on objects with no backbone and a device named by a string."""

    class Stand:
        lm = None
        device = "cpu"

    assert p.arrange(Stand(), p.AR, GIB, "the score") == []
    with p.acoustic(Stand()):
        pass


def test_the_groups_hold_the_released_model_exactly_once():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from yue2_comfy.vendor.yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM

    with torch.device("meta"):
        model = YuE2ForCausalLM(YuE2Config())
    found = p.groups(model)
    owner = {}
    for name, modules in found.items():
        for module in modules:
            for tensor in p._tensors(module):
                assert id(tensor) not in owner, (name, owner.get(id(tensor)))
                owner[id(tensor)] = name
    everything = list(model.parameters()) + list(model.buffers())
    assert {id(tensor) for tensor in everything} == set(owner)
    sizes = {name: sum(t.numel() for module in modules for t in p._tensors(module))
             * p.ELEMENT_BYTES / GIB for name, modules in found.items()}
    assert (round(sizes[p.AR], 2), round(sizes[p.NAR], 2), round(sizes[p.SHARED], 2)) == \
        (4.03, 2.63, 0.10)


def test_the_estimates_cover_the_peaks_measured_with_the_whole_model_on_the_card():
    """Peak GiB per stage on an RTX 5090, the 6.76 GiB of weights included.

    'auto' weighs an estimate plus MARGIN against the free memory, so that sum
    has to reach every measured peak, or a card that looks big enough runs out.
    Nor may it overshoot by much: an estimate two gigabytes too large swaps
    halves on cards that never needed it. The token counts are the real ones of
    the songs measured. Flow matching adds the prefill cache, which already
    exists when its estimate is made.

    The AR rows are from 2026-09-13 and stand. The acoustic rows were measured
    again on 2026-09-17, after ``runtime.fused_attention`` stopped the acoustic
    stage building whole attention matrices: a 40-second song (2197 tokens,
    1000 frames) and a 233-second one (8429 tokens, 5828 frames).
    """
    cases = (
        (p.ar_stage_bytes(94, 4096), 7.55),
        (p.ar_stage_bytes(562, 1000), 7.26),
        (p.ar_stage_bytes(346, 4096), 7.63),
        (p.ar_stage_bytes(2752, 6000), 8.71),
        (p.prefill_bytes(2197), 7.146),
        (p.kv_bytes(2197) + p.solve_bytes(2197, 1000), 7.100),
        (p.prefill_bytes(8429), 8.143),
        (p.kv_bytes(8429) + p.solve_bytes(8429, 5828), 8.068),
    )
    for estimate, peak in cases:
        need = (peak - 6.76) * GIB
        assert estimate + p.MARGIN >= need, (estimate / GIB, peak)
        assert estimate <= need + 2 * GIB, (estimate / GIB, peak)


def test_the_acoustic_estimates_land_close_to_the_work_measured():
    """The rows above leave two gigabytes of slack; these leave a tenth of one.

    Work here means what the stage allocated on top of what it inherited, read
    between the moves on 2026-09-17: for the prefill that includes the cache it
    builds and leaves behind, which is why its estimate carries kv_bytes. This
    is what the ladder for small cards is built on, so it is held tighter than
    the peaks above.
    """
    cases = (
        (p.prefill_bytes(2197), 0.350),
        (p.prefill_bytes(8429), 1.348),
        (p.solve_bytes(2197, 1000), 0.065),
        (p.solve_bytes(8429, 5828), 0.347),
    )
    for estimate, work in cases:
        assert estimate >= work * GIB, (estimate / GIB, work)
        assert estimate <= work * GIB + 0.1 * GIB, (estimate / GIB, work)


def test_the_acoustic_stage_replaces_the_offload_upstream_would_do_itself():
    """Upstream's ``_offload_ar`` reads a device with ``next(module.parameters())``.

    A packed ``mlp`` block -- see quantized.py -- holds buffers and no parameters
    at all, so that call raises on it. It is never reached, because this context
    replaces the function for the whole acoustic stage; this test is what keeps
    that true.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from yue2_comfy.vendor.yue2 import nar

    class Stand:
        lm = object()
        device = "cpu"

    engine, offload = nar.CachedNAR, nar._offload_ar
    with p.acoustic(Stand()):
        assert nar._offload_ar is not offload
        assert nar.CachedNAR is not engine
    assert (nar.CachedNAR, nar._offload_ar) == (engine, offload)


def _tiny(torch):
    """A model laid out the way ``groups`` reads one, a few kilobytes of it."""
    hidden = 8

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in p.AR_LAYER + p.NAR_LAYER:
                setattr(self, name, torch.nn.Linear(hidden, hidden, bias=False))

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList(Layer() for _ in range(2))
    model.model.embed_tokens = torch.nn.Embedding(4, hidden)
    model.model.norm = torch.nn.LayerNorm(hidden)
    model.lm_head = torch.nn.Linear(hidden, 4, bias=False)
    for name in ("llm2vae", "vae2llm", "time_embedder", "latent_pos_embed"):
        setattr(model, name, torch.nn.Linear(hidden, hidden))
    model.requires_grad_(False)
    return model


def _storages(keeper, half):
    return [p._held(owner, name, parameter).data_ptr()
            for owner, name, parameter in p._slots(keeper.groups[half])]


def test_a_file_backed_half_comes_back_to_the_tensors_it_left():
    """Off the card is the old storage put back: nothing copied, nothing allocated.

    The meta device stands in for the card, which is enough to follow every
    tensor there and back without one.
    """
    torch = pytest.importorskip("torch")
    model = p.load(_tiny(torch), torch.device("meta"), file_backed=True)
    keeper = p.Placement(model, torch.device("meta"))
    before = _storages(keeper, p.NAR)
    _, moved, copied = keeper._move(p.NAR, p.CARD)
    assert moved == copied == keeper.sizes[p.NAR]
    assert keeper.where(p.NAR) == p.CARD and keeper.where(p.AR) == p.CPU
    _, moved, copied = keeper._move(p.NAR, p.CPU)
    assert (moved, copied) == (keeper.sizes[p.NAR], 0)
    assert _storages(keeper, p.NAR) == before


def test_a_half_that_is_not_file_backed_leaves_no_copy_behind():
    """A copy kept of weights the loader built itself would be the model held twice."""
    torch = pytest.importorskip("torch")
    model = p.load(_tiny(torch), torch.device("meta"))
    p.Placement(model, torch.device("meta"))._move(p.NAR, p.CARD)
    assert not any(module.__dict__.get("_yue2_homes") for module in model.modules())


def test_a_move_counts_only_what_was_not_already_there():
    """The log says what moved, so a half already in place moves nothing."""
    torch = pytest.importorskip("torch")
    model = p.load(_tiny(torch), torch.device("meta"), file_backed=True)
    keeper = p.Placement(model, torch.device("meta"))
    assert keeper._move(p.AR, p.CPU)[1:] == (0, 0)
    keeper._move(p.AR, p.CARD)
    assert keeper._move(p.AR, p.CARD)[1:] == (0, 0)


def test_the_pinned_route_carries_the_bytes_exactly(monkeypatch):
    """Tiny buffers, so that one tensor spans several of them and both are reused."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA card")
    monkeypatch.setattr(p, "STAGE_BYTES", 4096)
    torch.manual_seed(0)
    tensors = [torch.randn(3001, dtype=torch.bfloat16),
               torch.randn(64, 48).t(),
               torch.randint(-127, 127, (10000,), dtype=torch.int8),
               torch.empty(0),
               torch.randn(5, 7, 3, dtype=torch.float32)]
    copies = p._to_card(tensors, torch.device("cuda"))
    for tensor, copy in zip(tensors, copies):
        assert copy.device.type == "cuda"
        assert copy.dtype == tensor.dtype and copy.shape == tensor.shape
        assert torch.equal(copy.cpu(), tensor)


def test_a_move_outside_inference_mode_may_follow_the_first_one_inside_it(monkeypatch):
    """The buffers are made on the first move; made in inference mode, no later move outside it could write them."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA card")
    monkeypatch.setattr(p, "_STAGING", {})
    monkeypatch.setattr(p, "STAGE_BYTES", 4096)
    tensor = torch.randn(3001)
    with torch.inference_mode():
        p._to_card([tensor], torch.device("cuda"))
    copy = p._to_card([tensor], torch.device("cuda"))[0]
    assert torch.equal(copy.cpu(), tensor)


def test_a_half_on_a_real_card_comes_back_without_a_copy():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA card")
    card = torch.device("cuda")
    for backed in (True, False):
        model = p.load(_tiny(torch), card, file_backed=backed)
        keeper = p.Placement(model, card)
        values = [t.clone() for t in model.parameters()]
        identities = [id(t) for t in model.parameters()]
        keeper._move(p.AR, p.CARD)
        assert keeper.where(p.AR) == p.CARD
        _, moved, copied = keeper._move(p.AR, p.CPU)
        assert copied == (0 if backed else moved)
        assert keeper.where(p.AR) == p.CPU
        assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(model.parameters(), values))
        assert [id(t) for t in model.parameters()] == identities


def test_the_decode_is_budgeted_by_what_it_holds_not_what_it_allocates():
    """Held on 2026-09-18: 4.0-4.5 GiB in the async pool, 5.07 native with a trim per
    tile, against 3.15 allocated; 1.79-1.82 for the low_vram tiles. The old 3 GiB
    left a 3070 Ti's NAR half on the card and the card at its ceiling."""
    assert p.decode_bytes(False) + p.MARGIN >= 5.07 * GIB
    assert p.decode_bytes(True) + p.MARGIN >= 1.82 * GIB
    assert p.decode_bytes(True) < p.decode_bytes(False)


def test_the_card_that_hit_the_ceiling_now_clears_the_nar_half_for_the_decode():
    """His numbers: the AR half already off, 4.05 GiB free with the NAR half on."""
    moved = p.moves(p.AUTO, None, where(p.CPU, p.CARD), SIZES, int(4.05 * GIB),
                    p.decode_bytes(False))
    assert moved == [(p.NAR, p.CPU)]


def test_the_weights_weight_norm_computes_do_not_outlive_the_decode():
    """They are recomputed before every call, so dropping them changes no sample."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from yue2_comfy import generate
    from yue2_comfy.vendor.yue2.modeling_vae import WNConv1d

    torch.manual_seed(0)
    block = torch.nn.Sequential(WNConv1d(4, 4, 3, padding=1), WNConv1d(4, 2, 3, padding=1))
    signal = torch.randn(1, 4, 16)
    first = block(signal)
    assert all("weight" in conv.__dict__ for conv in block)

    generate._forget_computed_weights(block)

    assert not any("weight" in conv.__dict__ for conv in block)
    assert torch.equal(block(signal), first)


def test_off_keeps_no_copy_of_the_file_behind():
    """Under 'off' a home would keep the checkpoint mapped and charged for nothing."""
    torch = pytest.importorskip("torch")
    model = p.load(_tiny(torch), torch.device("meta"), file_backed=True)
    keeper = p.Placement(model, torch.device("meta"))
    keeper._move(p.NAR, p.CARD, homes=False)
    assert not any(module.__dict__.get("_yue2_homes") for module in model.modules())

    keeper._move(p.AR, p.CARD)
    assert any(module.__dict__.get("_yue2_homes") for module in model.modules())
    keeper._move(p.AR, p.CPU)
    keeper._move(p.AR, p.CARD, homes=False)
    assert not any(module.__dict__.get("_yue2_homes") for module in model.modules())


def test_the_decode_trims_every_tile_and_forgets_the_computed_weights(monkeypatch):
    """What _decode has to call, checked on a stand-in decoder that says it is on a card."""
    torch = pytest.importorskip("torch")
    from yue2_comfy import generate

    trims, forgotten = [], []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: trims.append(1))
    monkeypatch.setattr(generate, "_forget_computed_weights", lambda vae: forgotten.append(vae))

    class Device:
        type = "cuda"

    class Weight:
        device = Device()

    class Config:
        decode_core_frames = 4
        decode_halo_frames = 1

    class Decoder:
        config = Config()

        def to(self, device):
            return self

        def parameters(self):
            return iter([Weight()])

        def decode_tiled(self, z, core_frames, halo_frames, output_device, on_progress):
            for done in range(3):
                on_progress(done + 1, 3)
            return torch.zeros(1, 2, 8)

    class Models:
        lm = None
        vae = Decoder()
        device = torch.device("cpu")
        low_vram = False

    generate._decode(Models, torch.zeros(12, 64), None, None, {})
    assert len(trims) >= 3
    assert forgotten == [Models.vae]
