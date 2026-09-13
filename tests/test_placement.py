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
    """RTX 5090, 2026-09-13: peak GiB per stage, the 6.76 GiB of weights included.

    'auto' weighs an estimate plus MARGIN against the free memory, so that sum
    has to reach every measured peak, or a card that looks big enough runs out.
    Nor may it overshoot by much: an estimate two gigabytes too large swaps
    halves on cards that never needed it. The token counts are the real ones of
    a 40-second and a 240-second song. Flow matching adds the prefill cache,
    which already exists when its estimate is made.
    """
    cases = (
        (p.ar_stage_bytes(94, 4096), 7.55),
        (p.ar_stage_bytes(562, 1000), 7.26),
        (p.prefill_bytes(1563), 7.46),
        (p.kv_bytes(1563) + p.solve_bytes(1563, 1000), 7.47),
        (p.ar_stage_bytes(346, 4096), 7.63),
        (p.ar_stage_bytes(2752, 6000), 8.71),
        (p.prefill_bytes(8753), 14.15),
        (p.kv_bytes(8753) + p.solve_bytes(8753, 6000), 14.96),
    )
    for estimate, peak in cases:
        need = (peak - 6.76) * GIB
        assert estimate + p.MARGIN >= need, (estimate / GIB, peak)
        assert estimate <= need + 2 * GIB, (estimate / GIB, peak)
