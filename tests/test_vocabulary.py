"""The window cut into the two vocabulary tables, checked without a card.

What has to hold for the song to come out the same is arithmetic, not luck:
the window must cover every row the phase's mask leaves alive plus the token
the graph starts from, the head must answer at full width so the sampler can
mask it as before, and what is outside the window must be finite so that CFG
does not turn it into NaN. All four are checked here; the memory it saves was
measured on a card and is recorded in the module.
"""

from __future__ import annotations

import pytest

from yue2_comfy.vendor.yue2.protocol import (ABC_END, CODEC_OFFSET, CODEC_SIZE,
                                             EOD, MUSIC_END, MUSIC_START, VOCAB_SIZE)

vocabulary = pytest.importorskip("yue2_comfy.vocabulary")
torch = pytest.importorskip("torch")

HIDDEN = 16


def table(rows=64, hidden=HIDDEN):
    """A small stand-in for a 184704-row table, on the CPU."""
    torch.manual_seed(3)
    inner = torch.nn.Embedding(rows, hidden)
    return inner


def test_the_song_window_holds_every_row_the_mask_leaves_alive():
    start, stop = vocabulary.window(vocabulary.MUSIC)
    assert start <= MUSIC_END < stop
    assert start <= CODEC_OFFSET and CODEC_OFFSET + CODEC_SIZE <= stop


def test_the_score_window_holds_the_text_vocabulary_and_its_end():
    start, stop = vocabulary.window(vocabulary.ABC)
    assert start == 0 and stop > EOD
    assert stop > ABC_END


def test_the_window_stretches_to_the_token_the_graph_starts_from():
    """MUSIC_START sits one row below the song's mask, and GraphAR embeds it."""
    bare = vocabulary.window(vocabulary.MUSIC)
    start, stop = vocabulary.window(vocabulary.MUSIC, [MUSIC_START])
    assert start <= MUSIC_START < stop
    assert start < bare[0]


def test_the_window_never_runs_past_the_vocabulary():
    for phase in (vocabulary.ABC, vocabulary.MUSIC):
        start, stop = vocabulary.window(phase, [VOCAB_SIZE - 1])
        assert 0 <= start < stop <= VOCAB_SIZE


def test_a_windowed_lookup_is_the_same_row():
    inner = table()
    sliced = vocabulary.SlicedEmbedding(inner, torch.device("cpu"))
    sliced.retune(8, 24)
    sliced.windowed = True
    ids = torch.tensor([[9, 23, 8]])
    assert torch.equal(sliced(ids), inner(ids))


def test_an_unwindowed_lookup_reads_the_whole_table():
    inner = table()
    sliced = vocabulary.SlicedEmbedding(inner, torch.device("cpu"))
    sliced.retune(8, 24)
    ids = torch.tensor([[0, 63, 40]])
    assert torch.equal(sliced(ids), inner(ids))


def test_the_head_answers_at_full_width_with_the_window_in_place():
    inner = torch.nn.Linear(HIDDEN, 64, bias=False)
    sliced = vocabulary.SlicedHead(inner, torch.device("cpu"))
    sliced.retune(8, 24)
    sliced.windowed = True
    hidden = torch.randn(2, 1, HIDDEN)
    answer = sliced(hidden)
    assert answer.shape == (2, 1, VOCAB_SIZE)
    assert torch.equal(answer[..., 8:24], torch.nn.functional.linear(hidden, inner.weight[8:24]))


def test_outside_the_window_the_head_is_finite():
    """CFG combines two branches before the mask, and -inf minus -inf is NaN."""
    inner = torch.nn.Linear(HIDDEN, 64, bias=False)
    sliced = vocabulary.SlicedHead(inner, torch.device("cpu"))
    sliced.retune(8, 24)
    sliced.windowed = True
    answer = sliced(torch.randn(2, 1, HIDDEN))
    outside = torch.cat([answer[..., :8], answer[..., 24:]], dim=-1)
    assert torch.isfinite(outside).all()
    guided = answer[:1] + 3.0 * (answer[:1] - answer[1:])
    assert torch.isfinite(guided).all()


def test_releasing_the_window_lets_go_of_its_memory():
    """A view of the window would keep the whole allocation alive."""
    inner = table(rows=64)
    sliced = vocabulary.SlicedEmbedding(inner, torch.device("cpu"))
    sliced.retune(8, 56)
    assert sliced.rows.shape[0] == 48
    sliced.release()
    assert sliced.rows.shape[0] == 1
    assert sliced.rows.untyped_storage().size() == sliced.rows.numel() * sliced.rows.element_size()


def test_the_table_is_not_reachable_as_a_parameter():
    """generate_tokens reads the model's device off the first parameter."""
    sliced = vocabulary.SlicedEmbedding(table(), torch.device("cpu"))
    assert not list(sliced.parameters())
    assert sliced.weight is sliced.rows


def test_placement_leaves_a_sliced_table_to_itself():
    from yue2_comfy import placement

    assert placement._self_placed(vocabulary.SlicedEmbedding(table(), torch.device("cpu")))
    assert not placement._self_placed(table())


def test_only_off_keeps_the_tables_on_the_card():
    from yue2_comfy import placement

    assert not placement.narrows(placement.OFF)
    assert placement.narrows(placement.ON)
    assert placement.narrows(placement.AUTO)


def test_tuning_a_model_without_tables_does_nothing():
    """Every mode calls tune; only some modes installed anything."""
    vocabulary.tune(object(), vocabulary.MUSIC, [1, 2, 3])
