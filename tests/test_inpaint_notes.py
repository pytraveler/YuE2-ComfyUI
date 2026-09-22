"""A change of notes: the whole score the song is to have, sung again over the bars it changes.

The score editor hands back a whole score, and the song must not be sung
again whole for a note or two, so the list carries that score and the node
works out which bars it changed. Bars changed far apart are sung one stretch
at a time, each edit taking only its own bars' music from the score, so the
song ends with every change in it and nothing between them sung again.
Nothing here needs torch.
"""

from __future__ import annotations

import json
import types

import pytest

from yue2_comfy import notation
from yue2_comfy.inpaint import track

from test_inpaint_track import LYRICS, RAP, rap_state


def rewritten(where, text=RAP):
    """*text* with the sung note starting at each tick of *where* moved to the pitch it maps to, as the editor writes it."""
    sheet = notation.read(text)
    for note in sheet["notes"]["Vocal"]:
        note["pitch"] = where.get(note["start"], note["pitch"])
    return notation.write(text, sheet)["abc"]


CHORUS_NOTE = rewritten({86: 64})
"""The rap with one note of bar 6 -- the second bar of the chorus -- a semitone lower."""

TWO_PLACES = rewritten({36: 67, 112: 71})
"""The rap with a note changed in bar 3 and in bar 8, far apart."""


def notes(score, bars=None, seed=5, takes=2):
    item = {"op": "notes", "score": score, "seed": seed, "takes": takes}
    if bars is not None:
        item["bars"] = list(bars)
    return track.read(json.dumps([item]))[0]


def test_a_change_of_notes_reads_as_the_window_wrote_it():
    edit = notes(CHORUS_NOTE)
    assert (edit.op, edit.bars, edit.seconds, edit.score) == ("notes", None, None, CHORUS_NOTE)
    assert (edit.seed, edit.takes, edit.take) == (5, 2, None)
    assert track.read(track.written([edit])) == (edit,)
    kept = notes(CHORUS_NOTE, bars=(4, 6))
    assert kept.bars == (4, 6)
    assert json.loads(track.written([kept]))[0] == {
        "op": "notes", "bars": [4, 6], "score": CHORUS_NOTE, "seed": 5, "takes": 2}


@pytest.mark.parametrize("item,said", [
    ({"op": "notes", "seed": 1}, "holds no score"),
    ({"op": "notes", "score": "  ", "seed": 1}, "holds no score"),
    ({"op": "notes", "score": 7, "seed": 1}, "holds no score"),
    ({"op": "notes", "score": "x" * (notation.LONGEST + 1)}, "far longer"),
    ({"op": "notes", "score": RAP, "seconds": [1.0, 2.0]}, "go by bars"),
    ({"op": "notes", "score": RAP, "bars": [3, 3]}, "not a stretch of a score"),
])
def test_a_change_of_notes_that_cannot_be_one_is_refused(item, said):
    with pytest.raises(ValueError, match="Edit 1.*" + said):
        track.read(json.dumps([item]))


def test_the_bars_whose_notes_changed_are_the_ones_sung_again_under_the_new_score():
    """The score the editor hands back is stripped; the song's own final newline stays, since the prompt is read from it."""
    state = rap_state()
    step = track.plan(state, notes(CHORUS_NOTE))
    retake = track.plan(state, track.read('[{"op": "retake", "bars": [5, 6]}]')[0])
    assert (step.kind, step.bars) == ("notes", (5, 6))
    assert (step.start, step.stop) == (retake.start, retake.stop)
    assert (step.score, step.lyrics, step.dropped) == (CHORUS_NOTE + "\n", LYRICS, ())


def test_the_song_after_a_change_of_notes_has_its_notes():
    state = rap_state()
    step = track.plan(state, notes(CHORUS_NOTE))
    after = track.after(state, step, step.stop - step.start)
    assert after.score == CHORUS_NOTE + "\n"
    assert after.sheet["notes"] == notation.read(CHORUS_NOTE)["notes"]
    assert after.clock == state.clock and after.frames == state.frames


def test_changes_far_apart_named_by_bars_are_sung_one_stretch_at_a_time():
    """The window writes one edit a place, each with the whole score; each takes its own bars."""
    state = rap_state()
    whole = track.plan(state, notes(TWO_PLACES))
    assert whole.bars == (2, 8)
    first = track.plan(state, notes(TWO_PLACES, bars=(2, 3)))
    assert first.bars == (2, 3)
    assert notation.changed(RAP, first.score) == [2]
    state = track.after(state, first, first.stop - first.start)
    second = track.plan(state, notes(TWO_PLACES, bars=(7, 8)))
    assert second.bars == (7, 8)
    assert notation.changed(state.score, second.score) == [7]
    assert second.score == TWO_PLACES + "\n"


def test_a_wider_stretch_than_the_change_is_sung_whole():
    step = track.plan(rap_state(), notes(CHORUS_NOTE, bars=(4, 8)))
    assert step.bars == (4, 8) and step.score == CHORUS_NOTE + "\n"


def test_a_change_of_notes_that_changes_nothing_says_so():
    with pytest.raises(ValueError, match="same notes as the song, so there is nothing"):
        track.plan(rap_state(), notes(RAP))
    with pytest.raises(ValueError, match="same notes as the song in bars 3 to 4"):
        track.plan(rap_state(), notes(CHORUS_NOTE, bars=(2, 4)))


def test_a_score_that_moves_more_than_its_notes_is_refused():
    with pytest.raises(ValueError, match="another tempo"):
        track.plan(rap_state(), notes(CHORUS_NOTE.replace("Q:1/4=120", "Q:1/4=100")))
    with pytest.raises(ValueError, match="names its sections otherwise"):
        track.plan(rap_state(), notes(CHORUS_NOTE.replace("% outro", "% ending")))


def test_a_song_without_a_score_has_no_notes_to_change():
    state = track.opened(types.SimpleNamespace(score="", lyrics=LYRICS, frames=500), None)
    with pytest.raises(ValueError, match="no score and no notes to change"):
        track.plan(state, notes(CHORUS_NOTE))


def test_other_notes_are_other_takes():
    assert track.name("song", [], notes(CHORUS_NOTE)) != track.name("song", [], notes(TWO_PLACES))
    assert track.name("song", [], notes(CHORUS_NOTE)) == track.name("song", [], notes(CHORUS_NOTE))
