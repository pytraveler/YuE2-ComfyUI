"""A song that goes on: past its last words, under a score the model writes on from where the singing ends.

The list carries only the words to go on with, so everything else is worked
out here from the song: the bar it goes on from, the part of the old score
the model writes on from, the lyrics it is sung under, how long the new part
should run once a take has written its score, and the song it leaves. Nothing
here needs torch.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from yue2_comfy import notation
from yue2_comfy.constants import FRAME_SECONDS
from yue2_comfy.inpaint import grid, lines, ops, track

from test_inpaint_track import FRAMES, LYRICS, RAP, rap_state, score

BRIDGE = "[Bridge]\nnine ten eleven\ntwelve and more"

TAIL = score(("intro", ["z16", "z8z2DDA2AA"]),
             ("verse", ["A2DDAAA2AGGGG2GG", "AGGGGFFFFDD2B4"]),
             ("chorus", ["A2z8D2B4", "A2z4FG3F2B2B2", "A2z8DDBB3", "A2z8z2G2G2-"]),
             ("outro", ["G4z12", "z16"]))
"""The rap whose last word is held into the outro's first bar, the way the model marks a last chorus's tail."""


def extend(text=BRIDGE, seed=5, takes=2):
    item = {"op": "extend", "seed": seed, "takes": takes}
    if text is not None:
        item["text"] = text
    return track.read(json.dumps([item]))[0]


def state_of(text=RAP, lyrics=LYRICS, frames=FRAMES, offset=0.5):
    sheet = notation.read(text)
    clock = grid.Grid(offset=offset, rate=1.0, tick=grid.tick_seconds(sheet),
                      starts=grid.starts_of(sheet))
    return track.State(lyrics=lyrics, score=text, sheet=sheet, clock=clock, frames=frames)


def test_a_song_going_on_reads_as_the_window_wrote_it():
    edit = extend()
    assert (edit.op, edit.bars, edit.seconds, edit.text) == ("extend", None, None, BRIDGE)
    assert (edit.seed, edit.takes, edit.take, edit.lines, edit.score) == (5, 2, None, None, None)
    assert track.read(track.written([edit])) == (edit,)
    assert json.loads(track.written([edit]))[0] == {"op": "extend", "text": BRIDGE, "seed": 5,
                                                    "takes": 2}
    assert extend(text=None).text == "", "no words at all is a new ending alone"


@pytest.mark.parametrize("item,said", [
    ({"op": "extend", "bars": [8, 10]}, "selects no bars and no seconds"),
    ({"op": "extend", "seconds": [18.0, 20.0]}, "selects no bars and no seconds"),
    ({"op": "extend", "text": 5}, "not text"),
    ({"op": "extend", "text": "la " * 2000}, "more than 4000 characters"),
])
def test_a_song_going_on_that_says_something_else_is_refused(item, said):
    with pytest.raises(ValueError, match=said):
        track.read(json.dumps([item]))


def test_it_goes_on_from_the_first_bar_after_the_last_sung_note():
    """The rap's chorus sings to the end of bar 8; its outro is the ending the new part replaces."""
    state = rap_state()
    step = track.plan(state, extend())
    marks = grid.seams(state.sheet)
    assert step.kind == "extend" and step.bars == (8, 10)
    assert (step.start, step.stop) == (
        grid.retake_frames(state.sheet, state.clock, marks, 8, 10, FRAMES)[0], FRAMES)
    assert step.score == notation.head(RAP, 8)
    assert notation.read(step.score)["sections"] == [
        {"name": "intro", "bar": 0, "bars": 2}, {"name": "verse", "bar": 2, "bars": 2},
        {"name": "chorus", "bar": 4, "bars": 4}]
    assert step.now == ("nine ten eleven", "twelve and more")
    assert track.going_on_from(state) == {"bar": 8, "bars": 10, "start": step.start}


def test_trailing_tags_stay_last_so_the_new_part_comes_before_the_ending():
    step = track.plan(rap_state(), extend())
    assert step.lyrics == LYRICS.replace("\n\n[Outro]", "") + "\n\n" + BRIDGE + "\n\n[Outro]"


def test_the_tail_of_the_last_chorus_is_not_written_on_as_an_outro():
    """A score ending on an outro comment is finished by writing more outro, so the comment goes."""
    step = track.plan(state_of(TAIL), extend())
    assert step.bars == (9, 10), "the held word ends in bar 9, the outro's first"
    assert "% outro" not in step.score
    assert notation.read(step.score)["sections"][-1] == {"name": "chorus", "bar": 4, "bars": 5}


def test_a_new_ending_alone_keeps_the_words_and_hears_nothing():
    step = track.plan(rap_state(), extend(text=""))
    assert step.lyrics == LYRICS and step.now == () and step.notices == ()


def test_words_under_no_tag_are_sung_as_a_verse_and_that_is_said():
    step = track.plan(rap_state(), extend(text="nine ten eleven"))
    assert step.lyrics.endswith("[Verse]\nnine ten eleven\n\n[Outro]")
    assert step.now == ("nine ten eleven",)
    assert [level for level, _text in step.notices] == ["notice"]
    assert "[Verse]" in step.notices[0][1]


def test_a_song_that_stopped_short_of_its_score_goes_on_from_the_last_bar_it_reached():
    """Twelve seconds of the rap reach bar 6; the bars after it were never sung, so the model writes them again."""
    state = state_of(frames=300)
    found = track.going_on_from(state)
    assert found["bar"] == 5
    assert track.plan(state, extend()).score == notation.head(RAP, 5)


def test_a_song_with_no_score_has_nothing_for_the_model_to_go_on_writing():
    state = dataclasses.replace(rap_state(), sheet=None, clock=None, score="")
    assert track.going_on_from(state) is None
    with pytest.raises(ValueError, match="no score for the model to go on writing"):
        track.plan(state, extend())


def written_on(step, *bars):
    """``step.score`` with ``bars`` more bars written after it, as a take would."""
    return step.score + "% bridge\nV: Vocal\n" + "|".join(bars) + "|\nV: Ins\n" \
        + "|".join(["Z"] * len(bars)) + "|\n"


def test_the_new_part_is_as_long_as_its_bars_at_the_songs_tempo_and_keeps_its_tail():
    state = rap_state()
    step = track.plan(state, extend())
    sheet = notation.read(written_on(step, "A2z8D2B4", "z16", "z16"))
    assert FRAMES < state.clock.moment(10), "this song stops before its score does: no tail"
    assert track.extension_frames(state, step, sheet) == (
        state.clock.moment(8) - step.start + state.clock.frames(3 * 16))
    longer = dataclasses.replace(state, frames=state.clock.moment(10) + 30)
    assert track.extension_frames(longer, step, sheet) == (
        state.clock.moment(8) - step.start + state.clock.frames(3 * 16) + 30)


def test_the_song_left_has_the_takes_score_and_its_bars_after_the_old_ones():
    state = rap_state()
    step = track.plan(state, extend())
    text = written_on(step, "A2z8D2B4", "z16", "z16")
    left = track.after(state, step, 200, text)
    assert left.score == text and left.lyrics == step.lyrics
    assert left.frames == step.start + 200
    assert len(left.sheet["bars"]) == 11
    assert left.clock.starts == grid.starts_of(left.sheet)
    assert [left.clock.at(bar) for bar in range(9)] == [state.clock.at(bar) for bar in range(9)]


def test_a_take_is_heard_for_its_new_words_from_a_beat_before_its_first_new_note():
    """Opened on bars of playing, the aligner puts the first word where the stretch opens."""
    state = rap_state()
    step = track.plan(state, extend())
    opening = step.start * FRAME_SECONDS
    assert opening == pytest.approx(16.56) and state.clock.at(8) == pytest.approx(16.5)
    later = notation.read(written_on(step, "z16", "z8A2z6", "A2z8D2B4"))
    assert track.sings_from(state, step, later) == pytest.approx(16.5 + 2.0 + 1.0 - 0.5), (
        "bar 9's third beat, less the beat of a quarter at 120")
    right_away = notation.read(written_on(step, "A2z8D2B4"))
    assert track.sings_from(state, step, right_away) == pytest.approx(opening), (
        "never before the new part begins")
    silent = notation.read(written_on(step, "z16", "z16"))
    assert track.sings_from(state, step, silent) == pytest.approx(opening)


def test_only_the_lines_a_song_went_on_with_are_timed_on_its_new_part():
    old = "one two\nthree"
    assert lines.added_after(old, old + "\nfour five") == "four five"
    assert lines.added_after(old, old) == "", "a new ending alone adds no words"
    assert lines.added_after("", "four five") == "four five"
    assert lines.added_after(old, "one two\nthree four") is None, "a line changed is not one added"
    assert lines.added_after(old, "one two") is None
    assert lines.went_on([("one", 0.0, 0.4)], [("four", 0.2, 0.6)], 10.0) == [
        ("one", 0.0, 0.4), ("four", 10.2, 10.6)]


def test_after_a_cut_the_new_bars_fall_at_the_tempo_from_where_the_lines_left_off():
    lined = grid.Grid(offset=0.5, rate=1.0, tick=0.125, starts=tuple(range(0, 176, 16)),
                      lines=(0.5, 2.5, 4.5, 6.5, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0))
    after = grid.after_extend(lined, tuple(range(0, 208, 16)), 8)
    assert after.lines[:9] == lined.lines[:9]
    assert after.lines[9:] == (18.0, 20.0, 22.0, 24.0)
    plain = dataclasses.replace(lined, lines=())
    assert grid.after_extend(plain, tuple(range(0, 208, 16)), 8) == dataclasses.replace(
        plain, starts=tuple(range(0, 208, 16)))


def test_the_head_keeps_its_bars_and_lets_go_of_a_tie_into_the_bars_it_leaves_out():
    text = notation.head(TAIL, 8)
    assert text.startswith(TAIL[:TAIL.index("V: Vocal\nA2z8D2B4")])
    assert "G2G2-" not in text and "A2z8z2G2G2|" in text
    assert len(notation.read(text)["bars"]) == 8
    assert notation.head(RAP, 10) == RAP
    for stop in (0, 11):
        with pytest.raises(ValueError, match="cannot end after bar"):
            notation.head(RAP, stop)


def test_new_words_keep_the_line_endings_and_refuse_a_book():
    crlf = LYRICS.replace("\n", "\r\n")
    going = ops.go_on(crlf, "[Bridge]\nnine ten")
    assert going.text == crlf.replace("\r\n\r\n[Outro]", "") + "\r\n\r\n[Bridge]\r\nnine ten" \
        + "\r\n\r\n[Outro]"
    assert ops.go_on("", "nine ten").text == "[Verse]\nnine ten"
    assert ops.go_on(LYRICS, "  \n ").text == LYRICS
    assert ops.go_on(LYRICS, "[Outro]").sung == ()
    with pytest.raises(ValueError, match="more than 4000"):
        ops.go_on(LYRICS, "la " * 2000)


def test_the_last_sung_section_is_found_by_the_words_paired_with_the_score():
    assert ops.last_sung(LYRICS, RAP) == 4
    assert ops.last_sung("[Intro]\n\n[Outro]", RAP) is None


def test_a_take_that_ended_the_song_itself_is_kept_over_one_stopped_at_its_limit():
    core = pytest.importorskip("yue2_comfy.inpaint.core")

    def take(ended, heard=None):
        return core.Take(seed=0, waveform=None, song=None, count=1, join=None, joins={},
                         ended=ended, timing={}, heard=heard)

    assert core.best([take(False, (5, 5)), take(True, (4, 5))]) == 1
    assert core.best([take(True, (3, 5)), take(True, (5, 5))]) == 1
    assert core.best([take(False), take(True)]) == 1
    assert core.best([take(True), take(True)]) == 0
