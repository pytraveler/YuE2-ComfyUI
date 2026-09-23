"""Sections moved to another place in a song: the score, the words, the frames and the grid after it.

A move keeps every piece of sound it lays out and sings only the last bar
before each seam again (``core.moved``), so most of what can go wrong is here:
which bars go where, that every piece is cut at the same place before its bar
line so the beat runs on across every seam, which stretch before each seam is
sung again and what its new length moves, that the words go with their
sections and keep their layout, and that the grid after it puts every bar where
its sound now is. Nothing here needs torch.
"""

from __future__ import annotations

import json

import pytest

from yue2_comfy import notation
from yue2_comfy.constants import FRAME_SECONDS
from yue2_comfy.inpaint import grid, lines, ops, track

from test_inpaint_track import FRAMES, LYRICS, RAP, rap_state, score


def move(first=4, stop=8, to=2):
    return track.read(json.dumps([{"op": "move", "bars": [first, stop], "to": to}]))[0]


def vocal_bars(text):
    """Every bar of the Vocal part as written, in order."""
    found = []
    rows = text.splitlines()
    for index, row in enumerate(rows):
        if row == "V: Vocal":
            music = next(line for line in rows[index + 1:] if not line.startswith(("M:", "K:")))
            found.extend(music[:-1].split("|"))
    return found


def test_the_moved_bars_are_read_before_the_bar_line_they_go_to():
    """Moving the verse after the chorus and the chorus before the verse is one move."""
    assert notation.order(4, 8, 2, 10) == [(0, 2), (4, 8), (2, 4), (8, 10)]
    assert notation.order(2, 4, 8, 10) == [(0, 2), (4, 8), (2, 4), (8, 10)]
    assert notation.order(0, 2, 10, 10) == [(2, 10), (0, 2)]
    assert notation.order(8, 10, 0, 10) == [(8, 10), (0, 8)]


@pytest.mark.parametrize("first,stop,to,said", [
    (4, 8, 4, "where they already are"),
    (4, 8, 8, "where they already are"),
    (4, 8, 6, "where they already are"),
    (4, 8, 11, "no bar line 12"),
    (4, 11, 2, "not bars of this score"),
    (4, 4, 2, "not bars of this score"),
])
def test_a_move_nowhere_or_outside_the_score_is_refused(first, stop, to, said):
    with pytest.raises(ValueError, match=said):
        notation.moved(RAP, first, stop, to)


def test_moved_bars_keep_their_characters_and_their_section_names():
    moved = notation.moved(RAP, 4, 8, 2)
    bars = vocal_bars(RAP)
    assert vocal_bars(moved) == bars[:2] + bars[4:8] + bars[2:4] + bars[8:]
    assert notation.read(moved)["sections"] == [
        {"name": "intro", "bar": 0, "bars": 2}, {"name": "chorus", "bar": 2, "bars": 4},
        {"name": "verse", "bar": 6, "bars": 2}, {"name": "outro", "bar": 8, "bars": 2}]
    assert notation.moved(RAP + "\n\n", 4, 8, 2).endswith("Z|Z|\n\n\n"), (
        "the text around the score stays")


def test_a_tie_off_the_end_of_a_stretch_comes_off_and_what_it_held_is_struck_again():
    tied = score(("intro", ["z16", "z16"]), ("verse", ["C16", "C8D8-"]),
                 ("chorus", ["D8E8", "F16"]), ("outro", ["z16"]))
    moved = notation.read(notation.moved(tied, 4, 6, 2))
    verse_end = moved["bars"][5]["start"]
    assert [n for n in moved["notes"]["Vocal"] if n["start"] == moved["bars"][2]["start"]] == [
        {"start": moved["bars"][2]["start"], "length": 8, "pitch": 62}], "the D of the chorus is its own"
    assert {"start": verse_end + 8, "length": 8, "pitch": 62} in moved["notes"]["Vocal"]


def test_a_stretch_that_starts_inside_a_section_is_named_after_it_again():
    """Bar 4 of the rap, the second of its verse, put before the outro stays a verse there."""
    moved = notation.read(notation.moved(RAP, 3, 4, 8))
    assert moved["sections"] == [
        {"name": "intro", "bar": 0, "bars": 2}, {"name": "verse", "bar": 2, "bars": 1},
        {"name": "chorus", "bar": 3, "bars": 4}, {"name": "verse", "bar": 7, "bars": 1},
        {"name": "outro", "bar": 8, "bars": 2}]


def test_a_stretch_carries_its_own_key_where_another_is_in_force():
    """The chorus in G put before a verse in C: both keep their keys, one field each."""
    text = (RAP.replace("% chorus\nV: Vocal\n", "% chorus\nV: Vocal\nK:G\n")
            .replace("A2z8z2G2G2|\nV: Ins\n", "A2z8z2G2G2|\nV: Ins\nK:G\n")
            .replace("% outro\nV: Vocal\n", "% outro\nV: Vocal\nK:C\n")
            .replace("z16|z16|\nV: Ins\n", "z16|z16|\nV: Ins\nK:C\n"))
    before = notation.read(text)
    assert [bar["key"] for bar in before["bars"]] == ["C"] * 4 + ["G"] * 4 + ["C"] * 2
    after = notation.read(notation.moved(text, 4, 8, 2))
    assert [bar["key"] for bar in after["bars"]] == ["C"] * 2 + ["G"] * 4 + ["C"] * 4


def test_bars_that_change_key_halfway_are_not_moved_past():
    text = RAP.replace("A2z8DDBB3|", "A2z8[K:G]DDBB3|").replace(
        "V: Ins\nZ|Z|Z|Z|\n", "V: Ins\nZ|Z|z8z2[K:G]z6|Z|\n")
    assert [bar["key"] for bar in notation.read(text)["bars"]][7] == "G"
    with pytest.raises(ValueError, match="changes key halfway through"):
        notation.moved(text, 2, 4, 8)
    assert notation.read(notation.moved(text, 0, 2, 4))["bars"][6]["editable"] is False, (
        "a move that leaves that bar where it was takes nothing past it")


def test_the_shift_of_a_move_lies_in_every_rest_it_can():
    rests = [grid.Seam(1, 0.0, 0.0, 10.0), grid.Seam(2, 2.0, 2.0, 30.0), grid.Seam(3, 0.0, 0.0, 12.0)]
    assert grid.shared_shift(rests, 4.0, 1.0) == 6.0, "a beat clear of the latest phrase, inside every rest"
    assert grid.shared_shift(rests[:1], 4.0, 1.0) == 4.0


def test_a_move_through_singing_that_runs_on_keeps_every_first_word():
    """The stand's rap runs every section on from the one before: the phrases give, never a pickup."""
    rests = [grid.Seam(1, 6.0, 6.0, 6.0), grid.Seam(2, 6.0, 6.0, 64.0), grid.Seam(3, 6.0, 6.0, 16.0)]
    assert grid.shared_shift(rests, 4.0, 0.5) == 6.5


def test_every_piece_of_a_move_is_cut_the_same_way_before_its_bar_line():
    """So the pieces are their bars' own lengths and the beat runs on across every seam."""
    state = rap_state()
    marks = grid.seams(state.sheet)
    stretches = notation.order(4, 8, 2, 10)
    pieces = grid.move_frames(state.sheet, state.clock, marks, stretches, FRAMES)
    beat, margin = grid._beat_and_margin(state.sheet, state.clock)
    shift = grid.shared_shift([marks[2], marks[4], marks[8]], beat, margin)
    at = {bar: state.clock.moment(bar, shift) for bar in (2, 4, 8)}
    assert pieces == [(0, at[2]), (at[4], at[8]), (at[2], at[4]), (at[8], FRAMES)]
    assert sum(stop - start for start, stop in pieces) == FRAMES
    assert at[4] - at[2] == state.clock.frames(32) and at[8] - at[4] == state.clock.frames(64)


def test_the_words_of_a_moved_section_go_with_it_and_the_layout_stays():
    words = ops.move_words(LYRICS, RAP, 4, 8, 2)
    assert words.text == ("[Intro]\n\n[Chorus]\nseven eight\n\n[Verse]\none two three\n"
                          "four five six\n\n[Outro]")
    assert (words.moved, words.matched, words.tagged) == (("Chorus",), True, False)
    assert ops.move_words(LYRICS, RAP, 2, 4, 8).text == words.text, "the same move said the other way"
    assert ops.move_words(LYRICS, RAP, 8, 10, 0).text == LYRICS, "the outro has no words to move"


def test_words_sung_before_the_first_tag_are_put_under_a_tag_when_they_no_longer_come_first():
    words = ops.move_words("one two three\nfour five six\n\n[Chorus]\nseven eight", RAP, 4, 8, 2)
    assert words.text == "[Chorus]\nseven eight\n\n[Verse]\none two three\nfour five six"
    assert words.tagged


def test_words_that_cannot_be_paired_with_the_score_stay_where_they_are():
    lyrics = "[Verse]\na\n\n[Verse]\nb\n\n[Verse]\nc\n\n[Verse]\nd"
    words = ops.move_words(lyrics, RAP, 4, 8, 2)
    assert (words.text, words.matched) == (lyrics, False)


def test_every_frame_of_a_move_keeps_the_noise_it_was_drawn_from():
    pieces = [(0, 100), (300, 400), (100, 300), (400, 500)]
    assert ops.moved_noise([[3, 0, 500]], pieces) == [[3, 0, 100], [3, 300, 100], [3, 100, 200],
                                                     [3, 400, 100]]
    assert ops.moved_noise([[3, 0, 200], [8, 0, 300]], [(200, 500), (0, 200)]) == [
        [8, 0, 300], [3, 0, 200]]


def test_a_move_reads_as_the_window_wrote_it():
    edit = move()
    assert (edit.op, edit.bars, edit.seconds, edit.to) == ("move", (4, 8), None, 2)
    assert (edit.seed, edit.takes, edit.take) == (0, 1, None)
    assert json.loads(track.written([edit])) == [{"op": "move", "bars": [4, 8], "to": 2}]
    assert track.read(track.written([edit])) == (edit,)
    assert track.name("song", [], move(4, 8, 2)) != track.name("song", [], move(4, 8, 10))


@pytest.mark.parametrize("item,said", [
    ({"op": "move", "bars": [4, 8]}, "'to' names the bar line"),
    ({"op": "move", "bars": [4, 8], "to": None}, "'to' names the bar line"),
    ({"op": "move", "seconds": [1.0, 2.0], "to": 2}, "it selects bars"),
    ({"op": "move", "to": 2}, "it selects bars"),
    ({"op": "move", "bars": [4, 8], "to": 2.5}, "not a whole number"),
    ({"op": "move", "bars": [4, 8], "to": -1}, "before the song"),
])
def test_a_move_that_does_not_say_where_is_refused(item, said):
    with pytest.raises(ValueError, match=said):
        track.read(json.dumps([item]))


@pytest.mark.parametrize("first,stop,to,said", [
    (5, 8, 2, "bar line 6 is inside the chorus"),
    (4, 8, 3, "bar line 4 is inside the verse"),
    (4, 7, 2, "bar line 8 is inside the chorus"),
    (4, 8, 12, "this score has 10 bars"),
    (4, 8, 8, "where they already are"),
])
def test_a_move_takes_whole_sections_to_where_another_starts(first, stop, to, said):
    with pytest.raises(ValueError, match=said):
        track.plan(rap_state(), move(first, stop, to))


def test_a_song_with_no_score_has_no_sections_to_move():
    state = track.State(lyrics=LYRICS, score="", sheet=None, clock=None, frames=FRAMES)
    with pytest.raises(ValueError, match="no sections to move"):
        track.plan(state, move())


def test_a_move_is_its_pieces_its_score_and_its_words():
    state = rap_state()
    step = track.plan(state, move())
    stretches = notation.order(4, 8, 2, 10)
    pieces = grid.move_frames(state.sheet, state.clock, grid.seams(state.sheet), stretches, FRAMES)
    assert step.kind == "move" and step.bars == (4, 8)
    assert step.pieces == tuple(pieces) and step.stretches == tuple(stretches)
    assert (step.start, step.stop) == pieces[1]
    assert step.score == notation.moved(RAP, 4, 8, 2)
    assert step.lyrics == ops.move_words(LYRICS, RAP, 4, 8, 2).text
    assert step.notices == ()
    assert track.placed(step) == (pieces[0][1], pieces[0][1] + pieces[1][1] - pieces[1][0])


def test_a_move_that_leaves_an_edge_bare_says_so():
    """The song opening in the middle of its intro's phrase, or ending before what came next."""
    state = rap_state()
    head = track.plan(state, move(0, 2, 8)).notices
    assert [level for level, _text in head] == ["notice"] and "opens where the verse began" in head[0][1]
    tail = track.plan(state, move(4, 8, 10)).notices
    assert [level for level, _text in tail] == ["notice"] and "ends where the chorus ended" in tail[0][1]


def test_after_a_move_every_bar_line_is_where_its_sound_now_is():
    state = rap_state()
    step = track.plan(state, move())
    after = track.after(state, step, step.stop - step.start)
    old = state.clock.bar_seconds()
    new = after.clock.bar_seconds()
    order = [bar for low, high in step.stretches for bar in range(low, high)]
    at = 0
    for (low, high), (start, stop) in zip(step.stretches, step.pieces):
        for bar in range(low, high):
            assert new[order.index(bar)] == pytest.approx(old[bar] + (at - start) * FRAME_SECONDS)
        at += stop - start
    assert new[-1] == pytest.approx(old[-1]) and after.frames == FRAMES
    assert (after.lyrics, after.score) == (step.lyrics, step.score)
    laid = grid.layout(after.sheet, after.clock, after.frames)
    assert [section["name"] for section in laid["sections"]] == ["intro", "chorus", "verse", "outro"]
    assert laid["bars"] == pytest.approx([round(second, 3) for second in new], abs=1e-3)


def test_a_bar_edit_after_a_move_finds_the_bars_where_they_went():
    state = rap_state()
    step = track.plan(state, move())
    after = track.after(state, step, step.stop - step.start)
    retake = track.plan(after, track.read('[{"op": "retake", "bars": [2, 6], "seed": 1}]')[0])
    chorus = track.placed(step)
    second = int(round(1.0 / FRAME_SECONDS))
    assert abs(retake.start - chorus[0]) <= second and abs(retake.stop - chorus[1]) <= second


def test_the_word_times_of_a_moved_song_go_with_their_sound():
    times = [("one", 5.0, 5.4), ("two", 5.5, 5.9), ("seven", 9.0, 9.4)]
    pieces = [(0.0, 4.0), (8.0, 12.0), (4.0, 8.0), (12.0, 20.0)]
    carried = lines.carried_along(times, "one two\nseven", pieces, "seven\none two")
    assert [word for word, _start, _stop in carried] == ["seven", "one", "two"]
    assert [time for _word, start, stop in carried for time in (start, stop)] == pytest.approx(
        [5.0, 5.4, 9.0, 9.4, 9.5, 9.9])
    assert lines.carried_along(times, "one two\nseven", pieces, "one two\nseven") is None, (
        "words that did not move with their sound are not these words")
    assert lines.carried_along(times, "one two three\nseven", pieces, "seven\none two") is None


def test_a_line_goes_whole_with_the_piece_most_of_it_is_sung_in():
    """The aligner glued the rap's first word to the intro, nine seconds early, and a held word runs past a cut."""
    times = [("I", 0.0, 9.6), ("was", 9.6, 9.8), ("born", 9.9, 10.2), ("sound", 19.6, 20.08),
             ("We", 20.1, 20.3), ("rise", 20.3, 21.0)]
    pieces = [(0.0, 9.48), (20.04, 41.12), (9.48, 20.04)]
    carried = lines.carried_along(times, "I was born sound\nWe rise", pieces, "We rise\nI was born sound")
    assert [word for word, _start, _stop in carried] == ["We", "rise", "I", "was", "born", "sound"]
    assert [(round(start, 2), round(stop, 2)) for _word, start, stop in carried] == [
        (9.54, 9.74), (9.74, 10.44), (30.56, 30.68), (30.68, 30.88), (30.98, 31.28),
        (40.68, 41.12)], "every word inside its line's piece, the held one ending where it does"


def test_a_move_cuts_where_the_voice_rests_when_the_score_would_cut_into_it():
    """The pop's second verse came in 0.4 s ahead of its score: the voice, not the score, knows."""
    downbeats = [10.0, 30.0, 50.0]
    unit = 0.1

    def resting(at):
        return lambda second: -90.0 if any(abs(second - (beat - at * unit)) < 0.01
                                           for beat in downbeats) else -10.0

    assert grid.voiced_shift(downbeats, unit, 4.0, 5.0, resting(7), 12) == 7.0
    assert grid.voiced_shift(downbeats, unit, 4.0, 5.0, resting(9), 12) == 9.0
    assert grid.voiced_shift(downbeats, unit, 4.0, 6.5, lambda second: -12.0, 12) == 6.5, (
        "where the voice never rests, the score's own shift stands")
    assert grid.voiced_shift(downbeats, unit, 4.0, 5.0, lambda second: -11.0 + second * 0.01,
                             12) == 5.0, "a few dB quieter is not reason enough to move"


def test_the_voice_is_asked_about_the_stretch_before_every_bar_line_a_move_cuts_at():
    state = rap_state()
    marks = grid.seams(state.sheet)
    stretches = notation.order(4, 8, 2, 10)
    windows = grid.move_windows(state.sheet, state.clock, marks, stretches)
    touched, default, top, beat = grid._move_lines(state.sheet, state.clock, marks, stretches)
    unit = state.clock.rate * state.clock.tick
    assert touched == [2, 4, 8] and top >= 2 * beat and top >= default + beat
    assert windows == [(pytest.approx(state.clock.at(bar) - top * unit - grid.QUIET_WINDOW),
                        pytest.approx(state.clock.at(bar) + grid.QUIET_WINDOW)) for bar in touched]
    quiet = int(default) + 2
    level = (lambda second: -90.0 if any(abs(second - (state.clock.at(bar) - quiet * unit)) < 0.01
                                         for bar in touched) else -10.0)
    pieces = grid.move_frames(state.sheet, state.clock, marks, stretches, FRAMES, level)
    at = {bar: state.clock.moment(bar, quiet) for bar in touched}
    assert pieces == [(0, at[2]), (at[4], at[8]), (at[2], at[4]), (at[8], FRAMES)]
    assert track.plan(state, move(), level=level).pieces == tuple(pieces)


def seams_of(step):
    """``[(new bar line, frame)]`` where two pieces of a move meet that were not neighbours."""
    found, bar, frame = [], 0, 0
    for index, ((low, high), (start, stop)) in enumerate(zip(step.stretches, step.pieces)):
        if index and step.pieces[index - 1][1] != start:
            found.append((bar, frame))
        bar += high - low
        frame += stop - start
    return found


@pytest.mark.parametrize("first,stop,to", [(4, 8, 2), (4, 8, 10), (0, 2, 8)])
def test_a_move_sings_the_last_bar_before_each_seam_again(first, stop, to):
    """From where a retake of that bar would open, a second at least, never back past the seam before."""
    state = rap_state()
    step = track.plan(state, move(first, stop, to))
    seams = seams_of(step)
    assert [seam for _start, seam in step.sung] == [frame for _bar, frame in seams]
    moved = notation.read(step.score)
    clock = grid.after_move(state.clock, grid.starts_of(moved), step.stretches, step.pieces)
    marks = grid.seams(moved)
    beat, margin = grid._beat_and_margin(moved, clock)
    before = 0
    for (start, seam), (bar, _frame) in zip(step.sung, seams):
        opens = clock.moment(bar - 1, grid.opening(marks[bar - 1], beat, margin))
        assert before <= start <= seam - track.SUNG_LEAST
        assert start == max(before, min(opens, seam - track.SUNG_LEAST))
        before = seam
    assert step.pulse == pytest.approx(
        state.clock.rate * state.clock.tick * state.sheet["per_quarter"] / 2), "an eighth note"


def sung_longer(step, gain):
    """``core.Take.sung`` for a move whose every stretch before a seam came out ``gain`` frames longer."""
    made, shift = [], 0
    for start, seam in step.sung:
        made.append((start + shift, seam + shift, seam - start + gain))
        shift += gain
    return tuple(made)


def test_what_comes_after_a_bar_sung_again_moves_by_what_it_gained():
    state = rap_state()
    step = track.plan(state, move())
    made = sung_longer(step, 3)
    plain = track.after(state, step, step.stop - step.start)
    after = track.after(state, step, step.stop - step.start, None, made)
    assert after.frames == FRAMES + 3 * len(made)
    first = made[0][0] * FRAME_SECONDS
    last = made[-1][1] * FRAME_SECONDS
    for old, new in zip(plain.clock.bar_seconds(), after.clock.bar_seconds()):
        if old <= first:
            assert new == pytest.approx(old)
        elif old >= last:
            assert new == pytest.approx(old + 3 * len(made) * FRAME_SECONDS)
    low, high = track.placed(step)
    assert track.placed(step, made) == (track.shifted(low, made), track.shifted(high, made))
    assert track.placed(step, made)[0] == low + 3, "the chorus comes after the first seam"


def test_a_frame_inside_a_bar_sung_again_is_spread_through_it():
    sung = ((20, 30, 12),)
    assert [track.shifted(frame, sung) for frame in (10, 20, 25, 30, 40)] == [10, 20, 26, 32, 42]
    assert track.shifted(40, ((20, 30, 12), (50, 60, 8))) == 42
    assert track.shifted(55, ((20, 30, 12), (50, 60, 8))) == 56


def test_word_times_move_past_the_bars_sung_again():
    times = [("one", 1.0, 1.5), ("two", 2.5, 3.0), ("three", 5.0, 5.5)]
    moved = lines.stretched(times, [(2.0, 4.0, 3.0)])
    assert [word for word, _start, _stop in moved] == ["one", "two", "three"]
    assert [time for _word, start, stop in moved for time in (start, stop)] == pytest.approx(
        [1.0, 1.5, 2.75, 3.5, 6.0, 6.5])
    assert lines.stretched(times, []) == times

