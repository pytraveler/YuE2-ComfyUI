"""The edit list, and the song it leaves behind: parsing, refusals, the grid after an edit, the names of results.

A track window writes a JSON list and the node reads it, so the shape of that
list is checked here rather than in the browser. The rest is the chain: a
second edit is made on the song the first one left, which is shorter than the
one that was sung, and its bars have moved. Nothing here needs torch.
"""

from __future__ import annotations

import types

import pytest

from yue2_comfy import notation
from yue2_comfy.inpaint import grid, track

HEADER = ('X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n'
          'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
          'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n')


def score(*sections):
    """A score of ``(name, bars)`` sections, the Ins part resting, at most four bars a line."""
    lines = [HEADER.rstrip("\n")]
    for name, bars in sections:
        lines.append("% " + name)
        for start in range(0, len(bars), 4):
            group = bars[start:start + 4]
            lines += ["V: Vocal", "|".join(group) + "|", "V: Ins", "|".join(["Z"] * len(group)) + "|"]
    return "\n".join(lines) + "\n"


RAP = score(("intro", ["z16", "z8z2DDA2AA"]),
            ("verse", ["A2DDAAA2AGGGG2GG", "AGGGGFFFFDD2B4"]),
            ("chorus", ["A2z8D2B4", "A2z4FG3F2B2B2", "A2z8DDBB3", "A2z8z2G2G2"]),
            ("outro", ["z16", "z16"]))
"""The grid tests' rap: intro 0-1, verse 2-3, chorus 4-7, outro 8-9, every bar two seconds long."""

LYRICS = "[Intro]\n\n[Verse]\none two three\nfour five six\n\n[Chorus]\nseven eight\n\n[Outro]"

FRAMES = 500
"""Twenty seconds: the ten bars from 0.5 s, and a little after them."""


def rap_song(score_text=RAP, lyrics=LYRICS):
    return types.SimpleNamespace(score=score_text, lyrics=lyrics, frames=FRAMES)


def rap_state(offset=0.5):
    sheet = notation.read(RAP)
    clock = grid.Grid(offset=offset, rate=1.0, tick=grid.tick_seconds(sheet),
                      starts=grid.starts_of(sheet))
    return track.opened(rap_song(), clock)


def test_an_empty_field_is_no_edits_at_all():
    """A node whose track has never been opened runs, hands the song on and draws the grid."""
    assert track.read("") == ()
    assert track.read("   \n") == ()
    assert track.read(None) == ()


def test_an_edit_list_reads_as_the_window_wrote_it():
    text = ('[{"op": "retake", "bars": [4, 8], "seed": 831001, "takes": 3, "take": 2},'
            ' {"op": "cut", "bars": [2, 4]},'
            ' {"op": "retake", "seconds": [1.5, 4.0], "seed": 7}]')
    first, second, third = track.read(text, takes=2)
    assert (first.op, first.bars, first.seconds) == ("retake", (4, 8), None)
    assert (first.seed, first.takes, first.take) == (831001, 3, 2)
    assert first.seeds() == (831001, 831002, 831003)
    assert (second.op, second.bars, second.takes, second.take) == ("cut", (2, 4), 1, None)
    assert (third.op, third.seconds, third.takes) == ("retake", (1.5, 4.0), 2)
    assert track.read(track.written((first, second, third)), takes=2) == (first, second, third)


def test_an_edit_list_that_is_not_one_is_refused_in_words_a_person_can_act_on():
    for text in ("{}", "[1, 2]", "not json", '"a string"'):
        with pytest.raises(ValueError):
            track.read(text)
    problems = [
        ('[{"op": "sing", "bars": [0, 1]}]', "not something an edit does"),
        ('[{"op": "retake"}]', "neither bars nor seconds"),
        ('[{"op": "retake", "bars": [0, 1], "seconds": [0, 1]}]', "neither bars nor seconds"),
        ('[{"op": "retake", "bars": [0]}]', "not a pair"),
        ('[{"op": "retake", "bars": [4, 4]}]', "not a stretch of a score"),
        ('[{"op": "retake", "bars": [4, 2]}]', "not a stretch of a score"),
        ('[{"op": "retake", "bars": [0.5, 2]}]', "not a whole number"),
        ('[{"op": "retake", "seconds": [4.0, 1.0]}]', "not a stretch of a song"),
        ('[{"op": "retake", "seconds": ["a", 1.0]}]', "not a number"),
        ('[{"op": "cut", "bars": [0, 2]}, {"op": "retake", "bars": [0, 2], "takes": 9}]',
         "between 1 and 4"),
        ('[{"op": "retake", "bars": [0, 2], "takes": 2, "take": 5}]', "keeps take 6 of 2"),
        ('[{"op": "retake", "bars": [0, 2], "seed": "x"}]', "not a whole number"),
    ]
    for text, said in problems:
        with pytest.raises(ValueError, match=said):
            track.read(text)


def test_an_edit_says_which_one_it_is_in_its_refusal():
    """A list is written by a window and read by a node, so a person has to be told which edit is wrong."""
    with pytest.raises(ValueError, match="Edit 2 "):
        track.read('[{"op": "cut", "bars": [0, 2]}, {"op": "cut"}]')


def test_a_retake_of_bars_takes_the_frames_the_grid_gives_it_and_leaves_the_words_alone():
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "retake", "bars": [4, 8]}]')[0])
    assert (step.start, step.stop) == (192, 414)
    assert (step.kind, step.bars, step.dropped) == ("retake", (4, 8), ())
    assert (step.lyrics, step.score) == (LYRICS, RAP)


def test_a_cut_of_bars_takes_them_out_of_the_score_and_the_words_with_them():
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "cut", "bars": [2, 4]}]')[0])
    assert (step.start, step.stop) == (92, 192)
    assert step.dropped == ("Verse",)
    assert "one two three" not in step.lyrics
    assert "seven eight" in step.lyrics
    assert len(notation.read(step.score)["bars"]) == 8
    assert step.notices and step.notices[0][0] == "notice"


def test_a_song_without_a_score_is_edited_by_the_second_and_cut_by_it_too():
    """'cot' off leaves no score and so no bars; the window then selects seconds."""
    state = track.opened(types.SimpleNamespace(score="", lyrics="la la", frames=FRAMES), None)
    step = track.plan(state, track.read('[{"op": "retake", "seconds": [2.0, 6.0]}]')[0])
    assert (step.start, step.stop) == (50, 150)
    cut = track.plan(state, track.read('[{"op": "cut", "seconds": [2.0, 6.0]}]')[0])
    assert (cut.start, cut.stop, cut.score, cut.dropped) == (50, 150, "", ())
    assert cut.lyrics == "la la"
    with pytest.raises(ValueError, match="no score and no bars"):
        track.plan(state, track.read('[{"op": "retake", "bars": [0, 2]}]')[0])


def test_a_cut_of_a_song_that_has_a_score_goes_by_bars():
    """Seconds would leave the score and the words describing a song that is no longer there."""
    with pytest.raises(ValueError, match="goes by whole bars"):
        track.plan(rap_state(), track.read('[{"op": "cut", "seconds": [2.0, 6.0]}]')[0])


def test_seconds_outside_the_song_are_refused_with_its_length():
    state = rap_state()
    with pytest.raises(ValueError, match="20.00 seconds long"):
        track.plan(state, track.read('[{"op": "retake", "seconds": [30.0, 40.0]}]')[0])


def test_the_bars_of_the_song_a_cut_leaves_are_where_the_sound_now_has_them():
    """Cutting the rap's verse takes bars 2 and 3 and the hundred frames they were: everything after
    comes back four seconds, and the chorus begins where the verse did."""
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "cut", "bars": [2, 4]}]')[0])
    after = track.after(state, step, 0)
    assert after.frames == FRAMES - 100
    assert len(after.sheet["bars"]) == 8
    assert after.clock.at(0) == pytest.approx(0.5)
    assert after.clock.at(2) == pytest.approx(4.5)
    assert after.clock.at(6) == pytest.approx(12.5)


def test_the_bars_after_a_retake_move_by_what_it_sang_and_the_ones_before_it_stay():
    """A retake of the chorus that comes out a second short pulls the end of the song a second
    earlier. The chorus opens at its pickup, frame 192, and closes past the outro's line, frame
    414, so its own bar lines are inside the new singing and are spread through it; the score
    keeps them, nobody knows where the new take put them."""
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "retake", "bars": [4, 8]}]')[0])
    assert (step.start, step.stop) == (192, 414)
    after = track.after(state, step, (step.stop - step.start) - 25)
    assert after.frames == FRAMES - 25
    assert after.sheet is state.sheet
    assert after.clock.at(2) == pytest.approx(4.5)
    assert after.clock.at(4) == pytest.approx(8.5 - 0.09, abs=0.01)
    assert after.clock.at(8) == pytest.approx(15.5, abs=0.01)
    assert after.clock.at(10) == pytest.approx(19.5)


def test_a_retake_that_came_out_as_long_as_it_was_leaves_the_grid_alone():
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "retake", "bars": [4, 8]}]')[0])
    assert track.after(state, step, step.stop - step.start).clock is state.clock


def test_a_second_edit_is_made_on_the_song_the_first_one_left():
    """After the verse is cut, the chorus is bars 2 to 6 and stands a hundred frames earlier, so
    its retake ends at 414 - 100. It opens earlier than it would have: the rap ran straight into
    the chorus and the seam was the margin before its pickup, and now the intro's rest lies
    before it, so the seam falls a beat before the pickup instead -- ten sixteenths before bar 2
    at 4.5 s, frame 81. The eleven frames between are the intro's rest, and singing them again
    costs nothing."""
    state = rap_state()
    first = track.plan(state, track.read('[{"op": "cut", "bars": [2, 4]}]')[0])
    assert (first.start, first.stop) == (92, 192)
    state = track.after(state, first, 0)
    second = track.plan(state, track.read('[{"op": "retake", "bars": [2, 6]}]')[0])
    assert (second.start, second.stop) == (81, 314)


def test_a_cut_that_runs_into_the_end_of_the_song_ends_the_bars_there():
    """A song can be shorter than its score. A cut of the last bars then takes less than their own
    length, and the lines after it cannot come back past where the cut began: the grid used to put
    the last line before the one before it (6.98 s after 14.5 s on this song of 30 s)."""
    sheet = notation.read(RAP)
    clock = grid.Grid(offset=0.5, rate=1.0, tick=grid.tick_seconds(sheet), starts=grid.starts_of(sheet))
    state = track.opened(types.SimpleNamespace(score=RAP, lyrics=LYRICS, frames=750), clock)
    step = track.plan(state, track.read('[{"op": "cut", "bars": [8, 10]}]')[0])
    assert (step.start, step.stop) == (412, 750)
    after = track.after(state, step, 0)
    assert after.frames == 412
    assert list(after.clock.lines) == pytest.approx([0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.48])
    assert len(after.clock.lines) == len(after.clock.starts) == len(after.sheet["bars"]) + 1


def test_a_cut_that_reaches_the_end_of_the_song_takes_the_bars_the_song_never_sang():
    """The song of 20 s stops inside bar 9 of 10. A cut of bars 4 to 9 takes the rest of the song,
    so the score loses bar 10 too, and says so. The score's last line stood half a second past
    the end of the song before the cut, and it still does after it."""
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "cut", "bars": [3, 9]}]')[0])
    assert (step.start, step.stop) == (212, 500)
    assert step.bars == (3, 10)
    assert any("bars 10 to 10 go with it" in text for _kind, text in step.notices)
    after = track.after(state, step, 0)
    assert len(after.sheet["bars"]) == 3
    assert list(after.clock.lines) == pytest.approx([0.5, 2.5, 4.5, 8.98])
    assert after.frames == 212


def test_the_beats_of_an_edited_bar_follow_its_own_length():
    """A retake that came out short makes the bars inside it shorter, and their beats with them."""
    state = rap_state()
    step = track.plan(state, track.read('[{"op": "retake", "bars": [4, 8]}]')[0])
    after = track.after(state, step, (step.stop - step.start) - 25)
    drawn = grid.layout(after.sheet, after.clock, after.frames)
    bar4, bar5 = drawn["bars"][4], drawn["bars"][5]
    beats = [beat for beat in drawn["beats"] if bar4 <= beat < bar5]
    assert len(beats) == 4
    assert beats[1] - beats[0] == pytest.approx((bar5 - bar4) / 4, abs=0.002)
    assert beats[3] < bar5


def test_infinity_is_not_a_second():
    """json reads Infinity; a selection of it would fall over as an OverflowError past every refusal."""
    for text in ('[{"op": "retake", "seconds": [1.0, Infinity]}]',
                 '[{"op": "retake", "seconds": [-Infinity, 1.0]}]',
                 '[{"op": "retake", "seconds": [NaN, 1.0]}]'):
        with pytest.raises(ValueError, match="not a number"):
            track.read(text)


def test_a_cut_reads_the_same_whatever_seed_or_take_it_was_written_with():
    """A cut has no seed and no take; one written with them must not get another name after a
    round trip through the field, or every take after it would be sung again."""
    plain = track.read('[{"op": "cut", "bars": [2, 4]}]')[0]
    dressed = track.read('[{"op": "cut", "bars": [2, 4], "seed": 42, "take": 0, "takes": 3}]')[0]
    assert plain == dressed
    assert track.name("song", [], plain) == track.name("song", [], dressed)
    assert track.read(track.written((dressed,))) == (plain,)


def test_the_words_before_the_first_tag_are_named_when_a_cut_takes_them():
    lyrics = "la la la\n\n[Verse]\none two three\nfour five six\n\n[Chorus]\nseven eight\n\n[Outro]"
    state = track.opened(rap_song(lyrics=lyrics), rap_state().clock)
    step = track.plan(state, track.read('[{"op": "cut", "bars": [0, 2]}]')[0])
    assert step.dropped == ("",)
    assert "the lines before the first tag" in step.notices[0][1]


def test_the_name_of_a_result_holds_the_song_the_edits_and_the_takes_kept():
    edits = track.read('[{"op": "cut", "bars": [2, 4]}, {"op": "retake", "bars": [2, 6], "seed": 5}]')
    cut, retake = edits
    first = track.name("song", [], cut)
    assert first == track.name("song", [], cut)
    assert first != track.name("other", [], cut)
    assert first != track.name("song", [(cut, 0)], cut)
    after = track.name("song", [(cut, 0)], retake)
    assert after != track.name("song", [(cut, 1)], retake)
    assert len(first) == 64


def test_asking_for_another_take_does_not_rename_what_is_already_sung():
    """More takes sings the one that is missing; the ones already there keep their name."""
    two = track.read('[{"op": "retake", "bars": [2, 6], "seed": 5, "takes": 2}]')[0]
    three = track.read('[{"op": "retake", "bars": [2, 6], "seed": 5, "takes": 3, "take": 2}]')[0]
    assert track.name("song", [], two) == track.name("song", [], three)
    assert three.seeds()[:2] == two.seeds()
