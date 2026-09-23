"""A break of playing put between two sections: the edit, the score, the words, the frames and the grid after it.

The model writes a break's bars itself and plays them (``core.broke``); what
can go wrong around that is here: where the break opens, that the first phrase
of the section after it goes on after it and the bars before keep their rest,
that the break holds no note to sing whatever the model wrote there, that the
words get their tag where the break is, and that the grid after it puts every
bar where its sound now is. Nothing here needs torch.
"""

from __future__ import annotations

import json

import pytest

from yue2_comfy import notation
from yue2_comfy.constants import CONTEXT, FRAME_SECONDS
from yue2_comfy.inpaint import grid, ops, track

from test_inpaint_track import HEADER, LYRICS, RAP, rap_state, score

PICKUP = score(("intro", ["z16", "z16"]), ("verse", ["C4D4E4F4", "G8z4C2D2"]),
               ("chorus", ["E8z8", "F16"]), ("outro", ["z16", "z16"]))
"""A song whose chorus comes in on a pickup: C and D at the end of the verse's last bar."""

PLAYED = "V: Vocal\nc4d4e4f4|g16|\nV: Ins\nC4E4G4c4|B16|\n"
"""Two bars the model might write for a break, with notes to sing in them that are not sung."""


def pickup_state():
    sheet = notation.read(PICKUP)
    clock = grid.Grid(offset=0.5, rate=1.0, tick=grid.tick_seconds(sheet),
                      starts=grid.starts_of(sheet))
    song = type("Song", (), {"score": PICKUP, "lyrics": LYRICS, "frames": 500})()
    return track.opened(song, clock)


def a_break(to=4, length=2, **more):
    item = dict({"op": "break", "to": to, "length": length, "seed": 5}, **more)
    return track.read(json.dumps([item]))[0]


def seam_of(state, to):
    beat, margin = grid._beat_and_margin(state.sheet, state.clock)
    opened = grid.opening(grid.seams(state.sheet)[to], beat, margin)
    return opened, int(round(state.sheet["bars"][to]["start"] - opened))


def notes(sheet, part="Vocal"):
    return sorted((note["start"], note["length"], note["pitch"]) for note in sheet["notes"][part])


def test_a_break_reads_and_writes_as_the_window_writes_it():
    edit = track.read('[{"op": "break", "to": 4, "length": 2, "seed": 5, "takes": 3, "take": 1}]')[0]
    assert (edit.op, edit.to, edit.length, edit.bars, edit.seconds) == ("break", 4, 2, None, None)
    assert (edit.seed, edit.takes, edit.take) == (5, 3, 1)
    assert track.read(track.written([edit])) == (edit,)
    assert track.read('[{"op": "break", "to": 4}]', takes=2)[0].length == track.BREAK_LENGTH, (
        "a break that does not say how long plays four bars")


@pytest.mark.parametrize("item,said", [
    ({"op": "break", "length": 4}, "'to' names"),
    ({"op": "break", "to": 4, "bars": [4, 6]}, "no bars and no seconds"),
    ({"op": "break", "to": 4, "seconds": [1.0, 2.0]}, "no bars and no seconds"),
    ({"op": "break", "to": -1}, "before the song"),
    ({"op": "break", "to": 4, "length": 0}, "1 to 16"),
    ({"op": "break", "to": 4, "length": 17}, "1 to 16"),
    ({"op": "break", "to": 4, "length": 2.5}, "not a whole number"),
    ({"op": "break", "to": 4, "length": None}, "not a whole number"),
])
def test_a_break_that_says_nowhere_or_no_length_is_refused(item, said):
    with pytest.raises(ValueError, match=said):
        track.read(json.dumps([item]))


def test_a_break_opens_where_a_retake_of_the_section_after_it_would_and_takes_nothing_out():
    state = pickup_state()
    step = track.plan(state, a_break(4, 2))
    opened, seam = seam_of(state, 4)
    assert step.kind == "break"
    assert step.start == step.stop == state.clock.moment(4, opened)
    assert step.start < state.clock.frame(state.sheet["bars"][4]["start"]), (
        "it opens before the downbeat, where the pickup begins")
    assert (step.bars, step.seam) == ((4, 6), seam)
    assert seam == state.sheet["bars"][3]["start"] + 10, (
        "in the rest between the verse's G and the pickup's C, a beat's half clear of both")


def test_a_break_goes_before_a_section_and_inside_the_song():
    state = rap_state()
    for to, said in ((0, "after the first bar"), (10, "after the first bar"),
                     (5, "inside the chorus"), (3, "inside the verse")):
        with pytest.raises(ValueError, match=said):
            track.plan(state, a_break(to))


def test_a_song_with_no_score_has_no_sections_to_break_between():
    song = type("Song", (), {"score": "", "lyrics": LYRICS, "frames": 500})()
    with pytest.raises(ValueError, match="no sections to put a break between"):
        track.plan(track.opened(song, None), a_break())


def test_the_score_the_model_writes_a_break_from_ends_before_the_pickup_under_an_interlude():
    state = pickup_state()
    _opened, seam = seam_of(state, 4)
    head = notation.interlude_head(PICKUP, 4, seam)
    assert head.endswith("% interlude\n") and head.count("% interlude") == 1
    read = notation.read(head[:-len("% interlude\n")])
    assert len(read["bars"]) == 4
    assert notes(read) == [note for note in notes(state.sheet) if note[0] < seam], (
        "the pickup is the chorus's first words, which come after the break")
    assert head.startswith(notation.head(PICKUP, 4)[:len(HEADER)])


def test_the_break_is_played_and_the_pickup_goes_on_after_it_with_its_section():
    state = pickup_state()
    _opened, seam = seam_of(state, 4)
    head = notation.interlude_head(PICKUP, 4, seam)
    broken = notation.interluded(PICKUP, head + PLAYED, 4, 2, seam)
    after = notation.read(broken)
    old = state.sheet
    shift = 2 * old["bars"][4]["length"]
    assert len(after["bars"]) == len(old["bars"]) + 2
    assert [(s["name"], s["bar"]) for s in after["sections"]] == [
        ("intro", 0), ("verse", 2), ("interlude", 4), ("chorus", 6), ("outro", 8)]
    before = [note for note in notes(old) if note[0] < seam]
    moved = [(start + shift, length, pitch) for start, length, pitch in notes(old) if start >= seam]
    assert notes(after) == before + moved
    pickup = [note for note in moved if note[0] < after["bars"][6]["start"]]
    assert pickup and all(after["bars"][5]["start"] <= note[0] for note in pickup), (
        "the pickup ends the break's last bar, as it ended the verse")
    played = [note for note in notes(after, "Ins")
              if after["bars"][4]["start"] <= note[0] < after["bars"][6]["start"]]
    assert len(played) == 5, "the break plays what the model wrote for it"
    assert notation.interluded(PICKUP + "\n\n", head + PLAYED, 4, 2, seam).endswith("\n\n\n"), (
        "the text around the score stays")


def test_a_break_the_model_stopped_writing_halfway_is_refused_to_be_written_again():
    state = pickup_state()
    _opened, seam = seam_of(state, 4)
    head = notation.interlude_head(PICKUP, 4, seam)
    for cut in (PLAYED[:PLAYED.index("V: Ins")], PLAYED[:-6], ""):
        with pytest.raises(ValueError, match="whole bars of the break"):
            notation.interluded(PICKUP, head + cut, 4, 2, seam)
    longer = "V: Vocal\nc4d4e4f4|g16|a16|b16|\nV: Ins\nZ4|\n% chorus\nV: Vocal\nc16|\nV: Ins\nZ|\n"
    assert len(notation.read(notation.interluded(PICKUP, head + longer, 4, 2, seam))["bars"]) == 10, (
        "bars written past the break are not the break")


def test_a_tie_into_the_break_comes_off_and_what_it_held_is_struck_again_after_it():
    tied = score(("intro", ["z16", "z16"]), ("verse", ["C16", "C8z6D2-"]),
                 ("chorus", ["D8E8", "F16"]), ("outro", ["z16"]))
    sheet = notation.read(tied)
    broken = notation.read(notation.interluded(tied, notation.interlude_head(
        tied, 4, sheet["bars"][4]["start"]) + PLAYED, 4, 2, sheet["bars"][4]["start"]))
    assert (sheet["bars"][3]["start"] + 14, 2, 62) in notes(broken), "the D before the break is short"
    assert (broken["bars"][6]["start"], 8, 62) in notes(broken), "and the chorus strikes its own"


def test_the_words_get_an_interlude_where_the_break_is():
    text = ops.break_words(LYRICS, RAP, 4)
    assert text.matched
    assert text.text == "[Intro]\n\n[Verse]\none two three\nfour five six\n\n[Interlude]\n\n[Chorus]" \
        "\nseven eight\n\n[Outro]"
    last = ops.break_words(LYRICS, RAP, 8)
    assert last.text.endswith("seven eight\n\n[Interlude]\n\n[Outro]"), (
        "after the last words sung, ahead of the tags with nothing under them that close them")
    bare = ops.break_words("[Verse]\none two three\n\n[Chorus]\nseven eight", RAP, 8)
    assert bare.text == "[Verse]\none two three\n\n[Chorus]\nseven eight\n\n[Interlude]"


def test_words_paired_only_as_far_as_the_break_still_take_its_tag():
    """A user's words had a verse where the model sang a bridge near the end; that is past the break."""
    later = LYRICS.replace("[Outro]", "[Bridge]\nnine ten\n\n[Outro]")
    text = ops.break_words(later, RAP, 4)
    assert text.matched and "four five six\n\n[Interlude]\n\n[Chorus]" in text.text
    lost = ops.break_words("[Bridge]\nnine ten\n\n[Chorus]\nseven eight", RAP, 4)
    assert (lost.matched, lost.text) == (False, "[Bridge]\nnine ten\n\n[Chorus]\nseven eight")


def test_a_break_whose_words_cannot_be_placed_says_so():
    state = rap_state()
    unmatched = type(state)(lyrics="[Bridge]\nnine ten", score=state.score, sheet=state.sheet,
                            clock=state.clock, frames=state.frames)
    step = track.plan(unmatched, a_break(4))
    assert step.notices == (("warn", track.UNMATCHED),) and step.lyrics == "[Bridge]\nnine ten"


def test_the_region_of_a_break_takes_nothing_out_and_leaves_room_for_the_join():
    region = ops.insert(100, 500, 120, 300)
    assert (region.start, region.stop, region.removed, region.length) == (100, 100, 0, 120)
    assert (region.shortest, region.longest) == (80, 160)
    for start in (0, 500):
        with pytest.raises(ValueError, match="between two frames"):
            ops.insert(start, 500, 120, 300)
    with pytest.raises(ValueError, match="no room"):
        ops.insert(100, 500, 120, CONTEXT - 100 - 120 - ops.JOIN_FRAMES + 1)
    assert ops.insert(100, 500, 120, CONTEXT - 100 - 120 - ops.JOIN_FRAMES - 10).width == 10, (
        "the search narrows to what the context has left")


def test_the_grid_after_a_break_moves_what_comes_after_it_by_what_came_in():
    state = pickup_state()
    _opened, seam = seam_of(state, 4)
    broken = notation.interluded(PICKUP, notation.interlude_head(PICKUP, 4, seam) + PLAYED, 4, 2, seam)
    step = track.plan(state, a_break(4, 2))
    count = track.break_frames(state, step, notation.read(broken))
    assert count == state.clock.frames(2 * state.sheet["bars"][4]["length"])
    after = track.after(state, step, count + 3, broken)
    old = state.clock.bar_seconds()
    new = after.clock.bar_seconds()
    moved = (count + 3) * FRAME_SECONDS
    assert new[:5] == old[:5], "the break's first downbeat is where the chorus's was"
    assert new[5] == pytest.approx(old[4] + moved / 2)
    assert new[6:] == pytest.approx([second + moved for second in old[4:]])
    assert after.frames == state.frames + count + 3
    assert (after.score, after.lyrics) == (broken, step.lyrics)
    assert after.sheet["sections"][2]["name"] == "interlude"
