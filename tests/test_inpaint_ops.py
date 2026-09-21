"""The plan of an edit, in plain Python: regions, noise runs, and the words a cut leaves.

None of this needs the model, so it is held exactly here, where CI runs. A
region has to ask for what the stand measured -- a join that may move two
seconds, less on a short stretch, never past the model's context. The noise of
an edited song has to name, for every frame, the very rows it was drawn from.
And a cut has to take out of the lyrics the sections it takes out of the song,
and leave every other character where it was.
"""

from __future__ import annotations

import pytest

from yue2_comfy.constants import CONTEXT
from yue2_comfy.inpaint import ops

HEADER = ('X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=90\n'
          'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
          'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n')

SUNG = "C8D8E8F8"
HUM = "C32"
REST = "Z"


def score(*sections):
    """A score of ``(name, bars)`` sections, the Ins part resting, at most four bars a line."""
    lines = [HEADER.rstrip("\n")]
    for name, bars in sections:
        lines.append("% " + name)
        for start in range(0, len(bars), 4):
            group = bars[start:start + 4]
            lines += ["V: Vocal", "|".join(group) + "|", "V: Ins", "|".join(["Z"] * len(group)) + "|"]
    return "\n".join(lines) + "\n"


SONG = score(("intro", [HUM, REST]), ("verse", [SUNG] * 4), ("chorus", [SUNG] * 4),
             ("verse", [SUNG] * 4), ("chorus", [SUNG] * 4), ("outro", [HUM, REST]))
"""Bars: intro 0-1, verse 2-5, chorus 6-9, verse 10-13, chorus 14-17, outro 18-19.
The intro and the outro have a vocal note each, as the model's scores often do."""

LYRICS = "[Verse]\none two three\nfour five six\n\n[Chorus]\nseven eight\n\n" \
         "[Verse]\nnine ten\neleven twelve\n\n[Chorus]\nthirteen fourteen"


def test_a_retake_may_move_its_join_two_seconds_either_way():
    region = ops.retake(740, 1269, 2497, 1145)
    assert (region.start, region.stop, region.length, region.width) == (740, 1269, 529, 50)
    assert (region.shortest, region.longest, region.removed) == (479, 579, 529)


def test_a_short_retake_moves_its_join_a_third_of_itself_at_most():
    assert ops.retake(100, 130, 1000, 500).width == 10
    tiny = ops.retake(100, 102, 1000, 500)
    assert tiny.width == 0
    assert (tiny.shortest, tiny.longest) == (2, 2)


def test_the_join_search_narrows_to_the_room_the_context_leaves():
    prompt = CONTEXT - 1000 - 20
    assert ops.retake(900, 1000, 1000, prompt).width == 20
    with pytest.raises(ValueError, match="no room"):
        ops.retake(900, 1000, 1000, CONTEXT - 999)


def test_a_cut_asks_for_nothing_new():
    region = ops.cut(10, 20, 100)
    assert (region.length, region.width, region.shortest, region.longest, region.removed) == (0, 0, 0, 0, 10)


@pytest.mark.parametrize("start,stop,frames", [(-1, 10, 100), (10, 10, 100), (20, 10, 100), (0, 101, 100)])
def test_a_stretch_outside_the_song_is_refused(start, stop, frames):
    with pytest.raises(ValueError, match="not a stretch"):
        ops.retake(start, stop, frames, 10)
    with pytest.raises(ValueError, match="not a stretch"):
        ops.cut(start, stop, frames)


def test_an_edit_is_counted_in_whole_frames():
    for start in (1.5, True):
        with pytest.raises(ValueError, match="whole frames"):
            ops.retake(start, 10, 100, 10)


def test_cutting_the_whole_song_is_refused():
    with pytest.raises(ValueError, match="whole song"):
        ops.cut(0, 100, 100)


def test_the_runs_of_a_stretch_keep_the_rows_their_frames_had():
    assert ops.runs_between([[7, 0, 100]], 10, 20) == [[7, 10, 10]]
    edited = [[7, 0, 5], [9, 0, 3], [7, 8, 4]]
    assert ops.runs_between(edited, 3, 10) == [[7, 3, 2], [9, 0, 3], [7, 8, 2]]
    assert ops.runs_between(edited, 0, 12) == edited
    assert ops.runs_between(edited, 12, 20) == []


def test_runs_that_go_on_with_one_draw_are_merged_and_no_others():
    assert ops.joined([[7, 0, 5]], [[7, 5, 3]]) == [[7, 0, 8]]
    assert ops.joined([[7, 0, 5]], [[7, 6, 3]]) == [[7, 0, 5], [7, 6, 3]]
    assert ops.joined([[7, 0, 5]], [[8, 5, 3]]) == [[7, 0, 5], [8, 5, 3]]
    assert ops.joined([], [[7, 0, 5]], []) == [[7, 0, 5]]


def test_a_retake_gives_its_new_frames_the_first_rows_of_its_seed():
    region = ops.retake(10, 20, 100, 50)
    runs = ops.edited_noise([[7, 0, 100]], region, 3, 12)
    assert runs == [[7, 0, 10], [3, 0, 12], [7, 20, 80]]
    assert sum(run[2] for run in runs) == 102


def test_a_cut_keeps_the_rows_on_both_sides():
    assert ops.edited_noise([[7, 0, 100]], ops.cut(10, 20, 100), 0, 0) == [[7, 0, 10], [7, 20, 80]]


def test_an_edit_of_an_edited_song_replaces_the_rows_of_the_edit_it_covers():
    first = ops.edited_noise([[7, 0, 100]], ops.retake(10, 20, 100, 50), 3, 12)
    again = ops.edited_noise(first, ops.retake(10, 22, 102, 50), 4, 11)
    assert again == [[7, 0, 10], [4, 0, 11], [7, 20, 80]]
    undone = ops.edited_noise(first, ops.cut(10, 22, 102), 0, 0)
    assert undone == [[7, 0, 10], [7, 20, 80]]


def test_the_lyrics_blocks_give_the_text_back_to_the_character():
    text = "  loose words\r\n\r\n[Verse 1] \r\nline one  \r\n\n[Hook]\nsung\n(ooh)\n..."
    blocks = ops._blocks(text)
    assert "".join("".join(block["raw"]) for block in blocks) == text
    assert [(block["tag"], block["lines"]) for block in blocks] == [("", 1), ("Verse 1", 1), ("Hook", 2)]


def test_a_cut_of_a_whole_section_takes_its_words_out():
    with pytest.raises(ValueError, match="not bars of this score"):
        ops.cut_words(LYRICS, SONG, 10, 99)
    words = ops.cut_words(LYRICS, SONG, 10, 14)
    assert words.matched and words.dropped == ("Verse",)
    assert words.text == "[Verse]\none two three\nfour five six\n\n[Chorus]\nseven eight\n\n" \
                         "[Chorus]\nthirteen fourteen"


def test_the_intro_and_the_outro_are_stepped_over_even_with_notes_in_them():
    assert [section[1] > 0 for section in ops._sections(SONG, 0, 1)] == [True] * 6
    words = ops.cut_words(LYRICS, SONG, 14, 18)
    assert words.dropped == ("Chorus",)
    assert words.text.endswith("[Verse]\nnine ten\neleven twelve")


def test_a_section_goes_when_the_cut_takes_most_of_its_notes():
    """The pickup of the next section is left behind, and the last bar of the one before is taken."""
    words = ops.cut_words(LYRICS, SONG, 9, 13)
    assert words.dropped == ("Verse",)
    assert "seven eight" in words.text and "nine ten" not in words.text


def test_a_section_that_loses_half_or_less_keeps_every_line():
    words = ops.cut_words(LYRICS, SONG, 12, 16)
    assert words.matched and words.dropped == ()
    assert words.text == LYRICS


def test_lyrics_that_cannot_be_paired_with_the_score_are_left_as_they_were():
    lyrics = "[Verse]\na b\n\n[Bridge]\nc d\n\n[Chorus]\ne f"
    words = ops.cut_words(lyrics, SONG, 10, 14)
    assert not words.matched and words.dropped == () and words.text == lyrics


def test_sections_the_names_do_not_pair_are_paired_by_order_when_the_counts_agree():
    named = score(("a", [SUNG] * 2), ("b", [SUNG] * 2), ("c", [REST]))
    words = ops.cut_words("[Verse]\nx y\n\n[Chorus]\nz w", named, 2, 4)
    assert words.matched and words.dropped == ("Chorus",)
    assert words.text == "[Verse]\nx y"


def test_words_before_the_first_tag_are_a_verse():
    lyrics = "one two\nthree four\n\n[Chorus]\nfive six\n\n[Verse]\nseven\n\n[Chorus]\neight"
    words = ops.cut_words(lyrics, SONG, 2, 6)
    assert words.dropped == ("",)
    assert words.text == "[Chorus]\nfive six\n\n[Verse]\nseven\n\n[Chorus]\neight"
