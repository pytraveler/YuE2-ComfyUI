"""Where a song's bars sit and where an edit of them opens: the seams, the shifts, the line, the frames.

The scores here are small copies of what the stand's songs taught. A rap
verse whose words run straight into the chorus, which the chorus's own
phrases say begins a beat and a half early. A ballad whose phrases start
after rests. A dance song whose verse holds its last word across the bar line
while the chorus comes in two beats later. A chorus whose last word is struck
on the first beat of an instrumental. Every one of them has to open where the
voice does, or a cut takes a word with it. The grid itself is checked on a
synthetic song whose start and speed are known.
"""

from __future__ import annotations

import math

import pytest

from yue2_comfy import notation
from yue2_comfy.constants import FRAME_SECONDS
from yue2_comfy.inpaint import grid

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
"""Bars: intro 0-1, verse 2-3, chorus 4-7, outro 8-9. The verse's pickup follows a rest; the
chorus's does not, and its own phrases begin six sixteenths before their bars twice out of three."""

BALLAD = score(("intro", ["z16", "z16"]),
               ("verse", ["z2AAA2A2A2e4d2", "B4z12", "z2AAA2A2A2e4d2", "z6aaa4f2e2"]),
               ("chorus", ["B4z12", "z6ggg4e2d2", "A4z12", "z16"]))
"""Bars: intro 0-1, verse 2-5, chorus 6-9. The verse comes in after its downbeat; the chorus's first
phrase starts in the verse's last bar, after a rest."""

DANCE = score(("verse", ["z8c4e4", "e4d4d4B4", "c4e4e4f4-", "f4g4e4c4-"]),
              ("chorus", ["c4z4b2b2g2b2-", "b4f4z8", "z16", "z16"]))
"""Bars: verse 0-3, chorus 4-7. The verse's last word is held across the line; the chorus starts two
beats into its first bar."""

ENDING = score(("chorus", ["z8c4d4", "e4d4c4d4", "e8d4c4"]),
               ("interlude", ["d8z8", "z16"]))
"""Bars: chorus 0-2, interlude 3-4. The chorus's last word is struck on the interlude's first beat."""


def marks_of(text):
    return grid.seams(notation.read(text))


def test_a_pickup_after_a_rest_opens_the_verse_in_that_rest():
    seam = marks_of(RAP)[2]
    assert (seam.pickup, seam.low, seam.high) == (6, 6, 32)


def test_a_chorus_the_verse_runs_into_opens_where_its_own_phrases_begin():
    seam = marks_of(RAP)[4]
    assert (seam.pickup, seam.low, seam.high) == (6, 6, 6)
    assert seam.room == 0


def test_a_section_the_voice_enters_late_opens_before_its_first_note():
    marks = marks_of(BALLAD)
    assert (marks[2].pickup, marks[2].low, marks[2].high) == (-2, -2, 32)
    assert (marks[6].pickup, marks[6].low, marks[6].high) == (10, 10, 16)


def test_a_word_held_across_the_line_belongs_to_the_section_before():
    seam = marks_of(DANCE)[4]
    assert (seam.pickup, seam.low, seam.high) == (-8, -8, -4)


def test_a_last_word_struck_on_an_instrumental_belongs_to_the_chorus():
    seam = marks_of(ENDING)[3]
    assert (seam.pickup, seam.low, seam.high) == (-32, -32, -8)


def test_the_start_and_the_end_of_the_song_are_seams_of_their_own():
    marks = marks_of(RAP)
    assert len(marks) == 11
    assert (marks[0].bar, marks[0].low, marks[0].high) == (0, 0, 0)
    assert marks[-1].bar == 10
    assert marks_of(RAP)[8].high == 0


def test_phrases_are_runs_of_touching_notes():
    notes = [{"start": 0, "length": 4}, {"start": 4, "length": 2}, {"start": 8, "length": 2},
             {"start": 10, "length": 6}]
    assert grid.phrases(notes) == [(0, 6, [0, 4]), (8, 16, [8, 10])]


def test_the_usual_pickup_is_the_commonest_and_the_shorter_on_a_tie():
    sheet = notation.read(RAP)
    runs = grid.phrases(sheet["notes"]["Vocal"])
    assert grid.usual_pickup(sheet, runs, 5) == 6
    assert grid.usual_pickup(notation.read(ENDING), grid.phrases(notation.read(ENDING)["notes"]["Vocal"]), 3) is None


BEAT, MARGIN = 4.0, 0.5


def seam(pickup, low, high):
    return grid.Seam(1, float(pickup), float(low), float(high))


def test_a_cut_keeps_a_beat_clear_of_both_phrases_when_the_rests_allow():
    assert grid.cut_shift(seam(6, 6, 16), seam(2, 2, 20), BEAT, MARGIN) == 10


def test_a_cut_stays_on_the_downbeat_when_the_rest_spans_it():
    assert grid.cut_shift(seam(-8, -8, 16), seam(-6, -6, 12), BEAT, MARGIN) == 0


def test_a_cut_into_continuous_singing_protects_the_first_word_kept():
    assert grid.cut_shift(seam(6, 6, 16), seam(6, 6, 6), BEAT, MARGIN) == 6.5


def test_a_cut_whose_rests_do_not_meet_still_keeps_the_next_phrase():
    assert grid.cut_shift(seam(2, 2, 3), seam(8, 8, 8), BEAT, MARGIN) == 8


@pytest.mark.parametrize("mark,expected", [((10, 10, 16), 13), ((-2, -2, 32), 2), ((-8, -8, 16), 0),
                                           ((6, 6, 6), 6.5)])
def test_a_retake_opens_a_beat_before_its_phrase_at_most(mark, expected):
    assert grid.opening(seam(*mark), BEAT, MARGIN) == expected


def test_the_line_through_sections_gives_start_and_drift():
    offset, drift = grid.line([(10.0, 1.0, 1.0), (20.0, 1.1, 1.0), (30.0, 1.2, 1.0)])
    assert offset == pytest.approx(0.9)
    assert drift == pytest.approx(0.01)


def test_a_section_far_off_the_line_is_left_out():
    offset, drift = grid.line([(10.0, 1.0, 1.0), (20.0, 1.1, 1.0), (30.0, 1.2, 1.0), (40.0, 3.0, 0.5)])
    assert (offset, drift) == (pytest.approx(0.9), pytest.approx(0.01))


def test_a_steep_line_is_held_at_the_steepest_slope_through_the_mean():
    assert grid.line([(0.0, 0.0, 1.0), (10.0, 1.0, 1.0)]) == (pytest.approx(0.5 - 5 * grid.STEEPEST), grid.STEEPEST)
    assert grid.line([(0.0, 1.0, 1.0), (10.0, 0.0, 1.0)]) == (pytest.approx(0.5 + 5 * grid.STEEPEST), -grid.STEEPEST)
    assert grid.line([(5.0, 0.7, 2.0)]) == (pytest.approx(0.7), 0.0)
    assert grid.line([]) == (0.0, 0.0)


def test_the_grid_counts_frames_from_its_own_clock():
    clock = grid.Grid(offset=1.0, rate=1.01, tick=0.125, starts=(0, 16, 32))
    assert clock.seconds(8) == pytest.approx(2.01)
    assert clock.frame(8) == 50
    assert clock.frames(32) == 101


def rap_grid(offset=0.5):
    sheet = notation.read(RAP)
    return sheet, grid.Grid(offset=offset, rate=1.0, tick=grid.tick_seconds(sheet), starts=grid.starts_of(sheet))


def test_a_cut_of_whole_sections_takes_out_whole_bars_moved_back_together():
    """The rap's verse runs from six sixteenths before bar 2 into bar 4 without a rest, and the chorus
    usually starts six before its bars too, so the rests to cut in barely meet and the shift is the margin
    past that: 6.64 sixteenths. Bar 2 is at 0.5 + 4.0 s, less 6.64 * 0.125 s, at frame 91.75; the two
    bars cut are 4.0 s, a hundred frames."""
    sheet, clock = rap_grid()
    marks = grid.seams(sheet)
    assert marks[2] == grid.Seam(2, 6.0, 6.0, 32.0)
    assert marks[4] == grid.Seam(4, 6.0, 6.0, 6.0)
    assert grid.cut_frames(sheet, clock, marks, 2, 4, 1000) == (92, 192)


def test_a_cut_to_the_end_takes_the_rest_and_one_from_the_start_begins_at_zero():
    """Before the verse's pickup lies a rest of 26 sixteenths, so the cut from the start stops a beat
    before the pickup: ten before bar 2, 3.25 s, frame 81.25."""
    sheet, clock = rap_grid()
    marks = grid.seams(sheet)
    assert grid.cut_frames(sheet, clock, marks, 4, 10, 1000)[1] == 1000
    assert grid.cut_frames(sheet, clock, marks, 0, 2, 1000) == (0, 81)


def test_an_edit_of_bars_the_song_never_reached_is_refused():
    """A song can come out shorter than its score; there is nothing to edit past its end."""
    sheet, clock = rap_grid()
    marks = grid.seams(sheet)
    with pytest.raises(ValueError, match="not a stretch of the song"):
        grid.cut_frames(sheet, clock, marks, 2, 4, 5)
    with pytest.raises(ValueError, match="not a stretch of the song"):
        grid.retake_frames(sheet, clock, marks, 2, 4, 5)
    with pytest.raises(ValueError, match="not a stretch of the song"):
        grid.cut_frames(sheet, grid.Grid(offset=-100.0, rate=1.0, tick=clock.tick, starts=clock.starts),
                        marks, 0, 2, 1000)
    with pytest.raises(ValueError, match="not bars of this score"):
        grid.cut_frames(sheet, clock, marks, 4, 2, 1000)
    with pytest.raises(ValueError, match="not bars of this score"):
        grid.retake_frames(sheet, clock, marks, 0, len(sheet["bars"]) + 1, 1000)


def test_a_retake_opens_each_end_by_its_own_seam():
    """The chorus opens the margin before its usual pickup, 6.64 sixteenths before bar 4 at 8.5 s: frame
    191.75. Nothing is sung after its last note, which ends on the line of bar 8 at 16.5 s, so the retake
    ends the margin after that note, frame 414.5, rounded to even."""
    sheet, clock = rap_grid()
    marks = grid.seams(sheet)
    assert marks[8] == grid.Seam(8, -32.0, -32.0, 0.0)
    assert grid.retake_frames(sheet, clock, marks, 4, 8, 1000) == (192, 414)
    assert grid.retake_frames(sheet, clock, marks, 0, 10, 1000) == (0, 1000)


SYNTHETIC = score(
    ("intro", ['"C"z16', '"Am"z16']),
    ("verse", ['"C"c4e4g4e4', '"Am"a4g4e4c4', '"F"f4a4c\'4a4', '"G"g4b4d\'4z4']),
    ("chorus", ['"F"c\'8a8', '"G"b8g8', '"C"e4g4c\'8', '"G"d\'8z8']),
    ("verse", ['"C"e4c4e4g4', '"Am"a8e8', '"F"a4c\'4f4a4', '"G"g8d4z4']),
    ("chorus", ['"F"c\'8a8', '"G"b8g8', '"C"e4g4c\'8', '"G"d\'8z8']),
    ("outro", ['"C"z16', '"C"z16']))


def test_a_section_weighs_what_it_sings():
    sheet = notation.read(RAP)
    assert grid.sung_ticks(sheet) == [6, 32, 34, 0]
    assert grid.sung_ticks(notation.read(ENDING))[1] == 8


def test_the_layout_gives_the_window_bars_beats_and_where_each_section_sings():
    sheet = notation.read(RAP)
    clock = grid.Grid(offset=0.5, rate=1.0, tick=grid.tick_seconds(sheet), starts=grid.starts_of(sheet))
    drawn = grid.layout(sheet, clock, 500)
    assert drawn["seconds"] == round(500 * FRAME_SECONDS, 3)
    assert drawn["by_voice"] is False
    assert len(drawn["bars"]) == len(sheet["bars"]) + 1
    assert drawn["bars"][0] == 0.5 and drawn["bars"][2] == pytest.approx(0.5 + 2 * 2.0)
    assert len(drawn["beats"]) == 4 * len(sheet["bars"])
    chorus = [s for s in drawn["sections"] if s["name"] == "chorus"][0]
    assert (chorus["bar"], chorus["bars"]) == (4, 4)
    assert chorus["start"] == pytest.approx(0.5 + 4 * 2.0)
    assert chorus["sung"] == pytest.approx(chorus["start"] - 6 * clock.tick)
    assert chorus["end"] == pytest.approx(0.5 + 8 * 2.0)
    assert [s for s in drawn["sections"] if s["name"] == "outro"][0]["sung"] is None


RATE = 16000


def sung_and_played(sheet, offset: float, rate: float):
    """The synthetic song at ``offset`` seconds and ``rate`` of its score's tempo: its mix and its voice alone."""
    import torch

    from yue2_comfy.midi.export import chord_pitches

    tick = grid.tick_seconds(sheet)
    seconds = offset + rate * tick * sheet["total"] + 1.0
    clock = torch.arange(int(seconds * RATE)) / RATE
    voice, band = torch.zeros_like(clock), torch.zeros_like(clock)

    def tone(into, pitch, start, end, level):
        a, b = int((offset + rate * tick * start) * RATE), int((offset + rate * tick * end) * RATE)
        hz = 440.0 * 2 ** ((pitch - 69) / 12.0)
        into[a:b] += level * torch.sin(2 * math.pi * hz * clock[a:b])

    for note in sheet["notes"]["Vocal"]:
        tone(voice, note["pitch"], note["start"], note["start"] + note["length"], 1.0)
    for index, chord in enumerate(sheet["chords"]):
        end = sheet["chords"][index + 1]["start"] if index + 1 < len(sheet["chords"]) else sheet["total"]
        for pitch in chord_pitches(chord["name"]):
            tone(band, pitch + 12, chord["start"], end, 0.3)
    return torch.stack([voice + band, voice + band]), torch.stack([voice, voice])


def test_the_grid_finds_the_start_and_the_speed_of_a_song_it_is_laid_on():
    pytest.importorskip("torch")
    sheet = notation.read(SYNTHETIC)
    mix, _voice = sung_and_played(sheet, 1.3, 1.006)
    found = grid.measured(sheet, mix, RATE)
    assert found.offset == pytest.approx(1.3, abs=0.03)
    assert found.rate == pytest.approx(1.006, abs=0.002)
    assert found.starts == grid.starts_of(sheet)
    assert found.by_voice is False


def test_the_passes_take_up_a_drift_steeper_than_one_pass_may_stretch(monkeypatch):
    """A song five percent fast is four seconds off its score at two minutes, twice the window a section is looked for in."""
    pytest.importorskip("torch")
    sheet = notation.read(SYNTHETIC)
    mix, _voice = sung_and_played(sheet, 1.3, 1.05)
    found = grid.measured(sheet, mix, RATE)
    assert found.offset == pytest.approx(1.3, abs=0.03)
    assert found.rate == pytest.approx(1.05, abs=0.002)
    monkeypatch.setattr(grid, "PASSES", 1)
    once = grid.measured(sheet, mix, RATE)
    assert once.rate == pytest.approx(1.0 + grid.STEEPEST)
    assert abs(once.offset - 1.3) > 0.2


def test_the_voice_places_a_song_that_starts_further_out_than_the_pitch_classes_look():
    pytest.importorskip("torch")
    sheet = notation.read(SYNTHETIC)
    far = grid.SEARCH_SECONDS + 4.0
    mix, voice = sung_and_played(sheet, far, 1.0)
    heard = grid.measured(sheet, mix, RATE, voice=voice)
    assert heard.by_voice is True
    assert heard.offset == pytest.approx(far, abs=0.05)
    blind = grid.measured(sheet, mix, RATE)
    assert blind.by_voice is False
    assert abs(blind.offset - far) > 1.0


def test_a_score_that_sings_nothing_leaves_the_voice_out_of_it():
    pytest.importorskip("torch")
    sheet = notation.read(score(("intro", ['"C"z16', '"C"z16'])))
    mix, voice = sung_and_played(notation.read(SYNTHETIC), 1.0, 1.0)
    assert grid.measured(sheet, mix, RATE, voice=voice).by_voice is False
