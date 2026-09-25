"""Moving a score to another key, proved note by note without the model.

What 'transpose' promises is narrow: every sounding note moves by the same
number of semitones, every chord and key moves with it, and nothing else about
the score changes. Upstream's own parser is the judge here, on a score written
to hit the awkward corners of the dialect and on ten scores the model really
wrote, in eight styles, with chords and without.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from yue2_comfy import constants, transpose
from yue2_comfy.vendor.yue2_music import abc_tools

HEADER = ('X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=90\n'
          'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
          'V: Ins clef=treble name="Ins Melody" snm="Inst."\n')

AWKWARD = HEADER + (
    'K:D\n% verse\nV: Vocal\n'
    '"D"F8A8"G/B"B8^c8-|"A7"c8e8-e8=c8|"E"^G8g8G16|\n'
    'V: Ins\nZ|z16F,16|Z|\n'
    '% chorus\nV: Vocal\nK:Bm\n'
    '"Bm"B,8^A,8B,16|"F#7"^a16z16|\n'
    'V: Ins\nK:Bm\nd8f8b16|Z|\n')
"""A tie across a barline, an accidental carried into another octave, a natural
cancelling the key, slash chords, full-bar rests and a key change between
groups: the places where shifting letters and nothing else goes wrong."""

AWKWARD_UP_2 = HEADER + (
    'K:E\n% verse\nV: Vocal\n'
    '"E"G8B8"A/C#"c8d8-|"B7"d8f8-f8=d8|"F#"^A8a8A16|\n'
    'V: Ins\nZ|z16G,16|Z|\n'
    '% chorus\nV: Vocal\nK:C#m\n'
    '"C#m"C8^B,8C16|"G#7"^b16z16|\n'
    'V: Ins\nK:C#m\ne8g8c\'16|Z|\n')

MODEL_SCORES = json.loads(
    (pathlib.Path(__file__).resolve().parent / "data" / "model_scores.json")
    .read_text(encoding="utf-8"))

STEPS = range(-constants.TRANSPOSE_LIMIT, constants.TRANSPOSE_LIMIT + 1)


def tiny(key, vocal="C8D8E8F8|"):
    """The smallest score upstream's parser accepts, in *key*."""
    return HEADER + "K:{}\n% verse\nV: Vocal\n{}\nV: Ins\nZ|\n".format(key, vocal)


def assert_moved(before_text, after_text, semitones):
    """Upstream's parser on both sides: every pitch up by *semitones*, all else equal."""
    before = abc_tools.parse_abc(before_text)
    after = abc_tools.parse_abc(after_text)
    assert after.bpm == before.bpm
    for name in abc_tools.VOICES:
        old, new = before.voices[name], after.voices[name]
        assert ([(onset, pitch + semitones, length) for onset, pitch, length in old.notes]
                == [tuple(note) for note in new.notes])
        assert old.bars == new.bars
        assert [onset for onset, _ in old.chords] == [onset for onset, _ in new.chords]
        assert [onset for onset, _ in old.keys] == [onset for onset, _ in new.keys]


def test_every_note_moves_by_the_same_step_and_nothing_else_moves():
    for semitones in STEPS:
        assert_moved(AWKWARD, transpose.move(AWKWARD, semitones).text, semitones)


def test_a_tone_up_reads_the_way_a_person_would_write_it():
    """Letters move with the key, and a mark is written only where it is needed.

    D major up a tone is E major, whose signature already sharpens F, C, G and
    D. The melody needs a mark exactly where the original had one the new key
    does not absorb: the natural that cancelled C-sharp now cancels D-sharp,
    and the G-sharp carried into the next octave is an A-sharp carried the same
    way. The redundant sharp on the tied c, which D major already gave it, is
    the one mark that is dropped.
    """
    moved = transpose.move(AWKWARD, 2)
    assert (moved.before, moved.after) == ("D", "E")
    assert moved.text == AWKWARD_UP_2.strip()


@pytest.mark.parametrize("key,semitones,expected", [
    ("C", 1, "Db"), ("C", -1, "B"), ("C", 6, "F#"), ("C", -6, "F#"),
    ("Am", 1, "Bbm"), ("F#", 1, "G"), ("C#m", -6, "Gm"),
    ("D", 11, "Db"), ("D", -11, "Eb"), ("Fm", 12, "Fm"),
])
def test_the_new_key_is_named_with_as_few_accidentals_as_it_can_be(key, semitones, expected):
    assert transpose.move(tiny(key), semitones).after == expected


def test_eleven_semitones_up_from_d_is_the_d_flat_an_octave_higher():
    """D and D-flat share a letter, and the letters still have to climb seven steps.

    Moving the letters by the distance between the two key names alone, which
    is zero here, would try to write each note eleven semitones away from its
    own letter.
    """
    moved = transpose.move(tiny("D", vocal="D8F8A8d8|"), 11)
    assert moved.after == "Db"
    assert "\nd8f8a8d'8|\n" in moved.text


RISING = HEADER + (
    'K:Bbm\n% verse\nV: Vocal\n"Bbm"B8d8f8b8|\nV: Ins\nZ|\n'
    '% chorus\nV: Vocal\nK:Bm\n"Bm"B8d8f8b8|\nV: Ins\nK:Bm\nZ|\n')
"""A song that rises a semitone from a key with flats to one with sharps, as a
transcription names B-flat minor and B minor since 0.9.3."""


def test_a_song_that_rises_a_semitone_moves_key_by_key_across_the_whole_range():
    for semitones in STEPS:
        assert_moved(RISING, transpose.move(RISING, semitones).text, semitones)
    up = transpose.move(RISING, 1)
    assert up.after == "Bm" and "\nK:Cm\n" in up.text and '"Cm"' in up.text


def test_a_score_the_model_wrote_comes_back_unchanged_from_a_move_of_zero():
    """The marks this module writes are the marks the model writes, character for character."""
    for name, score in MODEL_SCORES.items():
        assert transpose.move(score, 0).text == score.strip(), name


def test_every_score_the_model_wrote_moves_across_the_whole_range():
    for name, score in MODEL_SCORES.items():
        for semitones in STEPS:
            assert_moved(score, transpose.move(score, semitones).text, semitones)


def test_a_score_the_parser_cannot_read_is_refused_with_both_ways_out():
    with pytest.raises(ValueError) as error:
        transpose.move("X:1\nK:C\nCDEF|", 2)
    message = str(error.value)
    assert "'transpose' to 0" in message
    assert "another seed" in message


def test_a_move_past_an_octave_is_refused():
    with pytest.raises(ValueError) as error:
        transpose.move(AWKWARD, constants.TRANSPOSE_LIMIT + 1)
    assert "-12 to 12" in str(error.value)


def test_there_is_nothing_to_move_without_a_score():
    with pytest.raises(ValueError):
        transpose.move("  \n", 3)


def test_the_step_is_described_the_way_a_person_says_it():
    assert transpose.describe(2) == "up 2 semitones"
    assert transpose.describe(-1) == "down 1 semitone"
    assert transpose.describe(0) == "nowhere"
