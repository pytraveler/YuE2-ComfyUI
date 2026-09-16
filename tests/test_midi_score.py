"""A MIDI file as a score YuE2 sings: timing, grid, lines, octave, key, sections, words and chords.

Every score is read back with upstream's own reader, abc_tools, and checked as
the notes it sounds, so the tests hold whatever spelling the writer chooses.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from midi_files import chord, kar_texts, line, song_of

from yue2_comfy import edits
from yue2_comfy.midi import score
from yue2_comfy.vendor.yue2_music import abc_tools

SCALE = [60, 62, 64, 65, 67, 69, 71, 72]


def sung(abc, voice="Vocal"):
    return [(onset, pitch, length) for onset, pitch, length in abc_tools.parse(abc).voices[voice].notes]


def quarters(pitches, length=1):
    return [(Fraction(index * length), pitch, Fraction(length)) for index, pitch in enumerate(pitches)]


def test_a_plain_tune_is_written_note_for_note_at_its_tempo():
    out = score.convert(song_of(("Tune", 0, 0, line(SCALE), [])))
    header = out["abc"].splitlines()[:5]
    assert header[2:5] == ["M:4/4", "L:1/16", "Q:1/4=120"]
    assert sung(out["abc"]) == quarters(SCALE)
    assert out["lyrics"] == "[Verse]"
    assert out["facts"]["bars"] == 2 and out["facts"]["seconds"] == 4.0 and out["facts"]["grid"] == 16
    assert out["notices"] == []


def test_thirty_second_notes_get_a_grid_of_their_own():
    out = score.convert(song_of(("Tune", 0, 0, line(SCALE, step=60), [])))
    assert "L:1/32" in out["abc"].splitlines()
    assert sung(out["abc"]) == quarters(SCALE, Fraction(1, 8))


def test_notes_played_in_by_hand_are_rounded_to_sixteenths_and_the_node_says_so():
    notes = [(0, 470, 60), (490, 950, 62), (950, 1450, 64), (1430, 1920, 65)]
    out = score.convert(song_of(("Tune", 0, 0, notes, [])))
    assert sung(out["abc"]) == quarters([60, 62, 64, 65])
    assert score.PLAYED_IN in out["notices"]


@pytest.mark.parametrize("pitches, moved, words", [
    ([41, 43, 45, 43], 24, "moved up 2 octaves"),
    ([90, 91, 93, 91], -12, "moved down an octave"),
    ([70, 72, 74, 72], 0, None),
])
def test_a_line_outside_where_yue2_keeps_the_voice_is_moved_by_octaves(pitches, moved, words):
    out = score.convert(song_of(("Tune", 0, 0, line(pitches), [])))
    assert sung(out["abc"]) == quarters([pitch + moved for pitch in pitches])
    assert out["facts"]["voice_shift"] == moved
    assert any(words in notice for notice in out["notices"]) if words else out["notices"] == []


def test_the_top_note_of_a_chord_is_kept_and_the_octave_follows_the_line_that_is_kept():
    notes = chord([60, 72], 0, 480) + chord([62, 86], 480, 960) + chord([64, 88], 960, 1440)
    out = score.convert(song_of(("Lead", 0, 0, notes, [])))
    assert sung(out["abc"]) == quarters([60, 74, 76])
    assert out["facts"]["voice_shift"] == -12
    assert any("strikes 3 notes together" in notice for notice in out["notices"])


def test_a_lower_note_under_a_held_one_is_dropped_unless_most_of_it_comes_after():
    notes = [(0, 960, 72), (100, 520, 64), (800, 1920, 67)]
    assert sung(score.convert(song_of(("Tune", 0, 0, notes, [])))["abc"]) == [
        (Fraction(0), 72, Fraction(2)), (Fraction(2), 67, Fraction(2))]


def test_a_key_signature_that_fits_is_kept_and_one_that_does_not_is_replaced():
    tune = line([62, 66, 69, 74, 69, 66, 62, 62])
    kept = score.convert(song_of(("Tune", 0, 0, tune, []), keys=((0, 2, False),)))
    assert "K:D" in kept["abc"].splitlines() and kept["facts"]["key_source"] == "file"
    wrong = score.convert(song_of(("Tune", 0, 0, line([68, 72, 75, 80, 75, 72, 68, 68]), []), keys=((0, 0, False),)))
    assert wrong["facts"]["key"] == "Ab" and wrong["facts"]["key_source"] == "estimated"


def test_meters_come_from_the_file():
    changed = score.convert(song_of(("Tune", 0, 0, line(SCALE[:7]), []), meters=((0, 4, 4), (1920, 3, 4))))
    assert "M:3/4" in changed["abc"].splitlines() and changed["facts"]["bars"] == 2
    compound = score.convert(song_of(("Tune", 0, 0, line(SCALE[:6], step=240), []), meters=((0, 6, 8),)))
    assert compound["abc"].splitlines()[2:4] == ["M:6/8", "L:1/16"]
    assert sung(compound["abc"]) == quarters(SCALE[:6], Fraction(1, 2))


def test_markers_become_section_comments_and_tags():
    out = score.convert(song_of(("Tune", 0, 0, line(SCALE), []), markers=((0, "Chorus"),)))
    assert "% chorus" in out["abc"].splitlines() and out["lyrics"] == "[Chorus]"


def test_karaoke_words_are_the_lyrics_and_their_sections_mark_the_score():
    syllables = [b"\\Sing", b" a", b" song", b" of", b"/six", b"pence", b" a", b" pocket"]
    tune = line(SCALE, start=1920)
    out = score.convert(song_of(("Tune", 0, 0, tune, kar_texts(syllables, start=1920, step=480))))
    assert out["lyrics"] == "[Verse]\nSing a song of\nsixpence a pocket"
    lines = out["abc"].splitlines()
    assert lines.index("% intro") < lines.index("% verse")
    assert out["facts"]["karaoke"] is True


def test_full_guesses_chords_from_the_other_parts_and_melody_writes_none():
    parts = (("Vocal", 0, 0, line([64, 67, 72, 67, 65, 69, 72, 69]), []),
             ("Piano", 1, 0, chord([48, 52, 55], 0, 1920) + chord([41, 45, 48], 1920, 3840), []))
    full = score.convert(song_of(*parts), "full")
    assert '"C"' in full["abc"] and '"F"' in full["abc"] and score.CHORDS_GUESSED in full["notices"]
    melody = score.convert(song_of(*parts), "melody")
    assert edits.chordless(melody["abc"]) and sung(melody["abc"]) == sung(full["abc"])


def test_a_second_line_goes_to_the_instrument():
    out = score.convert(song_of(("Vocal", 0, 0, line([67, 69, 71, 72]), []), ("Ins", 1, 0, line([76, 77, 79, 81]), [])))
    assert sung(out["abc"], "Ins") == quarters([76, 77, 79, 81])
    assert out["facts"]["instrument"] == 2


def test_a_changing_tempo_is_averaged_and_the_node_says_so():
    out = score.convert(song_of(("Tune", 0, 0, line(SCALE), []), tempos=((0, 500000), (1920, 1000000))))
    assert "Q:1/4=80" in out["abc"].splitlines() and out["facts"]["seconds"] == 6.0
    assert any("between 60 and 120 BPM" in notice for notice in out["notices"])


@pytest.mark.parametrize("parts, mode, reason", [
    ((("Kit", 9, 0, line([36, 38]), []),), "melody", "no part with pitched notes"),
    ((), "melody", "has no notes"),
    ((("Tune", 0, 0, line(SCALE), []),), "loud", "'mode' must be one of melody, full"),
])
def test_what_cannot_be_written_is_refused_with_the_reason(parts, mode, reason):
    with pytest.raises(ValueError, match=reason):
        score.convert(song_of(*parts), mode)


def test_a_stray_thirty_second_among_sixteenths_is_kept():
    out = score.convert(song_of(("Tune", 0, 0, line(SCALE) + [(3840, 3900, 74), (3900, 4320, 72)], [])))
    assert out["facts"]["grid"] == 32
    assert sung(out["abc"])[-2:] == [(Fraction(8), 74, Fraction(1, 8)), (Fraction(65, 8), 72, Fraction(7, 8))]
