"""Scores saved as MIDI: the file holds the score's timing, keys, sections, lines and chords, and loads back.

The round trip runs on every score in tests/data -- the model's own and
SheetSage2's -- so the dialect as it is really written is what is checked.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from yue2_comfy.midi import export, score, smf
from yue2_comfy.vendor.yue2_music import abc_tools

DATA = pathlib.Path(__file__).resolve().parent / "data"
HEADER = ['X:1', 'T:', 'M:4/4', 'L:1/16', 'Q:1/4={bpm}', 'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
          'V: Ins clef=treble name="Ins Melody" snm="Inst."', 'K:{key}']


def written_scores() -> list:
    found = []

    def walk(value, path):
        if isinstance(value, str) and value.startswith("X:1"):
            found.append((path, value))
        elif isinstance(value, dict):
            for key, inner in value.items():
                walk(inner, path + "/" + str(key))
        elif isinstance(value, list):
            for index, inner in enumerate(value):
                walk(inner, path + "/" + str(index))

    for name in ("model_scores.json", "sheetsage_pack.json"):
        walk(json.loads((DATA / name).read_text(encoding="utf-8")), name)
    return found


SCORES = written_scores()


def abc(*music, bpm=90, key="G"):
    return "\n".join([line.format(bpm=bpm, key=key) for line in HEADER] + list(music)) + "\n"


def test_there_are_sixteen_scores_to_round_trip():
    assert len(SCORES) == 16


@pytest.mark.parametrize("path, text", SCORES, ids=[path for path, _text in SCORES])
def test_a_saved_score_loads_back_to_the_same_lines_in_the_same_bars_at_the_same_tempo(path, text):
    back = score.convert(smf.read(export.midi_of(text)), "melody")
    result = abc_tools.compare(abc_tools.parse(text), abc_tools.parse(back["abc"]))
    assert result["match"], result["differences"]


def test_the_file_carries_tempo_meters_keys_sections_lines_and_chords():
    text = abc("% verse", "V: Vocal", '"G"G4A4B4c4|"Em"d16|', "V: Ins", "Z2|",
               "% chorus", "V: Vocal", "M:3/4", "K:D", '"D/F#"d4e4f4|', "V: Ins", "M:3/4", "K:D", "Z|")
    song = smf.read(export.midi_of(text))
    assert song.tempos == [(0, 666667)]
    assert song.meters == [(0, 4, 4), (3840, 3, 4)]
    assert song.keys == [(0, 1, False), (3840, 2, False)]
    assert song.markers == [(0, b"verse"), (3840, b"chorus")]
    assert [track.name for track in song.tracks] == [b"YuE2 score", b"Vocal", b"Chords"]
    assert [(note.start, note.end, note.pitch) for note in song.tracks[1].notes] == [
        (0, 480, 67), (480, 960, 69), (960, 1440, 71), (1440, 1920, 72), (1920, 3840, 74),
        (3840, 4320, 74), (4320, 4800, 76), (4800, 5280, 78)]
    held = sorted((note.start, note.end, note.pitch) for note in song.tracks[2].notes)
    assert held == sorted([(0, 1920, p) for p in (55, 59, 62)] + [(1920, 3840, p) for p in (52, 55, 59)]
                          + [(3840, 5280, p) for p in (42, 50, 54, 57)])
    assert song.end == 5280


@pytest.mark.parametrize("symbol, pitches", [
    ("C", [48, 52, 55]), ("Am7", [57, 60, 64, 67]), ("F#m7b5", [54, 57, 60, 64]),
    ("Bbm(maj7)", [58, 61, 65, 69]), ("C/E", [40, 48, 52, 55]), ("Db7sus4", [49, 54, 56, 59]),
])
def test_a_chord_symbol_is_held_as_its_notes(symbol, pitches):
    assert export.chord_pitches(symbol) == pitches


def test_a_symbol_the_format_does_not_know_is_refused():
    with pytest.raises(ValueError, match="not a chord symbol"):
        export.chord_pitches("Cmaj13")


def test_bars_of_rest_at_the_end_are_kept():
    text = abc("V: Vocal", "G4A4B4c4|Z3|", "V: Ins", "Z4|")
    back = score.convert(smf.read(export.midi_of(text)))
    assert abc_tools.compare(abc_tools.parse(text), abc_tools.parse(back["abc"]))["match"]


def test_triads_come_back_as_the_same_chords_when_loaded_with_full():
    text = abc("V: Vocal", '"G"B8d8|"Em"B8G8|"C"c8e8|"D"d8A8|', "V: Ins", "Z4|", bpm=100)
    back = score.convert(smf.read(export.midi_of(text)), "full")
    assert abc_tools.parse(back["abc"]).voices["Vocal"].chords == abc_tools.parse(text).voices["Vocal"].chords


def test_a_score_that_cannot_be_read_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="Incomplete native two-voice ABC"):
        export.midi_of("X:1\nT:\n")
