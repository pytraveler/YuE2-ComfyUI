"""Keys and chords of a transcription named from the key they sound in.

The expected names in the first tests are m-a-p's own: the key names their
release uses, and the chords their tests of 2026-09-24 spell, so this pack and
their SheetSage2 name a transcription alike.
"""

from __future__ import annotations

import copy
import re

from yue2_comfy import transpose
from yue2_comfy.sheetsage import abc_rebuild, events, spelling, vocab

SHARPS = vocab.SHARPS
MAJOR_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MINOR_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B")
INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
             "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "7": (0, 4, 7, 10),
             "hdim7": (0, 3, 6, 10), "dim7": (0, 3, 6, 9), "minmaj7": (0, 3, 7, 11),
             "sus2": (0, 2, 7), "sus4": (0, 5, 7), "sus4(b7)": (0, 5, 7, 10),
             "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9)}


def pitch_class(name: str) -> int:
    return (abc_rebuild.NATURAL[name[0]] + name.count("#") - name.count("b")) % 12


def test_every_key_the_model_names_gets_the_name_m_a_ps_release_gives_it():
    for index, root in enumerate(SHARPS):
        assert spelling.key_name(root + ":major") == MAJOR_NAMES[index] + ":major"
        assert spelling.key_name(root + ":minor") == MINOR_NAMES[index] + ":minor"


def test_a_key_already_named_well_keeps_its_name_and_an_awkward_one_loses_it():
    for index in range(12):
        for label in (MAJOR_NAMES[index] + ":major", MINOR_NAMES[index] + ":minor"):
            assert spelling.key_name(label) == label
    assert spelling.key_name("Gb:major") == "F#:major"
    assert spelling.key_name("Eb:minor") == "D#:minor"
    assert spelling.key_name("Cb:major") == "B:major"
    assert spelling.key_name("E#:minor") == "F:minor"


def test_a_label_that_is_not_a_major_or_minor_key_is_left_alone():
    for label in ("D:dorian", "H:major", "C", "", "C#:Major"):
        assert spelling.key_name(label) == label
    assert spelling.chord_name("D#:maj", "D:dorian") == "D#:maj"


def test_the_signature_of_a_key_counts_sharps_up_and_flats_down():
    assert spelling.signature("A#:minor") == -5
    assert spelling.signature("C#:major") == -5
    assert spelling.signature("D#:minor") == 6
    assert spelling.signature("F#:major") == 6
    assert spelling.signature("A:minor") == 0
    for index, root in enumerate(SHARPS):
        name = abc_rebuild.key_text(spelling.key_name(root + ":major"))
        assert abc_rebuild.KEY_SIGNATURES[name] == spelling.signature(root + ":major")


def test_every_quality_the_model_has_is_spelled_with_its_own_tones():
    assert set(spelling.TONES) == set(vocab.QUALITIES) == set(INTERVALS)
    for quality, steps in spelling.TONES.items():
        assert steps[0] == 0
        assert tuple(7 * step % 12 for step in steps) == INTERVALS[quality]


def test_chords_in_c_minor_take_its_flats_as_m_a_p_spell_them():
    for raw, expected in (("D#:maj", "Eb:maj"), ("G#:maj", "Ab:maj"), ("A#:7/3", "Bb:7/3"),
                          ("D#:min7/b7", "Eb:min7/b7"), ("C:min", "C:min"), ("N", "N"), ("X", "X")):
        assert spelling.chord_name(raw, "C:minor") == expected


def test_a_chord_outside_the_key_takes_the_side_its_tones_lie_on():
    for raw, key, expected in (("C:maj", "E:major", "C:maj"), ("A#:maj", "G:major", "Bb:maj"),
                               ("D#:maj", "G:major", "Eb:maj"), ("G#:7", "C:major", "Ab:7"),
                               ("A#:min", "C:major", "Bb:min"), ("C#:hdim7", "G:major", "C#:hdim7"),
                               ("F:hdim7", "G#:minor", "E#:hdim7")):
        assert spelling.chord_name(raw, key) == expected


def test_b_flat_minor_is_written_with_its_own_flats():
    key = "A#:minor"
    assert [spelling.chord_name(raw, key) for raw in ("A#:min", "D#:min", "F#:maj", "G#:maj", "F:7", "C:min")] == [
        "Bb:min", "Eb:min", "Gb:maj", "Ab:maj", "F:7", "C:min"]


def test_every_chord_of_the_vocabulary_keeps_its_pitch_quality_and_inversion():
    for key in ("C:minor", "E:major", "A#:minor", "C#:major", "F#:major", "D#:minor"):
        for chord in vocab.FULL_CHORDS + vocab.MAJMIN_CHORDS:
            named = spelling.chord_name(chord, key)
            if chord == "N":
                assert named == "N"
                continue
            root, rest = chord.split(":", 1)
            new_root, new_rest = named.split(":", 1)
            assert new_rest == rest
            assert pitch_class(new_root) == pitch_class(root)
            assert len(new_root) <= 3
            abc_rebuild.chord_text(named)


def test_a_chord_goes_by_the_key_at_its_midpoint_and_a_tie_by_the_key_that_ends():
    rows = [[0, 2, "G#:maj"], [2, 6, "G#:maj"], [6, 8, "G#:maj"]]
    keys = [[0, 4, "C:minor"], [4, 8, "E:major"]]
    original = copy.deepcopy((rows, keys))
    assert [row[2] for row in spelling.respelled(rows, keys)] == ["Ab:maj", "Ab:maj", "G#:maj"]
    assert (rows, keys) == original
    assert spelling.respelled(rows, []) == rows
    assert spelling.respelled([], keys) == []


def test_before_the_first_key_and_after_the_last_the_nearest_one_holds():
    keys = [[2, 4, "C:minor"], [4, 6, "E:major"]]
    rows = [[0, 1, "D#:maj"], [9, 10, "D#:maj"]]
    assert [row[2] for row in spelling.respelled(rows, keys)] == ["Eb:maj", "D#:maj"]


def event(time, **values):
    return {"time": time, "values": values}


def song_in_b_flat_minor():
    """Four bars of four beats in A#:minor as the model names it, a chord a bar."""
    song = []
    for beat in range(17):
        values = {"rhythm": {"meter": (4, 4), "eighth_position": beat % 4 * 2}}
        if beat == 0:
            values.update(key="A#:minor", chord="A#:min", structure="verse")
        if beat % 4 == 0 and 0 < beat < 16:
            values["chord"] = ("D#:min", "F#:maj", "F:7")[beat // 4 - 1]
        if beat < 16:
            values["melody"] = [{"pitch": (70, 73, 77, 75)[beat % 4], "track": 0, "end_time": beat * 0.5 + 0.5}]
        song.append(event(beat * 0.5, **values))
    return song


def test_a_transcription_names_its_key_and_chords_from_its_key_and_leaves_its_events_alone():
    song = song_in_b_flat_minor()
    original = copy.deepcopy(song)
    rows = events.score_rows(song, 8.0)
    assert song == original
    assert [row[2] for row in rows["keys"]] == ["Bb:minor"]
    assert [row[2] for row in rows["chords"]] == ["Bb:min", "Eb:min", "Gb:maj", "F:7"]
    text = abc_rebuild.build(rows)
    assert "K:Bbm" in text and "A#" not in text
    for symbol in ('"Bbm"', '"Ebm"', '"Gb"', '"F7"'):
        assert symbol in text


def test_a_transcription_in_b_flat_minor_transposes_to_plain_names():
    moved = transpose.move(abc_rebuild.build(events.score_rows(song_in_b_flat_minor(), 8.0)), 2)
    assert (moved.before, moved.after) == ("Bbm", "Cm")
    assert set(re.findall(r'"([^"]*)"', moved.text)) - {"Vocal Melody", "Ins Melody", "Vocal", "Inst."} == {
        "Cm", "Fm", "Ab", "G7"}
