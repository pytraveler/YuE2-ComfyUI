"""A MIDI file's parts, words and harmony: which part is sung, what the words are, and which chords sound."""

from __future__ import annotations

import pytest
from midi_files import chord, kar_texts, line, song_of

from yue2_comfy.midi import chords, karaoke, smf, tracks

MELODY = "\u041c\u0435\u043b\u043e\u0434\u0438\u044f"
TRACK_ONE = "\u0414\u043e\u0440\u043e\u0436\u043a\u0430 1"
SONG = "\u043f\u0435\u0441\u043d\u044f"
MUELLER = "M\u00fcller"


@pytest.mark.parametrize("data, text", [
    (b"Lead", "Lead"),
    (MELODY.encode("cp1251"), MELODY),
    (TRACK_ONE.encode("cp1251").decode("latin-1").encode("utf-8"), TRACK_ONE),
    (MUELLER.encode("latin-1"), MUELLER),
    (MUELLER.encode("utf-8"), MUELLER),
    (b"  two\x00  words ", "two words"),
])
def test_names_are_read_in_the_encoding_they_were_written_in(data, text):
    assert tracks.decode(data) == text


def test_only_parts_with_notes_are_numbered_and_a_shared_track_splits_by_channel():
    song = song_of(("Piano", 0, 0, line([60, 62]), []), ("", 1, 0, [], []))
    events = [(0, smf.note_on(0, 72, 90)), (480, smf.note_off(0, 72)), (0, smf.note_on(9, 36, 90)),
              (480, smf.note_off(9, 36))]
    song.tracks.append(smf.read(smf.write([events])).tracks[0])
    found = tracks.parts(song)
    assert [(part.number, part.name, part.family) for part in found] == [
        (1, "Piano", "Piano"), (2, "channel 1", "Piano"), (3, "channel 10", "Drums")]


def test_the_highest_melodic_line_is_the_voice_and_a_bass_is_never_the_instrument():
    song = song_of(("Bass", 1, 33, line([36, 38, 40, 41]), []),
                   ("Tune", 0, 80, line([67, 69, 71, 72]), []),
                   ("Drums", 9, 0, line([36, 38, 36, 38]), []))
    chosen = tracks.choose(tracks.parts(song))
    assert chosen["voice"].name == "Tune" and chosen["why"]["voice"] == "highest"
    assert chosen["instrument"] is None


def test_a_pad_of_chords_above_the_tune_takes_neither_line():
    pad = chord([72, 76, 79], 0, 1920) + chord([74, 77, 81], 1920, 3840)
    song = song_of(("Pad", 0, 88, pad, []), ("Tune", 1, 0, line([65, 67, 69, 70, 72, 70, 69, 67]), []))
    chosen = tracks.choose(tracks.parts(song))
    assert chosen["voice"].name == "Tune" and chosen["instrument"] is None


@pytest.mark.parametrize("name", ["Vocal", "Lead Voice", MELODY])
def test_a_part_named_as_sung_is_the_voice(name):
    song = song_of(("Strings", 0, 48, line([79, 81, 83, 84]), []), (name, 1, 0, line([60, 62, 64, 65]), []))
    chosen = tracks.choose(tracks.parts(song))
    assert chosen["voice"].number == 2 and chosen["why"]["voice"] == "name"


def test_the_part_the_karaoke_syllables_fall_on_is_the_voice():
    song = song_of(("Flute", 0, 73, line([84, 86, 88, 89], step=240), []),
                   ("Singer?", 1, 0, line([60, 62, 64, 65, 67, 65, 64, 62]), []))
    chosen = tracks.choose(tracks.parts(song), syllables=[0, 480, 960, 1440])
    assert chosen["voice"].number == 2 and chosen["why"]["voice"] == "karaoke"


def test_numbers_are_obeyed_and_wrong_ones_are_refused_with_the_list():
    song = song_of(("Lead", 0, 80, line([67, 69]), []), ("Bass", 1, 33, line([40, 41]), []),
                   ("Kit", 9, 0, line([36, 38]), []))
    found = tracks.parts(song)
    chosen = tracks.choose(found, "2", "1")
    assert chosen["voice"].number == 2 and chosen["instrument"].number == 1
    assert chosen["why"] == {"voice": "chosen", "instrument": "chosen"}
    assert tracks.choose(found, "auto", "none")["instrument"] is None
    with pytest.raises(ValueError, match="track 7, but this file has 3 tracks with notes: 1 Lead"):
        tracks.choose(found, "7")
    with pytest.raises(ValueError, match="drums"):
        tracks.choose(found, "3")
    with pytest.raises(ValueError, match="both track 1"):
        tracks.choose(found, "1", "1")


def test_a_file_of_drums_alone_has_nothing_to_sing():
    with pytest.raises(ValueError, match="no part with pitched notes"):
        tracks.choose(tracks.parts(song_of(("Kit", 9, 0, line([36, 38]), []))))


def test_the_description_names_each_part_and_its_role():
    song = song_of(("Lead", 0, 80, chord([67, 79], 0, 480) + line([69], start=480), []))
    found = tracks.parts(song)
    rows = tracks.describe(found, tracks.choose(found))
    assert rows == [{"number": 1, "name": "Lead", "family": "Synth Lead", "notes": 3, "low": 67, "high": 79,
                     "drums": False, "polyphony": 2, "role": "voice", "why": "name"}]


def test_kar_words_become_sections_with_a_repeated_one_as_the_chorus():
    syllables = [b"\\Here", b" we", b" go", b"/Down", b" the", b" road",
                 b"\\Sing", b" it", b" loud", b"/All", b" night",
                 b"\\Walk", b" on", b"/by",
                 b"\\Sing", b" it", b" loud", b"/All", b" night"]
    song = song_of(("Words", 0, 0, line([60]), kar_texts(syllables)))
    words = karaoke.read(song)
    assert words.lyrics() == ("[Verse]\nHere we go\nDown the road\n\n[Chorus]\nSing it loud\nAll night\n\n"
                              "[Verse]\nWalk on\nby\n\n[Chorus]\nSing it loud\nAll night")
    assert [section["tick"] for section in words.sections] == [0, 1440, 2640, 3360]
    assert [section["label"] for section in words.sections] == ["verse", "chorus", "verse", "chorus"]
    assert len(words.syllables) == len(syllables)


def test_lyric_events_end_lines_at_returns_and_sections_at_blank_lines_and_long_pauses():
    pieces = [b"One ", b"line\r", b"two ", b"lines\r", b"\r", b"after ", b"blank\r", b"far ", b"later"]
    ticks = [0, 240, 480, 720, 720, 960, 1200, 1200 + 8 * 480, 1440 + 8 * 480]
    texts = [(tick, smf.LYRIC, piece) for tick, piece in zip(ticks, pieces)]
    words = karaoke.read(song_of(("Vocal", 0, 0, line([60]), texts)))
    assert [section["lines"] for section in words.sections] == [["One line", "two lines"], ["after blank"],
                                                                ["far later"]]


def test_syllables_in_windows_1251_are_read_as_cyrillic():
    syllables = [(" " + SONG[:3]).encode("cp1251"), SONG[3:].encode("cp1251")] * 4
    words = karaoke.read(song_of(("Words", 0, 0, line([60]), kar_texts(syllables))))
    assert words.sections[0]["lines"] == [" ".join([SONG] * 4)]


def test_markers_name_the_sections_they_sit_by():
    syllables = [b"\\A", b" b", b" c", b" d", b"\\E", b" f", b" g", b" h"]
    song = song_of(("Words", 0, 0, line([60]), kar_texts(syllables)), markers=((0, "Intro 1"), (960, "Bridge")))
    assert [section["tag"] for section in karaoke.read(song).sections] == ["Intro", "Bridge"]
    assert karaoke.marker_sections(song) == [(0, "intro"), (960, "bridge")]
    assert karaoke.section_label("Pre-Chorus 2") == "pre-chorus" and karaoke.section_label("Bar 17") is None


def test_a_title_alone_is_not_words():
    assert karaoke.read(song_of(("Words", 0, 0, line([60]), kar_texts([b"Hello"])))) is None


def weights_of(pitch_classes):
    weights = [0.0] * 12
    for pitch_class in pitch_classes:
        weights[pitch_class] += 1.0
    return weights


@pytest.mark.parametrize("pitch_classes, flats, expected", [
    ([0, 4, 7], False, "C:maj"),
    ([9, 0, 4], False, "A:min"),
    ([7, 11, 2, 5], False, "G:7"),
    ([10, 2, 5], True, "Bb:maj"),
    ([10, 2, 5], False, "A#:maj"),
    ([11, 2, 5], False, "B:dim"),
    ([0], False, "N"),
])
def test_the_chord_that_sounds_is_named(pitch_classes, flats, expected):
    assert chords.label(weights_of(pitch_classes), pitch_classes[0], flats) == expected


def test_chords_are_guessed_per_half_bar_and_equal_neighbours_merge():
    notes = [(0, 1920, 48, 1.0), (0, 1920, 52, 1.0), (0, 1920, 55, 1.0),
             (1920, 2880, 45, 1.0), (1920, 2880, 48, 1.0), (1920, 2880, 52, 1.0),
             (2880, 3840, 43, 1.0), (2880, 3840, 47, 1.0), (2880, 3840, 50, 1.0), (2880, 3840, 53, 1.0)]
    rows = chords.guess(notes, chords.spans([(0, 1920, 4), (1920, 3840, 4)]))
    assert rows == [[0, 1920, "C:maj"], [1920, 2880, "A:min"], [2880, 3840, "G:7"]]
    assert chords.spans([(0, 1440, 3)]) == [(0, 1440)]


def test_a_line_playing_alone_is_no_chord():
    bass = [(0, 240, 43, 1.0), (240, 480, 39, 1.0), (480, 720, 41, 1.0), (720, 960, 43, 1.0)]
    assert chords.guess(bass, [(0, 960)]) == [[0, 960, "N"]]


def test_an_open_fifth_takes_the_third_the_key_has():
    fifth = [0.0] * 12
    fifth[2] = fifth[9] = 1.0
    d_minor = {5, 7, 9, 10, 0, 2, 4}
    assert chords.label(fifth, 2) == "D:maj"
    assert chords.label(fifth, 2, scale=d_minor) == "D:min"
