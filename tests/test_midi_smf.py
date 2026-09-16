"""Standard MIDI Files: read the way files in the wild are written, and refused where a guess would be wrong.

The files here are built byte by byte rather than with ``smf.write``, so the
reader is not only checked against its own writer.
"""

from __future__ import annotations

import struct

import pytest

from yue2_comfy.midi import smf

END = bytes([0x00, 0xFF, 0x2F, 0x00])


def chunk(kind: bytes, body: bytes) -> bytes:
    return kind + struct.pack(">I", len(body)) + body


def midi_file(*tracks, format_=1, division=480) -> bytes:
    return chunk(b"MThd", struct.pack(">HHH", format_, len(tracks), division)) + b"".join(
        chunk(b"MTrk", body) for body in tracks)


def test_notes_pair_through_running_status_and_velocity_zero():
    body = bytes([0x00, 0x90, 60, 100, 0x60, 62, 90, 0x60, 60, 0, 0x60, 0x80, 62, 0]) + END
    song = smf.read(midi_file(body))
    assert song.tracks[0].notes == [smf.Note(0, 192, 60, 100, 0), smf.Note(96, 288, 62, 90, 0)]
    assert song.division == 480 and song.format == 1


def test_timing_names_and_texts_are_read_from_any_track():
    conductor = bytes([0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20,
                       0x00, 0xFF, 0x58, 0x04, 6, 3, 24, 8,
                       0x00, 0xFF, 0x59, 0x02, 0xFF, 0x01]) + END
    lead = bytes([0x00, 0xFF, 0x03, 0x04]) + b"Lead" + bytes([0x00, 0xC1, 81,
                                                             0x00, 0xFF, 0x06, 0x05]) + b"Verse" + bytes(
        [0x00, 0xFF, 0x05, 0x02]) + b"la" + bytes([0x00, 0x91, 67, 80, 0x83, 0x60, 0x81, 67, 0]) + END
    song = smf.read(midi_file(conductor, lead))
    assert song.tempos == [(0, 500000)]
    assert song.meters == [(0, 6, 8)]
    assert song.keys == [(0, -1, True)]
    assert song.markers == [(0, b"Verse")]
    track = song.tracks[1]
    assert track.name == b"Lead" and track.programs == {1: 81}
    assert [(tick, kind) for tick, kind, _payload in track.texts] == [(0, smf.NAME), (0, smf.MARKER), (0, smf.LYRIC)]
    assert track.notes == [smf.Note(0, 480, 67, 80, 1)]
    assert song.end == 480


def test_a_format_zero_file_keeps_its_channels_in_one_track():
    body = bytes([0x00, 0x90, 60, 100, 0x00, 0x99, 36, 100, 0x60, 0x80, 60, 0, 0x00, 0x89, 36, 0]) + END
    song = smf.read(midi_file(body, format_=0))
    assert sorted(note.channel for note in song.tracks[0].notes) == [0, smf.DRUM_CHANNEL]


def test_a_note_struck_again_before_release_pairs_its_offs_in_order():
    body = bytes([0x00, 0x90, 60, 100, 0x10, 60, 90, 0x10, 0x80, 60, 0, 0x10, 60, 0]) + END
    notes = smf.read(midi_file(body)).tracks[0].notes
    assert notes == [smf.Note(0, 32, 60, 100, 0), smf.Note(16, 48, 60, 90, 0)]


def test_a_note_left_sounding_ends_with_its_track():
    body = bytes([0x00, 0x90, 64, 100, 0x81, 0x40, 0xFF, 0x2F, 0x00])
    assert smf.read(midi_file(body)).tracks[0].notes == [smf.Note(0, 192, 64, 100, 0)]


def test_sysex_and_unknown_chunks_are_passed_over():
    body = bytes([0x00, 0xF0, 0x03, 0x7E, 0x7F, 0xF7, 0x00, 0x90, 60, 100, 0x40, 0x80, 60, 0]) + END
    data = midi_file(body)
    data = data[:14] + chunk(b"XFIH", b"abc") + data[14:]
    song = smf.read(data)
    assert song.tracks[0].notes == [smf.Note(0, 64, 60, 100, 0)]


def test_a_riff_wrapper_is_opened():
    inner = midi_file(bytes([0x00, 0x90, 60, 100, 0x40, 0x80, 60, 0]) + END)
    riff = b"RIFF" + struct.pack("<I", 4 + 8 + len(inner)) + b"RMID" + b"data" + struct.pack("<I", len(inner)) + inner
    assert smf.read(riff).tracks[0].notes == [smf.Note(0, 64, 60, 100, 0)]


@pytest.mark.parametrize("data, reason", [
    (b"RIFF\x00\x00\x00\x00WAVEfmt ", "not a MIDI file"),
    (midi_file(bytes([0x00, 0x90, 60])), "ends in the middle of track 1"),
    (midi_file(bytes([0x00, 60, 100]) + END), "data byte where an event should begin"),
    (chunk(b"MThd", struct.pack(">HHH", 1, 1, 0xE728)) + chunk(b"MTrk", END), "SMPTE"),
    (chunk(b"MThd", struct.pack(">HHH", 1, 0, 480)), "holds no track"),
    (midi_file(bytes([0x00, 0xFF, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F])), "runs past four bytes"),
])
def test_files_that_cannot_be_read_are_refused_with_the_reason(data, reason):
    with pytest.raises(ValueError, match=reason):
        smf.read(data)


@pytest.mark.parametrize("value, written", [(0, b"\x00"), (127, b"\x7f"), (128, b"\x81\x00"),
                                            (8192, b"\xc0\x00"), (0x0FFFFFFF, b"\xff\xff\xff\x7f")])
def test_variable_length_numbers_are_written_as_the_format_spells_them(value, written):
    assert smf.number(value) == written


def test_what_is_written_reads_back_and_a_repeated_pitch_is_not_cut_short():
    conductor = [(0, smf.tempo(600000)), (0, smf.meter(3, 4)), (0, smf.key(2, False)),
                 (960, smf.text(smf.MARKER, "Chorus"))]
    melody = [(0, smf.text(smf.NAME, "Vocal")), (0, smf.program(0, 53)),
              (0, smf.note_on(0, 62, 90)), (480, smf.note_off(0, 62)),
              (480, smf.note_on(0, 62, 90)), (960, smf.note_off(0, 62))]
    melody.reverse()
    song = smf.read(smf.write([conductor, melody]))
    assert song.tempos == [(0, 600000)] and song.meters == [(0, 3, 4)] and song.keys == [(0, 2, False)]
    assert song.markers == [(960, b"Chorus")]
    assert song.tracks[1].name == b"Vocal" and song.tracks[1].programs == {0: 53}
    assert song.tracks[1].notes == [smf.Note(0, 480, 62, 90, 0), smf.Note(480, 960, 62, 90, 0)]


def test_a_track_can_end_after_its_last_event():
    song = smf.read(smf.write([[(0, smf.note_on(0, 60, 90)), (480, smf.note_off(0, 60))]], end=1920))
    assert song.end == 1920 and song.tracks[0].notes == [smf.Note(0, 480, 60, 90, 0)]
