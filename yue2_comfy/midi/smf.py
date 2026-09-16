"""Standard MIDI Files: the notes and timing a file holds, and a file written from them.

A Standard MIDI File (MIDI Manufacturers Association, 1988) is a header chunk
followed by track chunks. A track is a list of events, each after a delta of
ticks, and a tick is the fraction of a quarter note the header names. Channel
events turn notes on and off and choose instruments; meta events carry the
tempo, the meter, the key, names, markers and lyrics. A file of format 0 keeps
every channel in one track; format 1 gives each part a track of its own.

Reading is forgiving where files in the wild are sloppy and strict where a
guess would put wrong notes into a score. Running status survives a meta
event, a note-on at velocity 0 ends a note, a note left sounding ends with its
track, a note struck again before it was released pairs its offs in order,
chunks of other kinds are passed over, and a RIFF 'RMID' wrapper is opened. A
file cut short, a data byte with no event to belong to, and a file timed in
SMPTE frames rather than beats are refused with the reason.

Nothing here opens a file or knows about scores.
"""

from __future__ import annotations

import struct
from typing import NamedTuple

DIVISION = 480
"""Ticks per quarter note in a file this module writes: a 1/32 note is 60 ticks and a triplet eighth 160."""

DRUM_CHANNEL = 9
"""The channel General MIDI keeps for drums, counted from 0 (channel 10 on an instrument's display)."""

DEFAULT_TEMPO = 500000
"""Microseconds per quarter note until a file sets a tempo: 120 BPM."""

TEXT = 0x01
COPYRIGHT = 0x02
NAME = 0x03
INSTRUMENT = 0x04
LYRIC = 0x05
MARKER = 0x06
CUE = 0x07
END = 0x2F
TEMPO = 0x51
METER = 0x58
KEY = 0x59
TEXTS = frozenset({TEXT, COPYRIGHT, NAME, INSTRUMENT, LYRIC, MARKER, CUE})
DATA_BYTES = {0x8: 2, 0x9: 2, 0xA: 2, 0xB: 2, 0xC: 1, 0xD: 1, 0xE: 2}
"""How many data bytes follow each kind of channel event."""


class Note(NamedTuple):
    """One note as it sounds: the ticks it starts and ends at, its pitch, velocity and channel."""

    start: int
    end: int
    pitch: int
    velocity: int
    channel: int


class Track:
    """A track chunk as read: its name, its notes, each channel's first program, and its text events.

    ``texts`` holds ``(tick, kind, bytes)`` for every text-like meta event, in
    file order; names and lyrics stay bytes, because only the reader of them
    can tell which encoding a file was written in.
    """

    def __init__(self, index: int):
        self.index = index
        self.name = b""
        self.notes = []
        self.programs = {}
        self.texts = []
        self.end = 0


class Song:
    """A whole file: its format and division, its tracks, and the timing every track shares.

    ``tempos`` holds ``(tick, microseconds per quarter)``, ``meters``
    ``(tick, numerator, denominator)``, ``keys`` ``(tick, sharps, minor)`` with
    flats as negative sharps, and ``markers`` ``(tick, bytes)`` -- gathered from
    every track, since files disagree about which track carries them.
    """

    def __init__(self, format_: int, division: int):
        self.format = format_
        self.division = division
        self.tracks = []
        self.tempos = []
        self.meters = []
        self.keys = []
        self.markers = []

    @property
    def end(self) -> int:
        """The last tick any track reaches."""
        return max((track.end for track in self.tracks), default=0)


class _Cursor:
    """Bytes read in order, with the part of the file they belong to named in every complaint."""

    def __init__(self, data: bytes, context: str):
        self.data = data
        self.at = 0
        self.context = context

    def more(self) -> bool:
        return self.at < len(self.data)

    def byte(self) -> int:
        if self.at >= len(self.data):
            raise ValueError("the file ends in the middle of {}".format(self.context))
        value = self.data[self.at]
        self.at += 1
        return value

    def take(self, count: int) -> bytes:
        if self.at + count > len(self.data):
            raise ValueError("the file ends in the middle of {}".format(self.context))
        value = self.data[self.at:self.at + count]
        self.at += count
        return value

    def number(self) -> int:
        value = 0
        for _ in range(4):
            byte = self.byte()
            value = (value << 7) | (byte & 0x7F)
            if byte < 0x80:
                return value
        raise ValueError("a length in {} runs past four bytes, which no MIDI file writes".format(self.context))


def _unwrap(data: bytes) -> bytes:
    """The Standard MIDI File inside a RIFF 'RMID' file, or the data as it is."""
    if data[:4] != b"RIFF" or data[8:12] != b"RMID":
        return data
    at = 12
    while at + 8 <= len(data):
        kind = data[at:at + 4]
        size = struct.unpack("<I", data[at + 4:at + 8])[0]
        if kind == b"data":
            return data[at + 8:at + 8 + size]
        at += 8 + size + (size & 1)
    raise ValueError("this RIFF file says it holds MIDI, but it has no 'data' chunk")


def _meta(song: Song, track: Track, tick: int, kind: int, payload: bytes) -> None:
    if kind == TEMPO and len(payload) == 3:
        value = int.from_bytes(payload, "big")
        if value > 0:
            song.tempos.append((tick, value))
    elif kind == METER and len(payload) >= 2:
        if payload[0] > 0 and payload[1] <= 6:
            song.meters.append((tick, payload[0], 1 << payload[1]))
    elif kind == KEY and len(payload) >= 2:
        sharps = payload[0] - 256 if payload[0] > 127 else payload[0]
        if -7 <= sharps <= 7 and payload[1] in (0, 1):
            song.keys.append((tick, sharps, bool(payload[1])))
    elif kind == MARKER:
        song.markers.append((tick, payload))
    elif kind == NAME and not track.name:
        track.name = payload
    if kind in TEXTS:
        track.texts.append((tick, kind, payload))


def _track(body: bytes, index: int, song: Song) -> Track:
    track = Track(index)
    cursor = _Cursor(body, "track {}".format(index + 1))
    tick = 0
    status = None
    sounding = {}
    while cursor.more():
        tick += cursor.number()
        first = cursor.byte()
        if first == 0xFF:
            kind = cursor.byte()
            payload = cursor.take(cursor.number())
            if kind == END:
                break
            _meta(song, track, tick, kind, payload)
            continue
        if first in (0xF0, 0xF7):
            cursor.take(cursor.number())
            continue
        if first & 0x80:
            if first >= 0xF0:
                raise ValueError("track {} holds a {:02X} event, which does not belong in a MIDI file"
                                 .format(index + 1, first))
            status = first
            values = [cursor.byte() for _ in range(DATA_BYTES[status >> 4])]
        elif status is None:
            raise ValueError("track {} has a data byte where an event should begin".format(index + 1))
        else:
            values = [first] + [cursor.byte() for _ in range(DATA_BYTES[status >> 4] - 1)]
        if any(value & 0x80 for value in values):
            raise ValueError("track {} has an event cut short of its data bytes".format(index + 1))
        kind, channel = status >> 4, status & 0x0F
        if kind == 0x9 and values[1] > 0:
            sounding.setdefault((channel, values[0]), []).append((tick, values[1]))
        elif kind in (0x8, 0x9):
            opened = sounding.get((channel, values[0]))
            if opened:
                start, velocity = opened.pop(0)
                track.notes.append(Note(start, tick, values[0], velocity, channel))
        elif kind == 0xC:
            track.programs.setdefault(channel, values[0])
    track.end = tick
    for (channel, pitch), opened in sounding.items():
        for start, velocity in opened:
            track.notes.append(Note(start, tick, pitch, velocity, channel))
    track.notes.sort()
    return track


def read(data: bytes) -> Song:
    """The song a Standard MIDI File holds; a ValueError says why a file cannot be read."""
    data = _unwrap(bytes(data))
    if data[:4] != b"MThd":
        raise ValueError("this is not a MIDI file: it does not begin with 'MThd'")
    if len(data) < 14:
        raise ValueError("the MIDI header is cut short")
    size = struct.unpack(">I", data[4:8])[0]
    if size < 6:
        raise ValueError("the MIDI header is {} bytes long, shorter than the format allows".format(size))
    format_, count, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("the file is timed in SMPTE frames rather than beats, so it has no bars to write a score in")
    if division == 0:
        raise ValueError("the file gives a quarter note no ticks")
    song = Song(format_, division)
    at = 8 + size
    while at + 8 <= len(data) and len(song.tracks) < count:
        kind = data[at:at + 4]
        length = struct.unpack(">I", data[at + 4:at + 8])[0]
        body = data[at + 8:at + 8 + length]
        at += 8 + length
        if kind != b"MTrk":
            continue
        song.tracks.append(_track(body, len(song.tracks), song))
    if not song.tracks:
        raise ValueError("the MIDI file holds no track")
    for rows in (song.tempos, song.meters, song.keys, song.markers):
        rows.sort(key=lambda row: row[0])
    return song


def number(value: int) -> bytes:
    """A variable-length quantity, the way delta times and meta lengths are written."""
    value = int(value)
    if not 0 <= value <= 0x0FFFFFFF:
        raise ValueError("{} does not fit a MIDI variable-length number".format(value))
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def meta(kind: int, payload: bytes) -> bytes:
    """A meta event of this kind."""
    return bytes([0xFF, kind]) + number(len(payload)) + bytes(payload)


def tempo(microseconds: int) -> bytes:
    """A tempo event: microseconds per quarter note."""
    return meta(TEMPO, int(microseconds).to_bytes(3, "big"))


def meter(numerator: int, denominator: int) -> bytes:
    """A time signature, with the metronome at every quarter and eight 32nds to a quarter."""
    return meta(METER, bytes([int(numerator), int(denominator).bit_length() - 1, 24, 8]))


def key(sharps: int, minor: bool) -> bytes:
    """A key signature: sharps positive, flats negative."""
    return meta(KEY, bytes([int(sharps) & 0xFF, 1 if minor else 0]))


def text(kind: int, value: str) -> bytes:
    """A text-like meta event -- a name, a marker, a lyric -- in UTF-8."""
    return meta(kind, str(value).encode("utf-8"))


def program(channel: int, value: int) -> bytes:
    """A program change: the instrument a channel plays from here on."""
    return bytes([0xC0 | int(channel), int(value)])


def note_on(channel: int, pitch: int, velocity: int) -> bytes:
    return bytes([0x90 | int(channel), int(pitch), int(velocity)])


def note_off(channel: int, pitch: int) -> bytes:
    return bytes([0x80 | int(channel), int(pitch), 0])


def _rank(event: bytes) -> int:
    """Meta events first at a tick, then note-offs, then everything else in the order it was given."""
    if event[0] == 0xFF:
        return 0
    if event[0] >> 4 == 0x8:
        return 1
    return 2


def write(tracks: list, division: int = DIVISION, end: int = 0) -> bytes:
    """A format-1 file from tracks, each a list of ``(tick, event bytes)``.

    Events are sorted by tick. At one tick, meta events come first and
    note-offs before the rest, so a note that ends where the same pitch starts
    again is not cut short by a player pairing ons and offs by pitch; events of
    the same rank keep the order they were given in. Every track ends at ``end``
    when that is later than its last event, so bars of rest closing a score are
    still in the file.
    """
    chunks = [b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), int(division))]
    for events in tracks:
        ordered = sorted(enumerate(events), key=lambda item: (int(item[1][0]), _rank(item[1][1]), item[0]))
        body = bytearray()
        last = 0
        for _position, (tick, event) in ordered:
            body += number(int(tick) - last) + bytes(event)
            last = int(tick)
        body += number(max(0, int(end) - last)) + meta(END, b"")
        chunks.append(b"MTrk" + struct.pack(">I", len(body)) + bytes(body))
    return b"".join(chunks)
