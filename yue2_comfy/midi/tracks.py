"""A MIDI file's parts: what each one is, and which of them the voice and the instrument take.

A part is a track, or, where one track holds several channels -- every
format-0 file does -- one channel of it. Only parts with notes are counted,
numbered from 1 in file order, so the numbers on the node are the ones in the
list it shows. Channel 10 is drums and is never sung.

Left to choose, the node gives the voice, in this order: the part the karaoke
syllables fall on; a part whose name says it is sung or is the melody; and
otherwise the highest melodic line -- not a bass (General MIDI programs 32-39
counted from 0, or a middle pitch below C3), carrying at least a quarter as
many notes as the busiest part, and marked down by up to an octave for the
notes it strikes in chords, so a pad above the tune does not take it. The
instrument is a part named for it, or the busiest remaining part above G3
that strikes at most half its notes in chords, or nothing:
in YuE2's own scores the second voice is a melodic line, not a bass. A part
named 'Chords' -- the harmony this pack writes when it saves a score as MIDI --
is neither.

Names are bytes, and the format names no encoding. UTF-8 is tried first;
failing that, text whose letters are mostly bytes from 0xC0 up is read as
Windows-1251, the usual encoding of Russian files, and anything else as
Latin-1. A name that is valid UTF-8 only because a converter encoded
Windows-1251 letters as if they were Latin-1 is read back to the Cyrillic.
"""

from __future__ import annotations

import bisect
import re
import statistics

from . import smf

FAMILIES = ("Piano", "Chromatic Percussion", "Organ", "Guitar", "Bass", "Strings", "Ensemble", "Brass", "Reed",
            "Pipe", "Synth Lead", "Synth Pad", "Synth Effects", "Ethnic", "Percussive", "Sound Effects")
"""General MIDI's sixteen instrument families, eight programs each, in program order."""

BASS_PROGRAMS = range(32, 40)
LOWEST_MELODY = 48
"""A part whose middle pitch is below C3 is a bass line whatever its program."""

INSTRUMENT_LOWEST = 55
BUSY_SHARE = 0.25
CHORD_LINE = 0.5
CHORD_PENALTY = 12.0
"""Semitones a part's middle pitch is marked down for striking all of its notes in chords: an octave."""

KARAOKE_SHARE = 0.5
"""The share of karaoke syllables that must fall on a part's note onsets for the words to name it the voice."""

TRACK_CHOICES = 64
"""How many parts the track widgets offer by number."""

VOICE_NAME = re.compile("vocal|voice|vox|melod|sing|lead|\u0432\u043e\u043a\u0430\u043b|\u0433\u043e\u043b\u043e\u0441"
                        "|\u043c\u0435\u043b\u043e\u0434", re.IGNORECASE)
INSTRUMENT_NAME = re.compile(r"^ins\b|instrument|solo", re.IGNORECASE)
CHORDS_NAME = re.compile(r"^\s*chords?\s*$", re.IGNORECASE)
DOUBLE_ENCODED = "utf-8-cp1251"


def encoding_of(data: bytes) -> str:
    """How to read these bytes: 'utf-8', 'cp1251', 'latin-1', or ``DOUBLE_ENCODED`` for the converter's mistake."""
    raw = bytes(data).replace(b"\x00", b"")
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError:
        high = sum(1 for byte in raw if byte >= 0xC0)
        letters = high + sum(1 for byte in raw if 65 <= byte <= 90 or 97 <= byte <= 122)
        return "cp1251" if high >= 3 and high * 2 >= letters else "latin-1"
    wide = [char for char in value if ord(char) > 127]
    letters = sum(1 for char in value if char.isalpha())
    if len(wide) >= 3 and all(ord(char) <= 0xFF for char in wide) and len(wide) * 2 >= letters:
        try:
            value.encode("latin-1").decode("cp1251")
        except UnicodeDecodeError:
            return "utf-8"
        return DOUBLE_ENCODED
    return "utf-8"


def text_of(data: bytes, encoding: str) -> str:
    """Bytes read in an encoding ``encoding_of`` named, with nothing folded: a karaoke syllable keeps its spaces."""
    raw = bytes(data).replace(b"\x00", b"")
    if encoding == DOUBLE_ENCODED:
        try:
            return raw.decode("utf-8").encode("latin-1").decode("cp1251")
        except UnicodeError:
            return raw.decode("utf-8", errors="replace")
    return raw.decode(encoding, errors="replace")


def decode(data: bytes) -> str:
    """Text from a MIDI file's bytes, its encoding guessed as the module docstring says, whitespace folded."""
    return " ".join(text_of(data, encoding_of(data)).split())


class Part:
    """One part: its number on the node, where it comes from, its name and program, and the notes it plays."""

    def __init__(self, number: int, track: int, channel: int, notes: list, program: int, name: str):
        self.number = number
        self.track = track
        self.channel = channel
        self.notes = notes
        self.program = program
        self.name = name
        self.median = statistics.median([note.pitch for note in notes])

    @property
    def drums(self) -> bool:
        return self.channel == smf.DRUM_CHANNEL

    @property
    def family(self) -> str:
        return "Drums" if self.drums else FAMILIES[self.program // 8]

    @property
    def bass(self) -> bool:
        return not self.drums and (self.program in BASS_PROGRAMS or self.median < LOWEST_MELODY)

    @property
    def polyphony(self) -> int:
        """The most notes the part sounds at once."""
        edges = sorted([(note.start, 1) for note in self.notes] + [(note.end, -1) for note in self.notes])
        most = sounding = 0
        for _tick, step in edges:
            sounding += step
            most = max(most, sounding)
        return most

    @property
    def chord_share(self) -> float:
        """The share of notes struck together with another note of the part."""
        onsets = {}
        for note in self.notes:
            onsets[note.start] = onsets.get(note.start, 0) + 1
        return sum(count for count in onsets.values() if count > 1) / len(self.notes)

    @property
    def label(self) -> str:
        return "{} {}".format(self.number, self.name or self.family)


def parts(song: smf.Song) -> list:
    """Every part of the song that plays a note, numbered from 1."""
    programs = {}
    for track in song.tracks:
        for channel, program in track.programs.items():
            programs.setdefault(channel, program)
    found = []
    for track in song.tracks:
        channels = sorted({note.channel for note in track.notes})
        name = decode(track.name)
        for channel in channels:
            notes = [note for note in track.notes if note.channel == channel]
            label = (name + " " if name else "") + "channel {}".format(channel + 1) if len(channels) > 1 else name
            found.append(Part(len(found) + 1, track.index, channel, notes,
                              track.programs.get(channel, programs.get(channel, 0)), label))
    return found


def listing(found: list) -> str:
    """The parts as a short list for a message: '1 Lead (Synth Lead), 2 Bass (Bass)'."""
    return ", ".join("{} ({})".format(part.label, part.family) for part in found)


def _numbered(found: list, value, widget: str):
    if value in (None, "", "auto", "none"):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError("'{}' is '{}', which is not 'auto' or a track number".format(widget, value)) from None
    for part in found:
        if part.number == number:
            if part.drums:
                raise ValueError("'{}' is track {}, which is drums: drums have no tune to sing. The file's tracks: {}"
                                 .format(widget, number, listing(found)))
            return part
    raise ValueError("'{}' is track {}, but this file has {} with notes: {}".format(
        widget, number, "one track" if len(found) == 1 else "{} tracks".format(len(found)), listing(found)))


def _onset_share(part: Part, ticks: list, tolerance: int) -> float:
    onsets = sorted(note.start for note in part.notes)
    hits = 0
    for tick in ticks:
        at = bisect.bisect_left(onsets, tick - tolerance)
        if at < len(onsets) and onsets[at] <= tick + tolerance:
            hits += 1
    return hits / len(ticks) if ticks else 0.0


def _auto_voice(pitched: list, syllables: list, division: int) -> tuple:
    usable = [part for part in pitched if not CHORDS_NAME.match(part.name)] or pitched
    if syllables:
        tolerance = max(1, division // 8)
        best = max(usable, key=lambda part: _onset_share(part, syllables, tolerance))
        if _onset_share(best, syllables, tolerance) >= KARAOKE_SHARE:
            return best, "karaoke"
    named = [part for part in usable if VOICE_NAME.search(part.name) and not part.bass]
    if named:
        return max(named, key=lambda part: len(part.notes)), "name"
    busiest = max(len(part.notes) for part in usable)
    melodic = [part for part in usable if not part.bass and len(part.notes) >= BUSY_SHARE * busiest]
    pool = melodic or usable
    return max(pool, key=lambda part: part.median - CHORD_PENALTY * part.chord_share), "highest"


def _auto_instrument(rest: list) -> tuple:
    usable = [part for part in rest if not CHORDS_NAME.match(part.name)]
    named = [part for part in usable if INSTRUMENT_NAME.search(part.name)]
    if named:
        return max(named, key=lambda part: len(part.notes)), "name"
    if not usable:
        return None, ""
    busiest = max(len(part.notes) for part in usable)
    lines = [part for part in usable if not part.bass and part.median >= INSTRUMENT_LOWEST
             and part.chord_share <= CHORD_LINE and len(part.notes) >= BUSY_SHARE * busiest]
    if not lines:
        return None, ""
    return max(lines, key=lambda part: len(part.notes) * (1.0 - part.chord_share)), "busiest"


def choose(found: list, vocal="auto", instrument="auto", syllables=(), division: int = smf.DIVISION) -> dict:
    """The parts the voice and the instrument take: ``{"voice", "instrument", "why"}``.

    ``vocal`` is 'auto' or a part number, ``instrument`` also 'none'. ``why``
    says how each was chosen -- 'chosen', 'karaoke', 'name', 'highest' or
    'busiest' -- for the list the node shows. Raises ValueError, naming the
    file's parts, for a number the file does not have, for drums, for one part
    asked to be both, and for a file with nothing but drums.
    """
    pitched = [part for part in found if not part.drums]
    if not pitched:
        raise ValueError("the file has no part with pitched notes{}".format(
            ", only drums: " + listing(found) if found else ""))
    why = {}
    voice = _numbered(found, vocal, "vocal_track")
    if voice is None:
        voice, why["voice"] = _auto_voice(pitched, list(syllables), division)
    else:
        why["voice"] = "chosen"
    chosen = None
    if instrument != "none":
        chosen = _numbered(found, instrument, "instrument_track")
        if chosen is None:
            chosen, reason = _auto_instrument([part for part in pitched if part is not voice])
            if chosen is not None:
                why["instrument"] = reason
        elif chosen is voice:
            raise ValueError("'instrument_track' and 'vocal_track' are both track {}; one part cannot be both "
                             "lines of the score".format(voice.number))
        else:
            why["instrument"] = "chosen"
    return {"voice": voice, "instrument": chosen, "why": why}


def describe(found: list, chosen: dict) -> list:
    """The parts as the node lists them: number, name, family, notes, range, drums, polyphony, role and why."""
    rows = []
    for part in found:
        role = "voice" if part is chosen.get("voice") else "instrument" if part is chosen.get("instrument") else ""
        rows.append({"number": part.number, "name": part.name, "family": part.family, "notes": len(part.notes),
                     "low": min(note.pitch for note in part.notes), "high": max(note.pitch for note in part.notes),
                     "drums": part.drums, "polyphony": part.polyphony, "role": role,
                     "why": chosen.get("why", {}).get(role, "")})
    return rows
