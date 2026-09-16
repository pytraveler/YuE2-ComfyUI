"""A score written out as a MIDI file: what 'Save as MIDI...' in the score editor hands over.

The score is read with upstream's reader, abc_tools, so the file holds the
notes the model reads, with ties joined. A conductor track carries the tempo,
every change of meter at the bar it starts in, every key signature, and a
marker at each section the score names. A track follows for each line of the
score that sounds -- 'Vocal' on channel 1 with a voice sound, 'Ins' on channel
2 with a piano -- and, where the score has chord symbols, 'Chords' on channel 3
with strings holding each chord until the next symbol: the root between C3 and
B3, the rest of the chord stacked above it, and a bass written after a slash an
octave below. Every track runs to the end of the score, so bars of rest at the
end are kept.

The track names are the ones 'YuE2 Load MIDI' looks for, so a file saved here
and loaded again gives the same two lines in the same bars at the same tempo.
Chord symbols come back only as far as the loader's guess reads them.
"""

from __future__ import annotations

import re
from fractions import Fraction

from ..sheetsage import sections as score_sections
from ..vendor.yue2_music import abc_tools
from . import smf

QUALITY_STEPS = {"": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8), "7": (0, 4, 7, 10),
                 "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10), "dim7": (0, 3, 6, 9), "m7b5": (0, 3, 6, 10),
                 "sus4": (0, 5, 7), "sus2": (0, 2, 7), "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9),
                 "7sus4": (0, 5, 7, 10), "m(maj7)": (0, 3, 7, 11)}
"""The steps above the root of every chord quality the score format knows, as the score spells them."""

NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SYMBOL = re.compile(r"(?P<root>[A-G](?:bb|##|b|#)?)(?P<quality>.*?)(?:/(?P<bass>[A-G](?:bb|##|b|#)?))?")
LINES = (("Vocal", 0, 53, 100), ("Ins", 1, 0, 90))
"""Each line of the score as a track: its name, channel, General MIDI program (from 0) and velocity."""

CHORDS = ("Chords", 2, 48, 64)
CHORD_ROOT = 48
CONDUCTOR = "YuE2 score"


def _pitch_class(name: str) -> int:
    return (NATURAL[name[0]] + name.count("#") - name.count("b")) % 12


def _tick(quarters) -> int:
    return int(round(Fraction(quarters) * smf.DIVISION))


def chord_pitches(symbol: str) -> list:
    """The pitches a chord symbol is held as: the root in C3-B3, the chord above it, a slash bass an octave below."""
    found = SYMBOL.fullmatch(str(symbol).strip())
    if found is None or found.group("quality") not in QUALITY_STEPS:
        raise ValueError("'{}' is not a chord symbol the score format knows".format(symbol))
    root = CHORD_ROOT + _pitch_class(found.group("root"))
    pitches = [root + step for step in QUALITY_STEPS[found.group("quality")]]
    if found.group("bass"):
        pitches.insert(0, CHORD_ROOT - 12 + _pitch_class(found.group("bass")))
    return pitches


def midi_of(text: str) -> bytes:
    """The score as a format-1 MIDI file; a ValueError says why the score cannot be read."""
    score = abc_tools.parse(text)
    vocal = score.voices["Vocal"]
    end = _tick(vocal.time)
    conductor = [(0, smf.text(smf.NAME, CONDUCTOR)), (0, smf.tempo(round(60e6 / score.bpm)))]
    meter = None
    for start, _length, bar_meter in vocal.bars:
        if bar_meter != meter:
            conductor.append((_tick(start), smf.meter(*bar_meter)))
            meter = bar_meter
    key = None
    for start, name in vocal.keys:
        if name != key:
            conductor.append((_tick(start), smf.key(abc_tools.KEYS[name], name.endswith("m"))))
            key = name
    for section in score_sections.sections(text):
        if section["label"]:
            start = Fraction(section["start"]).limit_denominator(1024)
            conductor.append((_tick(start), smf.text(smf.MARKER, section["label"])))
    tracks = [conductor]
    for name, channel, program, velocity in LINES:
        notes = score.voices[name].notes
        if not notes:
            continue
        events = [(0, smf.text(smf.NAME, name)), (0, smf.program(channel, program))]
        for onset, pitch, length in notes:
            events.append((_tick(onset), smf.note_on(channel, pitch, velocity)))
            events.append((_tick(onset + length), smf.note_off(channel, pitch)))
        tracks.append(events)
    if vocal.chords:
        name, channel, program, velocity = CHORDS
        events = [(0, smf.text(smf.NAME, name)), (0, smf.program(channel, program))]
        for index, (onset, symbol) in enumerate(vocal.chords):
            stop = vocal.chords[index + 1][0] if index + 1 < len(vocal.chords) else vocal.time
            if stop <= onset:
                continue
            for pitch in chord_pitches(symbol):
                events.append((_tick(onset), smf.note_on(channel, pitch, velocity)))
                events.append((_tick(stop), smf.note_off(channel, pitch)))
        tracks.append(events)
    return smf.write(tracks, smf.DIVISION, end)
