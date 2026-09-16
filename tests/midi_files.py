"""MIDI files built in memory for the tests: a conductor track, then one track per part.

Files go through ``smf.write`` and back through ``smf.read``, so every test
sees a file the way the node does; the reader itself is checked on bytes built
by hand in test_midi_smf.py.
"""

from __future__ import annotations

from yue2_comfy.midi import smf


def file_of(*parts, meters=((0, 4, 4),), tempos=((0, 500000),), keys=(), markers=(), division=480):
    """A file with a conductor track and one track per part, each ``(name, channel, program, notes, texts)``.

    ``notes`` are ``(start, end, pitch)`` in ticks and ``texts`` are
    ``(tick, meta kind, bytes)``; a name may be text or bytes.
    """
    conductor = [(tick, smf.tempo(value)) for tick, value in tempos]
    conductor += [(tick, smf.meter(numerator, denominator)) for tick, numerator, denominator in meters]
    conductor += [(tick, smf.key(sharps, minor)) for tick, sharps, minor in keys]
    conductor += [(tick, smf.text(smf.MARKER, label)) for tick, label in markers]
    written = [conductor]
    for name, channel, program, notes, texts in parts:
        events = []
        if name:
            events.append((0, smf.meta(smf.NAME, name if isinstance(name, bytes) else name.encode("utf-8"))))
        events.append((0, smf.program(channel, program)))
        for start, end, pitch in notes:
            events += [(start, smf.note_on(channel, pitch, 90)), (end, smf.note_off(channel, pitch))]
        events += [(tick, smf.meta(kind, payload)) for tick, kind, payload in texts]
        written.append(events)
    return smf.write(written, division)


def song_of(*parts, **timing):
    """The same file, read back the way the node reads it."""
    return smf.read(file_of(*parts, **timing))


def line(pitches, start=0, step=480):
    """Notes one after another, each ``step`` ticks long."""
    return [(start + index * step, start + (index + 1) * step, pitch) for index, pitch in enumerate(pitches)]


def chord(pitches, start, end):
    """Notes struck and released together."""
    return [(start, end, pitch) for pitch in pitches]


def kar_texts(syllables, start=0, step=240):
    """A .kar words track: two '@' header lines, then a text event per syllable."""
    texts = [(0, smf.TEXT, b"@KMIDI KARAOKE FILE"), (0, smf.TEXT, b"@TSong title")]
    return texts + [(start + index * step, smf.TEXT, piece) for index, piece in enumerate(syllables)]
