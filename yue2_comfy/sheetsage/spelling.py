"""Keys and chords named the way the key they sound in names them.

SheetSage2's vocabulary names every root with a sharp, so a song in B-flat
minor came out of it as ``A#:minor`` with ``A#:min``, ``D#:min`` and ``F#:maj``
over it, and the score said so: ``K:A#m``, seven sharps where five flats
belong, and chord names YuE2 does not write in its own scores, which have
``Bbmaj7`` and ``Ebmaj7`` in B-flat and ``Db`` and ``Gb`` in B-flat minor.
Transposing such a score only made it worse -- up a tone, its ``A#`` became
``B#`` in C major. m-a-p's SheetSage2 has named its labels from the key since
2026-09-21, and put right a flaw in that on 2026-09-24; this is the same rule,
written for this pack.

The name is not cosmetic. The notes are written against the key's signature,
and under ``K:A#m`` every F is a bare E, meaning E-sharp, and every C a bare B,
meaning B-sharp. YuE2 sings those letters without the sharp. Measured on a
song in B-flat minor that later rises to B minor, four seeds a score: sung
from ``K:A#m`` not one of its 71 F and C notes was on pitch, 94 percent a
semitone low; sung from ``K:Bbm``, the same notes written F and C, 94
percent on pitch, and the tune of that part went from 61 to 94 percent. The
part in B minor, written alike in both, came out alike (92 and 85).
F-sharp major and D-sharp minor keep an E-sharp in their signature under
this rule too, and are not measured.

A key takes the name with fewer accidentals in its signature, and of two with
six the sharp one: D-flat major rather than C-sharp, B-flat minor rather than
A-sharp, F-sharp major and D-sharp minor as they are.

A chord takes the root that brings its tones closest to the key's own notes.
On the line of fifths (C at 0, G at 1, F at -1, a sharp seven further up, a
flat seven further down) the seven notes of a key whose signature has ``s``
sharps -- flats counted negative -- lie from ``s - 1`` to ``s + 5``. A chord
tone outside that band costs how far outside it lies, the root is counted
twice, and of the seven letters the root could be written with, the cheapest
wins, the first from C to B on a tie. So D-sharp major in C minor is E-flat,
C-sharp half-diminished in G major stays C-sharp, and F half-diminished in
G-sharp minor is E-sharp.

Only names change. The pitch, the time, the quality and the inversion of every
label stay as the model wrote them, and so does the model's own event list:
this runs on the rows a score is written from.
"""

from __future__ import annotations

import bisect
import re

LETTERS = "CDEFGAB"
NATURAL = (0, 2, 4, 5, 7, 9, 11)
FIFTHS = (0, 2, 4, -1, 1, 3, 5)
"""Each letter's place on the line of fifths, C at 0, in the order of ``LETTERS``."""

NAME = re.compile(r"^(?P<letter>[A-G])(?P<accidental>#{0,2}|b{0,2})$")
MODES = {"major": 0, "minor": 3}
"""How many fifths a key's tonic lies above the major key that shares its signature."""

TONES = {
    "maj": (0, 4, 1), "min": (0, -3, 1), "dim": (0, -3, -6), "aug": (0, 4, 8),
    "maj7": (0, 4, 1, 5), "min7": (0, -3, 1, -2), "7": (0, 4, 1, -2),
    "hdim7": (0, -3, -6, -2), "dim7": (0, -3, -6, -9), "minmaj7": (0, -3, 1, 5),
    "sus2": (0, 2, 1), "sus4": (0, -1, 1), "sus4(b7)": (0, -1, 1, -2),
    "maj6": (0, 4, 1, 3), "min6": (0, -3, 1, 3),
}
"""Every quality the model names, as its tones' steps along the line of fifths from the root, root first.

A major third is four fifths up and a minor one three down, a fifth one up, a
diminished fifth six down and an augmented one eight up, a sixth three up, a
seventh five up and a minor seventh two down, a second two up and a fourth one
down. The diminished seventh is the doubly flattened seventh, nine down, as a
chord built of thirds spells it.
"""

NO_CHORD = frozenset({"N", "X"})


def _parts(name: str):
    """``(letter, alteration)`` of a pitch name such as ``A#`` or ``Bb``, or None for anything else."""
    found = NAME.fullmatch(name)
    if found is None:
        return None
    accidental = found.group("accidental")
    return LETTERS.index(found.group("letter")), accidental.count("#") - accidental.count("b")


def _pitch_class(letter: int, alteration: int) -> int:
    return (NATURAL[letter] + alteration) % 12


def _place(letter: int, alteration: int) -> int:
    """Where a spelled note lies on the line of fifths."""
    return FIFTHS[letter] + 7 * alteration


def _named(letter: int, alteration: int) -> str:
    return LETTERS[letter] + ("#" * alteration if alteration > 0 else "b" * -alteration)


def _spellings(pitch_class: int):
    """Every letter that can name a pitch class, C to B, each with the alteration it needs (-6 to 5)."""
    for letter, natural in enumerate(NATURAL):
        yield letter, (pitch_class - natural + 6) % 12 - 6


def _key_parts(label: str):
    """``(tonic, mode)`` of a key label such as ``A#:minor``, or None when it is not one."""
    tonic, _colon, mode = label.partition(":")
    parts = _parts(tonic)
    if parts is None or mode not in MODES:
        return None
    return parts, mode


def key_name(label: str) -> str:
    """A key label such as ``A#:minor`` under the name whose signature reads easiest: ``Bb:minor``.

    Anything that is not a major or minor key label comes back as it is.
    """
    found = _key_parts(label)
    if found is None:
        return label
    (letter, alteration), mode = found
    shift = MODES[mode]
    candidates = [spelled for spelled in _spellings(_pitch_class(letter, alteration)) if abs(spelled[1]) <= 1]
    best = min(candidates, key=lambda spelled: (abs(_place(*spelled) - shift), _place(*spelled) < shift))
    return _named(*best) + ":" + mode


def signature(label: str) -> int:
    """How many sharps the key of a label is written with once it is named well, flats counted negative."""
    (letter, alteration), mode = _key_parts(key_name(label))
    return _place(letter, alteration) - MODES[mode]


def chord_name(label: str, key: str) -> str:
    """A chord label such as ``D#:maj/3`` with its root named from ``key``: ``Eb:maj/3`` in C minor.

    No chord, a quality the model does not have, or a key that is not a major
    or minor key label leaves the label as it is.
    """
    if label in NO_CHORD or ":" not in label or _key_parts(key) is None:
        return label
    root, rest = label.split(":", 1)
    parts = _parts(root)
    steps = TONES.get(rest.split("/", 1)[0])
    if parts is None or steps is None:
        return label
    centre = signature(key) + 2

    def cost(spelled) -> int:
        places = [_place(*spelled) + step for step in steps]
        return sum(max(0, abs(place - centre) - 3) for place in places + places[:1])

    return _named(*min(_spellings(_pitch_class(*parts)), key=cost)) + ":" + rest


def respelled(chords: list, keys: list) -> list:
    """Chord rows ``[start, end, label]`` named from the key sounding at each one's midpoint.

    A midpoint exactly where one key gives way to the next goes by the key that
    ends there, as m-a-p's release has it; before the first key and after the
    last the nearest one holds, and with no key at all the labels stay as they
    are. The rows given are not changed; new ones come back.
    """
    if not keys:
        return [list(row) for row in chords]
    ends = [float(row[1]) for row in keys[:-1]]
    return [[start, end, chord_name(label, keys[bisect.bisect_left(ends, (float(start) + float(end)) / 2.0)][2])]
            for start, end, label in chords]
