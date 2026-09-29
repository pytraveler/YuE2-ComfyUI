"""Chord symbols from what a MIDI file's parts play, for 'cot' at 'full': read from a chords track, or guessed.

A file has no chord symbols, only notes, but a track named Chords -- the one
'Save as MIDI...' writes, perhaps put right by ear in a sequencer since -- is
held chords and nothing else, so it is read chord by chord. A chord starts
wherever a note of the track starts, and is the notes struck there with the
ones carried over from before that sound for at least half of it, so a common
tone held across a change counts and a note let ring a little late does not.
Those notes are named only when they are exactly one of the fifteen chords the
score format knows, on any root, with a bass under it: the lowest note, when
it is not the root. Some sets are two chords at once -- C, E, G and A is C6
and Am7 -- and the lowest note decides: the root when it is one, else the
chord over it. The track 'Save as MIDI...' writes has a layout of its own that
decides the rest: the root between C3 and B3 with the chord above it, and a
bass written after a slash below C3, so a lowest note below C3 under the rest
from C3 up is such a bass -- C/A rather than Am7, Am7/C rather than C6. A set
that is no chord of the format -- an added ninth, an open fifth, a single note
-- gets the chord the guess below would give it, and the node names its bars.

A file without such a track has the harmony in what its parts sound
together, and it is guessed. Each bar is cut in halves where its meter has an even number of
beats, four or more, and is taken whole otherwise. A piece in which no two
pitch classes ever sound at once -- a bass line or a tune alone -- is no
chord. In every other piece the pitch classes sounding are weighed by how long
they sound -- the voice at half weight, since a tune's passing notes are not
the harmony -- and the lowest note gets a bonus, since the bass usually plays
the root. The piece is compared with seven chord shapes on every root -- major,
minor, dominant seventh, major seventh, minor seventh, diminished and
suspended fourth -- every shape but the two plain triads needing a little more
evidence, and a shape whose notes all lie in the key needing a little less --
D minor over D major in a song in D minor, where the third is not played. A
piece with fewer than two pitch classes of any weight is no chord either, and
neighbours with the same chord are one.

This is a guess, and the node says so: close enough for 'cot' at 'full' to
follow the file's harmony, not a transcription of it.

Both name roots with sharps; the score names them from its key afterwards.
"""

from __future__ import annotations

import math

from ..sheetsage import abc_rebuild
from . import export

STEPS = {quality: export.QUALITY_STEPS[text] for quality, text in abc_rebuild.QUALITY_TEXT.items()}
"""The steps above the root of every quality the score format knows, under the labels the writer reads, triads first."""

DEGREES = ("1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7")
TONE_DEGREES = {"dim": {6: "b5"}, "hdim7": {6: "b5"}, "dim7": {6: "b5", 9: "bb7"}, "aug": {8: "#5"}}
"""A bass named as its degree above the root, so it is spelled from the root's name; chord tones as the chord spells them."""

SHAPES = (("maj", (0, 4, 7), 0.0), ("min", (0, 3, 7), 0.0), ("7", (0, 4, 7, 10), 0.03),
          ("maj7", (0, 4, 7, 11), 0.03), ("min7", (0, 3, 7, 10), 0.03), ("dim", (0, 3, 6), 0.03),
          ("sus4", (0, 5, 7), 0.05))
"""Each chord shape: its quality, its steps above the root, and what it costs against a plain major or minor triad."""

BASS_SHARE = 0.35
"""The lowest note's bonus weight, as a share of all the weight in the piece."""

ROOT_BONUS = 0.05
IN_KEY_BONUS = 0.02
VOICE_WEIGHT = 0.5
QUIET_SHARE = 0.1
"""A pitch class with less than this share of a piece's weight does not count towards it being a chord."""


def spans(bars: list) -> list:
    """``(start, end)`` of each piece of harmony, from bars given as ``(start, end, numerator)``."""
    found = []
    for start, end, numerator in bars:
        if numerator >= 4 and numerator % 2 == 0:
            middle = start + (end - start) / 2
            found.extend([(start, middle), (middle, end)])
        else:
            found.append((start, end))
    return found


def label(weights: list, bass: int, flats: bool = False, scale=None) -> str:
    """The chord -- ``C:min7``, or ``N`` for none -- that best fits twelve pitch-class weights and a bass pitch class."""
    total = sum(weights)
    if total <= 0 or sum(1 for weight in weights if weight >= QUIET_SHARE * total) < 2:
        return "N"
    weights = list(weights)
    weights[bass] += BASS_SHARE * total
    norm = math.sqrt(sum(weight * weight for weight in weights))
    names = abc_rebuild.FLAT_NAMES if flats else abc_rebuild.SHARP_NAMES
    best, best_score = "N", -math.inf
    for root in range(12):
        for quality, shape, cost in SHAPES:
            score = sum(weights[(root + step) % 12] for step in shape) / (norm * math.sqrt(len(shape))) - cost
            if root == bass:
                score += ROOT_BONUS
            if scale is not None and all((root + step) % 12 in scale for step in shape):
                score += IN_KEY_BONUS
            if score > best_score:
                best, best_score = names[root] + ":" + quality, score
    return best


def together(sounding: list) -> bool:
    """Whether two different pitch classes ever sound at once among ``(start, end, pitch class)``."""
    ordered = sorted(sounding)
    for index, (_start, end, pitch_class) in enumerate(ordered):
        for other_start, _other_end, other_class in ordered[index + 1:]:
            if other_start >= end:
                break
            if other_class != pitch_class:
                return True
    return False


def guess(notes: list, pieces: list, flats: bool = False, scale=None) -> list:
    """``[start, end, chord]`` for the pieces, from notes ``(start, end, pitch, weight)``; equal neighbours merged."""
    ordered = sorted(notes)
    rows = []
    for start, end in pieces:
        weights = [0.0] * 12
        sounding = []
        lowest = None
        for note_start, note_end, pitch, weight in ordered:
            if note_start >= end:
                break
            overlap = min(end, note_end) - max(start, note_start)
            if overlap <= 0:
                continue
            weights[pitch % 12] += float(overlap) * weight
            sounding.append((max(start, note_start), min(end, note_end), pitch % 12))
            lowest = pitch if lowest is None else min(lowest, pitch)
        chord = label(weights, lowest % 12, flats, scale) if together(sounding) else "N"
        if rows and rows[-1][2] == chord and rows[-1][1] == start:
            rows[-1][1] = end
        else:
            rows.append([start, end, chord])
    return rows


def _shape(classes: set, prefer: list):
    """``(root, quality)`` of the chord made of exactly these pitch classes, a preferred root first, or None."""
    found = [(root, quality) for quality, steps in STEPS.items() for root in range(12)
             if {(root + step) % 12 for step in steps} == classes]
    for root in prefer:
        for candidate in found:
            if candidate[0] == root:
                return candidate
    return found[0] if found else None


def spelled(pitches) -> str:
    """The chord label -- ``A:min7/b7`` -- that pitches sounding together make exactly, or None when they make none."""
    pitches = sorted(set(pitches))
    classes = {pitch % 12 for pitch in pitches}
    if len(classes) < 3:
        return None
    bass, upper = pitches[0], pitches[1:]
    above = {pitch % 12 for pitch in upper}
    tries = []
    if bass < export.CHORD_ROOT <= upper[0]:
        tries.append((above, [upper[0] % 12, bass % 12]))
    tries.append((classes, [bass % 12, upper[0] % 12]))
    if bass % 12 not in above:
        tries.append((above, [upper[0] % 12]))
    for wanted, prefer in tries:
        found = _shape(wanted, prefer)
        if found is None:
            continue
        root, quality = found
        label = abc_rebuild.SHARP_NAMES[root] + ":" + quality
        step = (bass - root) % 12
        if step:
            label += "/" + TONE_DEGREES.get(quality, {}).get(step, DEGREES[step])
        return label
    return None


def read(notes: list, end: int, scale=None) -> tuple:
    """``([start, end, chord], unnamed)`` from a chords track's notes ``(start, end, pitch)``; equal neighbours merged.

    Times are whatever the notes are given in -- the score's grid points --
    and the last chord runs to ``end``. ``unnamed`` holds the start of every
    chord whose notes are no chord of the score format, given the guess.
    """
    onsets = sorted({start for start, _stop, _pitch in notes})
    rows = []
    unnamed = []
    for index, onset in enumerate(onsets):
        stop = onsets[index + 1] if index + 1 < len(onsets) else max(end, onset + 1)
        sounding = [pitch for start, finish, pitch in notes
                    if start == onset or (start < onset < finish and 2 * (min(finish, stop) - onset) >= stop - onset)]
        chord = spelled(sounding)
        if chord is None:
            unnamed.append(onset)
            weights = [0.0] * 12
            for pitch in sounding:
                weights[pitch % 12] += 1.0
            chord = label(weights, min(sounding) % 12, scale=scale)
        if rows and rows[-1][2] == chord and rows[-1][1] == onset:
            rows[-1][1] = stop
        else:
            rows.append([onset, stop, chord])
    return rows, unnamed
