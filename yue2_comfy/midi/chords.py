"""Chord symbols guessed from what a MIDI file's parts play, for 'cot' at 'full'.

A file has no chord symbols, only notes: the harmony is in what sounds
together. Each bar is cut in halves where its meter has an even number of
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
"""

from __future__ import annotations

import math

from ..sheetsage import abc_rebuild

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
