"""A MIDI file as a score YuE2 sings: the rows ``sheetsage.abc_rebuild`` writes, and the lyrics to go with them.

Timing. A score has one tempo, so the file's tempo map is averaged over its
length, the way a transcription's is, and every beat is placed at that tempo.
Notes are placed by tick on the same beats, so they land on the grid points
exactly however the file's tempo moved. Bars come from the file's meters, a
meter that changes in the middle of a bar starts a bar there, and the song
runs to its last note or to the end of its longest track, whichever is later,
filled out to a whole bar. A file that states no meter is in 4/4.

The grid. A score is written on equal steps inside each beat. A file written
in a sequencer sits on a grid of its own: the score takes sixteenths when
every note starts and ends on one, and thirty-seconds when nine note edges in
ten lie on one, so a stray thirty-second is kept; a file played in by hand, on
neither, is rounded to sixteenths. The steps in a beat follow from the grid
and the longest beat among the file's meters, so no beat gets a coarser one.
A note shorter than one step is held for one.

The two lines. The voice and the instrument are the parts ``tracks.choose``
picks, and each is one note at a time, as a line of the score is: of notes
struck together the top one is kept, a lower note starting under a higher one
is dropped unless most of it sounds after the higher one ends, and a note is
cut where the next one begins. A line whose middle pitch lies outside where
YuE2's scores keep that line -- C4 to A#5 for the voice, C4 to E6 for the
instrument, the range of the middle pitches of sixteen of its own and
SheetSage2's scores -- is moved by whole octaves until it lies inside.

The key. The file's key signatures are used when there are several, or when
the only one fits the notes nearly as well as the best key does; otherwise
the key is estimated from how long each pitch class sounds, with the major and
minor key profiles of Krumhansl and Kessler (1982) as printed in Krumhansl,
"Cognitive Foundations of Musical Pitch" (1990).

Sections. Markers named after sections become the score's section comments;
without them the sections of the karaoke words do, with an intro before the
first words when a bar or more comes first; a file with neither has none.
With 'full', chord symbols are guessed by ``chords``.
"""

from __future__ import annotations

import bisect
import math
from fractions import Fraction

from ..sheetsage import abc_rebuild
from ..sheetsage import sections as score_sections
from . import chords, karaoke, smf, tracks

MODES = ("melody", "full")
VOICE_WINDOW = (60, 82)
INSTRUMENT_WINDOW = (60, 88)
ON_GRID = 0.9
KEY_SLACK = 0.05
TEMPO_SLACK = 0.01
MOST_BEATS = 20000
"""Beats a score may run to: an hour at 330 BPM. A file longer than that is refused rather than written."""

MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
MAJOR_KEYS = ("Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#")
MINOR_KEYS = ("Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m")

TEMPO_CHANGES = (
    "The file's tempo moves between {low:.0f} and {high:.0f} BPM. A score keeps one tempo, so it is written at "
    "the average, {bpm:.0f} BPM, and the song will not speed up or slow down where the file does."
)
MOVED = "The {line} line (track {number}) sat {where} for YuE2's scores, so it was moved {way} {count}."
STRUCK = (
    "Track {number} strikes {count} notes together with a higher one. A line of the score holds one note at a "
    "time, so the top note of each chord was kept."
)
PLAYED_IN = (
    "The notes of the file do not sit on a grid of sixteenths or thirty-seconds -- it was probably played in "
    "by hand -- so they were rounded to the nearest sixteenth."
)
CHORDS_GUESSED = (
    "The chord symbols were guessed from what the file's parts play together: a harmony to follow, not a "
    "transcription of it."
)


def seconds_at(song: smf.Song, tick) -> float:
    """Seconds from the start of the song to ``tick``, through its tempo map."""
    seconds = 0.0
    last_tick = 0
    tempo = smf.DEFAULT_TEMPO
    for change, value in song.tempos:
        if change >= tick:
            break
        seconds += (change - last_tick) * tempo / 1e6 / song.division
        last_tick, tempo = change, value
    return seconds + float(tick - last_tick) * tempo / 1e6 / song.division


def _meters(song: smf.Song) -> list:
    rows = []
    for tick, numerator, denominator in song.meters:
        if rows and rows[-1][0] == tick:
            rows[-1] = (tick, numerator, denominator)
        elif not rows or rows[-1][1:] != (numerator, denominator):
            rows.append((tick, numerator, denominator))
    if not rows or rows[0][0] > 0:
        rows.insert(0, (0, 4, 4))
    return rows


def beats(song: smf.Song, end: int) -> list:
    """``(tick, number, numerator, denominator)`` for every beat up to the first downbeat at or after ``end``."""
    meters = _meters(song)
    rows = []
    tick = Fraction(0)
    number = 1
    index = 0
    numerator, denominator = meters[0][1], meters[0][2]
    while True:
        while index + 1 < len(meters) and meters[index + 1][0] <= tick:
            index += 1
            numerator, denominator = meters[index][1], meters[index][2]
            number = 1
        rows.append((tick, number, numerator, denominator))
        if number == 1 and tick >= end and len(rows) > 1:
            return rows
        if len(rows) > MOST_BEATS:
            raise ValueError("the file runs past {} beats, longer than any song".format(MOST_BEATS))
        step = Fraction(song.division * 4, denominator)
        following = meters[index + 1][0] if index + 1 < len(meters) else None
        tick = Fraction(following) if following is not None and tick < following < tick + step else tick + step
        number = number % numerator + 1


def grid_of(notes: list, division: int) -> int:
    """16 or 32: sixteenths when every edge lies on one, thirty-seconds when nine in ten do, else sixteenths."""
    edges = [tick for note in notes for tick in (note.start, note.end)]
    if all((tick / Fraction(division, 4)).denominator == 1 for tick in edges):
        return 16
    if sum(1 for tick in edges if (tick / Fraction(division, 8)).denominator == 1) >= ON_GRID * len(edges):
        return 32
    return 16


def points_of(rows: list, subbeats: int) -> list:
    """The tick of every grid point: ``subbeats`` equal steps inside each beat, and the last beat."""
    points = []
    for (tick, *_rest), (following, *_more) in zip(rows, rows[1:]):
        step = (following - tick) / subbeats
        points.extend(tick + k * step for k in range(subbeats))
    points.append(rows[-1][0])
    return points


def snap(points: list, tick) -> int:
    """The index of the grid point nearest ``tick``; halfway goes to the earlier one."""
    at = bisect.bisect_left(points, tick)
    if at <= 0:
        return 0
    if at >= len(points):
        return len(points) - 1
    return at if points[at] - tick < tick - points[at - 1] else at - 1


def octave_shift(median: float, window: tuple) -> int:
    """Semitones, whole octaves, that bring a middle pitch inside ``window``; 0 when it is inside."""
    low, high = window
    if median < low:
        return 12 * math.ceil((low - median) / 12)
    if median > high:
        return -12 * math.ceil((median - high) / 12)
    return 0


def skyline(notes: list) -> tuple:
    """A line of ``[start, end, pitch]`` on grid indices, one note at a time, and how many chord notes it dropped."""
    kept = []
    struck = 0
    for start, end, pitch in sorted(notes, key=lambda note: (note[0], -note[2], note[1])):
        if not kept or start >= kept[-1][1]:
            kept.append([start, end, pitch])
            continue
        last = kept[-1]
        if start == last[0]:
            struck += 1
        elif pitch > last[2]:
            last[1] = start
            kept.append([start, end, pitch])
        elif end - last[1] > last[1] - start:
            kept.append([last[1], end, pitch])
    return kept, struck


def _line(part: tracks.Part, points: list, window: tuple) -> tuple:
    """A part as a line on the grid, moved into ``window`` by the middle pitch of the notes the line keeps."""
    notes = []
    for note in part.notes:
        start = snap(points, note.start)
        end = min(max(snap(points, note.end), start + 1), len(points) - 1)
        if end > start:
            notes.append((start, end, note.pitch))
    kept, struck = skyline(notes)
    if not kept:
        return [], 0, struck
    pitches = sorted(pitch for _start, _end, pitch in kept)
    shift = octave_shift((pitches[(len(pitches) - 1) // 2] + pitches[len(pitches) // 2]) / 2, window)
    return [[start, end, pitch + shift] for start, end, pitch in kept if 0 <= pitch + shift <= 127], shift, struck


def _correlation(weights: list, profile: tuple, root: int) -> float:
    rotated = [weights[(root + step) % 12] for step in range(12)]
    mean_a = sum(rotated) / 12
    mean_b = sum(profile) / 12
    top = sum((a - mean_a) * (b - mean_b) for a, b in zip(rotated, profile))
    bottom = math.sqrt(sum((a - mean_a) ** 2 for a in rotated) * sum((b - mean_b) ** 2 for b in profile))
    return top / bottom if bottom else 0.0


def estimate_key(weights: list) -> tuple:
    """``(correlation, root pitch class, minor)`` of the key whose profile best fits twelve pitch-class weights."""
    return max((_correlation(weights, profile, root), root, minor)
               for root in range(12) for profile, minor in ((MAJOR_PROFILE, False), (MINOR_PROFILE, True)))


def signature_name(sharps: int, minor: bool) -> str:
    """The ABC key a key signature names: sharps positive, flats negative."""
    return (MINOR_KEYS if minor else MAJOR_KEYS)[sharps + 7]


def _keys(song: smf.Song, weights: list) -> tuple:
    distinct = []
    for tick, sharps, minor in song.keys:
        if not distinct or distinct[-1][1:] != (sharps, minor):
            distinct.append((tick, sharps, minor))
    best = estimate_key(weights)
    if len(distinct) >= 2:
        return [(0 if index == 0 else tick, signature_name(sharps, minor))
                for index, (tick, sharps, minor) in enumerate(distinct)], "file"
    if len(distinct) == 1:
        _tick, sharps, minor = distinct[0]
        root = (7 * sharps + (9 if minor else 0)) % 12
        if _correlation(weights, MINOR_PROFILE if minor else MAJOR_PROFILE, root) >= best[0] - KEY_SLACK:
            return [(0, signature_name(sharps, minor))], "file"
    name = abc_rebuild.key_text(abc_rebuild.SHARP_NAMES[best[1]] + (":minor" if best[2] else ":major"))
    return [(0, name)], "estimated"


def _intervals(starts: list, last) -> list:
    """``[start, end, value]`` from ``(tick, value)`` starts, each ending where the next begins or at ``last``."""
    rows = []
    for index, (tick, value) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else last
        if tick < end:
            rows.append([tick, end, value])
    return rows


def _bars(rows: list) -> list:
    downbeats = [(tick, numerator) for tick, number, numerator, _denominator in rows if number == 1]
    return [(tick, following, numerator) for (tick, numerator), (following, _next) in zip(downbeats, downbeats[1:])]


def _sections(song: smf.Song, words, bars: list) -> list:
    marked = karaoke.marker_sections(song)
    if marked:
        return marked
    if words is None:
        return []
    starts = [(section["tick"], section["label"]) for section in words.sections]
    if bars and starts and starts[0][0] >= bars[0][1]:
        starts.insert(0, (0, "intro"))
    return starts


def convert(song: smf.Song, mode: str = "melody", vocal="auto", instrument="auto") -> dict:
    """A file's score and lyrics for one mode and one choice of parts, with what the node says about them.

    Returns ``{"abc", "lyrics", "parts", "facts", "notices"}``: ``parts`` is
    ``tracks.describe``'s list, ``facts`` what the node's summary shows, and
    ``notices`` sentences for the node to say. Raises ValueError with the reason
    a score cannot be written.
    """
    if mode not in MODES:
        raise ValueError("'mode' must be one of {}".format(", ".join(MODES)))
    found = tracks.parts(song)
    if not found:
        raise ValueError("the file has no notes")
    words = karaoke.read(song)
    chosen = tracks.choose(found, vocal, instrument, words.syllables if words else (), song.division)
    end = max(song.end, max(note.end for part in found for note in part.notes))
    rows = beats(song, end)
    lines = [(chosen["voice"], VOICE_WINDOW, "voice")]
    if chosen["instrument"] is not None:
        lines.append((chosen["instrument"], INSTRUMENT_WINDOW, "instrument"))
    grid = grid_of([note for part, _window, _name in lines for note in part.notes], song.division)
    subbeats = max(1, grid // min(row[3] for row in rows))
    points = points_of(rows, subbeats)
    last = rows[-1][0]
    length = seconds_at(song, last)
    per_tick = length / float(last)
    bpm = float(last) / song.division * 60.0 / length

    def at(tick) -> float:
        return float(tick) * per_tick

    notices = []
    tempos = [60e6 / value for _tick, value in song.tempos if _tick < last] or [60e6 / smf.DEFAULT_TEMPO]
    if max(tempos) > min(tempos) * (1 + TEMPO_SLACK):
        notices.append(TEMPO_CHANGES.format(low=min(tempos), high=max(tempos), bpm=bpm))
    edges = [tick for part, _window, _name in lines for note in part.notes for tick in (note.start, note.end)]
    if grid == 16 and edges and sum(1 for tick in edges if (tick / Fraction(song.division, 4)).denominator == 1) \
            < ON_GRID * len(edges):
        notices.append(PLAYED_IN)
    notes = []
    shifts = {}
    for index, (part, window, name) in enumerate(lines):
        kept, shift, struck = _line(part, points, window)
        shifts[name] = shift
        notes.extend([at(points[start]), at(points[stop]), pitch, index] for start, stop, pitch in kept)
        if shift:
            octaves = abs(shift) // 12
            notices.append(MOVED.format(line=name, number=part.number, where="low" if shift > 0 else "high",
                                        way="up" if shift > 0 else "down",
                                        count="an octave" if octaves == 1 else "{} octaves".format(octaves)))
        if struck:
            notices.append(STRUCK.format(number=part.number, count=struck))
    weights = [0.0] * 12
    for part in found:
        if not part.drums:
            for note in part.notes:
                weights[note.pitch % 12] += note.end - note.start
    key_starts, key_source = _keys(song, weights)
    bars = _bars(rows)
    structure = _intervals([(tick, label) for tick, label in _sections(song, words, bars) if tick < last], last)
    harmony = []
    if mode == "full":
        voice = chosen["voice"]
        heard = [(note.start, note.end, note.pitch, chords.VOICE_WEIGHT if part is voice else 1.0)
                 for part in found if not part.drums for note in part.notes]
        signature = abc_rebuild.KEY_SIGNATURES[key_starts[0][1]]
        tonic = (7 * signature) % 12
        scale = {(tonic + step) % 12 for step in (0, 2, 4, 5, 7, 9, 11)}
        harmony = chords.guess(heard, chords.spans(bars), signature < 0, scale)
        notices.append(CHORDS_GUESSED)
    built = {"beats": [[at(tick), number, numerator, denominator] for tick, number, numerator, denominator in rows],
             "chords": [[at(start), at(stop), chord] for start, stop, chord in harmony],
             "keys": [[at(start), at(stop), name] for start, stop, name in _intervals(key_starts, last)],
             "structures": [[at(start), at(stop), label] for start, stop, label in structure],
             "notes": sorted(notes)}
    abc = abc_rebuild.build(built, melody_only=(mode == "melody"), subbeats=subbeats)
    lyrics = words.lyrics() if words else score_sections.skeleton(score_sections.sections(abc))
    facts = {"bpm": int(round(bpm)), "meter": "{}/{}".format(rows[0][2], rows[0][3]), "bars": len(bars),
             "seconds": round(length, 1), "key": key_starts[0][1], "key_source": key_source, "grid": grid,
             "voice": chosen["voice"].number,
             "instrument": chosen["instrument"].number if chosen["instrument"] is not None else None,
             "voice_shift": shifts.get("voice", 0), "instrument_shift": shifts.get("instrument", 0),
             "karaoke": words is not None}
    return {"abc": abc, "lyrics": lyrics, "parts": tracks.describe(found, chosen), "facts": facts,
            "notices": notices}
