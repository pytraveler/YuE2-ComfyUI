"""Beats, chords, keys, sections and notes written out as two-voice ABC.

The score's bars come from the beats themselves: a bar runs from one downbeat
to the next, a first span before any downbeat is a pickup, and a last span
after the final downbeat is a partial bar. Every beat is split into four
subbeats, and everything else -- notes, chords, keys, section labels -- is
snapped to that grid by the midpoints between its points. The unit length is
the smallest note the grid needs in every meter present, and the tempo is the
average over the whole grid.

Bars are written in groups of at most four, and a new group starts where the
meter, the key or the section changes, with the section's name as a comment
above it -- the layout YuE2's own scores use. Chords go on the vocal voice
only; a bar of rests in a voice collapses into ``Z``. Notes are spelled from
the key, with an accidental written only where the bar's state changes.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import Counter

SUBBEATS = 4
VOICES = ("Vocal", "Ins")
NO_CHORDS = frozenset({"N", "X", "?"})
DURATIONS = frozenset({1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48})

QUALITY_TEXT = {"maj": "", "min": "m", "dim": "dim", "aug": "aug", "7": "7", "maj7": "maj7",
                "min7": "m7", "dim7": "dim7", "hdim7": "m7b5", "sus4": "sus4", "sus2": "sus2",
                "maj6": "6", "min6": "m6", "sus4(b7)": "7sus4", "minmaj7": "m(maj7)"}
NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
LETTERS = "CDEFGAB"
SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
ROOT = re.compile(r"^(?P<letter>[A-G])(?P<accidental>#{0,2}|b{0,2})$")
BASS_DEGREE = re.compile(r"^(?P<accidental>#{0,2}|b{0,2})(?P<degree>[1-9]|1[0-3])$")
KEY_SIGNATURES = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7, "F": -1,
                  "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7, "Am": 0, "Em": 1,
                  "Bm": 2, "F#m": 3, "C#m": 4, "G#m": 5, "D#m": 6, "A#m": 7, "Dm": -1, "Gm": -2,
                  "Cm": -3, "Fm": -4, "Bbm": -5, "Ebm": -6, "Abm": -7}
PITCH_NAMES = {
    7: ("B#", "C#", "C##", "D#", "D##", "E#", "F#", "F##", "G#", "G##", "A#", "B"),
    6: ("B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "G##", "A#", "B"),
    5: ("B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"),
    4: ("B#", "C#", "D", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"),
    3: ("B#", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"),
    2: ("C", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"),
    1: ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"),
    0: ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"),
    -1: ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"),
    -2: ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"),
    -3: ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"),
    -4: ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"),
    -5: ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"),
    -6: ("C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"),
    -7: ("C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "Bbb", "Bb", "Cb"),
}
"""The spelling of each pitch class in a key with that many sharps (positive) or flats."""

MUSIC_ELEMENT = re.compile(r'"(?P<quoted>[^"]*)"|\[K:(?P<key>[^\]]+)\]'
                           r"|(?P<note>[_=^]*[A-Ga-gz][,']*)(?P<duration>\d*)(?P<tie>-?)")


class NotationError(ValueError):
    """Why these events cannot be written as a score."""


class Beat:
    """One beat: when it sounds, its number in the bar, and the meter it declares."""

    __slots__ = ("time", "number", "numerator", "denominator")

    def __init__(self, time, number, numerator, denominator):
        self.time = float(time)
        self.number = int(number)
        self.numerator = int(numerator)
        self.denominator = int(denominator)


class Bar:
    """A bar as a span of beats, the meter it has, and the meter it is written in."""

    __slots__ = ("index", "start", "end", "numerator", "denominator", "pickup", "partial",
                 "written_numerator", "written_denominator", "pad_before")

    def __init__(self, index, start, end, numerator, denominator, pickup=False, partial=False,
                 written_numerator=None, written_denominator=None, pad_before=False):
        self.index = index
        self.start = start
        self.end = end
        self.numerator = numerator
        self.denominator = denominator
        self.pickup = pickup
        self.partial = partial
        self.written_numerator = written_numerator
        self.written_denominator = written_denominator
        self.pad_before = pad_before

    @property
    def start_t(self):
        return self.start * SUBBEATS

    @property
    def end_t(self):
        return self.end * SUBBEATS

    @property
    def abc_numerator(self):
        return self.written_numerator or self.numerator

    @property
    def abc_denominator(self):
        return self.written_denominator or self.denominator


def _pitch_class(root: str) -> tuple:
    found = ROOT.fullmatch(root)
    if found is None:
        raise NotationError("Invalid pitch spelling {!r}".format(root))
    accidental = found.group("accidental")
    shift = accidental.count("#") - accidental.count("b")
    return (NATURAL[found.group("letter")] + shift) % 12, found.group("letter"), accidental


def portable_name(root: str, keep_double: bool = False) -> str:
    """A root spelled with at most one accidental, unless asked to keep a double."""
    pitch_class, _letter, accidental = _pitch_class(root)
    if keep_double or len(accidental) <= 1:
        return root
    return (SHARP_NAMES if accidental.startswith("#") else FLAT_NAMES)[pitch_class]


def _bass_pitch(root: str, degree_text: str) -> str:
    if ROOT.fullmatch(degree_text):
        return portable_name(degree_text, keep_double=True)
    found = BASS_DEGREE.fullmatch(degree_text)
    if found is None:
        raise NotationError("Invalid chord bass degree {!r}".format(degree_text))
    root_pc, root_letter, root_accidental = _pitch_class(root)
    degree = int(found.group("degree"))
    degree_accidental = found.group("accidental")
    interval = (0, 2, 4, 5, 7, 9, 11)[(degree - 1) % 7] + 12 * ((degree - 1) // 7)
    interval += degree_accidental.count("#") - degree_accidental.count("b")
    target = (root_pc + interval) % 12
    letter = LETTERS[(LETTERS.index(root_letter) + degree - 1) % 7]
    difference = (target - NATURAL[letter] + 6) % 12 - 6
    if difference in (-2, -1, 0, 1, 2):
        return letter + {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}[difference]
    names = SHARP_NAMES if "#" in root_accidental + degree_accidental else FLAT_NAMES
    return names[target]


def chord_text(chord: str):
    """A model chord label such as ``A:min7/b3`` as an ABC chord symbol, or None for no chord."""
    chord = chord.strip()
    if chord in NO_CHORDS:
        return None
    if ":" not in chord:
        raise NotationError("Chord {!r} is missing the ':' quality separator".format(chord))
    root, descriptor = chord.split(":", 1)
    quality, bass = descriptor.split("/", 1) if "/" in descriptor else (descriptor, None)
    if quality not in QUALITY_TEXT:
        raise NotationError("Unsupported chord quality {!r} in {!r}".format(quality, chord))
    text = portable_name(root, keep_double=True) + QUALITY_TEXT[quality]
    if bass:
        text += "/" + _bass_pitch(root, bass)
    return text


def key_text(key: str) -> str:
    """A model key label such as ``A#:major`` as an ABC key with a standard signature."""
    key = key.strip()
    if ":" in key:
        root, mode = key.split(":", 1)
        if mode not in ("major", "minor"):
            raise NotationError("Unsupported key mode {!r} in {!r}".format(mode, key))
    elif key.endswith("m"):
        root, mode = key[:-1], "minor"
    else:
        root, mode = key, "major"
    root_pc, _letter, accidental = _pitch_class(root)
    suffix = "m" if mode == "minor" else ""
    candidate = portable_name(root) + suffix
    if candidate in KEY_SIGNATURES:
        return candidate
    names = FLAT_NAMES if "b" in accidental else SHARP_NAMES
    candidate = names[root_pc] + suffix
    if candidate not in KEY_SIGNATURES:
        candidate = (SHARP_NAMES if names is FLAT_NAMES else FLAT_NAMES)[root_pc] + suffix
    if candidate not in KEY_SIGNATURES:
        raise NotationError("Cannot encode portable ABC key for {!r}".format(key))
    return candidate


def key_accidentals(key: str) -> list:
    """Per letter C..B: +1 sharp, -1 flat, 0 natural in this key's signature."""
    count = KEY_SIGNATURES[key]
    accidentals = [0] * 7
    for letter in ("FCGDAEB" if count > 0 else "BEADGCF")[:abs(count)]:
        accidentals[LETTERS.index(letter)] = 1 if count > 0 else -1
    return accidentals


def note_text(pitch: int, signature: list, bar_state: dict) -> str:
    """A MIDI pitch in ABC, spelled from the key; ``bar_state`` holds the bar's accidentals by letter."""
    name = PITCH_NAMES[sum(signature)][pitch % 12]
    letter, accidental = name[0], name[1:]
    alteration = {"": 0, "#": 1, "##": 2, "b": -1, "bb": -2}[accidental]
    octave = (pitch - 60) // 12
    if pitch % 12 == 11 and alteration == -1:
        octave += 1
    elif pitch % 12 == 0 and alteration == 1:
        octave -= 1
    index = LETTERS.index(letter)
    mark = ""
    if bar_state.get(index, signature[index]) != alteration:
        bar_state[index] = alteration
        mark = {-2: "__", -1: "_", 0: "=", 1: "^", 2: "^^"}[alteration]
    if octave > 0:
        letter = letter.lower() + "'" * (octave - 1)
    elif octave < 0:
        letter += "," * -octave
    return mark + letter


def _first_most_common(values: list) -> int:
    counts = Counter(values)
    top = max(counts.values())
    return next(value for value in values if counts[value] == top)


def infer_bars(beats: list) -> list:
    """Bars from the downbeats, with the meter each one actually has."""
    downbeats = [i for i, beat in enumerate(beats) if beat.number == 1]
    if not downbeats:
        raise NotationError("no beat is numbered 1, so no bar has a downbeat")
    spans = []
    if downbeats[0] > 0:
        spans.append((0, downbeats[0], True, False))
    spans.extend((a, b, False, False) for a, b in zip(downbeats, downbeats[1:]))
    if downbeats[-1] < len(beats) - 1:
        spans.append((downbeats[-1], len(beats) - 1, False, True))
    if not spans:
        raise NotationError("the downbeats leave no bar between them")
    bars = []
    for index, (start, end, pickup, partial) in enumerate(spans):
        members = beats[start:end]
        count = len(members)
        numbers = [beat.number for beat in members]
        if numbers != list(range(numbers[0], numbers[0] + count)):
            raise NotationError("bar {} counts its beats {!r}, not in order".format(index, numbers))
        if not pickup and numbers[0] != 1:
            raise NotationError("bar {} is a whole bar that does not begin on beat 1".format(index))
        denominators = [beat.denominator for beat in members]
        numerators = [beat.numerator for beat in members]
        denominator = _first_most_common(denominators)
        declared = _first_most_common(numerators)
        pad_final = (partial and len(set(numerators)) == 1
                     and all(value == denominator for value in denominators) and declared >= count)
        bars.append(Bar(index, start, end, count, denominator, pickup, partial,
                        written_numerator=declared if pad_final else count))
    if len(bars) >= 2:
        first, second = bars[0], bars[1]
        if first.numerator / first.denominator < second.abc_numerator / second.abc_denominator:
            first.written_numerator = second.abc_numerator
            first.written_denominator = second.abc_denominator
            first.pad_before = True
    return bars


class Grid:
    """The subbeat grid: when each point sounds, its quarter-note position, and its beat's denominator."""

    def __init__(self, beats: list, bars: list):
        interval_denominators = [0] * (len(beats) - 1)
        for bar in bars:
            for i in range(bar.start, bar.end):
                interval_denominators[i] = bar.denominator
        if any(value == 0 for value in interval_denominators):
            raise NotationError("a beat lies in no bar")
        self.times = []
        self.denominators = []
        self.quarters = [0.0]
        quarter = 0.0
        for i in range(len(beats) - 1):
            start, end = beats[i].time, beats[i + 1].time
            step = (end - start) / SUBBEATS
            denominator = interval_denominators[i]
            for k in range(SUBBEATS):
                self.times.append(float(k) * step + start)
                self.denominators.append(denominator)
                quarter += 4.0 / denominator / SUBBEATS
                self.quarters.append(quarter)
        self.times.append(beats[-1].time)
        self.denominators.append(interval_denominators[-1])
        self.boundaries = [(a + b) / 2 for a, b in zip(self.times, self.times[1:])]

    def at(self, seconds: float) -> int:
        """The grid point a time belongs to."""
        return bisect.bisect_left(self.boundaries, float(seconds))

    def clamp(self, index: int) -> int:
        return max(0, min(index, len(self.times) - 1))


def _fill(rows: list, grid: Grid, default) -> list:
    """Each grid point's value from timed rows; a row on the grid's last point alone is dropped, not refused."""
    values = [default] * len(grid.times)
    for start, end, value in rows:
        a = grid.clamp(grid.at(start))
        b = grid.clamp(grid.at(end))
        if b <= a:
            if a == len(values) - 1:
                continue
            raise NotationError("{} from {:.3f} s to {:.3f} s is shorter than a subbeat of the grid"
                                .format(value, start, end))
        for t in range(a, b):
            values[t] = value
    if len(values) > 1:
        values[-1] = values[-2]
    return values


def _voice(notes: list, grid: Grid, name: str) -> list:
    """A voice's notes on the grid as sustain values, onsets odd; a note on the grid's last point alone is dropped."""
    values = [0] * len(grid.times)
    for start, end, pitch in sorted(notes, key=lambda note: (note[0], note[1], note[2])):
        a = grid.clamp(grid.at(start))
        b = grid.clamp(grid.at(end))
        if b <= a:
            if a == len(values) - 1:
                continue
            raise NotationError("{} note {} from {:.3f} s to {:.3f} s is shorter than a subbeat of the grid"
                                .format(name, pitch, start, end))
        if any(values[t] for t in range(a, b)):
            raise NotationError("{} notes overlap at subbeats {} to {} once quantized".format(name, a, b))
        sustain = pitch * 2 + 2
        for t in range(a, b):
            values[t] = sustain
        values[a] = sustain + 1
    return values


class Score:
    """Everything the writer reads: bars, grid, per-point keys and chords, section starts, voices."""

    def __init__(self, beats, bars, grid, keys, chords, sections, voices):
        self.beats = beats
        self.bars = bars
        self.grid = grid
        self.keys = keys
        self.chords = chords
        self.sections = sections
        self.voices = voices


def _parse_rows(rows: list, what: str, check=None) -> list:
    parsed = []
    previous_end = None
    for start, end, value in rows:
        start, end, value = float(start), float(end), str(value).strip()
        if end <= start:
            raise NotationError("{} end must be after start".format(what))
        if previous_end is not None and start < previous_end - 1e-6:
            raise NotationError("overlapping {} intervals".format(what))
        parsed.append((start, end, check(value) if check else value))
        previous_end = end
    return parsed


def score(beat_rows, chord_rows, key_rows, structure_rows, notes, melody_only: bool = False) -> Score:
    """The score these rows describe; notes are ``[start, end, pitch, track]``, track 0 the vocal."""
    beats = []
    for row in beat_rows:
        beat = Beat(*row[:4])
        if beat.number < 1 or beat.numerator < 1:
            raise NotationError("beat IDs and meter numerators must be positive")
        if beat.denominator < 1 or beat.denominator & (beat.denominator - 1):
            raise NotationError("meter denominator must be a positive power of two")
        if beats and beat.time <= beats[-1].time:
            raise NotationError("beat times must be strictly increasing")
        beats.append(beat)
    if len(beats) < 2:
        raise NotationError("at least two beat events are required")
    keys = _parse_rows(key_rows, "key", key_text)
    if not keys:
        raise NotationError("at least one key interval is required")
    sections = _parse_rows(structure_rows, "structure")
    chords = [] if melody_only else _parse_rows(chord_rows, "chord")
    for _start, _end, chord in chords:
        chord_text(chord)
    bars = infer_bars(beats)
    grid = Grid(beats, bars)
    voices = {name: _voice([(s, e, p) for s, e, p, track in notes if track == index], grid, name)
              for index, name in enumerate(VOICES)}
    key_values = _fill(keys, grid, keys[0][2])
    chord_values = ["N"] * len(grid.times) if melody_only else _fill(chords, grid, "N")
    section_starts = [(grid.clamp(grid.at(start)), label) for start, _end, label in sections]
    return Score(beats, bars, grid, key_values, chord_values, section_starts, voices)


def unit_denominator(score_: Score) -> int:
    """The ABC unit length that expresses every subbeat of every meter exactly."""
    values = [denominator * SUBBEATS for bar in score_.bars
              for denominator in (bar.denominator, bar.abc_denominator)]
    denominator = math.lcm(*values)
    if denominator > 1024:
        raise NotationError("the meters want a unit note of 1/{}, too short to write".format(denominator))
    return denominator


def _units(score_: Score, start: int, end: int, unit: int) -> int:
    total = 0
    for denominator in score_.grid.denominators[start:end]:
        divisor = denominator * SUBBEATS
        if unit % divisor:
            raise NotationError("a unit note of 1/{} does not divide a 1/{} subbeat".format(unit, divisor))
        total += unit // divisor
    return total


def tempo(score_: Score) -> float:
    """Quarter notes per minute over the whole grid."""
    seconds = score_.grid.times[-1] - score_.grid.times[0]
    quarters = score_.grid.quarters[-1] - score_.grid.quarters[0]
    if seconds <= 0 or quarters <= 0:
        raise NotationError("the score takes no time, so it has no tempo")
    return quarters / seconds * 60.0


def _same_segment(value: int, following: int) -> bool:
    if value == 0:
        return following == 0
    return following == (value // 2 - 1) * 2 + 2


def _continues(value: int, following: int) -> bool:
    return value > 0 and following == (value // 2 - 1) * 2 + 2


def split_units(duration: int) -> list:
    """A length as note values strict ABC readers accept, largest first."""
    if duration <= 0:
        raise NotationError("a length of {} units cannot be written".format(duration))
    pieces = []
    remaining = int(duration)
    while remaining:
        if remaining in DURATIONS:
            pieces.append(remaining)
            break
        smaller = [value for value in DURATIONS if value < remaining]
        if not smaller:
            raise NotationError("a length of {} units is not a sum of note values".format(duration))
        pieces.append(max(smaller))
        remaining -= max(smaller)
    return pieces


def _tokens(prefix: str, text: str, duration: int, tie_out: bool) -> list:
    pieces = split_units(duration)
    written = []
    for i, piece in enumerate(pieces):
        tied = text != "z" and (i + 1 < len(pieces) or tie_out)
        written.append((prefix if i == 0 else "") + text + ("" if piece == 1 else str(piece)) + ("-" if tied else ""))
    return written


def _padding(bar: Bar, unit: int) -> int:
    return (bar.abc_numerator * unit // bar.abc_denominator) - (bar.numerator * unit // bar.denominator)


def _bar_text(score_: Score, voice: str, bar: Bar, unit: int) -> str:
    values = score_.voices[voice]
    with_chords = voice == "Vocal"
    bar_state = {}
    current_key = score_.keys[bar.start_t]
    signature = key_accidentals(current_key)
    padding = _padding(bar, unit)
    if padding < 0:
        raise NotationError("bar {} holds more beats than its written meter".format(bar.index))
    leading = padding if bar.pad_before else 0
    trailing = 0 if bar.pad_before else padding
    parts = []
    t = bar.start_t
    while t < bar.end_t:
        changes = [bar.end_t]
        for probe in range(t + 1, bar.end_t):
            if not _same_segment(values[t], values[probe]):
                changes.append(probe)
                break
        for probe in range(t + 1, bar.end_t):
            if score_.keys[probe] != score_.keys[probe - 1]:
                changes.append(probe)
                break
        if with_chords:
            for probe in range(t + 1, bar.end_t):
                if score_.chords[probe] != score_.chords[probe - 1]:
                    changes.append(probe)
                    break
        following = min(changes)
        prefix = ""
        if t > bar.start_t and score_.keys[t] != current_key:
            current_key = score_.keys[t]
            signature = key_accidentals(current_key)
            bar_state = {}
            prefix += "[K:{}]".format(current_key)
        if with_chords and (t == bar.start_t or score_.chords[t] != score_.chords[t - 1]):
            symbol = chord_text(score_.chords[t])
            if symbol is not None:
                prefix += '"{}"'.format(symbol)
        value = values[t]
        text = "z" if value == 0 else note_text(value // 2 - 1, signature, bar_state)
        duration = _units(score_, t, following, unit)
        if t == bar.start_t and leading:
            if value == 0 and not prefix:
                duration += leading
            else:
                parts.extend(_tokens("", "z", leading, False))
            leading = 0
        if value == 0 and following == bar.end_t and trailing:
            duration += trailing
            trailing = 0
        tie_out = value > 0 and following < len(values) and _continues(value, values[following])
        parts.extend(_tokens(prefix, text, duration, tie_out))
        t = following
    if trailing:
        parts.extend(_tokens("", "z", trailing, False))
    return "".join(parts)


def _full_rest(text: str) -> bool:
    cursor = 0
    saw = False
    for found in MUSIC_ELEMENT.finditer(text):
        if text[cursor:found.start()]:
            return False
        cursor = found.end()
        if found.group("quoted") is not None or found.group("key") is not None:
            return False
        saw = True
        if found.group("note") != "z" or found.group("tie"):
            return False
    return saw and cursor == len(text)


def _voice_line(score_: Score, voice: str, bars: list, unit: int) -> str:
    texts = [_bar_text(score_, voice, bar, unit) for bar in bars]
    parts = []
    i = 0
    while i < len(texts):
        if not _full_rest(texts[i]):
            parts.append(texts[i] + "|")
            i += 1
            continue
        end = i + 1
        while end < len(texts) and _full_rest(texts[end]):
            end += 1
        parts.append("Z" + (str(end - i) if end - i > 1 else "") + "|")
        i = end
    return "".join(parts)


def _groups(score_: Score) -> list:
    first = score_.bars[0]
    meter = (first.abc_numerator, first.abc_denominator)
    key = score_.keys[first.start_t]
    section = ""
    groups = []
    for bar in score_.bars:
        bar_meter = (bar.abc_numerator, bar.abc_denominator)
        bar_key = score_.keys[bar.start_t]
        meter_changed = bar_meter != meter
        key_changed = bar_key != key
        labels = []
        for t, label in score_.sections:
            if not bar.start_t <= t < bar.end_t:
                continue
            clean = " ".join(str(label).split())
            if clean and clean != section:
                labels.append(clean)
                section = clean
        if not groups or len(groups[-1]["bars"]) >= 4 or meter_changed or key_changed or labels:
            groups.append({"bars": [bar], "labels": labels, "meter_changed": meter_changed,
                           "key_changed": key_changed})
        else:
            groups[-1]["bars"].append(bar)
        meter = bar_meter
        key = score_.keys[bar.end_t - 1]
    return groups


def write(score_: Score) -> str:
    """The score as ABC text in YuE2's two-voice dialect."""
    unit = unit_denominator(score_)
    first = score_.bars[0]
    lines = ["X:1", "T:", "M:{}/{}".format(first.abc_numerator, first.abc_denominator),
             "L:1/{}".format(unit), "Q:1/4={}".format(int(round(tempo(score_)))),
             'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
             'V: Ins clef=treble name="Ins Melody" snm="Inst."',
             "K:{}".format(score_.keys[first.start_t])]
    for group in _groups(score_):
        lines.extend("% " + label for label in group["labels"])
        head = group["bars"][0]
        for voice in VOICES:
            lines.append("V: " + voice)
            if group["meter_changed"]:
                lines.append("M:{}/{}".format(head.abc_numerator, head.abc_denominator))
            if group["key_changed"]:
                lines.append("K:{}".format(score_.keys[head.start_t]))
            lines.append(_voice_line(score_, voice, group["bars"], unit))
    return "\n".join(lines) + "\n"


def build(rows: dict, melody_only: bool = False) -> str:
    """ABC from the rows ``events.score_rows`` returns."""
    return write(score(rows["beats"], rows["chords"], rows["keys"], rows["structures"],
                       rows["notes"], melody_only))
