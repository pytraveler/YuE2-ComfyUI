"""A score as notes, and edited notes written back into the score.

The piano roll in the browser draws notes and the model reads text, so
something has to stand between the two. It lives here rather than in
JavaScript for the reason transpose.py does: upstream's own reader,
vendor/yue2_music/abc_tools.py, decides what every token of the dialect means,
and whatever this module writes is read back with it and compared with the
notes that were asked for before anything is sung.

Only the bars that changed are written again, together with any bar a tie joins
to one of them. Everything else -- the header, the section comments, the other
part, every bar nobody touched -- comes back character for character, so a
score that was opened and closed is still the model's score, and an edit shows
up in a diff as the bars it changed. A bar that is rewritten is spelled the way
the model spells: notes named from the key, a mark only where the key and the
marks already in the bar do not give the pitch, a rest filling the bar written
as Z. One mark more than the model would write is added on purpose: a letter
already altered in the bar gets its mark again in another octave. The dialect
carries an accidental to every octave of its letter, the notation most people
and most renderers read does not, and the extra mark means both read the same.

Measured before this was built (2026-09-14, three rap songs of the same style,
19 renders, vocals separated and pitch-tracked): the model sings its own score
within a semitone on 73-94% of the notes. Four bars with every note changed
were sung as edited in two songs of three and partly in the third, and the rest
of each song stayed on its notes. Reversing the lengths inside bars moved the
rhythm in all three; squeezing a phrase into half its bar with a rest after it
did not, because the voice filled the rest with the words. The same notes with
a mark written on every one were sung as well as the model's own spelling, so
the spelling here buys readable, minimal diffs, not better singing.

The bar grid does not move: notes and chord symbols change inside the bars
that exist, and tempo, meter and key stay as they are. A bar that changes key
halfway through is left to the ABC text.
"""

from __future__ import annotations

import re
from fractions import Fraction

from .vendor.yue2_music import abc_tools

LETTERS = "CDEFGAB"
MARKS = {-2: "__", -1: "_", 0: "=", 1: "^", 2: "^^"}
LENGTHS = tuple(sorted(abc_tools.DURATIONS, reverse=True))
FULL_REST = re.compile(r"\s*Z([2-4])?\s*")
PARTS = {"Vocal": "sung", "Ins": "instrumental"}

LONGEST = 200000
"""Characters of score either function will read. A four-minute song is a few
thousand; the cap only stops a pasted book from tying up a server thread."""

MOST_NOTES = 20000
"""Notes per part an edit may carry, for the same reason."""

UNREADABLE = (
    "This score cannot be read note by note: {reason}.\n\n"
    "The piano roll draws scores written in the dialect the model writes. The ABC "
    "tab still has the text, and the render node sings text the roll cannot draw."
)

NOT_WRITTEN = (
    "The edit could not be written into the score: {reason}.\n\n"
    "Nothing was changed. The score is as it was before the edit."
)

KEY_CHANGE_BAR = (
    "Bar {bar} changes key halfway through, and the piano roll leaves such bars "
    "as they are. Edit that bar in the ABC tab."
)

UNKNOWN_CHORD = (
    "'{name}' is not a chord symbol the score format knows. It knows a root from A "
    "to G with an optional # or b, then one of: major (nothing), m, dim, aug, 7, "
    "maj7, m7, dim7, m7b5, sus4, sus2, 6, m6, 7sus4, m(maj7) -- and an optional "
    "bass note after a slash, as in C/E."
)


def _reason(error) -> str:
    return str(error).strip().rstrip(".")


def _ticks(quarters: Fraction, per_quarter: Fraction) -> int:
    """A time in quarter notes as a whole number of L: units."""
    value = Fraction(quarters) * per_quarter
    if value.denominator != 1:
        raise ValueError("an event falls between the steps of L:")
    return int(value)


def _key_at(keys, time) -> str:
    """The key in force at *time* on a voice's key timeline."""
    current = keys[0][1]
    for start, key in keys:
        if start > time:
            break
        current = key
    return current


def _parsed(text: str):
    """``(text, score, lines, pieces, sections, inline)`` or the unreadable refusal."""
    source = (text or "").strip()
    if not source:
        raise ValueError("There is no score yet. Write one with the plan node first.")
    if len(source) > LONGEST:
        raise ValueError("That score is far longer than any song.")
    try:
        score = abc_tools.parse(source)
        if score.unit.denominator < 4:
            raise ValueError("L:1/{} is coarser than a quarter note".format(
                score.unit.denominator))
        lines, pieces, sections, inline = _layout(source, score)
    except (ValueError, KeyError, IndexError) as error:
        raise ValueError(UNREADABLE.format(reason=_reason(error))) from error
    return source, score, lines, pieces, sections, inline


def _layout(text: str, score):
    """Where every bar of every part sits in the text.

    Returns the lines with their endings, and for each part the pieces of its
    music lines in order -- one piece per barline-separated stretch, which is one
    bar, or up to four when it is a Z2 to Z4 rest -- plus the section each bar
    belongs to and the bars that change key inside themselves.
    """
    lines = text.splitlines(keepends=True)
    pieces = {name: [] for name in abc_tools.VOICES}
    counters = {name: 0 for name in abc_tools.VOICES}
    sections = []
    inline = set()
    section = (-1, "")
    for index, raw in enumerate(lines):
        body = raw.rstrip("\r\n")
        if body.startswith("% "):
            section = (section[0] + 1, body[2:].strip())
            continue
        voice = score.music_lines.get(index)
        if voice is None:
            continue
        for place, piece in enumerate(body[:-1].split("|")):
            rest = FULL_REST.fullmatch(piece)
            count = int(rest.group(1) or 1) if rest else 1
            bars = list(range(counters[voice], counters[voice] + count))
            counters[voice] += count
            pieces[voice].append({"line": index, "place": place, "bars": bars})
            if voice == "Vocal":
                sections.extend([section] * count)
            if "[K:" in piece:
                inline.update(bars)
    return lines, pieces, sections, inline


def read(text: str) -> dict:
    """The score as the piano roll draws it, or a ValueError saying why it cannot be.

    Times and lengths are whole L: units from the start of the song, so a
    quarter note is ``per_quarter`` of them. Each part's notes are sounding
    notes: a tie across a barline is one note, as upstream's reader resolves it.
    """
    source, score, _lines, _pieces, sections, inline = _parsed(text)
    per_quarter = Fraction(score.unit.denominator, 4)
    vocal = score.voices["Vocal"]

    def at(value):
        return _ticks(value, per_quarter)

    bars = []
    for number, (start, length, meter) in enumerate(vocal.bars):
        bars.append({"start": at(start), "length": at(length),
                     "meter": "{}/{}".format(*meter), "key": _key_at(vocal.keys, start),
                     "section": sections[number][1], "editable": number not in inline})
    groups = []
    for number, (identity, name) in enumerate(sections):
        if groups and groups[-1]["identity"] == identity:
            groups[-1]["bars"] += 1
        else:
            groups.append({"identity": identity, "name": name, "bar": number, "bars": 1})
    return {
        "unit": score.unit.denominator,
        "per_quarter": int(per_quarter),
        "bpm": score.bpm,
        "seconds": round(float(vocal.time) * 60.0 / score.bpm, 2),
        "total": at(vocal.time),
        "bars": bars,
        "sections": [{key: group[key] for key in ("name", "bar", "bars")} for group in groups],
        "notes": {name: [{"start": at(start), "length": at(duration), "pitch": int(pitch)}
                         for start, pitch, duration in score.voices[name].notes]
                  for name in abc_tools.VOICES},
        "chords": [{"start": at(start), "name": name} for start, name in vocal.chords],
        "signatures": {key: abc_tools.KEYS[key] for key in sorted({b["key"] for b in bars})},
    }


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _wanted(sheet, bars, total):
    """The notes and chords an edit asks for, checked as untrusted input."""
    if not isinstance(sheet, dict) or not isinstance(sheet.get("notes"), dict):
        raise ValueError("Send the edit as an object with 'notes' for each part.")
    starts = [bar["start"] for bar in bars]

    def bar_of(tick):
        number = 0
        for index, start in enumerate(starts):
            if start <= tick:
                number = index
        return number + 1

    notes = {}
    for name in abc_tools.VOICES:
        items = sheet["notes"].get(name, [])
        if not isinstance(items, list) or len(items) > MOST_NOTES:
            raise ValueError("The {} part must be a list of at most {} notes.".format(
                PARTS[name], MOST_NOTES))
        clean = []
        for item in items:
            if not isinstance(item, dict) or not all(
                    _whole(item.get(field)) for field in ("start", "length", "pitch")):
                raise ValueError("Every note needs a whole-number start, length and pitch.")
            start, length, pitch = item["start"], item["length"], item["pitch"]
            if start < 0 or length < 1 or start + length > total:
                raise ValueError("A {} note runs outside the song, which ends after bar "
                                 "{}.".format(PARTS[name], len(bars)))
            if not 0 <= pitch <= 127:
                raise ValueError("A {} note has pitch {}, and pitches go from 0 to 127."
                                 .format(PARTS[name], pitch))
            clean.append((start, length, pitch))
        clean.sort()
        for before, after in zip(clean, clean[1:]):
            if after[0] < before[0] + before[1]:
                raise ValueError(
                    "Two {} notes overlap in bar {}. Each part sings one note at a time, "
                    "so move or shorten one of them.".format(PARTS[name], bar_of(after[0])))
        notes[name] = clean

    items = sheet.get("chords", [])
    if not isinstance(items, list) or len(items) > MOST_NOTES:
        raise ValueError("The chords must be a list.")
    chords = {}
    for item in items:
        if not isinstance(item, dict) or not _whole(item.get("start")) \
                or not isinstance(item.get("name"), str):
            raise ValueError("Every chord needs a whole-number start and a name.")
        start, name = item["start"], item["name"].strip()
        if not 0 <= start < total:
            raise ValueError("A chord sits outside the song.")
        if abc_tools.CHORD.fullmatch(name) is None:
            raise ValueError(UNKNOWN_CHORD.format(name=name))
        if start in chords:
            raise ValueError("Two chords start at the same moment in bar {}.".format(
                bar_of(start)))
        chords[start] = name
    return notes, sorted(chords.items())


def _inside(items, start, length):
    """The notes of a sorted list that sound at some point of one bar."""
    end = start + length
    return tuple(note for note in items if note[0] < end and note[0] + note[1] > start)


def _dirty(grid, old, new, old_chords, new_chords) -> set:
    """The bars of one part that have to be written again.

    A bar is dirty when the notes sounding in it, or the chords starting in it,
    are not what they were. Then the dirt spreads along ties, old and new: a note
    that crosses from a dirty bar into its neighbour takes the neighbour with it,
    so a tie is always written from both ends by the same hand.
    """
    dirty = set()
    for number, (start, length) in enumerate(grid):
        if _inside(old, start, length) != _inside(new, start, length):
            dirty.add(number)
        elif [c for c in old_chords if start <= c[0] < start + length] != \
                [c for c in new_chords if start <= c[0] < start + length]:
            dirty.add(number)
    waiting = list(dirty)
    while waiting:
        number = waiting.pop()
        start, length = grid[number]
        for items in (old, new):
            for note in _inside(items, start, length):
                for neighbour in (number - 1, number + 1):
                    if 0 <= neighbour < len(grid) and neighbour not in dirty and \
                            _inside((note,), *grid[neighbour]):
                        dirty.add(neighbour)
                        waiting.append(neighbour)
    return dirty


def _spell(pitch: int, key: str):
    """``(letter, alteration, written)`` naming a sounding pitch from *key*.

    A name the key signature already gives wins; after that the fewest
    accidentals. Between two names that are left, a minor key raises -- its
    leading tone and raised sixth are C-sharp and B in D minor, not D-flat and
    C-flat -- and a major key leans the way its signature does: sharps in sharp
    and natural keys, flats in flat ones.
    """
    signature = abc_tools.key_accidentals(key)
    minor = key.endswith("m")
    flats = abc_tools.KEYS[key] < 0
    best = None
    for letter in LETTERS:
        for alteration in (-2, -1, 0, 1, 2):
            written = pitch - alteration
            if (written - abc_tools.NATURAL[letter]) % 12:
                continue
            if minor:
                leaning = alteration < signature[letter]
            else:
                leaning = alteration > 0 if flats else alteration < 0
            cost = (alteration != signature[letter], abs(alteration), leaning)
            if best is None or cost < best[0]:
                best = (cost, letter, alteration, written)
    return best[1], best[2], best[3]


def _pitch_text(letter: str, written: int) -> str:
    """A natural note at MIDI pitch *written*: C is 60, c 72, c' 84, C, 48."""
    octave = (written - abc_tools.NATURAL[letter]) // 12
    if octave >= 6:
        return letter.lower() + "'" * (octave - 6)
    return letter + "," * (5 - octave)


def _lengths(units: int):
    """A length in L: units as the native lengths that add up to it, longest first."""
    out = []
    while units:
        take = next(length for length in LENGTHS if length <= units)
        out.append(take)
        units -= take
    return out


def _digits(units: int) -> str:
    return "" if units == 1 else str(units)


def _bar_text(start, length, notes, chords, key, spelled) -> str:
    """One bar of one part, written from its sounding notes and its chords.

    The bar is cut wherever a note starts or ends or a chord starts, and each
    cut is written with the native lengths that fill it. A note cut by a chord
    or by a length the dialect cannot write in one token is tied across the cut,
    and a note running into the next bar ends the bar with a tie. Only a note's
    first token can carry a mark: a tied continuation keeps its pitch by the
    dialect's own rule.
    """
    end = start + length
    if not notes and not chords:
        return "Z"
    edges = {start, end}
    for note_start, note_length, _pitch in notes:
        edges.add(max(note_start, start))
        edges.add(min(note_start + note_length, end))
    edges.update(chords)
    edges = sorted(edge for edge in edges if start <= edge <= end)
    signature = abc_tools.key_accidentals(key)
    state = dict(signature)
    marked = {}
    out = []
    for left, right in zip(edges, edges[1:]):
        if left in chords:
            out.append('"' + chords[left] + '"')
        note = next((n for n in notes if n[0] <= left < n[0] + n[1]), None)
        pieces = _lengths(right - left)
        if note is None:
            out.extend("z" + _digits(units) for units in pieces)
            continue
        letter, alteration, written = spelled[note]
        octave = (written - abc_tools.NATURAL[letter]) // 12
        onward = note[0] + note[1] > right
        for index, units in enumerate(pieces):
            mark = ""
            if left == note[0] and index == 0:
                altered = alteration != signature[letter]
                if state[letter] != alteration or (
                        altered and octave not in marked.get(letter, ())):
                    if state[letter] != alteration:
                        marked[letter] = {octave}
                    else:
                        marked.setdefault(letter, set()).add(octave)
                    state[letter] = alteration
                    mark = MARKS[alteration]
            tied = index < len(pieces) - 1 or onward
            out.append(mark + _pitch_text(letter, written) + _digits(units)
                       + ("-" if tied else ""))
    return "".join(out)


def _check(text, notes, chords, score, per_quarter) -> None:
    """The written score read back: the asked-for notes and chords, the old grid."""
    result = abc_tools.parse(text)

    def at(value):
        return _ticks(value, per_quarter)

    for name in abc_tools.VOICES:
        got = sorted((at(start), at(duration), int(pitch))
                     for start, pitch, duration in result.voices[name].notes)
        if got != notes[name]:
            raise ValueError("the {} part does not read back as the notes asked for"
                             .format(PARTS[name]))
        if result.voices[name].bars != score.voices[name].bars:
            raise ValueError("the bars of the {} part moved".format(PARTS[name]))
        if result.voices[name].keys != score.voices[name].keys:
            raise ValueError("a key change of the {} part moved".format(PARTS[name]))
    if [(at(start), name) for start, name in result.voices["Vocal"].chords] != chords:
        raise ValueError("the chords do not read back as the chords asked for")
    if result.bpm != score.bpm or result.unit != score.unit:
        raise ValueError("the tempo or the note length changed")


def write(text: str, sheet) -> dict:
    """``{"abc": text, "bars": [...]}``: *text* with the notes of *sheet* in it.

    *sheet* is what :func:`read` returned, edited: ``notes`` per part and
    ``chords``; everything else in it is ignored, and the bar grid comes from
    the text, not from the sheet. ``bars`` lists the bars written again,
    counting from 0; an edit that changes nothing returns the text untouched and
    an empty list. A ValueError carries a message for the person editing.
    """
    source, score, lines, pieces, _sections, inline = _parsed(text)
    per_quarter = Fraction(score.unit.denominator, 4)
    base = read(source)
    grid = [(bar["start"], bar["length"]) for bar in base["bars"]]
    notes, chords = _wanted(sheet, base["bars"], base["total"])
    old = {name: sorted((n["start"], n["length"], n["pitch"]) for n in base["notes"][name])
           for name in abc_tools.VOICES}
    old_chords = [(c["start"], c["name"]) for c in base["chords"]]
    dirty = {"Vocal": _dirty(grid, old["Vocal"], notes["Vocal"], old_chords, chords),
             "Ins": _dirty(grid, old["Ins"], notes["Ins"], [], [])}
    touched = sorted(dirty["Vocal"] | dirty["Ins"])
    if not touched:
        return {"abc": source, "bars": []}
    locked = [number for number in touched if number in inline]
    if locked:
        raise ValueError(KEY_CHANGE_BAR.format(bar=locked[0] + 1))

    chord_at = dict(chords)
    replaced = {}
    for name in abc_tools.VOICES:
        voice = score.voices[name]
        spelled = {note: _spell(note[2], _key_at(voice.keys, Fraction(note[0]) / per_quarter))
                   for note in notes[name]}
        for piece in pieces[name]:
            if not dirty[name].intersection(piece["bars"]):
                continue
            written = []
            for number in piece["bars"]:
                if number not in dirty[name]:
                    written.append("Z")
                    continue
                start, length = grid[number]
                in_bar = {tick: chord for tick, chord in chord_at.items()
                          if start <= tick < start + length} if name == "Vocal" else {}
                written.append(_bar_text(start, length, _inside(notes[name], start, length),
                                         in_bar, base["bars"][number]["key"], spelled))
            replaced.setdefault(piece["line"], {})[piece["place"]] = "|".join(written)

    for index, places in replaced.items():
        raw = lines[index]
        body = raw.rstrip("\r\n")
        parts = body[:-1].split("|")
        for place, piece in places.items():
            parts[place] = piece
        lines[index] = "|".join(parts) + "|" + raw[len(body):]
    result = "".join(lines)
    try:
        _check(result, notes, chords, score, per_quarter)
    except (ValueError, KeyError, IndexError) as error:
        raise ValueError(NOT_WRITTEN.format(reason=_reason(error))) from error
    return {"abc": result, "bars": touched}
