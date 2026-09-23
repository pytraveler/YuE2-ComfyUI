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
that exist, and the meter and the key stay as they are. The tempo is the one
header the editor may set, because it is one number that moves nothing else:
the notes keep their lengths in bars, and the song is sung faster or slower. A
bar that changes key halfway through is left to the ABC text. The one exception
to the grid is ``without``, which takes whole bars out of both parts for a cut
of the song, and writes again only the lines that lose bars.

``changed`` and ``taken`` compare two scores of one song for the track editor,
which sings again only the bars whose notes changed: the first says which bars
those are, and the second lays one stretch of a change on the song by itself,
so that bars changed far apart can be sung as separate edits.
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

TEMPO_LINE = 4
"""Where ``Q:1/4=<BPM>`` sits.

The dialect fixes the order of the header and abc_tools.parse refuses a score
whose fifth line is not that tempo, so the line can be replaced without reading
the rest of the text.
"""

TEMPO_LOW = 40
TEMPO_HIGH = 200
"""The tempi an editor offers.

A score that came in outside them -- a transcription of something fast, a MIDI
file at 210 -- keeps its own as the far end of the range, so opening an editor
and closing it again cannot move the song.
"""

TEMPO_RANGE = "A tempo of {value} BPM is outside {low} to {high}."

HEADER_LINES = 8
"""Lines of header before the first group of music.

The dialect fixes them -- X:1, an empty title, the meter, the unit, the tempo,
the two voice definitions and the key -- and abc_tools.parse refuses a score
whose first eight lines are anything else, so a walk over the groups can begin
here without looking for where they start.
"""

SECTION_LONGEST = 40
"""Characters a section name may have. The model's own labels are a word or two."""

SECTION_NAME = (
    "'{name}' cannot be a section name: a name is a line of text of at most "
    "{longest} characters."
)

SECTION_TWICE = "Two sections start at bar {bar}, and every bar belongs to one section."

SECTION_BAR = "A section starts at bar {bar}, and this song has {bars} bars."

UNIT_LINE = 3
"""Where ``L:1/<n>`` sits, fixed by the dialect like the tempo line above."""

FINEST = 32
"""The shortest note length a score is rewritten at.

A grid step of the piano roll is one L: unit, and the dialect writes a length as
a whole number of them, so a thirty-second note exists in the roll only if the
score is written on thirty-seconds. The roll offers no finer grid, and a score
written finer would only be longer to read.
"""

UNIT_ASKED = (
    "A score can be rewritten on a note length of 1/{finest} at the most, and only "
    "on a power of two; 1/{unit} is not one."
)

GROUP_BARS = 4
"""Bars a group of music may hold, the most the dialect allows in one line."""

BLANK_BARS = 16
BLANK_BPM = 120
BLANK_METER = "4/4"
BLANK_UNIT = 16
BLANK_KEY = "C"
BLANK_SECTION = "verse"
"""What a score made from nothing starts as.

Sixteen bars of four four at 120 BPM is half a minute, enough of a grid to
draw a verse on and short enough to read at a glance. The unit is the one the
model itself writes for four four, and the roll offers thirty-seconds on top of
it when a finer grid is wanted.
"""

MOST_BARS = 2000
"""Bars either function will make.

An hour of music at four four, far past anything the model sings. The cap is
there so a mistyped number cannot build a score nothing can read, not to say
anything about how long a song may be.
"""

BARS_ASKED = "A score can be at most {most} bars long, and {bars} were asked for."

BARS_FEWER = (
    "This score is {have} bars long and {bars} were asked for. The editor only makes "
    "a song longer: bars are removed by deleting them, so that nothing is thrown away "
    "by a number typed in the wrong box."
)

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

NOT_CUT = (
    "The bars could not be cut out of the score: {reason}.\n\n"
    "Nothing was changed. The score is as it was before the cut."
)

KEY_CHANGE_CUT = (
    "Bar {bar} changes key halfway through, and cutting it out would move the key "
    "of everything after it. Cut up to that bar, or from the bar after it."
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


def _bar_count(body: str) -> int:
    """Bars in one music line, a Z2 to Z4 rest counting as the bars it stands for."""
    count = 0
    for piece in body[:-1].split("|"):
        rest = FULL_REST.fullmatch(piece)
        count += int(rest.group(1) or 1) if rest else 1
    return count


def _rest_text(bars: int) -> str:
    return "Z" + (str(bars) if bars > 1 else "")


def _cut_music(body: str, at: int) -> tuple:
    """One music line as two, the second beginning *at* bars into it.

    Only a whole-bar rest stands for more than one bar, so a cut that falls
    inside a piece falls inside such a rest and is written as two shorter ones.
    """
    head, tail, seen = [], [], 0
    for piece in body[:-1].split("|"):
        rest = FULL_REST.fullmatch(piece)
        bars = int(rest.group(1) or 1) if rest else 1
        if seen + bars <= at:
            head.append(piece)
        elif seen >= at:
            tail.append(piece)
        else:
            head.append(_rest_text(at - seen))
            tail.append(_rest_text(bars - (at - seen)))
        seen += bars
    return "|".join(head) + "|", "|".join(tail) + "|"


def _blocks(bodies: list) -> list:
    """The groups of a score: the names written above each, and each part's lines.

    The shape is the one upstream's reader demands -- any number of ``% name``
    comments, then for each part its ``V:`` line, a meter or key field if the
    group changes one, and one music line -- so a score that parsed needs no
    checking here.
    """
    found = []
    cursor = HEADER_LINES
    while cursor < len(bodies):
        names = []
        while bodies[cursor].startswith("% "):
            names.append(bodies[cursor][2:].strip())
            cursor += 1
        block = {"names": names, "voices": {}}
        for name in abc_tools.VOICES:
            head = [bodies[cursor]]
            cursor += 1
            while bodies[cursor].startswith(("M:", "K:")):
                head.append(bodies[cursor])
                cursor += 1
            block["voices"][name] = {"head": head, "music": bodies[cursor]}
            cursor += 1
        found.append(block)
    return found


def _split_block(block: dict, at: int) -> list:
    """A group cut in two on the same bar of both parts.

    A meter or key the group carries stays with its first half, where the bar it
    belongs to still is; the second half needs none, because a part keeps what
    it was given until something changes it.
    """
    first = {"names": block["names"], "voices": {}}
    second = {"names": [], "voices": {}}
    for name in abc_tools.VOICES:
        voice = block["voices"][name]
        head, tail = _cut_music(voice["music"], at)
        first["voices"][name] = {"head": voice["head"], "music": head}
        second["voices"][name] = {"head": ["V: " + name], "music": tail}
    return [first, second]


def _resection(text: str, wanted: list) -> str:
    """*text* with its section comments replaced by *wanted*, ``(bar, name)`` each.

    A comment can only stand between groups, and a group is one to four bars, so
    a section that begins inside one cuts it in two. Nothing else moves: every
    bar keeps the characters it had, and the caller reads the notes back to prove
    it.
    """
    lines = text.splitlines(keepends=True)
    bodies = [line.rstrip("\r\n") for line in lines]
    ending = lines[0][len(bodies[0]):] or "\n"
    blocks = _blocks(bodies)
    starts = []
    at = 0
    for block in blocks:
        starts.append(at)
        at += _bar_count(block["voices"]["Vocal"]["music"])
    for bar, _name in wanted:
        if bar in starts:
            continue
        index = max(place for place, start in enumerate(starts) if start < bar)
        blocks[index:index + 1] = _split_block(blocks[index], bar - starts[index])
        starts.insert(index + 1, bar)
    named = dict(wanted)
    out = list(lines[:HEADER_LINES])
    for start, block in zip(starts, blocks):
        if start in named:
            out.append("% " + named[start] + ending)
        for name in abc_tools.VOICES:
            out.extend(line + ending for line in block["voices"][name]["head"])
            out.append(block["voices"][name]["music"] + ending)
    if not lines[-1].endswith(("\n", "\r")):
        out[-1] = out[-1].rstrip("\r\n")
    return "".join(out)


def _squeezed(texts: list) -> str:
    """Bars as a music line, a run of whole-bar rests written as one Z.

    The dialect allows one to four bars in a line, so the run is cut into fours,
    which is also how the model writes a long silence.
    """
    out = []
    at = 0
    while at < len(texts):
        if texts[at] != "Z":
            out.append(texts[at])
            at += 1
            continue
        end = at
        while end < len(texts) and texts[end] == "Z" and end - at < 4:
            end += 1
        out.append(_rest_text(end - at))
        at = end
    return "|".join(out) + "|"


def _refine(text: str, unit: int) -> str:
    """*text* written again on ``L:1/unit``, finer than the one it came in.

    The song is not touched: a time and a length are the same number of quarter
    notes either way, and the caller reads both texts back and compares them.
    What changes is what the piano roll can hold, because its grid step is one
    unit and the dialect has no fractional lengths.
    """
    source, score, lines, _pieces, _sections, _inline = _parsed(text)
    sheet = read(source)
    scale = unit // sheet["unit"]
    per_quarter = Fraction(unit, 4)
    bodies = [line.rstrip("\r\n") for line in lines]
    ending = lines[0][len(bodies[0]):] or "\n"
    grid = [(bar["start"] * scale, bar["length"] * scale) for bar in sheet["bars"]]
    keys = [bar["key"] for bar in sheet["bars"]]
    notes = {name: sorted((note["start"] * scale, note["length"] * scale, note["pitch"])
                          for note in sheet["notes"][name]) for name in abc_tools.VOICES}
    chords = {chord["start"] * scale: chord["name"] for chord in sheet["chords"]}
    out = list(lines[:HEADER_LINES])
    out[UNIT_LINE] = "L:1/{}".format(unit) + lines[UNIT_LINE][len(bodies[UNIT_LINE]):]
    number = 0
    for block in _blocks(bodies):
        out.extend("% " + name + ending for name in block["names"])
        first = number
        for name in abc_tools.VOICES:
            number = first
            out.extend(line + ending for line in block["voices"][name]["head"])
            written = []
            for _ in range(_bar_count(block["voices"][name]["music"])):
                start, length = grid[number]
                inside = _inside(notes[name], start, length)
                spelled = {note: _spell(note[2], _key_at(score.voices[name].keys,
                                                         Fraction(note[0], per_quarter)))
                           for note in inside}
                in_bar = {tick: chord for tick, chord in chords.items()
                          if start <= tick < start + length} if name == "Vocal" else {}
                written.append(_bar_text(start, length, inside, in_bar, keys[number], spelled))
                number += 1
            out.append(_squeezed(written) + ending)
    if not lines[-1].endswith(("\n", "\r")):
        out[-1] = out[-1].rstrip("\r\n")
    return "".join(out)


def _same_music(before, after) -> None:
    """Two parses of the same song, however each is written down."""
    for name in abc_tools.VOICES:
        if [list(note) for note in before.voices[name].notes] != \
                [list(note) for note in after.voices[name].notes]:
            raise ValueError("the {} part does not read back as the same notes"
                             .format(PARTS[name]))
        if before.voices[name].bars != after.voices[name].bars:
            raise ValueError("the bars of the {} part moved".format(PARTS[name]))
        if before.voices[name].keys != after.voices[name].keys:
            raise ValueError("a key change of the {} part moved".format(PARTS[name]))
    if before.voices["Vocal"].chords != after.voices["Vocal"].chords:
        raise ValueError("the chords moved")
    if before.bpm != after.bpm:
        raise ValueError("the tempo changed")


def _wanted_unit(sheet, current: int):
    """The note length an edit asks the score to be rewritten on, or None.

    None is the ordinary answer: an edit sends the length its score already has,
    or sends none at all. A coarser one is ignored rather than refused, because
    the roll asks for a finer grid and never for a wider one.
    """
    if not isinstance(sheet, dict) or sheet.get("unit") is None:
        return None
    unit = sheet["unit"]
    if not _whole(unit) or unit <= current:
        return None
    if unit > FINEST or unit & (unit - 1):
        raise ValueError(UNIT_ASKED.format(unit=unit, finest=FINEST))
    return unit


def _wanted_sections(sheet, bars: int):
    """The sections an edit asks for, checked as untrusted input.

    None when the edit carries none, which is what every caller sent before
    sections could be edited and still sends when it only moves notes. A name
    that is empty is not a refusal but a stretch with no comment above it, which
    is how :func:`read` reports bars before the first one.
    """
    if not isinstance(sheet, dict) or sheet.get("sections") is None:
        return None
    items = sheet["sections"]
    if not isinstance(items, list) or len(items) > bars:
        raise ValueError("The sections must be a list, at most one for each bar.")
    wanted = []
    seen = set()
    for item in items:
        if not isinstance(item, dict) or not _whole(item.get("bar")) \
                or not isinstance(item.get("name"), str):
            raise ValueError("Every section needs a whole-number bar and a name.")
        name = " ".join(item["name"].split())
        if not name:
            continue
        if len(name) > SECTION_LONGEST or not name.isprintable():
            raise ValueError(SECTION_NAME.format(name=item["name"][:60].strip(),
                                                 longest=SECTION_LONGEST))
        if not 0 <= item["bar"] < bars:
            raise ValueError(SECTION_BAR.format(bar=item["bar"] + 1, bars=bars))
        if item["bar"] in seen:
            raise ValueError(SECTION_TWICE.format(bar=item["bar"] + 1))
        seen.add(item["bar"])
        wanted.append((item["bar"], name))
    wanted.sort()
    return wanted


def _rest_groups(bars: int, ending: str) -> list:
    """Lines for *bars* empty bars, cut into groups the dialect allows."""
    out = []
    left = bars
    while left > 0:
        count = min(GROUP_BARS, left)
        music = _rest_text(count) + "|"
        for name in abc_tools.VOICES:
            out.append("V: " + name + ending)
            out.append(music + ending)
        left -= count
    return out


def blank(bars: int = BLANK_BARS, bpm: int = BLANK_BPM) -> str:
    """A score of *bars* empty bars, for writing one from nothing.

    The header is the dialect's fixed eight lines and the music is whole-bar
    rests in both parts, so the piano roll opens on an empty grid of the right
    length instead of on a page telling the person to run something first.
    """
    _asked(bars)
    if not isinstance(bpm, int) or isinstance(bpm, bool) or not TEMPO_LOW <= bpm <= TEMPO_HIGH:
        raise ValueError(TEMPO_RANGE.format(value=bpm, low=TEMPO_LOW, high=TEMPO_HIGH))
    head = ["X:1", "T:", "M:" + BLANK_METER, "L:1/{}".format(BLANK_UNIT),
            "Q:1/4={}".format(bpm),
            'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
            'V: Ins clef=treble name="Ins Melody" snm="Inst."',
            "K:" + BLANK_KEY, "% " + BLANK_SECTION]
    return "".join(line + "\n" for line in head) + "".join(_rest_groups(bars, "\n"))


def lengthened(text: str, bars: int) -> str:
    """*text* with empty bars added at the end until the song is *bars* long.

    Nothing already written moves. The added groups carry only a ``V:`` line
    each, because a part keeps the meter and the key it was last given, and the
    song's last section runs on into them.
    """
    _asked(bars)
    source, score, _lines, _pieces, _sections, _inline = _parsed(text)
    have = len(score.voices["Vocal"].bars)
    if bars <= have:
        raise ValueError(BARS_FEWER.format(have=have, bars=bars))
    lines = source.splitlines(keepends=True)
    ending = lines[0][len(lines[0].rstrip("\r\n")):] or "\n"
    out = [line if line.endswith(("\n", "\r")) else line + ending for line in lines]
    out.extend(_rest_groups(bars - have, ending))
    return "".join(out)


def _asked(bars) -> None:
    """A bar count the editor may ask for, or the refusal saying what the limit is."""
    if not isinstance(bars, int) or isinstance(bars, bool) or not 1 <= bars <= MOST_BARS:
        raise ValueError(BARS_ASKED.format(most=MOST_BARS, bars=bars))


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


def _tempo(sheet, current: int) -> int:
    """The tempo the sheet asks for, as a whole quarter-note BPM.

    A sheet without one keeps the score's own, so a caller with no tempo to
    offer goes on sending what it always sent.
    """
    asked = sheet.get("bpm") if isinstance(sheet, dict) else None
    if asked is None:
        return current
    if isinstance(asked, bool) or not isinstance(asked, (int, float)):
        raise ValueError("the tempo must be a number")
    number = float(asked)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError("the tempo must be a number")
    value = int(round(number))
    low, high = min(TEMPO_LOW, current), max(TEMPO_HIGH, current)
    if not low <= value <= high:
        raise ValueError(TEMPO_RANGE.format(value=value, low=low, high=high))
    return value


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


def _check(text, notes, chords, score, per_quarter, bpm) -> None:
    """The written score read back: the asked-for notes, chords and tempo, the old grid."""
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
    if result.bpm != bpm or result.unit != score.unit:
        raise ValueError("the tempo or the note length is not the one asked for")


def _checked(text, notes, chords, score, per_quarter, bpm) -> None:
    """:func:`_check`, with the refusal a person editing can read.

    Every path that hands text back runs it, so a bar this module moved by
    mistake is caught here rather than sung.
    """
    try:
        _check(text, notes, chords, score, per_quarter, bpm)
    except (ValueError, KeyError, IndexError) as error:
        raise ValueError(NOT_WRITTEN.format(reason=_reason(error))) from error


def write(text: str, sheet) -> dict:
    """``{"abc": text, "bars": [...]}``: *text* with the notes of *sheet* in it.

    *sheet* is what :func:`read` returned, edited: ``notes`` per part,
    ``chords``, ``bpm`` when the tempo is to change, and ``sections`` when the
    names above the bars are to move; everything else in it is ignored, and the
    bar grid comes from the text, not from the sheet. ``bars`` lists the bars
    written again, counting from 0 -- a tempo or a section answers with an empty
    list, because neither rewrites a bar. An edit that changes nothing returns
    the text untouched. A ValueError carries a message for the person editing.
    """
    source, score, lines, pieces, _sections, inline = _parsed(text)
    as_it_came = source
    unit = _wanted_unit(sheet, score.unit.denominator)
    if unit is not None:
        came_as = score
        source = _refine(source, unit)
        _text, score, lines, pieces, _sections, inline = _parsed(source)
        try:
            _same_music(came_as, score)
        except (ValueError, KeyError, IndexError) as error:
            raise ValueError(NOT_WRITTEN.format(reason=_reason(error))) from error
    per_quarter = Fraction(score.unit.denominator, 4)
    base = read(source)
    wanted = _wanted_sections(sheet, len(base["bars"]))
    if wanted is not None and wanted != [(group["bar"], group["name"])
                                         for group in base["sections"] if group["name"]]:
        source = _resection(source, wanted)
        _text, _score, lines, pieces, _sections, inline = _parsed(source)
    bpm = _tempo(sheet, score.bpm)
    if bpm != score.bpm:
        raw = lines[TEMPO_LINE]
        body = raw.rstrip("\r\n")
        lines[TEMPO_LINE] = "Q:1/4={}".format(bpm) + raw[len(body):]
    grid = [(bar["start"], bar["length"]) for bar in base["bars"]]
    notes, chords = _wanted(sheet, base["bars"], base["total"])
    old = {name: sorted((n["start"], n["length"], n["pitch"]) for n in base["notes"][name])
           for name in abc_tools.VOICES}
    old_chords = [(c["start"], c["name"]) for c in base["chords"]]
    dirty = {"Vocal": _dirty(grid, old["Vocal"], notes["Vocal"], old_chords, chords),
             "Ins": _dirty(grid, old["Ins"], notes["Ins"], [], [])}
    touched = sorted(dirty["Vocal"] | dirty["Ins"])
    if not touched and bpm == score.bpm:
        if source is not as_it_came:
            _checked(source, notes, chords, score, per_quarter, bpm)
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
    _checked(result, notes, chords, score, per_quarter, bpm)
    return {"abc": result, "bars": touched}


def _rests(bars: int) -> list:
    """``bars`` whole-bar rests as the model writes them: Z4 for every four, then Z to Z3 for the rest."""
    found = ["Z4"] * (bars // 4)
    left = bars % 4
    if left:
        found.append("Z" if left == 1 else "Z{}".format(left))
    return found


def _folded(pieces) -> list:
    """The pieces of one line with neighbouring whole-bar rests folded together, as the model writes them."""
    found = []
    run = 0
    for piece in pieces:
        rest = FULL_REST.fullmatch(piece)
        if rest:
            run += int(rest.group(1) or 1)
            continue
        found.extend(_rests(run))
        run = 0
        found.append(piece)
    found.extend(_rests(run))
    return found


def _declared(lines, index: int) -> list:
    """The lines that open the music line at *index*: its V: line and any M: or K: lines under it."""
    found = []
    cursor = index - 1
    while cursor >= 0 and lines[cursor].startswith(("M:", "K:")):
        found.append(cursor)
        cursor -= 1
    if cursor < 0 or not lines[cursor].startswith("V: "):
        raise ValueError("a line of music has no V: line above it")
    return found + [cursor]


def _cut_back(before, after, start: int, stop: int) -> None:
    """The cut score read back: every bar left has the length, meter and key it had."""
    removed = stop - start
    for name in abc_tools.VOICES:
        old, new = before.voices[name], after.voices[name]
        if len(new.bars) != len(old.bars) - removed:
            raise ValueError("the {} part has {} bars where {} should be left".format(
                PARTS[name], len(new.bars), len(old.bars) - removed))
        kept = [bar for number, bar in enumerate(old.bars) if not start <= number < stop]
        for number, (was, now) in enumerate(zip(kept, new.bars), 1):
            if tuple(was[1:]) != tuple(now[1:]):
                raise ValueError("bar {} of the {} part changed its length".format(number, PARTS[name]))
            if _key_at(old.keys, was[0]) != _key_at(new.keys, now[0]):
                raise ValueError("bar {} of the {} part changed its key".format(number, PARTS[name]))
    if after.bpm != before.bpm or after.unit != before.unit:
        raise ValueError("the tempo or the note length changed")


def without(text: str, start: int, stop: int) -> str:
    """*text* with bars *start* to *stop* taken out of both parts: what a cut of the song does to its score.

    Bars count from 0 and *stop* is not included. A line that keeps all its bars
    comes back character for character, and so does everything around the
    score, its final newline included. A line that loses bars is written again
    from the ones it keeps, with neighbouring whole-bar rests folded into Z2 to
    Z4 the way the model writes them; a line left with none goes, together with
    the V: line above it, and a section comment with no music left under it goes
    too. A tie from the last bar before the cut is taken off in both parts: the
    note it held on into is gone.

    The result is read back before it is returned: every bar left must have the
    length, meter and key it had. A cut that would take a key or meter change
    with it is refused by that check rather than sung in the wrong key.
    """
    source, score, lines, pieces, _sections, inline = _parsed(text)
    count = len(score.voices["Vocal"].bars)
    if not (_whole(start) and _whole(stop) and 0 <= start < stop <= count):
        raise ValueError("Bars {} to {} are not bars of this score, which has {}.".format(
            start + 1 if _whole(start) else start, stop, count))
    if stop - start >= count:
        raise ValueError("That is every bar of the score, and a cut has to leave some.")
    locked = [number for number in range(start, stop) if number in inline]
    if locked:
        raise ValueError(KEY_CHANGE_CUT.format(bar=locked[0] + 1))

    lines = list(lines)
    gone = set()
    try:
        for name in abc_tools.VOICES:
            by_line = {}
            for piece in pieces[name]:
                by_line.setdefault(piece["line"], []).append(piece)
            for index, found in by_line.items():
                body = lines[index].rstrip("\r\n")
                texts = body[:-1].split("|")
                kept, changed = [], False
                for piece in sorted(found, key=lambda entry: entry["place"]):
                    left = [bar for bar in piece["bars"] if not start <= bar < stop]
                    written = texts[piece["place"]]
                    if len(left) < len(piece["bars"]):
                        changed = True
                        if not left:
                            continue
                        written = _rests(len(left))[0]
                    elif piece["bars"] == [start - 1] and written.rstrip().endswith("-"):
                        changed = True
                        written = written.rstrip()[:-1]
                    kept.append(written)
                if not changed:
                    continue
                if not kept:
                    gone.add(index)
                    gone.update(_declared(lines, index))
                    continue
                lines[index] = "|".join(_folded(kept)) + "|" + lines[index][len(body):]

        for index, line in enumerate(lines):
            if not line.startswith("% "):
                continue
            under = []
            for after in range(index + 1, len(lines)):
                if lines[after].startswith("% "):
                    if under:
                        break
                    continue
                if after in score.music_lines:
                    under.append(after)
            if under and all(after in gone for after in under):
                gone.add(index)

        result = "".join(line for index, line in enumerate(lines) if index not in gone)
        _cut_back(score, abc_tools.parse(result.strip()), start, stop)
    except (ValueError, KeyError, IndexError) as error:
        raise ValueError(NOT_CUT.format(reason=_reason(error))) from error
    raw = str(text or "")
    leading = raw[:len(raw) - len(raw.lstrip())]
    trailing = raw[len(raw.rstrip()):]
    return leading + result.strip() + trailing


NOT_HEAD = (
    "The score could not be ended after bar {stop}: {reason}.\n\n"
    "Nothing was changed. The score is as it was."
)


def head(text: str, stop: int, merged=()) -> str:
    """The first *stop* bars of *text*, for the model to write the rest of the score after them.

    This is how a song is made to go on past its last words: the model writes
    its score before a note of audio, so a score that ends at bar *stop* is
    where it picks up the pen, and whatever it writes next is sung from there.
    The bars kept come back character for character, cut into their own
    groups the way ``without`` cuts them, with every line ending in a newline
    so the model starts on a line of its own. A tie from the last bar kept is
    taken off in both parts, since the note it held on into is not written
    yet. *merged* names bars whose section comments are left out, so those
    bars run on in the section before them: the tail of a last chorus that the
    model marked as its outro, say, which would otherwise have the model write
    the rest as an outro.

    The result is read back before it is returned: it must be *stop* bars,
    each with the length, meter and key it had.
    """
    source, score, lines, _pieces, _sections, _inline = _parsed(text)
    count = len(score.voices["Vocal"].bars)
    if not (_whole(stop) and 0 < stop <= count):
        raise ValueError("The score cannot end after bar {}: it has {} bars.".format(stop, count))
    bodies = [line.rstrip("\r\n") for line in lines]
    ending = lines[0][len(bodies[0]):] or "\n"
    out = [body + ending for body in bodies[:HEADER_LINES]]
    at = 0
    try:
        for block in _blocks(bodies):
            if at >= stop:
                break
            bars = _bar_count(block["voices"]["Vocal"]["music"])
            if at + bars > stop:
                block = _split_block(block, stop - at)[0]
            last = at + bars >= stop
            if at not in merged:
                out.extend("% " + name + ending for name in block["names"])
            for name in abc_tools.VOICES:
                voice = block["voices"][name]
                music = voice["music"]
                if last and music.rstrip().endswith("-|"):
                    music = music.rstrip()[:-2] + "|"
                out.extend(line + ending for line in voice["head"])
                out.append(music + ending)
            at += bars
        result = "".join(out)
        _cut_back(score, abc_tools.parse(result.strip()), stop, count)
    except (ValueError, KeyError, IndexError) as error:
        raise ValueError(NOT_HEAD.format(stop=stop, reason=_reason(error))) from error
    return result


NOT_MOVED = (
    "The bars could not be moved in the score: {reason}.\n\n"
    "Nothing was changed. The score is as it was before the move."
)

KEY_CHANGE_MOVE = (
    "Bar {bar} changes key halfway through, and moving the bars around it would carry that "
    "change somewhere else. Move the sections on either side of it instead."
)


def order(first: int, stop: int, to: int, count: int) -> list:
    """The stretches ``(first bar, stop)`` a score of *count* bars is read in once bars *first* to *stop* go before bar *to*.

    Four stretches at most -- what stays before, what is moved, what it jumps
    over, what stays after -- with the empty ones left out. Bars count from
    0 and *stop* is not included; *to* is the bar line the moved bars go
    before, *count* for after the last bar.
    """
    for value in (first, stop, to, count):
        if not _whole(value):
            raise ValueError("A move is counted in whole bars.")
    if not 0 <= first < stop <= count:
        raise ValueError("Bars {} to {} are not bars of this score, which has {}.".format(
            first + 1, stop, count))
    if not 0 <= to <= count:
        raise ValueError("There is no bar line {} to move bars to in a score of {} bars.".format(
            to + 1, count))
    if first <= to <= stop:
        raise ValueError("Bars {} to {} would be moved to where they already are.".format(
            first + 1, stop))
    if to < first:
        found = [(0, to), (first, stop), (to, first), (stop, count)]
    else:
        found = [(0, first), (stop, to), (first, stop), (to, count)]
    return [(low, high) for low, high in found if high > low]


def _moved_back(before, after, stretches) -> None:
    """The moved score read back: every bar has the length, meter and key it had where it came from."""
    taken = [number for low, high in stretches for number in range(low, high)]
    if len(after["bars"]) != len(taken):
        raise ValueError("the score has {} bars where {} should be".format(
            len(after["bars"]), len(taken)))
    for number, (bar, was) in enumerate(zip(after["bars"], taken), 1):
        old = before["bars"][was]
        if (bar["length"], bar["meter"]) != (old["length"], old["meter"]):
            raise ValueError("bar {} changed its length".format(number))
        if bar["key"] != old["key"]:
            raise ValueError("bar {} changed its key".format(number))
    if after["bpm"] != before["bpm"] or after["unit"] != before["unit"]:
        raise ValueError("the tempo or the note length changed")


def moved(text: str, first: int, stop: int, to: int) -> str:
    """*text* with bars *first* to *stop* put before bar *to*: what moving part of a song does to its score.

    Bars count from 0 and *stop* is not included; *to* is the bar line the
    bars go before, the score's bar count for after its last bar. Every bar
    keeps its characters, cut into its own groups the way ``without`` cuts
    them, and a section comment goes with the bars under it. A stretch that
    starts in the middle of a section is named after that section again, so
    its bars are not read as the tail of whatever now comes before them;
    and a stretch that lands where another meter or key is in force carries
    its own at its head. A tie from the last bar of any stretch is taken off
    in both parts, since the note it held on into is somewhere else now, and
    what it held into is struck again. Everything around the score is kept,
    its final newline included.

    The result is read back before it is returned: every bar must have the
    length, meter and key it had. A bar that changes key halfway through
    cannot be moved past, see ``KEY_CHANGE_MOVE``.
    """
    source, score, lines, _pieces, sections, inline = _parsed(text)
    count = len(score.voices["Vocal"].bars)
    stretches = order(first, stop, to, count)
    low, high = min(first, to), max(stop, to)
    locked = sorted(number for number in inline if low <= number < high)
    if locked:
        raise ValueError(KEY_CHANGE_MOVE.format(bar=locked[0] + 1))
    bodies = [line.rstrip("\r\n") for line in lines]
    ending = lines[0][len(bodies[0]):] or "\n"
    try:
        blocks = _blocks(bodies)
        starts = []
        at = 0
        for block in blocks:
            starts.append(at)
            at += _bar_count(block["voices"]["Vocal"]["music"])
        for bar in sorted({first, stop, to}):
            if 0 < bar < count and bar not in starts:
                index = max(place for place, start in enumerate(starts) if start < bar)
                blocks[index:index + 1] = _split_block(blocks[index], bar - starts[index])
                starts.insert(index + 1, bar)
        sizes = [_bar_count(block["voices"]["Vocal"]["music"]) for block in blocks]
        state = {name: ("M:" + bodies[2][2:], "K:" + bodies[7][2:]) for name in abc_tools.VOICES}
        out = [body + ending for body in bodies[:HEADER_LINES]]
        for low, high in stretches:
            chosen = [index for index, start in enumerate(starts)
                      if low <= start and start + sizes[index] <= high]
            for place, index in enumerate(chosen):
                block = blocks[index]
                names = list(block["names"])
                if place == 0 and not names and sections[low][1]:
                    names = [sections[low][1]]
                out.extend("% " + name + ending for name in names)
                for name in abc_tools.VOICES:
                    voice = score.voices[name]
                    head = list(block["voices"][name]["head"])
                    if place == 0:
                        start, _length, meter = voice.bars[low]
                        wanted = ("M:{}/{}".format(*meter), "K:" + _key_at(voice.keys, start))
                        fields = [line for line in head[1:]]
                        for field, have in zip(wanted, state[name]):
                            if field != have and not any(line.startswith(field[:2]) for line in fields):
                                head.insert(1, field)
                    meter, key = state[name]
                    for line in head[1:]:
                        if line.startswith("M:"):
                            meter = line
                        elif line.startswith("K:"):
                            key = line
                    state[name] = (meter, key)
                    music = block["voices"][name]["music"]
                    if place == len(chosen) - 1 and music.rstrip().endswith("-|"):
                        music = music.rstrip()[:-2] + "|"
                    out.extend(line + ending for line in head)
                    out.append(music + ending)
        result = "".join(out)
        _moved_back(read(source), read(result), stretches)
    except (ValueError, KeyError, IndexError) as error:
        raise ValueError(NOT_MOVED.format(reason=_reason(error))) from error
    raw = str(text or "")
    leading = raw[:len(raw) - len(raw.lstrip())]
    trailing = raw[len(raw.rstrip()):]
    return leading + result.strip() + trailing


NOT_THE_BARS = (
    "The new score {what}. Only its notes and chords can change while the rest of the song is "
    "kept as it was sung: sing the song again for that, or change the notes alone."
)

SAME_NOTES = "The new score sings the same notes as the song in bars {first} to {stop}."


def _same_bars(old, new) -> None:
    """Two sheets from ``read`` that may differ in their notes and chords only, or a ValueError saying what else moved."""
    if new["unit"] != old["unit"]:
        raise ValueError(NOT_THE_BARS.format(what="is written on another note length"))
    if new["bpm"] != old["bpm"]:
        raise ValueError(NOT_THE_BARS.format(what="has another tempo"))
    if len(new["bars"]) != len(old["bars"]):
        raise ValueError(NOT_THE_BARS.format(what="has {} bars where the song has {}".format(
            len(new["bars"]), len(old["bars"]))))
    for number, (was, now) in enumerate(zip(old["bars"], new["bars"]), 1):
        if (was["start"], was["length"], was["meter"]) != (now["start"], now["length"], now["meter"]):
            raise ValueError(NOT_THE_BARS.format(what="changes the length of bar {}".format(number)))
        if was["key"] != now["key"]:
            raise ValueError(NOT_THE_BARS.format(what="changes the key of bar {}".format(number)))
    if new["sections"] != old["sections"]:
        raise ValueError(NOT_THE_BARS.format(what="names its sections otherwise"))


def _changed_bars(old, new) -> set:
    """The bars whose notes, in either part, or chords are not what they were, spread along ties as ``write`` spreads them."""
    grid = [(bar["start"], bar["length"]) for bar in old["bars"]]
    found = set()
    for name in abc_tools.VOICES:
        was = sorted((n["start"], n["length"], n["pitch"]) for n in old["notes"][name])
        now = sorted((n["start"], n["length"], n["pitch"]) for n in new["notes"][name])
        chords = ([(c["start"], c["name"]) for c in old["chords"]],
                  [(c["start"], c["name"]) for c in new["chords"]]) if name == "Vocal" else ([], [])
        found |= _dirty(grid, was, now, chords[0], chords[1])
    return found


def changed(before: str, after: str) -> list:
    """The bars, counted from 0, whose music *after* changes in *before*: what singing the new notes sings again.

    Both are scores of one song, and they must keep the same bars -- as many,
    each as long, in the same meter and key, under the same tempo, note
    length and section names -- because only the music inside the bars can
    change while the sound around them stays as it was sung. A bar is changed
    when a note sounding in it, in either part, or a chord starting in it is
    not what it was, and a note tied into a neighbouring bar takes that bar
    along, the way ``write`` spreads an edit. A ValueError says what else
    moved.
    """
    old, new = read(before), read(after)
    _same_bars(old, new)
    return sorted(_changed_bars(old, new))


def taken(before: str, after: str, first: int, stop: int) -> dict:
    """``{"abc": text, "bars": [first, stop]}``: *before* with the music *after* has in bars *first* to *stop*.

    This is one change of notes out of several made at once: *after* may
    change bars far apart, and each stretch of them can be sung on its own,
    each on the song the one before left. Everything outside the stretch
    stays as *before* has it. A note held across either edge of the stretch,
    in either score, takes the bar on the other side in too, so ``bars`` is
    the stretch as it grew until nothing crossed its edges. Only the bars
    whose music changed are written again (see ``write``), so a stretch
    with no change in it comes back as *before* itself, character for
    character, and so does what surrounds the score: the prompt a song was
    sung under is read from that text, a final newline included.
    """
    old, new = read(before), read(after)
    _same_bars(old, new)
    count = len(old["bars"])
    if not (_whole(first) and _whole(stop) and 0 <= first < stop <= count):
        raise ValueError("Bars {} to {} are not bars of this score, which has {}.".format(
            first + 1 if _whole(first) else first, stop, count))
    starts = [bar["start"] for bar in old["bars"]] + [old["total"]]
    held = [(n["start"], n["start"] + n["length"]) for sheet in (old, new)
            for name in abc_tools.VOICES for n in sheet["notes"][name]]
    while True:
        low, high = starts[first], starts[stop]
        earlier = first > 0 and any(begun < low < ended for begun, ended in held)
        later = stop < count and any(begun < high < ended for begun, ended in held)
        if not earlier and not later:
            break
        first, stop = first - earlier, stop + later

    def inside(item):
        return low <= item["start"] < high

    sheet = {"notes": {name: [n for n in old["notes"][name] if not inside(n)]
                       + [n for n in new["notes"][name] if inside(n)]
                       for name in abc_tools.VOICES},
             "chords": [c for c in old["chords"] if not inside(c)]
             + [c for c in new["chords"] if inside(c)]}
    raw = str(before or "")
    leading = raw[:len(raw) - len(raw.lstrip())]
    trailing = raw[len(raw.rstrip()):]
    return {"abc": leading + write(before, sheet)["abc"].strip() + trailing, "bars": [first, stop]}
