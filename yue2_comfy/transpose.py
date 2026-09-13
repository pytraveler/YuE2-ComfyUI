"""Moving a score to another key before it is sung.

YuE2 has no key control of its own. The style line cannot ask for one:
measured on two styles and three seeds, 'A minor', 'in the key of A minor' and
'E major' moved the key the model wrote 0 times in 18. Nor can the K: field on
its own, because the dialect reads every note relative to it -- replacing K:C
with K:G sings each F as F-sharp and leaves every other note where it was.

What the model does follow is the score. Moved as a whole -- every note by the
same number of semitones, every chord symbol and key field with it -- and sung
with the same seed, the audio lands exactly that far from the original: the
chroma of each render lined up with its score at the expected shift in every
case measured. It is a new take of the same tune rather than the old recording
pitched, because the model sings the moved score from its first note.

The dialect is upstream's, and so is the parser that reads it. The moved text
is parsed again with vendor/yue2_music/abc_tools.py and compared with the
original note by note before anything is sung, and a score that parser cannot
read is refused rather than guessed at.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from .constants import TRANSPOSE_LIMIT
from .vendor.yue2_music import abc_tools

LETTERS = "CDEFGAB"
MARKS = {-2: "__", -1: "_", 0: "=", 1: "^", 2: "^^"}
SPELLINGS = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}
NAME_SHIFTS = {"": 0, "#": 1, "##": 2, "b": -1, "bb": -2}
MARK_SHIFTS = {"=": 0, "_": -1, "__": -2, "^": 1, "^^": 2}
KEY_NAME = re.compile(r"([A-G])(#|b)?(m?)")
CHORD_NAME = re.compile(r"([A-G])(bb|##|b|#)?(.*?)(?:/([A-G])(bb|##|b|#)?)?")
FULL_REST = re.compile(r"\s*Z[2-4]?\s*")

COT_OFF = (
    "'transpose' moves the score just before it is sung, and with 'cot' set to "
    "'off' the model writes no score to move.\n\n"
    "Set 'cot' to 'full' or 'melody', or set 'transpose' back to 0."
)

UNREADABLE = (
    "This score cannot be moved {step}: {reason}.\n\n"
    "'transpose' only moves a score it can read note by note, so that the key is "
    "the one thing that changes. Set 'transpose' to 0 to sing the score as it "
    "is, or try another seed."
)


class Moved(NamedTuple):
    """A moved score, with the key it started in and the key it ended in."""

    text: str
    before: str
    after: str


def describe(semitones: int) -> str:
    """'up 2 semitones', 'down 1 semitone' or 'nowhere', for a log line or a message."""
    step = int(semitones)
    if not step:
        return "nowhere"
    return "{} {} semitone{}".format("up" if step > 0 else "down", abs(step),
                                     "" if abs(step) == 1 else "s")


def _pitch_class(letter: str, accidental: str | None) -> int:
    return (abc_tools.NATURAL[letter] + NAME_SHIFTS[accidental or ""]) % 12


def _key_parts(name: str):
    """``(letter, alteration, mode)`` of a key such as 'C#m'; the mode is '' or 'm'."""
    match = KEY_NAME.fullmatch(name.strip())
    if match is None:
        raise ValueError("'{}' is not a plain major or minor key".format(name.strip()))
    return match.group(1), NAME_SHIFTS[match.group(2) or ""], match.group(3)


def _note_text(letter: str, written: int) -> str:
    """A natural note at MIDI pitch *written*, in ABC: C is 60, c 72, C, 48, c' 84."""
    octave = (written - abc_tools.NATURAL[letter]) // 12
    if octave >= 6:
        return letter.lower() + "'" * (octave - 6)
    return letter + "," * (5 - octave)


class _Interval:
    """*semitones* up, spelled *steps* letter names up; the two carry the same sign.

    Moving the letter names along with the semitones is what keeps a score
    readable: C major up a tone is D major, where E becomes F-sharp rather than
    G-flat, and a leading tone written B-sharp stays a leading tone.
    """

    def __init__(self, semitones: int, steps: int):
        self.semitones = semitones
        self.steps = steps

    def name(self, letter: str, alteration: int):
        """A pitch-class name moved, as ``(letter, alteration)``.

        The planned letter is tried first, and a neighbouring one only when the
        planned one would need more than a double accidental.
        """
        index = LETTERS.index(letter) + self.steps
        target = (abc_tools.NATURAL[letter] + alteration + self.semitones) % 12
        for offset in (0, -1, 1):
            new_letter = LETTERS[(index + offset) % 7]
            new_alteration = (target - abc_tools.NATURAL[new_letter] + 6) % 12 - 6
            if abs(new_alteration) <= 2:
                return new_letter, new_alteration
        raise ValueError("pitch class {} has no spelling".format(target))

    def note(self, letter: str, written: int, pitch: int):
        """A sounding note moved, as ``(letter, natural MIDI pitch, alteration)``."""
        octave = (written - abc_tools.NATURAL[letter]) // 12
        degree = LETTERS.index(letter) + 7 * octave + self.steps
        target = pitch + self.semitones
        for offset in (0, -1, 1):
            moved = degree + offset
            new_letter = LETTERS[moved % 7]
            new_written = 12 * (moved // 7) + abc_tools.NATURAL[new_letter]
            if abs(target - new_written) <= 2:
                return new_letter, new_written, target - new_written
        raise ValueError("MIDI pitch {} has no spelling".format(target))

    def key(self, name: str) -> str:
        """A key field moved. A key the dialect has no name for is refused."""
        letter, alteration, mode = _key_parts(name)
        new_letter, new_alteration = self.name(letter, alteration)
        moved = new_letter + SPELLINGS[new_alteration] + mode
        if moved not in abc_tools.KEYS:
            raise ValueError("the key {} would become {}, which the score format has "
                             "no name for".format(name.strip(), moved))
        return moved

    def chord(self, text: str) -> str:
        """A chord symbol moved: the root and any slash bass, the quality untouched."""
        match = CHORD_NAME.fullmatch(text)
        if match is None:
            raise ValueError("'{}' is not a chord symbol the score format knows".format(text))
        root = self.name(match.group(1), NAME_SHIFTS[match.group(2) or ""])
        moved = root[0] + SPELLINGS[root[1]] + match.group(3)
        if match.group(4):
            bass = self.name(match.group(4), NAME_SHIFTS[match.group(5) or ""])
            moved += "/" + bass[0] + SPELLINGS[bass[1]]
        return moved


def _interval(key: str, semitones: int) -> _Interval:
    """The move from *key* into whichever spelling of the new key reads easiest.

    Of two names for the same key the one with fewer accidentals wins -- D-flat
    rather than C-sharp -- and a tie goes to the sharps, F-sharp over G-flat,
    only so that the same move is spelled the same way every time.

    The letter names move by as many steps as fit the size of the move, not just
    its direction: eleven semitones up from D is the D-flat an octave higher,
    seven letter names up, even though D and D-flat share a letter.
    """
    if semitones % 12 == 0:
        return _Interval(semitones, 7 * (semitones // 12))
    letter, alteration, mode = _key_parts(key)
    tonic = (abc_tools.NATURAL[letter] + alteration + semitones) % 12
    candidates = []
    for name, accidentals in abc_tools.KEYS.items():
        c_letter, c_alteration, c_mode = _key_parts(name)
        if c_mode == mode and (abc_tools.NATURAL[c_letter] + c_alteration) % 12 == tonic:
            candidates.append((abs(accidentals), -accidentals, c_letter))
    distance = (LETTERS.index(min(candidates)[2]) - LETTERS.index(letter)) % 7
    steps = min((distance - 7, distance, distance + 7),
                key=lambda option: abs(option - semitones * 7 / 12))
    return _Interval(semitones, steps)


def _bar(bar: str, state: dict, interval: _Interval) -> str:
    """One measure of one voice, moved.

    *state* carries what outlives a barline: the key in force and a tie still
    waiting for its continuation. Accidentals do not outlive it, in the original
    or in the copy, so both start each measure clean. Within the measure an
    accidental is written only where the new key signature and the marks already
    written would otherwise give the wrong pitch -- which is how the model writes
    them, so a move by 0 gives back the score character for character.

    The pitch of each original note is worked out exactly as upstream's parser
    works it out, including its convention that an accidental carries to every
    octave of its letter until the barline.
    """
    if FULL_REST.fullmatch(bar):
        return bar
    source_key = state["key"]
    target_key = interval.key(source_key)
    source_marks, target_marks = {}, {}
    out = []
    cursor = 0
    while cursor < len(bar):
        if bar[cursor].isspace():
            out.append(bar[cursor])
            cursor += 1
            continue
        match = abc_tools.TOKEN.match(bar, cursor)
        if match is None:
            raise ValueError("unsupported notation at {!r}".format(bar[cursor:cursor + 24]))
        cursor = match.end()
        chord, key = match.group("chord", "key")
        if chord is not None:
            out.append('"' + interval.chord(chord) + '"')
            continue
        if key is not None:
            source_key = state["key"] = key
            target_key = interval.key(key)
            source_marks, target_marks = {}, {}
            out.append("[K:" + target_key + "]")
            continue
        note, mark, octave, duration, tie = match.group("note", "acc", "oct", "duration", "tie")
        if note == "z":
            out.append(match.group(0))
            continue
        letter = note.upper()
        written = 60 + abc_tools.NATURAL[letter] + (12 if note.islower() else 0)
        written += 12 * (octave.count("'") - octave.count(","))
        alteration = source_marks.get(letter, abc_tools.key_accidentals(source_key)[letter])
        if mark:
            alteration = source_marks[letter] = MARK_SHIFTS[mark]
        pitch = written + alteration
        waiting = state["tie"]
        if waiting is not None and not mark and written == waiting[1]:
            pitch = waiting[0]
        new_letter, new_written, new_alteration = interval.note(letter, written, pitch)
        signature = abc_tools.key_accidentals(target_key)
        if waiting is not None and new_written == waiting[2]:
            new_mark = ""
        elif target_marks.get(new_letter, signature[new_letter]) == new_alteration:
            new_mark = ""
        else:
            new_mark = MARKS[new_alteration]
            target_marks[new_letter] = new_alteration
        out.append(new_mark + _note_text(new_letter, new_written) + duration + tie)
        state["tie"] = (pitch, written, new_written) if tie else None
    return "".join(out)


def _chord_moved(before: str, after: str, semitones: int) -> bool:
    """True when *after* is *before* with root and bass *semitones* higher."""
    old, new = CHORD_NAME.fullmatch(before), CHORD_NAME.fullmatch(after)
    if old is None or new is None or old.group(3) != new.group(3):
        return False
    if _pitch_class(new.group(1), new.group(2)) != (
            _pitch_class(old.group(1), old.group(2)) + semitones) % 12:
        return False
    if bool(old.group(4)) != bool(new.group(4)):
        return False
    return not old.group(4) or _pitch_class(new.group(4), new.group(5)) == (
        _pitch_class(old.group(4), old.group(5)) + semitones) % 12


def _key_moved(before: str, after: str, semitones: int) -> bool:
    """True when *after* is the key *semitones* above *before*, in the same mode."""
    old_letter, old_alteration, old_mode = _key_parts(before)
    new_letter, new_alteration, new_mode = _key_parts(after)
    return old_mode == new_mode and (
        abc_tools.NATURAL[new_letter] + new_alteration) % 12 == (
        abc_tools.NATURAL[old_letter] + old_alteration + semitones) % 12


def _check(source, result, semitones: int) -> None:
    """The moved score against the original: only pitches may differ, all by *semitones*."""
    if source.bpm != result.bpm or source.unit != result.unit:
        raise ValueError("the tempo or the note length changed")
    for name in abc_tools.VOICES:
        before, after = source.voices[name], result.voices[name]
        if before.bars != after.bars:
            raise ValueError("the bars of the {} part moved".format(name))
        if len(before.notes) != len(after.notes) or any(
                old[0] != new[0] or old[2] != new[2] or old[1] + semitones != new[1]
                for old, new in zip(before.notes, after.notes)):
            raise ValueError("a note of the {} part did not move by exactly {}".format(
                name, semitones))
        if len(before.chords) != len(after.chords) or any(
                old[0] != new[0] or not _chord_moved(old[1], new[1], semitones)
                for old, new in zip(before.chords, after.chords)):
            raise ValueError("a chord of the {} part did not move with its notes".format(name))
        if len(before.keys) != len(after.keys) or any(
                old[0] != new[0] or not _key_moved(old[1], new[1], semitones)
                for old, new in zip(before.keys, after.keys)):
            raise ValueError("a key change of the {} part did not move".format(name))


def move(text: str, semitones: int) -> Moved:
    """*text* moved by *semitones*, or a ValueError that says why it cannot be.

    The message is written for the person running the node: what went wrong and
    the two ways out, a move of 0 or another seed.
    """
    step = int(semitones)
    if abs(step) > TRANSPOSE_LIMIT:
        raise ValueError("'transpose' is {}, and it goes from -{} to {}.".format(
            step, TRANSPOSE_LIMIT, TRANSPOSE_LIMIT))
    source_text = (text or "").strip()
    if not source_text:
        raise ValueError("There is no score to move. 'transpose' needs the ABC score "
                         "the model writes before it sings.")
    try:
        source = abc_tools.parse(source_text)
        lines = source_text.splitlines(keepends=True)
        header = lines[7].rstrip("\r\n")[2:]
        interval = _interval(header, step)
        states = {name: {"key": header, "tie": None} for name in abc_tools.VOICES}
        voice = None
        output = []
        for index, raw in enumerate(lines):
            body = raw.rstrip("\r\n")
            ending = raw[len(body):]
            if index == 7:
                output.append("K:" + interval.key(header) + ending)
            elif index < 8:
                output.append(raw)
            elif body.startswith("V: "):
                voice = body[3:].strip()
                output.append(raw)
            elif body.startswith("K:"):
                states[voice]["key"] = body[2:]
                output.append("K:" + interval.key(body[2:]) + ending)
            elif index in source.music_lines:
                state = states[source.music_lines[index]]
                bars = body[:-1].split("|")
                output.append("|".join(_bar(bar, state, interval) for bar in bars)
                              + "|" + ending)
            else:
                output.append(raw)
        moved = "".join(output)
        _check(source, abc_tools.parse(moved), step)
    except (ValueError, KeyError, IndexError) as error:
        raise ValueError(UNREADABLE.format(step=describe(step), reason=error)) from error
    return Moved(moved, header.strip(), interval.key(header))
