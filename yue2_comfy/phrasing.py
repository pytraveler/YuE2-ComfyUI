"""Lyrics laid along a bare tune: the bars of a score each line of the words is sung on.

YuE2 learned from songs whose words and score belong together: a line of words
on a phrase of the tune, with about as many notes as the line has syllables and
a breath before the next line. A tune from a MIDI file has no words, and words
written for something else, sung over it as it stands, come out as something
else. Measured on 2026-09-16, Qwen3-ASR listening to four seeds of each:

- eight lines sung over the Pirates of the Caribbean theme as the file gives
  it -- no sections, about fifteen notes to a seven-syllable line, no breath
  between -- were not heard at all, 0 of 46 words in order;
- the same eight lines on the GTA San Andreas intro, a line to each of its
  phrases and the chorus on the verse's bars again, were heard 94 to 98
  percent in order with a word error rate of 0.02 to 0.07, as clearly as over
  the score the model wrote for them itself (96 to 100 percent, 0 to 0.04);
  over that tune as it stands the chorus fell on a riff of three notes a bar
  and was lost;
- the model's own score for those lines, its section comments taken out, was
  still sung cleanly in three seeds of four: the fit of the lines to the
  phrases matters, not the comments alone.

So a score that names no section -- a bare tune, as 'YuE2 Load MIDI' hands one
on -- is laid out for its words before it is sung:

- The voice's notes are split into phrases where a singer breathes: at a rest
  of an eighth note or more, and after a note held for a half note or longer.
  Phrases are also cut at their bar lines, so that a line can end at a bar
  line where the tune runs on without a breath -- in the GTA intro the verse's
  last phrase runs straight into a riff, and the words sung on the riff's
  first notes were words the model made up.
- Each line takes the phrase, or up to eight pieces in a row, whose notes come
  nearest its syllables at a pace a line is sung at, 0.8 to 3.5 syllables a
  second. It takes the next phrase when that fits, skips phrases that do not,
  and goes back to the start of the tune when the words outlast it.
- A section whose lines repeat an earlier section's -- a chorus sung again --
  is sung on that section's bars.
- Up to eight bars before the first phrase stay as the intro when the first
  line is sung on that phrase. The score ends one empty bar after the last
  line, as the model's own scores end: without that bar the singing ran on
  past the score in 5 songs of 18, by 14 to 52 seconds, and with it in 7 of
  64, six of them over the Pirates theme -- which is why the length ceiling at
  'max_seconds' 0 follows the tune too (see ``SLACK``).

Laid out this way and sung through template 7 as it ships, eight seeds each,
the template's lines were heard 83 to 100 percent in order over the GTA intro,
97 on average, with a median word error rate of 0.02: as clearly as over the
model's own score. Over the Pirates theme they were heard 22 to 98 percent, 73
on average, with a median error rate of 0.50 -- a theme that fast, with no
breath, keeps losing words or gaining words of the model's own however its
lines are laid -- so lines laid on half again as many notes as they have
syllables, or with half again as many syllables as notes, get a warning with
the numbers.

Syllables are counted from the letters: vowel groups in Latin script, with a
silent final e, a vowel before '-ing' and a sounded 'i' before another vowel
counted apart, and a syllable less for the few common words singers shorten,
'every' and 'different' among them; every vowel in Cyrillic; every character in
Chinese, Japanese and Korean. It is a count to lay lines out by, not a
dictionary.
"""

from __future__ import annotations

import bisect
import math
import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction

from .sheetsage import abc_rebuild
from .sheetsage import sections as score_sections
from .vendor.yue2_music import abc_tools

BREATH = Fraction(1, 2)
"""Quarter notes of rest that end a phrase: an eighth note, the breath between two lines in the model's own scores."""

HELD = Fraction(2)
"""Quarter notes a note is held for that end the phrase it closes: a half note."""

MOST_PIECES = 8
"""Pieces of the tune in a row that one line may be sung on."""

FASTEST = 3.5
"""Syllables a second above which a line is crowded: the model's own lines run 1.3 to 2.8, and 4.3 was mostly lost."""

SLOWEST = 0.8
"""Syllables a second below which a line drags over its notes."""

PACE_WEIGHT = 2.0
"""How much more a line sung too fast or too slow costs than the same mismatch between notes and syllables."""

JOIN = 0.15
"""What it costs one line to run on over a breath of the tune."""

MID_PHRASE = 0.2
"""What it costs a line to end where the tune does not breathe."""

JUMP = 0.5
"""What it costs a line not to follow on from the line before in the tune."""

SECTION_JUMP = 0.25
"""The same at the first line of a section, where songs move to another part of the tune anyway."""

SKIP = 0.05
"""What each phrase passed over costs on top, so a near phrase is taken before a far one."""

LOOP = 0.2
"""What going back to an earlier phrase costs on top, so a phrase ahead is taken before one already sung."""

LOOSE = math.log(1.5)
"""How far notes and syllables may part, on average over the lines, before the node warns.

Half again as many notes as syllables, or the other way round: the model's own
lines part by a factor of 1.3 on average, the Pirates theme laid out by 1.6.
"""

INTRO_BARS = 8
"""Bars before the first phrase that stay as the intro."""

SLACK = 1.1
"""How much longer than its laid-out score a song may run before it is stopped, as a factor..."""

SLACK_SECONDS = 2.0
"""...and in seconds on top.

The model's songs ended within three seconds of their scores: the GTA intro
laid out ran up to 2.4 seconds over, the model's own score ended up to 2.8
seconds early. Over the Pirates theme the singing ran on past its 19-second
score in 5 songs of 8, to 36 to 85 seconds, all of it words of the model's own.
With this ceiling the same eight seeds ended at 18 to 23 seconds.
"""

LISTED = 6
"""Sections and bar ranges a notice names before it says how many more there are."""

LAID = (
    "The score came without sections, so the lyrics were laid along its tune before it was sung: {where}. "
    "{unsung}The song ends one empty bar after the last line, {seconds:.0f} seconds in; with 'max_seconds' at 0 "
    "the singing is stopped at {ceiling:.0f} seconds should it run on."
)
UNSUNG = "Not sung: {bars} of the tune. "
CROWDED_NOTES = (
    "The tune has about {ratio} notes for each syllable of these lyrics -- its phrases hold about {notes} notes, "
    "the lines about {syllables} syllables -- so YuE2 is likely to sing sounds or words of its own on the notes "
    "left over."
)
CROWDED_WORDS = (
    "These lyrics have more syllables than the tune has notes for them -- its phrases hold about {notes} notes, "
    "the lines about {syllables} syllables -- so YuE2 is likely to crowd or drop words."
)

TAG = re.compile(r"\[\s*([^\]]*?)\s*\]")
WORD = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*")
LATIN_VOWELS = re.compile("[aeiouy\u00e6\u00f8\u0153]+")
GREEK_VOWELS = re.compile("[\u03b1\u03b5\u03b7\u03b9\u03bf\u03c5\u03c9]+")
SEPARATE_VOWEL = re.compile("(?<=[aeiouy])[\u00ef\u00eb\u00fc\u00ff]")
STRESSED_E = ("\u00e9", "\u00e8", "\u00ea", "\u00eb")
CYRILLIC_VOWELS = frozenset("\u0430\u0435\u0451\u0438\u043e\u0443\u044b\u044d\u044e\u044f\u0456\u0457\u0454")
SHORTENED = frozenset({"every", "everything", "everybody", "everywhere", "everyday", "evening", "evenings",
                       "different", "difference", "differently", "several", "favorite", "favourite", "camera",
                       "chocolate", "interest", "interesting", "business", "vegetable", "comfortable", "temperature"})
"""Words sung a syllable shorter than they are spelled: 'ev-ry', not 'ev-er-y'."""

SOUNDED_I = re.compile("(?<=[a-z])(?<![aeiouytscxgn])(?<!ll)i(?=[aou])")
"""An 'i' sung as a syllable of its own before another vowel: pi-a-no, ra-di-o; not in -tion, -cial or mil-lion."""

ALIASES = {"prechorus": "pre-chorus", "pre chorus": "pre-chorus", "post chorus": "post-chorus",
           "hook": "chorus", "refrain": "chorus"}
QUALITY_LABELS = {text: label for label, text in abc_rebuild.QUALITY_TEXT.items()}
CHORD_SYMBOL = re.compile(r"(?P<root>[A-G](?:bb|##|b|#)?)(?P<quality>.*?)(?:/(?P<bass>[A-G](?:bb|##|b|#)?))?")


@dataclass(frozen=True)
class Laid:
    """A score laid out for its words, and what the node says about how."""

    score: str
    notices: list
    seconds: float


@dataclass(frozen=True)
class Piece:
    """The part of a phrase inside one bar: when it sounds, its notes, and whether a breath comes before it."""

    start: Fraction
    end: Fraction
    notes: int
    breath: bool


def named(score: str) -> bool:
    """Whether a score names a section: a ``% label`` comment line anywhere in it."""
    return any(line.startswith("% ") for line in str(score or "").split("\n"))


def bare(score: str) -> str:
    """The score with its section comments taken out."""
    return "\n".join(line for line in str(score or "").split("\n") if not line.startswith("% "))


def labelled(score: str, label: str = "verse") -> str:
    """The score naming at least one section: a score that names none is one ``label`` from its first bar.

    Text with no ``V: Vocal`` line of music is given back as it is.
    """
    text = str(score or "")
    if named(text):
        return text
    lines = text.split("\n")
    if "V: Vocal" not in lines:
        return text
    at = lines.index("V: Vocal")
    return "\n".join(lines[:at] + ["% " + label] + lines[at:])


def _wide(character: str) -> bool:
    return ("\u4e00" <= character <= "\u9fff" or "\u3400" <= character <= "\u4dbf"
            or "\u3040" <= character <= "\u30ff" or "\uac00" <= character <= "\ud7af")


def _word_syllables(word: str) -> int:
    lowered = word.lower()
    if any("\u0400" <= character <= "\u04ff" for character in lowered):
        return max(1, sum(1 for character in lowered if character in CYRILLIC_VOWELS))
    wide = sum(1 for character in lowered if _wide(character))
    if wide:
        return wide
    parted = SEPARATE_VOWEL.sub(lambda found: "|" + found.group(0), lowered)
    plain = "".join(character for character in unicodedata.normalize("NFD", parted)
                    if not unicodedata.combining(character))
    if any("\u0370" <= character <= "\u03ff" for character in plain):
        return max(1, len(GREEK_VOWELS.findall(plain)))
    count = len(LATIN_VOWELS.findall(plain))
    if plain.endswith("ing") and len(plain) > 3 and plain[-4] in "aeiouy":
        count += 1
    count += len(SOUNDED_I.findall(plain))
    if plain in SHORTENED:
        count -= 1
    if count > 1 and len(plain) > 2:
        if plain.endswith("e") and not lowered.endswith(STRESSED_E) and not plain.endswith(("ee", "ie", "ye")) \
                and not (plain.endswith("le") and plain[-3] not in "aeiouy"):
            count -= 1
        elif plain.endswith("es") and len(plain) > 3 and plain[-3] not in "aeiouysxzcgh":
            count -= 1
        elif plain.endswith("ed") and len(plain) > 3 and plain[-3] not in "aeiouytd":
            count -= 1
    return max(1, count)


def syllables(line: str) -> int:
    """About how many syllables a line of words has: 0 for a line without a letter."""
    return sum(_word_syllables(word) for word in WORD.findall(str(line or "")))


def label_of(tag: str) -> str:
    """The section comment a lyrics tag becomes: '[Verse 2]' a 'verse', '[Hook]' a 'chorus', a tag it does not know a 'verse'."""
    clean = " ".join(str(tag).lower().replace("_", " ").split())
    clean = re.sub(r"[\s\d:.#-]+$", "", clean)
    clean = ALIASES.get(clean, clean)
    return clean if clean in score_sections.TAGS else "verse"


def lyric_sections(lyrics: str) -> list:
    """The lyrics as ``{"tag", "label", "lines"}`` sections in order; lines before any tag are a verse.

    A line is sung when it has a letter in it, so a line of punctuation or a
    bare number is left out.
    """
    found = []
    for raw in str(lyrics or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        tag = TAG.fullmatch(line)
        if tag:
            found.append({"tag": tag.group(1), "label": label_of(tag.group(1)), "lines": []})
            continue
        if not syllables(line):
            continue
        if not found:
            found.append({"tag": "Verse", "label": "verse", "lines": []})
        found[-1]["lines"].append(line)
    return found


def _phrases(notes: list) -> list:
    groups = []
    for note in sorted(notes):
        onset, _pitch, length = note
        if groups:
            last = groups[-1][-1]
            if onset - (last[0] + last[2]) < BREATH and last[2] < HELD:
                groups[-1].append(note)
                continue
        groups.append([note])
    return groups


def pieces(notes: list, bar_starts: list) -> list:
    """The voice's ``[onset, pitch, length]`` notes as the pieces lines are laid on, in order.

    A phrase ends at a rest of an eighth note or more and after a note held for
    a half note or longer, and is cut into a piece for each bar its notes start
    in. ``breath`` marks the first piece of each phrase.
    """
    found = []
    for group in _phrases(notes):
        chunks = []
        for note in group:
            bar = bisect.bisect_right(bar_starts, note[0]) - 1
            if chunks and chunks[-1][0] == bar:
                chunks[-1][1].append(note)
            else:
                chunks.append((bar, [note]))
        for position, (_bar, chunk) in enumerate(chunks):
            found.append(Piece(chunk[0][0], chunk[-1][0] + chunk[-1][2], len(chunk), position == 0))
    return found


def _path(found: list, counts: list, start: int, first: bool, bpm: int) -> list:
    """``(k, m)`` for each line of one section: it is sung on ``m`` pieces from piece ``k``.

    The cheapest way through, line by line: each line's cost is how far its
    notes are from its syllables and its pace from a singable one, and leaving
    the tune's order costs on top. ``start`` is the piece after the last one
    sung, 0 at the start of the song.
    """
    size = len(found)
    total = [0]
    for piece in found:
        total.append(total[-1] + piece.notes)
    per_quarter = 60.0 / bpm
    infinity = float("inf")

    def cost(k, m, count):
        notes = total[k + m] - total[k]
        seconds = float(found[k + m - 1].end - found[k].start) * per_quarter
        pace = count / seconds if seconds > 0 else FASTEST * 8
        value = abs(math.log(notes / count))
        value += PACE_WEIGHT * (max(0.0, math.log(pace / FASTEST)) + max(0.0, math.log(SLOWEST / pace)))
        value += JOIN * sum(1 for piece in found[k + 1:k + m] if piece.breath)
        if k + m < size and not found[k + m].breath:
            value += MID_PHRASE
        return value

    best = [infinity] * (size + 1)
    best[min(start, size)] = 0.0
    steps = []
    for index, count in enumerate(counts):
        jump = SECTION_JUMP if index == 0 and not first else JUMP
        ahead = [(infinity, -1)] * (size + 1)
        low = (infinity, -1)
        for k in range(size + 1):
            ahead[k] = low
            if best[k] - SKIP * k < low[0]:
                low = (best[k] - SKIP * k, k)
        behind = [(infinity, -1)] * (size + 1)
        low = (infinity, -1)
        for k in range(size, -1, -1):
            behind[k] = low
            if best[k] < low[0]:
                low = (best[k], k)
        following = [infinity] * (size + 1)
        chosen = [None] * (size + 1)
        for k in range(size):
            came, origin = best[k], k
            if ahead[k][1] >= 0 and ahead[k][0] + jump + SKIP * k < came:
                came, origin = ahead[k][0] + jump + SKIP * k, ahead[k][1]
            if behind[k][1] >= 0 and behind[k][0] + jump + LOOP < came:
                came, origin = behind[k][0] + jump + LOOP, behind[k][1]
            if came == infinity:
                continue
            for m in range(1, min(MOST_PIECES, size - k) + 1):
                value = came + cost(k, m, count)
                if value < following[k + m]:
                    following[k + m] = value
                    chosen[k + m] = (k, m, origin)
        steps.append(chosen)
        best = following
    end = min(range(size + 1), key=lambda k: (best[k], k))
    spans = []
    for chosen in reversed(steps):
        k, m, origin = chosen[end]
        spans.append((k, m))
        end = origin
    spans.reverse()
    return spans


def _key(lines: list) -> tuple:
    return tuple(" ".join(re.sub(r"[^\w\s']", " ", line.lower()).split()) for line in lines)


def _bar_at(bar_starts: list, time) -> int:
    return max(0, bisect.bisect_right(bar_starts, time) - 1)


def _last_bar(bar_starts: list, end) -> int:
    return max(0, bisect.bisect_left(bar_starts, end) - 1)


def _bars(numbers: list) -> str:
    """Bar numbers as a person reads them: 'bar 5', 'bars 5-9', 'bars 1-4 and 9'."""
    runs = []
    for number in numbers:
        if runs and number <= runs[-1][1] + 1:
            runs[-1][1] = max(runs[-1][1], number)
        else:
            runs.append([number, number])
    text = ["{}".format(low) if low == high else "{}-{}".format(low, high) for low, high in runs]
    if len(text) > LISTED:
        text = text[:LISTED] + ["{} more".format(len(text) - LISTED)]
    joined = ", ".join(text[:-1]) + " and " + text[-1] if len(text) > 1 else text[0]
    return ("bar " if len(runs) == 1 and runs[0][0] == runs[0][1] else "bars ") + joined


def chord_label(symbol: str) -> str:
    """An ABC chord symbol such as ``Bbm7/F`` as the label the score writer takes, ``Bb:min7/F``."""
    found = CHORD_SYMBOL.fullmatch(symbol)
    if found is None or found.group("quality") not in QUALITY_LABELS:
        raise ValueError("unsupported chord symbol {!r}".format(symbol))
    bass = found.group("bass")
    return found.group("root") + ":" + QUALITY_LABELS[found.group("quality")] + ("/" + bass if bass else "")


def _intervals(starts: list, end) -> list:
    rows = []
    for index, (time, value) in enumerate(starts):
        following = starts[index + 1][0] if index + 1 < len(starts) else end
        if time < following:
            rows.append([time, following, value])
    return rows


def _write(parsed, runs: list) -> str:
    vocal, instrument = parsed.voices["Vocal"], parsed.voices["Ins"]
    per_quarter = 60.0 / parsed.bpm
    chords = _intervals(vocal.chords, vocal.time)
    beats, notes, harmony, keys, labels = [], [], [], [], []
    now = Fraction(0)
    meter = vocal.bars[0][2]
    for run in runs:
        begin = vocal.bars[run["b0"]][0]
        finish = vocal.bars[run["b1"]][0] + vocal.bars[run["b1"]][1]
        shift = now - begin
        for start, _length, meter in vocal.bars[run["b0"]:run["b1"] + 1]:
            numerator, denominator = meter
            for beat in range(numerator):
                beats.append([float((start + shift + beat * Fraction(4, denominator)) * per_quarter),
                              beat + 1, numerator, denominator])
        for track, voice in enumerate((vocal, instrument)):
            for onset, pitch, length in voice.notes:
                if not begin <= onset < finish:
                    continue
                if track == 0 and not any(low <= onset < high for low, high in run["windows"]):
                    continue
                notes.append([float((onset + shift) * per_quarter),
                              float((min(onset + length, finish) + shift) * per_quarter), pitch, track])
        for low, high, symbol in chords:
            low, high = max(low, begin), min(high, finish)
            if low < high:
                harmony.append([float((low + shift) * per_quarter), float((high + shift) * per_quarter),
                                chord_label(symbol)])
        keyed = [(time, key) for time, key in vocal.keys if time <= begin]
        keys.append((now, keyed[-1][1] if keyed else vocal.keys[0][1]))
        keys.extend((time + shift, key) for time, key in vocal.keys if begin < time < finish)
        labels.extend((time + shift, label) for time, label in run["labels"])
        now += finish - begin
    numerator, denominator = meter
    for beat in range(numerator + 1):
        beats.append([float((now + beat * Fraction(4, denominator)) * per_quarter),
                      beat % numerator + 1, numerator, denominator])
    now += Fraction(4 * numerator, denominator)
    merged = []
    for time, key in keys:
        if merged and merged[-1][1] == key:
            continue
        if merged and merged[-1][0] == time:
            merged[-1] = (time, key)
        else:
            merged.append((time, key))
    rows = {"beats": beats,
            "chords": harmony,
            "keys": [[float(low * per_quarter), float(high * per_quarter), key]
                     for low, high, key in _intervals(merged, now)],
            "structures": [[float(low * per_quarter), float(high * per_quarter), label]
                           for low, high, label in _intervals(labels, now)],
            "notes": sorted(notes)}
    denominators = [bar[2][1] for run in runs for bar in vocal.bars[run["b0"]:run["b1"] + 1]]
    subbeats = max(1, parsed.unit.denominator // min(denominators))
    return abc_rebuild.build(rows, melody_only=not vocal.chords, subbeats=subbeats)


def lay(score: str, lyrics: str):
    """The score laid out for these lyrics, as a ``Laid``; None when it is sung as it came.

    None for a score that names a section, for one not in YuE2's dialect, for
    one whose voice has no notes, and for lyrics with no sung line.
    """
    text = str(score or "")
    if named(text):
        return None
    try:
        parsed = abc_tools.parse(text)
    except ValueError:
        return None
    vocal = parsed.voices["Vocal"]
    words = lyric_sections(lyrics)
    if not vocal.notes or not any(section["lines"] for section in words):
        return None
    bar_starts = [bar[0] for bar in vocal.bars]
    found = pieces(vocal.notes, bar_starts)
    sung = []
    done = {}
    position = 0
    for section in words:
        if not section["lines"]:
            continue
        counts = [max(1, syllables(line)) for line in section["lines"]]
        key = _key(section["lines"])
        if key in done:
            spans = done[key]
        else:
            spans = _path(found, counts, position, not sung, parsed.bpm)
            done[key] = spans
        position = spans[-1][0] + spans[-1][1]
        sung.append((section, counts, spans))

    runs = []
    previous = None
    for section, _counts, spans in sung:
        for index, (k, m) in enumerate(spans):
            begin, end = found[k].start, found[k + m - 1].end
            first_bar, last_bar = _bar_at(bar_starts, begin), _last_bar(bar_starts, end)
            if runs and previous is not None and k == previous:
                runs[-1]["b1"] = max(runs[-1]["b1"], last_bar)
                runs[-1]["windows"][-1][1] = end
            elif runs and previous is not None and k > previous and first_bar <= runs[-1]["b1"]:
                runs[-1]["b1"] = max(runs[-1]["b1"], last_bar)
                runs[-1]["windows"].append([begin, end])
            else:
                runs.append({"b0": first_bar, "b1": last_bar, "windows": [[begin, end]], "labels": []})
            if index == 0:
                runs[-1]["labels"].append((begin, section["label"]))
            previous = k + m
    intro = []
    if sung and sung[0][2][0][0] == 0 and runs[0]["b0"] > 0:
        leading = [section for section in words[:words.index(sung[0][0])] if not section["lines"]]
        runs[0]["b0"], intro = max(0, runs[0]["b0"] - INTRO_BARS), list(range(max(0, runs[0]["b0"] - INTRO_BARS),
                                                                          runs[0]["b0"]))
        runs[0]["labels"].insert(0, (vocal.bars[runs[0]["b0"]][0], leading[-1]["label"] if leading else "intro"))
    try:
        written = _write(parsed, runs)
        laid = abc_tools.parse(written)
    except ValueError:
        return None
    seconds = float(laid.voices["Vocal"].time) * 60.0 / laid.bpm

    where = []
    if intro:
        where.append("{} as the intro".format(_bars([bar + 1 for bar in intro])))
    for section, _counts, spans in sung:
        bars = sorted({bar + 1 for k, m in spans
                       for bar in range(_bar_at(bar_starts, found[k].start),
                                        _last_bar(bar_starts, found[k + m - 1].end) + 1)})
        where.append("[{}] on {}".format(section["tag"], _bars(bars)))
    if len(where) > LISTED:
        where = where[:LISTED] + ["{} more sections".format(len(where) - LISTED)]
    windows = [window for run in runs for window in run["windows"]]
    voiced = {_bar_at(bar_starts, onset) for onset, _pitch, _length in vocal.notes}
    heard = {_bar_at(bar_starts, onset) for onset, _pitch, _length in vocal.notes
             if any(low <= onset < high for low, high in windows)}
    unsung = sorted(bar + 1 for bar in voiced - heard)
    notices = [("notice", LAID.format(where="; ".join(where),
                                      unsung=UNSUNG.format(bars=_bars(unsung)) if unsung else "",
                                      seconds=seconds, ceiling=ceiling(seconds)))]
    ratios = []
    for _section, counts, spans in sung:
        for count, (k, m) in zip(counts, spans):
            ratios.append(sum(piece.notes for piece in found[k:k + m]) / count)
    if sum(abs(math.log(ratio)) for ratio in ratios) / len(ratios) > LOOSE:
        middle = sorted(ratios)[len(ratios) // 2]
        phrase_notes = sorted(len(group) for group in _phrases(vocal.notes))
        line_syllables = sorted(count for _section, counts, _spans in sung for count in counts)
        values = {"ratio": "{:.1f}".format(middle), "notes": phrase_notes[len(phrase_notes) // 2],
                  "syllables": line_syllables[len(line_syllables) // 2]}
        notices.append(("warn", (CROWDED_NOTES if middle > 1 else CROWDED_WORDS).format(**values)))
    return Laid(written, notices, seconds)


def ceiling(seconds: float) -> float:
    """The most seconds a song laid out on a tune this long is let sing: the tune, and a little over."""
    return round(float(seconds) * SLACK + SLACK_SECONDS, 1)
