"""Which words of a song belong to which line of its lyrics, and the stretch a line is sung in.

A change of words is sung over the stretch the words themselves are sung in,
and only the song's own sound can say where that is: the score has a note for
every syllable but no idea which line the singer was on, and the sections of
the grid are whole verses. The times come from the forced aligner
(``asr.aligner``), one pair a word, and this module is the arithmetic on them:
the words of each line, the seconds a line runs, and the stretch to hand an
edit.

The stand measured the shape of that stretch on 2026-09-19, three songs by
three seeds: a region opening at the new line's own first word let the model
resume inside the old line and the words ran late, while a region opening at
the last word of the line before -- which the model then sings again, as a
singer runs into a line -- came back with every word sung, 9 of 9.

The same stand picked the best of several takes of new words by hearing
them, and ``heard`` is how it counted: how many of the words asked for a
recogniser hears sung, in their order. A retake is heard the same way against
the words the song sang there, ``within``.

Standard library and the word rule the aligner counts by, so this runs
wherever the edit list runs, with no model and no torch.
"""

from __future__ import annotations

from ..asr.align import normal, words_of
from . import ops

EARLY = 0.08
"""How far before the first word a stretch opens: one step of the aligner's clock, 80 ms.

The times come back on that grid, so a word's start is already up to a step
late; opening a step earlier is opening on the word rather than inside it."""


def sung_lines(lyrics: str) -> list:
    """``[(line number, [words])]`` for every line of ``lyrics`` with words in it, in order.

    Line numbers count from 0 over the lyrics as they are written, tags and
    blank lines included, which is how an edit names the lines it rewrites.
    """
    found = []
    for number, line in enumerate(str(lyrics or "").splitlines()):
        if ops.tagged(line):
            continue
        words = words_of(line)
        if words:
            found.append((number, words))
    return found


def heard_text(lyrics: str) -> str:
    """The sung lines of ``lyrics``, one to a line, which is the text the aligner is given.

    The words are the cleaned ones, so the list that comes back lines up with
    ``sung_lines`` word for word however the lyrics were punctuated.
    """
    return chr(10).join(" ".join(words) for _number, words in sung_lines(lyrics))


def placed(lyrics: str, times) -> list:
    """``[(line number, start, stop)]`` in seconds for every sung line, from the aligner's word times.

    ``times`` is what ``asr.aligner.align`` returns for ``heard_text``: one
    ``(word, start, stop)`` for each word, in order. A line's start is its
    first word's start and its stop its last word's end.
    """
    lines = sung_lines(lyrics)
    wanted = sum(len(words) for _number, words in lines)
    if len(times) != wanted:
        raise ValueError("The song was heard word by word, but {} times came back for {} words. "
                         "The words the times were measured on are not these.".format(
                             len(times), wanted))
    found = []
    at = 0
    for number, words in lines:
        first, last = times[at], times[at + len(words) - 1]
        found.append((number, float(first[1]), float(last[2])))
        at += len(words)
    return found


def region(lyrics: str, times, first: int, stop: int):
    """The seconds to sing when lines ``first`` to ``stop`` of ``lyrics`` are rewritten.

    It opens at the last word of the line before -- the model sings that word
    again and runs from it into the new line -- and closes at the first word of
    the line after, each a step early. A first line has nothing to run in from
    and opens on its own first word; a last line closes at the end of the song,
    which is ``None`` here: how long the song is belongs to the caller, and a
    stretch that reaches the end is one the model may finish itself.
    """
    lines = sung_lines(lyrics)
    numbers = [number for number, _words in lines]
    inside = [index for index, number in enumerate(numbers) if first <= number < stop]
    if not inside:
        raise ValueError("Nothing is sung in the lines chosen, so there are no words to change.")
    at, last = inside[0], inside[-1]
    spans = placed(lyrics, times)
    if at > 0:
        away = sum(len(words) for _number, words in lines[:at]) - 1
        opening = float(times[away][1])
    else:
        opening = spans[at][1]
    closing = None if last + 1 >= len(lines) else spans[last + 1][1]
    start = max(0.0, opening - EARLY)
    if closing is None:
        return start, None
    return start, max(start, closing - EARLY)


def carried(spans, first: int, stop: int, lyrics: str, count: int) -> list:
    """The spans of a song's lines before a change of words, numbered as the lines after it.

    ``spans`` is ``placed`` on the words as they were, and lines ``first`` to
    ``stop`` of those words became the ``count`` lines of new words that
    stand at ``first`` in ``lyrics``. A line before the change keeps its
    number and a line after it moves by however many lines the change added
    or took away. A new line was never sung in the song as it was, so it is
    given the place of the line it replaced: one for one when there are as
    many sung lines on each side, and the whole stretch the old ones ran
    otherwise.
    """
    shift = int(count) - (int(stop) - int(first))
    fresh = [number for number, _words in sung_lines(lyrics)
             if first <= number < first + int(count)]
    old = [span for span in spans if first <= span[0] < stop]
    found = []
    for number, start, end in spans:
        if number < first:
            found.append((number, start, end))
        elif number >= stop:
            found.append((number + shift, start, end))
    if old and len(old) == len(fresh):
        found.extend((number, span[1], span[2]) for number, span in zip(fresh, old))
    elif old:
        whole = (min(span[1] for span in old), max(span[2] for span in old))
        found.extend((number, whole[0], whole[1]) for number in fresh)
    return sorted(found)


def within(times, start: float, stop: float) -> str:
    """The words sung between ``start`` and ``stop`` seconds, by the aligner's ``times``: what a retake there sings again.

    A word belongs to the stretch its middle lies in. A retake opens at a
    pickup and closes at the next, so a word cut across either end is the
    rare one, and the middle gives it to the side that sings most of it.
    """
    return " ".join(str(word) for word, begun, ended in times
                    if start <= (float(begun) + float(ended)) / 2.0 < stop)


def _plain(text: str) -> list:
    return [word for word in (normal(word) for word in words_of(text)) if word]


def heard(expected: str, said: str) -> tuple:
    """``(found, wanted)``: how many of the ``wanted`` words of ``expected`` are in ``said``, in order.

    ``said`` is what a recogniser heard sung and ``expected`` what the singing
    was asked for. A word counts when it is heard where the order of the
    words allows it -- the longest run of them in order, with anything
    between -- which is how the stand scored takes of new words on
    2026-09-19: a word heard besides them, like the last word of the line
    before that a change of words sings again, costs nothing, and a word
    heard out of its place is not one sung. Words are compared as
    ``asr.align.normal`` leaves them, so case, "yo" and punctuation do not
    count.
    """
    wanted = _plain(expected)
    got = _plain(said)
    row = [0] * (len(got) + 1)
    for word in wanted:
        before = 0
        for index, other in enumerate(got, 1):
            kept = row[index]
            row[index] = before + 1 if word == other else max(row[index], row[index - 1])
            before = kept
    return row[-1], len(wanted)
