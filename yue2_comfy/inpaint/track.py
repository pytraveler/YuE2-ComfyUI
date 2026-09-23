"""The list of edits a track window writes, and what each one does to the song under it.

An edit is a plain object in a JSON list, because that list has to survive
being saved inside a workflow, sent through ComfyUI's queue and read back by
both the window and the node. This module is the one place that knows its
shape, so the two cannot drift apart, and the tests read exactly what the
window writes::

    [{"op": "retake", "bars": [12, 16], "seed": 831001, "takes": 2, "take": 0},
     {"op": "cut", "bars": [20, 24]},
     {"op": "words", "lines": [7, 8], "text": "and the night comes down", "seed": 4},
     {"op": "notes", "score": "X:1 ...", "bars": [4, 6], "seed": 12},
     {"op": "extend", "text": "[Chorus]\nhold on ...", "seed": 9},
     {"op": "move", "bars": [13, 22], "to": 5},
     {"op": "break", "to": 13, "length": 4, "seed": 5, "takes": 2},
     {"op": "retake", "seconds": [48.0, 64.0], "seed": 7}]

Bars count from 0 and the second number is not included: [12, 16] is bars 13
to 16 as the score numbers them. Seconds are read as they are written.

Edits are made in order, each one on the song the ones before it left. A
selection is either bars of the score, which is how the window offers it, or
plain seconds, which is all a song sung with 'cot' off has: no score means no
bars. A retake sings from its own seed; asking for more takes sings the same
edit from seeds counted on from it, and 'take' says which one was kept. A cut
has neither, being the same cut however often it is made. A change of words
carries the lines it rewrites and what they become; it is sung like a retake,
and where it is sung is the stretch those lines are sung in -- which the node
finds from the song's own word times when the list names no bars and no
seconds. A change of notes carries the whole score the song is to have, as
the score editor leaves it; it is sung like a retake too, over the bars whose
notes that score changes, or over the bars it names -- and then only the
music of those bars is taken from it, so a score changed in two places far
apart is sung as two edits, each a stretch of its own. A song that goes on
carries the words it goes on with, or none for a new ending alone, and
selects nothing: it goes on from where the singing ends, the model writing
the score of the rest itself -- each take its own -- and ending the song its
own way. A move takes whole sections to the bar line ``to``, where another
section starts or the song ends. The words and the score go with the sound,
and the last bar before each seam is sung again so that the song runs on
across it; the model chooses its seeds itself, so like a cut a move has no
seed and one take. A break puts ``length`` bars of playing before the bar
line ``to``, where a section starts: the model writes their score itself, as
it does for a song that goes on, and plays them, and the section after goes
on as it was sung. It has seeds and takes as a retake has.

What an edit does to the words and the score is worked out here as well, so
the window can show it and the node can sing it without either of them
knowing how a cut finds its bars. Nothing in this module needs torch or a
model.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math

from .. import notation
from ..constants import FRAME_SECONDS, normalize_seed
from . import grid, ops

MAX_TAKES = 4
"""The most takes one edit may ask for at a time. Suno offers two; four is room to choose without a long wait."""

OPS = ("retake", "cut", "words", "notes", "extend", "move", "break")

ONE_TAKE = ("cut", "move")
"""The edits that sing nothing new, so they have no seed and come out one way."""

WRITTEN = ("extend", "break")
"""The edits whose takes write their own score: the song after one has the score of the take kept."""

NOT_A_LIST = ("The edits field does not hold a JSON list. The track window writes it -- open the "
              "track, or clear the field to start again.")

NO_SCORE = ("This song was sung with 'cot' set to 'off', so it has no score and no bars. Select "
            "the seconds to edit instead.")

SECONDS_CUT = ("A cut of a song that has a score goes by whole bars, so that its score and its "
               "words can be cut with it. Select bars rather than seconds.")

UNMATCHED = ("The sections of the lyrics could not be matched with the score's, so the words are "
             "left as they are. Read them before singing more of this song.")

NO_TIMES = ("This edit leaves the stretch to sing to the words themselves, and where they are sung "
            "was not worked out. Select the bars or the seconds to sing instead.")

FITS = ("The new words come to {} syllables where the old came to {}. The tune stays as the song "
        "sang it, so words that do not fit it are hurried, held or lost.")

NO_NOTES = ("This song was sung with 'cot' set to 'off', so it has no score and no notes to change. "
            "Select the seconds to retake instead.")

SECONDS_NOTES = ("{} changes notes, which go by bars, and it selects seconds. Select bars, or none: "
                 "the bars whose notes changed are the ones sung again.")

SAME_SCORE = "The new score sings the same notes as the song, so there is nothing to sing again."

NO_GOING_ON = ("This song was sung with 'cot' set to 'off', so it has no score for the model to go on "
               "writing. Retake its last seconds instead.")

EXTEND_WHERE = ("{} makes the song go on, which it does from where the singing ends, so it selects "
                "no bars and no seconds.")

TAGGED = ("The new words name no section, so they are sung as {}: the model writes a section's tune "
          "from its name. Start them with a tag such as [Chorus] or [Bridge] to say otherwise.")

NO_MOVE = ("This song was sung with 'cot' set to 'off', so it has no score and no sections to move. "
           "Cut and retake its seconds instead.")

MOVE_WHERE = ("{} moves sections, which go by bars: it selects bars, and 'to' names the bar line they "
              "go to.")

NOT_SECTIONS = ("{} moves bars {} to {} to bar line {}, and a move takes whole sections to where "
                "another starts or the song ends: bar line {} is inside the {}.")

UNTAGGED = ("The words sung before the first tag no longer come first, so they are sung as {}: "
            "without a tag they would run on in the section now before them.")

BARE_HEAD = ("The song now opens where the {} began in the middle of the song, with nothing before "
             "it. Retake its first bars if it starts too suddenly.")

BARE_TAIL = ("The song now ends where the {} ended in the middle of the song, just before what came "
             "next. Go on from there for a new ending, or retake its last bars.")

NO_BREAK = ("This song was sung with 'cot' set to 'off', so it has no score and no sections to put "
            "a break between. Retake a stretch of it instead.")

BREAK_WHERE = ("{} puts a break before a bar line, which 'to' names: it selects no bars and no "
               "seconds.")

BREAK_SECTION = ("{} puts a break before bar line {}, which is inside the {}. A break goes between "
                 "two sections, before the first bar of one.")

BREAK_EDGE = ("{} puts a break before bar line {}, and a break goes between two sections: after the "
              "first bar of the song and before its end. Go on past the end for a new ending.")

BREAK_BARS = (1, 16)
"""How many bars a break may have.

Measured on 2026-09-23 with four, on the stand's three songs and a user's:
bars the model writes itself after the section before, with no note to sing
in them, were played without a voice in seven takes of twelve, and the
section after went on with every word. Sixteen is a long instrumental; more
is another song."""

BREAK_LENGTH = 4
"""The bars a break has when the edit does not say."""


@dataclasses.dataclass(frozen=True)
class Edit:
    """One edit of the list: what to do, where, and which take of it was kept.

    ``bars`` counts from 0 with the second number not included, as the score
    does; ``seconds`` measures the song as the edits before this one left it.
    Exactly one of the two is given.

    ``vary`` and ``guide`` are a retake's own sampling, and ``fade`` a cut's
    own fade; each is None for "as the song was sung", which is what an edit
    written before the window could ask says. See ``VARY``, ``GUIDE`` and
    ``FADE``.

    ``lines`` and ``text`` belong to a change of words: which lines of the
    lyrics it rewrites, counted from 0 with the second number not included,
    and what they become. Both are None for anything else, but for a song
    that goes on, whose ``text`` is the words it goes on with -- "" for a new
    ending alone.

    ``score`` belongs to a change of notes: the whole score the song is to
    have, from which the music of ``bars`` is taken, or of every bar it
    changes when ``bars`` is None. It is None for anything else.

    ``to`` belongs to a move: the bar line the bars go to, counted as bars
    are, the score's bar count for after its last bar. A break has one too:
    the bar line it goes before. None for anything else.

    ``length`` belongs to a break: how many bars of playing it puts in. None
    for anything else.
    """

    op: str
    bars: tuple | None
    seconds: tuple | None
    seed: int
    takes: int
    take: int | None
    vary: float | None = None
    guide: float | None = None
    fade: float | None = None
    lines: tuple | None = None
    text: str | None = None
    score: str | None = None
    to: int | None = None
    length: int | None = None

    def seeds(self) -> tuple:
        """The seed of every take this edit asks for, counted on from its own."""
        return tuple(normalize_seed(self.seed + index) for index in range(self.takes))


@dataclasses.dataclass(frozen=True)
class State:
    """A song as the edits so far have left it: its words, its score, its bars and its length."""

    lyrics: str
    score: str
    sheet: dict | None
    clock: grid.Grid | None
    frames: int


@dataclasses.dataclass(frozen=True)
class Step:
    """One edit worked out against the song as it stands.

    ``start`` and ``stop`` are the frames it takes in hand. ``lyrics`` and
    ``score`` are what the song has after it, the same text as before for a
    retake. A song that goes on is the one exception: its score is written by
    each take for itself, so ``score`` is the part of the old one the model
    goes on writing from (``notation.head``), and ``bars`` is where that part
    ends and where the old score did. ``dropped`` names the sections a cut
    takes out, and ``notices`` what the person editing should be told about
    it. ``was`` and ``now`` are the lines a change of words swapped, or the
    lines a song goes on with, which is what the picker shows of an edit whose
    stretch alone says nothing about what was done to it.

    A move takes the frames of the bars it moves in hand, ``start`` to
    ``stop``, and says where everything goes: ``pieces`` are ``(start,
    stop)`` stretches of the old song in their new order, one for each of
    ``stretches``, the bars as ``notation.order`` reads them. ``sung`` is
    ``(start, seam)`` for each seam, in the frames of the song as the pieces
    lay it: the stretch before the seam that is sung again, the last bar of
    what comes before it (see ``SUNG_LEAST``). ``pulse`` is the grid's eighth
    note in seconds, which the beat across a seam is read by.

    A break takes nothing in hand: ``start`` and ``stop`` are the one frame
    it goes in at. Like a song that goes on, its score is written by each
    take for itself, so ``score`` is what the model writes on from
    (``notation.interlude_head``), ``bars`` the bars the break becomes in
    the score after it, and ``seam`` where it opens in units of L, which the
    notes sung after it are moved from (``notation.interluded``).
    """

    kind: str
    start: int
    stop: int
    bars: tuple | None
    lyrics: str
    score: str
    dropped: tuple
    notices: tuple
    was: tuple = ()
    now: tuple = ()
    pieces: tuple = ()
    stretches: tuple = ()
    sung: tuple = ()
    pulse: float | None = None
    seam: int | None = None


VARY = (0.0, 5.0)
"""What a retake's ``vary`` may be: the temperature an options node offers.

An edit is sung the way the song was sung, which is what keeps a retake in the
same voice as the song around it. ``vary`` is the window's way of saying
otherwise for one edit: it is the temperature that edit samples at, and the
higher it is the further the takes wander from each other and from the song."""

GUIDE = (1.0, 10.0)
"""What a retake's ``guide`` may be: the CFG scale of that edit alone, 1 being none.

A song sung without guidance can still have an edit sung with it -- the
unconditional branch is built for whatever run needs one -- which costs that
edit a second pass over the model."""

FADE = (0.0, 6.0)
"""How long a cut's fade may be, in seconds.

Only a cut that takes the first bars or the last leaves an edge with nothing
to fade into: anywhere else the two sides are already joined with a crossfade.
It is laid on the sound after the singing, so changing it costs nothing that
was already sung."""

SEED_LIMIT = 2 ** 53

SEED_RANGE = ("{}: the seed is not between 0 and 2**53 - 1, the numbers the track window can write "
              "back without changing them.")


def _whole(value, what: str) -> int:
    """``value`` as an int when it is a whole number, however JSON spelled it: 4 and 4.0 alike."""
    if isinstance(value, bool):
        raise ValueError("{} is not a whole number.".format(what))
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if not isinstance(value, int):
        raise ValueError("{} is not a whole number.".format(what))
    return value


def _range(item, index: int, op: str = "retake"):
    """The bars or the seconds one edit selects, exactly one of the two.

    A change of words may select neither, and then the stretch it is sung in
    is the one its own lines are sung in; see ``needs_times``. So may a
    change of notes, which is then sung over the bars whose notes changed.
    """
    where = "Edit {}".format(index + 1)
    bars, seconds = item.get("bars"), item.get("seconds")
    if op == "extend":
        if bars is not None or seconds is not None:
            raise ValueError(EXTEND_WHERE.format(where))
        return None, None
    if op == "break":
        if bars is not None or seconds is not None:
            raise ValueError(BREAK_WHERE.format(where))
        return None, None
    if op == "move" and (bars is None or seconds is not None):
        raise ValueError(MOVE_WHERE.format(where))
    if op in ("words", "notes") and bars is None and seconds is None:
        return None, None
    if op == "notes" and seconds is not None:
        raise ValueError(SECONDS_NOTES.format(where))
    if (bars is None) == (seconds is None):
        raise ValueError("{} selects neither bars nor seconds, or both at once. It takes one of "
                         "them.".format(where))
    chosen = bars if seconds is None else seconds
    if not isinstance(chosen, (list, tuple)) or len(chosen) != 2:
        raise ValueError("{} selects {}, which is not a pair.".format(
            where, "bars" if seconds is None else "seconds"))
    if seconds is None:
        first = _whole(chosen[0], "{}: the first bar".format(where))
        stop = _whole(chosen[1], "{}: the last bar".format(where))
        if not 0 <= first < stop:
            raise ValueError("{} selects bars {} to {}, which is not a stretch of a score.".format(
                where, first + 1, stop))
        return (first, stop), None
    pair = []
    for value, which in zip(chosen, ("first", "last")):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError("{}: the {} second is not a number.".format(where, which))
        pair.append(float(value))
    if not 0.0 <= pair[0] < pair[1]:
        raise ValueError("{} selects {:.2f} to {:.2f} seconds, which is not a stretch of a "
                         "song.".format(where, pair[0], pair[1]))
    return None, (pair[0], pair[1])


def _measure(item, key: str, where: str, what: str, span):
    """One number an edit may carry, checked; None when the edit does not say."""
    value = item.get(key)
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError("{}: {} is not a number.".format(where, what))
    if not span[0] <= float(value) <= span[1]:
        raise ValueError("{}: {} is {:g}, and it is between {:g} and {:g}.".format(
            where, what, float(value), span[0], span[1]))
    return float(value)


def _rewrite(item, where: str):
    """The lines a change of words rewrites and what they become, checked.

    What may be written there is ``ops.change_words``' business, which reads
    it against the words the song has; this is only the shape of the two
    fields, so that a list can be read without a song to read it against.
    """
    lines = item.get("lines")
    if not isinstance(lines, (list, tuple)) or len(lines) != 2:
        raise ValueError("{} does not say which lines of the words it rewrites.".format(where))
    first = _whole(lines[0], "{}: the first line".format(where))
    stop = _whole(lines[1], "{}: the last line".format(where))
    if not 0 <= first < stop:
        raise ValueError("{} rewrites lines {} to {}, which is not a stretch of words.".format(
            where, first + 1, stop))
    words = item.get("text")
    if not isinstance(words, str):
        raise ValueError("{} holds no new words to sing.".format(where))
    return (first, stop), words


def _new_score(item, where: str) -> str:
    """The score a change of notes carries, checked for being text a score could be.

    Whether it is a score of this song is ``notation.changed``'s business,
    which reads it against the one the song has.
    """
    score = item.get("score")
    if not isinstance(score, str) or not score.strip():
        raise ValueError("{} holds no score to sing.".format(where))
    if len(score) > notation.LONGEST:
        raise ValueError("{} holds a score far longer than any song.".format(where))
    return score


def _going(item, where: str) -> str:
    """The words a song goes on with, checked for being text: none at all is a new ending alone.

    What they may say is ``ops.go_on``'s business.
    """
    words = item.get("text", "")
    if words is None:
        return ""
    if not isinstance(words, str):
        raise ValueError("{} holds new words that are not text.".format(where))
    if len(words) > ops.LONGEST_WORDS:
        raise ValueError("{} goes on with more than {} characters of words.".format(
            where, ops.LONGEST_WORDS))
    return words


def _breaking(item, where: str) -> tuple:
    """The bar line a break goes before and how many bars it has, checked.

    Whether a section starts at that line is ``_break``'s business, which
    reads it against the song's score.
    """
    if item.get("to") is None:
        raise ValueError(BREAK_WHERE.format(where))
    to = _whole(item.get("to"), "{}: the bar line the break goes before".format(where))
    if to < 0:
        raise ValueError("{} puts a break before bar line {}, which is before the song.".format(
            where, to + 1))
    length = _whole(item.get("length", BREAK_LENGTH), "{}: the bars of the break".format(where))
    if not BREAK_BARS[0] <= length <= BREAK_BARS[1]:
        raise ValueError("{} asks for a break of {} bars; it has {} to {}.".format(
            where, length, BREAK_BARS[0], BREAK_BARS[1]))
    return to, length


def _edit(item, index: int, takes: int) -> Edit:
    where = "Edit {}".format(index + 1)
    if not isinstance(item, dict):
        raise ValueError("{} is not an object with an 'op' in it.".format(where))
    op = item.get("op")
    if op not in OPS:
        raise ValueError("{} asks for '{}', which is not something an edit does. It is 'retake', "
                         "'cut', 'words', 'notes', 'extend', 'move' or 'break'.".format(where, op))
    bars, seconds = _range(item, index, op)
    if op == "cut":
        return Edit(op=op, bars=bars, seconds=seconds, seed=0, takes=1, take=None,
                    fade=_measure(item, "fade", where, "the fade", FADE))
    if op == "move":
        if item.get("to") is None:
            raise ValueError(MOVE_WHERE.format(where))
        to = _whole(item.get("to"), "{}: the bar line the bars go to".format(where))
        if to < 0:
            raise ValueError("{} moves bars to bar line {}, which is before the song.".format(
                where, to + 1))
        return Edit(op=op, bars=bars, seconds=None, seed=0, takes=1, take=None, to=to)
    seed = _whole(item.get("seed", 0), "{}: the seed".format(where))
    if not 0 <= seed < SEED_LIMIT:
        raise ValueError(SEED_RANGE.format(where))
    seed = normalize_seed(seed)
    wanted = _whole(item.get("takes", takes), "{}: the number of takes".format(where))
    if not 1 <= wanted <= MAX_TAKES:
        raise ValueError("{} asks for {} takes; between 1 and {} can be sung.".format(
            where, wanted, MAX_TAKES))
    take = item.get("take")
    if take is not None:
        take = _whole(take, "{}: the take kept".format(where))
        if not 0 <= take < wanted:
            raise ValueError("{} keeps take {} of {}.".format(where, take + 1, wanted))
    lines, words, score, to, length = None, None, None, None, None
    if op == "words":
        lines, words = _rewrite(item, where)
    if op == "notes":
        score = _new_score(item, where)
    if op == "extend":
        words = _going(item, where)
    if op == "break":
        to, length = _breaking(item, where)
    return Edit(op=op, bars=bars, seconds=seconds, seed=seed, takes=wanted, take=take,
                vary=_measure(item, "vary", where, "the variety", VARY),
                guide=_measure(item, "guide", where, "the guide", GUIDE),
                lines=lines, text=words, score=score, to=to, length=length)


def read(text, takes: int = 1) -> tuple:
    """The edits written in ``text``, checked; no text at all is no edits.

    ``takes`` is how many takes a retake asks for when it does not say, which
    is the node's own widget.
    """
    if text is None:
        return ()
    if not isinstance(text, str):
        raise ValueError(NOT_A_LIST)
    body = text.lstrip("\ufeff").strip()
    if not body:
        return ()
    try:
        found = json.loads(body)
    except ValueError:
        raise ValueError(NOT_A_LIST)
    if not isinstance(found, list):
        raise ValueError(NOT_A_LIST)
    return tuple(_edit(item, index, takes) for index, item in enumerate(found))


def written(edits) -> str:
    """The edits as the field holds them, so a node can hand back the list it made."""
    items = []
    for edit in edits:
        item = {"op": edit.op}
        if edit.bars is not None:
            item["bars"] = [edit.bars[0], edit.bars[1]]
        elif edit.seconds is not None:
            item["seconds"] = [edit.seconds[0], edit.seconds[1]]
        if edit.op == "words":
            item["lines"] = [edit.lines[0], edit.lines[1]]
            item["text"] = edit.text
        if edit.op == "notes":
            item["score"] = edit.score
        if edit.op == "extend":
            item["text"] = edit.text
        if edit.op == "move":
            item["to"] = edit.to
        if edit.op == "break":
            item["to"] = edit.to
            item["length"] = edit.length
        if edit.op in ("retake", "words", "notes", "extend", "break"):
            item["seed"] = edit.seed
            item["takes"] = edit.takes
            if edit.take is not None:
                item["take"] = edit.take
            if edit.vary is not None:
                item["vary"] = edit.vary
            if edit.guide is not None:
                item["guide"] = edit.guide
        elif edit.fade is not None:
            item["fade"] = edit.fade
        items.append(item)
    return json.dumps(items)


def opened(song, clock, sheet=None) -> State:
    """The song before any edit: its own words and score, and the grid measured on it.

    ``sheet`` is the score as ``notation.read`` gives it, when the caller has
    read it already; without it the score is read here.
    """
    if clock is None or not song.score:
        sheet = None
    elif sheet is None:
        sheet = notation.read(song.score)
    return State(lyrics=song.lyrics, score=song.score, sheet=sheet,
                 clock=None if sheet is None else clock, frames=song.frames)


def _frames(state: State, edit: Edit, span=None):
    """The frames one edit takes in hand: the span found for it, its bars, or its seconds.

    ``span`` is a pair of frames worked out elsewhere -- for a change of words
    that names no bars, the stretch its own lines are sung in, which only the
    song's word times can say. It is clipped to the song, because word times
    are measured on sound and sound ends.
    """
    if span is not None:
        start, stop = int(span[0]), int(span[1])
        start, stop = max(0, min(state.frames, start)), max(0, min(state.frames, stop))
        if not start < stop:
            raise ValueError("The words chosen are not sung anywhere in this song.")
        return start, stop
    if edit.bars is None and edit.seconds is None:
        raise ValueError(NO_TIMES)
    if edit.bars is None:
        start = int(round(edit.seconds[0] / FRAME_SECONDS))
        stop = int(round(edit.seconds[1] / FRAME_SECONDS))
        start, stop = max(0, min(state.frames, start)), max(0, min(state.frames, stop))
        if not start < stop:
            raise ValueError("Seconds {:.2f} to {:.2f} are not a stretch of this song, which is "
                             "{:.2f} seconds long.".format(edit.seconds[0], edit.seconds[1],
                                                           state.frames * FRAME_SECONDS))
        return start, stop
    if state.sheet is None or state.clock is None:
        raise ValueError(NO_SCORE)
    first, stop = edit.bars
    marks = grid.seams(state.sheet)
    if edit.op == "cut":
        return grid.cut_frames(state.sheet, state.clock, marks, first, stop, state.frames)
    return grid.retake_frames(state.sheet, state.clock, marks, first, stop, state.frames)


def _notes(state: State, edit: Edit) -> Step:
    """A change of notes worked out: the stretch it sings again, and the score the song has after it.

    The stretch is the bars the edit names, or else the first to the last
    bar whose notes its score changes, grown over any note held across its
    edges (``notation.taken``). It opens where either score's phrases say --
    the edited notes may start their phrase earlier than the old ones did --
    and closes where the old score's say, because what follows is the song
    as it was sung.
    """
    if state.sheet is None or state.clock is None:
        raise ValueError(NO_NOTES)
    if edit.bars is None:
        found = notation.changed(state.score, edit.score)
        if not found:
            raise ValueError(SAME_SCORE)
        first, stop = found[0], found[-1] + 1
    else:
        first, stop = edit.bars
    laid = notation.taken(state.score, edit.score, first, stop)
    first, stop = laid["bars"]
    if edit.bars is not None and not notation.changed(state.score, laid["abc"]):
        raise ValueError(notation.SAME_NOTES.format(first=first + 1, stop=stop))
    sheet = notation.read(laid["abc"])
    start, end = grid.retake_frames(state.sheet, state.clock, grid.seams(state.sheet), first, stop,
                                    state.frames)
    earlier, _end = grid.retake_frames(sheet, state.clock, grid.seams(sheet), first, stop,
                                       state.frames)
    return Step(kind="notes", start=min(start, earlier), stop=end, bars=(first, stop),
                lyrics=state.lyrics, score=laid["abc"], dropped=(), notices=())


SOUND_REACHED = 0.5
"""Seconds a bar line must lie inside the song for the song to go on from it.

A song can stop short of its score, and the bars it never reached are not
something to go on from: the model writes those again, with the rest."""


def going_on_from(state: State):
    """The bar a song goes on from, and the second it opens at; None for a song with no score.

    That is the first bar after the last sung note ends -- whatever plays
    after it, an outro or a last chord, is the ending the new part replaces --
    or the last bar line the song reached, when it stopped short of its score.
    A score with nothing sung goes on after its last bar.
    """
    if state.sheet is None or state.clock is None:
        return None
    sheet, clock = state.sheet, state.clock
    count = len(sheet["bars"])
    starts = grid.starts_of(sheet)
    vocal = sheet["notes"]["Vocal"]
    bar = count
    if vocal:
        end = max(note["start"] + note["length"] for note in vocal)
        bar = next((number for number in range(1, count + 1) if starts[number] >= end), count)
    song = state.frames * FRAME_SECONDS
    while bar > 1 and clock.at(bar) > song - SOUND_REACHED:
        bar -= 1
    if bar < count:
        start, _stop = grid.retake_frames(sheet, clock, grid.seams(sheet), bar, count, state.frames)
    else:
        start = max(0, min(state.frames - 1, clock.moment(count)))
    return {"bar": bar, "bars": count, "start": start}


def _extend(state: State, edit: Edit) -> Step:
    """A song that goes on, worked out: where it goes on from, what the model writes on from, the words.

    The score the model goes on from is the old one up to the bar the song
    goes on from. The sections that start between the one the last lines are
    sung in and that bar lose their names there -- the model often marks the
    tail of a last chorus as its outro, and a score ending in an outro is
    finished by writing more outro, not by singing what comes next.
    """
    found = going_on_from(state)
    if found is None:
        raise ValueError(NO_GOING_ON)
    bar, count, start = found["bar"], found["bars"], found["start"]
    going = ops.go_on(state.lyrics, edit.text or "")
    last = ops.last_sung(state.lyrics, state.score)
    merged = () if last is None else tuple(
        section["bar"] for section in state.sheet["sections"] if last < section["bar"] < bar)
    notices = []
    if going.tagged:
        notices.append(("notice", TAGGED.format(ops.GO_ON_TAG)))
    return Step(kind="extend", start=start, stop=state.frames, bars=(bar, count),
                lyrics=going.text, score=notation.head(state.score, bar, merged), dropped=(),
                notices=tuple(notices), now=going.sung)


def extension_frames(state: State, step: Step, sheet) -> int:
    """Frames the song goes on for under ``sheet``, a score one of its takes wrote: from ``step.start`` to its end.

    The new bars last what the song's own tempo gives them, and the song
    keeps the tail it had past its score's last bar -- the ring of a last
    chord -- which the model sings again at the new end.
    """
    clock = state.clock
    bar, count = step.bars
    starts = grid.starts_of(sheet)
    tail = max(0, state.frames - clock.moment(count))
    return max(1, clock.moment(bar) - step.start + clock.frames(starts[-1] - starts[bar]) + tail)


def sings_from(state: State, step: Step, sheet) -> float:
    """The second a take of a song going on is heard from for its new words: a beat before its first new note.

    ``sheet`` is the score the take wrote, and its new bars fall at the song's
    own tempo from the bar the song goes on from, as ``grid.after_extend``
    lays them. The aligner, given a stretch that opens on seconds of playing,
    puts the first word where the stretch opens: a bridge after four bars of
    riff was lit six seconds early on a user's song, 2026-09-23. The takes of
    the stand's extensions and that song sang their first new note up to
    0.6 s ahead of where their scores have it and not behind it, so a beat
    early catches the singer who comes in first and leaves little playing to
    open on. Never before the new part begins, and that is where a take that
    sings no new note is heard from.
    """
    opening = step.start * FRAME_SECONDS
    clock = state.clock
    bar = step.bars[0]
    starts = grid.starts_of(sheet)
    if clock is None or bar >= len(starts):
        return opening
    notes = [note["start"] for note in sheet["notes"]["Vocal"] if note["start"] >= starts[bar]]
    if not notes:
        return opening
    unit = clock.rate * clock.tick
    first = clock.at(bar) + (min(notes) - starts[bar]) * unit
    return max(opening, first - float(sheet["per_quarter"]) * unit)


SUNG_LEAST = 25
"""The fewest frames before a seam of a move that are sung again: a second.

What is sung again is the last bar before the seam, from where a retake of
that bar would open to the seam itself -- often less than the bar, since the
seam lies ahead of its bar line by where the next section's first phrase
begins. On 2026-09-23 that was a second to three and a half on the stand's
songs and a user's. A seam closer to the one before it than this is sung
from that one on."""


def _sung(sheet, clock, pieces, stretches) -> tuple:
    """``(start, seam)`` for every seam of a move: the frames before it that are sung again.

    ``sheet`` and ``clock`` are the moved song's, its score read and its grid
    laid by ``grid.after_move``.
    """
    marks = grid.seams(sheet)
    beat, margin = grid._beat_and_margin(sheet, clock)
    found = []
    frame, bar, last = 0, 0, 0
    for index, ((low, high), (start, stop)) in enumerate(zip(stretches, pieces)):
        if index and pieces[index - 1][1] != start:
            opens = 0 if bar <= 1 else clock.moment(bar - 1, grid.opening(marks[bar - 1], beat, margin))
            opens = max(last, min(opens, frame - SUNG_LEAST))
            if opens < frame:
                found.append((opens, frame))
            last = frame
        frame += stop - start
        bar += high - low
    return tuple(found)


def _move(state: State, edit: Edit, level=None) -> Step:
    """A move worked out: the pieces of the song in their new order, and the words and score after it.

    Whole sections go, so that the words can go with them: the bars moved
    start and end where sections do, and they go to where another section
    starts or the song ends. Every bar line the move touches is cut at one
    shift before its downbeat (``grid.move_frames``), which keeps the beat
    whole across all three seams; ``level`` is the separated voice's
    loudness, which moves that shift to where the voice rests, when it is
    known (``grid.voiced_shift``). The last bar before each seam is sung
    again (``_sung``).
    """
    if state.sheet is None or state.clock is None:
        raise ValueError(NO_MOVE)
    sheet = state.sheet
    first, stop = edit.bars
    count = len(sheet["bars"])
    grid.bars_of(sheet, first, stop)
    if edit.to > count:
        raise ValueError("The bars are moved to bar line {}, and this score has {} bars.".format(
            edit.to + 1, count))
    named = {section["bar"]: section["name"] for section in sheet["sections"]}
    for bar in (first, stop, edit.to):
        if bar != count and bar not in named:
            inside = next(section["name"] for section in sheet["sections"]
                          if section["bar"] < bar < section["bar"] + section["bars"])
            raise ValueError(NOT_SECTIONS.format("The edit", first + 1, stop, edit.to + 1,
                                                 bar + 1, inside))
    stretches = notation.order(first, stop, edit.to, count)
    score = notation.moved(state.score, first, stop, edit.to)
    pieces = grid.move_frames(sheet, state.clock, grid.seams(sheet), stretches, state.frames,
                              level)
    start, end = pieces[stretches.index((first, stop))]
    if not start < end:
        raise ValueError(grid.OUTSIDE.format(first + 1, stop))
    words = ops.move_words(state.lyrics, state.score, first, stop, edit.to)
    notices = []
    if not words.matched:
        notices.append(("warn", UNMATCHED))
    if words.tagged:
        notices.append(("notice", UNTAGGED.format(ops.GO_ON_TAG)))
    sounding = [(piece, stretch) for piece, stretch in zip(pieces, stretches)
                if piece[1] > piece[0]]
    if sounding[0][0][0] > 0:
        notices.append(("notice", BARE_HEAD.format(named.get(sounding[0][1][0], "section"))))
    if sounding[-1][0][1] < state.frames:
        before = [name for bar, name in sorted(named.items()) if bar < sounding[-1][1][1]]
        notices.append(("notice", BARE_TAIL.format(before[-1] if before else "section")))
    moved = notation.read(score)
    clock = grid.after_move(state.clock, grid.starts_of(moved), stretches, pieces)
    pulse = state.clock.rate * state.clock.tick * float(sheet["per_quarter"]) / 2
    return Step(kind="move", start=start, stop=end, bars=(first, stop), lyrics=words.text,
                score=score, dropped=(), notices=tuple(notices), pieces=tuple(pieces),
                stretches=tuple(stretches), sung=_sung(moved, clock, pieces, stretches),
                pulse=pulse)


def _break(state: State, edit: Edit) -> Step:
    """A break worked out: where it goes in, what the model writes it from, and the words around it.

    It goes before a bar line where a section starts, and in at the frame a
    retake of that section would open at (``grid.opening``): as far ahead of
    the line as the section's first phrase begins, so that phrase is heard
    after the break and not cut off before it. The score the model writes on
    from is the old one up to that line, less the notes of that phrase
    (``notation.interlude_head``); the words get an empty ``ops.INTERLUDE_TAG``
    where the break is (``ops.break_words``).
    """
    if state.sheet is None or state.clock is None:
        raise ValueError(NO_BREAK)
    sheet = state.sheet
    count = len(sheet["bars"])
    where = "The edit"
    if not 0 < edit.to < count:
        raise ValueError(BREAK_EDGE.format(where, edit.to + 1))
    named = {section["bar"] for section in sheet["sections"]}
    if edit.to not in named:
        inside = next(section["name"] for section in sheet["sections"]
                      if section["bar"] < edit.to < section["bar"] + section["bars"])
        raise ValueError(BREAK_SECTION.format(where, edit.to + 1, inside))
    beat, margin = grid._beat_and_margin(sheet, state.clock)
    opened = grid.opening(grid.seams(sheet)[edit.to], beat, margin)
    start = state.clock.moment(edit.to, opened)
    if not 0 < start < state.frames:
        raise ValueError("Bar line {} lies outside the song as it was sung.".format(edit.to + 1))
    seam = int(round(sheet["bars"][edit.to]["start"] - opened))
    words = ops.break_words(state.lyrics, state.score, edit.to)
    notices = () if words.matched else (("warn", UNMATCHED),)
    return Step(kind="break", start=start, stop=start, bars=(edit.to, edit.to + edit.length),
                lyrics=words.text, score=notation.interlude_head(state.score, edit.to, seam),
                dropped=(), notices=notices, seam=seam)


def break_frames(state: State, step: Step, sheet) -> int:
    """Frames the break lasts under ``sheet``, a score one of its takes wrote: its bars at the song's own tempo."""
    first, stop = step.bars
    starts = grid.starts_of(sheet)
    return max(1, state.clock.frames(starts[stop] - starts[first]))


def placed(step: Step, sung=()) -> tuple:
    """The frames the bars a move took now lie at, in the song after it.

    ``sung`` is ``core.Take.sung`` of the move: the stretches before its
    seams that were sung again, which move what comes after them by what
    they gained or lost.
    """
    at = 0
    for (start, stop), stretch in zip(step.pieces, step.stretches):
        if tuple(stretch) == tuple(step.bars):
            return shifted(at, sung), shifted(at + stop - start, sung)
        at += stop - start
    raise ValueError("That edit is not a move.")


def shifted(frame: int, sung) -> int:
    """Where ``frame`` of a moved song lies once the ``(start, stop, count)`` stretches ``sung`` were sung again, in order.

    A frame after a stretch moves by what the stretch gained or lost, and one
    inside it is spread through the new frames in proportion.
    """
    for start, stop, count in sung:
        if frame >= stop:
            frame += count - (stop - start)
        elif frame > start:
            frame = start + int(round((frame - start) * count / float(stop - start)))
    return int(frame)


def needs_times(edit: Edit) -> bool:
    """Whether the stretch this edit sings has to be found from the song's own word times."""
    return edit.op == "words" and edit.bars is None and edit.seconds is None


def plan(state: State, edit: Edit, span=None, level=None) -> Step:
    """What ``edit`` does to the song ``state`` describes: its frames, and the words and score after it.

    ``span`` is the pair of frames found for an edit that names no bars and no
    seconds; see ``_frames``. A change of notes finds its own; see ``_notes``.
    ``level`` is how loud the separated voice is, for a move; see ``_move``.
    """
    if edit.op == "notes":
        return _notes(state, edit)
    if edit.op == "extend":
        return _extend(state, edit)
    if edit.op == "move":
        return _move(state, edit, level)
    if edit.op == "break":
        return _break(state, edit)
    start, stop = _frames(state, edit, span)
    if edit.op == "words":
        rewrite = ops.change_words(state.lyrics, edit.lines[0], edit.lines[1], edit.text)
        notices = []
        old, new = rewrite.syllables
        if abs(new - old) >= 2 and abs(new - old) * 4 > old:
            notices.append(("notice", FITS.format(new, old)))
        return Step(kind="words", start=start, stop=stop, bars=edit.bars, lyrics=rewrite.text,
                    score=state.score, dropped=(), notices=tuple(notices),
                    was=rewrite.before, now=rewrite.after)
    if edit.op == "retake":
        return Step(kind="retake", start=start, stop=stop, bars=edit.bars, lyrics=state.lyrics,
                    score=state.score, dropped=(), notices=())
    if edit.bars is None:
        if state.score:
            raise ValueError(SECONDS_CUT)
        return Step(kind="cut", start=start, stop=stop, bars=None, lyrics=state.lyrics,
                    score=state.score, dropped=(), notices=())
    first, last = edit.bars
    notices = []
    count = len(state.sheet["bars"])
    if stop >= state.frames and last < count:
        notices.append(("notice", "The cut reaches the end of the song, which stops short of "
                                  "its score, so bars {} to {} go with it.".format(
                                      last + 1, count)))
        last = count
    score = notation.without(state.score, first, last)
    words = ops.cut_words(state.lyrics, state.score, first, last)
    if words.dropped:
        notices.append(("notice", "The cut takes the words of {} out of the song.".format(
            ", ".join(tag or "the lines before the first tag" for tag in words.dropped))))
    elif not words.matched:
        notices.append(("warn", UNMATCHED))
    return Step(kind="cut", start=start, stop=stop, bars=(first, last), lyrics=words.text,
                score=score, dropped=words.dropped, notices=tuple(notices))


def after(state: State, step: Step, count: int, score: str | None = None, sung=()) -> State:
    """The song an edit leaves, once its new part has come out ``count`` frames long.

    A cut is ``count`` 0. The bars of the score move with the sound: a cut
    takes its own out and pulls the rest back, and a retake that came out
    longer or shorter than what it replaced pushes everything after it along.
    A change of notes leaves the bars where a retake would, and its notes in
    them. A song that goes on has the score its take wrote, ``score``: the
    old bars where they were, and the new ones after them. A move lays the
    bars with their pieces, and then as ``sung`` left them: each stretch
    before a seam that was sung again, ``core.Take.sung``, is a retake of
    its own. A break has the score its take wrote too, ``score``: its bars go
    in before the section it was put before, and everything after moves by
    ``count``.
    """
    frames = state.frames - (step.stop - step.start) + count
    if step.kind == "move":
        sheet = notation.read(step.score)
        clock = grid.after_move(state.clock, grid.starts_of(sheet), step.stretches, step.pieces)
        frames = state.frames
        for start, stop, made in sung:
            clock = grid.after_retake(clock, start, stop, made)
            frames += made - (stop - start)
        return State(lyrics=step.lyrics, score=step.score, sheet=sheet, clock=clock,
                     frames=frames)
    if step.kind == "extend":
        sheet = notation.read(score)
        clock = grid.after_extend(state.clock, grid.starts_of(sheet), step.bars[0])
        return State(lyrics=step.lyrics, score=score, sheet=sheet, clock=clock, frames=frames)
    if step.kind == "break":
        sheet = notation.read(score)
        first, stop = step.bars
        clock = grid.after_break(state.clock, grid.starts_of(sheet), first, stop - first, count)
        return State(lyrics=step.lyrics, score=score, sheet=sheet, clock=clock, frames=frames)
    if step.kind == "cut" and step.bars is not None:
        sheet = notation.read(step.score)
        clock = grid.after_cut(state.clock, grid.starts_of(sheet), step.bars[0], step.bars[1],
                               step.start, step.stop - step.start)
        return State(lyrics=step.lyrics, score=step.score, sheet=sheet, clock=clock, frames=frames)
    clock = state.clock if state.clock is None else grid.after_retake(
        state.clock, step.start, step.stop, count)
    sheet = notation.read(step.score) if step.kind == "notes" else state.sheet
    return State(lyrics=step.lyrics, score=step.score, sheet=sheet, clock=clock, frames=frames)


def _spelled(edit: Edit, seed) -> dict:
    """One edit as the name of a result spells it: what it does and where, and which take was kept.

    How it was sampled belongs here -- takes sung at another temperature are
    other takes -- while a cut's fade does not: it is laid on the sound
    afterwards, so moving it sings nothing again.
    """
    item = {"op": edit.op, "bars": list(edit.bars) if edit.bars else None,
            "seconds": list(edit.seconds) if edit.seconds else None, "seed": edit.seed}
    if edit.op == "words":
        item["lines"] = list(edit.lines)
        item["text"] = edit.text
    if edit.op == "notes":
        item["score"] = edit.score
    if edit.op == "extend":
        item["text"] = edit.text
    if edit.op == "move":
        item["to"] = edit.to
    if edit.op == "break":
        item["to"] = edit.to
        item["length"] = edit.length
    for key, value in (("vary", edit.vary), ("guide", edit.guide)):
        if value is not None:
            item[key] = value
    if seed is not None:
        item["kept"] = int(seed)
    return item


def sound_name(song: str, history) -> str:
    """The name the sound the edits so far have made is known by.

    Anything measured on that sound rather than sung into it -- when each word
    of it is sung, say -- is kept under this, because the same edits on the
    same song make the same sound.
    """
    body = json.dumps({"song": song, "made": [_spelled(made, seed) for made, seed in history]},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def name(song: str, history, edit: Edit) -> str:
    """The name the takes of ``edit`` are remembered under, after the edits in ``history``.

    ``history`` is the edits already made, each with the seed of the take that
    was kept, because that take is the song this one is made on. How many
    takes were asked for is left out: asking for another one sings that one
    alone and leaves the rest where they are.
    """
    spelled = {"song": song, "made": [_spelled(made, seed) for made, seed in history],
               "next": _spelled(edit, None)}
    body = json.dumps(spelled, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()
