"""The list of edits a track window writes, and what each one does to the song under it.

An edit is a plain object in a JSON list, because that list has to survive
being saved inside a workflow, sent through ComfyUI's queue and read back by
both the window and the node. This module is the one place that knows its
shape, so the two cannot drift apart, and the tests read exactly what the
window writes::

    [{"op": "retake", "bars": [12, 16], "seed": 831001, "takes": 2, "take": 0},
     {"op": "cut", "bars": [20, 24]},
     {"op": "retake", "seconds": [48.0, 64.0], "seed": 7}]

Bars count from 0 and the second number is not included: [12, 16] is bars 13
to 16 as the score numbers them. Seconds are read as they are written.

Edits are made in order, each one on the song the ones before it left. A
selection is either bars of the score, which is how the window offers it, or
plain seconds, which is all a song sung with 'cot' off has: no score means no
bars. A retake sings from its own seed; asking for more takes sings the same
edit from seeds counted on from it, and 'take' says which one was kept. A cut
has neither, being the same cut however often it is made.

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

OPS = ("retake", "cut")

NOT_A_LIST = ("The edits field does not hold a JSON list. The track window writes it -- open the "
              "track, or clear the field to start again.")

NO_SCORE = ("This song was sung with 'cot' set to 'off', so it has no score and no bars. Select "
            "the seconds to edit instead.")

SECONDS_CUT = ("A cut of a song that has a score goes by whole bars, so that its score and its "
               "words can be cut with it. Select bars rather than seconds.")

UNMATCHED = ("The sections of the lyrics could not be matched with the score's, so the words are "
             "left as they are. Read them before singing more of this song.")


@dataclasses.dataclass(frozen=True)
class Edit:
    """One edit of the list: what to do, where, and which take of it was kept.

    ``bars`` counts from 0 with the second number not included, as the score
    does; ``seconds`` measures the song as the edits before this one left it.
    Exactly one of the two is given.
    """

    op: str
    bars: tuple | None
    seconds: tuple | None
    seed: int
    takes: int
    take: int | None

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
    retake. ``dropped`` names the sections a cut takes out, and ``notices``
    what the person editing should be told about it.
    """

    kind: str
    start: int
    stop: int
    bars: tuple | None
    lyrics: str
    score: str
    dropped: tuple
    notices: tuple


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


def _range(item, index: int):
    """The bars or the seconds one edit selects, exactly one of the two."""
    where = "Edit {}".format(index + 1)
    bars, seconds = item.get("bars"), item.get("seconds")
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


def _edit(item, index: int, takes: int) -> Edit:
    where = "Edit {}".format(index + 1)
    if not isinstance(item, dict):
        raise ValueError("{} is not an object with an 'op' in it.".format(where))
    op = item.get("op")
    if op not in OPS:
        raise ValueError("{} asks for '{}', which is not something an edit does. It is 'retake' "
                         "or 'cut'.".format(where, op))
    bars, seconds = _range(item, index)
    if op == "cut":
        return Edit(op=op, bars=bars, seconds=seconds, seed=0, takes=1, take=None)
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
    return Edit(op=op, bars=bars, seconds=seconds, seed=seed, takes=wanted, take=take)


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
        else:
            item["seconds"] = [edit.seconds[0], edit.seconds[1]]
        if edit.op == "retake":
            item["seed"] = edit.seed
            item["takes"] = edit.takes
            if edit.take is not None:
                item["take"] = edit.take
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


def _frames(state: State, edit: Edit):
    """The frames one edit takes in hand, from its bars or from its seconds."""
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


def plan(state: State, edit: Edit) -> Step:
    """What ``edit`` does to the song ``state`` describes: its frames, and the words and score after it."""
    start, stop = _frames(state, edit)
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


def after(state: State, step: Step, count: int) -> State:
    """The song an edit leaves, once its new part has come out ``count`` frames long.

    A cut is ``count`` 0. The bars of the score move with the sound: a cut
    takes its own out and pulls the rest back, and a retake that came out
    longer or shorter than what it replaced pushes everything after it along.
    """
    frames = state.frames - (step.stop - step.start) + count
    if step.kind == "cut" and step.bars is not None:
        sheet = notation.read(step.score)
        clock = grid.after_cut(state.clock, grid.starts_of(sheet), step.bars[0], step.bars[1],
                               step.start, step.stop - step.start)
        return State(lyrics=step.lyrics, score=step.score, sheet=sheet, clock=clock, frames=frames)
    clock = state.clock if state.clock is None else grid.after_retake(
        state.clock, step.start, step.stop, count)
    return State(lyrics=step.lyrics, score=step.score, sheet=state.sheet, clock=clock,
                 frames=frames)


def _spelled(edit: Edit, seed) -> dict:
    """One edit as the name of a result spells it: what it does and where, and which take was kept."""
    item = {"op": edit.op, "bars": list(edit.bars) if edit.bars else None,
            "seconds": list(edit.seconds) if edit.seconds else None, "seed": edit.seed}
    if seed is not None:
        item["kept"] = int(seed)
    return item


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
