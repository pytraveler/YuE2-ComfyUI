"""The plan of an edit: which frames go, how many new ones come, whose noise they get, what the words become.

An edit replaces the old frames ``start`` to ``stop`` of a song with new ones.
A retake sings the same stretch again with the same words and lets the join
choose how many frames it takes; a cut takes the stretch out, and only the seam
where the two sides now meet is drawn again. Everything here is arithmetic on
frames, noise runs and text. ``core`` does what needs the model.

The numbers are the ones the inpainting stand settled on and measured on
2026-09-19, three songs by three seeds on an RTX 5090: every retake sang its
words, the join the model chose was within two frames of the length the bars
asked for, and nothing outside an edit changed.
"""

from __future__ import annotations

import dataclasses

from .. import notation, phrasing
from ..constants import CONTEXT

MARGIN = 12
"""Frames drawn again on each side of the new ones, 0.48 s: the seam itself.

Every other old frame is held on its old latent, so what the acoustic stage
may change is the new part and these twelve frames at each end of it."""

JOIN_FRAMES = 75
"""Old frames after an edit that judge where the new part should end: 3 s.

For every length the search allows, the model scores how well these frames
follow the new part, and the best-scored length wins. On the stand the winner
was within two frames of the length the bars asked for in every run."""

WIDTH = 50
"""How far the length of the new part may move from the length selected: 2 s either way.

A short selection moves at most a third of itself, so a one-bar retake cannot
come back as a bar and a half."""

FADE_SAMPLES = 4800
"""The crossfade at each end of the decoded window, 0.1 s at 48 kHz.

It lies where the old and the new latents are the same, so what it blends is
one sound decoded twice, which differs only in rounding."""


@dataclasses.dataclass(frozen=True)
class Region:
    """Old frames ``start`` to ``stop`` replaced by about ``length`` new ones.

    ``width`` is how far the number of new frames may move from ``length``
    either way to put the join where the old song goes on best; at 0 it is
    exactly ``length``. A cut asks for no new frames at all.
    """

    start: int
    stop: int
    length: int
    width: int

    @property
    def removed(self) -> int:
        return self.stop - self.start

    @property
    def shortest(self) -> int:
        """The fewest new frames the join may choose: never none, for an edit that sings."""
        return max(1, self.length - self.width) if self.length else 0

    @property
    def longest(self) -> int:
        return self.length + self.width


def _within(start, stop, frames) -> None:
    """Refuse a stretch that is not inside the song, with a ValueError saying so."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, stop, frames)):
        raise ValueError("An edit is measured in whole frames of the song.")
    if not 0 <= start < stop <= frames:
        raise ValueError("Frames {} to {} are not a stretch of this song, which has {} frames."
                         .format(start, stop, frames))


def retake(start: int, stop: int, frames: int, prompt_tokens: int) -> Region:
    """The region of a retake of frames ``start`` to ``stop`` of a song of ``frames`` frames.

    ``prompt_tokens`` is the length of the song's prompt. The context the model
    sings on from is that prompt and every frame before ``start``, and it has to
    hold the new part at its longest as well; a song that fitted its own context
    always leaves room for the length selected, and the search narrows to what
    is left over.
    """
    _within(start, stop, frames)
    length = stop - start
    room = CONTEXT - int(prompt_tokens) - start - length
    if room < 0:
        raise ValueError("This song's prompt and the part before the edit leave no room in the "
                         "model's context of {} tokens for the part to sing.".format(CONTEXT))
    return Region(start, stop, length, max(0, min(WIDTH, length // 3, room)))


def cut(start: int, stop: int, frames: int) -> Region:
    """The region of a cut of frames ``start`` to ``stop``: they go, and nothing comes in their place."""
    _within(start, stop, frames)
    if stop - start >= frames:
        raise ValueError("That would cut the whole song.")
    return Region(start, stop, 0, 0)


def runs_between(runs, start: int, stop: int) -> list:
    """The noise runs of frames ``start`` to ``stop`` of a song whose noise is ``runs``.

    A run is ``[seed, offset, count]``: rows ``offset`` to ``offset + count`` of
    the draw the acoustic stage makes from ``seed``. A run the stretch cuts
    through is cut with it, so the frames keep the rows they had.
    """
    found = []
    at = 0
    for seed, offset, count in runs:
        low, high = max(start, at), min(stop, at + count)
        if low < high:
            found.append([int(seed), int(offset) + low - at, high - low])
        at += count
    return found


def joined(*parts) -> list:
    """Lists of runs one after another, a run that goes on with the same draw merged into the one before."""
    found = []
    for part in parts:
        for seed, offset, count in part:
            if found and found[-1][0] == seed and found[-1][1] + found[-1][2] == offset:
                found[-1][2] += int(count)
            else:
                found.append([int(seed), int(offset), int(count)])
    return found


def edited_noise(runs, region: Region, seed: int, count: int) -> list:
    """The noise of a song after an edit: the old rows around it, the first ``count`` rows of ``seed`` inside it.

    Every old frame keeps the noise it was drawn from, the frames of the seams
    included, so the acoustic stage can hold it on the line from that noise to
    its latent. The new frames get rows of the edit's own seed.
    """
    total = sum(int(run[2]) for run in runs)
    new = [[int(seed), 0, int(count)]] if count else []
    return joined(runs_between(runs, 0, region.start), new, runs_between(runs, region.stop, total))


@dataclasses.dataclass(frozen=True)
class Words:
    """The lyrics a cut leaves, and what happened to them.

    ``dropped`` holds the tags of the sections taken out, in order. ``matched``
    is False when the lyrics' sections could not be paired with the score's
    (see ``_paired``); the lyrics are then left as they were.
    """

    text: str
    dropped: tuple
    matched: bool


def _blocks(lyrics: str) -> list:
    """The lyrics as written, cut where a tag starts a section: ``{"tag", "raw", "lines"}``.

    ``raw`` holds every line of the block with its line ending, so the blocks
    joined give the text back to the character. ``lines`` counts the sung lines
    the way ``phrasing.lyric_sections`` does, and the text before the first tag
    is a block with no tag.
    """
    blocks = [{"tag": "", "raw": [], "lines": 0}]
    for raw in str(lyrics or "").splitlines(keepends=True):
        line = raw.strip()
        found = phrasing.TAG.fullmatch(line) if line else None
        if found:
            blocks.append({"tag": found.group(1), "raw": [raw], "lines": 0})
            continue
        blocks[-1]["raw"].append(raw)
        if line and phrasing.syllables(line):
            blocks[-1]["lines"] += 1
    return blocks


def _sections(score: str, start: int, stop: int) -> list:
    """``(label, notes, cut)`` for each section of the score: its label, its vocal notes, how many the cut takes."""
    sheet = notation.read(score)
    bars = sheet["bars"]
    if not 0 <= start < stop <= len(bars):
        raise ValueError("Bars {} to {} are not bars of this score, which has {}.".format(
            start + 1, stop, len(bars)))
    onsets = [note["start"] for note in sheet["notes"]["Vocal"]]
    begin = bars[start]["start"]
    end = bars[stop - 1]["start"] + bars[stop - 1]["length"]
    found = []
    for section in sheet["sections"]:
        first, last = section["bar"], section["bar"] + section["bars"] - 1
        low, high = bars[first]["start"], bars[last]["start"] + bars[last]["length"]
        inside = [tick for tick in onsets if low <= tick < high]
        found.append((" ".join(str(section["name"]).lower().split()), len(inside),
                      sum(1 for tick in inside if begin <= tick < end)))
    return found


def _paired(labels, sections):
    """The score section each lyrics section is sung in, in order, or None when that cannot be told.

    First by name: each lyrics section takes the next score section named as its
    tag reads (see ``phrasing.label_of``), which steps over the intro, the outro
    and the interludes -- the model often gives those a few vocal notes too, so
    counting the sections it sings in does not line them up. When the names do
    not line up and the voice sings in exactly as many sections as the lyrics
    have, those are paired in order.
    """
    found = []
    at = 0
    for label in labels:
        while at < len(sections) and sections[at][0] != label:
            at += 1
        if at == len(sections):
            break
        found.append(at)
        at += 1
    if len(found) == len(labels):
        return found
    sung = [index for index, section in enumerate(sections) if section[1]]
    return sung if len(sung) == len(labels) else None


def cut_words(lyrics: str, score: str, start: int, stop: int) -> Words:
    """The lyrics without the sections a cut of bars ``start`` to ``stop`` of ``score`` takes out.

    Each lyrics section with lines is paired with the score section it is sung
    in (see ``_paired``). It goes when the cut takes more than half of that
    section's vocal notes. Most rather than all, because a sung line starts on
    the beat before its section more often than not: on the stand the stretch cut
    for a verse, from the downbeat before its first word to the downbeat before
    the next chorus's, took the last bar of the chorus before it -- the verse's
    pickup -- and left the verse's own last bar, the chorus's pickup, which is
    6 to 12 percent of those sections' notes on the wrong side of a section
    comment. A section that loses less keeps every line, because which of its
    lines were sung in the bars cut is not something the score can say.

    Everything that stays is the text as it was, to the character. Bars count
    from 0 and ``stop`` is not included.
    """
    sections = _sections(score, start, stop)
    blocks = _blocks(lyrics)
    singing = [index for index, block in enumerate(blocks) if block["lines"]]
    labels = [phrasing.label_of(blocks[index]["tag"]) if blocks[index]["tag"] else "verse"
              for index in singing]
    paired = _paired(labels, sections)
    if paired is None:
        return Words(str(lyrics or ""), (), False)
    dropped = sorted(index for index, at in zip(singing, paired)
                     if sections[at][1] and sections[at][2] * 2 > sections[at][1])
    if not dropped:
        return Words(str(lyrics or ""), (), True)
    text = "".join("".join(block["raw"]) for index, block in enumerate(blocks)
                   if index not in dropped)
    return Words(text.strip(), tuple(blocks[index]["tag"] for index in dropped), True)
