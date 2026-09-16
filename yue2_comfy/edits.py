"""An edited score, and the words it was edited for.

A score is written for one style, one set of lyrics and one 'cot' mode, so an
edit of it is an edit for those words. Kept on a node, it outlives them: the
lyrics on 'YuE2 Generate Song' change in a click, and words that arrive through
a wire change without the browser ever seeing them. Old notes sung under new
words is the one outcome nobody asks for, so an edit carries a mark of the
words it belongs to, and a node that sings it checks the mark against the words
it actually has.

The mark is the last line of the edited text: an ABC comment the score editor
adds on Apply and every node takes off before anything is compared or sung, so
the model never reads it. It names the words and not the seed, on purpose. A
new seed with the same words sings the same edit as a new take, which is how a
bar that did not take gets another chance. A score without a mark -- pasted,
wired in, or saved before marks existed -- is sung as it is, whatever the words.

Nothing here touches torch or the model, so every rule in this file is checked
on a machine that has never downloaded the weights.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re

SCORE_UI = "yue2_score"
"""Where a node hands the score the model wrote to the browser.

The score editor reads it from the node it edits for: 'YuE2 Generate Song'
hands over its own, and 'YuE2 Render Plan' is fed by whichever plan node is on
its plan input. Any node's 'ui' reaches the browser, but only an output node
can be run on its own, which is why the plan nodes are output nodes: the editor
queues one of them alone to get a score without a song. 'YuE2 Generate Song'
is not one, because running it alone is running the whole song."""

WORDS_UI = "yue2_words"
"""Where the same node hands over the mark of the words that score was written for."""

AUTO_SECONDS_UI = "yue2_auto_seconds"
"""Where a node hands over the ceiling its lyrics give a song when 'max_seconds' is 0.

The score editor draws where the singing stops. It counts the lines itself when
the lyrics are on the canvas, and needs this when they came in through a wire."""

MARK_PREFIX = "%yue2-words "
MARK_LINE = re.compile(r"%yue2-words ([0-9a-f]{16})[ \t\r]*")

OTHER_WORDS = (
    "The edited score on this node was made for other words: the style, the lyrics "
    "or 'cot' have changed since it was edited. It was not sung, and {instead}, so "
    "old notes do not end up under new words. Put the words back and the edit is "
    "sung again, or open 'Edit score...' to edit the new score."
)

COT_OFF = (
    "'cot' is 'off', which sings straight from the lyrics without a score, so the "
    "edited score on this node was not used. It stays on the node for when 'cot' "
    "is 'full' or 'melody' again."
)


@dataclasses.dataclass(frozen=True)
class Edit:
    """What a score box holds: the score, trimmed and without its mark, and the mark."""

    score: str
    words: str | None


def mark(style, lyrics, cot) -> str:
    """Sixteen hex digits that name the words a score was written for.

    The style and the lyrics are trimmed first, the way the nodes trim them
    before the model sees them, so a trailing newline is not a change of words.
    """
    payload = json.dumps([str(style or "").strip(), str(lyrics or "").strip(),
                          str(cot or "")])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def read(text) -> Edit:
    """The score and its mark out of a node's score box.

    Every line that is a mark is taken out, wherever it sits, and the last one
    wins; a line that only looks like one stays in the score. The score is
    trimmed at the ends, as the nodes have always trimmed an edited score.
    """
    words = None
    kept = []
    for line in str(text or "").split("\n"):
        found = MARK_LINE.fullmatch(line)
        if found:
            words = found.group(1)
        else:
            kept.append(line)
    return Edit("\n".join(kept).strip(), words)


def attach(score, words) -> str:
    """The score with the mark as its last line, or the bare score when there is no mark."""
    clean = str(score or "").rstrip()
    return clean + "\n" + MARK_PREFIX + words if words else clean


def mismatch(edit, style, lyrics, cot, instead) -> str:
    """Why this edit is not sung with these words, or '' when nothing stops it.

    *instead* says what the node does in its place, and is written into the
    message. No edit at all is never a mismatch.
    """
    if not edit.score:
        return ""
    if cot == "off":
        return COT_OFF
    if edit.words is not None and edit.words != mark(style, lyrics, cot):
        return OTHER_WORDS.format(instead=instead)
    return ""


LYRICS_UI = "yue2_lyrics"
"""Where 'YuE2 Transcribe' and 'YuE2 Load MIDI' hand the lyrics they would output to the browser."""

TRACK_UI = "yue2_track"
"""Where 'YuE2 Transcribe' hands over the mark of the recording it transcribed, and 'YuE2 Load MIDI' that of its file."""

MARKS_UI = "yue2_marks"
"""Where those two nodes hand over the score marks of that recording or file in every mode, so the browser need not hash."""

MIDI_UI = "yue2_midi"
"""Where 'YuE2 Load MIDI' hands over the file's tracks and the facts of the score written from them, for its list."""


def audio_mark(data: bytes, rate) -> str:
    """Sixteen hex digits that name a recording: its samples and their rate.

    The transcriber's edits belong to a recording the way the singer's belong
    to words, so this is the mark they carry instead of ``mark``.
    """
    digest = hashlib.sha256()
    digest.update(str(int(rate)).encode("ascii") + b"\0")
    digest.update(memoryview(data))
    return digest.hexdigest()[:16]


def track_mark(audio: str, mode: str) -> str:
    """The mark of a score transcribed from a recording in one mode: melody alone, or with chords."""
    return hashlib.sha256(json.dumps([str(audio), str(mode)]).encode("utf-8")).hexdigest()[:16]


def file_mark(data: bytes) -> str:
    """Sixteen hex digits that name a file by its bytes: the MIDI loader's edits belong to a file, not a name."""
    return hashlib.sha256(bytes(data)).hexdigest()[:16]


def midi_mark(file: str, mode: str, vocal, instrument) -> str:
    """The mark of a score written from a file in one mode, with one choice of voice and instrument tracks."""
    payload = json.dumps([str(file), str(mode), str(vocal), str(instrument)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


CHORDLESS = (
    "The score to sing has no chord symbols -- a melody-only transcription looks "
    "like that -- but 'cot' is 'full', which tells the model the score carries the "
    "harmony as well. It is sung anyway; set 'cot' to 'melody' in YuE2 Options to "
    "let the accompaniment follow the style instead."
)

_CHORD_SYMBOL = re.compile(r'"[^"\n]*"')
_FIELD_LINE = re.compile(r"^\s*(%|[A-Za-z]:)")


def chordless(score) -> bool:
    """Whether a score has music in it but not one chord symbol.

    Header, voice and comment lines are passed over: the ``V:`` lines quote the
    voice names, and a quoted name is not a chord.
    """
    music = [line for line in str(score or "").split("\n")
             if line.strip() and not _FIELD_LINE.match(line)]
    return bool(music) and not any(_CHORD_SYMBOL.search(line) for line in music)
