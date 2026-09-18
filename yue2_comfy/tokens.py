"""Where the YuE2 tokenizer cuts the lyrics, for the song editor to draw.

The editor lets a person click a letter to flip its case. What that changes is
not a stress mark -- the model is never told about stress -- but where the BPE
splits a word, and the only honest way to show that is to run the real
tokenizer and draw its cuts. This module does the running.

The cuts are taken from the whole prompt the model reads, ``SongRequest.text()``,
and not from the lyrics on their own. Qwen's pre-tokenizer glues a closing
bracket or a comma to the newline after it, so a line tokenized in isolation
can disagree with the same line in context exactly at its edges.

Nothing here downloads. The vocabulary arrives with the weights, and before the
weights are on disk the editor simply draws no cuts and says why.
"""

from __future__ import annotations

import bisect
import logging
import os
import threading
import unicodedata

from .vendor.yue2.protocol import SongRequest

log = logging.getLogger(__name__)

NO_VOCABULARY = (
    "Token cuts appear once the YuE2 weights are on disk: the vocabulary is "
    "part of them. Everything else in the editor works without it."
)

NOT_NFC = (
    "These lyrics contain characters the tokenizer normalises before reading "
    "(NFC), so the cuts cannot be drawn on the text as typed."
)

_lock = threading.Lock()
_held = {"path": "", "tokenizer": None}


def lyric_start(style: str) -> int:
    """Where the lyrics begin inside the prompt built for this style line.

    ``text()`` ends with the lyrics and one newline, so the prompt built around
    empty lyrics is exactly the prefix plus that newline.
    """
    return len(SongRequest(style=style, lyrics="").text()) - 1


def cuts(pieces, text: str, start: int, end: int) -> tuple:
    """``(cuts, torn)`` for the stretch ``text[start:end]``, relative to ``start``.

    ``pieces`` are the tokens of ``text`` as raw bytes. A cut is the character
    offset where a new token begins. A torn character is one a token boundary
    runs through the middle of, and its cut is placed after it, so that the
    character is drawn with the token that finishes it.

    Tearing is rare in lyrics. Measured on the real vocabulary, no letter was
    torn in Russian, Chinese, Japanese, Korean or accented Latin samples; an
    emoji was, its four bytes spread over two tokens.
    """
    encoded = text.encode("utf-8")
    if b"".join(pieces) != encoded:
        raise ValueError("the tokens do not spell the text they were made from")

    starts = []
    position = 0
    for char in text:
        starts.append(position)
        position += len(char.encode("utf-8"))

    found, torn = [], []
    boundary = 0
    for piece in pieces[:-1]:
        boundary += len(piece)
        index = bisect.bisect_right(starts, boundary) - 1
        if starts[index] != boundary:
            if start <= index < end:
                torn.append(index - start)
            index += 1
        if start < index < end:
            found.append(index - start)
    return sorted(set(found)), sorted(set(torn))


def _vocabulary() -> str:
    """The qwen.tiktoken this machine has, written out of a repack if need be."""
    from . import discovery, repack

    files = discovery.locate("standard", "bf16")
    if files.repack:
        return repack.merges_for(files.repack)
    return files.merges


def tokenizer():
    """The YuE2 text tokenizer, built once and kept.

    A located file that has since disappeared is looked for again rather than
    trusted, which is what lets the editor start drawing cuts the moment a
    download finishes, without a restart.
    """
    with _lock:
        if _held["tokenizer"] is not None and os.path.isfile(_held["path"]):
            return _held["tokenizer"]
        from .loader import load_tokenizer

        path = _vocabulary()
        _held["tokenizer"] = load_tokenizer(path)
        _held["path"] = path
        log.info("[yue2_comfy.tokens] vocabulary loaded from %s", path)
        return _held["tokenizer"]


def lyric_cuts(style: str, lyrics: str, source=None) -> dict:
    """Everything the editor needs to draw the cuts in ``lyrics``, as typed.

    The nodes strip both prompts before building the request, so the same is
    done here and the offsets are moved back onto the unstripped text.
    ``source`` stands in for the real tokenizer in tests.
    """
    style = (style or "").strip()
    raw = lyrics or ""
    body = raw.strip()
    lead = len(raw) - len(raw.lstrip())

    if unicodedata.normalize("NFC", body) != body:
        return {"ok": True, "available": False, "problem": NOT_NFC}
    if unicodedata.normalize("NFC", style) != style:
        style = unicodedata.normalize("NFC", style)

    try:
        made = source if source is not None else tokenizer()
    except FileNotFoundError:
        return {"ok": True, "available": False, "problem": NO_VOCABULARY}

    text = SongRequest(style=style, lyrics=body).text()
    ids = made.encode(text)
    pieces = [made._enc.decode_single_token_bytes(token) for token in ids]
    start = lyric_start(style)
    found, torn = cuts(pieces, text, start, start + len(body))
    return {
        "ok": True,
        "available": True,
        "cuts": [lead + at for at in found],
        "torn": [lead + at for at in torn],
        "tokens": len(found) + (1 if body else 0),
    }
