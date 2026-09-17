"""Only the rows of the two big tables that the running phase can use.

YuE2 writes its score out of the text vocabulary and its song out of the codec
one, and ``sampling.distribution`` masks every other row to -inf before it
samples. So while the song is being written, only 32770 of the 184704 rows of
``embed_tokens`` and of ``lm_head`` can matter: 0.125 GiB each on the card
instead of 0.705, with the tables themselves left in system RAM.

Measured on an RTX 5090 on 2026-09-17, with ``offload`` at ``on``: a 40-second
song peaked at 4.43 GiB instead of 4.67 and a 233-second one at 4.50 instead of
5.66, both a second or two faster because less is carried to the card, and both
the same song to the last byte -- the rows this leaves behind were multiplied
by -inf either way. With CFG at 3.0 the song matched to the byte as well.

The lookup cannot simply move to the CPU: ``cuda_graph._decode`` embeds inside
the captured graph, and a graph holds no CPU work. A window is an ordinary CUDA
tensor and captures like any other, so the decode loop reads the window and
everything outside the graph -- the prefills, and the acoustic stage, which
embeds a mixed prefix -- reads the table in RAM.

Three details that bite, each found by running it:

* The table must not be reachable through ``model.parameters()``, or
  ``sampling.generate_tokens`` reads the device off it and decides the model is
  on the CPU. It is held in a list, which ``nn.Module`` does not walk.
* Outside the window the head writes zeros, not -inf. With CFG the caller
  combines two branches before masking, and -inf minus -inf is NaN.
* The window must hold the token the graph starts from. GraphAR fills its
  static token buffer with the last token of the prefix and embeds it while
  warming the capture up, and for the song that token is MUSIC_START, one row
  below the mask.
"""

from __future__ import annotations

import contextlib
import logging

import torch
import torch.nn.functional as F

from .vendor.yue2.protocol import (ABC_END, CODEC_OFFSET, CODEC_SIZE, MUSIC_END,
                                   VOCAB_SIZE)

log = logging.getLogger(__name__)

ABC, MUSIC = "abc", "music"


def window(phase: str, seeds=()) -> tuple:
    """The contiguous rows a phase can read or sample, and the tokens it starts from."""
    start, stop = (0, ABC_END + 1) if phase == ABC else (MUSIC_END, CODEC_OFFSET + CODEC_SIZE)
    for token in seeds:
        start, stop = min(start, int(token)), max(stop, int(token) + 1)
    return start, stop


class Sliced(torch.nn.Module):
    """One vocabulary table: all of it in RAM, one window of it on the card."""

    yue2_self_placed = True
    """Tells ``placement.groups`` that no half owns this module."""

    def __init__(self, inner, card):
        super().__init__()
        self.held = [inner.weight.detach()]
        self.card = card
        self.start, self.stop = 0, 0
        self.windowed = False
        self.register_buffer("rows", self.held[0][:1].to(card, copy=True))

    @property
    def weight(self):
        """What GraphAR reads the model's device and dtype from."""
        return self.rows

    def table(self):
        return self.held[0]

    def retune(self, start: int, stop: int):
        self.start, self.stop = int(start), int(stop)
        self.rows = self.table()[self.start:self.stop].to(self.card, copy=True)

    def release(self):
        """Give the window's memory back.

        ``copy=True`` says what is meant. Without it a window that is already
        on the card, or a run on the CPU, hands back a view of the whole table
        and frees nothing at all.
        """
        self.rows = self.table()[:1].to(self.card, copy=True)


class SlicedEmbedding(Sliced):
    """``embed_tokens``: a row lookup, from the window or from RAM."""

    def forward(self, ids):
        if self.windowed:
            return F.embedding(ids - self.start, self.rows)
        return F.embedding(ids.to(self.table().device), self.table()).to(self.card)


class SlicedHead(Sliced):
    """``lm_head``: the window's logits, written into a full-width answer."""

    def __init__(self, inner, card):
        super().__init__(inner, card)
        self.full = None

    def forward(self, hidden):
        if not self.windowed:
            return F.linear(hidden.to(self.table().device), self.table()).to(self.card)
        shape = tuple(hidden.shape[:-1]) + (VOCAB_SIZE,)
        if self.full is None or tuple(self.full.shape) != shape:
            self.full = torch.zeros(shape, dtype=self.rows.dtype, device=self.card)
        self.full[..., self.start:self.stop] = F.linear(hidden, self.rows)
        return self.full

    def release(self):
        super().release()
        self.full = None


def tune(models, phase: str, *prefixes):
    """Cut both windows to the phase about to be decoded.

    Does nothing when the tables are not narrowed, so a caller does not have to
    know which mode it is running under.
    """
    tables = _tables(getattr(models, "lm", None))
    if not tables:
        return
    seeds = [prefix[-1] for prefix in prefixes if prefix is not None and len(prefix)]
    start, stop = window(phase, seeds)
    for table in tables:
        table.retune(start, stop)


def _tables(lm) -> list:
    """The two sliced tables of a model, or nothing if it has none."""
    if lm is None:
        return []
    found = [getattr(getattr(lm, "model", None), "embed_tokens", None),
             getattr(lm, "lm_head", None)]
    return [table for table in found if isinstance(table, Sliced)]


def _windowed(lm, value: bool):
    """Turn the windows on once the prefill, which needs every row, is done."""
    for table in _tables(lm):
        table.windowed = bool(value)
        if not value:
            table.release()


@contextlib.contextmanager
def narrowed(models, enabled: bool = True):
    """Keep the two tables in RAM and only a phase's window on the card.

    The swap lasts one stage and is undone afterwards, including on failure.
    ``placement`` is told to forget what it measured, because the AR half is
    1.41 GiB smaller while this holds and that size decides what it moves.
    """
    lm = getattr(models, "lm", None)
    card = getattr(models, "device", None)
    if not enabled or lm is None or getattr(card, "type", "cpu") != "cuda":
        yield
        return
    try:
        kept = (lm.model.embed_tokens, lm.lm_head)
    except AttributeError:
        log.debug("[yue2_comfy.vocabulary] this model has no tables to narrow", exc_info=True)
        yield
        return

    from .vendor.yue2 import cuda_graph

    lm.model.embed_tokens = SlicedEmbedding(kept[0], card)
    lm.lm_head = SlicedHead(kept[1], card)
    lm._yue2_placement = None
    original = cuda_graph.GraphAR

    class WindowedGraphAR(original):
        """Upstream's graph, with the windows turned on around the capture."""

        def _capture(self):
            _windowed(lm, True)
            super()._capture()

        def close(self):
            _windowed(lm, False)
            super().close()

    cuda_graph.GraphAR = WindowedGraphAR
    try:
        yield
    finally:
        cuda_graph.GraphAR = original
        _windowed(lm, False)
        lm.model.embed_tokens, lm.lm_head = kept
        lm._yue2_placement = None
