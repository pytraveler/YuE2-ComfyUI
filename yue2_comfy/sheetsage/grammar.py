"""Which tokens the decoder may choose next.

The model is decoded greedily, but never freely: at each step only the tokens
that keep the sequence well formed are allowed. After a shift more shifts may
follow (four in a row at most), and once an event has a value it may end, with
a shift or with eos. Fields come in their fixed order, so a key cannot follow a
chord in the same event. A meter may be followed by its eighth-note position,
and a pitch by its length or another pitch; a meter left without its position
is read as the meter alone (see ``vocab.decode``).

Only the full-chord and full-melody tasks are ever opened, so the major/minor
chord block is never allowed. The allowed set is returned as id ranges; the
decoder turns them into a mask.
"""

from __future__ import annotations

from . import vocab

FIELD_INDEX = {field: index for index, field in enumerate(vocab.FIELDS)}


class Grammar:
    """The state of one sequence being decoded."""

    def __init__(self):
        self.shifting = True
        self.shifts = 0
        self.filled = 0
        self.field = -1
        self.owed = None

    def allowed(self) -> list:
        """The ``[start, end)`` id ranges the next token may come from."""
        ranges = []
        if self.filled > 0:
            ranges.append((vocab.EOS, vocab.EOS + 1))
        if (self.filled > 0 or self.shifting) and self.shifts < 4:
            ranges.append(vocab.BLOCKS["shift"])
        if self.owed == "eighth":
            ranges.append(vocab.BLOCKS["eighth"])
            return ranges
        if self.owed == "duration":
            ranges.append(vocab.BLOCKS["duration"])
            ranges.append(vocab.BLOCKS["pitch"])
            return ranges
        if self.field < FIELD_INDEX["timestamp"]:
            ranges.append(vocab.BLOCKS["time"])
        if self.field < FIELD_INDEX["rhythm"]:
            ranges.append(vocab.BLOCKS["meter"])
            ranges.append(vocab.BLOCKS["eighth"])
        if self.field < FIELD_INDEX["structure"]:
            ranges.append(vocab.BLOCKS["structure"])
        if self.field < FIELD_INDEX["key"]:
            ranges.append(vocab.BLOCKS["key"])
        if self.field < FIELD_INDEX["chord"]:
            ranges.append(vocab.BLOCKS["chord_full"])
        if self.field <= FIELD_INDEX["melody"]:
            ranges.append(vocab.BLOCKS["pitch"])
        return ranges

    def update(self, token) -> bool:
        """Take one token; True when it ends the sequence."""
        token = int(token)
        if token == vocab.EOS:
            return True
        token_kind = vocab.kind(token)
        if token_kind == "shift":
            if not self.shifting and self.filled > 0:
                self.filled = 0
                self.field = -1
                self.owed = None
            self.shifting = True
            self.shifts += 1
            return False
        self.shifting = False
        self.shifts = 0
        self.filled += 1
        field = vocab.KIND_FIELDS.get(token_kind)
        if field is None or token_kind == "chord_majmin":
            raise ValueError("unexpected token kind {!r} while decoding".format(token_kind))
        self.field = FIELD_INDEX[field]
        self.owed = {"meter": "eighth", "pitch": "duration"}.get(token_kind)
        return False

    def follow(self, prefix) -> "Grammar":
        """Take the events of a prefix: everything after its out token."""
        prefix = [int(token) for token in prefix]
        for token in prefix[prefix.index(vocab.OUT) + 1:]:
            self.update(token)
        return self
