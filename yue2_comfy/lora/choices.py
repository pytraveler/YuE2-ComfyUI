"""The adapters a run sings with, as the settings carry them and as the model is folded with them.

The node hands these on as plain dictionaries -- a name, the file, its SHA-256
and the two strengths -- because they travel inside the settings: into a plan
ComfyUI caches between runs, and into the song memory, which writes them down.
Here they become Choice objects, in the one order folding adds them in.

That order is not the order of the rows. Adding two differences in float32
and then the other way round can differ in the last bit, and after the round
to bf16 that is sometimes a different weight -- a different song for the same
seed because two rows were dragged past each other. Sorted by the file's hash,
the rows can be arranged however the person likes.

Nothing here imports torch: ``placement`` asks which halves a run touches
before any weight moves.
"""

from __future__ import annotations

import dataclasses

from . import catalogue, formats

AR, NAR = formats.AR, formats.NAR


@dataclasses.dataclass(frozen=True)
class Choice:
    """One adapter of a run: which file, and how strongly on each half."""

    name: str
    path: str
    sha256: str
    ar: float
    nar: float

    def strength(self, half: str) -> float:
        return self.ar if half == AR else self.nar

    def halves(self) -> tuple:
        """The halves this file changes, as its header says, whatever the strengths."""
        found = catalogue.verdict(self.path) or {}
        return tuple(half for half in formats.HALVES if half in (found.get("halves") or {}))


def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number and abs(number) != float("inf") else 0.0


def from_settings(specs) -> tuple:
    """Choices from what the settings carry, sorted into folding order, rows at zero left out."""
    found = []
    for spec in specs or ():
        if not isinstance(spec, dict) or not spec.get("path"):
            continue
        choice = Choice(name=str(spec.get("name") or ""), path=str(spec["path"]),
                        sha256=str(spec.get("sha256") or ""), ar=_number(spec.get("ar")),
                        nar=_number(spec.get("nar")))
        if choice.ar or choice.nar:
            found.append(choice)
    found.sort(key=lambda choice: (choice.sha256, choice.name, choice.path, choice.ar, choice.nar))
    return tuple(found)


def touching(choices, half: str) -> tuple:
    """The choices that change ``half``: a file with parts there, at a strength other than zero."""
    return tuple(choice for choice in choices or ()
                 if choice.strength(half) and half in choice.halves())


def signature(choices, half: str) -> tuple:
    """What a half holds once these choices are folded in, comparable between runs.

    The file's hash and its identity on disk both go in, so a file replaced
    under the same name is folded again even before its hash is known.
    """
    return tuple((choice.sha256, catalogue.stamp(choice.path) or choice.path,
                  repr(float(choice.strength(half))))
                 for choice in touching(choices, half))


def halves(choices) -> set:
    """Every half some choice changes."""
    return {half for half in formats.HALVES if touching(choices, half)}
