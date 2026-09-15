"""Recognised words laid out as lyrics: section tags, and lines where the singer phrases.

A language model is asked for one thing only: to break each section's words
into lines, leaving out ad-libs and backing-vocal echoes if it likes. It is not
trusted with more. Its answer is kept only when every section comes back under
its own tag, in order, with words that are the recognised words in their order
-- some may be left out, none added, changed or moved -- and most of them still
there. A section that fails keeps its words and is broken into lines at its
punctuation instead, which also serves when no model answers at all.

Standard library only.
"""

from __future__ import annotations

import re

from .align import SENTENCE_END, normal

KEEP_SHARE = 0.7
"""An answer that leaves out more than three words in ten has dropped singing, not ad-libs."""
SHORTEST_LINE = 4
LONGEST_LINE = 10

SYSTEM = (
    "You lay out song lyrics that a speech recogniser heard, for a singing model.\n\n"
    "You get the song's sections in order, each under its tag, with the words heard in it run together. "
    "For every section, write its tag on a line of its own, exactly as given, and under it the same words "
    "broken into short lines where the singer phrases -- usually four to ten words a line. Leave one "
    "empty line between sections.\n\n"
    "Keep every word as it is and in its order: do not correct, translate, add or repeat words, even when "
    "one looks misheard. You may leave out ad-libs and backing-vocal echoes. Punctuation may change.\n\n"
    "Answer with the lyrics only: no explanations, no headings, no quotes."
)

_TAG = re.compile(r"\s*\[([^\]\n]+)\]\s*")
_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


def messages(sections: list, language: str = "") -> list:
    """Chat messages asking for the layout of the sections that have words."""
    blocks = ["[" + section["tag"] + "]\n" + section["text"].strip()
              for section in sections if section["text"].strip()]
    head = "The words are in {}.\n\n".format(language) if language else ""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": head + "\n\n".join(blocks)}]


def read_blocks(answer: str) -> list:
    """``[(tag, [lines])]`` in the order the answer gives them."""
    text = _THINKING.sub("", answer or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    blocks = []
    for line in text.splitlines():
        found = _TAG.fullmatch(line)
        if found:
            blocks.append((found.group(1).strip(), []))
        elif blocks and line.strip():
            blocks[-1][1].append(line.strip())
    return blocks


def faithful(lines: list, words: str) -> bool:
    """Whether ``lines`` hold the recognised words in order, with none added and most kept."""
    given = [found for found in (normal(word) for word in words.split()) if found]
    kept = [found for found in (normal(word) for line in lines for word in line.split()) if found]
    if not given or len(kept) < KEEP_SHARE * len(given):
        return False
    position = 0
    for word in kept:
        while position < len(given) and given[position] != word:
            position += 1
        if position == len(given):
            return False
        position += 1
    return True


def _clean(line: str) -> str:
    return line.strip().rstrip(".,;:").strip()


def punctuated_lines(text: str) -> list:
    """The words broken into lines after each full stop, after a comma once a line is long enough, and at ten words."""
    lines, current = [], []
    for word in text.split():
        current.append(word)
        ends = word[-1:]
        if (ends in SENTENCE_END or (ends in ",;:" and len(current) >= SHORTEST_LINE)
                or len(current) >= LONGEST_LINE):
            lines.append(" ".join(current))
            current = []
    if current:
        lines.append(" ".join(current))
    joined = []
    for line in lines:
        if joined and len(line.split()) <= 2 and joined[-1][-1:] in ",;:":
            joined[-1] += " " + line
        else:
            joined.append(line)
    return [cleaned for cleaned in (_clean(line) for line in joined) if cleaned]


def lay_out(sections: list, answer: str = "") -> tuple:
    """The lyrics, and how many sections kept the model's lines.

    ``sections`` are ``{"tag", "text"}`` in order; those without words are left
    out. The answer's blocks are matched to them in order and by tag.
    """
    wanted = [section for section in sections if section["text"].strip()]
    blocks = read_blocks(answer)
    usable = len(blocks) == len(wanted) and all(
        tag.lower() == section["tag"].lower() for (tag, _lines), section in zip(blocks, wanted))
    parts = []
    kept = 0
    for index, section in enumerate(wanted):
        lines = None
        if usable and faithful(blocks[index][1], section["text"]):
            lines = [cleaned for cleaned in (_clean(line) for line in blocks[index][1]) if cleaned]
            kept += 1
        parts.append("[" + section["tag"] + "]\n" + "\n".join(lines or punctuated_lines(section["text"])))
    return "\n\n".join(parts), kept
