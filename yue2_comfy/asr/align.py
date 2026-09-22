"""Cutting the words of a whole song into its sections, guided by a rougher hearing of each section.

Recognised in one pass, a song's words come out best, but with no times. Heard
section by section, the words are worse -- the cuts fall in the middle of a
word and a short piece loses its bearings -- yet each piece says which words
belong to it. So the section texts only guide: their words are matched to the
whole song's words in order, and every whole-song word takes the section of
the guide word it matches.

Words between two matches are shared out in proportion to the guide words
between them, and a cut that falls within two words of a comma or a full stop
moves there. A section time falls on the downbeat, so the two or three words
sung before it -- a pickup such as "and then" -- are heard at the end of the
earlier section; when they follow the last full stop there, they move on to
the section they lead into. Standard library only.
"""

from __future__ import annotations

import difflib

PUNCTUATION = ".!?;:,\u2026"
SENTENCE_END = ".!?\u2026"
SNAP = 2
PICKUP_WORDS = 3


def normal(word: str) -> str:
    """A word as compared: lower case, yo as ye, letters and digits only."""
    lowered = word.lower().replace("\u0451", "\u0435")
    return "".join(character for character in lowered if character.isalnum())


def _guide(sections: list) -> tuple:
    words, owners = [], []
    for index, text in enumerate(sections):
        for word in (text or "").split():
            found = normal(word)
            if found:
                words.append(found)
                owners.append(index)
    return words, owners


def anchors(tokens: list, sections: list, shortest: int = 2) -> tuple:
    """``[(token index, guide index, section)]`` for whole-song words matched in runs of ``shortest`` or more, and each guide word's section."""
    guide, owners = _guide(sections)
    target = [normal(token) for token in tokens]
    matcher = difflib.SequenceMatcher(None, target, guide, autojunk=False)
    found = []
    for block in matcher.get_matching_blocks():
        if block.size >= shortest:
            found.extend((block.a + step, block.b + step, owners[block.b + step]) for step in range(block.size))
    return found, owners


def _snap(tokens: list, cut: int, low: int, high: int) -> int:
    for distance in range(SNAP + 1):
        for place in (cut - distance, cut + distance):
            if low <= place <= high and place > 0 and tokens[place - 1][-1:] in PUNCTUATION:
                return place
    return cut


def _share(tokens: list, start: int, stop: int, groups: list) -> list:
    """Sections for ``tokens[start:stop]``, given ``[section, weight]`` groups in order."""
    merged = []
    for owner, weight in groups:
        if merged and merged[-1][0] == owner:
            merged[-1][1] += weight
        else:
            merged.append([owner, weight])
    if len(merged) > 1 and all(weight == 0 for _owner, weight in merged):
        for group in merged:
            group[1] = 1
    total = sum(weight for _owner, weight in merged)
    labels = []
    position = start
    done = 0
    for index, (owner, weight) in enumerate(merged):
        done += weight
        if index == len(merged) - 1:
            end = stop
        else:
            end = start + (round((stop - start) * done / total) if total else 0)
            end = _snap(tokens, end, position, stop)
        labels.extend([owner] * max(0, end - position))
        position = max(position, end)
    return labels[:stop - start]


def move_pickups(texts: list, longest: int = PICKUP_WORDS) -> list:
    """Up to ``longest`` words after a section's last full stop, moved to the start of the next section with words."""
    texts = list(texts)
    for index in range(len(texts) - 1):
        words = texts[index].split()
        following = next((later for later in range(index + 1, len(texts)) if texts[later].split()), None)
        if not words or following is None:
            continue
        last = max((place for place, word in enumerate(words) if word[-1:] in SENTENCE_END), default=-1)
        tail = words[last + 1:]
        if last >= 0 and 0 < len(tail) <= longest:
            texts[index] = " ".join(words[:last + 1])
            texts[following] = " ".join(tail + texts[following].split())
    return texts


def split(whole: str, sections: list, shortest: int = 2) -> list:
    """The whole song's text cut into one text per section, in order; a section may get none."""
    tokens = (whole or "").split()
    if not sections:
        return []
    first = next((index for index, text in enumerate(sections) if (text or "").split()), 0)
    everything = [""] * len(sections)
    everything[first] = " ".join(tokens)
    if len(sections) == 1 or not tokens:
        return everything
    matched, owners = anchors(tokens, sections, shortest)
    if not matched:
        return everything
    labels = [None] * len(tokens)
    previous = (-1, -1)
    for a, b, owner in matched + [(len(tokens), len(owners), None)]:
        pa, pb = previous
        groups = [[owners[pb], 0]] if pb >= 0 else []
        groups.extend([owners[index], 1] for index in range(pb + 1, b))
        if owner is not None:
            groups.append([owner, 0])
        if a > pa + 1:
            labels[pa + 1:a] = _share(tokens, pa + 1, a, groups or [[0, 1]])
        if owner is not None:
            labels[a] = owner
        previous = (a, b)
    texts = [[] for _ in sections]
    for token, owner in zip(tokens, labels):
        texts[owner if owner is not None else first].append(token)
    return move_pickups([" ".join(words) for words in texts])


def words_of(text: str) -> list:
    """The words of ``text`` as the aligner counts them: whitespace apart, letters, digits and apostrophes kept.

    Qwen's own processor cleans a word this way before asking for its times,
    for every language written with spaces, and the answer comes back one pair
    of times a word -- so a caller that counts words differently reads the
    wrong times for every word after the first difference.
    """
    import unicodedata

    found = []
    for segment in str(text or "").split():
        kept = "".join(character for character in segment
                       if character == "'" or unicodedata.category(character)[0] in "LN")
        if kept:
            found.append(kept)
    return found
