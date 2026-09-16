"""The words of a karaoke MIDI file: lyrics YuE2 can sing, and the ticks their sections begin at.

Karaoke files carry the words as syllables timed to the tune, in one of two
conventions. Files in the .kar layout keep them as text events, with lines
starting '@' for the title and other headers; a syllable starting with a
backslash begins a new paragraph and one starting with '/' a new line. Other
files use lyric events and end a line with a carriage return or a newline.
Both are read the same way. Lyric events win when a file has enough of them,
since the text events of such a file are usually titles and credits.

A paragraph is a section; so is the start of a verse after a silence of two
bars of 4/4 or more in a file that marks no paragraphs, and an empty line
ends one too. A paragraph whose words repeat an earlier paragraph's is the
chorus, and so is that earlier one; the rest are verses. Where a file has
markers named after sections -- 'Verse 2', 'Chorus' -- the marker nearest a
paragraph's start names it instead.

The encoding is guessed once, from all the syllables together, because a
syllable alone is too short to tell Windows-1251 from Latin-1.
"""

from __future__ import annotations

import re

from ..sheetsage import sections as score_sections
from . import smf, tracks

LEAST_SYLLABLES = 8
"""Fewer timed syllables than this are a title or a credit, not the words of a song."""

PAUSE_QUARTERS = 8
"""Quarter notes of silence between syllables that start a new section in a file marking no paragraphs."""

MARKER_REACH = 4
"""Quarter notes either side of a paragraph's first syllable a marker may sit and still name it."""

BREAK = re.compile(r"\r\n|\r|\n")
LABELS = sorted(score_sections.TAGS, key=len, reverse=True)


class Words:
    """A song's words: ``sections`` as ``{"tick", "tag", "label", "lines"}``, and ``syllables``, a tick each."""

    def __init__(self, sections: list, syllables: list):
        self.sections = sections
        self.syllables = syllables

    def lyrics(self) -> str:
        """The words as lyrics for YuE2: a tag above each section, a blank line between sections."""
        return "\n\n".join("[" + section["tag"] + "]\n" + "\n".join(section["lines"]) for section in self.sections)


def section_label(text: str):
    """The section a marker names, as a label ``sheetsage.sections.TAGS`` knows, or None: 'Chorus 2' is 'chorus'."""
    clean = " ".join(str(text).lower().replace("_", " ").split())
    for label in LABELS:
        if re.match(re.escape(label) + r"(?:[\s\d:.-]|$)", clean):
            return label
    return None


def marker_sections(song: smf.Song) -> list:
    """``(tick, label)`` for every marker that names a section, in order."""
    found = []
    for tick, payload in song.markers:
        label = section_label(tracks.decode(payload))
        if label is not None:
            found.append((tick, label))
    return found


def _syllables(song: smf.Song) -> list:
    lyric = [(tick, payload) for track in song.tracks for tick, kind, payload in track.texts
             if kind == smf.LYRIC and not payload.startswith(b"@")]
    if len(lyric) >= LEAST_SYLLABLES:
        return sorted(lyric, key=lambda item: item[0])
    best = []
    for track in song.tracks:
        found = [(tick, payload) for tick, kind, payload in track.texts
                 if kind == smf.TEXT and not payload.startswith(b"@")]
        if len(found) > len(best):
            best = found
    return best if len(best) >= LEAST_SYLLABLES else []


class _Builder:
    """Sections of lines, filled a syllable at a time."""

    def __init__(self):
        self.sections = []
        self.lines = []
        self.line = ""
        self.start = None

    def add(self, tick: int, text: str) -> None:
        if text.strip() and self.start is None:
            self.start = tick
        self.line += text

    def end_line(self) -> None:
        text = " ".join(self.line.split())
        self.line = ""
        if text:
            self.lines.append(text)
        elif self.lines:
            self.end_section()

    def end_section(self) -> None:
        text = " ".join(self.line.split())
        self.line = ""
        if text:
            self.lines.append(text)
        if self.lines:
            self.sections.append({"tick": self.start, "lines": self.lines})
        self.lines = []
        self.start = None


def _tag(sections: list, song: smf.Song) -> None:
    keys = [re.sub(r"\W+", "", " ".join(section["lines"]).lower()) for section in sections]
    for section in sections:
        section["tag"] = "Verse"
    for index, key in enumerate(keys):
        for earlier in range(index):
            if key and keys[earlier] == key:
                sections[index]["tag"] = sections[earlier]["tag"] = "Chorus"
    markers = marker_sections(song)
    reach = MARKER_REACH * song.division
    for section in sections:
        near = [(abs(tick - section["tick"]), label) for tick, label in markers if abs(tick - section["tick"]) <= reach]
        if near:
            section["tag"] = score_sections.TAGS[min(near)[1]]
    for section in sections:
        section["label"] = section["tag"].lower()


def read(song: smf.Song):
    """The song's words, or None when the file carries no timed words."""
    syllables = _syllables(song)
    if not syllables:
        return None
    encoding = tracks.encoding_of(b" ".join(payload for _tick, payload in syllables))
    pause = PAUSE_QUARTERS * song.division
    builder = _Builder()
    previous = None
    for tick, payload in syllables:
        piece = tracks.text_of(payload, encoding)
        if previous is not None and tick - previous >= pause and (builder.lines or builder.line.strip()):
            builder.end_section()
        previous = tick
        if piece.startswith("\\"):
            builder.end_section()
            piece = piece[1:]
        elif piece.startswith("/"):
            builder.end_line()
            piece = piece[1:]
        pieces = BREAK.split(piece)
        builder.add(tick, pieces[0])
        for rest in pieces[1:]:
            builder.end_line()
            builder.add(tick, rest)
    builder.end_section()
    if not builder.sections:
        return None
    _tag(builder.sections, song)
    return Words(builder.sections, [tick for tick, payload in syllables if tracks.text_of(payload, encoding).strip(" \\/\r\n")])
