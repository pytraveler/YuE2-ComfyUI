"""The sections of a written score, and the lyrics skeleton they give.

A transcribed score names its sections in comments above the bars -- intro,
verse, chorus -- in the model's own vocabulary of 23 labels. Lyrics for YuE2
are tagged with a smaller set. Each section is mapped to the tag the writer
uses, and a section in which the vocal sings nothing gets no tag: there is
nothing to write under it.
"""

from __future__ import annotations

import re

from ..vendor.yue2_music import abc_tools

TAGS = {"intro": "Intro", "verse": "Verse", "pre-chorus": "Pre-Chorus", "chorus": "Chorus",
        "post-chorus": "Chorus", "bridge": "Bridge", "outro": "Outro", "rap": "Verse",
        "interlude": "Interlude", "instrumental": "Instrumental", "solo": "Instrumental",
        "fade-out": "Outro", "pre-outro": "Bridge", "loop": "Chorus", "intro and verse": "Verse",
        "pre-chorus and chorus": "Chorus", "verse and pre-chorus": "Verse", "theme": "Verse",
        "development": "Bridge", "variation": "Verse", "irregular": "Verse", "preshot": "Intro",
        "silence": "Intro"}
"""SheetSage2's section labels to the tags ``writer.TAGS`` knows; anything else becomes a verse."""

_FULL_REST = re.compile(r"Z([2-4])?")


def _bar_count(line: str) -> int:
    count = 0
    for bar in line[:-1].split("|"):
        found = _FULL_REST.fullmatch(bar.strip())
        count += int(found.group(1) or "1") if found else 1
    return count


def sections(text: str) -> list:
    """``{"label", "tag", "start", "notes"}`` for each section, in order, from a score's comments.

    ``start`` is in quarter notes and ``notes`` counts the vocal notes that begin
    inside the section. Music before the first comment is a section labelled
    ''. Raises ValueError when the score is not in YuE2's dialect.
    """
    score = abc_tools.parse(text)
    vocal = score.voices["Vocal"]
    lines = text.splitlines()
    found = []
    pending = []
    bar_index = 0
    for index, line in enumerate(lines):
        if line.startswith("% "):
            pending.append(line[2:].strip())
            continue
        if score.music_lines.get(index) != "Vocal":
            continue
        if bar_index < len(vocal.bars) and (pending or not found):
            label = pending[-1] if pending else ""
            found.append({"label": label, "tag": TAGS.get(label.lower(), "Verse"),
                          "start": vocal.bars[bar_index][0], "notes": 0})
        pending = []
        bar_index += _bar_count(line)
    for position, section in enumerate(found):
        end = found[position + 1]["start"] if position + 1 < len(found) else None
        section["notes"] = sum(1 for onset, _pitch, _length in vocal.notes
                               if onset >= section["start"] and (end is None or onset < end))
        section["start"] = float(section["start"])
    return found


def timed(rows: dict, seconds: float) -> list:
    """``{"label", "tag", "start", "end", "notes"}`` in seconds for each section of a transcription.

    ``rows`` is what ``events.score_rows`` gives. Neighbouring intervals with the
    same label are one section, as the score writes them; the first section
    starts the recording, the last ends it, and each ends where the next begins.
    ``notes`` counts the vocal notes that begin inside it.
    """
    found = []
    for start, _end, label in rows.get("structures", ()):
        label = " ".join(str(label).split())
        if found and found[-1]["label"] == label:
            continue
        found.append({"label": label, "tag": TAGS.get(label.lower(), "Verse"), "start": float(start)})
    if not found:
        found = [{"label": "", "tag": "Verse", "start": 0.0}]
    found[0]["start"] = 0.0
    for position, section in enumerate(found):
        section["end"] = found[position + 1]["start"] if position + 1 < len(found) else float(seconds)
        section["notes"] = sum(1 for onset, _end, _pitch, track in rows.get("notes", ())
                               if track == 0 and section["start"] <= onset < section["end"])
    return found


def skeleton(found: list) -> str:
    """Lyrics with only the tags: one per section the vocal sings in, a blank line to write under each."""
    return "\n".join("[" + section["tag"] + "]\n" for section in found if section["notes"] > 0).rstrip("\n")
