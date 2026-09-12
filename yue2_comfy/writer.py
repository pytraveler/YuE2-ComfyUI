"""Turning one line of intent into a style line and lyrics YuE2 can sing.

The writing rules here are not folklore. YuE2's own prompt is
``[Tags]<style>[Lyrics]<lyrics>`` with the lyrics passed through verbatim, its
documentation asks for "genre, instruments, vocal character, language, and
tempo" in the style and "section tags such as [Verse] and [Chorus]" in the
lyrics, and the example shipped with the model is the one quoted in the system
prompt below. "Such as" is the operative phrase: the tags are a convention the
model was trained around, not an enumeration it validates, and the Chinese
example upstream does not even put blank lines between its sections. So this
module normalises towards that convention and refuses nothing for departing
from it.

Nothing here loads a model or talks to a network. It builds the messages, and
it reads the answer back. llm.py runs it.
"""

from __future__ import annotations

import re

from .constants import (LANGUAGE_CHOICES, WRITER_CONTEXT_MARGIN, WRITER_LINES,
                        WRITER_MAX_LINES, WRITER_MIN_LINES, auto_seconds)

SECTION_LINES = 4

EXAMPLE_STYLE = (
    "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and "
    "light drums, lyrical memorable melody, unhurried phrasing, 88 BPM"
)

SYSTEM = """You write songs for YuE2, a model that sings exactly what you hand it.

Answer with these two sections and nothing else. No title, no explanation, no markdown, no code fences.

STYLE
One line, comma separated, in this order: the language the song is sung in, the genre, the singing voice, two or three instruments, the character of the melody, the phrasing, and the tempo as a number followed by BPM. Name sounds, not feelings. The whole line looks like this:

{example}

LYRICS
Section tags on their own line in square brackets -- [Verse], [Chorus], [Bridge], [Outro] -- and under each tag the lines that are sung, one per line, with a blank line between sections. Nothing on a sung line but the words: no tags, no brackets, no parentheses, no stage directions, no numbering, no quotation marks.

Write {sections} sections of about {per} sung lines each, {lines} sung lines in total, not counting the tags. Bring the same chorus back rather than writing a new one.{language}

Lay the answer out exactly like this:

STYLE
<the one line>

LYRICS
[Verse]
<line>
<line>

[Chorus]
<line>
<line>"""

LANGUAGE_LINE = (
    "\n\nSing it in {language}: write the lyrics in {language}, and name {language} first "
    "in the style line."
)

REPAIR = (
    "\n\nYour last answer did not have the two sections. Answer again with the word STYLE "
    "on its own line, then the one style line, then the word LYRICS on its own line, then "
    "the tagged lyrics. Nothing else."
)

TAGS = ("intro", "verse", "pre-chorus", "prechorus", "chorus", "hook",
        "bridge", "refrain", "outro", "instrumental", "interlude")

PRETTY = {"prechorus": "Pre-Chorus", "pre-chorus": "Pre-Chorus"}

_HEADING = re.compile(r"^[\s>#*_`-]*(style|lyrics)\s*[:.\-]*\s*$", re.IGNORECASE)
_TAG_LINE = re.compile(r"^[\s>*_]*\[?\s*([A-Za-z][A-Za-z -]*?)\s*(\d+)?\s*\]?[\s:*_]*$")
_FENCE = re.compile(r"^\s*```")
_NUMBERED = re.compile(r"^\s*\d+\s*[.)]\s+")
_PARENTHETICAL = re.compile(r"^\s*[(\[][^)\]]*[)\]]\s*$")
_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think\s*>", re.IGNORECASE)


def line_target(lines) -> int:
    """Clamp a requested line count into the range the prompt will ask for."""
    try:
        wanted = int(lines)
    except (TypeError, ValueError):
        return WRITER_LINES
    return min(WRITER_MAX_LINES, max(WRITER_MIN_LINES, wanted))


def section_plan(lines) -> tuple:
    """``(sections, lines per section)`` for a line budget.

    Asking for a total alone was measured to come back 40 to 50 per cent short
    on a 4B, every time; naming a fixed six sections was exact at twelve lines
    and double at eight. Scaling the section count with the budget was the
    phrasing that held across four, eight, twelve, sixteen and twenty-eight
    lines -- twenty runs with no answer that failed to parse.
    """
    wanted = line_target(lines)
    sections = max(2, round(wanted / float(SECTION_LINES)))
    return sections, max(2, round(wanted / float(sections)))


def system_prompt(lines, language: str = "auto") -> str:
    """The whole instruction sheet, with the line budget and language filled in."""
    named = (language or "").strip()
    if not named or named.lower() == "auto" or named not in LANGUAGE_CHOICES:
        spoken = ""
    else:
        spoken = LANGUAGE_LINE.format(language=named)
    sections, per = section_plan(lines)
    return SYSTEM.format(example=EXAMPLE_STYLE, lines=line_target(lines),
                         sections=sections, per=per, language=spoken)


def build_messages(idea: str, language: str = "auto", lines=WRITER_LINES,
                   instructions: str = "", repair: bool = False) -> list:
    """The chat turns to send: the rules as system, the idea as user.

    ``instructions`` is whatever the person typed into the node's own box and
    goes after the rules, so it can override them -- someone who wants no
    chorus at all should get no chorus.
    """
    system = system_prompt(lines, language)
    extra = (instructions or "").strip()
    if extra:
        system = system + "\n\n" + extra
    if repair:
        system = system + REPAIR
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": (idea or "").strip()},
    ]


def context_needed(messages: list, max_new_tokens: int) -> int:
    """A context window wide enough for these messages and the answer.

    Deliberately generous: three characters per token under-counts on every
    tokenizer worth using, and the cost of over-counting is a few megabytes of
    KV cache, while the cost of under-counting is a song that stops mid-word.
    """
    characters = sum(len(message.get("content") or "") for message in messages)
    return int(characters / 3) + int(max_new_tokens) + WRITER_CONTEXT_MARGIN


def strip_thinking(text: str) -> str:
    """Whatever the model wrote after it finished reasoning.

    A reasoning model asked to turn thinking off still sometimes opens a block,
    and an answer that is nothing but an unclosed one is not an answer -- it
    comes back empty, which is what makes the caller try again.
    """
    body = text or ""
    match = None
    for match in _THINK_CLOSE.finditer(body):
        pass
    if match is not None:
        return body[match.end():]
    if _THINK_OPEN.search(body):
        return ""
    return body


def _tag(line: str) -> str:
    """The normalised section tag this line is, or "" if it is not one."""
    match = _TAG_LINE.match(line)
    if not match:
        return ""
    word = match.group(1).strip().lower()
    if word not in TAGS:
        return ""
    pretty = PRETTY.get(word) or "-".join(part.capitalize() for part in word.split("-"))
    number = match.group(2)
    return "[" + pretty + (" " + number if number else "") + "]"


def tidy_style(block: str) -> str:
    """The style line, taken out of whatever the model wrapped it in."""
    for line in (block or "").splitlines():
        line = line.strip().strip("`")
        if not line or _FENCE.match(line) or _HEADING.match(line):
            continue
        line = line.strip("*_ ").strip()
        if line[:1] in ('"', "'") and line[-1:] == line[:1] and len(line) > 1:
            line = line[1:-1].strip()
        if line:
            return re.sub(r"\s+", " ", line)
    return ""


def tidy_lyrics(block: str) -> str:
    """The lyrics, with the tags normalised and the model's debris removed.

    Blank lines are rebuilt rather than preserved: small models drop them, and
    a tag welded to the line above it reads as a lyric with a bracket in it.
    """
    kept: list = []
    for raw in (block or "").splitlines():
        line = raw.strip()
        if not line or _FENCE.match(line):
            continue
        if _HEADING.match(line):
            continue
        tag = _tag(line)
        if tag:
            if kept:
                kept.append("")
            kept.append(tag)
            continue
        if _PARENTHETICAL.match(line):
            continue
        line = _NUMBERED.sub("", line).strip()
        line = line.strip("*_`").strip()
        if line[:1] in ('"', "'") and line[-1:] == line[:1] and len(line) > 1:
            line = line[1:-1].strip()
        if line:
            kept.append(line)
    while kept and not kept[-1]:
        kept.pop()
    return "\n".join(kept)


def split(text: str) -> tuple:
    """Pull ``(style, lyrics)`` out of one raw answer.

    The headed layout is what the prompt asks for and what models give back.
    When the headings are missing the first section tag is the seam instead:
    everything above it was describing the song, everything below it is the
    song. That fallback is why a model that ignores the layout still produces
    something usable rather than a refusal.
    """
    body = strip_thinking(text)
    buckets = {"style": [], "lyrics": []}
    head: list = []
    current = ""
    seen = False
    for line in body.splitlines():
        heading = _HEADING.match(line)
        if heading:
            current = heading.group(1).lower()
            seen = True
            continue
        if current:
            buckets[current].append(line)
        elif not seen:
            head.append(line)

    if seen and (buckets["style"] or buckets["lyrics"] or head):
        style = tidy_style("\n".join(buckets["style"]))
        if not style:
            style = tidy_style("\n".join(head))
        return style, tidy_lyrics("\n".join(buckets["lyrics"]))

    lines = body.splitlines()
    for position, line in enumerate(lines):
        if _tag(line.strip()):
            return (tidy_style("\n".join(lines[:position])),
                    tidy_lyrics("\n".join(lines[position:])))
    return tidy_style(body), ""


def sung(lyrics: str) -> int:
    """Lines that carry words, which is what the length ceiling is built on."""
    count = 0
    for line in (lyrics or "").splitlines():
        line = line.strip()
        if line and not (line.startswith("[") and line.endswith("]")):
            count += 1
    return count


def complete(style: str, lyrics: str) -> bool:
    """Whether this answer is worth handing on rather than asking again."""
    return bool((style or "").strip()) and sung(lyrics) > 0


def findings(style: str, lyrics: str, wanted) -> list:
    """What to say on the node about an answer that came back usable anyway.

    None of these are errors. A style line with no BPM still sings, and six
    lines when twelve were asked for is a shorter song, not a broken one -- so
    they are reported where the person can see them and nothing is rewritten
    behind their back.
    """
    notes: list = []
    target = line_target(wanted)
    written = sung(lyrics)
    if written and abs(written - target) > max(2, target // 3):
        notes.append(("info", "Asked for about {} sung lines, got {}, so the length ceiling "
                              "works out at about {:.0f} s.".format(
                                  target, written, auto_seconds(lyrics))))
    if style and not re.search(r"\d+\s*bpm", style, re.IGNORECASE):
        notes.append(("info", "The style line names no tempo. YuE2 will pick one."))
    if lyrics and not any(_tag(line.strip()) for line in lyrics.splitlines()):
        notes.append(("info", "The lyrics carry no section tags, so the whole song is one "
                              "block. Add [Verse] and [Chorus] lines to shape it."))
    return notes
