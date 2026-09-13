"""The song editor's JavaScript: it parses, it does what it says, and it agrees with Python.

The browser half of the pack has no build step and no test runner of its own.
Node runs the pure half here -- web/js/yue2_sheet.js imports nothing from
ComfyUI for exactly this reason -- and everything is skipped on a machine
without Node rather than failing there.

Two lists live on both sides of the wire, the languages and the section tags.
They are checked against the Python that owns them, because a tag the editor
cycles to and the writer does not know is a disagreement nobody would see.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess

import pytest

from yue2_comfy import constants, writer

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "js"
SHEET = WEB / "yue2_sheet.js"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

TOP_LEVEL = re.compile(r"^(?:export\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", re.MULTILINE)


def modules():
    return sorted(WEB.glob("*.js"))


def test_there_are_modules_to_check():
    assert len(modules()) >= 3, "no JavaScript found -- the checks below would pass vacuously"


@needs_node
@pytest.mark.parametrize("path", modules(), ids=lambda p: p.name)
def test_every_module_parses(path):
    done = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("path", modules(), ids=lambda p: p.name)
def test_no_top_level_function_is_shadowed_by_a_binding(path):
    """Carried over from the MiniMax pack, where this shipped once.

    A module-level function whose name is also bound with const or let in some
    inner scope parses fine and throws at the first click: the inner binding
    hoists over its whole scope, and the call lands in the temporal dead zone.
    """
    source = path.read_text(encoding="utf-8")
    clashes = []
    for match in TOP_LEVEL.finditer(source):
        name = match.group(1)
        bound = re.compile(r"\b(?:const|let|var)\s+(?:\[[^\]]*\b" + re.escape(name)
                           + r"\b[^\]]*\]|\{[^}]*\b" + re.escape(name) + r"\b[^}]*\}|"
                           + re.escape(name) + r"\b)")
        if bound.search(source):
            clashes.append(name)
    assert not clashes, "{}: shadowed functions: {}".format(path.name, clashes)


def run_sheet(script: str):
    """Run ``script`` with the sheet imported as ``s``; it must print one JSON value."""
    program = "import * as s from {};\n{}".format(json.dumps(SHEET.as_uri()), script)
    done = subprocess.run([NODE, "--input-type=module", "-e", program],
                          capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def js_array(name: str) -> list:
    """An exported string array, read from the source without running it."""
    source = SHEET.read_text(encoding="utf-8")
    match = re.search(r"export const " + name + r" = \[(.*?)\];", source, re.DOTALL)
    assert match, name + " is not exported as a literal array"
    return re.findall(r'"([^"]*)"', match.group(1))


def test_the_languages_are_the_writers_less_auto():
    assert js_array("LANGUAGES") == [c for c in constants.LANGUAGE_CHOICES if c != "auto"]


def test_the_tags_are_the_writers_in_its_spelling():
    spelled = []
    for word in writer.TAGS:
        name = writer._tag("[" + word + "]")[1:-1]
        if name not in spelled:
            spelled.append(name)
    assert js_array("TAGS") == spelled
    assert set(js_array("CYCLE")) <= set(spelled)


@needs_node
def test_a_tag_line_is_read_the_way_the_writer_reads_it():
    lines = ["[verse]", "[VERSE 2]", "[Pre-Chorus]", "[prechorus]",
             "  [ Chorus ]  ", "[outro]", "[Instrumental 3]"]
    got = run_sheet("console.log(JSON.stringify({}.map((l) => s.header(s.parseLyrics(l)[0]))));"
                    .format(json.dumps(lines)))
    assert got == [writer._tag(line.strip()) for line in lines]


@needs_node
def test_the_default_style_parses_into_its_parts_and_writes_back_unchanged():
    got = run_sheet("""
        const parts = s.parseStyle({style});
        console.log(JSON.stringify({{
            language: s.languageOf(parts), bpm: s.bpmOf(parts), voice: s.voiceOf(parts),
            others: parts.filter((p) => p.kind === "other").map((p) => p.text),
            back: s.formatStyle(parts),
        }}));
    """.format(style=json.dumps(constants.DEFAULT_STYLE)))
    assert got == {
        "language": "English", "bpm": 88, "voice": "expressive female voice",
        "others": ["warm piano pop", "acoustic piano", "rounded bass and light drums",
                   "unhurried phrasing"],
        "back": constants.DEFAULT_STYLE,
    }


@needs_node
def test_style_parts_are_changed_in_place_and_added_where_the_writer_puts_them():
    got = run_sheet("""
        const f = (parts) => s.formatStyle(parts);
        const base = s.parseStyle("English, pop, piano, 88 BPM");
        const bare = s.parseStyle("pop, piano");
        console.log(JSON.stringify([
            f(s.setBpm(base, 120)),
            f(s.setBpm(base, 999)),
            f(s.setBpm(base, null)),
            f(s.setLanguage(base, "")),
            f(s.setLanguage(bare, "Russian")),
            f(s.setVoice(bare, "soft female voice")),
            f(s.addPart(base, "staccato phrasing")),
            f(s.addPart(bare, "staccato, phrasing")),
            f(s.movePart(base, 2, -1)),
            f(s.movePart(base, 1, -1)),
            f(s.editPart(base, 1, "")),
            s.bpmOf(s.parseStyle("bpm 140, rock")),
        ]));
    """)
    assert got == [
        "English, pop, piano, 120 BPM",
        "English, pop, piano, 200 BPM",
        "English, pop, piano",
        "pop, piano, 88 BPM",
        "Russian, pop, piano",
        "pop, soft female voice, piano",
        "English, pop, piano, staccato phrasing, 88 BPM",
        "pop, piano, staccato phrasing",
        "English, piano, pop, 88 BPM",
        "English, pop, piano, 88 BPM",
        "English, piano, 88 BPM",
        140,
    ]


@needs_node
def test_a_style_pasted_over_several_lines_becomes_one_line():
    got = run_sheet(r'''console.log(JSON.stringify([
        s.oneLine("Russian, rap,\nmale rap vocal\r\n\nheavy bass, 127 BPM"),
        s.oneLine("  one line already  "),
        s.oneLine("a,\nb"),
        s.oneLine(""),
    ]));''')
    assert got == ["Russian, rap, male rap vocal, heavy bass, 127 BPM", "one line already",
                   "a, b", ""]


@needs_node
def test_several_voices_are_one_style_part_joined_with_and():
    got = run_sheet("""
        const f = (parts) => s.formatStyle(parts);
        const duet = s.parseStyle("Russian, rap, male rap vocals and female melodic vocals, heavy bass, 105 BPM");
        const one = s.parseStyle("pop, male and female duet, piano");
        console.log(JSON.stringify([
            s.voicesOf(duet),
            s.voicesOf(one),
            f(s.setVoices(duet, ["male rap vocals"])),
            f(s.setVoices(duet, ["male rap vocals", "female melodic vocals", " backing choir, big "])),
            f(s.setVoices(duet, [])),
            f(s.setVoices(s.parseStyle("pop, piano"), ["soft female voice", "", "deep male voice"])),
        ]));
    """)
    assert got == [
        ["male rap vocals", "female melodic vocals"],
        ["male and female duet"],
        "Russian, rap, male rap vocals, heavy bass, 105 BPM",
        "Russian, rap, male rap vocals and female melodic vocals and backing choir big, heavy bass, 105 BPM",
        "Russian, rap, heavy bass, 105 BPM",
        "pop, soft female voice and deep male voice, piano",
    ]


@needs_node
def test_lyrics_round_trip_through_blocks():
    written = writer.tidy_lyrics("[verse]\nOne\nTwo\n\nThree\n[chorus]\nFour\n\n\n")
    samples = [constants.DEFAULT_LYRICS, written, "untagged first\n\n[Verse 2]\nA\nB"]
    got = run_sheet("console.log(JSON.stringify({}.map((t) => s.formatLyrics(s.parseLyrics(t)))));"
                    .format(json.dumps(samples)))
    assert got == samples


@needs_node
def test_layout_offsets_count_code_points_like_the_server_does():
    lyrics = "[Verse]\nla \U0001f3b5 la\nsecond line"
    got = run_sheet("""
        const lay = s.layout(s.parseLyrics({lyrics}));
        const chars = s.points(lay.text);
        console.log(JSON.stringify(lay.lines.map((l) => [l.start, chars.slice(l.start, l.start + 11).join("")])));
    """.format(lyrics=json.dumps(lyrics)))
    points = list(lyrics)
    assert got == [[8, "".join(points[8:19])], [16, "".join(points[16:27])]]
    assert got[1][1] == "second line"


@needs_node
def test_lines_stanzas_and_sections_move_and_multiply():
    got = run_sheet("""
        const base = s.parseLyrics("[Verse]\\nA\\nB\\n\\nC\\nD\\n\\n[Chorus]\\nE");
        const f = (blocks) => s.formatLyrics(blocks);
        const down = s.moveLine(base, 0, 4, 1);
        const up = s.moveLine(base, 1, 0, -1);
        console.log(JSON.stringify([
            f(down.blocks), [down.b, down.l],
            f(up.blocks), [up.b, up.l],
            f(s.duplicateStanza(base, 0, 0)),
            f(s.removeStanza(base, 0, 3)),
            f(s.removeStanza(base, 0, 0)),
            f(s.duplicateBlock(base, 1)),
            f(s.moveBlock(base, 1, -1)),
            f(s.addBlock(base, 0, "Bridge")),
            f(s.duplicateLine(base, 1, 0)),
            f(s.removeLine(base, 0, 0)),
            f(s.setTag(base, 1, s.cycleTag("Chorus"))),
        ]));
    """)
    assert got == [
        "[Verse]\nA\nB\n\nC\n\n[Chorus]\nD\nE", [1, 0],
        "[Verse]\nA\nB\n\nC\nD\nE\n\n[Chorus]", [0, 5],
        "[Verse]\nA\nB\n\nA\nB\n\nC\nD\n\n[Chorus]\nE",
        "[Verse]\nA\nB\n\n[Chorus]\nE",
        "[Verse]\nC\nD\n\n[Chorus]\nE",
        "[Verse]\nA\nB\n\nC\nD\n\n[Chorus]\nE\n\n[Chorus]\nE",
        "[Chorus]\nE\n\n[Verse]\nA\nB\n\nC\nD",
        "[Verse]\nA\nB\n\nC\nD\n\n[Bridge]\n\n\n[Chorus]\nE",
        "[Verse]\nA\nB\n\nC\nD\n\n[Chorus]\nE\nE",
        "[Verse]\nB\n\nC\nD\n\n[Chorus]\nE",
        "[Verse]\nA\nB\n\nC\nD\n\n[Bridge]\nE",
    ]


@needs_node
def test_one_click_steps_through_the_common_tags():
    got = run_sheet('console.log(JSON.stringify(["Verse", "Pre-Chorus", "Chorus", "Bridge", '
                    '"Outro", "Intro", "Hook", "Rap"].map(s.cycleTag)));')
    assert got == ["Pre-Chorus", "Chorus", "Bridge", "Outro", "Intro", "Verse", "Verse", "Verse"]


@needs_node
def test_a_letter_click_flips_its_case_and_a_second_click_flips_it_back():
    got = run_sheet("""
        const once = s.toggleCase("record", 3);
        console.log(JSON.stringify([
            once, s.toggleCase(once, 3),
            s.toggleCase("stra\\u00dfe", 4),
            s.toggleCase("88 BPM", 0),
            s.toggleCase("\\u{1F3B5} la", 2),
            s.toggleCase("\\u0434\\u043e\\u0440\\u043e\\u0433\\u0430", 3),
            [0, 1, 3, 5].map((i) => s.isInnerCapital("RecORD x", i)),
        ]));
    """)
    assert got == [
        "recOrd", "record",
        "stra\u00dfe",
        "88 BPM",
        "\U0001f3b5 La",
        "\u0434\u043e\u0440\u041e\u0433\u0430",
        [False, False, True, True],
    ]


@needs_node
def test_the_summary_counts_sung_lines_per_section():
    got = run_sheet("console.log(JSON.stringify(s.summarize({}, {})));".format(
        json.dumps(constants.DEFAULT_STYLE), json.dumps(constants.DEFAULT_LYRICS)))
    assert got["sung"] == constants.sung_lines(constants.DEFAULT_LYRICS)
    assert [(x["name"], x["lines"]) for x in got["sections"]] == [("Verse", 2), ("Chorus", 2)]
