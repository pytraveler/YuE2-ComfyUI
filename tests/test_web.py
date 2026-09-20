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


def js_number(name: str) -> int:
    """A single number exported by the roll, read out of its source."""
    source = (WEB / "yue2_roll.js").read_text(encoding="utf-8")
    match = re.search(r"export const " + name + r" = (\d+);", source)
    assert match, name + " is not exported as a number"
    return int(match.group(1))


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


STYLES = WEB / "yue2_styles.js"


def run_styles(script: str):
    """Run ``script`` with the examples as ``x`` and the sheet as ``s``."""
    program = "import * as x from {};\nimport * as s from {};\n{}".format(
        json.dumps(STYLES.as_uri()), json.dumps(SHEET.as_uri()), script)
    done = subprocess.run([NODE, "--input-type=module", "-e", program],
                          capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@needs_node
def test_the_examples_are_named_once_each_and_come_from_one_of_two_places():
    """The picker groups them by where they were run, so both groups must exist.

    The names are what the list shows; two examples with one name would read as
    a duplicate row nobody could tell apart.
    """
    got = run_styles("console.log(JSON.stringify(x.EXAMPLES));")
    assert len(got) >= 20
    assert len({one["name"] for one in got}) == len(got)
    assert {one["from"] for one in got} == {"demo", "gallery"}
    for one in got:
        assert one["style"].strip() == one["style"] and len(one["style"]) > 20
        assert one["language"] in ["", *js_array("LANGUAGES")]


@needs_node
def test_every_example_style_is_what_the_editor_gives_back():
    """An example the chips cannot hold would be quietly rewritten on the way out.

    Spacing after a comma is the one difference allowed: the gallery lines were
    typed without it, and the editor writes its parts back the way this pack's
    writer spells them. Everything else about the spacing is checked too, because
    the tool that copies these lines in wraps them across source lines, and its
    first run cut "double-kick drums" in half at the hyphen and left the halves
    a space apart.
    """
    got = run_styles("""
        console.log(JSON.stringify(x.EXAMPLES.map((one) => {
            const parts = s.parseStyle(one.style);
            return { back: s.formatStyle(parts), empty: parts.filter((p) => !p.text.trim()).length,
                     kinds: parts.map((p) => p.kind) };
        })));
    """)
    styles = run_styles("console.log(JSON.stringify(x.EXAMPLES.map((one) => one.style)));")
    for style, one in zip(styles, got):
        assert one["back"] == re.sub(r"\s*,\s*", ", ", style)
        assert not re.search(r"[A-Za-z]- |\s,|  ", style), "spacing: " + style
        assert one["empty"] == 0
        for kind in ("language", "bpm", "voice"):
            assert one["kinds"].count(kind) <= 1


@needs_node
def test_a_voice_suggestion_reads_as_a_voice_and_a_sound_does_not():
    """The two datalists feed two different slots of the same line.

    A voice the parser does not recognise becomes an ordinary part as soon as
    the line is read back, and the Voices row loses what the user chose there.
    """
    got = run_styles("""
        const kind = (text) => s.parseStyle(text)[0].kind;
        console.log(JSON.stringify({
            voices: s.VOICES.filter((text) => kind(text) !== "voice"),
            sounds: s.SUGGESTIONS.filter((text) => kind(text) !== "other"),
        }));
    """)
    assert got == {"voices": [], "sounds": []}


def test_the_genre_names_are_offered_beside_the_sounds():
    """The gallery's own genre names, which the Sound box completes from."""
    source = STYLES.read_text(encoding="utf-8")
    match = re.search(r"export const GENRES = \[(.*?)\];", source, re.DOTALL)
    assert match, "GENRES is not exported as a literal array"
    genres = re.findall(r'"([^"]*)"', match.group(1))
    assert len(genres) >= 60
    assert genres == sorted(set(genres))
    assert '[...sheet.SUGGESTIONS, ...GENRES]' in (WEB / "yue2_editor.js").read_text(encoding="utf-8")


PIANO = WEB / "yue2_piano.js"
PIANO_DIR = WEB / "piano"
SCORE = WEB / "yue2_score.js"

PIANO_CEILING = 700 * 1024
"""What the sounds may weigh, all together.

They are the only binary this pack ships and the only part of it that can grow
by a megabyte from one flag of the tool that writes them, so the limit is here
rather than in a reviewer's memory.
"""


def piano_module():
    """The pitch list as the browser reads it, without starting Node."""
    source = PIANO.read_text(encoding="utf-8")
    match = re.search(r"export const PITCHES = \[(.*?)\];", source, re.DOTALL)
    assert match, "PITCHES is not exported as a literal array"
    return [int(found) for found in re.findall(r"\d+", match.group(1))]


def test_every_key_of_the_roll_has_a_sound_within_reach():
    """A gap here is silence in the editor, or a note played at the wrong speed.

    Salamander is sampled in minor thirds, so one semitone of stretching covers
    the keyboard. Its middle C is the region written without pitch_keycenter,
    which a reader that insists on the opcode drops without a word, and the hole
    would be exactly where most songs sit.
    """
    source = PIANO.read_text(encoding="utf-8")
    reach = int(re.search(r"export const REACH = (\d+);", source).group(1))
    pitches = piano_module()
    assert pitches == sorted(set(pitches))
    low, high = js_number("LOWEST"), js_number("HIGHEST")
    for pitch in range(low, high + 1):
        near = min(abs(pitch - have) for have in pitches)
        assert near <= reach, "nothing within {} semitones of MIDI {}".format(reach, pitch)


def test_the_module_and_the_files_on_disk_say_the_same_thing():
    """A sound the list does not name is never fetched; a name with no file is a 404."""
    on_disk = sorted(int(path.stem) for path in PIANO_DIR.glob("*.ogg"))
    assert on_disk == piano_module()
    assert sum(path.stat().st_size for path in PIANO_DIR.glob("*.ogg")) <= PIANO_CEILING


def test_the_sounds_are_ogg_vorbis_in_one_channel():
    """Read from the file rather than trusted from the tool that wrote it.

    A stereo copy weighs twice as much for a preview nobody pans, and a file
    that is not Ogg at all would only show up as a decode error in a browser.
    """
    for path in sorted(PIANO_DIR.glob("*.ogg")):
        head = path.read_bytes()[:64]
        assert head[:4] == b"OggS", path.name
        at = head.index(b"vorbis")
        assert head[at - 1] == 1, "{}: not a vorbis identification header".format(path.name)
        assert head[at + 10] == 1, "{}: {} channels".format(path.name, head[at + 10])
        rate = int.from_bytes(head[at + 11:at + 15], "little")
        assert 16000 <= rate <= 48000, "{}: {} Hz".format(path.name, rate)


def test_the_player_strikes_the_piano_and_keeps_the_synth_for_a_bad_install():
    """The sounds are fetched, so they can fail to arrive; silence is not an option."""
    source = SCORE.read_text(encoding="utf-8")
    assert 'import { PITCHES } from "./yue2_piano.js";' in source
    assert "createBufferSource()" in source and "playbackRate" in source
    assert "createOscillator()" in source, "the fallback synth is gone"
    assert "PIANO.size" in source, "nothing chooses between the two"


ROLL = WEB / "yue2_roll.js"

ROLL_SCORE = ('X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=120\n'
              'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
              'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
              'K:F\n% verse\nV: Vocal\n"F"F16A16|"Bb"B16d16|Z2|\n'
              'V: Ins\nZ|z16f16|Z2|\n'
              '% chorus\nV: Vocal\n"C"c32-|c16z16|\nV: Ins\nZ2|\n')
"""Flat key, a full rest over two bars, a tie across a barline and two sections."""


def run_roll(script: str):
    """Run ``script`` with the roll logic imported as ``r``; it must print one JSON value."""
    program = "import * as r from {};\n{}".format(json.dumps(ROLL.as_uri()), script)
    done = subprocess.run([NODE, "--input-type=module", "-e", program],
                          capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_section_names_offered_are_the_ones_the_model_writes():
    """A name the editor suggests and the transcriber never writes would be an invention."""
    from yue2_comfy.sheetsage import vocab
    source = ROLL.read_text(encoding="utf-8")
    match = re.search(r"export const SECTION_NAMES = \[(.*?)\];", source, re.DOTALL)
    assert match, "SECTION_NAMES is not exported as a literal array"
    assert sorted(re.findall(r'"([^"]*)"', match.group(1))) == sorted(vocab.STRUCTURE_LABELS)


def test_the_roll_and_the_score_writer_agree_on_their_limits():
    from yue2_comfy import notation
    assert js_number("SECTION_LONGEST") == notation.SECTION_LONGEST
    assert js_number("FINEST") == notation.FINEST


@needs_node
def test_the_roll_moves_a_section_without_touching_its_neighbours():
    got = run_roll("""
        const sheet = {unit: 16, per_quarter: 4, bpm: 120, notes: {Vocal: [], Ins: []}, chords: [],
            sections: [{name: "", bar: 0, bars: 1}, {name: "verse", bar: 1, bars: 2}]};
        const model = r.modelOf(sheet);
        const named = r.setSection(model, 0, "  Intro  ");
        console.log(JSON.stringify({
            read: model.sections, unit: model.unit, named: named.sections,
            moved: r.moveSection(named, 1, 2).sections,
            onto: r.moveSection(named, 1, 0),
            gone: r.removeSection(named, 0).sections,
            missing: r.removeSection(model, 0),
            spans: r.sectionSpans(named, 3),
            empty: r.setSection(model, 0, "   "),
            long: r.sectionName("x".repeat(41)),
            sent: r.sheetOf(named).sections,
        }));
    """)
    assert got["read"] == [{"bar": 1, "name": "verse"}]
    assert got["unit"] == 16
    assert got["named"] == [{"bar": 0, "name": "Intro"}, {"bar": 1, "name": "verse"}]
    assert got["moved"] == [{"bar": 0, "name": "Intro"}, {"bar": 2, "name": "verse"}]
    assert got["onto"] is None, "two sections cannot start at one bar"
    assert got["gone"] == [{"bar": 1, "name": "verse"}]
    assert got["missing"] is None, "removing a section that is not there is not an edit"
    assert got["spans"] == [{"name": "Intro", "bar": 0, "bars": 1},
                            {"name": "verse", "bar": 1, "bars": 2}]
    assert got["empty"] is None and got["long"] == ""
    assert got["sent"] == got["named"], "the sections travel to the server on the sheet"


@needs_node
def test_the_finer_grid_multiplies_every_tick_the_model_holds():
    got = run_roll("""
        const sheet = {unit: 16, per_quarter: 4, bpm: 120, sections: [],
            notes: {Vocal: [{start: 4, length: 2, pitch: 60}], Ins: []},
            chords: [{start: 4, name: "C"}]};
        console.log(JSON.stringify(r.sheetOf(r.scaledModel(r.modelOf(sheet), 2, 32))));
    """)
    assert got["notes"]["Vocal"] == [{"start": 8, "length": 4, "pitch": 60}]
    assert got["chords"] == [{"start": 8, "name": "C"}]
    assert got["unit"] == 32


def test_the_editor_edits_the_strip_and_offers_the_grid_the_score_cannot_hold_yet():
    """The strip is the only place sections can be touched, so it has to answer the pointer."""
    source = SCORE.read_text(encoding="utf-8")
    assert "sectionDown(event, px, tick)" in source and "openSectionInput" in source
    assert 'input.setAttribute("list", "yue2-s-sections")' in source
    for call in ("roll.setSection", "roll.removeSection", "roll.moveSection", "roll.sectionSpans"):
        assert call in source, call + " is never used, so that edit cannot be made"
    assert "finerGrid" in source and "roll.scaledModel" in source
    assert "SECTION_H" in source, "the strip has no band of its own to click in"


def test_the_roll_knows_the_chord_qualities_upstream_knows():
    from yue2_comfy.vendor.yue2_music import abc_tools
    source = ROLL.read_text(encoding="utf-8")
    match = re.search(r"export const QUALITIES = \[(.*?)\];", source, re.DOTALL)
    assert re.findall(r'"([^"]*)"', match.group(1)) == list(abc_tools.QUALITIES)


@needs_node
def test_the_roll_accepts_exactly_the_chords_upstream_accepts():
    from yue2_comfy.vendor.yue2_music import abc_tools
    names = ["C", "Dm7", "F#m7b5", "Bbmaj7/D", "Ebm(maj7)", "C##dim7", "Dbaug", "A7/E",
             "G7sus4", "Cmaj9", "C13", "H7", "C/", "c", "", "Am7/", "Cm(maj7", " Dm "]
    got = run_roll("console.log(JSON.stringify({}.map(r.isChord)));".format(json.dumps(names)))
    assert got == [abc_tools.CHORD.fullmatch(name.strip()) is not None for name in names]


@needs_node
def test_note_names_snaps_and_chord_tones():
    got = run_roll("""console.log(JSON.stringify([
        [r.noteName(61), r.noteName(61, true), r.noteName(60), r.noteName(21), r.noteName(108)],
        r.snapChoices(8).map((s) => s.ticks), r.snapChoices(4).map((s) => s.ticks),
        [r.snapTo(13, 4), r.snapDown(15, 4)],
        r.chordPitches("Am7/G"), r.chordPitches("Bb"), r.chordPitches("nope"),
        r.clock(125.9),
    ]));""")
    assert got == [["C#4", "Db4", "C4", "A0", "C8"], [8, 4, 2, 1], [4, 2, 1], [12, 12],
                   [43, 57, 60, 64, 67], [58, 62, 65], [], "2:05"]


@needs_node
def test_the_tempo_range_holds_the_scores_own_and_the_model_carries_it():
    """A transcription of something fast opens and closes without being pulled back to 200.

    The tempo rides on the model rather than beside it, so undo, redo and the
    write that follows every edit carry it without a second path of their own.
    """
    got = run_roll("""const sheet = {notes: {Vocal: [{start: 0, length: 8, pitch: 60}], Ins: []},
        chords: [], bpm: 210, per_quarter: 8, total: 32};
    const model = r.modelOf(sheet);
    console.log(JSON.stringify([
        r.tempoRange(120), r.tempoRange(210), r.tempoRange(20),
        [r.tempoOf(96, 120), r.tempoOf(96.4, 120), r.tempoOf(1000, 120), r.tempoOf(1, 120),
         r.tempoOf(215, 210), r.tempoOf("x", 120)],
        [model.bpm, r.sheetOf(model).bpm, r.sheetOf({ ...model, bpm: 96 }).bpm],
        r.secondsAt({ bpm: 120, per_quarter: 8 }, 960),
    ]));""")
    assert got == [{"low": 40, "high": 200}, {"low": 40, "high": 210}, {"low": 20, "high": 200},
                   [96, 96, 200, 40, 210, None], [210, 210, 96], 60]


@needs_node
def test_a_note_stacked_on_another_is_offered_as_the_chord_it_most_likely_starts():
    """A chord drawn as stacked notes cannot live in a part, so the window offers a chord symbol.

    Only the intervals that name a chord from its lower note are guessed; the rest
    offer an empty chord box. Whatever is guessed must be a chord upstream accepts.
    """
    from yue2_comfy.vendor.yue2_music import abc_tools
    pairs = [[60, 64, False], [64, 60, False], [57, 60, False], [66, 72, False], [66, 72, True],
             [62, 72, False], [60, 71, False], [60, 62, False], [60, 72, False], [55, 60, False]]
    got = run_roll("console.log(JSON.stringify({}.map(([one, other, flats]) => r.chordGuess(one, other, flats))));"
                   .format(json.dumps(pairs)))
    assert got == ["C", "C", "Am", "F#dim", "Gbdim", "D7", "Cmaj7", "", "", ""]
    assert all(abc_tools.CHORD.fullmatch(name) for name in got if name)


def roll_sheet():
    from yue2_comfy import notation
    return notation.read(ROLL_SCORE)


@needs_node
def test_the_sheet_survives_the_roll_and_comes_back_untouched():
    from yue2_comfy import notation
    sheet = roll_sheet()
    got = run_roll("console.log(JSON.stringify(r.sheetOf(r.modelOf({}))));".format(json.dumps(sheet)))
    assert notation.write(ROLL_SCORE, got) == {"abc": ROLL_SCORE.strip(), "bars": []}


@needs_node
def test_notes_cannot_overlap_leave_the_song_or_move_apart():
    sheet = roll_sheet()
    got = run_roll("""
        const sheet = {sheet};
        const m = r.modelOf(sheet);
        const vocal = m.notes.Vocal;
        const total = sheet.total;
        console.log(JSON.stringify([
            r.addNote(m, "Vocal", 8, 8, 60, total),
            r.addNote(m, "Vocal", 96, 16, 60, total) !== null,
            r.addNote(m, "Vocal", total - 4, 8, 60, total),
            r.moveNotes(m, "Vocal", [vocal[0].id, vocal[1].id], 32, 2, total) === null,
            r.moveNotes(m, "Vocal", [vocal[2].id, vocal[3].id], 64, -1, total).notes.Vocal
                .map((n) => [n.start, n.pitch]),
            r.stretchNote(m, "Vocal", vocal[0].id, 24, total).notes.Vocal.slice(0, 2).map((n) => [n.start, n.length]),
            r.stretchNote(m, "Vocal", vocal[0].id, 8, total).notes.Vocal[0].length,
            r.deleteNotes(m, "Ins", [m.notes.Ins[0].id]).notes.Ins.length,
            r.notesIn(m, "Vocal", 0, 32, 60, 70).length,
            r.noteAt(m, "Vocal", 20, 69).pitch,
        ]));
    """.format(sheet=json.dumps(sheet)))
    assert got[0] is None and got[1] is True and got[2] is None
    assert got[3] is True
    assert got[4] == [[0, 65], [16, 69], [96, 69], [112, 73], [128, 72]]
    assert got[5] == [[0, 24], [24, 8]] and got[6] == 8
    assert got[7] == 0 and got[8] == 2 and got[9] == 69


@needs_node
def test_a_note_stretched_into_the_next_one_takes_time_from_it():
    """A part sings one note at a time, so a stretch moves the boundary with the next note.

    Refused, the stretch left a bar whose notes touch with no note that could be made
    longer. The next note keeps its end and at least one tick, the song's end is a wall,
    and a length that changes nothing hands back the same model, so a click on an edge
    records no edit.
    """
    from yue2_comfy import notation
    sheet = roll_sheet()
    got = run_roll("""
        const sheet = {sheet};
        const m = r.modelOf(sheet);
        const [f, , , d, c] = m.notes.Vocal;
        const spans = (model) => model.notes.Vocal.map((n) => [n.start, n.length]);
        const taken = r.stretchNote(m, "Vocal", f.id, 24, sheet.total);
        console.log(JSON.stringify({{
            taken: spans(taken),
            wall: spans(r.stretchNote(m, "Vocal", f.id, 40, sheet.total)),
            shorter: spans(r.stretchNote(m, "Vocal", f.id, 8, sheet.total)),
            gap: spans(r.stretchNote(m, "Vocal", d.id, 40, sheet.total)),
            end: spans(r.stretchNote(m, "Vocal", c.id, 100, sheet.total)),
            same: r.stretchNote(m, "Vocal", f.id, 16, sheet.total) === m,
            missing: r.stretchNote(m, "Vocal", 999, 8, sheet.total),
            bars: r.changedBars(sheet, taken),
            sheet: r.sheetOf(taken),
        }}));
    """.format(sheet=json.dumps(sheet)))
    kept = [[32, 16], [48, 16], [128, 48]]
    assert got["taken"] == [[0, 24], [24, 8]] + kept
    assert got["wall"] == [[0, 31], [31, 1]] + kept
    assert got["shorter"] == [[0, 8], [16, 16]] + kept
    assert got["gap"] == [[0, 16], [16, 16], [32, 16], [48, 40], [128, 48]]
    assert got["end"] == [[0, 16], [16, 16], [32, 16], [48, 16], [128, 64]]
    assert got["same"] is True and got["missing"] is None
    assert notation.write(ROLL_SCORE, got["sheet"])["bars"] == got["bars"] == [0]


@needs_node
def test_the_room_for_a_click_runs_to_the_next_note_or_the_end_of_the_song():
    """A click in a gap shorter than the last note drawn gets a note cut to the gap, not a refusal."""
    sheet = roll_sheet()
    got = run_roll("""
        const sheet = {sheet};
        const m = r.modelOf(sheet);
        const ticks = [0, 4, 64, 100, 176, 191, 192, -1, 4.5];
        console.log(JSON.stringify([...ticks.map((tick) => r.roomAt(m, "Vocal", tick, sheet.total)),
                                    r.roomAt(m, "Ins", 0, sheet.total), r.roomAt(m, "Ins", 50, sheet.total)]));
    """.format(sheet=json.dumps(sheet)))
    assert got == [0, 0, 64, 28, 16, 1, 0, 0, 0, 48, 0]


@needs_node
def test_altgr_counts_as_alt_and_not_as_ctrl():
    """Windows reports AltGr, the right Alt of many keyboard layouts, as Ctrl and Alt together.

    Read as Ctrl, it selected notes instead of taking them off the grid.
    """
    got = run_roll("""
        const events = [{altKey: true}, {ctrlKey: true}, {metaKey: true}, {ctrlKey: true, altKey: true},
                        {getModifierState: (name) => name === "AltGraph"}, {shiftKey: true, ctrlKey: true}, {}];
        console.log(JSON.stringify(events.map((event) => r.modifiersOf(event))));
    """)
    assert [[x["alt"], x["ctrl"], x["shift"]] for x in got] == [
        [True, False, False], [False, True, False], [False, True, False], [True, False, False],
        [True, False, False], [False, True, True], [False, False, False]]


@needs_node
def test_the_bars_the_roll_marks_are_the_bars_the_server_rewrites():
    from yue2_comfy import notation
    sheet = roll_sheet()
    got = run_roll("""
        const sheet = {sheet};
        let m = r.modelOf(sheet);
        m = r.moveNotes(m, "Vocal", [m.notes.Vocal[2].id], 0, 2, sheet.total);
        m = r.setChord(m, 96, "Dm7");
        const added = r.addNote(m, "Ins", 72, 8, 65, sheet.total);
        console.log(JSON.stringify({{bars: r.changedBars(sheet, added.model), sheet: r.sheetOf(added.model),
                                    bad: r.setChord(m, 0, "Cmaj9"), cut: r.removeChord(m, 0).chords.length}}));
    """.format(sheet=json.dumps(sheet)))
    written = notation.write(ROLL_SCORE, got["sheet"])
    assert got["bars"] == written["bars"] == [1, 2, 3]
    assert got["bad"] is None and got["cut"] == 3


@needs_node
def test_rests_open_up_for_the_engraver_and_the_header_is_read():
    sheet = roll_sheet()
    got = run_roll("console.log(JSON.stringify([r.forNotation({abc}), r.headerFacts({abc})]));"
                   .format(abc=json.dumps(ROLL_SCORE)))
    assert "Z|Z|" in got[0] and "Z2" not in got[0]
    assert got[1] == {"key": "F", "meter": "4/4", "bpm": 120, "bars": len(sheet["bars"])}


@needs_node
def test_playback_starts_inside_a_held_note_and_holds_each_chord_to_the_next():
    sheet = roll_sheet()
    got = run_roll("""
        const sheet = {sheet};
        const m = r.modelOf(sheet);
        console.log(JSON.stringify([
            r.events(sheet, m, 136, {{Vocal: true, Ins: false, chords: false}}),
            r.events(sheet, m, 0, {{Vocal: false, Ins: false, chords: true}}).slice(0, 3),
            r.secondsAt(sheet, 32),
        ]));
    """.format(sheet=json.dumps(sheet)))
    tick = 60 / (120 * 8)
    assert got[0] == [{"at": 0, "length": 40 * tick, "pitch": 72, "part": "Vocal"}]
    assert got[1] == [{"at": 0, "length": 32 * tick, "pitch": p, "part": "chords"} for p in (53, 57, 60)]
    assert got[2] == 2.0


@needs_node
def test_history_undoes_and_redoes_in_order():
    got = run_roll("""
        const h = new r.History(2);
        h.push("a"); h.push("b"); h.push("c");
        const back = h.undo("d");
        const again = h.undo(back);
        const forward = h.redo(again);
        console.log(JSON.stringify([back, again, forward, h.undo(forward), h.undo("x"), h.canRedo]));
    """)
    assert got == ["c", "b", "c", "b", None, True]


@needs_node
def test_the_mark_of_the_words_comes_off_and_goes_on_as_the_server_does_it():
    """The editor puts the mark on and the node takes it off; a disagreement would sing a comment."""
    from yue2_comfy import edits
    words = edits.mark("style", "lyrics", "full")
    marked = edits.attach(ROLL_SCORE, words)
    texts = [ROLL_SCORE, marked, marked.replace("\n", "\r\n") + "\r\n",
             ROLL_SCORE + "%yue2-words not-a-mark\n", "", edits.attach("", words),
             "X:1\n%yue2-words " + words + "\nK:C"]
    got = run_roll("""
        const texts = {texts};
        console.log(JSON.stringify({{split: texts.map((t) => r.splitMark(t)),
                                    attached: [r.attachMark({score}, {words}), r.attachMark({score}, null)]}}));
    """.format(texts=json.dumps(texts), score=json.dumps(ROLL_SCORE), words=json.dumps(words)))
    assert got["attached"] == [marked, edits.attach(ROLL_SCORE, None)]
    assert [[x["score"], x["words"]] for x in got["split"]] == [
        [found.score, found.words] for found in map(edits.read, texts)]


@needs_node
def test_the_length_limit_is_the_ceiling_the_singing_stage_stops_at():
    """The red line on the piano roll has to fall where the song really stops."""
    lyrics = ["", "[Intro]\n[Outro]", "one line", constants.DEFAULT_LYRICS,
              "[Verse]\r\na\r\n\r\n  b  \n[Chorus]\n\tc", "line\n" * 40, "[Verse] sung anyway\n[x]",
              "[Only bass] Neon fades along the lane [Stab]\n[Solo] [Claps]\n[End",
              "one two three four five six seven eight nine ten eleven twelve\n" * 3,
              "\u041f\u0435\u0441\u043d\u044f \u043e \u0437\u0438\u043c\u0435\n[Male: \"Yeah...\"] ...",
              "\u4f60\u597d\u4e16\u754c\u6211\u4eec\u4e00\u8d77\u5531\u6b4c"]
    maxima = [0, 60, 12.5, -3]
    got = run_roll("""
        const lyrics = {lyrics};
        console.log(JSON.stringify({{
            lines: lyrics.map((text) => r.sungLines(text)),
            limits: {maxima}.flatMap((most) => lyrics.map((text) => r.lengthLimit(most, text).seconds)),
            unknown: r.lengthLimit(null, "a"),
            wired: [r.lengthLimit(0, null, 204), r.lengthLimit(60, null, 204), r.lengthLimit(0, null)],
        }}));
    """.format(lyrics=json.dumps(lyrics), maxima=json.dumps(maxima)))
    assert got["lines"] == [constants.sung_lines(text) for text in lyrics]
    assert got["limits"] == [constants.length_ceiling(most, text) for most in maxima for text in lyrics]
    assert got["unknown"] is None
    assert got["wired"] == [{"seconds": 204, "auto": True}, {"seconds": 60, "auto": False}, None]


@needs_node
def test_the_bars_past_the_limit_are_the_bars_that_start_after_it():
    """A bar the limit cuts through is still partly heard, so only the bars after it are named."""
    sheet = roll_sheet()
    got = run_roll("""
        const sheet = {sheet};
        console.log(JSON.stringify([r.limitTick(sheet, 5), r.barsAfter(sheet, 5), r.barsAfter(sheet, 6),
                                    r.barsAfter(sheet, sheet.seconds), r.barsAfter(sheet, 0)]));
    """.format(sheet=json.dumps(sheet)))
    assert sheet["seconds"] == 12.0
    assert got == [80, [3, 4, 5], [3, 4, 5], [], [0, 1, 2, 3, 4, 5]]


def test_the_browser_listens_for_the_notices_the_nodes_send():
    """A notice nobody listens for lands in the console as an unhandled message and nowhere else.

    That is how every warning and refusal toast of this pack went unseen until
    the score editor's first warning was looked for on screen.
    """
    from yue2_comfy import progress
    source = (WEB / "yue2_notices.js").read_text(encoding="utf-8")
    assert 'const NOTICES_EVENT = "{}";'.format(progress.NOTICES_EVENT) in source
    assert "api.addEventListener(NOTICES_EVENT," in source


@needs_node
def test_the_summary_counts_sung_lines_per_section():
    got = run_sheet("console.log(JSON.stringify(s.summarize({}, {})));".format(
        json.dumps(constants.DEFAULT_STYLE), json.dumps(constants.DEFAULT_LYRICS)))
    assert got["sung"] == constants.sung_lines(constants.DEFAULT_LYRICS)
    assert [(x["name"], x["lines"]) for x in got["sections"]] == [("Verse", 2), ("Chorus", 2)]


@needs_node
def test_an_edit_goes_on_the_node_only_when_it_differs_and_reads_back_as_written():
    """What Apply puts on a node is what the node reads back: the text and its mark, or no edit at all."""
    from yue2_comfy import edits
    track = edits.audio_mark(b"recording", 44100)
    cases = [["[Verse]\nla la\n", "[Verse]\n", track], ["  [Verse]\n\n", "[Verse]", track],
             ["", "[Verse]", track], ["[Chorus]\nhey", None, None], [ROLL_SCORE, ROLL_SCORE + "\n", track]]
    got = run_roll("console.log(JSON.stringify({}.map(([t, b, w]) => r.editValue(t, b, w))));".format(
        json.dumps(cases)))
    assert [(found.score, found.words) for found in map(edits.read, got)] == [
        ("[Verse]\nla la", track), ("", None), ("", None), ("[Chorus]\nhey", None), ("", None)]


def test_the_transcribe_node_is_named_and_heard_the_same_on_both_sides():
    """The editors find the node, its modes and its ui keys by name; a rename on one side loses them silently."""
    from yue2_comfy import edits, transcribe
    score = (WEB / "yue2_score.js").read_text(encoding="utf-8")
    editor = (WEB / "yue2_editor.js").read_text(encoding="utf-8")
    name = next(iter(transcribe.TRANSCRIBE_CLASSES))
    for source in (score, editor):
        assert 'const TRANSCRIBE = "{}";'.format(name) in source
        assert 'const TRACK_UI = "{}";'.format(edits.TRACK_UI) in source
    assert 'const LYRICS_UI = "{}";'.format(edits.LYRICS_UI) in editor
    assert 'const MARKS_UI = "{}";'.format(edits.MARKS_UI) in score
    assert re.search(r"const NODES = \[[^\]]*\bTRANSCRIBE\b[^\]]*\]", editor)
    assert 'const MODE = "mode";' in score
    assert "mode" in transcribe.YuE2Transcribe.INPUT_TYPES()["required"]


def test_the_midi_node_is_named_and_heard_the_same_on_both_sides():
    """The editors and the list find the node, its widgets, its ui key and its routes by name."""
    from yue2_comfy import edits, load_midi
    score = (WEB / "yue2_score.js").read_text(encoding="utf-8")
    editor = (WEB / "yue2_editor.js").read_text(encoding="utf-8")
    midi = (WEB / "yue2_midi.js").read_text(encoding="utf-8")
    routes = (ROOT / "yue2_comfy" / "routes.py").read_text(encoding="utf-8")
    name = next(iter(load_midi.MIDI_CLASSES))
    for source in (score, editor, midi):
        assert 'const LOAD_MIDI = "{}";'.format(name) in source
    assert re.search(r"const NODES = \[[^\]]*\bLOAD_MIDI\b[^\]]*\]", editor)
    assert 'const MIDI_UI = "{}";'.format(edits.MIDI_UI) in midi
    required = load_midi.YuE2LoadMidi.INPUT_TYPES()["required"]
    for constant, widget in (("MIDI", "midi"), ("MODE", "mode"), ("VOCAL", "vocal_track"),
                             ("INSTRUMENT", "instrument_track")):
        assert widget in required and 'const {} = "{}";'.format(constant, widget) in midi
    assert 'const VOCAL_TRACK = "vocal_track";' in score and 'const INSTRUMENT_TRACK = "instrument_track";' in score
    assert 'const TRACKS_ROUTE = "/yue2/midi/tracks";' in midi and '"/midi/tracks"' in routes
    assert 'const MIDI_ROUTE = "/yue2/score/midi";' in score and '"/score/midi"' in routes
    listed = re.search(r"export const MIDI_EXTENSIONS = \[(.*?)\];", ROLL.read_text(encoding="utf-8")).group(1)
    assert tuple(re.findall(r'"([^"]*)"', listed)) == load_midi.EXTENSIONS


@needs_node
def test_the_midi_list_and_file_names_read_as_the_node_describes_them():
    from yue2_comfy import load_midi
    names = ["a.mid", "B.KAR", "c.midi.txt", "d.RMI", "", "song.MIDI"]
    parts = [{"number": 1, "name": "Track 1", "family": "Brass", "notes": 78, "low": 55, "high": 91, "drums": False,
              "role": "voice", "why": "highest"},
             {"number": 2, "name": "", "family": "Bass", "notes": 1, "low": 40, "high": 40, "drums": False,
              "role": "instrument", "why": "chosen"},
             {"number": 3, "name": "Drumkit", "family": "Drums", "notes": 48, "low": 35, "high": 38, "drums": True,
              "role": "", "why": ""}]
    facts = {"bars": 12, "bpm": 97, "meter": "4/4", "key": "Gm", "seconds": 29.7, "voice_shift": -12, "karaoke": True}
    got = run_roll("console.log(JSON.stringify([{}.map(r.isMidiFile), {}.map(r.partLine), {}.map(r.partRole), "
                   "r.midiFacts({}), r.clock(29.7), [r.octaveMove(24), r.octaveMove(0)], "
                   "[r.midiFileName('YuE2 Load MIDI'), r.midiFileName('a/b:c*'), r.midiFileName(''), "
                   "r.midiFileName('{}')]]));".format(
                       json.dumps(names), json.dumps(parts), json.dumps(parts), json.dumps(facts), SONG_TITLE))
    listed, lines, roles, summary, clock, moves, files = got
    assert listed == [name.lower().endswith(load_midi.EXTENSIONS) for name in names]
    assert lines == ["1 Track 1 \u00b7 Brass \u00b7 78 notes \u00b7 G3\u2013G6", "2 Bass \u00b7 1 note \u00b7 E2\u2013E2",
                     "3 Drumkit \u00b7 Drums \u00b7 48 notes"]
    assert roles == ["voice (auto: highest line)", "instrument", "drums, not sung"]
    assert summary == " \u00b7 ".join(["12 bars", "97 BPM", "4/4", "Key Gm", clock, "voice down an octave",
                                      "karaoke words"])
    assert moves == ["up 2 octaves", ""]
    assert files == ["YuE2 Load MIDI.mid", "a_b_c.mid", "YuE2 score.mid", SONG_TITLE + ".mid"]


SONG_TITLE = "\u041f\u0435\u0441\u043d\u044f 1"


def test_the_editors_length_controls_match_what_the_writer_will_make():
    from yue2_comfy import notation
    score = (WEB / "yue2_score.js").read_text(encoding="utf-8")
    routes = (ROOT / "yue2_comfy" / "routes.py").read_text(encoding="utf-8")
    assert 'const LENGTH_ROUTE = "/yue2/score/length";' in score and '"/score/length"' in routes
    assert "const NEW_BARS = {};".format(notation.BLANK_BARS) in score
    assert "const NEW_BPM = {};".format(notation.BLANK_BPM) in score
    added = re.search(r"const ADD_BARS = \[([^\]]*)\];", score)
    assert added, "ADD_BARS is not a literal array"
    counts = [int(value) for value in re.findall(r"\d+", added.group(1))]
    assert counts and counts == sorted(counts)
    assert all(0 < count <= notation.MOST_BARS for count in counts)


def test_the_empty_editor_offers_a_score_to_start_from():
    score = (WEB / "yue2_score.js").read_text(encoding="utf-8")
    body = score[score.index("    fillEmpty(problem) {"):score.index("    refresh() {")]
    assert "Make an empty score" in body and "this.newScore()" in body
    assert body.index("this.source") < body.index("Make an empty score")
    assert "async newScore()" in score and "async addBars(" in score and "async relength(" in score
    assert "if (this.writePending) await this.write();" in score


def test_the_bar_strip_is_tall_enough_to_hit_and_the_lanes_still_stack():
    score = (WEB / "yue2_score.js").read_text(encoding="utf-8")
    numbers = dict(re.findall(r"const (SECTION_H|BAR_H|CHORD_H|ROW_H) = (\d+);", score))
    assert int(numbers["BAR_H"]) >= 28
    assert "const RULER_H = SECTION_H + BAR_H;" in score
    assert re.search(r"c\.fillRect\(Math\.round\(this\.x\(bar\.start\)\), SECTION_H \+ 1, 1, "
                     r"RULER_H - SECTION_H - 1\);", score)
    assert 'c.fillText("bar", 6, RULER_H - 6);' in score


def test_the_right_button_moves_the_playhead_and_drops_a_chord():
    score = (WEB / "yue2_score.js").read_text(encoding="utf-8")
    section = score[score.index("    sectionDown(event, px, tick) {"):score.index("    openSectionInput(")]
    assert "event.button === 2" in section and "this.movePlayhead(tick)" in section
    down = score[score.index("    pointerDown(event) {"):score.index("    movePlayhead(tick) {")]
    assert "if (event.button === 2) this.dropChord(px);" in down
    assert "movePlayhead(tick) {" in score and "dropChord(px) {" in score
    assert "roll.removeChord(this.model, found.start)" in score
    assert 'this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());' in score
def test_a_widget_the_editors_write_is_recorded_as_a_change_to_the_workflow():
    """Measured on a live ComfyUI: without this the page reloads to an empty score box.

    ComfyUI autosaves a draft of the open workflow and restores it on a reload,
    but only what its change tracker has seen. It sees a widget the person
    typed into; it does not see one an editor writes a second later, after the
    click that opened it is long over. So every write of ours says so itself.
    """
    controls = (WEB / "yue2_controls.js").read_text(encoding="utf-8")
    assert "export function graphChanged()" in controls
    assert "activeWorkflow?.changeTracker" in controls
    assert "captureCanvasState" in controls and "checkState" in controls
    setter = controls.split("export function setWidgetValue")[1].split("\n}")[0]
    assert "graphChanged();" in setter, "setWidgetValue writes without recording the change"

    for name in ("yue2_lora.js", "yue2_midi.js", "yue2_score.js"):
        source = (WEB / name).read_text(encoding="utf-8")
        assert "graphChanged" in source, name + " changes the node without recording it"
        assert "changeTracker" not in source, name + " should use the shared helper"

    for name, source in ((path.name, text) for path in modules() for text in
                         [path.read_text(encoding="utf-8")]):
        if name in ("yue2_controls.js", "yue2_roll.js", "yue2_sheet.js", "yue2_piano.js"):
            continue
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("widget.value =") or stripped.startswith("widget.value="):
                assert "graphChanged" in source, name + " sets a widget without recording it"
