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
def test_a_copy_from_the_section_view_is_lyrics_the_model_reads():
    """Copying rows of the lyrics editor gives its words, never the badges and buttons drawn between them.

    The letters picked in each row become the line, a stanza break picked whole
    stays a break, and a header comes along only when its chip was picked, so
    a copy over two sections pastes back as two sections.
    """
    lyrics = "[Verse]\nfirst line\nsecond line\n\nthird line\n\n[Chorus]\nla \U0001f3b5 la"
    got = run_sheet("""
        const base = s.parseLyrics({lyrics});
        const all = base.map((block) => ({{header: true,
            lines: Object.fromEntries(block.lines.map((line, l) => [l, [0, Infinity]]))}}));
        console.log(JSON.stringify([
            s.excerpt(base, all) === s.formatLyrics(base),
            s.excerpt(base, [{{header: false, lines: {{0: [6, 10], 1: [0, 3]}}}}]),
            s.excerpt(base, [{{header: false, lines: {{2: [0, Infinity], 3: [0, Infinity]}}}},
                             {{header: true, lines: {{0: [0, 4]}}}}]),
            s.excerpt(base, [{{header: false, lines: {{1: [0, Infinity], 2: [0, Infinity], 3: [0, 5]}}}}]),
            s.excerpt(base, [{{header: false, lines: {{2: [0, Infinity]}}}}]),
            s.excerpt(base, [null, {{header: true, lines: {{}}}}]),
        ]));
    """.format(lyrics=json.dumps(lyrics)))
    assert got == [
        True,
        "line\nsec",
        "third line\n\n[Chorus]\nla \U0001f3b5",
        "second line\n\nthird",
        "",
        "[Chorus]",
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


SOUNDS = WEB / "yue2_sounds.js"

FAKE_AUDIO = """
const made = { osc: 0, noise: 0, gain: 0, filter: 0, started: 0, stopped: 0 };
function node(kind) {
    return {
        connect: (next) => next, start: () => { made.started += 1; }, stop: () => { made.stopped += 1; },
        frequency: { value: 0, setValueAtTime: () => {}, exponentialRampToValueAtTime: () => {},
                     linearRampToValueAtTime: () => {} },
        gain: { value: 0, setValueAtTime: () => {}, linearRampToValueAtTime: () => {},
                exponentialRampToValueAtTime: () => {} },
        detune: { value: 0 }, Q: { value: 0 }, pan: { value: 0 }, type: "", loop: false, buffer: null, kind,
    };
}
const context = {
    sampleRate: 48000, currentTime: 0,
    createOscillator: () => { made.osc += 1; return node("osc"); },
    createBufferSource: () => { made.noise += 1; return node("noise"); },
    createGain: () => { made.gain += 1; return node("gain"); },
    createBiquadFilter: () => { made.filter += 1; return node("filter"); },
    createStereoPanner: () => node("pan"),
    createBuffer: () => ({ getChannelData: () => new Float32Array(8) }),
};
"""
"""Enough of a Web Audio context for the sounds to be built and counted in Node."""


def run_sounds(script: str):
    """Run ``script`` with the sounds imported as ``s`` and a fake audio context ready."""
    program = "import * as s from {};\n{}\n{}".format(json.dumps(SOUNDS.as_uri()), FAKE_AUDIO, script)
    done = subprocess.run([NODE, "--input-type=module", "-e", program],
                          capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_each_part_is_offered_the_piano_first_and_only_sounds_that_exist():
    """The list in the window and the list the player can build are one list.

    The parts start on the piano, the chords on held strings: a chord lasts until
    the next one, and a struck piano has died away long before that.
    """
    offered = run_sounds("console.log(JSON.stringify(s.CHOICES));")
    assert sorted(offered) == ["Ins", "Vocal", "chords"]
    defaults = s_defaults()
    for part, choices in offered.items():
        assert choices[0][0] == "piano", part + " does not start on the piano"
        assert defaults[part] in [id for id, _ in choices], part
    assert defaults == {"Vocal": "piano", "Ins": "piano", "chords": "strings"}
    assert [id for id, _ in offered["Ins"]].count("drums") == 1
    assert "drums" not in [id for id, _ in offered["Vocal"] + offered["chords"]]


def s_defaults():
    return run_sounds("console.log(JSON.stringify(s.DEFAULTS));")


def test_a_sound_a_part_does_not_offer_falls_back_to_its_own_first_sound():
    """A remembered choice outlives the list it was picked from."""
    assert run_sounds("console.log(JSON.stringify(["
                      "s.known('Ins', 'drums'), s.known('Vocal', 'drums'),"
                      "s.known('chords', 'pad'), s.known('Vocal', undefined),"
                      "s.known('chords', 'drums'), s.known('chords', 'piano'), s.known('Vocal', 'strings')]));"
                      ) == ["drums", "piano", "pad", "piano", "strings", "piano", "piano"]


def test_the_kit_names_one_drum_for_every_row_and_repeats_each_octave():
    kit = run_sounds("console.log(JSON.stringify(s.KIT));")
    assert len(kit) == 12 and len(set(kit)) == 12
    named = run_sounds("console.log(JSON.stringify([60, 62, 66, 72, 48, 61].map(s.drumName)));")
    assert named == ["Kick", "Snare", "Hat", "Kick", "Kick", kit[1]]


def test_every_sound_builds_and_stops_what_it_started():
    """A voice that never stops leaves the note sounding after Stop."""
    for kind in ("synth", "bass", "pluck", "pad", "strings", "drums"):
        counted = run_sounds(
            "const out = [];\n"
            "for (const pitch of [36, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 84]) {\n"
            "    out.push(...s.play(context, node('master'), 0, 0.5, pitch, "
            + json.dumps(kind) + ", 0.1));\n}\n"
            "console.log(JSON.stringify({ voices: out.length, made }));")
        assert counted["voices"] >= 14, kind + " played nothing"
        assert counted["made"]["started"] == counted["made"]["stopped"] == counted["voices"], kind


def test_the_window_remembers_the_sounds_without_trusting_the_browser():
    """Storage throws in a private window; the editor must open there anyway."""
    source = SCORE.read_text(encoding="utf-8")
    assert 'import * as sounds from "./yue2_sounds.js";' in source
    kept = source.split("function rememberedSounds()")[1].split("\n}")[0]
    assert "localStorage" in kept and "catch" in kept
    assert "localStorage" in source.split("function rememberSounds(")[1].split("\n}")[0]
    assert "widgets_values" not in source.split("rememberSounds(")[1][:400]


def test_the_window_says_the_sound_is_not_what_yue2_hears():
    """The one thing that would be a lie by omission: this picks no instrument for the song."""
    source = SCORE.read_text(encoding="utf-8")
    assert "never an instrument" in source
    for part in ("Vocal:", "Ins:", "chords:"):
        assert part in source.split("const SOUND_TITLE = {")[1].split("};")[0]


def test_the_keyboard_names_the_kit_only_while_the_drums_are_the_part_being_edited():
    source = SCORE.read_text(encoding="utf-8")
    rows = source.split("drumRows() {")[1].split("\n    }")[0]
    assert 'this.part === "Ins"' in rows and "sounds.DRUMS" in rows
    assert "drawKit(c, top, rows)" in source and "sounds.drumName(pitch)" in source


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


@needs_node
def test_metronome_follows_meter_tempo_and_a_start_inside_the_bar():
    result = run_roll("""
        const sheet = { bpm: 120, per_quarter: 4, bars: [
            {start: 0, length: 12, meter: "3/4"},
            {start: 12, length: 12, meter: "6/8"},
            {start: 24, length: 16, meter: "4/4"},
        ]};
        console.log(JSON.stringify([
            r.metronomeEvents(sheet, 5),
            r.metronomeEvents({...sheet, bpm: 60}, 5),
        ]));
    """)
    fast, slow = result
    assert fast[0] == {"at": 0.375, "accent": False}
    assert fast[1] == {"at": 0.875, "accent": True}
    assert len(fast) == 11
    assert sum(event["accent"] for event in fast) == 2
    assert [event["at"] for event in slow] == [2 * event["at"] for event in fast]


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


def plainest_roots(mode: str) -> list:
    """Each pitch class named as the key with the fewest accidentals, a tie going to the sharps.

    The rule transpose.py picks a moved key by, over upstream's own key table.
    """
    from yue2_comfy.vendor.yue2_music import abc_tools
    names = []
    for pitch_class in range(12):
        found = []
        for name, accidentals in abc_tools.KEYS.items():
            root, minor = name[:-1] if name.endswith("m") else name, name.endswith("m")
            shift = {"": 0, "#": 1, "b": -1}[root[1:]]
            if minor == (mode == "minor") and (abc_tools.NATURAL[root[0]] + shift) % 12 == pitch_class:
                found.append((abs(accidentals), -accidentals, root))
        names.append(min(found)[2])
    return names


@needs_node
def test_a_moved_chord_is_named_by_the_plainest_key_of_its_new_root():
    """Named from the new root alone, so two steps down spell what one move of two does."""
    got = run_roll("""console.log(JSON.stringify({
        major: r.MAJOR_ROOTS, minor: r.MINOR_ROOTS,
        some: [["Am", -2], ["C/E", -2], ["F#m7b5", 1], ["Bbmaj7", 12], ["Ebm", 0], ["G#dim", 2],
               ["Dbaug", 1], ["C7sus4", -1], ["Cm(maj7)", 3], [" D ", 1], ["x", 1], ["C", 0.5]]
            .map(([name, step]) => r.movedChord(name, step)),
        twice: r.movedChord(r.movedChord("C", -1), -1) === r.movedChord("C", -2),
        all: r.QUALITIES.flatMap((q) => ["C", "F#", "Bb"].flatMap((root) =>
            Array.from({length: 25}, (_, i) => r.movedChord(root + q + "/E", i - 12))))
            .every((name) => r.isChord(name)),
    }));""")
    assert got["major"] == plainest_roots("major")
    assert got["minor"] == plainest_roots("minor")
    assert got["some"] == ["Gm", "Bb/D", "Gm7b5", "Bbmaj7", "Ebm", "Bbdim", "Daug", "B7sus4",
                           "D#m(maj7)", "Eb", None, None]
    assert got["twice"] is True and got["all"] is True


@needs_node
def test_the_key_list_names_each_key_as_transpose_will_and_keeps_the_voice_where_it_can():
    """Each key of the song's mode is offered once, by the name 'transpose' gives the moved key.

    The move is the nearest one, -5 to +6, unless it would take the middle of the voice line out of
    the window the model's scores keep it in while the move the other way round keeps it inside.
    """
    from yue2_comfy import transpose
    from yue2_comfy.vendor.yue2_music import abc_tools
    keys = list(abc_tools.KEYS)
    got = run_roll("""console.log(JSON.stringify({{
        plain: {keys}.map((key) => r.keyChoices(key)),
        high: r.keyChoices("F#m", 80).map((c) => c.shift),
        low: r.keyChoices("C", 61).map((c) => c.shift),
        tritone: [r.keyChoices("C", 74), r.keyChoices("C", 68)].map((list) => list.find((c) => c.name === "F#").shift),
        none: [r.keyChoices("Dmix"), r.keyChoices(""), r.keyChoices(null)],
        words: Array.from({{length: 23}}, (_, i) => r.shiftWords(i - 11)),
        marks: [r.keyMove(0), r.keyMove(3), r.keyMove(-5)],
    }}));""".format(keys=json.dumps(keys)))
    for key, choices in zip(keys, got["plain"]):
        assert [choice["shift"] for choice in choices] == list(range(-5, 7)), key
        for choice in choices:
            expected = key if not choice["shift"] else transpose._interval(key, choice["shift"]).key(key)
            assert choice["name"] == expected, (key, choice)
    assert got["high"] == [-5, -4, -3, -2, -1, 0, 1, 2, -9, -8, -7, -6]
    assert got["low"] == [7, 8, 9, 10, -1, 0, 1, 2, 3, 4, 5, 6]
    assert got["tritone"] == [-6, 6]
    assert got["none"] == [[], [], []]
    assert got["words"] == [transpose.describe(step) for step in range(-11, 12)]
    assert got["marks"] == ["", "+3", "\u22125"]


@needs_node
def test_the_history_shows_what_undo_and_redo_would_hand_back():
    """A key change is kept as a whole window's state; the window asks which kind comes next."""
    got = run_roll("""const h = new r.History();
        const seen = [h.lastDone, h.lastUndone];
        h.push("a");
        h.push({keyed: true});
        seen.push(h.lastDone, h.lastUndone);
        seen.push(h.undo("c"), h.lastDone, h.lastUndone);
        console.log(JSON.stringify(seen));""")
    assert got == [None, None, {"keyed": True}, None, {"keyed": True}, "a", "c"]


def test_the_editor_moves_the_key_through_the_transpose_route_and_can_take_it_back():
    score = SCORE.read_text(encoding="utf-8")
    routes = (ROOT / "yue2_comfy" / "routes.py").read_text(encoding="utf-8")
    assert 'const TRANSPOSE_ROUTE = "/yue2/score/transpose";' in score and '"/score/transpose"' in routes
    assert "roll.keyChoices(" in score and "this.keyControl()" in score
    assert "this.history.lastDone?.keyed" in score and "this.history.lastUndone?.keyed" in score
    assert "The tempo, the key, the number of bars" in score, "the notes window of a sung song must say why"


@needs_node
def test_the_roll_warns_about_the_voice_where_the_midi_import_would_move_it():
    """One window for both: the middle of the voice line, as Load MIDI judges it."""
    from yue2_comfy.midi import score
    sheet = roll_sheet()
    got = run_roll("""
        const m = r.modelOf({sheet});
        const high = r.moveNotes(m, "Vocal", m.notes.Vocal.map((n) => n.id), 0, 12, {total});
        console.log(JSON.stringify([r.VOICE_WINDOW, r.voiceMiddle(m), r.voiceMiddle(high),
                                    r.voiceMiddle({{notes: {{Vocal: []}}}})]));
    """.format(sheet=json.dumps(sheet), total=sheet["total"]))
    assert got[0] == list(score.VOICE_WINDOW)
    pitches = sorted(n["pitch"] for n in sheet["notes"]["Vocal"])
    assert got[1] == (pitches[(len(pitches) - 1) // 2] + pitches[len(pitches) // 2]) / 2 == 70
    assert got[2] == 82 and got[3] is None


@needs_node
def test_both_parts_and_the_chords_above_them_move_as_one():
    """What 'Both' does: notes of either part and the picked chords, all or nothing.

    A box takes a chord the way it takes a note: by the stretch it lasts, here up
    to the next chord, so a box begun just after a barline still takes that bar's.
    """
    from yue2_comfy import notation
    sheet = roll_sheet()
    got = run_roll("""
        const sheet = {sheet};
        const m = r.modelOf(sheet);
        const total = sheet.total;
        const [, , bb, d] = m.notes.Vocal;
        const f = m.notes.Ins[0];
        const spans = (model) => [model.notes.Vocal.map((n) => [n.start, n.pitch]),
                                  model.notes.Ins.map((n) => [n.start, n.pitch]),
                                  model.chords.map((c) => [c.start, c.name])];
        const moved = r.moveTogether(m, [bb.id, d.id, f.id], [32], 32, -2, total);
        console.log(JSON.stringify({{
            moved: spans(moved),
            sheet: r.sheetOf(moved),
            onto: r.moveTogether(m, [bb.id, d.id], [], 96, 0, total),
            chordOnto: r.moveTogether(m, [], [0], 32, 0, total),
            chordsInRow: spans(r.moveTogether(m, [], [0, 32], 32, 0, total))[2],
            pastTheEnd: r.moveTogether(m, [], [128], 64, 0, total),
            still: r.moveTogether(m, [bb.id], [32], 0, 0, total) === m,
            chordOnly: spans(r.moveTogether(m, [], [128], 0, 5, total)),
            parts: [r.partOf(m, bb.id), r.partOf(m, f.id), r.partOf(m, 999)],
            boxed: [r.chordsIn(m, 0, 64), r.chordsIn(m, 64, 0), r.chordsIn(m, 1, 32), r.chordsIn(m, 100, 110),
                    r.chordsIn(m, 130, 140)],
            dropped: spans(r.deleteTogether(m, [bb.id, f.id], [0])),
        }}));
    """.format(sheet=json.dumps(sheet)))
    vocal, ins, chords = got["moved"]
    assert vocal == [[0, 65], [16, 69], [64, 68], [80, 72], [128, 72]]
    assert ins == [[80, 75]] and chords == [[0, "F"], [64, "Ab"], [128, "C"]]
    assert got["onto"] is None and got["chordOnto"] is None and got["pastTheEnd"] is None
    assert got["chordsInRow"] == [[32, "F"], [64, "Bb"], [128, "C"]]
    assert got["still"] is True
    assert got["chordOnly"][0][4] == [128, 72] and got["chordOnly"][2][2] == [128, "F"]
    assert got["parts"] == ["Vocal", "Ins", None]
    assert got["boxed"] == [[0, 32], [0, 32], [0], [32], [128]]
    assert got["dropped"][0] == [[0, 65], [16, 69], [48, 74], [128, 72]]
    assert got["dropped"][1] == [] and got["dropped"][2] == [[32, "Bb"], [128, "C"]]
    written = notation.write(ROLL_SCORE, got["sheet"])
    assert written["bars"] == [1, 2]
    back = notation.read(written["abc"])
    assert [(n["start"], n["pitch"]) for n in back["notes"]["Vocal"]] == [tuple(n) for n in vocal]
    assert [(n["start"], n["pitch"]) for n in back["notes"]["Ins"]] == [tuple(n) for n in ins]
    assert [(c["start"], c["name"]) for c in back["chords"]] == [tuple(c) for c in chords]


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
def test_a_second_press_counts_as_a_double_click_only_near_the_first_one_and_soon():
    """The roll draws on the press, so the gesture is read before a note is down.

    The browser's own dblclick arrives after both presses have already done
    their work; by then the first one has drawn a note. The editor asks this
    instead, while the press can still be turned into play or stop.
    """
    got = run_roll("""
        const last = {at: 1000, px: 200, py: 300};
        const tries = [[last, 1100, 200, 300], [last, 1399, 204, 296], [last, 1500, 200, 300],
                       [last, 1100, 212, 300], [last, 1100, 200, 312], [null, 1100, 200, 300]];
        console.log(JSON.stringify([r.DOUBLE_CLICK_MS, r.DOUBLE_CLICK_PX,
                                    ...tries.map((one) => r.isDoubleClick(...one))]));
    """)
    assert got == [400, 5, True, True, False, False, False, False]


@needs_node
def test_the_history_hands_back_the_last_step_without_offering_it_as_a_redo():
    """The double click undraws the note the first press drew, and that is not an undo.

    A Redo button lit by a gesture meant to play the song would read as a bug,
    so the step is taken off the stack instead of being undone. The draw wipes
    whatever was waiting to be redone, as any edit does, so the step hands that
    list back and the take-back puts it where it was.
    """
    got = run_roll("""
        const past = new r.History();
        past.push("first");
        const back = past.undo("second");
        const dropped = past.push("drawn");
        const taken = past.take(dropped);
        console.log(JSON.stringify({back, dropped, taken, canUndo: past.canUndo, canRedo: past.canRedo,
                                    empty: new r.History().take()}));
    """)
    assert got == {"back": "first", "dropped": ["second"], "taken": "drawn",
                   "canUndo": False, "canRedo": True, "empty": None}


def test_a_double_click_plays_the_song_and_leaves_no_note_behind():
    """Asked for by the user on 2026-09-21: two left clicks start and stop the sound.

    The bar strip already moves the marker on a press, so there the second
    press only toggles and the song starts from that bar. On the roll the
    first press has drawn a note, so the second takes it back before playing.

    The note is only taken back while it is still the last step on the stack,
    which is why an undo or a redo in between forgets it: without that, a
    Ctrl+Z between the two presses would have cost the step before it too.
    """
    source = SCORE.read_text(encoding="utf-8")
    down = source.split("pointerDown(event) {")[1].split("\n    }")[0]
    assert "roll.isDoubleClick(this.lastDown, at, px, py)" in down
    assert "event.button === 0 && !keys.alt && !keys.ctrl && !keys.shift" in down
    assert "if (again) this.togglePlay();\n            else this.movePlayhead(tick);" in down
    assert "this.takeBackDraw();" in down
    back = source.split("takeBackDraw() {")[1].split("\n    }")[0]
    assert "this.history.take(redo)" in back and "roll.DOUBLE_CLICK_MS" in back
    assert "if (kept && drag.drawn) this.drawn = { at: now(), redo: this.dropped };" in source
    assert "this.dropped = this.history.push(previous);" in source
    for step in ("undo() {", "redo() {"):
        assert "this.drawn = null;" in source.split(step)[1].split("\n    }")[0], step
    assert "this.drawn = null;" in source.split("commit(next, previous = this.model) {")[1].split("\n    }")[0]


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
    assert "this.removeChordAt(found);" in score and "roll.removeChord(this.model, span.start)" in score
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


EDITLIST = WEB / "yue2_editlist.js"
TRACK = WEB / "yue2_track.js"

EDIT_LISTS = [
    "",
    "   ",
    "[]",
    '[{"op": "retake", "bars": [12, 16], "seed": 5, "takes": 2, "take": 1}]',
    '[{"op": "cut", "bars": [0, 4]}]',
    '[{"op": "retake", "seconds": [1.5, 9.25], "seed": 7}]',
    '[{"op": "retake", "bars": [12, 16]}, {"op": "cut", "bars": [2, 3]}]',
    '[{"op": "cut", "bars": [2, 4], "seed": 42, "take": 0, "takes": 3}]',
    '[{"op": "retake", "bars": [1, 2], "take": null}]',
    "not json",
    "{}",
    '"[]"',
    '[{"op": "reverse", "bars": [1, 2]}]',
    '[{"op": "retake"}]',
    '[{"op": "retake", "bars": [1, 2], "seconds": [1.0, 2.0]}]',
    '[{"op": "cut", "bars": [4, 4]}]',
    '[{"op": "cut", "bars": [-1, 4]}]',
    '[{"op": "cut", "bars": [1.5, 4]}]',
    '[{"op": "cut", "bars": [true, 2]}]',
    '[{"op": "cut", "bars": true}]',
    '[{"op": "cut", "bars": [1, 2, 3]}]',
    '[{"op": "retake", "seconds": [1.0, Infinity]}]',
    '[{"op": "retake", "seconds": [NaN, 1.0]}]',
    '[{"op": "retake", "seconds": [5.0, 1.0]}]',
    '[{"op": "retake", "bars": [1, 2], "takes": 9}]',
    '[{"op": "retake", "bars": [1, 2], "takes": 2, "take": 2}]',
    '[{"op": "retake", "bars": [1, 2], "seed": "x"}]',
    '[{"op": "retake", "bars": [1, 2], "seed": null}]',
    '[{"op": "retake", "bars": [1, 2], "seed": -3}]',
    '[{"op": "words", "lines": [7, 8], "text": "and the night comes down"}]',
    '[{"op": "words", "lines": [7, 8], "text": "new", "bars": [12, 16], "seed": 5, "takes": 2, "take": 1}]',
    '[{"op": "words", "lines": [7, 8], "text": "new", "seconds": [1.5, 9.25]}]',
    '[{"op": "words", "lines": [7, 8]}]',
    '[{"op": "words", "text": "new"}]',
    '[{"op": "words", "lines": [8, 7], "text": "new"}]',
    '[{"op": "words", "lines": [7, 8], "text": 5}]',
    '[{"op": "words", "lines": [1.5, 8], "text": "new"}]',
    '[{"op": "words", "lines": [7, 8], "text": "new", "bars": [1, 2], "seconds": [1.0, 2.0]}]',
    '[{"op": "retake", "lines": [7, 8], "text": "new", "bars": [1, 2]}]',
    '[{"op": "notes", "score": "X:1", "seed": 5, "takes": 2, "take": 1}]',
    '[{"op": "notes", "score": "X:1", "bars": [4, 6], "seed": 5, "vary": 1.5}]',
    '[{"op": "notes", "score": "X:1", "lines": [1, 2], "text": "new"}]',
    '[{"op": "notes", "score": "X:1", "seconds": [1.0, 2.0]}]',
    '[{"op": "notes", "score": "X:1", "bars": [1, 2], "seconds": [1.0, 2.0]}]',
    '[{"op": "notes", "score": "X:1", "bars": [2, 2]}]',
    '[{"op": "notes", "seed": 1}]',
    '[{"op": "notes", "score": " \\u3000\\n"}]',
    '[{"op": "notes", "score": "\\u001c"}]',
    '[{"op": "notes", "score": 5}]',
    '[{"op": "retake", "bars": [1, 2], "score": "X:1"}]',
    '[{"op": "extend", "text": "[Bridge]\\nnine ten", "seed": 5, "takes": 2, "take": 1}]',
    '[{"op": "extend", "seed": 5}]',
    '[{"op": "extend", "text": null, "seed": 5, "vary": 1.5, "guide": 3.0}]',
    '[{"op": "extend", "text": "", "bars": [8, 10]}]',
    '[{"op": "extend", "text": "", "seconds": [1.0, 2.0]}]',
    '[{"op": "extend", "text": 5}]',
    '[{"op": "extend", "text": ["nine"]}]',
    '[{"op": "extend", "text": "nine", "lines": [1, 2], "score": "X:1"}]',
    '[{"op": "extend", "text": "' + "la " * 1400 + '"}]',
    '[{"op": "extend", "text": "' + "\\u0431" * 4000 + '"}]',
    '[{"op": "extend", "text": "' + "\\ud83c\\udfb5" * 2100 + '"}]',
    '[{"op": "move", "bars": [13, 22], "to": 5}]',
    '[{"op": "move", "bars": [5, 13], "to": 30, "seed": 5, "takes": 3, "take": 1}]',
    '[{"op": "move", "bars": [5, 13], "to": 2.0}]',
    '[{"op": "move", "bars": [5, 13]}]',
    '[{"op": "move", "bars": [5, 13], "to": null}]',
    '[{"op": "move", "bars": [5, 13], "to": 2.5}]',
    '[{"op": "move", "bars": [5, 13], "to": -1}]',
    '[{"op": "move", "bars": [5, 13], "to": true}]',
    '[{"op": "move", "bars": [5, 13], "to": "3"}]',
    '[{"op": "move", "seconds": [1.0, 2.0], "to": 3}]',
    '[{"op": "move", "to": 3}]',
    '[{"op": "move", "bars": [5, 13], "seconds": [1.0, 2.0], "to": 3}]',
    '[{"op": "move", "bars": [5, 5], "to": 3}]',
    '[{"op": "retake", "bars": [1, 2], "to": 3}]',
    '[{"op": "break", "to": 13, "length": 4, "seed": 5, "takes": 2, "take": 1}]',
    '[{"op": "break", "to": 13, "seed": 5}]',
    '[{"op": "break", "to": 13.0, "length": 2.0, "vary": 1.5, "guide": 3.0}]',
    '[{"op": "break", "to": 13, "length": 0}]',
    '[{"op": "break", "to": 13, "length": 17}]',
    '[{"op": "break", "to": 13, "length": 2.5}]',
    '[{"op": "break", "to": 13, "length": null}]',
    '[{"op": "break", "to": 13, "length": "4"}]',
    '[{"op": "break", "length": 4}]',
    '[{"op": "break", "to": null}]',
    '[{"op": "break", "to": -1}]',
    '[{"op": "break", "to": "3"}]',
    '[{"op": "break", "to": 13, "bars": [13, 17]}]',
    '[{"op": "break", "to": 13, "seconds": [1.0, 2.0]}]',
    '[{"op": "break", "to": 13, "takes": 9}]',
    '[{"op": "break", "to": 13, "length": 4, "at": -6, "seed": 5}]',
    '[{"op": "break", "to": 13, "at": 2.0}]',
    '[{"op": "break", "to": 13, "at": 0}]',
    '[{"op": "break", "to": 13, "at": null}]',
    '[{"op": "break", "to": 13, "at": 1.5}]',
    '[{"op": "break", "to": 13, "at": "2"}]',
    '[{"op": "break", "to": 13, "at": true}]',
    '[{"op": "retake", "bars": [1, 2], "length": 4}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "hip hop", "seed": 5, "takes": 2, "take": 1}]',
    '[{"op": "restyle", "bars": [4, 8], "also": [[12, 16]], "style": " rap \\u3000",'
    ' "melody": "new", "octave": -1, "vary": 1.5}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "x", "melody": null, "octave": null,'
    ' "also": null}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "x", "octave": 1.0}]',
    '[{"op": "restyle", "bars": [4, 8]}]',
    '[{"op": "restyle", "bars": [4, 8], "style": " \\u001c "}]',
    '[{"op": "restyle", "bars": [4, 8], "style": 5}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "' + "x" * 1001 + '"}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "' + "\\u0431" * 1000 + '"}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "x", "melody": "rap"}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "x", "octave": 2}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "x", "octave": true}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "x", "also": [[6, 10]]}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "x", "also": [[8, 12], [1, 3]]}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "x", "also": [[8]]}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "x", "also": [[8, 8]]}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "x", "also": [[8.5, 9]]}]',
    '[{"op": "restyle", "bars": [4, 8], "style": "x", "also": "8-12"}]',
    '[{"op": "restyle", "seconds": [1.0, 2.0], "style": "x"}]',
    '[{"op": "restyle", "style": "x"}]',
    '["retake"]',
    "[null]",
]

TRACK_GRID = {
    "seconds": 20.0, "offset": 0.5, "rate": 1.0, "by_voice": True,
    "bars": [0.5, 2.5, 4.5, 6.5, 8.5, 10.5],
    "beats": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
              8.5, 9.0, 9.5, 10.0],
    "sections": [{"name": "verse", "bar": 0, "bars": 2, "start": 0.5, "end": 4.5, "sung": 0.2},
                 {"name": "chorus", "bar": 2, "bars": 3, "start": 4.5, "end": 10.5, "sung": 4.1}],
}


def run_edits(script: str):
    """Run ``script`` with the edit list imported as ``e``; it must print one JSON value."""
    program = "import * as e from {};\n{}".format(json.dumps(EDITLIST.as_uri()), script)
    done = subprocess.run([NODE, "--input-type=module", "-e", program],
                          capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def spelled(edit):
    """One of the node's edits in the shape the window holds it in; a restyle's own fields only on a restyle."""
    shape = {"op": edit.op, "bars": None if edit.bars is None else list(edit.bars),
             "seconds": None if edit.seconds is None else list(edit.seconds),
             "seed": edit.seed, "takes": edit.takes, "take": edit.take,
             "vary": edit.vary, "guide": edit.guide, "fade": edit.fade,
             "lines": None if edit.lines is None else list(edit.lines), "text": edit.text,
             "score": edit.score, "to": edit.to, "length": edit.length, "at": edit.at}
    if edit.op == "restyle":
        shape.update(style=edit.style, melody=edit.melody, octave=edit.octave,
                     also=[list(stretch) for stretch in edit.also])
    return shape


@needs_node
@pytest.mark.parametrize("text", EDIT_LISTS, ids=[str(n) for n in range(len(EDIT_LISTS))])
def test_the_window_reads_an_edit_list_exactly_as_the_node_does(text):
    """Both sides read the field the window writes, so a list one takes and the other refuses is
    either an edit made in the window that the node will not sing, or a track that will not draw.

    Seeds are compared as they are: both sides take 0 to 2**53 - 1 and refuse the rest, so the
    list the node answers with is always one the window can read back. A whole number written
    as 2.0 is a whole number on both sides too.
    """
    from yue2_comfy.inpaint import track

    made = run_edits("console.log(JSON.stringify(e.readEdits({}, 2)));".format(json.dumps(text)))
    try:
        wanted = track.read(text, 2)
    except ValueError:
        assert made["error"], "the window takes a list the node refuses: " + text
        return
    assert not made["error"], "the window refuses a list the node takes: " + text
    assert made["edits"] == [spelled(edit) for edit in wanted]


@needs_node
def test_what_the_window_writes_is_what_the_node_sings():
    from yue2_comfy.inpaint import track

    written = run_edits("""
        const made = [
            e.editFor({from: 24.0, to: 32.0, first: 12, stop: 16}, "retake", 3, 831001),
            e.editFor({from: 40.0, to: 48.0, first: 20, stop: 24}, "cut", 2, 5),
            e.editFor({from: 4.5, to: 9.25, first: null, stop: null}, "retake", 9, 7),
        ];
        const kept = e.withTake(made, 0, 2);
        console.log(JSON.stringify([e.writeEdits(kept), e.writeEdits(e.dropLast(kept)),
            e.writeEdits(e.withTakes(kept, 0, 2))]));
    """)
    edits = track.read(written[0], 1)
    assert [spelled(edit) for edit in edits] == [
        {"op": "retake", "bars": [12, 16], "seconds": None, "seed": 831001, "takes": 3,
         "take": 2, "vary": None, "guide": None, "fade": None, "lines": None,
         "text": None, "score": None, "to": None, "length": None,
         "at": None},
        {"op": "cut", "bars": [20, 24], "seconds": None, "seed": 0, "takes": 1, "take": None,
         "vary": None, "guide": None, "fade": None, "lines": None, "text": None, "score": None,
         "to": None, "length": None,
         "at": None},
        {"op": "retake", "bars": None, "seconds": [4.5, 9.25], "seed": 7, "takes": 4,
         "take": None, "vary": None, "guide": None, "fade": None, "lines": None,
         "text": None, "score": None, "to": None, "length": None,
         "at": None},
    ]
    assert json.loads(track.written(edits)) == json.loads(written[0])
    assert len(track.read(written[1], 1)) == 2
    assert track.read(written[2], 1)[0].take is None, "a take kept that is no longer sung"


RESTYLE_GRID = {
    "bars": [2.0 * bar for bar in range(17)],
    "sections": [{"name": "verse", "bar": 0, "bars": 4, "start": 0.0, "end": 8.0},
                 {"name": "Chorus", "bar": 4, "bars": 4, "start": 8.0, "end": 16.0},
                 {"name": "verse", "bar": 8, "bars": 4, "start": 16.0, "end": 24.0},
                 {"name": "chorus", "bar": 12, "bars": 4, "start": 24.0, "end": 32.0}],
}
"""Two choruses, their names written two ways: the window takes them for one name, as the node does."""


@needs_node
def test_a_restyle_the_window_writes_is_the_one_the_node_sings():
    from yue2_comfy.inpaint import track

    made = run_edits("""
        const grid = %s;
        const chosen = e.selectBars(grid, 4, 8);
        const also = e.sameSections(grid, chosen);
        const edit = e.restyleFor(chosen, also, "  hip hop  ", "new", -1, 2, 77, { vary: 1.5 });
        console.log(JSON.stringify([e.writeEdits([edit]), also,
            e.sameSections(grid, e.selectBars(grid, 4, 7)),
            e.sameSections(grid, e.selectBars(grid, 0, 8)), e.describeEdit(edit, 0),
            e.describeEdit(e.restyleFor(chosen, [], "folk", "same", 0, 1, 5), 1)]));
    """ % json.dumps(RESTYLE_GRID))
    written, also, part, two, said, plain = made
    assert also == [[12, 16]], "the other chorus"
    assert part == [] and two == [], "only a whole section of one name has others like it"
    edit, = track.read(written)
    assert (edit.bars, edit.also, edit.style, edit.melody, edit.octave) == (
        (4, 8), ((12, 16),), "hip hop", "new", -1)
    assert (edit.seed, edit.takes, edit.vary) == (77, 2, 1.5)
    assert json.loads(track.written([edit])) == json.loads(written)
    assert said == ('1. Style "hip hop" in bars 5-8 and bars 13-16, a new melody, an octave down, '
                    'variety 1.50')
    assert plain == '2. Style "folk" in bars 5-8'


@needs_node
@pytest.mark.parametrize("seed", [9007199254740993, -3, -9223372036854775808])
def test_a_seed_the_window_cannot_write_back_is_refused_on_both_sides(seed):
    """A browser cannot hold a number past 2**53 without changing it, and a seed that changes
    is every take after it sung again; a negative seed the node used to fold landed past 2**53
    every time, so the list it answered with was one the window could not read back and the
    takes never came home. Both sides now refuse the same seeds with the same reason."""
    from yue2_comfy.inpaint import track

    text = '[{{"op": "retake", "bars": [1, 2], "seed": {}}}]'.format(seed)
    made = run_edits("console.log(JSON.stringify(e.readEdits({}, 1)));".format(json.dumps(text)))
    assert "2**53" in made["error"]
    with pytest.raises(ValueError, match="2\\*\\*53"):
        track.read(text, 1)


@needs_node
def test_a_drag_snaps_to_bars_and_with_alt_to_beats():
    """Bars are what a cut takes, so a drag lands on them; Alt is the way to a stretch that is
    not whole bars, and a cut of one of those is refused here as the node refuses it."""
    got = run_edits("""
        const grid = {grid};
        const bars = e.selectionBetween(grid, 2.9, 6.1, false);
        const beats = e.selectionBetween(grid, 2.9, 6.1, true);
        const tiny = e.selectionBetween(grid, 2.6, 2.7, false);
        const section = e.selectBars(grid, 2, 5);
        console.log(JSON.stringify({{
            bars: bars, beats: beats, tiny: tiny, section: section,
            said: e.describeSelection(bars), saidBeats: e.describeSelection(beats),
            cutBars: e.whyNotCut(bars, true), cutBeats: e.whyNotCut(beats, true),
            cutFree: e.whyNotCut(beats, false), late: e.whyNotEdit(bars, 2.0),
            barAt: [e.barAt(grid, 0.0), e.barAt(grid, 5.0), e.barAt(grid, 99.0)],
            here: e.sectionAt(grid, 5.0).name, count: e.barCount(grid),
            said1: e.describeEdit(e.editFor(bars, "cut", 1, 0), 0),
        }}));
    """.format(grid=json.dumps(TRACK_GRID)))
    assert got["bars"] == {"from": 2.5, "to": 6.5, "first": 1, "stop": 3}
    assert got["beats"] == {"from": 3.0, "to": 6.0, "first": None, "stop": None}
    assert got["tiny"] == {"from": 2.5, "to": 4.5, "first": 1, "stop": 2}
    assert got["section"] == {"from": 4.5, "to": 10.5, "first": 2, "stop": 5}
    assert got["said"] == "bars 2-3, 0:02-0:06"
    assert got["saidBeats"] == "0:03-0:06"
    assert got["cutBars"] == "" and "select bars" in got["cutBeats"].lower()
    assert got["cutFree"] == "" and "stops before" in got["late"]
    assert got["barAt"] == [0, 2, 4] and got["here"] == "chorus" and got["count"] == 5
    assert got["said1"] == "1. Cut of bars 2-3"


@needs_node
def test_the_window_writes_a_seed_the_node_takes_as_it_is():
    from yue2_comfy.constants import normalize_seed

    seeds = run_edits("""
        const made = [];
        for (let i = 0; i < 200; i += 1) made.push(e.newSeed());
        console.log(JSON.stringify([Math.min(...made), Math.max(...made), e.SEED_CEILING]));
    """)
    assert seeds[0] >= 0 and seeds[1] < seeds[2] <= 2 ** 31
    assert normalize_seed(seeds[1]) == seeds[1]


@needs_node
def test_notes_changed_far_apart_are_written_as_an_edit_a_place():
    """Two or more bars left alone between two changes is two edits, each sung by itself; the
    node takes only its own bars' music from the score each carries. Changes inside a selection
    of bars sing that selection whole."""
    from yue2_comfy.inpaint import track

    got = run_edits("""
        const apart = e.notesFor("X:1", [9, 2, 3, 6], 2, [11, 12, 13]);
        const near = e.notesFor("X:1", [2, 4], 3, [7]);
        const chosen = e.notesFor("X:1", [2, 4], 2, [8], {first: 1, stop: 6, from: 0, to: 1});
        const outside = e.notesFor("X:1", [2, 9], 2, [8, 9], {first: 1, stop: 6, from: 0, to: 1});
        const typed = e.notesFor("X:1", [], 2, [4], {first: 1, stop: 6, from: 0, to: 1});
        console.log(JSON.stringify({
            stretches: e.stretchesOf([9, 2, 3, 6, 6, -1]),
            apart: apart.map((edit) => [edit.bars, edit.seed, edit.takes]),
            near: near.map((edit) => edit.bars), chosen: chosen.map((edit) => edit.bars),
            outside: outside.map((edit) => edit.bars), typed: typed.map((edit) => edit.bars),
            written: e.writeEdits(apart), said: e.describeEdit(apart[0], 0),
            bare: e.describeEdit({...apart[0], bars: null}, 1),
        }));
    """)
    assert got["stretches"] == [[2, 4], [6, 7], [9, 10]]
    assert got["apart"] == [[[2, 4], 11, 2], [[6, 7], 12, 2], [[9, 10], 13, 2]]
    assert got["near"] == [[2, 5]]
    assert got["chosen"] == [[1, 6]]
    assert got["outside"] == [[2, 3], [9, 10]]
    assert got["typed"] == [None], "bars the window cannot know are the node's to find"
    assert [edit.score for edit in track.read(got["written"], 1)] == ["X:1"] * 3
    assert got["said"] == "1. New notes in bars 3-4"
    assert got["bare"] == "2. New notes in the bars they change"


@needs_node
def test_a_song_going_on_is_written_and_named_as_the_node_reads_it():
    """The box's words go on the list trimmed, a new ending alone is the empty box, and the list
    names the first line sung, since a song going on selects no stretch to name."""
    from yue2_comfy.inpaint import track

    got = run_edits("""
        const made = e.extendFor("  [Chorus]\\nhold on\\nsing it loud  ", 3, 17, {vary: 1.2});
        const bare = e.extendFor("", 9, 4);
        console.log(JSON.stringify({written: e.writeEdits([made, bare]),
            said: e.describeEdit(made, 0), ending: e.describeEdit(bare, 1),
            kept: e.describeEdit({...made, take: 0}, 2),
            last: e.lastSung("[Verse]\\none two\\n\\n[Chorus]\\nthree four\\nfive\\n\\n[Outro]\\n"),
            none: e.lastSung("[Intro]\\n\\n[Outro]"), bareWords: e.lastSung("one\\ntwo")}));
    """)
    edits = track.read(got["written"])
    assert [edit.op for edit in edits] == ["extend", "extend"]
    assert (edits[0].text, edits[0].takes, edits[0].seed, edits[0].vary) == (
        "[Chorus]\nhold on\nsing it loud", 3, 17, 1.2)
    assert (edits[1].text, edits[1].takes) == ("", 4)
    assert got["said"] == '1. Go on with "hold on" and 1 more line, variety 1.20'
    assert got["ending"] == "2. A new ending"
    assert got["kept"] == '3. Go on with "hold on" and 1 more line, take 1 of 3, variety 1.20'
    assert got["last"] == "[Chorus]\nthree four\nfive"
    assert got["none"] == "" and got["bareWords"] == "one\ntwo"


def test_the_two_sides_offer_the_same_number_of_takes():
    from yue2_comfy import edit_track
    from yue2_comfy.inpaint import track

    from yue2_comfy import notation

    source = EDITLIST.read_text(encoding="utf-8")
    assert "export const MAX_TAKES = {};".format(track.MAX_TAKES) in source
    assert "export const LONGEST_SCORE = {};".format(notation.LONGEST) in source
    from yue2_comfy.inpaint import ops

    assert "export const LONGEST_WORDS = {};".format(ops.LONGEST_WORDS) in source
    assert edit_track.YuE2EditTrack.INPUT_TYPES()["required"]["takes"][1]["max"] == track.MAX_TAKES


def test_the_track_window_names_the_node_its_widgets_and_its_ui_key_the_same_way():
    """The window finds the node and everything it hands over by name; the names live in Python."""
    from yue2_comfy import edit_track, edits

    source = TRACK.read_text(encoding="utf-8")
    assert 'const NODE = "{}";'.format(next(iter(edit_track.EDIT_CLASSES))) in source
    assert 'const TRACK_UI = "{}";'.format(edits.EDIT_TRACK_UI) in source
    shape = edit_track.YuE2EditTrack.INPUT_TYPES()
    for constant, widget in (("EDITS", "edits"), ("TAKES", "takes")):
        assert 'const {} = "{}";'.format(constant, widget) in source
        assert widget in shape["required"] or widget in shape["optional"]
    assert "queueNodeIds: [String(node.id)]" in source, "a run of ours runs this node alone"
    assert "await runAlone(this.node);" in source, "which is what Render does"
    doc = edit_track.YuE2EditTrack._payload.__doc__
    for key in ("peaks", "rms", "grid", "lyrics", "takes", "chosen", "dropped", "at", "kind",
                "edits", "seconds", "total", "song", "lines", "heard", "said", "mumbled"):
        assert re.search(r"\b(?:payload|take|drawn|entry)\??\." + key + r"\b", source), (
            key + " is written but never drawn")
        assert re.search(r"\b" + key + r"\b", doc), key + " is drawn but not written down"


def test_the_window_keeps_one_player_for_each_song_it_plays():
    """A player per file, made and kept in one place, and an interrupted play is no fault.

    The song the user edits is 3:44, which is 43 MB of WAV a take. The window
    used to make a new Audio every time anything was played, and each of them
    went on pulling its 43 MB down a connection nothing ever closed; swapping
    takes under the needle then ended in a live Stop button, a playhead that
    stood still and silence. Now each song has one player, kept so a take
    already fetched plays the moment it is chosen, and the oldest is let go of
    when there are more than the window needs. Changing the source of a
    playing element rejects its play promise with AbortError, which is not a
    fault to report either.
    """
    source = TRACK.read_text(encoding="utf-8")
    assert source.count("new Audio(") == 1, "players are made in one place"
    made = source.index("new Audio(")
    assert source.index("    sounder(url) {") < made < source.index("    release(sound) {"), (
        "the player is made where it is kept, not afresh in playSong")
    assert "const sound = this.sounder(url);" in source, "playSong asks for the player of that url"
    assert 'removeAttribute("src")' in source, "the fetch of a song no longer wanted is let go"
    assert '"AbortError"' in source, "a play we interrupted ourselves is not a failure"
    play = source.index("    playSong(")
    assert source.index("this.follow(sound, mine);", play) < source.index("sound.play()", play), (
        "the playhead must follow before the browser answers, or a song that never starts looks "
        "exactly like one that plays")


def test_the_end_of_a_selection_is_watched_by_more_than_the_animation_frame():
    """A window off screen gets no frames, and Play selected ran on past its end.

    So the end of a selection is armed as a timer when playing starts, and the
    element's own timeupdate -- which keeps coming when nothing is drawn --
    checks it too.
    """
    source = TRACK.read_text(encoding="utf-8")
    arming = source[source.index("    armStop() {"):source.index("    disarmStop() {")]
    assert "this.timer = setTimeout(" in arming and "clearTimeout(" in source
    assert "this.stopAtEnd()" in arming
    ticking = source[source.index("    ticked(sound) {"):source.index("    stopAtEnd() {")]
    assert "this.stopAtEnd()" in ticking, "the element's own clock watches for the end too"
    for event in ('"timeupdate"', '"playing"', '"loadedmetadata"', '"ended"', '"error"'):
        assert "addEventListener({}".format(event) in source, event


def test_escape_leaves_the_sound_the_message_and_the_selection_before_the_window():
    """Escape unwinds one thing at a time: it used to close the window over a red message."""
    source = TRACK.read_text(encoding="utf-8")
    ladder = source[source.index("handle.onEscape = () => {"):]
    ladder = ladder[:ladder.index("\n        };")]
    steps = ["this.stopSound();", "this.hushSound();", "this.clearPick();", "this.close();"]
    for step in steps:
        assert step in ladder, step + " is not on the way out"
    where = [ladder.index(step) for step in steps]
    assert where == sorted(where), "the sound, then the message, then the selection, then out"


SECTION_TAGS_READ = ["Verse", "verse 2", "CHORUS", "Hook", "refrain", "Pre_Chorus", "pre chorus",
                     "post chorus", "Bridge 3", "Outro:", "Intro #1", "interlude", "instrumental",
                     "solo", "fade-out", "rap", "wobble", "", "   ", "Verse-", "theme", "silence",
                     "loop", "PRE-CHORUS 2", "post-chorus", u"Chorus\u0662", u"chorus\uff12",
                     "chorus\x1c", "\x1fverse\x1f", u"\u041f\u0440\u0438\u043f\u0435\u0432"]

PAIRINGS = [
    ((["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "interlude", "verse",
       "chorus", "outro"], [3, 20, 24, 20, 24, 12, 24, 2, 20, 12, 4]),
     ["verse", "chorus", "verse", "chorus", "verse", "chorus", "bridge", "outro"]),
    ((["verse", "chorus"], [10, 10]), ["verse", "chorus"]),
    ((["intro", "verse", "chorus", "outro"], [0, 10, 10, 0]), ["verse", "chorus"]),
    ((["intro", "verse", "chorus", "outro"], [2, 10, 10, 1]), ["verse", "chorus"]),
    ((["intro", "verse", "chorus", "verse", "chorus", "outro"], [0, 9, 9, 9, 9, 0]),
     ["verse", "chorus", "verse", "chorus"]),
    ((["verse", "verse", "chorus"], [9, 9, 9]), ["verse", "chorus"]),
    ((["chorus", "verse"], [9, 9]), ["verse", "chorus"]),
    ((["intro", "verse", "chorus"], [1, 1, 0]), ["verse", "bridge"]),
    ((["intro", "verse", "chorus"], [0, 1, 0]), ["verse", "bridge"]),
    ((["intro", "verse", "outro"], [0, 5, 3]), ["verse", "chorus"]),
    (([], []), ["verse"]),
    ((["verse"], [1]), []),
    ((["intro"], [1]), ["verse"]),
    ((["intro"], [0]), ["verse"]),
]


@needs_node
def test_a_section_tag_is_read_the_same_way_on_both_sides():
    """A tag has to become the same section name here as it does there.

    The window lights the words under the play mark by laying the blocks of
    the lyrics on the sections of the score, which are named by the model, and
    the node lays them the same way to know which words a cut takes out. Two
    readings of '[Hook]' would light one block and drop another.
    """
    from yue2_comfy import phrasing

    got = run_edits("console.log(JSON.stringify({}.map(e.labelOf)));".format(
        json.dumps(SECTION_TAGS_READ)))
    assert got == [phrasing.label_of(tag) for tag in SECTION_TAGS_READ]


@needs_node
@pytest.mark.parametrize("scored,labels", PAIRINGS,
                         ids=[str(n) for n in range(len(PAIRINGS))])
def test_the_window_lays_the_words_on_the_score_as_the_node_does(scored, labels):
    """Wherever the node can pair the two, the window pairs them the same.

    The node goes by name first and by the count of sections with vocal notes
    second, and the grid now carries that count, so the window can go the same
    way; it is the node's answer whenever the node has one. The node gives up
    where neither lines up; the window then keeps guessing, because all it
    does with the answer is light a block of words, while the node would be
    deciding which words a cut throws away, and only the shape of that guess
    is checked here.
    """
    from yue2_comfy.inpaint import ops

    names, notes = scored
    theirs = ops._paired(labels, [(name, count, 0) for name, count in zip(names, notes)])
    ours = run_edits("console.log(JSON.stringify(e.pairSections({}, {}, {})));".format(
        json.dumps(names), json.dumps(labels), json.dumps(notes)))
    if theirs is not None:
        assert ours == theirs
        return
    assert ours is None or (len(ours) <= len(labels) and sorted(set(ours)) == ours
                            and all(0 <= at < len(names) for at in ours))


@needs_node
def test_the_window_counts_a_block_as_sung_the_way_the_node_does():
    """A block whose lines have no letter in them is not a sung block on either side."""
    from yue2_comfy.inpaint import ops

    lyrics = "[Verse]\nla la\n[Instrumental]\n---\n[Chorus]\nsing sing\n[Verse 2]\n1 2 3 4\n"
    theirs = [ops.phrasing.label_of(block["tag"]) if block["tag"] else "verse"
              for block in ops._blocks(lyrics) if block["lines"]]
    ours = run_sheet("""
        const blocks = s.parseLyrics({});
        console.log(JSON.stringify(blocks.filter((b) => b.lines.some((l) => /\\p{{L}}/u.test(l)))
            .map((b) => b.tag)));
    """.format(json.dumps(lyrics)))
    assert [tag.lower() for tag in ours] == theirs


@needs_node
def test_a_byte_order_mark_and_a_whole_number_spelled_as_a_float_read_alike():
    """Pasted text can start with U+FEFF and a saved workflow can spell 4 as 4.0."""
    from yue2_comfy.inpaint import track

    for text in ('\ufeff[{"op": "cut", "bars": [0, 4]}]',
                 '[{"op": "retake", "bars": [0.0, 4.0], "seed": 3.0, "takes": 2.0}]'):
        made = run_edits("console.log(JSON.stringify(e.readEdits({}, 2)));".format(
            json.dumps(text)))
        theirs = track.read(text, 2)
        assert not made["error"] and made["edits"] == [spelled(edit) for edit in theirs]


def test_the_window_plays_through_the_packs_own_route_and_falls_back_once():
    """The players ask /yue2/sound for a sound the node wrote, and /view only when that route is not there.

    /view sends a file through sendfile, and on client editions of Windows one
    take held open by a browser starves every other file body the server has
    to send; measured on the user's own server on 2026-09-22. The pack's route
    writes plain chunks. A page newer than its server gets a 404 from the route
    and must still play, so the players drop to /view once and say nothing.
    """
    source = TRACK.read_text(encoding="utf-8")
    assert 'const SOUND_ROUTE = "/yue2/sound";' in source
    assert 'const SOUND_SUBFOLDER = "yue2_edit";' in source, "the folder the node writes into"
    assert source.count("soundUrl(") == 3, "one builder, asked by the song and by the warmed takes"
    assert "viewUrl(" not in source, "nothing goes to /view first"
    url = source[source.index("function soundUrl("):source.index("function takeFacts(")]
    assert "entry.subfolder === SOUND_SUBFOLDER" in url, "only the node's own files take the route"
    assert 'route = "/view?"' in url, "everything else, and the fallback, goes to /view"
    failed = source[source.index("    soundFailed(sound) {"):source.index("    playPick() {")]
    assert "this.viaView = true;" in failed and "this.dropSound();" in failed, (
        "on a missing route every player is let go and the song asked for again through /view")
    assert "(code === 2 || code === 4)" in failed, "a network or unsupported-source fault, not a decode one"
    assert "this.playSong(at, until);" in failed, "from the same place, with the same end"


def test_play_starts_at_the_mark_and_a_stalled_player_is_nudged_once():
    """A click on the wave puts the mark there and Play starts at it, selection or not; the end starts over.

    The user clicked outside the selection, pressed Play and heard the
    selection's start: the selection came first. Play selected is the button
    for the selection. And a player whose time stands still is seeked in place
    once, which makes the browser ask for the bytes again, before the red line.
    """
    source = TRACK.read_text(encoding="utf-8")
    start = source[source.index("    startAt() {"):source.index("    sounder(url) {")]
    mark = start.index("this.playhead > 0.05")
    assert mark < start.index("if (this.selection) return this.selection.from;"), (
        "the mark comes before the selection")
    assert "this.playhead < this.total() - 0.1" in start, "a mark at the end starts over"
    follow = source[source.index("    follow(sound, mine) {"):source.index("    whyNotMore(")]
    assert "let nudged = false;" in follow
    assert "this.seekTo(now);" in follow, "the nudge is a seek to where it stands"
    assert follow.index("this.seekTo(now);") < follow.index("this.saySound(STUCK_SAID, true);"), (
        "the nudge comes before the red line")
    assert "nudged = false;" in follow[follow.index("if (now !== last) {"):follow.index("} else if")], (
        "a player that moved again may be nudged again later")


def test_a_take_the_session_never_sang_is_a_row_the_window_can_ask_for():
    """A saved list asks for four takes; a fresh session has the one it kept. The rest are offers.

    The node sings only the take the list keeps, so the window has to show the
    others as what they are: seeds with no sound. They get no radio, because
    there is nothing to switch the track to, and a button that puts one on the
    list instead.
    """
    source = TRACK.read_text(encoding="utf-8")
    facts = source[source.index("function takeFacts("):source.index("function growNode(")]
    assert "if (take.sung === false) return NOT_SUNG;" in facts, (
        "a take with no sound has no length and no join to show")
    shown = source[source.index("    shownTake() {"):source.index("    wave() {")]
    assert "take.sung === false ? null : take" in shown, (
        "the track never follows a take that was not sung")
    takes = source[source.index("    paintTakes() {"):source.index("    paintWords() {")]
    assert "const gone = take.sung === false;" in takes
    loop = takes[takes.index("for (const take of takes) {"):]
    assert loop.index("if (gone) {") < loop.index("box.type = \"radio\";"), (
        "a take that was not sung gets no radio")
    assert "GONE_WITH_SESSION" in takes, "the row says why it is not there"
    assert "this.singTake(take.index)" in takes
    sing = source[source.index("    singTake(index) {"):source.index("    moreTakes() {")]
    assert "list.withTake(edits, last, index)" in sing, "it keeps that take instead"
    assert "this.render();" in sing, "and it is sung at once, not on a later Render"


SONGS = WEB / "yue2_songs.js"


def test_the_song_as_it_was_stands_first_among_the_takes_of_a_retake():
    """Hearing the retake against what it replaced is the whole point of asking for one."""
    source = TRACK.read_text(encoding="utf-8")
    takes = source[source.index("    paintTakes() {"):source.index("    paintWords() {")]
    assert takes.index("const was = this.payload.before;") < takes.index("for (const take of"), (
        "what the retake replaced comes first, above the takes of it")
    assert "this.showTake(BEFORE)" in takes and "WAS_NAME" in takes
    shown = source[source.index("    shownTake() {"):source.index("    wave() {")]
    assert "if (this.take === BEFORE) return this.payload.before || null;" in shown, (
        "so the track, the bars, the sound and the mark all follow it like a take")
    show = source[source.index("    showTake(index) {"):source.index("    dropLast() {")]
    assert "if (mine && index !== BEFORE)" in show, "choosing it writes no take on the list"
    work = source[source.index("    nextWork() {"):source.index("    paintNext() {")]
    assert "this.take === BEFORE" in work and "NEXT_DROP" in work, (
        "and the block offers the one thing that follows: take the edit off")
    assert "act: () => this.dropLast()" in work
    paint = source[source.index("    paintNext() {"):source.index("    paintTakes() {")]
    assert "work.act ? work.act() : this.render()" in paint, "the button does the work itself"
    drop = source[source.index("    dropLast() {"):source.index("    singTake(index) {")]
    assert "this.undo();" in drop and "this.render();" in drop


def test_the_window_carries_the_knobs_the_next_edit_will_be_made_with():
    """A retake with no say over how it is sung is a retake sung again and again by hand."""
    source = TRACK.read_text(encoding="utf-8")
    made = source[source.index('const knobs = element("div", "yue2-t-bar yue2-t-knobs");'):
                  source.index("left.appendChild(knobs);")]
    for name in ("Variety", "Guide", "Fade"):
        assert 'this.knob(knobs, "' + name + '"' in made, name + " has a knob of its own"
    assert "this.knobs.seed = list.newSeed();" in made, "and another seed is a button away"
    paint = source[source.index("    paintKnobs() {"):source.index("    paintButtons() {")]
    assert "this.seedKnob.hidden = !sings;" in paint
    assert "Boolean(this.editing)" in paint, (
        "a change of words is sung by these knobs too, so they are there before it is")
    assert "this.fadeKnob.box.hidden = !canFade;" in paint
    assert "Boolean(edge)" in paint, "the fade is offered where the cut leaves an edge bare"
    assert "AS_SUNG" in paint, "and a knob nobody moved says it is the song's own"
    moved = source[source.index("    knobMoved(name, value) {"):source.index("    seedTyped() {")]
    assert "Math.abs(value - own) < step ? null : value" in moved, (
        "dragged back to the song's own, it asks for nothing again")
    added = source[source.index("    addEdit(op) {"):source.index("    undo() {")]
    assert "vary: this.knobs.vary, guide: this.knobs.guide," in added
    assert "fade: edge ? this.fadeFor(edge) : null," in added, (
        "a cut in the middle of the song carries no fade at all")


def test_only_a_cut_that_reaches_an_end_of_the_song_is_offered_a_fade():
    """Anywhere else the two sides are joined with a crossfade already."""
    source = EDITLIST.read_text(encoding="utf-8")
    edge = source[source.index("export function cutEdge("):
                  source.index("export function whyNotCut(")]
    assert "selection.first === 0" in edge and "selection.stop >= bars" in edge
    assert 'if (head === tail) return "";' in edge, (
        "a selection that is the whole song reaches both ends and is neither")
    written = source[source.index("export function writeEdits("):
                     source.index("export function newSeed(")]
    assert 'if (typeof edit.fade === "number") {' in written
    assert 'if (typeof edit.vary === "number") item.vary = edit.vary;' in written


def test_a_song_opened_from_the_picker_is_drawn_without_anyone_pressing_render():
    """Drawing it is the only thing anybody does next, and it costs a second or two."""
    source = TRACK.read_text(encoding="utf-8")
    chosen = source[source.index("function chooseSong(node) {"):source.index("function runAlone(")]
    assert "openPicked(node);" in chosen, "the song just written on the node is the one drawn"
    assert chosen.index("setWidgetValue(node, SONG, row.key);") < chosen.index("openPicked(node)")
    picked = source[source.index("function openPicked(node) {"):source.index("function nodeText(")]
    assert "audioJoined(node)" in picked, (
        "a song joined to 'audio' and switched on wins, and running the node would run whatever "
        "sings it")
    assert "!nodeSong(node)" in picked, "and a song dropped under the node leaves nothing to draw"
    assert "shown.render();" in picked, "an open window runs it itself, so it follows the run"
    alone = source[source.index("function drawAlone(node) {"):source.index("function openPicked(")]
    for event in ("execution_error", "execution_interrupted", "execution_success"):
        assert alone.count('api.addEventListener("' + event) == 1
        assert alone.count('api.removeEventListener("' + event) == 1, (
            "the end of the run is let go of again, so a song picked twice leaves one listener")


def test_the_track_window_can_open_another_saved_song_without_closing():
    """The picker is a window over a window, which is what frame's stack is for."""
    source = TRACK.read_text(encoding="utf-8")
    assert 'this.songsButton = element("button", "", SONGS_LABEL);' in source
    assert "chooseSong(this.node)" in source, "the same picker the node's own button opens"
    assert "this.songsButton.disabled = busy;" in source, "though not while the node is running"


def test_the_pack_asks_its_questions_in_a_window_of_its_own():
    """The browser's own dialog carries a checkbox that switches it off for the whole site.

    Tick it once and every confirm answers no for good: deleting a song, or
    being warned that an edit is about to be thrown away, would quietly stop
    working with nothing on screen to say why.
    """
    for path in sorted(WEB.glob("*.js")):
        source = path.read_text(encoding="utf-8")
        for call in ("window.confirm(", "window.alert(", "window.prompt("):
            assert call not in source, path.name + " still asks through the browser"
    source = (WEB / "yue2_controls.js").read_text(encoding="utf-8")
    asked = source[source.index("export function confirmed("):
                   source.index("export function warned(")]
    assert "onClose: () => resolve(answer)" in asked, (
        "closing it any way at all -- Escape, the backdrop, Cancel -- answers no")
    assert "answer = true;" in asked, "and only the one button answers yes"
    assert "yes.focus();" in asked, "Enter answers it, as the browser's own did"
    assert "yue2-asked-danger" in asked, "a question that deletes something says so in red"


def test_the_picker_shows_a_song_by_what_it_is_not_by_its_key():
    """A key is sixty-four characters of nothing to look at, so the row is what the song is.

    Style, the first of its words, where it came from, how long it is and when
    it was last opened -- the store keeps all of it beside the latents, so a
    row costs one small read and no sound.
    """
    source = SONGS.read_text(encoding="utf-8")
    assert 'const ROUTE = "/yue2/songs";' in source
    facts = source[source.index("function facts(row) {"):source.index("function matches(")]
    for field in ("row.origin", "row.seconds", "row.seed"):
        assert field in facts, "a row without " + field + " is two songs that look alike"
    head = source[source.index("    function headRow(line,"):source.index("    function kidRow(")]
    order = [head.index("yue2-s-" + part) for part in ("when", "style", "words", "facts")]
    assert order == sorted(order), (
        "four lines: when it was made, the style, the words, where it came from. Songs edited "
        "one after another differ by nothing but the time, so the time has a line of its own")
    first = head[head.index('element("div", "yue2-s-head")'):head.index("yue2-s-style")]
    assert "yue2-s-when" in first and "yue2-s-style" not in first, (
        "the first line carries the time, the marks and how many edits, nothing else")
    assert "when(born(row))" in head, (
        "the time a song was made, not the time it was last read: reading touches the file")
    said = source[source.index("function words(row) {"):source.index("function facts(")]
    assert 'line.startsWith("[") && line.endsWith("]")' in said, "section tags are not words"
    assert "row.whole === false" in said, "a listing carries only the first of the lyrics"


def test_the_picker_will_not_hand_over_a_song_the_node_would_refuse():
    """A voice-only song holds the whole mix while it sounds like the voice; the node refuses it."""
    source = SONGS.read_text(encoding="utf-8")
    take = source[source.index("    function take() {"):source.index("    function pick(")]
    assert "if (!openable(row)) return;" in take
    able = source[source.index("    function openable(row) {"):source.index("    function take()")]
    assert "!row.voice" in able, "a voice-only song is not one that can be opened"
    mark = source[source.index("function badges(row, chosen) {"):source.index("function matches(")]
    assert "VOICE_ONLY" in mark and "row.key === chosen" in mark, (
        "a voice-only song says so, and the song the node is on is marked")
    assert "else if (!row.bars)" in mark, "a song with no score is offered, and says so"
    rows = source[source.index("    function headRow(line,"):source.index("    function draw() {")]
    assert rows.count("yue2-s-barred") == 2, "and it is dimmed wherever it shows"


def test_a_song_opened_on_a_node_with_audio_joined_switches_audio_off():
    """The link wins over the field, so opening a song here unticks the square first.

    The same user's report of 2026-10-18: the button was greyed out while a
    song was joined, and the only word why was its tooltip, in English. Now it
    says what it does -- 'Switch 'audio' off and open' -- and does it. Saying
    no to the question about the edits leaves the square as it was, and
    opening the song the node is on already only switches it off.
    """
    source = SONGS.read_text(encoding="utf-8")
    able = source[source.index("    function openable(row) {"):source.index("    function take()")]
    assert "joined" not in able, "a song joined to the node no longer stops a song opening"
    every = [line.strip() for line in source.splitlines() if "open.disabled" in line]
    assert every and all("openable(" in line or line == "open.disabled = true;" for line in every), (
        "every place that turns the button back on asks the same question: two of them did not, "
        "and the button came up live on a voice-only song")
    assert 'const open = element("button", "yue2-go", joined ? OPEN_JOINED : OPEN);' in source
    assert "if (joined) open.title = JOINED;" in source, "the button says what happens to the link"
    assert 'let mine = joined ? "" : (chosen || "");' in source, (
        "and no row says it is the one on the node while the link is what the node is on")
    track = TRACK.read_text(encoding="utf-8")
    choose = track[track.index("function chooseSong(node) {"):track.index("function runAlone(")]
    off = "if (audioJoined(node)) setWidgetValue(node, USE_AUDIO, false);"
    assert choose.index("await confirmed(") < choose.index(off) < choose.index(
        "setWidgetValue(node, SONG, row.key);"), (
        "the square goes off after the edits are given up and before the song is written")
    same = choose[:choose.index("const { edits } = nodeList(node);")]
    assert "if (audioJoined(node)) toggleAudio(node);" in same, (
        "the song already on the node keeps its edits and only loses the link")
    assert "this.songsButton.disabled = busy;" in track, "the button that opens it stays where it is"


def test_the_edits_of_a_song_fold_under_the_song_they_were_made_from():
    """Five edits of one song are five rows that differ by nothing but the time, in a flat list.

    The song holds the line it belongs to, so the picker can group them
    without walking a chain that a dropped song would break.
    """
    source = SONGS.read_text(encoding="utf-8")
    group = source[source.index("function families(shown) {"):source.index("export function ")]
    assert "const id = row.root || row.key;" in group, "a song and its edits are one line"
    assert "if (!line.head) line.head = line.kids.shift() || null;" in group, (
        "a line whose first song has gone is headed by the oldest edit left of it")
    assert "line.kids.sort((one, two) => born(one) - born(two));" in group, "oldest edit first"
    assert "order.sort((one, two) => two.used - one.used);" in group, (
        "and the line used last is the first line")
    draw = source[source.index("    function draw() {"):source.index("    find.addEventListener")]
    assert "wanted || unfolded === line.id" in draw, (
        "one line is open at a time, and a search opens whatever it found")
    head = source[source.index("    function headRow(line,"):source.index("    function kidRow(")]
    assert 'unfolded = unfolded === line.id ? "" : line.id;' in head, "the turn folds and unfolds"
    assert "picture(" not in head, "the song itself draws no strip"
    kid = source[source.index("    function kidRow(row) {"):source.index("    function draw() {")]
    assert "picture(row)" in kid, "an edit draws the song with the stretch it changed"
    assert "yue2-s-style" not in kid and "words(row)" not in kid, (
        "and not the style and words of the song it is an edit of, which are already above it")
    assert "what(row)" in kid, "it says what it did instead"


def test_the_strip_beside_an_edit_is_the_song_with_the_stretch_it_changed():
    """No sound is kept, so the picture is: a kilobyte of peaks saved with the song."""
    source = SONGS.read_text(encoding="utf-8")
    drawn = source[source.index("function picture(row) {"):source.index("function badges(")]
    assert 'setAttribute("preserveAspectRatio", "none")' in drawn, (
        "the strip is drawn once and stretched to whatever width the row has")
    assert "yue2-s-flat" in drawn, "a song saved before pictures were kept still draws a bar"
    band = source[source.index("function band(mark, seconds) {"):source.index("function picture(")]
    assert 'cut ? "yue2-s-seam" : "yue2-s-span"' in band, "a cut is a seam, a retake a stretch"
    assert "(Number(at[0]) / seconds) * WIDE" in band, "placed by the seconds of this song"
    read = source[source.index("function bytes(text) {"):source.index("function envelope(")]
    assert "atob(String(text))" in read, "the peaks travel as base64, a byte a slice"
    assert "return [];" in read, "and anything else draws nothing rather than throwing"


def test_the_picker_makes_the_row_it_is_on_plain_and_opens_one_on_a_second_click():
    """A tint behind a row with a waveform drawn across it is not a selection anybody can see."""
    source = SONGS.read_text(encoding="utf-8")
    style = source[source.index("const STYLE = `"):]
    assert "inset 4px 0 0 #3B7DD8" in style, "the row chosen carries a bar of colour, not a tint"
    twice = source[source.index("    function struckTwice(row) {"):
                   source.index("    function noteBox(")]
    assert "now - struck.at < TWICE" in twice, "a second click on the same row, soon after"
    assert "take();" in twice, "opens the song and closes the window"
    head = source[source.index("    function headRow(line,"):source.index("    function kidRow(")]
    kid = source[source.index("    function kidRow(row) {"):source.index("    function draw() {")]
    for part in (head, kid):
        assert "if (struckTwice(row)) return;" in part, "on a song and on its edits alike"
    assert "dblclick" not in source, (
        "counted here rather than left to the browser, which drops its own dblclick when the "
        "row is replaced between the two clicks -- and picking a row redraws the list")


def test_a_word_can_be_written_on_a_song_and_is_shown_on_its_row():
    """The one thing a song cannot say about itself is what it was wanted for."""
    source = SONGS.read_text(encoding="utf-8")
    assert 'const NOTE_ROUTE = "/yue2/songs/note";' in source
    acts = source[source.index("    function acts(row, whole) {"):
                  source.index("    function headRow(")]
    assert "writing = row.key;" in acts, "the pencil opens the box on that row"
    box = source[source.index("    function noteBox(row) {"):source.index("    function noted(")]
    assert 'if (event.key === "Enter") end(true);' in box
    assert 'else if (event.key === "Escape") end(false);' in box
    assert "setTimeout(" in box, (
        "the write waits for the draw under way, which is what took the box away")
    write = source[source.index("    async function writeNote(row, text) {"):
                   source.index("    async function erase(")]
    assert "row.note = was;" in write, "a note the server would not take goes back as it was"
    hay = source[source.index("function matches(row, wanted) {"):
                 source.index("function iconButton(")]
    assert "row.note" in hay, "and the search finds a song by the word on it"


def test_a_song_or_one_of_its_edits_can_be_deleted_from_the_picker():
    """Deleting it by hand means finding a folder of keys; the row that shows it can do it."""
    source = SONGS.read_text(encoding="utf-8")
    assert 'const DROP_ROUTE = "/yue2/songs/drop";' in source
    erase = source[source.index("    async function erase(row, whole) {"):
                   source.index("    async function setSounds(")]
    assert 'confirmed(said, { title: asked, ok: "Delete", danger: true })' in erase, (
        "nothing goes without being asked first, in a window of the pack's own")
    assert "{ key: row.key, family: Boolean(whole) }" in erase, (
        "the song takes its edits with it, an edit goes alone")
    assert "await load();" in erase, "the list is read again rather than patched by hand"
    assert "onDrop?.(gone)" in erase, "and a node opened on a song that has gone is told"
    head = source[source.index("    function headRow(line,"):source.index("    function kidRow(")]
    kid = source[source.index("    function kidRow(row) {"):source.index("    function draw() {")]
    assert "acts(row, true)" in head and "acts(row, false)" in kid
    track = TRACK.read_text(encoding="utf-8")
    choose = track[track.index("function chooseSong(node) {"):track.index("function nodeText(")]
    assert 'setWidgetValue(node, SONG, "");' in choose, "the key it was on is cleared"


def test_the_window_can_keep_each_songs_sound_beside_the_song():
    """A read off the disk beats a decode, and the window is where that is decided."""
    source = SONGS.read_text(encoding="utf-8")
    assert 'const SOUNDS_ROUTE = "/yue2/songs/sounds";' in source
    keep = source[source.index("    function paintKeep() {"):
                  source.index("    async function writeNote(")]
    assert "keepBox.disabled = busy || !can;" in keep, (
        "a machine with no PyAV is told why rather than offered a switch that does nothing")
    assert "size(keeping.bytes)" in keep and "size(keeping.budget)" in keep, (
        "what they take, against what they may take")
    assert "sweep.hidden = !count;" in keep, "nothing kept, nothing to press"
    swept = source[source.index("    async function setSounds(body) {"):
                   source.index("    async function load(")]
    assert "if (body.sweep) await load();" in swept, "a sweep changes every row"


def test_the_node_writes_the_chosen_key_and_says_which_song_it_is():
    """The key is written for the node, and the summary says the song, because the key says nothing."""
    track = TRACK.read_text(encoding="utf-8")
    choose = track[track.index("function chooseSong(node) {"):track.index("function nodeText(")]
    assert "setWidgetValue(node, SONG, row.key)" in choose, "and setWidgetValue records the change"
    assert "audioJoined(node)" in choose, "the picker says so when the graph wins"
    assert "showWidget(node, SONG, false);" in track, "the key is chosen, never typed"
    assert "SONGS_LABEL" in track and "chooseSong(node)" in track
    summary = track[track.index("function paintSummary(node) {"):track.index("function resetTrack(")]
    assert "Saved song: " in summary and "Press Render to open it." in summary


def test_the_window_runs_the_work_itself_instead_of_asking_for_render():
    """Nobody reads 'press Render'. Every step that needs the node carries the button that runs it.

    The block above the takes says what is about to happen -- the stretch, the
    takes, or for a cut the seconds and the words that go -- and its button
    runs this node alone. 'More takes' and 'Sing it' run straight away, and a
    take switched on the track offers to be kept, because until the node runs
    again the song it hands on carries the take it kept last time.
    """
    source = TRACK.read_text(encoding="utf-8")
    work = source[source.index("    nextWork() {"):source.index("    paintNext() {")]
    assert "NEXT_OPEN" in work, "a node with a song and no track offers to open it"
    assert "NEXT_CATCHUP" in work, "a list the track cannot be reached from says so"
    assert "kept !== chosen" in work, "a take chosen but not kept is work to do"
    paint = source[source.index("    paintNext() {"):source.index("    paintTakes() {")]
    assert 'element("button", "yue2-t-go", work.label)' in paint
    assert "work.act ? work.act() : this.render()" in paint, (
        "the button does the work itself, and is never a hint to press Render")
    assert source.index("    paintNext() {") < source.index("    paintTakes() {"), (
        "the block is built before the rows it stands above")
    takes = source[source.index("    paintTakes() {"):source.index("    paintWords() {")]
    assert "this.paintNext();" in takes and takes.index("this.paintNext();") < takes.index(
        "const takes = this.payload?.takes"), "the block shows even when no takes are drawn"
    for name in ("    singTake(index) {", "    moreTakes() {"):
        piece = source[source.index(name):]
        piece = piece[:piece.index("\n    }")]
        assert "this.render();" in piece, name.strip() + " still waits for Render"


def test_a_cut_says_what_it_takes_out_before_it_is_made():
    """A cut cannot be listened to first, so what it removes is spelled out: seconds and words."""
    source = TRACK.read_text(encoding="utf-8")
    facts = source[source.index("    cutFacts(edit) {"):source.index("    wordsCut(from, to) {")]
    assert "Takes out " in facts and "What is left plays " in facts
    assert "words.sections" in facts and "words.lines" in facts
    assert "CUT_MOVES" in facts, "the node moves the ends, and the block admits it"
    words = source[source.index("    wordsCut(from, to) {"):source.index("    nextWork() {")]
    assert "this.wordAt(from)" in words and "this.blocks[at]" in words, (
        "the lines come from the words already laid against the track")


def test_choosing_another_saved_song_clears_the_track_it_replaces():
    """The edits are bars of the song that was open; on another song they mean somewhere else."""
    source = TRACK.read_text(encoding="utf-8")
    choose = source[source.index("function chooseSong(node) {"):source.index("function nodeText(")]
    same = "if (row.key === nodeSong(node)) {\n            if (audioJoined(node)) toggleAudio(node);\n            return;"
    assert same in choose, "picking the same song changes nothing but the link"
    assert "await confirmed(" in choose, "edits are not thrown away silently"
    assert "writeList(node, []);" in choose
    assert "node.__yue2TrackDrawn = null;" in choose, "the old song's takes must not stay on screen"
    assert "arrived();" in choose, "and the window redraws from nothing"
@needs_node
def test_the_window_writes_a_change_of_words_the_node_can_sing():
    """The words half of the list: the box writes the lines and the text, the node sings them.

    A change of words may name no bars at all -- the node then hears the song
    to find where that line is -- so the window has to be able to write an
    edit with neither bars nor seconds, and the node to read one.
    """
    from yue2_comfy.inpaint import track

    written = run_edits("""
        const made = [
            e.wordsFor([3, 4], "one two four", 2, 831001),
            e.wordsFor([7, 9], "seven nine", 1, 42, {from: 24.0, to: 32.0, first: 12, stop: 16}),
        ];
        console.log(JSON.stringify([e.writeEdits(made), e.describeEdit(made[0], 0),
            e.describeEdit(made[1], 1)]));
    """)
    edits = track.read(written[0], 1)
    assert [(edit.op, edit.lines, edit.text, edit.bars, edit.takes) for edit in edits] == [
        ("words", (3, 4), "one two four", None, 2),
        ("words", (7, 9), "seven nine", (12, 16), 1),
    ]
    assert json.loads(track.written(edits)) == json.loads(written[0])
    assert written[1] == "1. New words for line 4"
    assert written[2] == "2. New words for bars 13-16"


@needs_node
def test_the_track_window_offers_a_line_of_the_words_to_rewrite():
    """Clicking a line is the whole way in, so the pieces it needs have to be in the file."""
    source = TRACK.read_text(encoding="utf-8")
    for wanted in ("pickLine(number, event)", "list.wordsFor(", "this.editing", "wordBox()",
                   "singWords()", "stopEditing()"):
        assert wanted in source, wanted
    assert "if (this.stopEditing()) {" in source, (
        "Escape has to put the box away before it stops the sound or closes the window")
    assert source.count("this.editing = null;") >= 4, (
        "a new payload, a cancel, a sung edit and the second click all put the box away")


@needs_node
def test_typing_in_the_track_window_is_not_the_play_key():
    """Space is Play in the window and a space between words inside the box."""
    source = TRACK.read_text(encoding="utf-8")
    where = source.index('event.key !== " " && event.code !== "Space"')
    guarded = source[where:where + 400]
    assert "typing(event.target)" in guarded.split("togglePlay")[0], (
        "the play key has to let a box holding words keep the space")
    assert 'tag === "TEXTAREA"' in source and "isContentEditable" in source


def test_the_picker_tells_a_change_of_words_from_a_retake():
    """Two edits of the same seconds look alike on a row; only the words say which was which.

    A change of notes and a song that went on name themselves too: both were once called retakes."""
    source = SONGS.read_text(encoding="utf-8")
    what = source[source.index("function what(row) {"):source.index("function swapSaid(")]
    assert 'mark.op === "words" ? "new words " : mark.op === "notes" ? "new notes "' in what
    assert 'mark.op === "extend" ? "went on " : mark.op === "move" ? "moved "' in what
    assert 'mark.op === "break" ? "break " : mark.op === "restyle" ? "style " : "retake "' in what
    mark = source[source.index("function badges(row, chosen) {"):source.index("function matches(")]
    assert '(mark.op === "words" || mark.op === "extend") && (mark.was || mark.now)' in mark
    assert "NEW_WORDS" in mark and "swapSaid" in mark, (
        "the badge is there to be pointed at, and what it holds is both texts")
    said = source[source.index("function swapSaid(mark) {"):source.index("function matches(")]
    assert "WAS_SUNG" in said and "IS_SUNG" in said
    hay = source[source.index("function matches(row, wanted) {"):source.index("function iconButton(")]
    assert "mark.was" in hay and "mark.now" in hay, "and the words are searched like the rest"


def test_the_box_for_new_words_is_the_size_of_what_is_in_it():
    """A panel-wide rule gives every textarea 320px, which is most of the words beside the track."""
    source = TRACK.read_text(encoding="utf-8")
    assert ".yue2-panel textarea.yue2-t-wordarea" in source, (
        "the box has to out-rank the panel's own textarea rule to be smaller than 320px")
    rule = source[source.index(".yue2-panel textarea.yue2-t-wordarea"):
                  source.index(".yue2-t-wordwhy")]
    assert "min-height: 0;" in rule and "resize: none;" in rule, (
        "it grows with the words instead, so there is no handle to drag")
    fit = source[source.index("    fitArea() {"):source.index("    wordBox() {")]
    assert "area.style.height = \"auto\";" in fit and "area.scrollHeight" in fit
    assert "WORDS_TALL" in fit and "WORDS_SHORT" in fit, "and it stops growing somewhere"
    assert "this.fitArea();" in source[source.index("    paintWords() {"):
                                       source.index("    overWords(")], (
        "a box painted again is sized again, or it opens at one row")
    assert ".yue2-t-wordwhy" in source and "yue2-t-wordbox" in source, (
        "the note under the buttons is inside the box, not loose among the lyrics")


def test_a_line_under_the_pointer_is_shown_on_the_track():
    """The window already lights a line up while the song plays; this is that the other way round."""
    source = TRACK.read_text(encoding="utf-8")
    span = source[source.index("    guessedSpan(block, line) {"):source.index("    whereLine(")]
    at = source[source.index("    guessedAt(second) {"):source.index("    markWords() {")]
    for shared in ("section.sung", "stop <= from", "sections[index + 1]"):
        assert shared in span and shared in at, (
            "the line's place and the place's line have to be the same guess: " + shared)
    assert "this.paired.indexOf(block)" in span
    over = source[source.index("    overWords(block, line) {"):source.index("    lineSpan(")]
    assert "this.draw();" in over, "the track is drawn again, or nothing is seen"
    drawn = source[source.index("    drawSaid(c, top, tall) {"):source.index("    drawPick(")]
    assert "SKIN.said" in drawn and "SKIN.saidEdge" in drawn
    assert "said: " in source and "saidEdge: " in source, "a colour of its own, not the selection's"
    said = source[source.index("    saidSpan() {"):source.index("    keepWordSeen(")
                  if source.index("    keepWordSeen(") > source.index("    saidSpan() {")
                  else source.index("    saidSpan() {") + 1200]
    assert "this.editing" in said, (
        "an open box shows where it will sing without the pointer being anywhere")


def test_lines_light_up_where_the_song_sings_them_once_their_times_are_known():
    """The node hears the song and hands over when each line is sung; the bars are the fallback."""
    source = TRACK.read_text(encoding="utf-8")
    timed = source[source.index("    timedLines() {"):source.index("    lineTime(")]
    assert "take ? take.lines : this.payload?.lines" in timed, (
        "a take is a song of its own length, so its lines are its own")
    at = source[source.index("    wordAt(second) {"):source.index("    guessedAt(second) {")]
    assert "this.timedLines()" in at and "this.guessedAt(second)" in at
    assert "LINE_HOLD" in at and "LINE_GAP" in at, (
        "a line stays lit through a breath, not through an interlude")
    span = source[source.index("    lineSpan(block, line) {"):source.index("    guessedSpan(")]
    assert "this.lineTime(number)" in span and "this.guessedSpan(block, line)" in span
    cut = source[source.index("    wordsCut(from, to) {"):source.index("    nextWork() {")]
    assert "this.timedLines()" in cut, "a cut names the lines it takes by where they are sung"
    words = source[source.index("    paintWords() {"):source.index("    overWords(")]
    assert "WORDS_TIMED" in words and "WORDS_WHERE_HEARD" in words, (
        "the panel says which it is: heard, or guessed from the bars")


def test_a_change_of_words_offers_its_takes_as_a_retake_does():
    """The takes of new words were sung and compared but never shown: none could be heard side by
    side, none kept but the join's, and More takes was not there. A change of notes has takes too."""
    source = TRACK.read_text(encoding="utf-8")
    assert ('return op === "retake" || op === "words" || op === "notes" || op === "extend" '
            '|| op === "break"\n        || op === "restyle";' in source)
    assert source.count('op === "retake"') == 1
    for pattern in ('op !== "retake"', 'kind === "retake"', 'kind !== "retake"'):
        assert pattern not in source, pattern
    for method in ("    shownTake() {", "    paintTakes() {", "    nextWork() {",
                   "    onlyTheTake() {", "    showTake(index) {", "    singTake(index) {",
                   "    moreTakes() {"):
        start = source.index(method)
        body = source[start:source.index("\n    }\n", start)]
        assert "singsAgain(" in body, method


def test_a_take_of_new_words_says_how_much_of_them_was_heard():
    source = TRACK.read_text(encoding="utf-8")
    facts = source[source.index("function heardFacts(take) {"):source.index("function wasFacts(")]
    assert '"heard " + take.heard[0] + " of " + take.heard[1]' in facts
    takes = source[source.index("    paintTakes() {"):source.index("    paintWords() {")]
    assert "heardFacts(take)" in takes and "take.mumbled" in takes and "take.said" in takes
    assert "PICK_WORDS" in takes, "the pick is explained by what picked it"
    assert "was.said" in takes, "the song as it was says what was heard there before"
    fact = source[source.index("    takeFact(edit) {"):source.index("    editSpan(edit) {")]
    assert 'edit.op === "words"' in fact, "the blue block says how the take will be picked"


def test_a_retake_heard_on_this_machine_says_so_and_shows_the_song_as_it_was_heard():
    """A retake is heard only when the speech models are already there, so the window asks the node.

    The song as it was is heard against the same words, which is what a take's
    count means anything against.
    """
    source = TRACK.read_text(encoding="utf-8")
    fact = source[source.index("    takeFact(edit) {"):source.index("    editSpan(edit) {")]
    assert "this.payload?.hears" in fact
    takes = source[source.index("    paintTakes() {"):source.index("    paintWords() {")]
    assert "heardFacts(was)" in takes and "HEARD_BEFORE" in takes
    assert "words ? HEARD_WHY : HEARD_AGAIN" in takes
    assert "words ? PICK_WORDS : heardAny ? PICK_HEARD : PICK_JOIN" in takes
    assert "this.payload.asked" in takes, "what the takes were heard against is shown too"
    from yue2_comfy import edit_track

    doc = edit_track.YuE2EditTrack._payload.__doc__
    for key in ("hears", "asked"):
        assert "``" + key + "``" in doc, key


def test_the_audio_input_is_switched_off_by_a_square_beside_it():
    """The MiniMax rewriter's way: a square drawn by the input, its state a widget of the node.

    Everything that asked whether a song is joined to 'audio' asks whether one
    is joined and switched on, or the picker would refuse a song the node is
    about to edit.
    """
    from yue2_comfy import edit_track

    source = TRACK.read_text(encoding="utf-8")
    assert 'const USE_AUDIO = "use_audio";' in source
    assert "use_audio" in edit_track.YuE2EditTrack.INPUT_TYPES()["optional"]
    hooks = source[source.index("app.registerExtension({"):]
    for said in ("drawSwitch(this, ctx)", "onSwitch(this, pos)", "toggleAudio(this)"):
        assert said in hooks, said
    assert "showWidget(node, USE_AUDIO, vueNodes())" in source, (
        "Vue nodes draw no square on the canvas, so there the switch is shown as a widget")
    assert source.count("sourceOf(node, AUDIO_IN)") == 2, "audioJoined and the summary, no more"
    assert "sourceOf(this.node, AUDIO_IN)" not in source
    for place, end in (("function chooseSong(node) {", "\n}\n"), ("function openPicked(node) {", "\n}\n"),
                       ("    nextWork() {", "\n    }\n")):
        start = source.index(place)
        assert "audioJoined(" in source[start:source.index(end, start)], place
    toggled = source[source.index("function toggleAudio(node) {"):source.index("function chooseSong(")]
    assert "setWidgetValue(node, USE_AUDIO" in toggled, "written so a saved workflow keeps it"


def test_a_run_on_the_joined_song_leaves_its_key_in_song_key():
    """Switching 'audio' off goes on with the song the node last edited, not one picked days before.

    A user's report of 2026-10-18: 'Load Audio' into the node, an edit, the
    square unticked, Render -- and the window drew another song, with the edit
    laid on it, because 'song_key' was written by the picker alone. Now the
    node says the song came in through 'audio' and the window writes its key,
    through setWidgetValue so a reload keeps it. A song picked while that run
    was out is not overwritten when it comes back, unless the square is ticked
    again by then, when the joined song is what the node is on anyway.
    """
    source = TRACK.read_text(encoding="utf-8")
    hooks = source[source.index("nodeType.prototype.onExecuted = function (message) {"):]
    hooks = hooks[:hooks.index("\n        };\n")]
    wrote = ("if (drawn.joined && typeof drawn.was === \"string\" && drawn.was\n"
             "                    && (audioOn(this) || !this.__yue2SongChosen)) {\n"
             "                    setWidgetValue(this, SONG, drawn.was);")
    assert wrote in hooks
    assert hooks.index(wrote) < hooks.index("this.__yue2SongChosen = false;") < hooks.index(
        "this.__yue2TrackDrawn = drawn;")
    chose = source[source.index("function chooseSong(node) {"):source.index("function runAlone(")]
    assert "setWidgetValue(node, SONG, row.key);\n        node.__yue2SongChosen = true;" in chose
    assert source.count("setWidgetValue(this, SONG") + source.count("setWidgetValue(node, SONG") == 3, (
        "the pick, the pick deleted, and a run on the joined song")


@needs_node
def test_the_list_the_window_writes_names_its_song_as_the_node_reads_it():
    """The node refuses a list made on another song; the window says which song it made it on.

    Every write of the list goes through writeList, which names the song the
    drawn track is ('was'), or keeps the name the list already has when
    nothing is drawn yet, as after a reload. An empty list names nothing.
    """
    from yue2_comfy.inpaint import track

    key = "ab" * 32
    got = run_edits("""
        const cut =[{op: "cut", bars: [2, 4], seconds: null, seed: 0, takes: 1, take: null,
                      vary: null, guide: null, fade: null, lines: null, text: null, score: null}];
        const named = e.writeEdits(cut, "%s");
        console.log(JSON.stringify({named, bare: e.writeEdits(cut), back: e.readEdits(named, 2),
            old: e.readEdits(e.writeEdits(cut), 2).song,
            bad: e.readEdits('{"song": 5, "edits": []}').error}));
    """ % key)
    assert track.song_of(got["named"]) == key and track.read(got["named"]) == track.read(got["bare"])
    assert track.song_of(got["bare"]) == ""
    assert got["back"]["song"] == key and got["back"]["edits"][0]["op"] == "cut"
    assert got["old"] == "" and got["bad"]
    source = TRACK.read_text(encoding="utf-8")
    assert source.count("setWidgetValue(node, EDITS") == 1
    assert "list.writeEdits(edits, listSong(node))" in source
    song = source[source.index("function listSong(node) {"):]
    song = song[:song.index("\n}\n")]
    assert "node.__yue2TrackDrawn?.was" in song and "nodeList(node).song" in song



@needs_node
def test_a_section_is_dropped_on_the_section_line_nearest_the_pointer():
    """Whole sections move, to where another starts or the song ends, never into themselves."""
    from yue2_comfy.inpaint import track

    got = run_edits("""
        const grid = {grid};
        console.log(JSON.stringify({{
            lines: e.sectionLines(grid),
            whole: [e.wholeSections(grid, 0, 2), e.wholeSections(grid, 2, 5),
                    e.wholeSections(grid, 1, 2), e.wholeSections(grid, 0, 5)],
            before: e.dropLine(grid, 2, 5, 0.3),
            after: e.dropLine(grid, 0, 2, 9.0),
            nowhere: e.dropLine(grid, 0, 5, 3.0),
            said: e.describeEdit(e.moveFor(2, 5, 0), 0, 5),
            end: e.describeEdit(e.moveFor(0, 2, 5), 1, 5),
            written: e.writeEdits([e.moveFor(2, 5, 0)]),
        }}));
    """.format(grid=json.dumps(TRACK_GRID)))
    assert got["lines"] == [0, 2, 5]
    assert got["whole"] == [True, True, False, True]
    assert (got["before"], got["after"], got["nowhere"]) == (0, 5, None)
    assert got["said"] == "1. Move bars 3-5 before bar 1"
    assert got["end"] == "2. Move bars 1-2 to the end"
    edit = track.read(got["written"])[0]
    assert (edit.op, edit.bars, edit.to) == ("move", (2, 5), 0)


@needs_node
def test_the_break_the_window_writes_is_the_break_the_node_plays():
    """A break names the bar line it goes before and how many bars it plays; it has takes as a retake has."""
    from yue2_comfy.inpaint import track

    got = run_edits("""
        const made = [e.breakFor(13, 4, 3, 831001, {vary: 1.5}), e.breakFor(2, 99, 9, 5),
                      e.breakFor(13, 4, 2, 7, {}, -6), e.breakFor(13, 4, 2, 7, {}, 1),
                      e.breakFor(13, 4, 2, 7, {}, 0)];
        console.log(JSON.stringify({written: e.writeEdits(made),
            said: made.map((edit, index) => e.describeEdit(edit, index, 40))}));
    """)
    first, second, ahead, into, down = track.read(got["written"], 1)
    assert (first.op, first.to, first.length, first.takes, first.seed, first.vary, first.at) == (
        "break", 13, 4, 3, 831001, 1.5, None)
    assert (second.to, second.length, second.takes) == (2, 16, 4), (
        "the window keeps a break to the bars and takes the node plays")
    assert (ahead.at, into.at, down.at) == (-6, 1, 0), "the beat the arrows chose goes to the node"
    assert got["said"] == ["1. A break of 4 bars before bar 14, variety 1.50",
                          "2. A break of 16 bars before bar 3",
                          "3. A break of 4 bars before bar 14, 6 beats ahead of it",
                          "4. A break of 4 bars before bar 14, 1 beat into it",
                          "5. A break of 4 bars before bar 14, on its downbeat"]


@needs_node
def test_a_break_goes_before_a_section_after_the_first_and_before_the_end():
    got = run_edits("console.log(JSON.stringify(e.breakLines({})));".format(json.dumps(TRACK_GRID)))
    assert got == [2], "the chorus is the one section a break can go before on that grid"


@needs_node
def test_a_break_names_both_sections_and_a_selected_section_gets_it_after_it():
    """A user's ask of 2026-10-18: a break 'after every verse or chorus' came out 'at random'.

    The window put the break before the section selected, while a person who
    selects a verse means after it, and its list named one section only. Now
    every place names both sections, and a selection gets the break at its end:
    on that section line when it ends on one, else on the line nearest its end.
    With nothing selected it still goes before the first chorus.
    """
    grid = dict(TRACK_GRID, sections=[
        {"name": "verse", "bar": 0, "bars": 2, "start": 0.5, "end": 4.5, "sung": 0.2},
        {"name": "chorus", "bar": 2, "bars": 1, "start": 4.5, "end": 6.5, "sung": 4.1},
        {"name": "bridge", "bar": 3, "bars": 2, "start": 6.5, "end": 10.5, "sung": 6.4}])
    got = run_edits("""
        const grid = {grid};
        console.log(JSON.stringify({{
            lines: e.breakLines(grid),
            named: e.breakLines(grid).map((bar) => e.breakBetween(grid, bar)),
            short: e.breakLines(grid).map((bar) => e.breakPair(grid, bar)),
            verse: e.breakLineFor(grid, e.selectBars(grid, 0, 2)),
            chorus: e.breakLineFor(grid, e.selectBars(grid, 2, 3)),
            last: e.breakLineFor(grid, e.selectBars(grid, 3, 5)),
            inside: e.breakLineFor(grid, e.selectBars(grid, 0, 1)),
            seconds: e.breakLineFor(grid, e.selectSeconds(1.0, 6.0)),
            none: e.breakLineFor(grid, null),
            bare: e.breakLineFor({{bars: [0, 1]}}, null),
        }}));
    """.format(grid=json.dumps(grid)))
    assert got["lines"] == [2, 3]
    assert got["named"] == ["after the verse, before the chorus", "after the chorus, before the bridge"]
    assert got["short"] == ["verse \u2192 chorus", "chorus \u2192 bridge"], (
        "the list itself sits in the narrow words column, so it says it with an arrow")
    assert (got["verse"], got["chorus"]) == (2, 3), "a selected section gets the break after it"
    assert got["last"] == 3, "nothing comes after the last section, so the nearest line it has"
    assert (got["inside"], got["seconds"]) == (2, 3), "any other stretch: the line nearest its end"
    assert (got["none"], got["bare"]) == (2, None), "nothing selected: before the first chorus"
    source = TRACK.read_text(encoding="utf-8")
    assert "return list.breakLineFor(this.grid(), this.selection);" in source
    box = source[source.index("    breakBox() {"):source.index("    paintBreakFact() {")]
    fact = source[source.index("    breakFact("):source.index("    breakPlaces(to) {")]
    assert "list.breakPair(grid, bar)" in box and "list.breakBetween(grid, to)" in fact, (
        "the list and the line under it say the same two sections")
    assert 'option.title = list.breakBetween(grid, bar) + ", bar " + (bar + 1);' in box, (
        "and the option says it in words, with the bar the list has no room for")
    assert ".yue2-t-breakbox select { max-width: 100%; min-width: 0; }" in source, (
        "the list was cut off at the column's edge; it may be no wider than the column")


def test_the_track_offers_a_break_beside_going_on_and_says_why_it_cannot():
    source = TRACK.read_text(encoding="utf-8")
    assert 'element("button", "", "Break\\u2026")' in source
    paint = source[source.index("    paintButtons() {"):source.index("    paintList() {")]
    assert "this.breakButton.disabled" in paint and "NO_BREAK_HERE" in paint
    takes = source[source.index("    paintTakes() {"):source.index("    paintWords() {")]
    assert "voiceFacts(take)" in takes and "PICK_BREAK" in takes and "VOICE_LEFT_WHY" in takes
    assert "if (this.stopBreaking())" in source, "Escape puts the break box away"
    box = source[source.index("    breakBox() {"):source.index("    paintBreakFact() {")]
    assert box.index('"\\u2190"') < box.index('"\\u2192"') < box.index("const bars = "), (
        "the arrows come before the bars, where the person asked for them")
    assert "this.nudgeBreak(-1)" in box and "this.nudgeBreak(1)" in box
    said = source[source.index("    saidSpan() {"):source.index("    rawLines() {")]
    assert "this.breakPlace()" in said, "the green mark is where the node puts the break in"
    put = source[source.index("    putBreak() {"):source.index("    whyNotBreak() {")]
    assert "this.breaking.at)" in put, "the beat chosen goes on the list"
    for method in ("    whyNotMore(selection) {", "    whyNotNotes() {", "    whyNotGoOn() {",
                   "    whyNotBreak() {"):
        start = source.index(method)
        body = source[start:source.index("\n    }\n", start)]
        assert 'edit.op === "break"' in body, method


def run_roll_from(folder: pathlib.Path, script: str):
    """``run_roll`` for a script too long for a command line: it is written to a file first."""
    path = folder / "roll_check.mjs"
    path.write_text("import * as r from {};\n{}".format(json.dumps(ROLL.as_uri()), script), encoding="utf-8")
    done = subprocess.run([NODE, str(path)], capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def imported_name(pitches, key):
    """The chord Load MIDI reads from these notes on a Chords track, named in ``key`` as the score writes it."""
    from yue2_comfy.midi import chords, score
    from yue2_comfy.sheetsage import abc_rebuild, spelling
    label = chords.spelled(pitches)
    return None if label is None else abc_rebuild.chord_text(spelling.chord_name(label, score.key_label(key)))


def test_the_roll_lays_chords_out_from_the_root_load_midi_reads():
    from yue2_comfy.midi import export
    assert js_number("CHORD_ROOT") == export.CHORD_ROOT


@needs_node
def test_the_roll_names_chord_notes_exactly_as_load_midi_names_a_chords_track(tmp_path):
    """A chord changed note by note in the roll is the chord the file saved from it reads back as.

    Every layout the roll draws -- fifteen kinds on twelve roots, bare or over each
    bass -- in every key, and sets of notes no one would draw: named alike, or both
    refused.
    """
    import random
    from yue2_comfy.midi import export, score
    from yue2_comfy.sheetsage import abc_rebuild, spelling
    from yue2_comfy.vendor.yue2_music import abc_tools
    keys = list(abc_tools.KEYS)
    layouts = [export.chord_pitches(abc_rebuild.SHARP_NAMES[root] + quality
                                    + ("" if bass is None else "/" + abc_rebuild.SHARP_NAMES[bass]))
               for quality in abc_tools.QUALITIES for root in range(12) for bass in [None] + list(range(12))]
    rng = random.Random(27)
    loose = [sorted(rng.sample(range(34, 74), rng.randint(2, 5))) for _ in range(600)]
    sets = layouts + loose
    got = run_roll_from(tmp_path, "const sets = {}; const keys = {};\n"
                        "console.log(JSON.stringify({{sharps: keys.map((k) => r.keySharps(k)), "
                        "names: keys.map((k) => sets.map((p) => r.namedChord(p, k)))}}));"
                        .format(json.dumps(sets), json.dumps(keys)))
    assert got["sharps"] == [spelling.signature(score.key_label(key)) for key in keys]
    wrong = [(key, pitches, name, imported_name(pitches, key))
             for key, row in zip(keys, got["names"]) for pitches, name in zip(sets, row)
             if name != imported_name(pitches, key)]
    assert not wrong, wrong[:5]
    named = [name for row in got["names"] for name in row[:len(layouts)]]
    assert all(abc_tools.CHORD.fullmatch(name) for name in named)
    assert any(name is None for row in got["names"] for name in row[len(layouts):])


@needs_node
def test_a_chord_drawn_on_a_row_takes_the_kind_its_degree_has_in_the_key():
    """C, Dm, Em, F, G, Am, Bdim in C; a root outside the scale gets a major chord."""
    from yue2_comfy.vendor.yue2_music import abc_tools
    keys = list(abc_tools.KEYS)
    got = run_roll("console.log(JSON.stringify({}.map((k) => Array.from({{length: 12}}, (_, pc) => "
                   "r.chordByDegree(60 + pc, k)))));".format(json.dumps(keys)))
    shapes = {"maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6)}
    for key, row in zip(keys, got):
        tonic = (7 * abc_tools.KEYS[key]) % 12
        scale = [(tonic + step) % 12 for step in (0, 2, 4, 5, 7, 9, 11)]
        for pitch_class, name in enumerate(row):
            quality = "maj"
            if pitch_class in scale:
                at = scale.index(pitch_class)
                steps = ((scale[(at + 2) % 7] - pitch_class) % 12, (scale[(at + 4) % 7] - pitch_class) % 12)
                quality = {(4, 7): "maj", (3, 7): "min", (3, 6): "dim"}[steps]
            wanted = imported_name([48 + pitch_class + step for step in shapes[quality]], key)
            assert name == wanted, (key, pitch_class, name, wanted)
    assert got[keys.index("C")] == ["C", "Db", "Dm", "Eb", "Em", "F", "F#", "G", "Ab", "Am", "Bb", "Bdim"]


@needs_node
def test_a_chord_moved_whole_keeps_its_kind_and_bass_and_is_named_from_the_key():
    """Up or down by its root, a chord is what its moved notes read as in the key where it lands."""
    from yue2_comfy.midi import export
    from yue2_comfy.sheetsage import abc_rebuild
    from yue2_comfy.vendor.yue2_music import abc_tools
    names = [abc_rebuild.SHARP_NAMES[root] + quality
             + ("/" + abc_rebuild.SHARP_NAMES[(root + 4) % 12] if slash else "")
             for quality in abc_tools.QUALITIES for root in (0, 3, 8) for slash in (False, True)]
    keys = ["C", "Bb", "F#m", "Ebm", "E"]
    shifts = [-7, -1, 1, 5, 12]
    got = run_roll("console.log(JSON.stringify({}.flatMap((k) => {}.flatMap((n) => {}.map((s) => "
                   "r.transposedChord(n, s, k))))));".format(json.dumps(keys), json.dumps(names), json.dumps(shifts)))
    at = 0
    for key in keys:
        for name in names:
            for shift in shifts:
                moved = got[at]
                at += 1
                if shift == 12:
                    assert moved == name
                    continue
                layout = export.chord_pitches(moved)
                wanted = sorted({(pitch + shift) % 12 for pitch in export.chord_pitches(name)})
                assert sorted({pitch % 12 for pitch in layout}) == wanted, (name, shift)
                assert moved == imported_name(layout, key), (key, name, shift, moved)


@needs_node
def test_a_chord_moves_between_its_neighbours_and_its_end_is_where_the_next_begins():
    sheet = roll_sheet()
    got = run_roll("""
        const sheet = {sheet};
        const m = r.modelOf(sheet);
        const names = (model) => model && model.chords.map((c) => [c.start, c.name]);
        console.log(JSON.stringify({{
            spans: r.chordSpans(m, sheet.total),
            rooms: [r.chordRoom(m, 0, sheet.total), r.chordRoom(m, 32, sheet.total), r.chordRoom(m, 128, sheet.total),
                    r.chordRoom(m, 5, sheet.total)],
            later: names(r.moveChord(m, sheet, 32, 16, 0)),
            past: r.moveChord(m, sheet, 32, 96, 0),
            up: names(r.moveChord(m, sheet, 32, 0, 1)),
            same: r.moveChord(m, sheet, 32, 0, -12) === m,
            all: names(r.moveChords(m, sheet, [0, 32], 32, 2)),
            onto: r.moveChords(m, sheet, [0], 32, 0),
            root: [r.rootPitch("Bb/D"), r.rootPitch("nope")],
            key: [r.keyAt(sheet, 0), r.keyAt(sheet, 150)],
        }}));
    """.format(sheet=json.dumps(sheet)))
    assert got["spans"] == [{"start": 0, "name": "F", "end": 32}, {"start": 32, "name": "Bb", "end": 128},
                            {"start": 128, "name": "C", "end": 192}]
    assert got["rooms"] == [{"low": 0, "high": 31}, {"low": 1, "high": 127}, {"low": 33, "high": 191}, None]
    assert got["later"] == [[0, "F"], [48, "Bb"], [128, "C"]]
    assert got["past"] is None, "a chord does not jump over the next one"
    assert got["up"] == [[0, "F"], [32, "B"], [128, "C"]]
    assert got["same"] is True
    assert got["all"] == [[32, "G"], [64, "C"], [128, "C"]]
    assert got["onto"] is None
    assert got["root"] == [58, None] and got["key"] == ["F", "F"]


REPEAT_SCORE = ('X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=100\n'
                'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
                'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
                'K:C\n'
                '% verse\nV: Vocal\n"C"c16|"Am"A16|\nV: Ins\nZ2|\n'
                '% chorus\nV: Vocal\n"F"f16|"G"g16|\nV: Ins\nZ2|\n'
                '% verse\nV: Vocal\n"C"c16|"Am"A16|\nV: Ins\nZ2|\n'
                '% chorus\nV: Vocal\nf16|"G"g16|\nV: Ins\nZ2|\n'
                '% verse\nV: Vocal\n"C"c16|"Em"e16|\nV: Ins\nZ2|\n'
                '% chorus\nV: Vocal\n"F"f16|"G"g16|"C"c16|\nV: Ins\nZ3|\n')
"""Three verses, the last with other chords; a chorus that starts on the chord the verse before it left
ringing, and a last chorus a bar longer than the first."""


@needs_node
def test_a_chord_edit_is_made_again_in_the_repeats_that_still_have_the_old_chords():
    """What 'repeats too' does: the same change in each section of the same name whose chords matched.

    A repeat with chords of its own is left alone and named, and the chord that
    rang on past the end of a repeat still rings there, so the section after it
    is not changed by the edit. A longer or shorter repeat is matched over the
    bars the two have in common, as a last chorus that runs on usually is.
    """
    from yue2_comfy import notation
    sheet = notation.read(REPEAT_SCORE)
    got = run_roll("""
        const sheet = {sheet};
        const m = r.modelOf(sheet);
        const names = (model) => model.chords.map((c) => [c.start, c.name]);
        const edited = r.setChord(m, 16, "Dm");
        const mirrored = r.mirrorChords(sheet, m, edited);
        const chorus = r.mirrorChords(sheet, m, r.setChord(m, 32, "Dm"));
        const moved = r.mirrorChords(sheet, m, r.moveChord(m, sheet, 16, 8, 0));
        console.log(JSON.stringify({{
            chords: names(mirrored.model), done: mirrored.done, skipped: mirrored.skipped,
            sheet: r.sheetOf(mirrored.model),
            chorus: [names(chorus.model), chorus.done, chorus.skipped],
            moved: [names(moved.model), moved.done],
            nothing: r.mirrorChords(sheet, m, m).model === m,
            kinds: [r.sectionKind(" Verse 2 "), r.sectionKind("pre-chorus"), r.sectionKind("verse")],
        }}));
    """.format(sheet=json.dumps(sheet)))
    assert got["chords"] == [[0, "C"], [16, "Dm"], [32, "F"], [48, "G"], [64, "C"], [80, "Dm"], [96, "Am"],
                             [112, "G"], [128, "C"], [144, "Em"], [160, "F"], [176, "G"], [192, "C"]]
    assert got["done"] == [{"bar": 4, "bars": 2, "name": "verse"}]
    assert got["skipped"] == [{"bar": 8, "bars": 2, "name": "verse"}]
    written = notation.write(REPEAT_SCORE, got["sheet"])
    assert written["bars"] == [1, 5, 6]
    back = notation.read(written["abc"])
    assert [[c["start"], c["name"]] for c in back["chords"]] == got["chords"]
    chords, done, skipped = got["chorus"]
    assert done == [{"bar": 10, "bars": 3, "name": "chorus"}]
    assert skipped == [{"bar": 6, "bars": 2, "name": "chorus"}], "the chorus that starts on Am is another chorus"
    assert [160, "Dm"] in chords and [192, "C"] in chords and [96, "Am"] not in chords
    assert got["moved"][0][:6] == [[0, "C"], [24, "Am"], [32, "F"], [48, "G"], [64, "C"], [88, "Am"]]
    assert got["moved"][1] == [{"bar": 4, "bars": 2, "name": "verse"}]
    assert got["nothing"] is True
    assert got["kinds"] == ["verse", "pre-chorus", "verse"]


def test_the_window_edits_chords_as_notes_in_a_part_of_their_own():
    """The Chords part of the switch: drawn from the chord symbols, edited whole or note by note."""
    source = SCORE.read_text(encoding="utf-8")
    assert '[CHORDS, "Chords",' in source and 'const CHORDS = "chords";' in source
    down = source.split("    chordDown(event, px, py, tick, keys, again) {")[1].split("\n    }\n")[0]
    for call in ("this.removeChordAt(hit.span)", "this.openChordInput(hit.span.start, hit.span.name)",
                 "this.toggleTone(tick, pitch)", "roll.chordByDegree(pitch, roll.keyAt(this.sheet, start))",
                 'mode: "edge"', 'mode: "chord"', 'mode: "tone"'):
        assert call in down, call
    dragged = source.split("    chordDragged(event, tick, pitch) {")[1].split("\n    }\n")[0]
    for call in ("roll.moveChords(", "roll.moveChord(", "roll.chordRoom(", "roll.namedChord("):
        assert call in dragged, call
    assert "this.chordDown(event, px, py, tick, keys, again);" in source
    assert "this.chordDragged(event, tick, pitch);" in source and "this.chordReleased(drag, result);" in source


def test_every_chord_edit_in_the_chords_part_can_reach_the_repeats():
    source = SCORE.read_text(encoding="utf-8")
    assert 'checkControl("repeats too", false, REPEATS_TOOLTIP)' in source
    commit = source.split('    commitChords(next, previous = this.model, said = "") {')[1].split("\n    }\n")[0]
    assert "this.chordPart && this.repeats.box.checked" in commit and "roll.mirrorChords(" in commit
    for path in ("this.commitChords(roll.removeChord(this.model, start))",
                 "this.commitChords(roll.setChord(this.model, start, text))",
                 "if (this.chordPart) this.commitChords(next);",
                 "const kept = this.commitChords(result, drag.from);"):
        assert path in source, path


def test_a_chord_name_on_the_lane_stops_where_the_next_chord_starts():
    """Two chords a beat apart drew their names over each other, and a click took the first."""
    source = SCORE.read_text(encoding="utf-8")
    chips = source.split("    chordChips(c, model) {")[1].split("\n    }\n")[0]
    assert "this.x(span.end) - left - 2" in chips and "Math.min(" in chips
    assert source.count("this.chordChips(c, ") == 2, "the lane and the click must measure the same chips"
    lane = source.split("for (const chip of this.chordChips(c, model)) {")[1].split("\n        }\n")[0]
    assert "c.clip();" in lane



@needs_node
def test_a_mark_kept_for_new_words_comes_off_and_goes_on_as_the_server_does_it():
    """The editor writes ' keep' after the mark and the node reads it; a disagreement would sing a comment."""
    from yue2_comfy import edits
    words = edits.mark("style", "lyrics", "full")
    kept = edits.attach(ROLL_SCORE, words, keep=True)
    texts = [kept, kept.replace("\n", "\r\n") + "\r\n", edits.attach(ROLL_SCORE, words),
             ROLL_SCORE + "%yue2-words " + words + " keeps\n", ROLL_SCORE]
    got = run_roll("""
        const texts = {texts};
        const score = {score};
        const changed = score + "G|";
        console.log(JSON.stringify({{
            split: texts.map((t) => r.splitMark(t)),
            attached: [r.attachMark(score, {words}, true), r.attachMark(score, {words}), r.attachMark(score, null, true)],
            values: [r.editValue(score, score, {words}, true), r.editValue(score, score, {words}, false),
                     r.editValue(score, score, null, true), r.editValue(changed, score, {words}, true),
                     r.editValue(changed, score, {words}, false), r.editValue("", score, {words}, true)],
        }}));
    """.format(texts=json.dumps(texts), score=json.dumps(ROLL_SCORE), words=json.dumps(words)))
    assert got["attached"] == [kept, edits.attach(ROLL_SCORE, words), edits.attach(ROLL_SCORE, None, keep=True)]
    assert [[x["score"], x["words"], x["keep"]] for x in got["split"]] == [
        [found.score, found.words, found.keep] for found in map(edits.read, texts)]
    changed = (ROLL_SCORE + "G|").strip()
    assert got["values"] == [kept.strip(), "", "", edits.attach(changed, words, keep=True),
                             edits.attach(changed, words), ""]


def test_the_window_keeps_an_edit_for_new_words_only_when_asked():
    """A box beside Apply, remembered in the browser, read back from the box, and named where the node refuses."""
    from yue2_comfy import edits, nodes, staged
    source = SCORE.read_text(encoding="utf-8")
    assert 'const KEEP_LABEL = "Keep for new words";' in source
    label = "'Keep for new words'"
    assert label in edits.OTHER_WORDS and label in nodes.EDITED_SCORE_TOOLTIP and label in staged.SCORE_TOOLTIP
    russian = json.loads((ROOT / "locales" / "ru" / "nodeDefs.json").read_text(encoding="utf-8"))
    for node in ("YuE2GenerateSong", "YuE2RenderPlan"):
        assert "Keep for new words" in russian[node]["inputs"]["score_abc"]["tooltip"], node
    assert "this.keepWords.holder.hidden = Boolean(this.track || this.source);" in source
    assert "this.keepWords.box.checked = box.score && box.words ? box.keep : rememberedKeep();" in source
    assert "roll.editValue(text, model, this.baseWords, this.keepWords.box.checked)" in source
    keep = source.split("    keepChanged() {")[1].split("\n    }\n")[0]
    assert "rememberKeep(keep);" in keep
    remembered = source.split("function rememberedKeep() {")[1].split("\n}\n")[0]
    assert "try {" in remembered and "catch" in remembered


def test_typing_in_the_abc_tab_keeps_the_words_an_edit_belongs_to():
    """Only a score that is the plan's own, or carries its own mark, changes the words; a typed edit keeps them."""
    source = SCORE.read_text(encoding="utf-8")
    typed = source.split("    async readTyped() {")[1].split("\n    }\n")[0]
    assert "this.baseWords || this.planWords" in typed
    assert "if (typed.words) this.keepWords.box.checked = typed.keep;" in typed


COPY_SCORE = ('X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=100\n'
              'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
              'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
              'K:C\n'
              '% verse\nV: Vocal\n"C"z4c4d4e4|"G"g8f8|"Am"e16|"F"c8z8|\nV: Ins\nC16|G,16|A,16|F,16|\n'
              '% outro\nV: Vocal\nZ2|\nV: Ins\nZ2|\n')
"""Six bars of sixteen ticks: a phrase over bars 1-2, a held note from bar 3 into nothing, chords until bar 4."""


def copy_run(tmp_path, script):
    from yue2_comfy import notation
    sheet = notation.read(COPY_SCORE)
    return sheet, run_roll_from(tmp_path, """
        const sheet = {sheet};
        const model = r.modelOf(sheet);
        const ids = (part, from, to) => model.notes[part].filter((n) => n.start >= from && n.start < to).map((n) => n.id);
        const plain = (m, part) => m.notes[part].map((n) => [n.start, n.length, n.pitch]);
        {script}
    """.replace("{sheet}", json.dumps(sheet)).replace("{script}", script))


@needs_node
def test_a_copy_is_measured_from_the_bar_line_before_it_and_runs_to_the_bar_line_after_it(tmp_path):
    """Pasted into another bar, the notes keep their places in the bar; duplicated, the copy starts a bar line on."""
    _sheet, got = copy_run(tmp_path, """
        const voice = r.clipOf(model, sheet, ids("Vocal", 0, 32), []);
        const both = r.clipOf(model, sheet, ids("Vocal", 0, 32), [0, 16, 32]);
        const chords = r.clipOf(model, sheet, [], [16, 32]);
        const none = r.clipOf(model, sheet, [], []);
        console.log(JSON.stringify({voice, both: both.chords, chords, none}));
    """)
    voice = got["voice"]
    assert (voice["origin"], voice["span"]) == (0, 32)
    assert [[n["at"], n["length"], n["pitch"]] for n in voice["notes"]["Vocal"]] == [
        [4, 4, 72], [8, 4, 74], [12, 4, 76], [16, 8, 79], [24, 8, 77]]
    assert voice["notes"]["Ins"] == [] and voice["chords"] == []
    assert got["both"] == [{"at": 0, "name": "C", "end": 16}, {"at": 16, "name": "G", "end": 32}], \
        "with notes, the chords are the ones above their bars"
    assert got["chords"] == {"origin": 16, "span": 32, "notes": {"Vocal": [], "Ins": []},
                             "chords": [{"at": 0, "name": "G", "end": 16}, {"at": 16, "name": "Am", "end": 32}]}, \
        "chords alone are copied as the blocks they are, each until the next chord"
    assert got["none"] is None


@needs_node
def test_a_paste_replaces_the_notes_under_it_and_shortens_the_one_that_ran_into_it(tmp_path):
    from yue2_comfy import notation
    _sheet, got = copy_run(tmp_path, """
        const clip = r.clipOf(model, sheet, ids("Vocal", 0, 32), []);
        const pasted = r.pasteClip(model, sheet, clip, 32, [["Vocal", "Vocal"]], false);
        const again = r.pasteClip(pasted.model, sheet, r.clipOf(pasted.model, sheet, pasted.ids, []), 64,
                                  [["Vocal", "Vocal"]], false);
        const other = r.pasteClip(model, sheet, clip, 32, [["Vocal", "Ins"]], false);
        const late = r.pasteClip(model, sheet, clip, 80, [["Vocal", "Vocal"]], false);
        console.log(JSON.stringify({vocal: plain(pasted.model, "Vocal"), ins: plain(pasted.model, "Ins"),
            ids: pasted.ids.length, fresh: pasted.ids.every((id) => !model.notes.Vocal.some((n) => n.id === id)),
            again: plain(again.model, "Vocal"), otherIns: plain(other.model, "Ins"),
            otherVocal: plain(other.model, "Vocal"), late, sheet: r.sheetOf(pasted.model)}));
    """)
    phrase = [[4, 4, 72], [8, 4, 74], [12, 4, 76], [16, 8, 79], [24, 8, 77]]
    moved = lambda shift: [[start + shift, length, pitch] for start, length, pitch in phrase]
    assert got["vocal"] == phrase + [[32, 4, 76]] + moved(32), "the e held from bar 3 is cut where the copy begins"
    assert got["ins"] == [[0, 16, 60], [16, 16, 55], [32, 16, 57], [48, 16, 53]]
    assert got["ids"] == 5 and got["fresh"]
    assert got["again"] == phrase + [[32, 4, 76]] + moved(32) + moved(64), "a copy of the copy goes on"
    assert got["otherIns"] == [[0, 16, 60], [16, 16, 55], [32, 4, 57]] + moved(32), "into the part being edited"
    assert got["otherVocal"] == phrase + [[32, 16, 76], [48, 8, 72]]
    assert got["late"] is None, "past the end the window adds bars first"
    written = notation.write(COPY_SCORE, got["sheet"])
    back = notation.read(written["abc"])
    assert [[n["start"], n["length"], n["pitch"]] for n in back["notes"]["Vocal"]] == got["vocal"]
    assert written["bars"] == [2, 3]


@needs_node
def test_pasted_chords_are_written_as_blocks_and_the_chord_after_them_rings_on(tmp_path):
    from yue2_comfy import notation
    _sheet, got = copy_run(tmp_path, """
        const chords = r.pasteClip(model, sheet, r.clipOf(model, sheet, [], [16, 32]), 48, [], true);
        const both = r.pasteClip(model, sheet, r.clipOf(model, sheet, ids("Vocal", 0, 32), [0, 16]), 32,
                                 [["Vocal", "Vocal"], ["Ins", "Ins"]], true);
        const none = r.pasteClip(model, sheet, r.clipOf(model, sheet, [], [16]), 0, [["Vocal", "Vocal"]], false);
        console.log(JSON.stringify({chords: chords.model.chords, starts: chords.starts, both: both.model.chords,
                                    bothStarts: both.starts, bothIns: plain(both.model, "Ins"), none,
                                    sheet: r.sheetOf(both.model)}));
    """)
    assert got["chords"] == [{"start": 0, "name": "C"}, {"start": 16, "name": "G"}, {"start": 32, "name": "Am"},
                             {"start": 48, "name": "G"}, {"start": 64, "name": "Am"}, {"start": 80, "name": "F"}]
    assert got["starts"] == [48, 64]
    assert got["both"] == [{"start": 0, "name": "C"}, {"start": 16, "name": "G"}, {"start": 32, "name": "C"},
                           {"start": 48, "name": "G"}, {"start": 64, "name": "F"}]
    assert got["bothStarts"] == [32, 48]
    assert got["bothIns"] == [[0, 16, 60], [16, 16, 55], [32, 16, 57], [48, 16, 53]], "the copy had no Ins notes"
    assert got["none"] is None, "nothing the part being edited can take"
    back = notation.read(notation.write(COPY_SCORE, got["sheet"])["abc"])
    assert [(c["start"], c["name"]) for c in back["chords"]] == [(c["start"], c["name"]) for c in got["both"]]


@needs_node
def test_a_shortcut_is_the_key_under_the_finger_in_any_layout():
    """With a Russian layout Ctrl+Z reports the key as a Cyrillic letter; the physical key is still Z."""
    got = run_roll("""
        console.log(JSON.stringify([
            {key: "z", code: "KeyZ"}, {key: "Z", code: "KeyZ"}, {key: "\u044f", code: "KeyZ"},
            {key: "\u0441", code: "KeyC"}, {key: "a", code: "KeyQ"}, {key: "Delete", code: "Delete"},
            {key: "\u043c", code: ""}].map((event) => r.shortcutLetter(event))));
    """)
    assert got == ["z", "z", "z", "c", "a", "delete", "\u043c"]


def test_the_window_copies_and_pastes_without_touching_the_graph_underneath():
    """Buttons at the top and the keys; the clipboard events never reach ComfyUI while the window is on top."""
    source = SCORE.read_text(encoding="utf-8")
    assert 'const CLIP_KEYS = { c: "copy", x: "cut", v: "paste", d: "duplicate", b: "duplicate" };' in source
    assert 'const CLIPBOARD_EVENTS = ["copy", "cut", "paste"];' in source
    for label in ('"Copy"', '"Cut"', '"Paste"', '"Duplicate"'):
        assert "[" + label.lower() + ", " + label + "," in source, label
    key = source.split("    key(event) {")[1].split("\n    }\n")[0]
    assert "const letter = roll.shortcutLetter(event);" in key
    branch = key.split("CLIP_KEYS[letter]) {")[1].split("}")[0]
    for call in ("event.preventDefault();", "event.stopPropagation();", "this.clipAction(CLIP_KEYS[letter]);"):
        assert call in branch, call
    assert "for (const type of CLIPBOARD_EVENTS) document.addEventListener(type, this.onClipboard, true);" in source
    assert "for (const type of CLIPBOARD_EVENTS) document.removeEventListener(type, this.onClipboard, true);" in source
    guard = source.split("    clipboardEvent(event) {")[1].split("\n    }\n")[0]
    assert "event.stopPropagation();" in guard and "if (!typing) event.preventDefault();" in guard
    place = source.split("    async placeClip(clip, at, verb) {")[1].split("\n    }\n")[0]
    assert "await this.addBars(count);" in place and "if (this.track) {" in place
    assert "roll.pasteClip(this.model, this.sheet, clip, at, routes, withChords)" in place
    draw = source.split("    draw() {")[1].split("\n    }\n")[0]
    assert "this.clipButtons.paste.disabled = !CLIP || !this.sheet;" in draw


def test_a_pack_window_is_a_modal_dialog_so_comfyui_keeps_its_keys_to_itself():
    """ComfyUI's change tracker undoes the graph on Ctrl+Z from a window-level capture listener, before any
    listener of ours runs; it stands aside only while a modal dialog is open, which it finds by these attributes."""
    source = (WEB / "yue2_controls.js").read_text(encoding="utf-8")
    made = source.split("export function frame(")[1].split("\n}\n")[0]
    assert 'panel.setAttribute("role", "dialog");' in made
    assert 'panel.setAttribute("aria-modal", "true");' in made
    assert "back.remove();" in made, "closing takes the dialog out of the page, and ComfyUI's keys come back"


def test_a_window_asked_to_be_sizable_gets_grips_a_fill_button_and_keeps_its_size():
    """M30.1: the lyrics editor was a fixed 920 px with a text box of about sixteen lines.

    Eight grips sit on the panel's edges and corners, inside its padding, so no
    content is under them. A drag grows the panel on both sides at once: it stays
    centred where `frame` puts it, and the dragged edge stays under the pointer
    because the size changes by twice the pointer's travel. The button in the top
    right corner fills the browser tab (the owner's choice over the browser's own
    full screen, which takes Escape for itself) and brings the size back.

    Size and fill are kept per kind of window, so the editor opens as it was last
    left on every node. A width or height nobody dragged is not written, and the
    window's own stylesheet keeps deciding it. Storage throws in a private window;
    the window must open there anyway, at its usual size. Only windows that ask
    are sizable: a question box stays the size of its question.
    """
    source = (WEB / "yue2_controls.js").read_text(encoding="utf-8")
    made = source.split("export function frame(")[1].split("\n}\n")[0]
    assert made.startswith("{ onClose, sticky, size } = {})")
    assert "if (size) sizable(panel, size);" in made
    sizing = source.split("function sizable(panel, kind) {")[1].split("\n}\n")[0]
    assert 'const GRIPS = ["n", "s", "e", "w", "ne", "nw", "se", "sw"];' in source
    assert "for (const side of GRIPS)" in sizing and "setPointerCapture" in sizing
    assert 'side.includes("e") ? 2 : side.includes("w") ? -2 : 0' in sizing
    assert 'side.includes("s") ? 2 : side.includes("n") ? -2 : 0' in sizing
    assert "window.innerWidth - EDGE" in sizing and "window.innerHeight - EDGE" in sizing
    assert '"lostpointercapture", end' in sizing and sizing.count("save()") >= 2
    assert 'panel.classList.toggle("yue2-max", filled);' in sizing
    assert "if (kept.w > 0)" in sizing and "if (kept.h > 0)" in sizing
    for name in ("function rememberedSize(", "function rememberSize("):
        body = source.split(name)[1].split("\n}\n")[0]
        assert "localStorage" in body and "catch" in body, name
    assert '"yue2.window." + kind' in source
    assert ".yue2-panel.yue2-sized.yue2-max { width: calc(100vw - 16px) !important;" in source
    for asked in ("export function confirmed(", "export function warned("):
        assert "size" not in source.split(asked)[1].split("\n}\n")[0], asked
    editor = (WEB / "yue2_editor.js").read_text(encoding="utf-8")
    assert 'size: this.lyricsOnly ? "lyrics" : "song",' in editor


def test_the_lyrics_editor_opens_as_text_when_it_was_left_as_text():
    """M30.1: people who edit lyrics dozens of times pressed 'Edit as text' every time.

    The mode is remembered for the browser, not per node, and only offered where
    the lyrics are written in this window (a wire brings them otherwise). In text
    mode the box fills what the window has left and the token count follows the
    typing. A token answer used to redraw the lyrics card; with the editor now
    opening straight into the text box, that would replace the box under the
    person's fingers a moment after they start typing, so in text mode an answer
    only repaints the count. 'Back to the transcription' keeps the mode it found.
    """
    source = (WEB / "yue2_editor.js").read_text(encoding="utf-8")
    assert "this.asText = this.lyricsFree && remembered(TEXT_KEY);" in source
    for name in ("function remembered(", "function remember("):
        body = source.split(name)[1].split("\n}\n")[0]
        assert "localStorage" in body and "catch" in body, name
    toggle = source.split("    toggleText() {")[1].split("\n    }\n")[0]
    assert "remember(TEXT_KEY, false);" in toggle and "remember(TEXT_KEY, true);" in toggle
    tokens = source.split("    async fetchTokens() {")[1].split("\n    }\n")[0]
    text_branch = tokens.split("if (this.asText) {")[1].split("}")[0]
    assert "this.paintStatus(this.status)" in text_branch and "return;" in text_branch
    assert tokens.index("if (this.asText) {") < tokens.index("this.renderLyrics()")
    lyrics = source.split("    renderLyrics(keep) {")[1].split("\n    }\n")[0]
    text_mode = lyrics.split("if (this.asText) {")[1].split("\n        }\n")[0]
    assert 'area.className = "yue2-text-all";' in text_mode and "this.scheduleTokens();" in text_mode
    assert 'scroller?.classList.toggle("yue2-textmode", this.asText && this.lyricsFree);' in lyrics
    back = source.split("    async backToTags() {")[1].split("\n    }\n")[0]
    assert "this.asText" not in back
    assert ".yue2-panel textarea.yue2-text-all { flex: 1 1 auto; min-height: 0; resize: none; }" in source


def test_the_style_folds_into_one_line_to_give_the_lyrics_the_window():
    """M30.1: on Generate Song the style card takes half the window above the lyrics box.

    Folded, the card is its title and the style line; a click on the line opens
    it again. The choice is remembered like the text mode, and the lyrics-only
    editor of Transcribe and Load MIDI has no style card to fold.
    """
    source = (WEB / "yue2_editor.js").read_text(encoding="utf-8")
    assert "this.styleFolded = !this.lyricsOnly && remembered(FOLD_KEY);" in source
    fold = source.split("    foldStyle(folded) {")[1].split("\n    }\n")[0]
    assert "remember(FOLD_KEY, folded);" in fold and "this.renderStyle();" in fold
    style = source.split("    renderStyle() {")[1].split("\n    }\n")[0]
    folded = style.split("if (this.styleFolded) {")[1].split("\n        }\n")[0]
    assert "this.foldStyle(false)" in folded and "return;" in folded
    assert style.index("if (this.styleFolded) {") < style.index("if (!this.styleFree) {")


def test_a_sized_window_never_cuts_its_own_content():
    """M30: the score window's toolbars wrap as it narrows, and its body has a floor of its own.

    A drag first grows the height to what the content needs (scrollHeight over
    clientHeight), so the window stops shrinking where Apply would go under its
    edge. When even the whole tab is too short for the content at that width,
    the step is refused, so a narrowing drag stops at the narrowest width that
    still fits. Whatever slips through anyway -- a tab with taller content
    chosen after the drag, a smaller browser -- scrolls inside the panel rather
    than being cut. The fill button takes no focus on a click, so the roll and
    the lyrics box keep the keys they had.
    """
    source = (WEB / "yue2_controls.js").read_text(encoding="utf-8")
    sizing = source.split("function sizable(panel, kind) {")[1].split("\n}\n")[0]
    assert "max-height: calc(100vh - 16px); overflow: auto; }" in source
    fit = sizing.split("    function fit() {")[1].split("\n    }\n")[0]
    assert "panel.scrollHeight - panel.clientHeight" in fit and "window.innerHeight - EDGE" in fit
    move = sizing.split("const move = (moved) => {")[1].split("\n        };\n")[0]
    assert "over: overflowing()" in move
    assert move.index("fit();") < move.index("if (!was.over && overflowing()) {")
    refused = move.split("if (!was.over && overflowing()) {")[1].split("}")[0]
    assert "panel.style.width = was.width;" in refused and "panel.style.height = was.height;" in refused
    assert 'fill.addEventListener("mousedown", (event) => event.preventDefault());' in sizing


def test_the_score_window_is_sizable_and_its_roll_cannot_spill_onto_the_status_line():
    """M30: the second window asked for, after the lyrics editor.

    The roll used to carry the 240 px floor while its body could shrink to
    nothing, so a short window let the roll and its scroll bar run over the
    status line: the panel itself never overflowed, and nothing could see it.
    The floor now sits on the body, which the roll fills, so a window too short
    for it overflows where the sizing guard sees it.
    """
    source = SCORE.read_text(encoding="utf-8")
    assert 'frame({ sticky: true, size: "score", onClose: () => this.closed() });' in source
    assert ".yue2-s-body { flex: 1 1 auto; min-height: 270px;" in source
    assert ".yue2-s-roll { position: relative; flex: 1 1 auto; min-height: 0;" in source


def test_text_on_the_node_is_a_switch_in_the_node_properties_and_off_by_default():
    """M30.2: a pack user retypes lyrics dozens of times and wanted to do it on the node.

    The owner chose a switch, off by default. It lives in the node's properties,
    not in a widget: widgets_values are positional and a new value widget would
    shift every saved file. Switched off, the property is deleted rather than set
    to false, so a saved workflow is byte for byte what it was before. Song nodes
    show ComfyUI's own style and lyrics boxes (the style one capped, the lyrics
    one taking the room); a box whose input is wired stays hidden. The switch is
    put last in the shared button row by CSS order, since the score editor
    appends its buttons to the same row after this one. A graph undo or a reload
    configures the node again, which is where the switch is read back.
    """
    source = (WEB / "yue2_editor.js").read_text(encoding="utf-8")
    assert 'const TEXT_PROPERTY = "yue2_text_on_node";' in source
    assert "return node?.properties?.[TEXT_PROPERTY] === true;" in source
    flip = source.split("function flipText(node) {")[1].split("\n}\n")[0]
    assert "if (on) node.properties[TEXT_PROPERTY] = true;" in flip
    assert "else delete node.properties[TEXT_PROPERTY];" in flip
    assert "if (on) roomForText(node);" in flip and "graphChanged();" in flip
    show = source.split("function showText(node) {")[1].split("\n}\n")[0]
    assert "const style = on && !sourceOf(node, STYLE);" in show
    assert "const lyrics = on && !sourceOf(node, LYRICS);" in show
    assert "showWidget(node, SUMMARY_UI, !boxes);" in show
    assert "styleBox.options.getMaxHeight = () => STYLE_BOX_H;" in source
    assert ".addWidget(" not in source
    assert ".yue2-buttons button.yue2-toggle { order: 1;" in source
    for hook in ("nodeType.prototype.onConfigure = function () {",
                 "nodeType.prototype.onConnectionsChange = function () {"):
        assert "setTimeout(() => refresh(this), 0);" in source.split(hook)[1].split("\n        };\n")[0], hook


def test_the_lyrics_typed_on_transcribe_are_kept_exactly_as_apply_keeps_them():
    """M30.2: Transcribe and Load MIDI keep an edit with a mark naming its recording.

    Their lyrics widget holds that marked text, so it is never shown raw: the
    switch shows a field of the pack's own that reads the edit or the node's
    own lyrics and writes through the same editValue call as the editor's
    Apply, so typing the node's own words back keeps no edit at all. The field
    is not refilled under a caret. Typing settles into one graph change after
    a pause (and on leaving the field) instead of one per key, which is how a
    native box behaves too; checked live on 8188 with a reload in between.
    """
    source = (WEB / "yue2_editor.js").read_text(encoding="utf-8")
    apply = source.split("    apply() {")[1].split("\n    }\n")[0]
    assert ("editValue(this.lyricsText(), this.node.__yue2Lyrics,\n"
            "                    this.node.__yue2Track || this.keptWords)") in apply
    field = source.split("function lyricsField(node) {")[1].split("\n}\n")[0]
    assert "widget.value = editValue(area.value, node.__yue2Lyrics,\n" \
           "            node.__yue2Track || splitMark(widget.value ?? \"\").words);" in field
    assert "settle = setTimeout(saved, FIELD_SETTLE);" in field
    assert 'area.addEventListener("change", saved);' in field
    assert "panelWidget(node, FIELD_UI, holder, () => FIELD_H, true);" in field
    fill = source.split("function fillField(node, box) {")[1].split("\n}\n")[0]
    assert "document.activeElement === area" in fill
    show = source.split("function showText(node) {")[1].split("\n}\n")[0]
    owned = show.split("if (sourceTexts(node)) {")[1].split("} else {")[0]
    assert "showWidget(node, FIELD_UI, boxes);" in owned and "LYRICS, " not in owned.replace("sourceOf(node, LYRICS)", "")


def test_the_track_window_is_sizable_and_its_wave_takes_the_height_it_is_given():
    """M30.3: the pack user spends most of their time in Edit Track and asked to stretch it.

    The window sizes and fills the tab like the others. It used to be as tall
    as its content, with a wave of 150-300 px and a body capped at 62vh, and it
    still opens that way until it is given a height. A height of its own (a drag
    that sets one, or filling the tab) marks the panel 'yue2-held', and then the
    cap goes and the wave grows into all the height the window gained: measured
    on 8188, a window 86 px taller gave the wave 325 px more, and the old fixed
    wave put back stayed at 272 px. The canvas follows its box through the
    ResizeObserver it already had.
    """
    track = (WEB / "yue2_track.js").read_text(encoding="utf-8")
    assert 'frame({ sticky: true, size: "track", onClose: () => this.closed() });' in track
    assert "    height: clamp(150px, 30vh, 300px);" in track
    assert ".yue2-panel.yue2-track.yue2-held .yue2-t-body { max-height: none; }" in track
    assert (".yue2-panel.yue2-track.yue2-held .yue2-t-canvas { flex: 1 1 auto; "
            "min-height: clamp(150px, 30vh, 300px); }") in track
    controls = (WEB / "yue2_controls.js").read_text(encoding="utf-8")
    sizing = controls.split("function sizable(panel, kind) {")[1].split("\n}\n")[0]
    assert ('const hold = () => panel.classList.toggle("yue2-held",\n'
            '        Boolean(panel.style.height) || panel.classList.contains("yue2-max"));') in sizing
    show = sizing.split("const show = (filled) => {")[1].split("\n    };\n")[0]
    assert "hold();" in show
    move = sizing.split("const move = (moved) => {")[1].split("\n        };\n")[0]
    assert move.index("hold();") < move.index("fit();")
    assert move.rstrip().endswith("hold();")


def test_the_words_column_beside_the_wave_has_a_splitter_that_remembers():
    """M30.3: the words column was a fixed 230 px.

    A splitter between the wave and the words moves it; the width is kept for
    the browser in try/catch like the window sizes, never squeezes the wave
    under LEFT_LEAST, and a double click puts it back. A song without words
    hides the splitter with the column.
    """
    track = (WEB / "yue2_track.js").read_text(encoding="utf-8")
    assert 'const WORDS_KEY = "yue2.track.words";' in track
    for name in ("function keptWordsWidth(", "function keepWordsWidth("):
        body = track.split(name)[1].split("\n}\n")[0]
        assert "localStorage" in body and "catch" in body, name
    width = track.split("    wordsWidth(width, keep) {")[1].split("\n    }\n")[0]
    assert "room - LEFT_LEAST" in width and "if (keep) keepWordsWidth(fitted);" in width
    assert 'this.split.addEventListener("dblclick", () => this.wordsWidth(WORDS_W, true));' in track
    assert "this.wordsWidth(keptWordsWidth(), false);" in track
    words = track.split("    paintWords() {")[1].split("\n    }\n")[0]
    assert words.count("this.split.hidden = ") == 2


WORDS_SONG = (
    'X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=100\n'
    'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
    'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
    'K:C\n% verse\nV: Vocal\nC2D2E2F2G2A2B2z2|C2D2E2F2G2A2B2z2|\nV: Ins\nZ2|\n')
"""A verse of two phrases of seven notes, each closed by a rest."""

BULLET = chr(0x2022)
DASH = chr(0x2014)


def laid_words(lyrics: str) -> dict:
    from yue2_comfy import notation, underlay
    return underlay.lay(notation.read(WORDS_SONG), lyrics)


def test_the_words_under_the_notes_go_by_the_same_names_on_both_sides():
    """M29: the window asks a route and reads a ui key that Python writes; a rename on one side draws no words."""
    from yue2_comfy import edits, routes, underlay
    score = (WEB / "yue2_score.js").read_text(encoding="utf-8")
    assert 'const SUNG_UI = "{}";'.format(edits.SUNG_UI) in score
    assert 'const UNDERLAY_ROUTE = "{}/score/words";'.format(routes.PREFIX) in score
    assert '@routes.post(PREFIX + "/score/words")' in (ROOT / "yue2_comfy" / "routes.py").read_text(encoding="utf-8")
    assert 'export const WORD_KINDS = ["{}", "{}", "{}"];'.format(
        underlay.FIT, underlay.SPARE, underlay.CROWD) in ROLL.read_text(encoding="utf-8")
    assert 'if (typeof sung === "string") this.__yue2Sung = sung;' in score


def test_the_words_are_laid_again_whenever_the_notes_or_the_lyrics_may_have_changed():
    """M29: a score read, an edit written, a run that brought new lyrics, and the tick of the box itself.

    The lyrics are looked up on every asking, so words typed on the node
    while the window is open are the ones laid under the next edit.
    """
    score = (WEB / "yue2_score.js").read_text(encoding="utf-8")
    adopt = score.split('    adopt(sheet, fit = true, prefer = "Eighth notes") {')[1].split("\n    }\n")[0]
    assert adopt.rstrip().endswith("this.layWords();")
    write = score.split("    async write() {")[1].split("\n    }\n")[0]
    assert "this.showDescription();\n            this.layWords();" in write
    arrived = score.split("    scoreArrived(detail) {")[1].split("\n    }\n")[0]
    assert "this.layWords();" in arrived
    lay = score.split("    async layWords() {")[1].split("\n    }\n")[0]
    assert "lyricsOf(this.track ? null : this.node, this.track)" in lay
    assert "if (this.isClosed || sequence !== this.wordsSequence) return;" in lay
    assert 'const SHOW_WORDS_KEY = "yue2.score.showWords";' in score
    for name in ("function rememberedShowWords(", "function rememberShowWords("):
        body = score.split(name)[1].split("\n}\n")[0]
        assert "localStorage" in body and "catch" in body, name


def test_the_notes_window_of_edit_track_is_given_the_songs_lyrics():
    track = (WEB / "yue2_track.js").read_text(encoding="utf-8")
    opened = track.split("this.notesEditor = openScoreFor(this.payload.score, {")[1].split("});")[0]
    assert 'lyrics: String(this.payload.lyrics || ""),' in opened


@needs_node
def test_the_voices_notes_are_counted_in_the_order_python_counts_them():
    """A syllable is drawn under the note Python numbered: by start, then by pitch, the model's own order left alone."""
    notes = [{"id": 1, "start": 8, "pitch": 60}, {"id": 2, "start": 0, "pitch": 64},
             {"id": 3, "start": 8, "pitch": 55}, {"id": 4, "start": 4, "pitch": 62}]
    found = run_roll("const notes = {};\n"
                     "console.log(JSON.stringify([r.inOrder(notes).map((n) => n.id), notes.map((n) => n.id), "
                     "r.inOrder(null)]));".format(json.dumps(notes)))
    assert found == [[2, 4, 3, 1], [1, 2, 3, 4], []]


@needs_node
def test_the_next_note_on_a_row_is_looked_for_a_few_notes_ahead():
    pitches = [60, 62, 64, 59, 60, 67, 67, 67, 67, 67, 67, 67, 67, 67, 60]
    notes = [{"start": index * 4, "pitch": pitch} for index, pitch in enumerate(pitches)]
    found = run_roll("const notes = {};\n"
                     "const at = (index, pitch, reach) => {{ const n = r.nextOn(notes, index, pitch, reach); "
                     "return n ? n.start / 4 : null; }};\n"
                     "console.log(JSON.stringify([at(0, 59), at(0, 60), at(0, 61), at(4, 60), at(4, 60, 20), "
                     "at(14, 60), at(99, 60)]));".format(json.dumps(notes)))
    assert found == [3, 4, None, None, 14, None, None]


PLACED_VIEW = ("{ from: 0, to: 1000, room: 150, least: 9, x: (tick) => tick * 5, "
               "width: (text) => text.length * 6 + 6 }")
"""A roll five pixels a tick wide, with letters six pixels wide."""


def placed(notes: list, words: list, view: str = PLACED_VIEW) -> list:
    """``[text, row, left, width, clipped]`` for each word the roll would draw, rows counted down from the note."""
    found = run_roll("const notes = {};\n"
                     "console.log(JSON.stringify(r.wordPlaces(notes, {}, {}).map((p) => "
                     "[p.text, notes[p.index].pitch - p.pitch, p.left, p.width, p.clipped])));"
                     .format(json.dumps(notes), json.dumps(words), view))
    return found


@needs_node
def test_a_word_goes_a_row_lower_while_the_word_before_it_is_still_running():
    """Three quick notes on one pitch: the second word steps down a row, the third is back under its note."""
    notes = [{"start": 0, "pitch": 65}, {"start": 4, "pitch": 65}, {"start": 8, "pitch": 65}]
    words = [[0, 0, "Mor-"], [1, 0, "ning"], [2, 0, "breaks"]]
    assert placed(notes, words) == [["Mor-", 1, 0, 30, False], ["ning", 2, 20, 30, False],
                                    ["breaks", 1, 40, 42, False]]


@needs_node
def test_a_word_steps_past_a_note_that_stands_on_its_row():
    """The next note a semitone down sits where the text would run: the row under that is free."""
    notes = [{"start": 40, "pitch": 60}, {"start": 44, "pitch": 59}]
    assert placed(notes, [[0, 0, "tomorrow"]]) == [["tomorrow", 2, 200, 54, False]]


@needs_node
def test_a_word_with_no_free_row_is_clipped_to_the_room_it_has_or_left_out():
    notes = [{"start": 60, "pitch": 70}, {"start": 62, "pitch": 69}, {"start": 63, "pitch": 68}]
    assert placed(notes, [[0, 0, "wonderful"]]) == [["wonderful", 2, 300, 13, True]]
    tight = [{"start": 60, "pitch": 70}, {"start": 61, "pitch": 69}, {"start": 61, "pitch": 68}]
    assert placed(tight, [[0, 0, "wonderful"]]) == []


@needs_node
def test_words_outside_the_view_are_not_placed_and_none_is_wider_than_its_room():
    notes = [{"start": index * 40, "pitch": 60} for index in range(6)]
    words = [[index, 0, "la"] for index in range(6)] + [[9, 0, "gone"]]
    view = PLACED_VIEW.replace("from: 0, to: 1000", "from: 60, to: 130")
    assert [row[2] for row in placed(notes, words, view)] == [400, 600]
    long = placed(notes[:1], [[0, 0, "a" * 80]])
    assert long == [["a" * 80, 1, 0, 150, False]]


@needs_node
def test_rows_grow_in_steps_and_the_first_step_draws_the_roll_as_before():
    """A studio asked for taller rows to read the notes and the words on a big screen.

    The remembered height is snapped to a known step, so a stored value from
    another version cannot leave the roll at a size no button reaches. At the
    default height every text is drawn at the size and on the line it had when
    the height was a constant: notes 9 px four up from the row's foot, keys 10 px
    three up, words 10 px on their 13 px box.
    """
    found = run_roll(
        "const sizes = r.ROW_HEIGHTS.map((h) => [h, r.rowTextSizes(h)]);\n"
        "console.log(JSON.stringify({heights: r.ROW_HEIGHTS, first: r.ROW_HEIGHT, least: r.NOTE_TEXT_LEAST,\n"
        "  snapped: ['14', '13', null, '48', 'abc', 17, ''].map(r.rowHeight),\n"
        "  steps: [[14, 1], [14, -1], [8, -1], [48, 1], [13, 1], [48, -20]].map(([h, s]) => r.nextRowHeight(h, s)),\n"
        "  sizes, lines: [r.textBaseline(100, 14, 9), r.textBaseline(100, 14, 10), r.textBaseline(100, 13, 10)],\n"
        "  inside: sizes.map(([h, s]) => [s.note, s.word, s.key, s.kit].map((size) => r.textBaseline(0, h, size)))}));")
    heights = found["heights"]
    assert heights == sorted(set(heights)) and found["first"] in heights
    assert heights[0] <= 10 and heights[-1] >= 40, "the steps reach well below and well above the old height"
    assert found["snapped"] == [14, 14, 14, 48, 14, 17, 14]
    assert found["steps"] == [17, 12, 8, 48, 17, 8]
    sizes = dict((height, size) for height, size in found["sizes"])
    assert sizes[found["first"]] == {"note": 9, "word": 10, "key": 10, "kit": 9}
    assert found["lines"] == [110, 111, 110]
    for kind in ("note", "word", "key", "kit"):
        grown = [sizes[height][kind] for height in heights]
        assert grown == sorted(grown), kind + " text shrinks while the rows grow"
        assert grown[-1] > grown[heights.index(found["first"])], kind + " text never grows"
    for height, lines in zip(heights, found["inside"]):
        assert all(0 < line <= height for line in lines), height
    assert sizes[heights[0]]["note"] < found["least"] <= sizes[12]["note"], \
        "notes are named from 12 px rows up; lower rows leave the names to the keyboard"


@needs_node
def test_a_row_zoom_keeps_the_row_under_the_pointer():
    """The wheel zooms where the pointer is, as Ctrl+wheel does in time: the row under it stays under it."""
    found = run_roll(
        "const out = [];\n"
        "for (const from of r.ROW_HEIGHTS) for (const to of r.ROW_HEIGHTS) for (let offset = 0; offset < 700; offset += 7) {\n"
        "  const top = r.keptPitchTop(79, offset, from, to);\n"
        "  out.push([79 - Math.floor(offset / from), top - Math.floor(offset / to)]);\n"
        "}\n"
        "console.log(JSON.stringify(out));")
    assert found and all(before == after for before, after in found)


def test_the_rows_are_zoomed_by_buttons_beside_the_time_zoom_and_by_alt_wheel():
    """The person looked for it beside '-' and '+', so it sits there, and the roll has no fixed row height left."""
    score = (WEB / "yue2_score.js").read_text(encoding="utf-8")
    assert not re.search(r"\bROW_H\b", score), "a row drawn at a constant height would not follow the zoom"
    assert "this.rowH = rememberedRowHeight();" in score
    tools = score.split("this.rollTools.append(")[1].split(");")[0]
    assert "across, upDown, fit" in tools
    assert 'rowsDown.addEventListener("click", () => this.zoomRows(-1));' in score
    assert 'rowsUp.addEventListener("click", () => this.zoomRows(1));' in score
    wheel = score.split("    wheel(event) {")[1].split("\n    }\n")[0]
    assert wheel.index("held.alt") < wheel.index("held.ctrl"), "AltGr reads as Ctrl too, so Alt is asked first"
    assert "this.zoomRows(delta < 0 ? 1 : -1, py)" in wheel
    zoom = score.split("    zoomRows(step, anchorPy) {")[1].split("\n    }\n")[0]
    assert "roll.keptPitchTop(" in zoom and "rememberRowHeight(height);" in zoom and "this.clampView();" in zoom
    assert 'const ROW_HEIGHT_KEY = "yue2.score.rowHeight";' in score
    for name in ("function rememberedRowHeight(", "function rememberRowHeight("):
        body = score.split(name)[1].split("\n}\n")[0]
        assert "localStorage" in body and "catch" in body, name


@needs_node
def test_what_the_route_answers_is_what_the_window_draws():
    payload = laid_words("[Verse]\nla la la la la la la\nna na na na na\n")
    found = run_roll("const laid = r.wordsLaid({});\n"
                     "console.log(JSON.stringify([laid.notes.length, laid.spare, laid.lines.map((l) => l.kind), "
                     "laid.paired, laid.bare, laid.sung, r.wordsLaid(null), r.wordsLaid({{lines: []}}), "
                     "r.wordsLaid({{lines: [{{kind: 'odd'}}], notes: []}}), "
                     "r.wordsLaid({{lines: [], notes: [[0, 0, 'la'], 'x', [1, 2]]}}).notes]));"
                     .format(json.dumps(payload)))
    assert found == [12, [12, 13], ["fit", "spare"], True, False, 14, None, None, None, []]


@needs_node
def test_the_count_beside_the_facts_names_the_lines_that_do_not_fit():
    fit = laid_words("[Verse]\nla la la la la la la\nna na na na na na na\n")
    off = laid_words("[Verse]\nla la la la la la la\nna na na na na\n")
    crowded = laid_words("[Verse]\nla la la la la la la la la\nna na na na na na na\n")
    apart = laid_words("[Verse]\nla la la la la la la\n[Chorus]\nna na na na na na na\n[Bridge]\noh\n")
    bare = laid_words("")
    bare["bare"] = True
    found = run_roll(
        "const facts = (payload, problem) => r.wordsFacts(payload && r.wordsLaid(payload), problem);\n"
        "console.log(JSON.stringify([facts({}), facts({}), facts({}), facts({}), facts({}), facts({}), "
        "facts(null), facts(null, 'No lyrics yet.'), r.WORDS_FIT, r.WORDS_OFF, r.WORDS_UNPAIRED, r.WORDS_BARE, "
        "r.WORDS_NONE]));".format(*[json.dumps(item) for item in (fit, off, crowded, apart, bare, laid_words(""))]))
    whole, spare, crowd, unpaired, unnamed, none, nothing, problem = found[:8]
    fits, differs, unpaired_text, bare_text, none_text = found[8:]
    assert whole == {"text": "words fit", "title": fits, "warn": False}
    assert spare["text"] == "1 of 2 lines fit" and spare["warn"]
    assert spare["title"] == differs + "\n\n" + BULLET + " na na na na na " + DASH + " 5 syllables, 7 notes"
    assert crowd["title"].endswith(BULLET + " la la la la la la la la la " + DASH + " 9 syllables, 7 notes")
    assert unpaired["warn"] and unpaired["title"].endswith("\n\n" + unpaired_text)
    assert unnamed == {"text": "words not shown", "title": bare_text, "warn": False}
    assert none == {"text": "no words", "title": none_text, "warn": False}
    assert nothing is None
    assert problem == {"text": "words not shown", "title": "No lyrics yet.", "warn": False}

