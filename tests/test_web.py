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


SOUNDS = WEB / "yue2_sounds.js"

FAKE_AUDIO = """
const made = { osc: 0, noise: 0, gain: 0, filter: 0, started: 0, stopped: 0 };
function node(kind) {
    return {
        connect: (next) => next, start: () => { made.started += 1; }, stop: () => { made.stopped += 1; },
        frequency: { value: 0, setValueAtTime: () => {}, exponentialRampToValueAtTime: () => {} },
        gain: { value: 0, setValueAtTime: () => {}, linearRampToValueAtTime: () => {},
                exponentialRampToValueAtTime: () => {} },
        detune: { value: 0 }, Q: { value: 0 }, type: "", loop: false, buffer: null, kind,
    };
}
const context = {
    sampleRate: 48000, currentTime: 0,
    createOscillator: () => { made.osc += 1; return node("osc"); },
    createBufferSource: () => { made.noise += 1; return node("noise"); },
    createGain: () => { made.gain += 1; return node("gain"); },
    createBiquadFilter: () => { made.filter += 1; return node("filter"); },
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
    """The list in the window and the list the player can build are one list."""
    offered = run_sounds("console.log(JSON.stringify(s.CHOICES));")
    assert sorted(offered) == ["Ins", "Vocal", "chords"]
    for part, choices in offered.items():
        assert choices[0][0] == "piano", part + " does not start on the piano"
        assert s_defaults()[part] == "piano"
    assert [id for id, _ in offered["Ins"]].count("drums") == 1
    assert "drums" not in [id for id, _ in offered["Vocal"] + offered["chords"]]


def s_defaults():
    return run_sounds("console.log(JSON.stringify(s.DEFAULTS));")


def test_a_sound_a_part_does_not_offer_falls_back_to_the_piano():
    """A remembered choice outlives the list it was picked from."""
    assert run_sounds("console.log(JSON.stringify(["
                      "s.known('Ins', 'drums'), s.known('Vocal', 'drums'),"
                      "s.known('chords', 'pad'), s.known('Vocal', undefined)]));"
                      ) == ["drums", "piano", "pad", "piano"]


def test_the_kit_names_one_drum_for_every_row_and_repeats_each_octave():
    kit = run_sounds("console.log(JSON.stringify(s.KIT));")
    assert len(kit) == 12 and len(set(kit)) == 12
    named = run_sounds("console.log(JSON.stringify([60, 62, 66, 72, 48, 61].map(s.drumName)));")
    assert named == ["Kick", "Snare", "Hat", "Kick", "Kick", kit[1]]


def test_every_sound_builds_and_stops_what_it_started():
    """A voice that never stops leaves the note sounding after Stop."""
    for kind in ("synth", "bass", "pluck", "pad", "drums"):
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
    """One of the node's edits in the shape the window holds it in."""
    return {"op": edit.op, "bars": None if edit.bars is None else list(edit.bars),
            "seconds": None if edit.seconds is None else list(edit.seconds),
            "seed": edit.seed, "takes": edit.takes, "take": edit.take,
            "vary": edit.vary, "guide": edit.guide, "fade": edit.fade}


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
         "take": 2, "vary": None, "guide": None, "fade": None},
        {"op": "cut", "bars": [20, 24], "seconds": None, "seed": 0, "takes": 1, "take": None,
         "vary": None, "guide": None, "fade": None},
        {"op": "retake", "bars": None, "seconds": [4.5, 9.25], "seed": 7, "takes": 4,
         "take": None, "vary": None, "guide": None, "fade": None},
    ]
    assert json.loads(track.written(edits)) == json.loads(written[0])
    assert len(track.read(written[1], 1)) == 2
    assert track.read(written[2], 1)[0].take is None, "a take kept that is no longer sung"


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


def test_the_two_sides_offer_the_same_number_of_takes():
    from yue2_comfy import edit_track
    from yue2_comfy.inpaint import track

    source = EDITLIST.read_text(encoding="utf-8")
    assert "export const MAX_TAKES = {};".format(track.MAX_TAKES) in source
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
                "edits", "seconds", "total", "song"):
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
    assert "this.seedKnob.hidden = !canRetake;" in paint
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
    assert "sourceOf(node, AUDIO_IN)" in picked, (
        "a song joined to 'audio' wins, and running the node would run whatever sings it")
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


def test_a_song_joined_to_the_node_leaves_the_list_to_read_and_not_to_open():
    """The link wins over the field, so opening one here would move nothing but the field."""
    source = SONGS.read_text(encoding="utf-8")
    able = source[source.index("    function openable(row) {"):source.index("    function take()")]
    assert "!joined" in able, "a second click on a row opens nothing either, take() asking this"
    every = [line.strip() for line in source.splitlines() if "open.disabled" in line]
    assert every and all("openable(" in line or line == "open.disabled = true;" for line in every), (
        "every place that turns the button back on asks the same question: two of them did not, "
        "and the button came up live on a node with a link in it")
    assert 'let mine = joined ? "" : (chosen || "");' in source, (
        "and no row says it is the one on the node while the link is what the node is on")
    assert "if (joined) open.title = JOINED;" in source, "the button says why it is off"
    assert "labelled and deleted from here" in source, (
        "the rest of the list is what the picker is for as well, so it is not taken away")
    assert "this.songsButton.disabled = busy;" in TRACK.read_text(encoding="utf-8"), (
        "which is why the button that opens it stays where it is")


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
    assert "sourceOf(node, AUDIO_IN)" in choose, "the picker says so when the graph wins"
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
    assert "if (row.key === nodeSong(node)) return;" in choose, "picking the same song changes nothing"
    assert "await confirmed(" in choose, "edits are not thrown away silently"
    assert "writeList(node, []);" in choose
    assert "node.__yue2TrackDrawn = null;" in choose, "the old song's takes must not stay on screen"
    assert "arrived();" in choose, "and the window redraws from nothing"
