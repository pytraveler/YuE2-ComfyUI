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
              "[Verse]\r\na\r\n\r\n  b  \n[Chorus]\n\tc", "line\n" * 40, "[Verse] sung anyway\n[x]"]
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
