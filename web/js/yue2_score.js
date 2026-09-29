import { app } from "../../scripts/app.js";
import {
    ask, buttonRow, confirmed, element, frame, graphChanged, installStyle, onTop, panelWidget,
    setWidgetValue, showWidget, sourceOf, warned, widgetNamed,
} from "./yue2_controls.js";
import * as roll from "./yue2_roll.js";
import * as sounds from "./yue2_sounds.js";
import { PITCHES } from "./yue2_piano.js";
import { stretchesOf } from "./yue2_editlist.js";

const RENDER = "YuE2RenderPlan";
const GENERATE = "YuE2GenerateSong";
const TRANSCRIBE = "YuE2Transcribe";
const LOAD_MIDI = "YuE2LoadMidi";
const PLAN_NODES = ["YuE2Plan", "YuE2SelectPlan"];
const SELECT = "YuE2SelectPlan";
const BATCH = "YuE2PlanBatch";
const OPTIONS_NODE = "YuE2Options";
const SCORE = "score_abc";
const PLAN = "plan";
const PLANS = "plans";
const OPTIONS = "options";
const LYRICS = "lyrics";
const MAX_SECONDS = "max_seconds";
const MODE = "mode";
const VOCAL_TRACK = "vocal_track";
const INSTRUMENT_TRACK = "instrument_track";
const MUTED_MODES = [2, 4];
const SCORE_UI = "yue2_score";
const WORDS_UI = "yue2_words";
const TRACK_UI = "yue2_track";
const MARKS_UI = "yue2_marks";
const AUTO_SECONDS_UI = "yue2_auto_seconds";
const SCORE_EVENT = "yue2-score-written";
const READ_ROUTE = "/yue2/score/read";
const WRITE_ROUTE = "/yue2/score/write";
const MIDI_ROUTE = "/yue2/score/midi";
const LENGTH_ROUTE = "/yue2/score/length";
const TRANSPOSE_ROUTE = "/yue2/score/transpose";
const NEW_BARS = 16;
const NEW_BPM = 120;
const ADD_BARS = [1, 2, 4, 8, 16, 32];
const LONGER = "longer";
const LONGER_LABEL = "Add bars\u2026";
const KEEP_KEY = "yue2.score.keepWords";
const KEEP_LABEL = "Keep for new words";
const KEEP_TOOLTIP =
    "Sing this edit even after the lyrics or the style change, instead of a new score written for the new "
    + "words. A changed line is sung on the notes of the old one; a line added moves the lines after it "
    + "along the tune, and the last line of a section can be lost.";
const WRITE_DELAY = 250;
const READ_DELAY = 450;
const SUMMARY_H = 58;
const SONG_SUMMARY_H = 42;
const SCORE_SUMMARY = "yue2_score_summary";
const SONG_SUMMARY = "yue2_song_summary";
const ABCJS_URL = new URL("./vendor/abcjs-basic-min.cjs", import.meta.url).href;

const KEYS_W = 58;
const SECTION_H = 16;
const BAR_H = 30;
const RULER_H = SECTION_H + BAR_H;
const CHORD_H = 24;
const ROW_H = 14;
const EDGE_PX = 7;
const CHORD_BOX_W = 150;
const CHANGED_RED = "#C0392B";
const TIMES_KEPT = 200;
const NOTE_RADIUS = 2;
const SKIN = {
    back: "#243035",
    rowWhite: "#34434A",
    rowBlack: "#2D3A40",
    octave: "#222C31",
    snap: "#303E45",
    beat: "#28343A",
    bar: "#1A2226",
    changed: "rgba(240, 176, 88, 0.12)",
    locked: "rgba(255, 255, 255, 0.05)",
    note: "#A3F5B5",
    noteEdge: "#173D22",
    noteText: "#08200F",
    insNote: "#9FD0F7",
    insEdge: "#123650",
    insText: "#061B2B",
    chordNote: "#D9C2F5",
    chordRoot: "#B990EC",
    chordEdge: "#3A1F57",
    chordText: "#1D0B2E",
    refused: "rgba(236, 226, 240, 0.55)",
    refusedEdge: "#E08A8A",
    picked: "#FF9A91",
    pickedEdge: "#5E1B16",
    pickedText: "#2B0805",
    shine: "rgba(255, 255, 255, 0.55)",
    ghost: "rgba(196, 220, 230, 0.20)",
    ghostEdge: "rgba(196, 220, 230, 0.34)",
    box: "rgba(255, 255, 255, 0.75)",
    ruler: "#1C2529",
    rulerText: "#AFC3CC",
    rulerTick: "#3B4A51",
    lane: "#212B30",
    laneText: "#8FA3AC",
    chip: "#33444C",
    chipEdge: "#526973",
    chipText: "#E4EEF2",
    playhead: "#F5A623",
    limit: "#F2D15C",
    limitShade: "rgba(8, 12, 14, 0.45)",
    limitChip: "#4A4119",
    limitText: "#FFF1B8",
    keyWhite: "#E3E6E8",
    keyEdge: "#9EA6AA",
    keyBlack: "#1B1E20",
    keyBlackTop: "#3B4146",
    keyText: "#4A555B",
    keyTextC: "#1A2226",
    keyUnder: "#B7D2E4",
    keyBlackUnder: "#2F4553",
    keyBlackText: "#E4EEF2",
    keysEdge: "#11171A",
    sectionEdge: "#F0F4F6",
    sectionHeld: "rgba(255, 255, 255, 0.35)",
};

const BOTH = "both";
const CHORDS = "chords";
const PART_CHOICES = [
    ["Vocal", "Voice", "Draw and edit the voice part. The instrument part is drawn faintly behind it."],
    ["Ins", "Instrument", "Draw and edit the instrument part. The voice part is drawn faintly behind it."],
    [CHORDS, "Chords", "Draw and edit the chords as notes, each held until the next chord. Both parts are "
        + "drawn faintly behind them."],
    [BOTH, "Both", "Select and move the notes of both parts at once, with the chords above them. "
        + "To draw or stretch notes, pick one part."],
];
const CHORDS_NOTICE =
    "Chords: each chord is its notes, held until the next one; the named block is its root. Click an empty place "
    + "for a new chord in the key. Drag the root up or down for another root, sideways for another moment, and a "
    + "chord's right end to move the next one. Drag another note, or Ctrl+click a row, to change a chord note by "
    + "note. Right-click removes, a double click names. Measured on three songs: YuE2 plays an edited chord on "
    + "most of its beats and keeps the tune.";
const REPEATS_TOOLTIP =
    "Make each chord edit in the other sections of the same name too, where their chords are still the ones "
    + "this section had. YuE2 now and then carries an edit into a repeat by itself; this writes it there.";
const BOTH_NOTICE =
    "Both parts: the voice in green, the instrument in blue. Drag a box to select notes and the chords "
    + "above them, Ctrl+click to add or drop one, then drag or use the arrow keys (Shift for an octave). "
    + "Pick one part to draw notes.";
const BOTH_CHORDS = "In Both, a click on a chord selects it. Pick one part to add or change chords.";
const CLIP_TOOLS = [
    ["copy", "Copy", "Copy the selected notes or chords (Ctrl+C)."],
    ["cut", "Cut", "Copy the selected notes or chords and take them out (Ctrl+X)."],
    ["paste", "Paste", "Put the copy into the bar the playhead is in, at the same places in the bar (Ctrl+V). "
        + "It replaces what is under it, and bars it needs past the end are added."],
    ["duplicate", "Duplicate", "Put a copy of the selection right after it, from the next bar line (Ctrl+D or "
        + "Ctrl+B). Press again to go on."],
];
const PITCH_MOVES = [
    ["\u2212oct", -12, "Move what is selected down an octave (Shift+\u2193)."],
    ["\u22121", -1, "Move what is selected down a semitone (\u2193)."],
    ["+1", 1, "Move what is selected up a semitone (\u2191)."],
    ["+oct", 12, "Move what is selected up an octave (Shift+\u2191)."],
];
const SECTION_COLORS = [
    ["pre", "#9B7FD1"], ["intro", "#8C96A3"], ["verse", "#4E98C4"], ["chorus", "#D19A3F"],
    ["hook", "#D9774B"], ["bridge", "#4FA37A"], ["outro", "#7C8FA6"], ["solo", "#C98B5B"],
    ["inter", "#5FA3A3"], ["inst", "#5FA3A3"],
];
const COMMON_CHORDS = ["C", "C#", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    .flatMap((root) => ["", "m", "7", "maj7", "m7", "sus4", "dim", "aug"].map((q) => root + q));

const EDIT_LABEL = "Edit score\u2026";
const EDIT_TOOLTIP =
    "Open the score editor: a piano roll, the score drawn as notes, and its ABC text. " +
    "Nothing changes on the node until you press Apply.";
const RESET_LABEL = "Reset score";
const RESET_TOOLTIP_OWN =
    "Throw away the edit kept on this node. The next run writes the model's score again, " +
    "so a new seed gives a new tune.";
const RESET_TOOLTIP_PLAN =
    "Throw away the edit kept on this node. The next run sings the plan's own score.";
const RESET_TOOLTIP_TRANSCRIBE =
    "Throw away the edit kept on this node. The next run outputs the transcription again.";

const HONEST_NOTE =
    "YuE2 usually sings an edited melody, not always: measured on three songs, four rewritten " +
    "bars were sung as written in two and partly in the third. A small change of rhythm is " +
    "followed; a rest cut into a sung line gets filled with the words. Every edit is a new take " +
    "of the whole song -- ";
const RETRY_ON_PLAN =
    "if a bar does not take, change the seed on the plan node; the edited score stays in this node.";
const RETRY_HERE =
    "if a bar does not take, change the seed on this node: the edit stays, and only the performance " +
    "changes. The edit belongs to these words: with other lyrics or another style the node writes a " +
    "new score instead, unless '" + KEEP_LABEL + "' is ticked.";
const TRANSCRIBE_NOTE =
    "SheetSage2 writes down what it hears, and a real recording is heard less exactly than a clean mix: " +
    "play the melody, and fix a wrong note or chord here before the song is sung. The edit belongs to " +
    "this recording and this 'mode': with another recording the node transcribes anew instead.";

const RESET_TOOLTIP_MIDI =
    "Throw away the edit kept on this node. The next run outputs the file's score again.";
const MIDI_NOTE =
    "The score is written from the file's notes: a line of one note at a time on the file's own grid. Play it, " +
    "fix what the file or the choice of tracks got wrong, or pick other tracks on the node. The edit belongs to " +
    "this file, this 'mode' and these tracks: with another the node writes the file's score instead.";

const SOURCE_WORDS = {
    [TRANSCRIBE]: {
        title: "YuE2 Transcribe",
        back: "Back to the transcription",
        backTitle: "Throw away the edits and load the score this node transcribed on its last run.",
        result: "the transcription",
        heading: "The transcription",
        made: "Written from the recording, the same on every run. Edit score\u2026 to fix notes.",
        rest: "; the rest as transcribed.",
        wired: "Sent on as it arrives, instead of a transcription.",
        other: "Made for another recording or mode: this node transcribes anew instead.",
        pending: "This edit was made for another recording or the other 'mode', so the node transcribes anew "
            + "instead of sending it on.",
        last: "the last transcription",
        noScore: "Run once, and the transcription can be edited here.",
        empty: "There is no score yet. Run the workflow once: the transcription appears here, and fixing "
            + "a note after that does not listen to the recording again. Or paste a score into the ABC tab.",
        note: TRANSCRIBE_NOTE,
        resetTooltip: RESET_TOOLTIP_TRANSCRIBE,
    },
    [LOAD_MIDI]: {
        title: "YuE2 Load MIDI",
        back: "Back to the file's score",
        backTitle: "Throw away the edits and load the score this node wrote from the file on its last run.",
        result: "the file's score",
        heading: "The file's score",
        made: "Written from the MIDI file, the same on every run. Edit score\u2026 to change notes.",
        rest: "; the rest as in the file.",
        wired: "Sent on as it arrives, instead of the file's score.",
        other: "Made for another file, mode or choice of tracks: this node sends the file's score instead.",
        pending: "This edit was made for another file, the other 'mode' or other tracks, so the node sends "
            + "the file's score instead.",
        last: "the file's score from the last run",
        noScore: "Run once, and the score written from the file can be edited here.",
        empty: "There is no score yet. Run the workflow once: the score written from the file appears here. "
            + "Or paste a score into the ABC tab.",
        note: MIDI_NOTE,
        resetTooltip: RESET_TOOLTIP_MIDI,
    },
};

const LIMIT_TOOLTIP =
    "'max_seconds' in YuE2 Options ends the singing here, however far the score runs on. " +
    "Bars after the dashed line are not sung; raising 'max_seconds' brings them in, and makes a new take.";
const LIMIT_AUTO_TOOLTIP =
    "With 'max_seconds' at 0 a song may run " + roll.AUTO_SECONDS_PER_LINE + " seconds for each sung line plus "
    + roll.AUTO_BASE_SECONDS + ", at least " + roll.AUTO_MIN_SECONDS + " and at most " + roll.MAX_SECONDS
    + ", or " + roll.AUTO_INSTRUMENTAL_SECONDS + " when there are no lines to count. Bars after the dashed line "
    + "are not sung; a higher 'max_seconds' in YuE2 Options brings them in, and makes a new take.";

const TEMPO_TOOLTIP =
    "The tempo written into the score, Q:1/4 in the ABC. The notes keep their lengths in bars, "
    + "so the whole song is sung faster or slower and the dashed line moves with it. It goes to "
    + "the node with the rest of the edit, on Apply.";

const KEY_TOOLTIP =
    "The key the song is written in. Pick another and the whole score moves there by the semitones "
    + "shown: both parts, every chord and the key itself, the way 'transpose' moves it. Typing another "
    + "K: into the ABC text does not do this: the same letters are read in the new key, so an F-sharp "
    + "turns into an F. Ctrl+Z moves the song back.";

const CHORD_HELP =
    "Chord symbols are a root from A to G with an optional # or b, then one of: nothing (major), " +
    "m, dim, aug, 7, maj7, m7, dim7, m7b5, sus4, sus2, 6, m6, 7sus4, m(maj7) -- and an optional " +
    "bass note after a slash, as in C/E. Leave the box empty to remove the chord.";

const KINDS_HELP =
    "The score knows fifteen kinds on any root: major, m, dim, aug, 7, maj7, m7, dim7, m7b5, sus4, sus2, 6, "
    + "m6, 7sus4 and m(maj7), each with a bass note under it if wanted.";

const SECTION_HELP =
    "A section name is a line of text of at most " + roll.SECTION_LONGEST + " characters. The list " +
    "offers the names the model itself writes, and the words under the lyrics follow them.";

const SECTION_NOTICE =
    "Sections are the top strip: drag a boundary to move it, click a name to change it or to " +
    "empty the box and join it to the section before, and click anywhere else up there to start " +
    "a new one at that bar.";

const DOUBLE_NOTICE =
    " A double click starts and stops the sound, as Space does: on the bar strip it plays from " +
    "the bar you clicked, and on the roll it leaves no note behind.";

const TRACK_SUB =
    "The notes the song is sung to. Change them here: only the bars whose notes change are sung "
    + "again, under the new score, and the rest of the song stays as it was sung. Nothing is "
    + "sung until the track window's blue button.";

const TRACK_NOTE =
    "YuE2 follows a changed note most of the time, not every time: measured on three songs, the "
    + "new note was sung on 49 of 50 changed notes of a pop verse, 61 of 81 in a rap and 17 of 24 "
    + "in a ballad, and none of them on the old score. Each change is sung as takes to choose "
    + "from, and another seed is one more try.";

const TRACK_APPLY = "Sing the changed bars";

const TRACK_APPLY_WHY =
    "Puts the change on the track's list: the bars whose notes changed, as one edit, or one edit "
    + "a place when they lie far apart. Render sings them.";

const TRACK_BACK = "Back to the song's notes";

const TRACK_BACK_WHY = "Throw away the changes made here and show the notes the song has.";

const TRACK_FIXED =
    "The tempo, the key, the number of bars and the sections belong to the whole song, which is kept as it "
    + "was sung, so they stay as they are here. Only notes and chords change.";

const TRACK_SAME = "No note has changed, so there is nothing to sing again.";

const CUT_NOTICE =
    "This score stops in the middle of a line: the model ran out of room while writing it. The roll shows "
    + "the whole bars before that, and an edit leaves the unfinished end out.";

const FINER = "finer";
const FINER_LABEL = "Thirty-second notes (rewrites the score)";

const FINER_NOTICE =
    "The whole score is written again on thirty-second notes, because a grid step is one note " +
    "length and the format has no fractions. Not a note moves, and the ABC tab shows the new " +
    "text; Cancel still leaves the node as it was.";

const SCORE_STYLE_ID = "yue2-score-style";
const SCORE_STYLE = `
.yue2-panel.yue2-score { width: min(1500px, 98vw); }
.yue2-score [hidden] { display: none !important; }
.yue2-score h3 { font-size: 15px; font-weight: 600; margin: 0; }
.yue2-score .yue2-s-sub { font-size: 11px; color: var(--descrip-text, #999); margin: 2px 0 10px; }
.yue2-score button, .yue2-score select, .yue2-score input[type="text"],
.yue2-score input[type="number"], .yue2-score textarea {
    box-sizing: border-box; font: inherit; font-size: 12px; padding: 4px 9px; border-radius: 6px;
    color: var(--input-text, #ddd); background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-score button { cursor: pointer; }
.yue2-score button:hover { background: var(--comfy-menu-bg, #353535); border-color: #6a6a6a; }
.yue2-score button.yue2-s-on, .yue2-score button.yue2-s-go {
    background: #3B7DD8; border-color: #3B7DD8; color: #fff; }
.yue2-score button[disabled] { opacity: 0.45; cursor: not-allowed; }
.yue2-s-bar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 8px; }
.yue2-s-grow { flex: 1 1 auto; }
.yue2-s-gap { width: 10px; }
.yue2-s-facts { font-size: 12px; color: var(--descrip-text, #999); white-space: nowrap; }
.yue2-s-facts .yue2-s-warn { cursor: help; }
.yue2-s-check { font-size: 12px; display: inline-flex; gap: 4px; align-items: center;
    color: var(--descrip-text, #bbb); user-select: none; }
.yue2-score select.yue2-s-sound { font-size: 11px; padding: 2px 4px; margin-left: -1px;
    max-width: 96px; }
.yue2-score button.yue2-s-on:hover { background: #4A8BE3; border-color: #4A8BE3; }
.yue2-s-seg { display: inline-flex; }
.yue2-score .yue2-s-seg button { border-radius: 0; margin-left: -1px; }
.yue2-score .yue2-s-seg button:first-child { border-radius: 6px 0 0 6px; margin-left: 0; }
.yue2-score .yue2-s-seg button:last-child { border-radius: 0 6px 6px 0; }
.yue2-s-tempo { font-size: 12px; display: inline-flex; gap: 6px; align-items: center;
    color: var(--descrip-text, #bbb); user-select: none; }
.yue2-s-tempo input[type="range"] { width: 120px; margin: 0; accent-color: #3B7DD8; }
.yue2-score input.yue2-s-bpm { width: 62px; padding: 4px 6px; text-align: right; }
.yue2-s-body { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
.yue2-s-roll { position: relative; flex: 1 1 auto; min-height: 240px; outline: none;
    border: 1px solid var(--border-color, #4e4e4e); border-radius: 6px; overflow: hidden; }
.yue2-s-roll:focus { border-color: #3B7DD8; }
.yue2-s-roll canvas { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; }
.yue2-s-scroll { width: 100%; margin: 6px 0 0; accent-color: #3B7DD8; }
.yue2-s-paper { flex: 1 1 auto; min-height: 240px; overflow: auto; background: #fbfaf7; color: #222;
    border-radius: 6px; padding: 10px 14px; }
.yue2-s-paper .yue2-s-note { font-size: 12px; color: #555; margin: 0 0 6px; }
.yue2-score textarea.yue2-s-text { flex: 1 1 auto; min-height: 240px; width: 100%; resize: none;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace; line-height: 1.5; }
.yue2-s-status { font-size: 12px; min-height: 18px; margin: 6px 0 2px; color: var(--descrip-text, #aaa); }
.yue2-s-status.yue2-s-bad { color: #E08A8A; }
.yue2-s-status button { margin-left: 8px; padding: 1px 9px; font-size: 11px; }
.yue2-s-hint { font-size: 11px; line-height: 1.45; color: var(--descrip-text, #999); margin: 2px 0 0; }
.yue2-s-foot { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
.yue2-s-foot button { font-size: 13px; padding: 6px 14px; }
.yue2-s-or { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--border-color, #444); }
.yue2-s-empty { margin: auto; padding: 40px 10px; text-align: center; font-size: 13px; line-height: 1.6;
    color: var(--descrip-text, #aaa); max-width: 560px; }
.yue2-s-empty button { margin-top: 12px; }
.yue2-score input.yue2-s-chord { position: absolute; z-index: 2; width: 150px; height: 22px; padding: 1px 6px; }
.yue2-score input.yue2-s-chord.yue2-s-bad { border-color: #E08A8A; }
.yue2-s-summary { width: 100%; height: 100%; box-sizing: border-box; overflow: hidden; padding: 5px 8px;
    border-radius: 6px; cursor: pointer; font-family: system-ui, sans-serif; font-size: 11px;
    line-height: 16px; color: var(--input-text, #ddd); background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-s-summary:hover { border-color: #3B7DD8; }
.yue2-s-summary div { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.yue2-s-key { font-weight: 600; }
.yue2-s-dim { color: var(--descrip-text, #999); }
.yue2-s-warn { color: #E0A45A; }
.yue2-buttons button[hidden] { display: none !important; }
`;

let AUDIO = null;
let PIANO_LOADING = null;
const PIANO = new Map();
const PIANO_LEVEL = { Vocal: 0.32, Ins: 0.14, chords: 0.09 };
const WAVE_LEVEL = { Vocal: 0.16, Ins: 0.07, chords: 0.03 };
const RELEASE = 0.09;
let ABCJS_LOADING = null;
let CLIP = null;
const CLIPBOARD_EVENTS = ["copy", "cut", "paste"];
const CLIP_KEYS = { c: "copy", x: "cut", v: "paste", d: "duplicate", b: "duplicate" };
const TIMES = new Map();
const TIMES_ASKED = new Set();

function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
}

function now() {
    return typeof performance === "object" && performance ? performance.now() : Date.now();
}

function hashText(text) {
    let hash = 0x811c9dc5;
    const source = String(text || "");
    for (let i = 0; i < source.length; i++) {
        hash ^= source.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash.toString(16);
}

function barList(numbers) {
    const sorted = [...new Set(numbers)].sort((a, b) => a - b);
    const out = [];
    let first = null;
    let last = null;
    const flush = () => {
        if (first === null) return;
        out.push(first === last ? String(first + 1) : (first + 1) + "-" + (last + 1));
    };
    for (const n of sorted) {
        if (first !== null && n === last + 1) {
            last = n;
            continue;
        }
        flush();
        first = n;
        last = n;
    }
    flush();
    return out.join(", ");
}

function sectionColor(name) {
    const lower = String(name || "").toLowerCase();
    const found = SECTION_COLORS.find(([word]) => lower.includes(word));
    return found ? found[1] : "#9A9A9A";
}

function linkOrigin(node, slot) {
    const link = node?.inputs?.[slot]?.link;
    if (link === null || link === undefined) return null;
    const record = app.graph?.links?.get?.(link) ?? app.graph?.links?.[link];
    return record ? app.graph.getNodeById(record.origin_id) : null;
}

function planSource(node) {
    let origin = sourceOf(node, PLAN);
    for (let hop = 0; origin && hop < 16; hop++) {
        if (PLAN_NODES.includes(origin.type) || PLAN_NODES.includes(origin.comfyClass)) return origin;
        if (origin.type !== "Reroute") return null;
        origin = linkOrigin(origin, 0);
    }
    return null;
}

function isGenerate(node) {
    return node?.type === GENERATE || node?.comfyClass === GENERATE;
}

function isTranscribe(node) {
    return node?.type === TRANSCRIBE || node?.comfyClass === TRANSCRIBE;
}

function isLoadMidi(node) {
    return node?.type === LOAD_MIDI || node?.comfyClass === LOAD_MIDI;
}

function sourceWords(node) {
    return isTranscribe(node) ? SOURCE_WORDS[TRANSCRIBE] : isLoadMidi(node) ? SOURCE_WORDS[LOAD_MIDI] : null;
}

function isOwn(node) {
    return isGenerate(node) || Boolean(sourceWords(node));
}

function backLabel(node) {
    return sourceWords(node)?.back || "Back to the model's score";
}

function wordsWanted(origin) {
    if (!sourceWords(origin)) return origin?.__yue2Words || null;
    const chosen = widgetNamed(origin, MODE)?.value;
    return origin.__yue2Marks?.[chosen] || origin.__yue2Words || null;
}

function isRender(node) {
    return node?.type === RENDER || node?.comfyClass === RENDER;
}

function scoreSource(node) {
    if (isOwn(node)) return { node, own: true };
    const origin = planSource(node);
    return origin ? { node: origin, own: false } : null;
}

function isNodeOf(node, type) {
    return node?.type === type || node?.comfyClass === type;
}

function upstreamOf(node, name, wanted) {
    let origin = node ? sourceOf(node, name) : null;
    for (let hop = 0; origin && hop < 16; hop++) {
        if (wanted(origin)) return origin;
        if (origin.type !== "Reroute") return null;
        origin = linkOrigin(origin, 0);
    }
    return null;
}

function optionsNodeFor(node) {
    const found = upstreamOf(node, OPTIONS, (origin) => isNodeOf(origin, OPTIONS_NODE));
    return found && !MUTED_MODES.includes(found.mode) ? found : null;
}

function limitFor(node) {
    if (sourceWords(node)) return null;
    const own = isGenerate(node);
    const source = own ? node : planSource(node);
    if (!source) return null;
    const writer = isNodeOf(source, SELECT)
        ? upstreamOf(source, PLANS, (origin) => isNodeOf(origin, BATCH))
        : source;
    const settings = optionsNodeFor(node) || (own ? null : optionsNodeFor(writer));
    let most = settings || writer ? 0 : null;
    if (settings) {
        most = sourceOf(settings, MAX_SECONDS) ? null : Number(widgetNamed(settings, MAX_SECONDS)?.value) || 0;
    }
    const typed = writer && !sourceOf(writer, LYRICS) ? widgetNamed(writer, LYRICS)?.value : null;
    return roll.lengthLimit(most, typeof typed === "string" ? typed : null, source.__yue2AutoSeconds);
}

function limitReason(limit) {
    return limit.auto ? "the length limit worked out from the lyrics" : "'max_seconds'";
}

function limitRemedy(limit) {
    return limit.auto ? "unless 'max_seconds' is set higher in YuE2 Options" : "until it is raised";
}

function sheetTimes(abc) {
    const text = String(abc || "").trim();
    if (!text) return null;
    if (TIMES.has(text)) return TIMES.get(text);
    if (TIMES_ASKED.has(text)) return null;
    TIMES_ASKED.add(text);
    ask(READ_ROUTE, { abc: text })
        .then(({ ok, payload }) => (ok && payload?.ok ? {
            bpm: payload.sheet.bpm, per_quarter: payload.sheet.per_quarter, seconds: payload.sheet.seconds,
            bars: payload.sheet.bars.map((bar) => ({ start: bar.start })),
        } : false))
        .catch(() => false)
        .then((times) => {
            TIMES_ASKED.delete(text);
            TIMES.set(text, times);
            while (TIMES.size > TIMES_KEPT) TIMES.delete(TIMES.keys().next().value);
            refreshScoreSummaries();
        });
    return null;
}

function graphNodes() {
    return app.graph?._nodes ?? app.graph?.nodes ?? [];
}

const SOUND_KEY = "yue2.score.sounds";

const SOUND_TITLE = {
    Vocal: "What the voice part sounds like in this window.",
    Ins: "What the instrument part sounds like in this window. 'drums' plays the rows as a kit, "
        + "and while this part is the one being edited the keyboard names them -- Kick, Snare, "
        + "Hat -- instead of the notes. The kit repeats in every octave.",
    chords: "What the chord lane sounds like in this window.",
};

const SOUND_NOTICE = " The sound is for your ear here only: YuE2 is given this score and the style "
    + "line, never an instrument, so what plays the song is the style line.";

function rememberedSounds() {
    const kept = { ...sounds.DEFAULTS };
    try {
        const stored = JSON.parse(window.localStorage.getItem(SOUND_KEY) || "{}");
        for (const part of Object.keys(kept)) kept[part] = sounds.known(part, stored[part]);
    } catch (error) {
        void error;
    }
    return kept;
}

function rememberSounds(chosen) {
    try {
        window.localStorage.setItem(SOUND_KEY, JSON.stringify(chosen));
    } catch (error) {
        void error;
    }
}

function rememberedKeep() {
    try {
        return window.localStorage.getItem(KEEP_KEY) === "1";
    } catch (error) {
        void error;
        return false;
    }
}

function rememberKeep(keep) {
    try {
        window.localStorage.setItem(KEEP_KEY, keep ? "1" : "0");
    } catch (error) {
        void error;
    }
}

function audioContext() {
    if (!AUDIO) AUDIO = new (window.AudioContext || window.webkitAudioContext)();
    if (AUDIO.state === "suspended") AUDIO.resume();
    return AUDIO;
}

function midiHz(pitch) {
    return 440 * Math.pow(2, (pitch - 69) / 12);
}

function pianoUrl(pitch) {
    return new URL("./piano/" + pitch + ".ogg", import.meta.url).href;
}

function nearestSampled(pitch) {
    let best = PITCHES[0];
    for (const have of PITCHES) {
        if (Math.abs(have - pitch) < Math.abs(best - pitch)) best = have;
    }
    return best;
}

function loadPiano() {
    if (!PIANO_LOADING) {
        const context = audioContext();
        PIANO_LOADING = Promise.all(PITCHES.map((pitch) => fetch(pianoUrl(pitch))
            .then((response) => {
                if (!response.ok) throw new Error("HTTP " + response.status + " for " + pianoUrl(pitch));
                return response.arrayBuffer();
            })
            .then((bytes) => context.decodeAudioData(bytes))
            .then((buffer) => PIANO.set(pitch, buffer))))
            .catch((error) => {
                PIANO_LOADING = null;
                throw error;
            });
    }
    return PIANO_LOADING;
}

function loadAbcjs() {
    if (window.ABCJS) return Promise.resolve(window.ABCJS);
    if (!ABCJS_LOADING) {
        ABCJS_LOADING = fetch(ABCJS_URL)
            .then((response) => {
                if (!response.ok) throw new Error("HTTP " + response.status + " for " + ABCJS_URL);
                return response.text();
            })
            .then((code) => {
                const script = document.createElement("script");
                script.textContent = "(function (define, exports, module) {\n" + code
                    + "\n}).call(window);\n//# sourceURL=abcjs-basic-min.js";
                document.head.appendChild(script);
                if (!window.ABCJS) throw new Error("abcjs loaded but did not start");
                return window.ABCJS;
            })
            .catch((error) => {
                ABCJS_LOADING = null;
                throw error;
            });
    }
    return ABCJS_LOADING;
}

class Player {
    constructor() {
        this.voices = [];
        this.playing = false;
        this.started = 0;
        this.master = null;
        this.sounds = { ...sounds.DEFAULTS };
    }

    play(list) {
        this.stop();
        const context = audioContext();
        const master = context.createGain();
        master.gain.value = 0.9;
        master.connect(context.destination);
        this.master = master;
        const origin = context.currentTime + 0.06;
        this.started = origin;
        for (const item of list) {
            this.voices.push(...this.tone(context, master, origin + item.at, item.length, item.pitch, item.part));
        }
        this.playing = true;
    }

    soundOf(part) {
        return sounds.known(part, this.sounds?.[part]);
    }

    tone(context, master, start, length, pitch, part) {
        const kind = this.soundOf(part);
        if (kind !== "piano") {
            return sounds.play(context, master, start, length, pitch, kind,
                               WAVE_LEVEL[part] ?? WAVE_LEVEL.chords);
        }
        return [PIANO.size
            ? this.struck(context, master, start, length, pitch, part)
            : this.wave(context, master, start, length, pitch, part)];
    }

    struck(context, master, start, length, pitch, part) {
        const take = nearestSampled(pitch);
        const source = context.createBufferSource();
        const gain = context.createGain();
        const level = PIANO_LEVEL[part] ?? PIANO_LEVEL.chords;
        source.buffer = PIANO.get(take);
        source.playbackRate.value = Math.pow(2, (pitch - take) / 12);
        const stop = start + Math.max(0.06, length);
        gain.gain.setValueAtTime(level, start);
        gain.gain.setValueAtTime(level, stop);
        gain.gain.linearRampToValueAtTime(0, stop + RELEASE);
        source.connect(gain).connect(master);
        source.start(start);
        source.stop(stop + RELEASE + 0.02);
        return source;
    }

    wave(context, master, start, length, pitch, part) {
        const oscillator = context.createOscillator();
        const gain = context.createGain();
        const level = WAVE_LEVEL[part] ?? WAVE_LEVEL.chords;
        oscillator.type = part === "Vocal" ? "triangle" : part === "Ins" ? "square" : "sine";
        oscillator.frequency.value = midiHz(pitch);
        const stop = start + Math.max(0.06, length);
        gain.gain.setValueAtTime(0, start);
        gain.gain.linearRampToValueAtTime(level, start + 0.012);
        gain.gain.setValueAtTime(level, Math.max(start + 0.012, stop - 0.04));
        gain.gain.linearRampToValueAtTime(0, stop);
        oscillator.connect(gain).connect(master);
        oscillator.start(start);
        oscillator.stop(stop + 0.02);
        return oscillator;
    }

    blip(pitch, part) {
        const context = audioContext();
        const gain = context.createGain();
        gain.gain.value = 0.9;
        gain.connect(context.destination);
        this.tone(context, gain, context.currentTime + 0.01, 0.22, pitch, part);
    }

    elapsed() {
        return this.playing ? Math.max(0, audioContext().currentTime - this.started) : 0;
    }

    stop() {
        for (const oscillator of this.voices) {
            try {
                oscillator.stop();
            } catch (error) {
                void error;
            }
        }
        this.voices = [];
        if (this.master) this.master.disconnect();
        this.master = null;
        this.playing = false;
    }
}

function selectControl(options, value, title, onChange) {
    const list = document.createElement("select");
    for (const [key, label] of options) {
        const option = document.createElement("option");
        option.value = String(key);
        option.textContent = label;
        option.selected = String(key) === String(value);
        list.appendChild(option);
    }
    list.title = title;
    list.addEventListener("change", () => onChange(list.value));
    return list;
}

function checkControl(label, checked, title) {
    const holder = element("label", "yue2-s-check");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = checked;
    holder.title = title;
    holder.append(box, label);
    return { holder, box };
}

class ScoreEditor {
    constructor(node, track = null) {
        this.track = track;
        this.node = node || { title: track?.title || "", properties: {}, widgets: [] };
        const source = track ? null : scoreSource(this.node);
        this.origin = source?.node ?? null;
        this.own = Boolean(source?.own);
        this.source = track ? null : sourceWords(this.node);
        this.back = track ? TRACK_BACK : backLabel(this.node);
        this.planScore = track ? String(track.text || "").trim() : this.origin?.__yue2Score || null;
        this.planWords = this.origin?.__yue2Words || null;
        this.limit = track ? null : limitFor(this.node);
        this.baseWords = null;
        this.player = new Player();
        this.sounds = rememberedSounds();
        this.player.sounds = this.sounds;
        this.sequence = 0;
        this.writeSequence = 0;
        this.isClosed = false;
        this.tab = "roll";
        this.part = "Vocal";
        this.both = false;
        this.chordPart = false;
        this.snap = 1;
        this.lastLength = 0;
        this.selection = new Set();
        this.chordPicks = new Set();
        this.drag = null;
        this.working = null;
        this.hoverPitch = null;
        this.sheetTempo = null;
        this.shownTempo = null;
        this.tempoFrom = null;
        this.moved = null;
        this.keyBusy = false;
        this.shownKeys = null;
        this.playhead = 0;
        this.playTick = null;
        this.chordInput = null;
        this.sectionInput = null;
        this.sectionBoxes = [];
        this.notice = null;
        this.changed = [];
        this.localChanged = [];
        this.sheet = null;
        this.model = null;
        this.good = null;
        this.history = new roll.History();
        this.tick0 = 0;
        this.pxPerTick = 4;
        this.pitchTop = 79;
        this.pianoWait = false;
        this.width = 0;
        this.height = 0;
        this.ratio = 1;
        installStyle(SCORE_STYLE_ID, SCORE_STYLE);
        this.build();
        this.open();
        loadPiano().catch((error) => {
            void error;
        });
    }

    build() {
        const { panel, close, handle } = frame({ sticky: true, onClose: () => this.closed() });
        panel.classList.add("yue2-score");
        this.panel = panel;
        this.close = close;
        this.handle = handle;
        handle.onEscape = () => this.escape();

        panel.appendChild(element("h3", "", (this.track ? "Notes \u2014 " : "Score \u2014 ")
            + (this.node.title || (this.source ? this.source.title : "YuE2 Render Plan"))));
        panel.appendChild(element("p", "yue2-s-sub", this.track ? TRACK_SUB
            : (this.source ? "The notes this node sends on." : "The notes this node sings.")
                + " Nothing is written to the node until Apply."));

        const tabs = element("div", "yue2-s-bar");
        this.tabButtons = {};
        for (const [id, label, title] of [
            ["roll", "Piano roll", "Draw, move and stretch notes; click the chord lane to set chords."],
            ["notes", "Notes", "The score drawn as sheet music, with the bars the edit rewrites in red."],
            ["abc", "ABC", "The score as the model reads it. Paste a score here to load it."],
        ]) {
            const button = element("button", "", label);
            button.title = title;
            button.addEventListener("click", () => this.showTab(id));
            this.tabButtons[id] = button;
            tabs.appendChild(button);
        }
        tabs.appendChild(element("span", "yue2-s-grow"));
        this.facts = element("span", "yue2-s-facts");
        tabs.appendChild(this.facts);
        panel.appendChild(tabs);

        const tools = element("div", "yue2-s-bar");
        this.playButton = element("button", "", "Play");
        this.playButton.title = "Play from the marker (Space, or a double click on the roll). "
            + "A sampled piano and a few synthesised "
            + "voices, not YuE2: the song itself is made when the workflow runs, and the sound "
            + "picked beside each part is for your ear here only.";
        this.playButton.addEventListener("click", () => this.togglePlay());
        const rewind = element("button", "", "From start");
        rewind.title = "Put the play marker back at the first bar.";
        rewind.addEventListener("click", () => {
            this.stopPlaying();
            this.playhead = 0;
            this.tick0 = 0;
            this.clampView();
            this.draw();
        });
        this.hearVoice = checkControl("voice", true, "Play the voice part.");
        this.hearIns = checkControl("instrument", true, "Play the instrument part.");
        this.hearChords = checkControl("chords", false, "Play the chord symbols as soft held chords.");
        tools.append(this.playButton, rewind,
            this.hearVoice.holder, this.soundControl("Vocal"),
            this.hearIns.holder, this.soundControl("Ins"),
            this.hearChords.holder, this.soundControl("chords"),
            element("span", "yue2-s-gap"), this.tempoControl(), element("span", "yue2-s-gap"),
            this.keyControl(), element("span", "yue2-s-gap"));

        this.rollTools = element("span", "yue2-s-bar");
        this.rollTools.style.margin = "0";
        this.partHolder = element("span", "yue2-s-seg");
        this.partButtons = {};
        for (const [id, label, title] of PART_CHOICES) {
            const button = element("button", "", label);
            button.title = title;
            button.addEventListener("click", () => this.pickPart(id));
            this.partButtons[id] = button;
            this.partHolder.appendChild(button);
        }
        this.repeats = checkControl("repeats too", false, REPEATS_TOOLTIP);
        this.repeats.holder.hidden = true;
        this.repeats.box.addEventListener("change", () => this.rollBox.focus());
        this.showPart();
        this.snapHolder = element("span");
        const zoomOut = element("button", "", "\u2212");
        zoomOut.title = "Zoom out (Ctrl + wheel).";
        zoomOut.addEventListener("click", () => this.zoom(1 / 1.4));
        const zoomIn = element("button", "", "+");
        zoomIn.title = "Zoom in (Ctrl + wheel).";
        zoomIn.addEventListener("click", () => this.zoom(1.4));
        const fit = element("button", "", "Whole song");
        fit.title = "Fit the whole song into the window.";
        fit.addEventListener("click", () => {
            this.pxPerTick = Math.max(0.05, (this.width - KEYS_W) / Math.max(1, this.sheet?.total || 1));
            this.tick0 = 0;
            this.clampView();
            this.draw();
        });
        this.undoButton = element("button", "", "Undo");
        this.undoButton.title = "Undo (Ctrl+Z).";
        this.undoButton.addEventListener("click", () => this.undo());
        this.redoButton = element("button", "", "Redo");
        this.redoButton.title = "Redo (Ctrl+Shift+Z).";
        this.redoButton.addEventListener("click", () => this.redo());
        this.barsHolder = element("span");
        const clipTools = element("span", "yue2-s-seg");
        this.clipButtons = {};
        for (const [id, label, title] of CLIP_TOOLS) {
            const button = element("button", "", label);
            button.title = title;
            button.disabled = true;
            button.addEventListener("click", () => {
                this.clipAction(id);
                this.rollBox.focus();
            });
            clipTools.appendChild(button);
            this.clipButtons[id] = button;
        }
        const pitch = element("span", "yue2-s-seg");
        this.pitchButtons = PITCH_MOVES.map(([label, semitones, title]) => {
            const button = element("button", "", label);
            button.title = title;
            button.disabled = true;
            button.addEventListener("click", () => {
                this.shiftPicked(0, semitones);
                this.rollBox.focus();
            });
            pitch.appendChild(button);
            return button;
        });
        this.rollTools.append(this.partHolder, this.repeats.holder, this.snapHolder, this.barsHolder, zoomOut, zoomIn, fit,
            this.undoButton, this.redoButton, element("span", "yue2-s-gap"), clipTools, element("span", "yue2-s-gap"),
            pitch);
        tools.appendChild(this.rollTools);
        panel.appendChild(tools);

        const body = element("div", "yue2-s-body");
        this.rollBox = element("div", "yue2-s-roll");
        this.rollBox.tabIndex = 0;
        this.canvas = document.createElement("canvas");
        this.canvas.title = SECTION_NOTICE + DOUBLE_NOTICE;
        this.rollBox.appendChild(this.canvas);
        this.scroller = document.createElement("input");
        this.scroller.type = "range";
        this.scroller.className = "yue2-s-scroll";
        this.scroller.min = "0";
        this.scroller.step = "1";
        this.scroller.addEventListener("input", () => {
            this.tick0 = Number(this.scroller.value);
            this.draw();
        });
        this.paper = element("div", "yue2-s-paper");
        this.textBox = document.createElement("textarea");
        this.textBox.className = "yue2-s-text";
        this.textBox.spellcheck = false;
        this.textBox.addEventListener("input", () => {
            clearTimeout(this.readTimer);
            this.readTimer = setTimeout(() => this.readTyped(), READ_DELAY);
        });
        this.empty = element("div", "yue2-s-empty");
        body.append(this.rollBox, this.scroller, this.paper, this.textBox, this.empty);
        panel.appendChild(body);

        this.status = element("div", "yue2-s-status");
        panel.appendChild(this.status);
        panel.appendChild(element("p", "yue2-s-hint", this.track ? TRACK_NOTE
            : this.source ? this.source.note : HONEST_NOTE + (this.own ? RETRY_HERE : RETRY_ON_PLAN)));

        const foot = element("div", "yue2-s-foot");
        this.writeButton = element("button", "", "Write the score");
        this.writeButton.title = "Run only the plan node feeding this one: the score is written, nothing is sung.";
        this.writeButton.addEventListener("click", () => this.writeScore());
        this.resetButton = element("button", "", this.back);
        this.resetButton.title = this.track
            ? TRACK_BACK_WHY
            : this.source
            ? this.source.backTitle
            : this.own
            ? "Throw away the edits and load the score this node wrote on its last run."
            : "Throw away the edits and load the score the plan node wrote.";
        this.resetButton.addEventListener("click", () => this.reset());
        const cancel = element("button", "", "Cancel");
        cancel.addEventListener("click", () => this.escape());
        this.applyButton = element("button", "yue2-s-go", this.track ? TRACK_APPLY : "Apply");
        if (this.track) this.applyButton.title = TRACK_APPLY_WHY;
        this.applyButton.addEventListener("click", () => this.apply());
        this.saveButton = element("button", "", "Save as MIDI\u2026");
        this.saveButton.title = "Download the score as it stands in this window, edits included, as a MIDI file: "
            + "the voice, the instrument line and the chords, each on a track of its own.";
        this.saveButton.addEventListener("click", () => this.saveMidi());
        this.keepWords = checkControl(KEEP_LABEL, false, KEEP_TOOLTIP);
        this.keepWords.holder.hidden = Boolean(this.track || this.source);
        this.keepWords.box.addEventListener("change", () => this.keepChanged());
        foot.append(this.writeButton, this.resetButton, this.saveButton, element("span", "yue2-s-grow"),
            this.keepWords.holder, cancel, this.applyButton);
        panel.appendChild(foot);

        const chords = document.createElement("datalist");
        chords.id = "yue2-s-chords";
        for (const name of COMMON_CHORDS) {
            const option = document.createElement("option");
            option.value = name;
            chords.appendChild(option);
        }
        panel.appendChild(chords);

        const sections = document.createElement("datalist");
        sections.id = "yue2-s-sections";
        for (const name of roll.SECTION_NAMES) {
            const option = document.createElement("option");
            option.value = name;
            sections.appendChild(option);
        }
        panel.appendChild(sections);

        this.canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
        this.canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
        this.canvas.addEventListener("pointerleave", () => this.hover(null));
        this.canvas.addEventListener("pointerup", (event) => this.pointerUp(event));
        this.canvas.addEventListener("pointercancel", () => this.cancelDrag());
        this.canvas.addEventListener("mousedown", (event) => event.preventDefault());
        this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());
        this.canvas.addEventListener("wheel", (event) => this.wheel(event), { passive: false });
        panel.addEventListener("keydown", (event) => this.key(event));
        panel.addEventListener("keyup", (event) => {
            if (event.key === "Alt" || event.key === "AltGraph") event.preventDefault();
        });
        this.onStrayKey = (event) => this.strayKey(event);
        document.addEventListener("keydown", this.onStrayKey, true);
        this.onClipboard = (event) => this.clipboardEvent(event);
        for (const type of CLIPBOARD_EVENTS) document.addEventListener(type, this.onClipboard, true);
        this.observer = new ResizeObserver(() => this.resize());
        this.observer.observe(this.rollBox);

        this.onScore = (event) => this.scoreArrived(event.detail);
        window.addEventListener(SCORE_EVENT, this.onScore);
    }

    open() {
        if (this.track) {
            this.fromBox = false;
            this.opened = this.planScore;
            this.baseWords = null;
            this.load(this.opened);
            return;
        }
        const box = roll.splitMark(widgetNamed(this.node, SCORE)?.value ?? "");
        const base = this.node.properties?.yue2_score_base;
        this.fromBox = Boolean(box.score);
        this.opened = box.score || String(this.planScore || "").trim();
        this.baseWords = box.score ? box.words : this.planWords;
        this.keepWords.box.checked = box.score && box.words ? box.keep : rememberedKeep();
        const wanted = this.origin ? wordsWanted(this.origin) : null;
        const other = "This edit was made for other words than " + (this.own ? "the last run had" : "the plan has now");
        if (box.score && box.words && wanted && box.words !== wanted && box.keep && !this.source) {
            this.pendingNote = other + ", and it is kept for new words, so it is sung with them. '" + this.back
                + "' loads the new score.";
            this.pendingBad = false;
        } else if (box.score && box.words && wanted && box.words !== wanted) {
            this.pendingNote = this.source
                ? this.source.pending + " '" + this.back + "' loads " + this.source.last + "."
                : other + ", so it is not sung until they come back. Tick '" + KEEP_LABEL + "' and Apply to sing "
                    + "it with the new ones, or '" + this.back + "' loads the new score.";
            this.pendingBad = true;
        } else if (box.score && this.planScore && base && base !== hashText(String(this.planScore).trim())) {
            this.pendingNote = box.words
                ? "This edit was made on an earlier take of these words, and it is still sung. "
                    + "'" + this.back + "' loads the newer take."
                : "This edit was made on an earlier score; the plan has written a new one since. "
                    + "'" + this.back + "' loads the new one.";
            this.pendingBad = !box.words;
        }
        this.load(this.opened);
    }

    closed() {
        this.isClosed = true;
        this.stopPlaying();
        clearTimeout(this.writeTimer);
        clearTimeout(this.readTimer);
        this.observer?.disconnect();
        window.removeEventListener(SCORE_EVENT, this.onScore);
        document.removeEventListener("keydown", this.onStrayKey, true);
        for (const type of CLIPBOARD_EVENTS) document.removeEventListener(type, this.onClipboard, true);
    }

    clipboardEvent(event) {
        if (this.isClosed || !onTop(this.handle)) return;
        const target = event.target;
        const typing = target?.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName);
        if (typing && !this.panel.contains(target)) return;
        event.stopPropagation();
        if (!typing) event.preventDefault();
    }

    strayKey(event) {
        const target = event.target;
        if (this.isClosed || !onTop(this.handle) || this.panel.contains(target)) return;
        if (target?.closest?.(".yue2-panel") || target?.isContentEditable
            || ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName)) return;
        this.key(event);
        if (event.defaultPrevented) event.stopPropagation();
    }

    setStatus(text, bad = false) {
        this.status.textContent = text || "";
        this.status.classList.toggle("yue2-s-bad", Boolean(bad));
    }

    edited() {
        return (this.current || "").trim() !== (this.opened || "").trim();
    }

    async load(text) {
        this.stopPlaying();
        const clean = String(text || "").trim();
        this.base = clean;
        this.current = clean;
        this.changed = [];
        this.localChanged = [];
        this.sheet = null;
        this.model = null;
        this.good = null;
        this.history = new roll.History();
        this.selection = new Set();
        this.chordPicks = new Set();
        this.notice = null;
        this.moved = null;
        this.textBox.value = clean;
        this.notesDrawn = null;
        if (!clean) {
            this.fillEmpty();
            this.showTab(this.tab === "abc" ? "abc" : "roll");
            this.refresh();
            return;
        }
        const sequence = ++this.sequence;
        this.setStatus("Reading the score\u2026");
        const { ok, payload } = await ask(READ_ROUTE, { abc: clean });
        if (this.isClosed || sequence !== this.sequence) return;
        if (!ok || !payload.ok) {
            this.setStatus(payload.error || "The score could not be read.", true);
            this.fillEmpty(payload.error);
            this.showTab("abc");
            this.refresh();
            return;
        }
        this.adopt(payload.sheet);
        if (this.pendingNote) {
            this.setStatus(this.pendingNote, this.pendingBad);
            this.pendingNote = null;
        }
    }

    adopt(sheet, fit = true, prefer = "Eighth notes") {
        this.sheet = sheet;
        this.model = roll.modelOf(sheet);
        this.good = this.model;
        this.localChanged = [];
        this.sheetTempo = sheet.bpm;
        this.shownTempo = null;
        this.tempoFrom = null;
        const tempi = roll.tempoRange(sheet.bpm);
        for (const box of [this.tempoSlider, this.tempoBox]) {
            box.min = String(tempi.low);
            box.max = String(tempi.high);
        }
        this.fillSnap(sheet, prefer);
        this.fillBars();
        this.paintFacts();
        this.showKey();
        this.playhead = 0;
        if (fit) this.fitView();
        if (sheet.cut) this.notice = CUT_NOTICE;
        this.showTab(this.tab);
        this.showDescription();
        this.refresh();
    }

    soundControl(part) {
        const pick = selectControl(sounds.CHOICES[part], this.sounds[part],
            SOUND_TITLE[part] + SOUND_NOTICE, (value) => {
                this.sounds[part] = sounds.known(part, value);
                this.player.sounds = this.sounds;
                rememberSounds(this.sounds);
                this.stopPlaying();
                this.draw();
            });
        pick.className = "yue2-s-sound";
        return pick;
    }

    soundOf(part) {
        return sounds.known(part, this.sounds[part]);
    }

    drumRows() {
        return !this.both && !this.chordPart && this.part === "Ins" && this.soundOf("Ins") === sounds.DRUMS;
    }

    pickPart(id) {
        const both = id === BOTH;
        const chords = id === CHORDS;
        if (!both && !chords) this.part = id;
        const wasChords = this.chordPart;
        this.both = both;
        this.chordPart = chords;
        const kept = new Set(this.model ? this.editedParts().flatMap((part) => this.model.notes[part].map((n) => n.id)) : []);
        this.selection = new Set([...this.selection].filter((noteId) => kept.has(noteId)));
        if (!both && !chords) this.chordPicks.clear();
        this.showPart();
        if (this.sheet) {
            this.notice = both ? BOTH_NOTICE : chords ? CHORDS_NOTICE : null;
            this.showDescription();
            if (chords && !wasChords) this.showChordRows();
        }
        this.draw();
        this.rollBox.focus();
    }

    showPart() {
        const shown = this.both ? BOTH : this.chordPart ? CHORDS : this.part;
        for (const [id, button] of Object.entries(this.partButtons)) button.classList.toggle("yue2-s-on", id === shown);
        if (this.repeats) this.repeats.holder.hidden = !this.chordPart;
    }

    editedParts() {
        if (this.chordPart) return [];
        return this.both ? [this.part, this.part === "Vocal" ? "Ins" : "Vocal"] : [this.part];
    }

    clearPicks() {
        this.selection.clear();
        this.chordPicks.clear();
    }

    chordsPickable() {
        return this.both || this.chordPart;
    }

    pickedChords() {
        return this.chordsPickable()
            ? this.model.chords.map((c) => c.start).filter((start) => this.chordPicks.has(start)) : [];
    }

    showChordRows() {
        const pitches = this.model.chords.flatMap((chord) => roll.chordPitches(chord.name));
        const low = pitches.length ? Math.min(...pitches) : roll.CHORD_ROOT;
        const high = pitches.length ? Math.max(...pitches) : roll.CHORD_ROOT + 16;
        const rows = this.visibleRows();
        if (low >= this.pitchTop - rows + 1 && high <= this.pitchTop) return;
        this.pitchTop = Math.round((low + high) / 2) + Math.floor(rows / 2);
        this.clampView();
    }

    noteById(model, id) {
        const part = roll.partOf(model, id);
        return part ? { note: model.notes[part].find((n) => n.id === id), part } : null;
    }

    tempoControl() {
        const holder = element("span", "yue2-s-tempo");
        holder.title = this.track ? TRACK_FIXED : TEMPO_TOOLTIP;
        this.tempoSlider = document.createElement("input");
        this.tempoSlider.type = "range";
        this.tempoSlider.step = "1";
        this.tempoSlider.min = String(roll.TEMPO_LOW);
        this.tempoSlider.max = String(roll.TEMPO_HIGH);
        this.tempoSlider.addEventListener("input", () => this.previewTempo(this.tempoSlider.value));
        this.tempoSlider.addEventListener("change", () => this.setTempo(this.tempoSlider.value));
        this.tempoBox = document.createElement("input");
        this.tempoBox.type = "number";
        this.tempoBox.className = "yue2-s-bpm";
        this.tempoBox.step = "1";
        this.tempoBox.addEventListener("change", () => this.setTempo(this.tempoBox.value));
        holder.append(element("span", "", "Tempo"), this.tempoSlider, this.tempoBox,
            element("span", "", "BPM"));
        return holder;
    }

    previewTempo(value) {
        if (!this.model || !this.sheet) return;
        const bpm = roll.tempoOf(value, this.sheetTempo);
        if (bpm === null || bpm === this.model.bpm) return;
        if (!this.tempoFrom) this.tempoFrom = this.model;
        this.model = { ...this.model, bpm };
        this.draw();
    }

    setTempo(value) {
        if (!this.model || !this.sheet) return;
        const bpm = roll.tempoOf(value, this.sheetTempo);
        const from = this.tempoFrom || this.model;
        this.tempoFrom = null;
        this.model = from;
        if (bpm === null || bpm === from.bpm) {
            this.shownTempo = null;
            this.draw();
            return;
        }
        this.commit({ ...from, bpm }, from);
    }

    showTempo() {
        const bpm = this.model && this.model.bpm;
        this.tempoSlider.disabled = !bpm || Boolean(this.track);
        this.tempoBox.disabled = !bpm || Boolean(this.track);
        if (!bpm) {
            this.shownTempo = null;
            this.tempoBox.value = "";
            return;
        }
        if (!this.sheet || this.shownTempo === bpm) return;
        this.shownTempo = bpm;
        this.sheet.bpm = bpm;
        this.sheet.seconds = roll.secondsAt(this.sheet, this.sheet.total);
        this.tempoSlider.value = String(bpm);
        this.tempoBox.value = String(bpm);
        this.paintFacts();
    }

    keyControl() {
        const holder = element("span", "yue2-s-tempo");
        holder.title = this.track ? TRACK_FIXED : KEY_TOOLTIP;
        this.keyList = document.createElement("select");
        this.keyList.addEventListener("change", () => {
            const shift = Number(this.keyList.value);
            this.rollBox.focus();
            if (shift) this.changeKey(shift);
        });
        holder.append(element("span", "", "Key"), this.keyList);
        this.showKey();
        return holder;
    }

    showKey(force = false) {
        if (!this.keyList) return;
        const key = this.sheet?.bars[0]?.key || "";
        const choices = this.sheet && this.model ? roll.keyChoices(key, roll.voiceMiddle(this.model)) : [];
        const shown = JSON.stringify(choices);
        this.keyList.disabled = !choices.length || Boolean(this.track) || this.keyBusy;
        if (!force && shown === this.shownKeys) {
            this.keyList.value = "0";
            return;
        }
        this.shownKeys = shown;
        this.keyList.replaceChildren();
        for (const choice of choices.length ? choices : [{ name: key || "-", shift: 0 }]) {
            const option = document.createElement("option");
            option.value = String(choice.shift);
            option.textContent = choice.shift ? choice.name + "  " + roll.keyMove(choice.shift) : choice.name;
            option.selected = choice.shift === 0;
            this.keyList.appendChild(option);
        }
    }

    async changeKey(shift) {
        if (!this.sheet || this.track || this.keyBusy) return;
        if (this.chordInput) this.chordInput.finish(true);
        if (this.sectionInput) this.sectionInput.finish(true);
        if (this.writePending) await this.write();
        if (this.isClosed || !this.sheet) return;
        this.keyBusy = true;
        this.showKey();
        this.setStatus("Moving the song " + roll.shiftWords(shift) + "\u2026");
        const sequence = ++this.writeSequence;
        const { ok, payload } = await ask(TRANSPOSE_ROUTE, { abc: this.current, semitones: shift });
        this.keyBusy = false;
        if (this.isClosed) return;
        if (sequence !== this.writeSequence) {
            this.showKey();
            this.setStatus("The key stayed as it was: the notes changed while the song was being moved. "
                + "Pick the key again.", true);
            return;
        }
        if (!ok || !payload.ok) {
            this.showKey();
            this.setStatus(payload.error || "The song could not be moved to another key.", true);
            return;
        }
        this.stopPlaying();
        this.dropped = this.history.push(this.snapshot());
        this.drawn = null;
        const total = (this.moved?.shift || 0) + shift;
        this.moved = total ? { from: this.moved?.from || payload.before, shift: total } : null;
        this.base = payload.abc;
        this.current = payload.abc;
        this.sheet = payload.sheet;
        this.model = roll.modelOf(payload.sheet);
        this.good = this.model;
        this.changed = [];
        this.localChanged = [];
        this.clearPicks();
        this.notesDrawn = null;
        if (document.activeElement !== this.textBox) this.textBox.value = payload.abc;
        this.shownTempo = null;
        this.paintFacts();
        this.showKey();
        this.notice = this.voiceWarning() || null;
        this.showDescription();
        this.draw();
        if (this.tab === "notes") this.drawNotes();
    }

    snapshot() {
        return { keyed: true, model: this.model, base: this.base, current: this.current, sheet: this.sheet,
                 good: this.good, changed: this.changed, moved: this.moved };
    }

    restore(entry) {
        if (!entry?.keyed) {
            this.model = entry;
            return;
        }
        this.base = entry.base;
        this.current = entry.current;
        this.sheet = entry.sheet;
        this.good = entry.good;
        this.changed = entry.changed;
        this.moved = entry.moved;
        this.model = entry.model;
        this.notesDrawn = null;
        if (document.activeElement !== this.textBox) this.textBox.value = entry.current;
        this.shownTempo = null;
        this.paintFacts();
    }

    paintFacts() {
        const sheet = this.sheet;
        const key = sheet.bars[0]?.key || "";
        const meter = sheet.bars[0]?.meter || "";
        this.facts.replaceChildren([key && "Key " + key, meter, sheet.bpm + " BPM",
            sheet.bars.length + " bars", roll.clock(sheet.seconds)].filter(Boolean).join(" \u00B7 "));
        if (this.cutTick() === null) return;
        const sung = element("span", "yue2-s-warn", " \u00B7 sung up to " + roll.clock(this.limit.seconds));
        sung.title = this.limit.auto ? LIMIT_AUTO_TOOLTIP : LIMIT_TOOLTIP;
        this.facts.appendChild(sung);
    }

    cutTick() {
        if (!this.sheet || !this.limit) return null;
        const tick = roll.limitTick(this.sheet, this.limit.seconds);
        return tick < this.sheet.total ? tick : null;
    }

    lateBars(bars) {
        if (this.cutTick() === null) return [];
        const late = new Set(roll.barsAfter(this.sheet, this.limit.seconds));
        return bars.filter((bar) => late.has(bar));
    }

    describe() {
        if (!this.sheet) return "";
        const retimed = this.model && this.model.bpm !== this.sheetTempo
            ? "The score is now " + this.model.bpm + " BPM, where it came in at " + this.sheetTempo
                + ": the notes keep their lengths, so the whole song is sung "
                + (this.model.bpm > this.sheetTempo ? "faster" : "slower") + ". "
            : "";
        if (this.track) {
            if (!this.changed.length) return "No changes. These are the notes the song is sung to.";
            const places = stretchesOf(this.changed);
            return (this.changed.length === 1 ? "Bar " : "Bars ") + barList(this.changed)
                + " will be sung again under the new notes"
                + (places.length > 1 ? ", as " + places.length + " edits, one a place" : "")
                + "; the rest of the song stays as it was sung.";
        }
        const keys = new Set(this.sheet.bars.map((bar) => bar.key));
        const keyed = this.moved
            ? "The song is moved " + roll.shiftWords(this.moved.shift) + ", from " + this.moved.from + " to "
                + (this.sheet.bars[0]?.key || "") + ": both parts, every chord"
                + (keys.size > 1 ? " and every key change" : "") + ", so every bar is written again. "
            : "";
        const before = retimed + keyed;
        if (!this.changed.length) {
            return before + (before ? "Nothing else changed." : "No changes. This is the score as it came in.")
                + (this.cutTick() === null ? ""
                : " Only its first " + roll.clock(this.limit.seconds) + " is sung: " + limitReason(this.limit)
                    + " ends the song at the dashed line.");
        }
        const late = this.lateBars(this.changed);
        return before
            + (this.changed.length === 1 ? "Bar " : "Bars ") + barList(this.changed)
            + (keyed ? (this.changed.length === 1 ? " is" : " are") + " changed on top of that."
                : " will be written again; every other bar stays exactly as it was.")
            + (late.length ? " Bars " + barList(late) + " come after " + roll.clock(this.limit.seconds) + ", where "
                + limitReason(this.limit) + " ends the song: they will not be heard " + limitRemedy(this.limit) + "."
                : "");
    }

    showDescription() {
        const text = this.describe() + (this.sheet?.cut && this.edited()
            ? " The unfinished end the model left is not written back." : "");
        this.setStatus(this.notice ? this.notice + " " + text : text, this.lateBars(this.changed).length > 0);
    }

    fillEmpty(problem) {
        this.empty.replaceChildren();
        if (problem) {
            this.empty.appendChild(element("div", "", "The piano roll cannot draw this score. The ABC tab has its text."));
            return;
        }
        if (this.source) {
            this.empty.appendChild(element("div", "", this.source.empty));
        } else if (this.own) {
            this.empty.appendChild(element("div", "",
                "There is no score yet. Run the workflow once: the score this node writes appears here, "
                + "and every edit after that costs only the singing. Or paste a score into the ABC tab."));
        } else if (this.origin) {
            this.empty.appendChild(element("div", "",
                "There is no score yet. '" + (this.origin.title || "YuE2 Plan") + "' writes one: "
                + "the button runs only that node, so nothing is sung, and it takes seconds once the "
                + "model is loaded. Or paste a score into the ABC tab."));
            const button = element("button", "yue2-s-go", "Write the score");
            button.addEventListener("click", () => this.writeScore());
            this.empty.appendChild(button);
        } else {
            this.empty.appendChild(element("div", "",
                "Connect a 'YuE2 Plan' or 'YuE2 Select Plan' to this node's plan input to edit the "
                + "score it writes, or paste a score into the ABC tab."));
        }
        this.empty.appendChild(element("div", "yue2-s-or",
            "Or start from nothing: an empty grid of " + NEW_BARS + " bars to draw on, which the "
            + "node sings in place of anything else."));
        const blank = element("button", "", "Make an empty score");
        blank.addEventListener("click", () => this.newScore());
        this.empty.appendChild(blank);
    }

    refresh() {
        this.showTempo();
        this.undoButton.disabled = !this.history.canUndo;
        this.redoButton.disabled = !this.history.canRedo;
        this.writeButton.hidden = !this.origin || this.own;
        this.resetButton.disabled = !this.planScore && !this.fromBox;
        this.saveButton.disabled = !(this.current || "").trim();
    }

    showTab(id) {
        this.tab = id;
        for (const [key, button] of Object.entries(this.tabButtons)) {
            button.classList.toggle("yue2-s-on", key === id);
        }
        const drawn = Boolean(this.sheet);
        this.rollBox.hidden = id !== "roll" || !drawn;
        this.scroller.hidden = id !== "roll" || !drawn;
        this.rollTools.hidden = id !== "roll" || !drawn;
        this.paper.hidden = id !== "notes" || !drawn;
        this.textBox.hidden = id !== "abc";
        this.empty.hidden = drawn || id === "abc";
        if (id === "roll" && drawn) {
            requestAnimationFrame(() => {
                this.resize();
                this.rollBox.focus();
            });
        }
        if (id === "notes" && drawn) this.drawNotes();
    }

    fillSnap(sheet, prefer) {
        const choices = roll.snapChoices(sheet.per_quarter);
        const preferred = choices.find((choice) => choice.name === prefer) || choices[0];
        this.snap = preferred ? preferred.ticks : 1;
        const options = choices.map((choice) => [String(choice.ticks), choice.name]);
        if (sheet.unit < roll.FINEST && !this.track) options.push([FINER, FINER_LABEL]);
        this.snapHolder.replaceChildren(selectControl(options, String(this.snap),
            "The grid notes snap to while drawing, moving and stretching.", (value) => {
                if (value === FINER) {
                    this.finerGrid();
                    return;
                }
                this.snap = Number(value);
                this.draw();
            }));
    }

    fillBars() {
        if (this.track) {
            this.barsHolder.replaceChildren();
            return;
        }
        const options = [[LONGER, LONGER_LABEL]].concat(
            ADD_BARS.map((count) => [String(count), "+ " + count + (count === 1 ? " bar" : " bars")]));
        this.barsHolder.replaceChildren(selectControl(options, LONGER,
            "Add empty bars to the end of the song. The score is written again, so the "
            + "undo history starts over.", (value) => {
                if (value !== LONGER) this.addBars(Number(value));
            }));
    }

    async addBars(count) {
        if (!this.sheet) return;
        if (this.writePending) await this.write();
        if (this.isClosed || !this.sheet) return;
        await this.relength(this.current, this.sheet.bars.length + count,
            count + (count === 1 ? " bar" : " bars") + " added at the end. The undo history starts "
            + "here, and the new bars are silent until something is drawn in them.");
        this.fillBars();
    }

    async newScore() {
        await this.relength("", NEW_BARS,
            "An empty score of " + NEW_BARS + " bars at " + NEW_BPM + " BPM. Draw notes in the roll, "
            + "or write it out in the ABC tab. 'Add bars' at the top makes it longer.");
    }

    async relength(abc, bars, note) {
        const sequence = ++this.writeSequence;
        const moved = abc ? this.moved : null;
        this.setStatus("Writing the score\u2026");
        const { ok, payload } = await ask(LENGTH_ROUTE, { abc, bars, bpm: NEW_BPM });
        if (this.isClosed || sequence !== this.writeSequence) return;
        if (!ok || !payload.ok) {
            this.setStatus(payload.error || "The score could not be written.", true);
            this.fillBars();
            return;
        }
        this.pendingNote = note;
        this.pendingBad = false;
        await this.load(payload.abc);
        if (!this.isClosed && this.sheet) this.moved = moved;
    }

    async finerGrid() {
        if (!this.sheet || this.sheet.unit >= roll.FINEST) return;
        const factor = roll.FINEST / this.sheet.unit;
        const zoom = this.pxPerTick / factor;
        const at = this.tick0 * factor;
        this.setStatus("Writing the score again on thirty-second notes\u2026");
        const asked = roll.scaledModel(this.model, factor, roll.FINEST);
        const sequence = ++this.writeSequence;
        const { ok, payload } = await ask(WRITE_ROUTE, { abc: this.base, sheet: roll.sheetOf(asked) });
        if (this.isClosed || sequence !== this.writeSequence) return;
        if (!ok || !payload.ok) {
            this.fillSnap(this.sheet, "Eighth notes");
            this.setStatus(payload.error || "The score could not be written on a finer grid.", true);
            return;
        }
        this.base = payload.abc;
        this.current = payload.abc;
        this.changed = [];
        this.history = new roll.History();
        this.notesDrawn = null;
        if (document.activeElement !== this.textBox) this.textBox.value = payload.abc;
        this.adopt(payload.sheet, false, "Thirty-second notes");
        this.pxPerTick = zoom;
        this.tick0 = at;
        this.clampView();
        this.notice = FINER_NOTICE;
        this.showDescription();
        this.draw();
    }

    fitView() {
        const span = roll.pitchSpan(this.model);
        const rows = Math.max(8, Math.floor(((this.height || 520) - RULER_H - CHORD_H) / ROW_H));
        const middle = Math.round((span.low + span.high) / 2);
        this.pitchTop = clamp(Math.max(span.high, middle + Math.floor(rows / 2)), roll.LOWEST + rows - 1, roll.HIGHEST);
        const shown = Math.min(this.sheet.bars.length, 8);
        const last = this.sheet.bars[Math.max(0, shown - 1)];
        const ticks = last ? last.start + last.length : this.sheet.total;
        this.pxPerTick = Math.max(0.05, ((this.width || 1100) - KEYS_W) / Math.max(1, ticks));
        this.tick0 = 0;
        const focus = this.track?.focus;
        const count = this.sheet.bars.length;
        if (focus && Number.isInteger(focus.first) && focus.first < count) {
            const first = Math.max(0, focus.first - 1);
            const stop = Math.min(count, Math.max(focus.stop + 1, first + 8));
            const end = this.sheet.bars[stop - 1];
            const from = this.sheet.bars[first].start;
            this.pxPerTick = Math.max(0.05, ((this.width || 1100) - KEYS_W)
                / Math.max(1, end.start + end.length - from));
            this.tick0 = from;
            this.playhead = this.sheet.bars[focus.first].start;
        }
        this.clampView();
    }

    visibleTicks() {
        return Math.max(1, (this.width - KEYS_W) / this.pxPerTick);
    }

    visibleRows() {
        return Math.max(1, Math.floor((this.height - RULER_H - CHORD_H) / ROW_H));
    }

    clampView() {
        if (!this.sheet) return;
        const room = Math.max(0, this.sheet.total - this.visibleTicks());
        this.tick0 = clamp(this.tick0, 0, room);
        this.pitchTop = clamp(this.pitchTop, roll.LOWEST + this.visibleRows() - 1, roll.HIGHEST);
        this.scroller.max = String(Math.ceil(room));
        this.scroller.value = String(Math.round(this.tick0));
        this.scroller.disabled = room <= 0;
    }

    resize() {
        if (this.rollBox.hidden) return;
        const room = { width: this.rollBox.clientWidth, height: this.rollBox.clientHeight };
        if (!room.width || !room.height) return;
        const firstTime = !this.width;
        this.ratio = window.devicePixelRatio || 1;
        this.canvas.width = Math.floor(room.width * this.ratio);
        this.canvas.height = Math.floor(room.height * this.ratio);
        this.width = this.canvas.width / this.ratio;
        this.height = this.canvas.height / this.ratio;
        this.canvas.style.width = this.width + "px";
        this.canvas.style.height = this.height + "px";
        if (firstTime && this.sheet) this.fitView();
        this.clampView();
        this.draw();
    }

    zoom(factor, anchorPx) {
        if (!this.sheet) return;
        const px = anchorPx ?? (KEYS_W + (this.width - KEYS_W) / 2);
        const anchor = this.tickAt(px);
        this.pxPerTick = clamp(this.pxPerTick * factor, 0.05, 60);
        this.tick0 = anchor - (px - KEYS_W) / this.pxPerTick;
        this.clampView();
        this.draw();
    }

    x(tick) {
        return KEYS_W + (tick - this.tick0) * this.pxPerTick;
    }

    tickAt(px) {
        return this.tick0 + (px - KEYS_W) / this.pxPerTick;
    }

    y(pitch) {
        return RULER_H + CHORD_H + (this.pitchTop - pitch) * ROW_H;
    }

    pitchAt(py) {
        return this.pitchTop - Math.floor((py - RULER_H - CHORD_H) / ROW_H);
    }

    shownModel() {
        return this.working || this.model;
    }

    draw() {
        this.showTempo();
        const picked = this.selection.size > 0 || (this.chordsPickable() && this.chordPicks.size > 0);
        for (const button of this.pitchButtons) button.disabled = !picked;
        for (const id of ["copy", "cut", "duplicate"]) this.clipButtons[id].disabled = !picked || !this.sheet;
        this.clipButtons.paste.disabled = !CLIP || !this.sheet;
        if (!this.sheet || this.rollBox.hidden || !this.width) return;
        const sheet = this.sheet;
        const model = this.shownModel();
        const c = this.canvas.getContext("2d");
        const W = this.width;
        const H = this.height;
        const top = RULER_H + CHORD_H;
        const px = this.pxPerTick;
        const first = this.tick0;
        const lastTick = this.tickAt(W);
        c.setTransform(this.ratio, 0, 0, this.ratio, 0, 0);
        c.font = "10px system-ui, sans-serif";
        c.textBaseline = "alphabetic";
        c.fillStyle = SKIN.back;
        c.fillRect(0, 0, W, H);

        c.save();
        c.beginPath();
        c.rect(KEYS_W, top, W - KEYS_W, H - top);
        c.clip();
        const rows = Math.ceil((H - top) / ROW_H) + 1;
        for (let r = 0; r < rows; r++) {
            const pitch = this.pitchTop - r;
            const rowY = top + r * ROW_H;
            c.fillStyle = roll.isBlack(pitch) ? SKIN.rowBlack : SKIN.rowWhite;
            c.fillRect(KEYS_W, rowY, W - KEYS_W, ROW_H);
            if (((pitch % 12) + 12) % 12 === 0) {
                c.fillStyle = SKIN.octave;
                c.fillRect(KEYS_W, rowY + ROW_H - 1, W - KEYS_W, 1);
            }
        }
        const marked = new Set(this.changed.length ? this.changed : this.localChanged);
        const localMarked = new Set(this.localChanged);
        let index = roll.barAt(sheet, Math.max(0, Math.floor(first)));
        for (; index < sheet.bars.length; index++) {
            const bar = sheet.bars[index];
            if (bar.start > lastTick) break;
            const barX = this.x(bar.start);
            const barW = bar.length * px;
            if (marked.has(index) || localMarked.has(index)) {
                c.fillStyle = SKIN.changed;
                c.fillRect(barX, top, barW, H - top);
            }
            if (!bar.editable) {
                c.fillStyle = SKIN.locked;
                for (let stripe = -H; stripe < barW; stripe += 12) {
                    c.fillRect(barX + stripe + (H - top) / 2, top, 3, H - top);
                }
            }
            const den = Number(bar.meter.split("/")[1]) || 4;
            const beat = (sheet.per_quarter * 4) / den;
            if (this.snap * px >= 7) {
                c.fillStyle = SKIN.snap;
                for (let t = bar.start + this.snap; t < bar.start + bar.length; t += this.snap) {
                    c.fillRect(Math.round(this.x(t)), top, 1, H - top);
                }
            }
            c.fillStyle = SKIN.beat;
            for (let t = bar.start + beat; t < bar.start + bar.length; t += beat) {
                c.fillRect(Math.round(this.x(t)), top, 1, H - top);
            }
            c.fillStyle = SKIN.bar;
            c.fillRect(Math.round(barX), top, 2, H - top);
        }
        if (this.chordPart) {
            this.drawPart(c, model, "Ins", true, first, lastTick);
            this.drawPart(c, model, "Vocal", true, first, lastTick);
            this.drawChords(c, model, first, lastTick);
        } else {
            this.drawPart(c, model, this.part === "Vocal" ? "Ins" : "Vocal", !this.both, first, lastTick);
            this.drawPart(c, model, this.part, false, first, lastTick);
        }
        const cut = this.cutTick();
        const cutX = cut === null ? null : this.x(cut);
        if (cutX !== null && cutX < W) {
            const shadeFrom = Math.max(KEYS_W, cutX);
            c.fillStyle = SKIN.limitShade;
            c.fillRect(shadeFrom, top, W - shadeFrom, H - top);
            if (cutX >= KEYS_W) this.dashedLine(c, cutX, top, H);
        }
        const marker = this.playTick ?? this.playhead;
        const markerX = Math.round(this.x(marker)) + 0.5;
        const markerShown = markerX >= KEYS_W && markerX <= W;
        if (markerShown) {
            c.strokeStyle = SKIN.playhead;
            c.lineWidth = 1;
            c.beginPath();
            c.moveTo(markerX, top);
            c.lineTo(markerX, H);
            c.stroke();
        }
        if (this.drag?.mode === "box") {
            const x0 = this.x(this.drag.tick);
            const x1 = this.x(this.drag.toTick);
            const y0 = this.y(Math.max(this.drag.pitch, this.drag.toPitch));
            const y1 = this.y(Math.min(this.drag.pitch, this.drag.toPitch)) + ROW_H;
            c.strokeStyle = SKIN.box;
            c.lineWidth = 1;
            c.setLineDash([4, 3]);
            c.strokeRect(Math.min(x0, x1), y0, Math.abs(x1 - x0), y1 - y0);
            c.setLineDash([]);
        }
        c.restore();

        c.fillStyle = SKIN.ruler;
        c.fillRect(0, 0, W, RULER_H);
        c.fillStyle = SKIN.lane;
        c.fillRect(0, RULER_H, W, CHORD_H);
        c.save();
        c.beginPath();
        c.rect(KEYS_W, 0, W - KEYS_W, top);
        c.clip();
        const labels = [];
        this.sectionBoxes = [];
        for (const section of roll.sectionSpans(model, sheet.bars.length)) {
            const startBar = sheet.bars[section.bar];
            const endBar = sheet.bars[section.bar + section.bars - 1];
            if (!startBar || !endBar) continue;
            const left = this.x(startBar.start);
            const right = this.x(endBar.start + endBar.length);
            if (right < KEYS_W || left > W) continue;
            c.fillStyle = sectionColor(section.name);
            c.globalAlpha = 0.8;
            c.fillRect(left + 1, 1, right - left - 2, SECTION_H - 2);
            c.globalAlpha = 1;
            if (left >= KEYS_W) {
                const held = this.drag?.mode === "section"
                    && (this.drag.bar ?? this.drag.from) === section.bar;
                c.fillStyle = held ? SKIN.playhead : SKIN.sectionEdge;
                c.fillRect(Math.round(left), 0, 2, SECTION_H);
            }
            c.fillStyle = "#fff";
            const labelX = Math.max(left, KEYS_W) + 6;
            c.fillText(section.name, labelX, 12);
            const labelW = c.measureText(section.name).width + 4;
            labels.push([labelX, labelX + labelW]);
            this.sectionBoxes.push({ bar: section.bar, name: section.name,
                                     from: labelX - 2, to: labelX + labelW });
        }
        const every = 28 / (px * (sheet.bars[0]?.length || 1)) > 1 ? 4 : 1;
        index = roll.barAt(sheet, Math.max(0, Math.floor(first)));
        for (; index < sheet.bars.length; index++) {
            const bar = sheet.bars[index];
            if (bar.start > lastTick) break;
            c.fillStyle = SKIN.rulerTick;
            c.fillRect(Math.round(this.x(bar.start)), SECTION_H + 1, 1, RULER_H - SECTION_H - 1);
            if (index % every === 0) {
                c.fillStyle = SKIN.rulerText;
                c.fillText(String(index + 1), this.x(bar.start) + 4, RULER_H - 6);
            }
        }
        c.font = "11px system-ui, sans-serif";
        c.lineWidth = 1;
        const picks = this.shownPicks();
        for (const chip of this.chordChips(c, model)) {
            const chord = chip.chord;
            if (chip.left > W || chip.left + chip.width < KEYS_W) continue;
            const picked = this.chordsPickable() && picks.has(chord.start);
            const chipX = Math.round(chip.left) + 0.5;
            this.box(c, chipX, RULER_H + 4.5, chip.width, CHORD_H - 9,
                picked ? SKIN.picked : SKIN.chip, picked ? SKIN.pickedEdge : SKIN.chipEdge);
            if (chip.width < 12) continue;
            c.save();
            c.beginPath();
            c.rect(chipX, RULER_H, chip.width - 2, CHORD_H);
            c.clip();
            c.fillStyle = picked ? SKIN.pickedText : SKIN.chipText;
            c.fillText(chord.name, chip.left + 5, RULER_H + CHORD_H - 8);
            c.restore();
        }
        c.font = "10px system-ui, sans-serif";
        if (cutX !== null && cutX >= KEYS_W && cutX <= W) {
            this.dashedLine(c, cutX, 0, top);
            const chip = (this.limit.auto ? "lyrics limit " : "max_seconds ") + roll.clock(this.limit.seconds);
            const chipW = c.measureText(chip).width + 10;
            const onRight = Math.round(cutX + 3) + 0.5;
            const onLeft = Math.round(cutX - 3 - chipW) + 0.5;
            const covers = (from) => labels.some(([a, b]) => a < from + chipW && b > from);
            const chipX = onRight + chipW > W || (covers(onRight) && !covers(onLeft) && onLeft >= KEYS_W)
                ? onLeft : onRight;
            this.box(c, chipX, 1.5, chipW, 14, SKIN.limitChip, SKIN.limit);
            c.fillStyle = SKIN.limitText;
            c.fillText(chip, chipX + 5, 12);
        }
        if (markerShown) {
            const head = SECTION_H + 1;
            c.fillStyle = SKIN.playhead;
            c.beginPath();
            c.moveTo(markerX - 5, head);
            c.lineTo(markerX + 5, head);
            c.lineTo(markerX, head + 7);
            c.closePath();
            c.fill();
            c.fillRect(markerX - 0.5, head + 7, 1, top - head - 7);
        }
        c.restore();

        c.fillStyle = SKIN.ruler;
        c.fillRect(0, 0, KEYS_W, top);
        c.fillStyle = SKIN.laneText;
        c.fillText("section", 6, 12);
        c.fillText("bar", 6, RULER_H - 6);
        c.fillText("chord", 6, RULER_H + CHORD_H - 8);
        c.save();
        c.beginPath();
        c.rect(0, top, KEYS_W, H - top);
        c.clip();
        this.drawKeys(c, top, rows);
        c.restore();
        c.fillStyle = SKIN.keysEdge;
        c.fillRect(KEYS_W - 1, 0, 1, H);
        this.undoButton.disabled = !this.history.canUndo;
        this.redoButton.disabled = !this.history.canRedo;
    }

    drawKeys(c, top, rows) {
        if (this.drumRows()) {
            this.drawKit(c, top, rows);
            return;
        }
        const blackW = Math.round(KEYS_W * 0.6);
        c.fillStyle = SKIN.keyWhite;
        c.fillRect(0, top, KEYS_W, rows * ROW_H);
        for (let r = 0; r < rows; r++) {
            const pitch = this.pitchTop - r;
            const rowY = top + r * ROW_H;
            const tone = ((pitch % 12) + 12) % 12;
            const black = roll.isBlack(pitch);
            const under = pitch === this.hoverPitch;
            if (black) {
                c.fillStyle = SKIN.keyEdge;
                c.fillRect(blackW, rowY + ROW_H / 2, KEYS_W - blackW, 1);
                c.fillStyle = under ? SKIN.keyBlackUnder : SKIN.keyBlack;
                c.fillRect(0, rowY + 1, blackW, ROW_H - 2);
                c.fillStyle = SKIN.keyBlackTop;
                c.fillRect(blackW - 4, rowY + 3, 2, ROW_H - 6);
            } else {
                if (under) {
                    c.fillStyle = SKIN.keyUnder;
                    c.fillRect(0, rowY + 1, KEYS_W, ROW_H - 2);
                }
                if (tone === 0 || tone === 5) {
                    c.fillStyle = SKIN.keyEdge;
                    c.fillRect(0, rowY + ROW_H - 1, KEYS_W, 1);
                }
            }
            if (tone !== 0 && !under) continue;
            c.font = (tone === 0 ? "600 " : "") + "10px system-ui, sans-serif";
            c.fillStyle = black ? SKIN.keyBlackText : tone === 0 ? SKIN.keyTextC : SKIN.keyText;
            c.fillText(roll.noteName(pitch), black ? 5 : KEYS_W - 24, rowY + ROW_H - 3);
        }
    }

    drawKit(c, top, rows) {
        c.fillStyle = SKIN.keyWhite;
        c.fillRect(0, top, KEYS_W, rows * ROW_H);
        c.font = "9px system-ui, sans-serif";
        for (let r = 0; r < rows; r++) {
            const pitch = this.pitchTop - r;
            const rowY = top + r * ROW_H;
            if (pitch === this.hoverPitch) {
                c.fillStyle = SKIN.keyUnder;
                c.fillRect(0, rowY + 1, KEYS_W, ROW_H - 2);
            }
            c.fillStyle = SKIN.keyEdge;
            c.fillRect(0, rowY + ROW_H - 1, KEYS_W, 1);
            c.fillStyle = ((pitch % 12) + 12) % 12 === 0 ? SKIN.keyTextC : SKIN.keyText;
            c.fillText(sounds.drumName(pitch), 4, rowY + ROW_H - 3);
        }
    }

    drawPart(c, model, part, ghost, first, lastTick) {
        const px = this.pxPerTick;
        const top = RULER_H + CHORD_H;
        const blue = this.both && part === "Ins";
        const fill = blue ? SKIN.insNote : SKIN.note;
        const edge = blue ? SKIN.insEdge : SKIN.noteEdge;
        const text = blue ? SKIN.insText : SKIN.noteText;
        c.save();
        c.font = "600 9px system-ui, sans-serif";
        c.lineWidth = 1;
        for (const note of model.notes[part]) {
            if (note.start + note.length < first || note.start > lastTick) continue;
            const noteY = this.y(note.pitch);
            if (noteY + ROW_H < top || noteY > this.height) continue;
            const noteX = Math.round(this.x(note.start)) + 0.5;
            const width = Math.max(3, Math.round(note.length * px) - 1);
            if (ghost) {
                this.box(c, noteX, noteY + 1.5, width, ROW_H - 3, SKIN.ghost, SKIN.ghostEdge);
                continue;
            }
            const picked = this.selection.has(note.id);
            this.box(c, noteX, noteY + 1.5, width, ROW_H - 3, picked ? SKIN.picked : fill,
                picked ? SKIN.pickedEdge : edge);
            if (width > 5) {
                c.fillStyle = SKIN.shine;
                c.fillRect(noteX + 1.5, noteY + 2, width - 4, 1);
            }
            const bar = this.sheet.bars[roll.barAt(this.sheet, note.start)];
            const label = part === "Ins" && this.soundOf("Ins") === sounds.DRUMS
                ? sounds.drumName(note.pitch)
                : roll.noteName(note.pitch, roll.flatsIn(this.sheet, bar.key));
            if (c.measureText(label).width + 6 <= width) {
                c.fillStyle = picked ? SKIN.pickedText : text;
                c.fillText(label, noteX + 3, noteY + ROW_H - 4);
            }
        }
        c.restore();
    }

    chordChips(c, model) {
        c.font = "11px system-ui, sans-serif";
        return roll.chordSpans(model, this.sheet.total).map((span) => {
            const left = this.x(span.start);
            const room = this.x(span.end) - left - 2;
            const width = Math.max(4, Math.min(c.measureText(span.name).width + 10, room));
            return { chord: span, left, width };
        });
    }

    drawChords(c, model, first, lastTick) {
        const top = RULER_H + CHORD_H;
        const drag = this.drag;
        const picks = this.shownPicks();
        c.save();
        c.font = "600 9px system-ui, sans-serif";
        c.lineWidth = 1;
        for (const span of roll.chordSpans(model, this.sheet.total)) {
            if (span.end < first || span.start > lastTick) continue;
            const toned = drag?.mode === "tone" && drag.start === span.start;
            const pitches = toned ? drag.pitches : roll.chordPitches(span.name);
            const root = toned ? null : roll.rootPitch(span.name);
            const picked = picks.has(span.start);
            const refused = toned && !drag.named;
            const noteX = Math.round(this.x(span.start)) + 0.5;
            const width = Math.max(3, Math.round((span.end - span.start) * this.pxPerTick) - 1);
            const bar = this.sheet.bars[roll.barAt(this.sheet, span.start)];
            for (const pitch of pitches) {
                const noteY = this.y(pitch);
                if (noteY + ROW_H < top || noteY > this.height) continue;
                const fill = refused ? SKIN.refused : picked ? SKIN.picked : pitch === root ? SKIN.chordRoot : SKIN.chordNote;
                const edge = refused ? SKIN.refusedEdge : picked ? SKIN.pickedEdge : SKIN.chordEdge;
                this.box(c, noteX, noteY + 1.5, width, ROW_H - 3, fill, edge);
                const label = pitch === root ? span.name
                    : toned && pitch === drag.toPitch ? (drag.named || "?")
                    : roll.noteName(pitch, roll.flatsIn(this.sheet, bar.key));
                if (c.measureText(label).width + 6 <= width) {
                    c.fillStyle = picked ? SKIN.pickedText : SKIN.chordText;
                    c.fillText(label, noteX + 3, noteY + ROW_H - 4);
                }
            }
        }
        c.restore();
    }

    box(c, x, y, w, h, fill, edge) {
        c.beginPath();
        if (c.roundRect) c.roundRect(x, y, w, h, NOTE_RADIUS);
        else c.rect(x, y, w, h);
        c.fillStyle = fill;
        c.fill();
        c.strokeStyle = edge;
        c.stroke();
    }

    dashedLine(c, x, from, to) {
        c.save();
        c.strokeStyle = SKIN.limit;
        c.lineWidth = 2;
        c.setLineDash([6, 4]);
        c.beginPath();
        c.moveTo(Math.round(x), from);
        c.lineTo(Math.round(x), to);
        c.stroke();
        c.restore();
    }

    local(event) {
        const box = this.canvas.getBoundingClientRect();
        return { px: event.clientX - box.left, py: event.clientY - box.top };
    }

    hitNote(px, py) {
        const pitch = this.pitchAt(py);
        const tick = this.tickAt(px);
        for (const part of this.editedParts()) {
            const found = this.model.notes[part].find((note) => note.pitch === pitch && note.start <= tick
                && tick < note.start + note.length + EDGE_PX / this.pxPerTick);
            if (found) return found;
        }
        return null;
    }

    barLocked(tick) {
        const bar = this.sheet.bars[roll.barAt(this.sheet, Math.max(0, Math.floor(tick)))];
        if (bar && !bar.editable) {
            this.setStatus("Bar " + (roll.barAt(this.sheet, Math.floor(tick)) + 1) + " changes key halfway "
                + "through, and the roll leaves it as it is. Edit it in the ABC tab.", true);
            return true;
        }
        return false;
    }

    blip(pitch, part = this.part) {
        try {
            this.player.blip(pitch, part);
        } catch (error) {
            void error;
        }
    }

    pointerDown(event) {
        if (!this.sheet) return;
        this.rollBox.focus();
        const keys = roll.modifiersOf(event);
        const { px, py } = this.local(event);
        const top = RULER_H + CHORD_H;
        const tick = this.tickAt(px);
        const at = now();
        const again = event.button === 0 && !keys.alt && !keys.ctrl && !keys.shift
            && roll.isDoubleClick(this.lastDown, at, px, py);
        this.lastDown = { at, px, py };
        if (px < KEYS_W) {
            if (py >= top) this.blip(this.pitchAt(py));
            return;
        }
        if (py < SECTION_H) {
            this.sectionDown(event, px, tick);
            return;
        }
        if (py < RULER_H) {
            if (again) this.togglePlay();
            else this.movePlayhead(tick);
            return;
        }
        if (py < top) {
            if (event.button === 2) this.dropChord(px);
            else if (this.both) this.pickChord(px, keys);
            else this.chordAt(tick, px);
            return;
        }
        if (this.chordPart) {
            this.chordDown(event, px, py, tick, keys, again);
            return;
        }
        if (again) {
            this.takeBackDraw();
            this.togglePlay();
            return;
        }
        const pitch = this.pitchAt(py);
        const hit = this.hitNote(px, py);
        if (hit) {
            if (event.button === 2) {
                const all = this.selection.has(hit.id);
                const ids = all ? [...this.selection] : [hit.id];
                const starts = all && this.both ? [...this.chordPicks] : [];
                this.clearPicks();
                this.commit(this.both ? roll.deleteTogether(this.model, ids, starts)
                    : roll.deleteNotes(this.model, this.part, ids));
                return;
            }
            const edge = !this.both && this.x(hit.start + hit.length) - px <= EDGE_PX;
            if ((keys.shift || keys.ctrl) && !edge) {
                if (this.selection.has(hit.id)) this.selection.delete(hit.id);
                else this.selection.add(hit.id);
                this.draw();
                return;
            }
            if (this.barLocked(hit.start)) return;
            if (!this.selection.has(hit.id)) {
                this.selection = new Set([hit.id]);
                this.chordPicks.clear();
            }
            this.canvas.setPointerCapture(event.pointerId);
            this.blip(hit.pitch, roll.partOf(this.model, hit.id) || this.part);
            this.drag = edge
                ? { mode: "resize", id: hit.id, from: this.model, start: hit.start, length: hit.length }
                : { mode: "move", ids: [...this.selection], chords: this.pickedChords(),
                    from: this.model, base: this.model, tick, pitch, sounded: hit.pitch, ticks: 0 };
            this.draw();
            return;
        }
        if (event.button !== 0) return;
        this.canvas.setPointerCapture(event.pointerId);
        if (keys.shift || keys.ctrl || this.both) {
            if (!keys.shift && !keys.ctrl) this.clearPicks();
            this.drag = { mode: "box", tick, pitch, toTick: tick, toPitch: pitch, kept: new Set(this.selection),
                          keptChords: new Set(this.chordPicks) };
            this.draw();
            return;
        }
        if (this.barLocked(tick)) return;
        const start = keys.alt ? Math.floor(tick) : roll.snapDown(tick, this.snap);
        const under = this.model.notes[this.part].find((note) => note.start <= start && start < note.start + note.length);
        if (under) {
            this.offerChord(under, start, pitch);
            return;
        }
        const room = roll.roomAt(this.model, this.part, start, this.sheet.total);
        const placed = room
            ? roll.addNote(this.model, this.part, start, Math.min(this.lastLength || this.snap, room), pitch, this.sheet.total)
            : null;
        if (!placed) {
            this.setStatus("The song ends before that point.", true);
            return;
        }
        this.selection = new Set([placed.id]);
        this.blip(pitch);
        this.working = placed.model;
        this.drag = { mode: "move", drawn: true, ids: [placed.id], from: this.model, base: placed.model,
                      tick: start, pitch, sounded: pitch };
        this.draw();
    }

    takeBackDraw() {
        if (!this.drawn || now() - this.drawn.at > roll.DOUBLE_CLICK_MS) return;
        const redo = this.drawn.redo;
        this.drawn = null;
        const before = this.history.take(redo);
        if (!before) return;
        this.notice = null;
        this.model = before;
        this.clearPicks();
        this.localChanged = roll.changedBars(this.sheet, this.model);
        this.showKey();
        this.draw();
        this.scheduleWrite();
    }

    hitChord(px, py) {
        const pitch = this.pitchAt(py);
        const tick = this.tickAt(px);
        const total = this.sheet.total;
        const reach = EDGE_PX / this.pxPerTick;
        for (const span of roll.chordSpans(this.model, total)) {
            if (tick < span.start || tick >= span.end + (span.end < total ? reach : 0)) continue;
            const pitches = roll.chordPitches(span.name);
            if (!pitches.includes(pitch)) continue;
            const edge = span.end < total && Math.abs(this.x(span.end) - px) <= EDGE_PX;
            return { span, pitch, pitches, root: pitch === roll.rootPitch(span.name), edge };
        }
        return null;
    }

    blipChord(pitches) {
        for (const pitch of pitches) this.blip(pitch, "chords");
    }

    shownPicks() {
        const drag = this.drag;
        if (!drag || !this.working || (drag.mode !== "move" && drag.mode !== "chord")) return this.chordPicks;
        const starts = drag.mode === "chord" ? drag.starts : [...this.chordPicks];
        return new Set(starts.map((start) => start + (drag.ticks || 0)));
    }

    chordDown(event, px, py, tick, keys, again) {
        const hit = this.hitChord(px, py);
        if (event.button === 2) {
            if (hit) this.removeChordAt(hit.span);
            else this.setStatus("Right-click a chord's notes to remove it; click an empty place for a new one.");
            return;
        }
        if (again) {
            if (this.drawn && now() - this.drawn.at <= roll.DOUBLE_CLICK_MS) {
                this.takeBackDraw();
                this.togglePlay();
            } else if (hit) {
                this.openChordInput(hit.span.start, hit.span.name);
            } else {
                this.togglePlay();
            }
            return;
        }
        if (event.button !== 0) return;
        const pitch = this.pitchAt(py);
        if (keys.ctrl && !keys.shift) {
            this.toggleTone(tick, pitch);
            return;
        }
        if (keys.shift || keys.ctrl) {
            if (hit) {
                if (this.chordPicks.has(hit.span.start)) this.chordPicks.delete(hit.span.start);
                else this.chordPicks.add(hit.span.start);
                this.draw();
                return;
            }
            this.canvas.setPointerCapture(event.pointerId);
            this.drag = { mode: "box", tick, pitch, toTick: tick, toPitch: pitch, kept: new Set(),
                          keptChords: new Set(this.chordPicks) };
            this.draw();
            return;
        }
        if (hit) {
            if (this.barLocked(hit.span.start)) return;
            const many = hit.root && !hit.edge && this.chordPicks.has(hit.span.start) && this.chordPicks.size > 1;
            if (!many) this.chordPicks = new Set([hit.span.start]);
            this.canvas.setPointerCapture(event.pointerId);
            this.blipChord(hit.pitches);
            if (hit.edge) {
                this.drag = { mode: "edge", from: this.model, start: hit.span.end, to: hit.span.end };
            } else if (hit.root) {
                this.drag = { mode: "chord", from: this.model, base: this.model, start: hit.span.start,
                              starts: many ? this.pickedChords() : [hit.span.start], tick, pitch, ticks: 0, semitones: 0 };
            } else {
                this.drag = { mode: "tone", from: this.model, start: hit.span.start, name: hit.span.name,
                              index: hit.pitches.indexOf(hit.pitch), pitch: hit.pitch, toPitch: hit.pitch,
                              pitches: hit.pitches, named: hit.span.name };
            }
            this.draw();
            return;
        }
        const start = keys.alt ? Math.floor(tick) : roll.snapDown(tick, this.snap);
        if (start < 0 || start >= this.sheet.total) {
            this.setStatus("The song ends before that point.", true);
            return;
        }
        if (this.barLocked(start)) return;
        const name = roll.chordByDegree(pitch, roll.keyAt(this.sheet, start));
        const placed = roll.setChord(this.model, start, name);
        if (!placed) return;
        this.chordPicks = new Set([start]);
        this.blipChord(roll.chordPitches(name));
        this.canvas.setPointerCapture(event.pointerId);
        this.working = placed;
        this.drag = { mode: "chord", drawn: true, from: this.model, base: placed, start, starts: [start],
                      tick: start, pitch, ticks: 0, semitones: 0 };
        this.draw();
    }

    chordDragged(event, tick, pitch) {
        const drag = this.drag;
        const total = this.sheet.total;
        const alt = roll.modifiersOf(event).alt;
        const step = alt ? 1 : this.snap;
        if (drag.mode === "chord") {
            let ticks = alt ? Math.round(tick - drag.tick) : roll.snapTo(tick - drag.tick, this.snap);
            const semitones = pitch - drag.pitch;
            let moved;
            if (drag.starts.length > 1) {
                moved = roll.moveChords(drag.base, this.sheet, drag.starts, ticks, semitones);
            } else {
                const room = roll.chordRoom(drag.base, drag.start, total);
                const low = drag.start - Math.floor((drag.start - room.low) / step) * step;
                const high = drag.start + Math.floor((room.high - drag.start) / step) * step;
                ticks = clamp(drag.start + ticks, low, high) - drag.start;
                moved = drag.drawn
                    ? roll.setChord(drag.from, drag.start + ticks, roll.chordByDegree(pitch, roll.keyAt(this.sheet, drag.start + ticks)))
                    : roll.moveChord(drag.base, this.sheet, drag.start, ticks, semitones);
            }
            if (!moved) return;
            this.working = moved;
            drag.ticks = ticks;
            if (semitones !== drag.semitones) {
                drag.semitones = semitones;
                const lead = moved.chords.find((c) => c.start === drag.start + ticks);
                if (lead) this.blipChord(roll.chordPitches(lead.name));
            }
        } else if (drag.mode === "edge") {
            const room = roll.chordRoom(drag.from, drag.start, total);
            const low = Math.ceil(room.low / step) * step;
            const high = Math.floor(room.high / step) * step;
            const at = alt ? Math.round(tick) : roll.snapTo(tick, this.snap);
            const to = low <= high ? clamp(at, low, high) : clamp(Math.round(tick), room.low, room.high);
            const moved = roll.moveChord(drag.from, this.sheet, drag.start, to - drag.start, 0);
            if (!moved) return;
            this.working = moved;
            drag.to = to;
        } else if (drag.mode === "tone") {
            if (pitch === drag.toPitch || pitch < roll.LOWEST || pitch > roll.HIGHEST) return;
            const pitches = drag.pitches.slice();
            pitches[drag.index] = pitch;
            drag.pitches = pitches;
            drag.toPitch = pitch;
            drag.named = roll.namedChord(pitches, roll.keyAt(this.sheet, drag.start));
            this.working = drag.named ? roll.setChord(drag.from, drag.start, drag.named) : drag.from;
            this.blipChord(pitches);
        }
    }

    chordReleased(drag, result) {
        if (drag.mode === "tone") {
            if (drag.toPitch === drag.pitch) {
                this.draw();
            } else if (!drag.named) {
                this.setStatus(this.refusal(drag.pitches, drag.name), true);
                this.draw();
            } else if (drag.named === drag.name) {
                this.setStatus(this.toneNames(drag.pitches) + " are still " + drag.name + ".");
                this.draw();
            } else {
                this.commitChords(result, drag.from, this.madeOf(drag.pitches, drag.named));
            }
            return;
        }
        if (!result || result === drag.from) {
            this.draw();
            return;
        }
        const kept = this.commitChords(result, drag.from);
        if (kept && drag.drawn) this.drawn = { at: now(), redo: this.dropped };
        if (kept && drag.mode === "chord") {
            this.chordPicks = new Set(drag.starts.map((start) => start + (drag.ticks || 0)));
            this.draw();
        }
    }

    toneNames(pitches) {
        const flats = roll.flatsIn(this.sheet, this.sheet.bars[0]?.key);
        const names = [...new Set(pitches)].sort((a, b) => a - b).map((pitch) => roll.noteName(pitch, flats));
        return names.length > 1 ? names.slice(0, -1).join(", ") + " and " + names[names.length - 1] : names.join("");
    }

    refusal(pitches, name) {
        return this.toneNames(pitches) + " make no chord the score can name, so it stays " + name + ". " + KINDS_HELP;
    }

    madeOf(pitches, name) {
        return this.toneNames(pitches) + " make " + name + ", laid out the way the MIDI file holds it.";
    }

    toggleTone(tick, pitch) {
        const span = roll.chordSpans(this.model, this.sheet.total).find((one) => one.start <= tick && tick < one.end);
        if (!span) {
            this.setStatus("No chord holds here to add a note to. Click without Ctrl for a new chord.", true);
            return;
        }
        if (this.barLocked(span.start)) return;
        const layout = roll.chordPitches(span.name);
        const pitches = layout.includes(pitch) ? layout.filter((one) => one !== pitch) : [...layout, pitch];
        const named = roll.namedChord(pitches, roll.keyAt(this.sheet, span.start));
        this.chordPicks = new Set([span.start]);
        if (!named) {
            this.setStatus(this.refusal(pitches, span.name), true);
            this.draw();
            return;
        }
        this.blipChord(pitches);
        if (named === span.name) {
            this.setStatus(this.toneNames(pitches) + " are still " + span.name + ".");
            this.draw();
            return;
        }
        this.commitChords(roll.setChord(this.model, span.start, named), this.model, this.madeOf(pitches, named));
    }

    removeChordAt(span) {
        if (this.barLocked(span.start)) return;
        this.chordPicks.delete(span.start);
        this.closeChordInput();
        this.commitChords(roll.removeChord(this.model, span.start), this.model,
            span.name + " is gone from bar " + (roll.barAt(this.sheet, span.start) + 1) + ". Ctrl+Z brings it back.");
    }

    shiftChords(starts, ticks, semitones) {
        if (!ticks && semitones % 12 === 0) {
            this.setStatus("A chord has no octave here: the roll lays every chord out from C3, as the MIDI file holds it.");
            return;
        }
        const moved = roll.moveChords(this.model, this.sheet, starts, ticks, semitones);
        if (!moved) {
            this.setStatus("That move does not fit: a chord would land where another one starts, or leave the song.", true);
            return;
        }
        if (!this.commitChords(moved)) return;
        this.chordPicks = new Set(starts.map((start) => start + ticks));
        this.draw();
        const lead = moved.chords.find((chord) => chord.start === starts[0] + ticks);
        if (lead && semitones) this.blipChord(roll.chordPitches(lead.name));
    }

    commitChords(next, previous = this.model, said = "") {
        let mirrored = null;
        if (next && next !== previous && this.chordPart && this.repeats.box.checked) {
            mirrored = roll.mirrorChords(this.sheet, previous, next);
            next = mirrored.model;
        }
        const kept = this.commit(next, previous);
        const told = [said, kept && mirrored ? this.mirrorWords(mirrored) : ""].filter(Boolean).join(" ");
        if (kept && told) {
            this.notice = told;
            this.showDescription();
        }
        return kept;
    }

    mirrorWords(mirrored) {
        const where = (list) => list.map((span) => barList(Array.from({ length: span.bars }, (_, i) => span.bar + i))
            + " (" + span.name + ")").join(", ");
        const pieces = [];
        if (mirrored.done.length) pieces.push("Also made in bars " + where(mirrored.done) + ".");
        if (mirrored.skipped.length) {
            pieces.push("Bars " + where(mirrored.skipped) + " have other chords there, so they stay as they were.");
        }
        return pieces.join(" ");
    }

    movePlayhead(tick) {
        const at = clamp(roll.snapDown(tick, this.snap), 0, this.sheet.total);
        const wasPlaying = this.player.playing;
        this.stopPlaying();
        this.playhead = at;
        if (wasPlaying) this.startPlaying();
        this.draw();
    }

    sectionEdgeAt(px) {
        return this.model.sections.find((section) => {
            const bar = this.sheet.bars[section.bar];
            const left = bar ? this.x(bar.start) : null;
            return left !== null && left >= KEYS_W && Math.abs(left - px) <= EDGE_PX;
        }) || null;
    }

    sectionDown(event, px, tick) {
        if (event.button === 2) {
            this.movePlayhead(tick);
            return;
        }
        if (event.button !== 0) return;
        if (this.track) {
            this.setStatus(TRACK_FIXED);
            return;
        }
        const edge = this.sectionEdgeAt(px);
        if (edge) {
            this.canvas.setPointerCapture(event.pointerId);
            this.drag = { mode: "section", from: edge.bar, base: this.model };
            this.draw();
            return;
        }
        const named = (this.sectionBoxes || []).find((box) => px >= box.from && px <= box.to);
        if (named) {
            this.openSectionInput(named.bar, named.name);
            return;
        }
        const bar = roll.barAt(this.sheet, clamp(Math.floor(tick), 0, this.sheet.total - 1));
        const here = roll.sectionStarting(this.model, bar);
        this.openSectionInput(bar, here ? here.name : "");
    }

    openSectionInput(bar, name) {
        this.closeChordInput();
        this.closeSectionInput();
        const input = document.createElement("input");
        input.type = "text";
        input.className = "yue2-s-chord";
        input.value = name;
        input.placeholder = name ? "name \u2014 empty joins it to the one before" : "verse, chorus\u2026";
        input.setAttribute("list", "yue2-s-sections");
        for (const side of ["width", "min-width", "max-width"]) {
            input.style.setProperty(side, CHORD_BOX_W + "px", "important");
        }
        const at = this.sheet.bars[bar];
        input.style.left = clamp(at ? this.x(at.start) : KEYS_W, KEYS_W,
            Math.max(KEYS_W, this.width - CHORD_BOX_W - 10)) + "px";
        input.style.top = "1px";
        const entry = { input, done: false };
        entry.finish = (keep) => {
            if (entry.done) return true;
            const text = input.value.trim();
            const clean = roll.sectionName(text);
            if (keep && text && !clean) {
                input.classList.add("yue2-s-bad");
                this.setStatus(SECTION_HELP, true);
                return false;
            }
            this.closeSectionInput(entry);
            if (keep && !clean && name) this.commit(roll.removeSection(this.model, bar));
            else if (keep && clean && clean !== name) this.commit(roll.setSection(this.model, bar, clean));
            this.showDescription();
            return true;
        };
        input.addEventListener("keydown", (event) => {
            event.stopPropagation();
            if (event.key !== "Enter") return;
            event.preventDefault();
            if (entry.finish(true)) this.rollBox.focus();
        });
        input.addEventListener("blur", () => {
            if (!entry.finish(true)) this.closeSectionInput(entry);
        });
        this.sectionInput = entry;
        this.rollBox.appendChild(input);
        input.focus();
        input.select();
        this.setStatus(name
            ? "Name this section again and press Enter, or empty the box to join it to the one before."
            : "Name a section starting at bar " + (bar + 1) + " and press Enter. Esc leaves the strip as it was.");
    }

    closeSectionInput(entry = this.sectionInput) {
        if (!entry) return;
        entry.done = true;
        if (this.sectionInput === entry) this.sectionInput = null;
        entry.input.remove();
    }

    hover(pitch) {
        if (this.hoverPitch === pitch) return;
        this.hoverPitch = pitch;
        if (!this.drag) this.draw();
    }

    pointerMove(event) {
        if (!this.sheet) return;
        const { px, py } = this.local(event);
        this.hover(py > RULER_H + CHORD_H ? this.pitchAt(py) : null);
        if (!this.drag) {
            const grid = py > RULER_H + CHORD_H && px > KEYS_W;
            const hit = grid && !this.chordPart ? this.hitNote(px, py) : null;
            const chord = grid && this.chordPart ? this.hitChord(px, py) : null;
            let cursor = "crosshair";
            if (py < SECTION_H && px > KEYS_W) cursor = this.sectionEdgeAt(px) ? "ew-resize" : "pointer";
            else if (py <= RULER_H + CHORD_H) cursor = "pointer";
            else if (chord) cursor = chord.edge ? "ew-resize" : chord.root ? "move" : "ns-resize";
            else if (hit) cursor = !this.both && this.x(hit.start + hit.length) - px <= EDGE_PX ? "ew-resize" : "move";
            this.canvas.style.cursor = cursor;
            return;
        }
        const tick = this.tickAt(px);
        const pitch = this.pitchAt(py);
        const total = this.sheet.total;
        if (this.drag.mode === "section") {
            const bar = roll.barAt(this.sheet, clamp(Math.floor(tick), 0, total - 1));
            const moved = roll.moveSection(this.drag.base, this.drag.from, bar);
            if (moved) {
                this.working = moved;
                this.drag.bar = bar;
            }
        } else if (this.drag.mode === "box") {
            this.drag.toTick = tick;
            this.drag.toPitch = pitch;
            const inside = this.editedParts().flatMap((part) =>
                roll.notesIn(this.model, part, this.drag.tick, tick, this.drag.pitch, pitch));
            this.selection = new Set([...this.drag.kept, ...inside]);
            if (this.chordsPickable()) {
                this.chordPicks = new Set([...this.drag.keptChords, ...roll.chordsIn(this.model, this.drag.tick, tick)]);
            }
        } else if (["chord", "edge", "tone"].includes(this.drag.mode)) {
            this.chordDragged(event, tick, pitch);
        } else if (this.drag.mode === "move") {
            const ticks = roll.modifiersOf(event).alt ? Math.round(tick - this.drag.tick)
                : roll.snapTo(tick - this.drag.tick, this.snap);
            const semitones = pitch - this.drag.pitch;
            const moved = this.both
                ? roll.moveTogether(this.drag.base, this.drag.ids, this.drag.chords, ticks, semitones, total)
                : roll.moveNotes(this.drag.base, this.part, this.drag.ids, ticks, semitones, total);
            if (moved) {
                this.working = moved;
                this.drag.ticks = ticks;
                const lead = this.noteById(moved, this.drag.ids[0]);
                if (lead && lead.note.pitch !== this.drag.sounded) {
                    this.drag.sounded = lead.note.pitch;
                    this.blip(lead.note.pitch, lead.part);
                }
            }
        } else if (this.drag.mode === "resize") {
            const end = roll.modifiersOf(event).alt ? Math.round(tick) : roll.snapTo(tick, this.snap);
            const stretched = roll.stretchNote(this.drag.from, this.part, this.drag.id, Math.max(1, end - this.drag.start), total);
            if (stretched) {
                this.working = stretched;
                this.lastLength = stretched.notes[this.part].find((n) => n.id === this.drag.id)?.length || this.lastLength;
            }
        }
        this.draw();
    }

    pointerUp(event) {
        if (!this.drag) return;
        try {
            this.canvas.releasePointerCapture(event.pointerId);
        } catch (error) {
            void error;
        }
        const drag = this.drag;
        const result = this.working;
        this.drag = null;
        this.working = null;
        if (drag.mode === "section") {
            if (result && result !== drag.base) {
                this.commit(result, drag.base);
                const moved = result.sections.find((section) => section.bar === drag.bar);
                this.notice = moved ? moved.name + " starts at bar " + (moved.bar + 1) + " now." : null;
                this.showDescription();
                return;
            }
            this.draw();
            return;
        }
        if (["chord", "edge", "tone"].includes(drag.mode)) {
            this.chordReleased(drag, result);
            return;
        }
        if ((drag.mode === "move" || drag.mode === "resize") && result && result !== drag.from) {
            if (drag.mode === "move") {
                const lead = this.noteById(result, drag.ids[0]);
                if (lead && !this.lastLength) this.lastLength = lead.note.length;
            }
            const kept = this.commit(result, drag.from);
            if (kept && drag.drawn) this.drawn = { at: now(), redo: this.dropped };
            if (kept && drag.chords?.length) {
                this.chordPicks = new Set(drag.chords.map((start) => start + drag.ticks));
                this.draw();
            }
            if (kept && drag.mode === "move" && drag.sounded !== drag.pitch) this.warnVoice();
            return;
        }
        this.draw();
    }

    cancelDrag() {
        this.drag = null;
        this.working = null;
        this.draw();
    }

    wheel(event) {
        if (!this.sheet) return;
        event.preventDefault();
        const { px } = this.local(event);
        if (roll.modifiersOf(event).ctrl) {
            this.zoom(event.deltaY < 0 ? 1.2 : 1 / 1.2, px);
            return;
        }
        if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
            this.tick0 += (event.deltaX || event.deltaY) / this.pxPerTick;
        } else {
            this.pitchTop += event.deltaY > 0 ? -2 : 2;
        }
        this.clampView();
        this.draw();
    }

    key(event) {
        if (event.key === "Alt" || event.key === "AltGraph") {
            event.preventDefault();
            return;
        }
        const tag = event.target?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        const ctrl = roll.modifiersOf(event).ctrl;
        const letter = roll.shortcutLetter(event);
        if (ctrl && letter === "z") {
            event.preventDefault();
            if (event.shiftKey) this.redo();
            else this.undo();
            return;
        }
        if (ctrl && letter === "y") {
            event.preventDefault();
            this.redo();
            return;
        }
        if (event.code === "Space") {
            event.preventDefault();
            this.togglePlay();
            return;
        }
        if (this.tab !== "roll" || !this.sheet) return;
        if (ctrl && letter === "a") {
            event.preventDefault();
            this.selection = new Set(this.editedParts().flatMap((part) => this.model.notes[part].map((n) => n.id)));
            if (this.chordsPickable()) this.chordPicks = new Set(this.model.chords.map((c) => c.start));
            this.draw();
            return;
        }
        if (ctrl && !event.altKey && CLIP_KEYS[letter]) {
            event.preventDefault();
            event.stopPropagation();
            this.clipAction(CLIP_KEYS[letter]);
            return;
        }
        const chords = this.pickedChords();
        if (!this.selection.size && !chords.length) return;
        if (event.key === "Delete" || event.key === "Backspace") {
            event.preventDefault();
            const ids = [...this.selection];
            this.clearPicks();
            const next = this.chordsPickable() ? roll.deleteTogether(this.model, ids, chords)
                : roll.deleteNotes(this.model, this.part, ids);
            if (this.chordPart) this.commitChords(next);
            else this.commit(next);
            return;
        }
        const moves = {
            ArrowUp: [0, event.shiftKey || ctrl ? 12 : 1], ArrowDown: [0, event.shiftKey || ctrl ? -12 : -1],
            ArrowLeft: [-this.snap, 0], ArrowRight: [this.snap, 0],
        };
        if (moves[event.key]) {
            event.preventDefault();
            this.shiftPicked(...moves[event.key]);
        }
    }

    clipAction(id) {
        if (!this.sheet || !this.model || this.keyBusy) return;
        if (id === "paste") this.pasteCopy();
        else if (id === "duplicate") this.duplicatePicked();
        else this.copyPicked(id === "cut");
    }

    pickedClip() {
        const ids = this.chordPart ? [] : this.both ? [...this.selection]
            : [...this.selection].filter((id) => roll.partOf(this.model, id) === this.part);
        const clip = roll.clipOf(this.model, this.sheet, ids, this.pickedChords());
        if (!clip) {
            this.setStatus(this.chordPart
                ? "Select chords first: click one, and Shift+click or drag a box for more."
                : "Select notes first: Ctrl+click them, or drag a box with Shift or Ctrl held.", true);
        }
        return clip;
    }

    clipWords(clip) {
        const notes = Object.values(clip.notes).reduce((sum, list) => sum + list.length, 0);
        const words = [];
        if (notes) words.push(notes + (notes === 1 ? " note" : " notes"));
        if (clip.chords.length) words.push(clip.chords.length + (clip.chords.length === 1 ? " chord" : " chords"));
        return words.join(" and ");
    }

    copyPicked(cut) {
        const clip = this.pickedClip();
        if (!clip) return;
        CLIP = clip;
        const first = roll.barAt(this.sheet, clip.origin);
        const last = roll.barAt(this.sheet, clip.origin + clip.span - 1);
        const said = (cut ? "Cut " : "Copied ") + this.clipWords(clip) + " from bars "
            + barList(Array.from({ length: last - first + 1 }, (_, i) => first + i))
            + ". Paste (Ctrl+V) puts them into the bar the playhead is in; Duplicate (Ctrl+D) puts them right after.";
        if (!cut) {
            this.setStatus(said);
            this.draw();
            return;
        }
        const ids = [...this.selection];
        const chords = this.pickedChords();
        this.clearPicks();
        const next = this.chordsPickable() ? roll.deleteTogether(this.model, ids, chords)
            : roll.deleteNotes(this.model, this.part, ids);
        if (this.chordPart) {
            this.commitChords(next, this.model, said);
        } else if (this.commit(next)) {
            this.notice = said;
            this.showDescription();
        }
    }

    pasteCopy() {
        if (!CLIP) {
            this.setStatus("Nothing is copied yet: select notes and press Copy (Ctrl+C).", true);
            return;
        }
        const total = this.sheet.total;
        const at = this.playhead >= total ? total : this.sheet.bars[roll.barAt(this.sheet, this.playhead)].start;
        this.placeClip(CLIP, at, "Pasted");
    }

    duplicatePicked() {
        const clip = this.pickedClip();
        if (clip) this.placeClip(clip, clip.origin + clip.span, "Duplicated");
    }

    clipRoutes(clip) {
        if (this.chordPart) return { routes: [], withChords: true };
        if (this.both) return { routes: roll.PARTS.map((part) => [part, part]), withChords: true };
        const source = clip.notes[this.part].length ? this.part : roll.PARTS.find((part) => clip.notes[part].length);
        return { routes: source ? [[source, this.part]] : [], withChords: false };
    }

    async placeClip(clip, at, verb) {
        const { routes, withChords } = this.clipRoutes(clip);
        const reach = roll.clipReach(clip, routes, withChords);
        if (!reach) {
            this.setStatus(this.chordPart
                ? "The copy has no chords. Pick Voice, Instrument or Both to paste its notes."
                : "The copy holds only chords. Pick Chords or Both to paste them.", true);
            return;
        }
        let added = 0;
        const need = at + reach.to;
        if (need > this.sheet.total) {
            if (this.track) {
                this.setStatus("The copy would run past the end of the song, and this window cannot add bars.", true);
                return;
            }
            const last = this.sheet.bars[this.sheet.bars.length - 1];
            const count = Math.ceil((need - this.sheet.total) / Math.max(1, last.length));
            const perQuarter = this.sheet.per_quarter;
            await this.addBars(count);
            if (this.isClosed || !this.sheet) return;
            if (this.sheet.total < need || this.sheet.per_quarter !== perQuarter) {
                this.setStatus("The bars the copy needs could not be added.", true);
                return;
            }
            added = count;
        }
        const first = roll.barAt(this.sheet, at + reach.from);
        const last = roll.barAt(this.sheet, at + reach.to - 1);
        const locked = this.sheet.bars.slice(first, last + 1).findIndex((bar) => bar.editable === false);
        if (locked >= 0) {
            this.setStatus("Bar " + (first + locked + 1) + " changes key halfway through, and the roll leaves it as it "
                + "is, so the copy was not put there. Edit it in the ABC tab.", true);
            return;
        }
        const pasted = roll.pasteClip(this.model, this.sheet, clip, at, routes, withChords);
        if (!pasted) {
            this.setStatus("The copy does not fit there.", true);
            return;
        }
        const said = verb + " " + this.clipWords({ notes: { Vocal: pasted.ids, Ins: [] }, chords: pasted.starts })
            + " into bars " + barList(Array.from({ length: last - first + 1 }, (_, i) => first + i)) + "."
            + (added ? " " + added + (added === 1 ? " bar was" : " bars were") + " added at the end for it, so "
                + "the undo history starts before this." : "");
        const kept = this.chordPart ? this.commitChords(pasted.model, this.model, said) : this.commit(pasted.model);
        if (!kept) return;
        if (!this.chordPart) {
            this.notice = said;
            this.showDescription();
        }
        this.selection = new Set(pasted.ids);
        this.chordPicks = new Set(this.chordsPickable() ? pasted.starts : []);
        this.reveal(at + reach.from, at + reach.to);
        this.draw();
    }

    reveal(from, to) {
        const shown = this.visibleTicks();
        if (from >= this.tick0 && to <= this.tick0 + shown) return;
        this.tick0 = Math.max(0, from - shown * 0.1);
        this.clampView();
    }

    shiftPicked(ticks, semitones) {
        const chords = this.pickedChords();
        if (!this.sheet || (!this.selection.size && !chords.length)) return;
        if (this.chordPart) {
            this.shiftChords(chords, ticks, semitones);
            return;
        }
        const moved = this.both
            ? roll.moveTogether(this.model, [...this.selection], chords, ticks, semitones, this.sheet.total)
            : roll.moveNotes(this.model, this.part, [...this.selection], ticks, semitones, this.sheet.total);
        if (!moved) {
            this.setStatus(this.both
                ? "That move does not fit: it would overlap another note or chord, or leave the song."
                : "That move does not fit: it would overlap another note or leave the song.", true);
            return;
        }
        this.commit(moved);
        if (chords.length) {
            this.chordPicks = new Set(chords.map((start) => start + ticks));
            this.draw();
        }
        const lead = this.noteById(moved, [...this.selection][0]);
        if (lead && semitones) this.blip(lead.note.pitch, lead.part);
        if (semitones) this.warnVoice();
    }

    voiceWarning() {
        const middle = roll.voiceMiddle(this.model);
        const [low, high] = roll.VOICE_WINDOW;
        if (middle === null || (middle >= low && middle <= high)) return "";
        return "The voice line now centres on " + roll.noteName(Math.round(middle)) + ", "
            + (middle > high ? "above" : "below") + " the " + roll.noteName(low) + "-" + roll.noteName(high)
            + " where the model's own scores keep it; how YuE2 sings it there was not measured.";
    }

    warnVoice() {
        if (!this.both && this.part !== "Vocal") return;
        const warning = this.voiceWarning();
        if (!warning) return;
        this.notice = warning;
        this.showDescription();
    }

    offerChord(under, start, pitch) {
        const bar = roll.barAt(this.sheet, start);
        const guess = roll.chordGuess(under.pitch, pitch, roll.flatsIn(this.sheet, this.sheet.bars[bar].key));
        const existing = this.model.chords.find((chord) => chord.start === start)?.name || "";
        const why = "A part sings one note at a time, so a chord is not stacked notes here: "
            + "it goes in the chord lane above the grid.";
        if (guess && guess === existing) {
            this.setStatus(why + " " + guess + " is already there.", true);
            return;
        }
        const action = !guess ? "Add a chord" : existing ? "Change " + existing + " to " + guess : "Add the chord " + guess;
        const button = element("button", "", action + " at bar " + (bar + 1));
        button.addEventListener("click", () => (guess ? this.placeChord(start, guess) : this.openChordInput(start, existing)));
        this.status.replaceChildren(why, button);
        this.status.classList.add("yue2-s-bad");
    }

    placeChord(start, name) {
        if (this.barLocked(start) || !this.commit(roll.setChord(this.model, start, name))) return;
        this.notice = name + " is on the chord lane at bar " + (roll.barAt(this.sheet, start) + 1)
            + ": click it there to change or remove it.";
        this.setStatus(this.notice);
        this.rollBox.focus();
    }

    chordUnder(px) {
        const c = this.canvas.getContext("2d");
        c.save();
        const chip = this.chordChips(c, this.model).find((one) => px >= one.left && px <= one.left + one.width);
        c.restore();
        return chip ? { start: chip.chord.start, name: chip.chord.name } : null;
    }

    chordAt(tick, px) {
        if (this.barLocked(tick)) return;
        const found = this.chordUnder(px);
        const start = found ? found.start : clamp(roll.snapDown(tick, this.snap), 0, this.sheet.total - 1);
        this.openChordInput(start, found ? found.name : "");
    }

    pickChord(px, keys) {
        const found = this.chordUnder(px);
        if (!found) {
            this.setStatus(BOTH_CHORDS);
            return;
        }
        if (keys.shift || keys.ctrl) {
            if (this.chordPicks.has(found.start)) this.chordPicks.delete(found.start);
            else this.chordPicks.add(found.start);
        } else {
            this.selection.clear();
            this.chordPicks = new Set([found.start]);
        }
        this.draw();
    }

    dropChord(px) {
        const found = this.chordUnder(px);
        if (!found) {
            this.setStatus(this.both ? BOTH_CHORDS : "Right-click a chord to remove it; click the lane to add one.");
            return;
        }
        if (this.barLocked(found.start)) return;
        this.removeChordAt(found);
    }

    openChordInput(start, name) {
        this.chordInput?.finish(true);
        this.closeChordInput();
        const input = document.createElement("input");
        input.type = "text";
        input.className = "yue2-s-chord";
        input.value = name;
        input.placeholder = "Am7, F/C \u2014 empty removes";
        input.setAttribute("list", "yue2-s-chords");
        for (const side of ["width", "min-width", "max-width"]) input.style.setProperty(side, CHORD_BOX_W + "px", "important");
        input.style.left = clamp(this.x(start), KEYS_W, Math.max(KEYS_W, this.width - CHORD_BOX_W - 10)) + "px";
        input.style.top = RULER_H + 1 + "px";
        const entry = { input, done: false };
        entry.finish = (keep) => {
            if (entry.done) return true;
            const text = input.value.trim();
            if (keep && text && text !== name && !roll.isChord(text)) {
                input.classList.add("yue2-s-bad");
                this.setStatus(CHORD_HELP, true);
                return false;
            }
            this.closeChordInput(entry);
            if (keep && !text && name) this.commitChords(roll.removeChord(this.model, start));
            else if (keep && text && text !== name) this.commitChords(roll.setChord(this.model, start, text));
            this.showDescription();
            return true;
        };
        input.addEventListener("keydown", (event) => {
            event.stopPropagation();
            if (event.key !== "Enter") return;
            event.preventDefault();
            if (entry.finish(true)) this.rollBox.focus();
        });
        input.addEventListener("input", () => {
            input.classList.toggle("yue2-s-bad", Boolean(input.value.trim()) && !roll.isChord(input.value));
        });
        input.addEventListener("blur", () => {
            if (!entry.finish(true)) this.closeChordInput(entry);
        });
        this.chordInput = entry;
        this.rollBox.appendChild(input);
        input.focus();
        input.select();
        this.setStatus("Type a chord symbol and press Enter. An empty box removes the chord; Esc leaves it as it was.");
    }

    closeChordInput(entry = this.chordInput) {
        if (!entry) return;
        entry.done = true;
        if (this.chordInput === entry) this.chordInput = null;
        entry.input.remove();
    }

    commit(next, previous = this.model) {
        this.notice = null;
        this.drawn = null;
        if (!next) {
            this.setStatus("That does not fit: each part sings one note at a time, inside the song.", true);
            this.draw();
            return false;
        }
        if (next === previous) return false;
        this.dropped = this.history.push(previous);
        this.model = next;
        this.localChanged = roll.changedBars(this.sheet, this.model);
        this.showKey();
        this.draw();
        this.scheduleWrite();
        return true;
    }

    undo() {
        if (!this.model || this.keyBusy) return;
        const previous = this.history.undo(this.history.lastDone?.keyed ? this.snapshot() : this.model);
        if (!previous) return;
        this.drawn = null;
        this.stepped(previous);
    }

    redo() {
        if (!this.model || this.keyBusy) return;
        const next = this.history.redo(this.history.lastUndone?.keyed ? this.snapshot() : this.model);
        if (!next) return;
        this.drawn = null;
        this.stepped(next);
    }

    stepped(entry) {
        this.notice = null;
        if (entry.keyed) this.stopPlaying();
        this.restore(entry);
        this.clearPicks();
        this.localChanged = roll.changedBars(this.sheet, this.model);
        this.showKey();
        this.draw();
        this.scheduleWrite();
    }

    scheduleWrite() {
        this.writePending = true;
        clearTimeout(this.writeTimer);
        this.writeTimer = setTimeout(() => this.write(), WRITE_DELAY);
    }

    async write() {
        clearTimeout(this.writeTimer);
        this.writePending = false;
        if (!this.sheet) return;
        const sequence = ++this.writeSequence;
        const sent = this.model;
        const { ok, payload } = await ask(WRITE_ROUTE, { abc: this.base, sheet: roll.sheetOf(sent) });
        if (this.isClosed || sequence !== this.writeSequence) return;
        if (ok && payload.ok) {
            this.current = payload.abc;
            this.changed = payload.bars;
            this.good = sent;
            if (document.activeElement !== this.textBox) this.textBox.value = payload.abc;
            this.notesDrawn = null;
            if (this.tab === "notes") this.drawNotes();
            this.showDescription();
        } else {
            this.setStatus(payload.error || "The edit could not be written into the score.", true);
            if (this.good && this.good !== this.model) {
                this.model = this.good;
                this.localChanged = roll.changedBars(this.sheet, this.model);
                this.showKey();
            }
        }
        this.draw();
    }

    async readTyped() {
        clearTimeout(this.readTimer);
        this.readTimer = null;
        const typed = roll.splitMark(this.textBox.value);
        const text = typed.score;
        if (text === (this.current || "").trim()) return true;
        if (!text) {
            this.base = "";
            this.current = "";
            this.sheet = null;
            this.model = null;
            this.changed = [];
            this.fillEmpty();
            this.setStatus(this.source ? "The box is empty: the node outputs " + this.source.result + " on its next run."
                : this.own ? "The box is empty: the node writes a new score on its next run."
                : "The box is empty: the node will sing the plan's own score.");
            return true;
        }
        const sequence = ++this.sequence;
        const { ok, payload } = await ask(READ_ROUTE, { abc: text });
        if (this.isClosed || sequence !== this.sequence) return false;
        this.typedProblem = null;
        if (!ok || !payload.ok) {
            this.typedProblem = payload.error || "The text cannot be read as a score.";
            this.setStatus(this.typedProblem, true);
            return false;
        }
        this.base = text;
        this.current = text;
        this.baseWords = typed.words
            || (text === String(this.planScore || "").trim() ? this.planWords : this.baseWords || this.planWords);
        if (typed.words) this.keepWords.box.checked = typed.keep;
        this.changed = [];
        this.history = new roll.History();
        this.moved = null;
        this.notesDrawn = null;
        this.adopt(payload.sheet);
        this.setStatus("Read as " + payload.sheet.bars.length + " bars. The piano roll and the notes show this text now.");
        return true;
    }

    async drawNotes() {
        const text = this.current || "";
        if (this.notesDrawn === text + "|" + this.changed.join(",")) return;
        this.paper.replaceChildren(element("p", "yue2-s-note", "Drawing the notes\u2026"));
        try {
            const ABCJS = await loadAbcjs();
            if (this.isClosed || this.tab !== "notes") return;
            const note = element("p", "yue2-s-note", this.changed.length
                ? "Bars " + barList(this.changed) + " (in red) are written again by this edit."
                : "The score as it stands. Measured: the singer sits an octave below the written vocal line.");
            const holder = element("div");
            this.paper.replaceChildren(note, holder);
            ABCJS.renderAbc(holder, roll.forNotation(text), {
                responsive: "resize", add_classes: true,
                staffwidth: Math.max(600, this.paper.clientWidth - 60),
                paddingtop: 6, paddingbottom: 16,
                wrap: { minSpacing: 1.4, maxSpacing: 2.5, preferredMeasuresPerLine: 4 },
            });
            for (const bar of this.changed) {
                for (const found of holder.querySelectorAll(".abcjs-mm" + bar)) {
                    found.style.fill = CHANGED_RED;
                    found.style.stroke = CHANGED_RED;
                    for (const child of found.querySelectorAll("*")) child.style.fill = CHANGED_RED;
                }
            }
            this.notesDrawn = text + "|" + this.changed.join(",");
        } catch (error) {
            this.paper.replaceChildren(element("p", "yue2-s-note", "The notes could not be drawn: " + error.message));
        }
    }

    togglePlay() {
        if (this.player.playing) this.stopPlaying();
        else this.startPlaying();
    }

    needsPiano() {
        return Object.keys(sounds.DEFAULTS).some((part) => this.soundOf(part) === "piano");
    }

    startPlaying() {
        if (!this.sheet || this.pianoWait) return;
        audioContext();
        if (PIANO.size || !this.needsPiano()) {
            this.playNow();
            return;
        }
        this.pianoWait = true;
        this.setStatus("Loading the piano\u2026");
        let failed = null;
        loadPiano()
            .catch((error) => {
                failed = error;
            })
            .then(() => {
                this.pianoWait = false;
                if (this.isClosed) return;
                this.setStatus(failed
                    ? "The piano did not load, so this plays as a plain synth: " + failed.message
                    : this.notice, Boolean(failed));
                this.playNow();
            });
    }

    playNow() {
        if (!this.sheet) return;
        const from = this.playhead >= this.sheet.total ? 0 : this.playhead;
        const list = roll.events(this.sheet, this.model, from, {
            Vocal: this.hearVoice.box.checked, Ins: this.hearIns.box.checked, chords: this.hearChords.box.checked,
        });
        try {
            this.player.play(list);
        } catch (error) {
            this.setStatus("The browser would not play sound: " + error.message, true);
            return;
        }
        this.playFrom = from;
        this.playButton.textContent = "Stop";
        const perTick = roll.secondsAt(this.sheet, 1);
        const step = () => {
            if (this.isClosed || !this.player.playing) return;
            const tick = this.playFrom + this.player.elapsed() / perTick;
            if (tick >= this.sheet.total) {
                this.stopPlaying();
                this.playhead = 0;
                this.draw();
                return;
            }
            this.playTick = tick;
            if (!this.rollBox.hidden && this.width) {
                const shown = this.visibleTicks();
                if (tick > this.tick0 + shown * 0.92 || tick < this.tick0) {
                    this.tick0 = tick - shown * 0.08;
                    this.clampView();
                }
                this.draw();
            }
            requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    }

    stopPlaying() {
        if (this.player.playing) {
            if (this.playTick !== null) this.playhead = Math.floor(this.playTick);
            this.player.stop();
        }
        this.playTick = null;
        if (this.playButton) this.playButton.textContent = "Play";
    }

    async writeScore() {
        if (!this.origin || this.own) return;
        this.setStatus("Writing the score on '" + (this.origin.title || "YuE2 Plan")
            + "'. Only that node runs; nothing is sung.");
        try {
            await app.queuePrompt(0, 1, { queueNodeIds: [String(this.origin.id)] });
        } catch (error) {
            this.setStatus("The plan node could not be queued: " + (error?.message || error), true);
        }
    }

    scoreArrived(detail) {
        if (!detail || !this.origin || String(detail.id) !== String(this.origin.id)) return;
        this.limit = limitFor(this.node);
        if (this.sheet) {
            this.paintFacts();
            this.showDescription();
            this.draw();
        }
        if (typeof detail.words === "string") this.planWords = detail.words;
        if (typeof detail.score !== "string") {
            this.refresh();
            return;
        }
        this.planScore = detail.score || null;
        this.refresh();
        if (!this.current || (!this.fromBox && !this.edited())) {
            this.opened = String(detail.score || "").trim();
            this.baseWords = this.planWords;
            this.load(detail.score);
            return;
        }
        this.setStatus((this.own ? "This node" : "The plan node") + " has written a new score. "
            + "'" + this.back + "' loads it; the edit stays until then.");
    }

    async reset() {
        if (this.track) {
            if (this.edited() && !(await confirmed("The notes in this window have been changed.",
                { title: "Put back the notes the song has?", ok: "Put them back",
                  cancel: "Keep the changes", danger: true }))) return;
            this.load(this.planScore);
            return;
        }
        if (this.planScore) {
            if (this.edited() && !(await confirmed(
                "The notes in this window have been edited.",
                { title: "Load the score " + (this.own ? "this node" : "the plan node")
                         + " wrote?", ok: "Load it", cancel: "Keep the edits",
                  danger: true }))) return;
            this.fromBox = false;
            this.baseWords = this.planWords;
            this.load(this.planScore);
            return;
        }
        if (!this.fromBox) return;
        if (!(await confirmed("The score it was made on has not arrived since the page was "
            + "opened, so the window stays empty until the next run.",
            { title: "Remove the edit kept on this node?", ok: "Remove it",
              danger: true }))) return;
        this.fromBox = false;
        this.baseWords = null;
        this.load("");
        this.setStatus("Apply removes the edit, and the next run " + (this.source
            ? "outputs " + this.source.result + " again." : this.own
            ? "writes the model's score again." : "sings the plan's own score."));
    }

    async escape() {
        if (this.chordInput || this.sectionInput) {
            this.closeChordInput();
            this.closeSectionInput();
            this.showDescription();
            this.rollBox.focus();
            return;
        }
        if (this.drag) {
            this.cancelDrag();
            return;
        }
        if ((this.selection.size || this.chordPicks.size) && this.tab === "roll") {
            this.clearPicks();
            this.draw();
            return;
        }
        if (this.edited() && !(await confirmed(
            "The notes in this window have been edited. Closing now leaves the node as it was.",
            { title: "Close the score editor?", ok: "Close and lose them",
              cancel: "Keep editing", danger: true }))) return;
        this.close();
    }

    async saveMidi() {
        if (this.chordInput) this.chordInput.finish(true);
        if (this.readTimer) await this.readTyped();
        if (this.writePending) await this.write();
        if (this.isClosed) return;
        const text = (this.current || "").trim();
        if (!text) return;
        const { ok, payload } = await ask(MIDI_ROUTE, { abc: text });
        if (this.isClosed) return;
        if (!ok || !payload.ok) {
            this.setStatus(payload.error || "The score could not be written as a MIDI file.", true);
            return;
        }
        const bytes = Uint8Array.from(atob(payload.data), (letter) => letter.charCodeAt(0));
        const link = document.createElement("a");
        link.href = URL.createObjectURL(new Blob([bytes], { type: "audio/midi" }));
        link.download = roll.midiFileName(this.node.title || "YuE2 score");
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(link.href), 10000);
        this.setStatus("Saved as " + link.download + ": the voice, the instrument line and the chords, "
            + "each on a track of its own.");
    }

    keepChanged() {
        const keep = this.keepWords.box.checked;
        rememberKeep(keep);
        this.setStatus(!this.baseWords
            ? "This score names no words yet, because the node has not written one, so it is sung with any words."
            : keep
            ? "Apply keeps this edit for new words: the node sings it with whatever lyrics and style it is given."
            : "Apply ties this edit to the words it was made for: with other words the node writes a new score.");
    }

    async apply() {
        if (this.chordInput) this.chordInput.finish(true);
        if (this.readTimer) await this.readTyped();
        if (this.writePending) await this.write();
        if (this.isClosed) return;
        if (this.track) {
            const typed = roll.splitMark(this.textBox.value).score;
            const text = (this.current || "").trim();
            if (this.typedProblem && typed !== text) {
                this.setStatus(this.typedProblem, true);
                return;
            }
            if (!text || text === this.planScore) {
                this.setStatus(TRACK_SAME, true);
                return;
            }
            this.track.apply(text, this.base === this.planScore ? [...this.changed] : []);
            this.close();
            return;
        }
        const typed = roll.splitMark(this.textBox.value).score;
        let text = (this.current || "").trim();
        if (this.typedProblem && typed !== text) {
            if (!(await confirmed(this.typedProblem
                + "\n\nThe node will try to sing it as it is.",
                { title: "The ABC text cannot be read note by note", ok: "Put it on anyway",
                  danger: true }))) return;
            text = typed;
        }
        const model = String(this.planScore || "").trim();
        const value = roll.editValue(text, model, this.baseWords, this.keepWords.box.checked);
        setWidgetValue(this.node, SCORE, value);
        this.node.properties = this.node.properties || {};
        if (value) {
            const earlier = this.fromBox && this.base !== model ? this.node.properties.yue2_score_bars || [] : [];
            const every = this.moved && this.sheet ? this.sheet.bars.map((_bar, index) => index) : [];
            this.node.properties.yue2_score_bars = [...new Set([...earlier, ...this.changed, ...every])]
                .sort((a, b) => a - b);
            this.node.properties.yue2_score_base = model ? hashText(model) : "";
        } else {
            delete this.node.properties.yue2_score_bars;
            delete this.node.properties.yue2_score_base;
        }
        paintScoreSummary(this.node);
        graphChanged();
        this.close();
    }
}

function openScoreEditor(node) {
    try {
        new ScoreEditor(node);
    } catch (error) {
        console.error("[YuE2] the score editor failed to open:", error);
        warned("The score editor could not open: " + (error?.message || error)
            + "\n\nThe score_abc box is still on the node's properties panel.");
    }
}

export function openScoreFor(text, track) {
    try {
        return new ScoreEditor(null, { ...track, text });
    } catch (error) {
        console.error("[YuE2] the score editor failed to open:", error);
        warned("The score editor could not open: " + (error?.message || error));
        return null;
    }
}

async function resetScoreEdit(node) {
    const box = roll.splitMark(widgetNamed(node, SCORE)?.value ?? "");
    if (!box.score) return;
    const next = sourceWords(node)
        ? "The next run outputs " + sourceWords(node).result + " again."
        : isGenerate(node)
        ? "The next run writes the model's score again, and a new seed gives a new tune."
        : "The next run sings the plan's own score.";
    if (!(await confirmed(next, { title: "Throw away the edited score kept on this node?",
                                  ok: "Throw it away", danger: true }))) return;
    setWidgetValue(node, SCORE, "");
    if (node.properties) {
        delete node.properties.yue2_score_bars;
        delete node.properties.yue2_score_base;
    }
    paintScoreSummary(node);
    graphChanged();
}

function plainButton(label, tooltip, onClick) {
    const button = document.createElement("button");
    button.textContent = label;
    button.title = tooltip;
    button.addEventListener("click", onClick);
    return button;
}

function paintScoreSummary(node) {
    const holder = node.__yue2ScoreSummary;
    if (!holder) return;
    holder.replaceChildren();
    fillScoreSummary(node, holder);
    const count = Math.max(2, holder.childElementCount);
    if (node.__yue2ScoreRows !== count) {
        node.__yue2ScoreRows = count;
        growToFit(node);
    }
}

function growToFit(node) {
    try {
        const wanted = node.computeSize?.()?.[1];
        if (wanted && node.size && node.size[1] < wanted) node.setSize?.([node.size[0], wanted]);
    } catch (error) {
        console.warn("[YuE2] the score summary could not resize its node:", error);
    }
    node.setDirtyCanvas?.(true, true);
}

function fillScoreSummary(node, holder) {
    const row = (...parts) => {
        const line = element("div");
        line.append(...parts);
        holder.appendChild(line);
    };
    const dim = (text) => element("span", "yue2-s-dim", text);
    const strong = (text) => element("span", "yue2-s-key", text);
    const warn = (text) => element("span", "yue2-s-warn", text);
    const own = isOwn(node);
    const source = sourceWords(node);
    const wired = sourceOf(node, SCORE);
    const box = roll.splitMark(widgetNamed(node, SCORE)?.value ?? "");
    if (node.__yue2ResetButton) node.__yue2ResetButton.hidden = Boolean(wired) || !box.score;
    if (wired) {
        row(dim("The score comes in through a wire from "), strong(wired.title || "another node"));
        row(dim(source ? source.wired
            : own ? "Sung as it arrives, whatever the words." : "Edit it where it is written."));
        if (node.__yue2ScoreButton) node.__yue2ScoreButton.disabled = true;
        return;
    }
    if (node.__yue2ScoreButton) node.__yue2ScoreButton.disabled = false;
    const origin = scoreSource(node)?.node ?? null;
    const planScore = origin?.__yue2Score || null;
    const planWords = origin ? wordsWanted(origin) : null;
    const facts = (abc) => {
        const found = roll.headerFacts(abc);
        return [found.key && "Key " + found.key, found.meter, found.bpm && found.bpm + " BPM",
            found.bars && found.bars + " bars"].filter(Boolean).join(" \u00B7 ");
    };
    const limitRow = (abc, edited) => {
        const limit = limitFor(node);
        const times = limit ? sheetTimes(abc) : null;
        if (!times || times.seconds <= limit.seconds) return;
        const late = roll.barsAfter(times, limit.seconds).filter((bar) => edited.includes(bar));
        const why = limit.auto ? " (lyrics limit)" : " (max_seconds)";
        row(late.length
            ? warn("Bars " + barList(late) + " come after " + roll.clock(limit.seconds) + why + " and are not sung.")
            : dim("Sung up to " + roll.clock(limit.seconds) + " of " + roll.clock(times.seconds) + why + "."));
    };
    if (box.score) {
        row(strong("Edited score"), dim(" \u00B7 " + facts(box.score)));
        const bars = node.properties?.yue2_score_bars || [];
        const base = node.properties?.yue2_score_base;
        const rewritten = bars.length ? "Bars " + barList(bars) + " rewritten"
            : source ? "Sent on as it stands" : "Sung as it stands";
        const otherWords = Boolean(box.words && planWords && box.words !== planWords);
        const kept = Boolean(box.keep && box.words && !source);
        if (otherWords && kept) {
            row(dim("Made for other words and kept for new ones: sung with the words " + (own ? "this node" : "the plan")
                + " has now."));
        } else if (otherWords) {
            row(warn(source ? source.other
                : own ? "Made for other words: this node writes a new score instead."
                : "Made for other words: the plan's own score is sung instead."));
        } else if (planScore && base && base !== hashText(String(planScore).trim())) {
            row(box.words ? dim(rewritten + " on an earlier take; still sung.")
                : warn("Edited on an earlier score; the plan has changed since."));
        } else if (source) {
            row(dim(bars.length ? rewritten + source.rest : rewritten + " instead of " + source.result + "."));
        } else if (own) {
            row(dim(rewritten + (kept ? ". A new seed and new words keep these notes." : ". A new seed keeps these notes.")));
        } else {
            row(dim((bars.length ? rewritten + "; the rest as the model wrote it." : "Sung as it stands in this node.")
                + (kept ? " Kept for new words." : "")));
        }
        if (!otherWords || kept) limitRow(box.score, bars);
        return;
    }
    if (planScore) {
        row(strong(source ? source.heading : "The model's score"), dim(" \u00B7 " + facts(planScore)));
        row(dim(source ? source.made
            : own ? "Written by the model on each run. Edit score\u2026 to change notes."
            : "Sung exactly as written. Edit score\u2026 to change notes."));
        limitRow(planScore, []);
        return;
    }
    if (own) {
        row(strong("No score yet"));
        row(dim(source ? source.noScore
            : "Run once, and the score this node writes can be edited here."));
        return;
    }
    if (origin) {
        row(strong("No score yet"));
        row(dim("Run '" + (origin.title || "YuE2 Plan") + "', or open the editor and write one."));
        return;
    }
    row(strong("No plan connected"));
    row(dim("Connect a YuE2 Plan to edit the score it writes."));
}

function refreshScoreSummaries() {
    for (const node of graphNodes()) {
        if (isRender(node) || isOwn(node)) paintScoreSummary(node);
    }
}

function unbindScoreSocket(node) {
    for (const input of node.inputs || []) {
        if (input.name === SCORE && input.widget) input.widget = undefined;
    }
    node.setDirtyCanvas?.(true, true);
}

function rebindScoreOnSave(saved) {
    for (const input of saved?.inputs || []) {
        if (input.name === SCORE && !input.widget) input.widget = { name: SCORE };
    }
}

function settleBesideTheSongButton(node, buttons) {
    const song = node.__yue2Button;
    if (song?.parentElement) {
        song.parentElement.append(...buttons);
    } else {
        const { buttons: made } = buttonRow(node, "yue2_score_edit", [
            { label: EDIT_LABEL, tooltip: EDIT_TOOLTIP, onClick: () => openScoreEditor(node) },
            {
                label: RESET_LABEL, tooltip: sourceWords(node)?.resetTooltip || RESET_TOOLTIP_OWN,
                onClick: () => resetScoreEdit(node),
            },
        ]);
        node.__yue2ScoreButton = made[0];
        node.__yue2ResetButton = made[1];
    }
    const widgets = node.widgets || [];
    const mine = widgets.findIndex((w) => w.name === SCORE_SUMMARY);
    const songAt = widgets.findIndex((w) => w.name === SONG_SUMMARY);
    if (mine >= 0 && songAt > mine) {
        const [moved] = widgets.splice(mine, 1);
        widgets.splice(widgets.findIndex((w) => w.name === SONG_SUMMARY) + 1, 0, moved);
    }
    paintScoreSummary(node);
    node.setDirtyCanvas?.(true, true);
}

function installScoreEditor(node) {
    installStyle(SCORE_STYLE_ID, SCORE_STYLE);
    const summary = element("div", "yue2-s-summary");
    summary.title = "Click to open the score editor";
    summary.addEventListener("click", () => {
        if (!node.__yue2ScoreButton?.disabled) openScoreEditor(node);
    });
    node.__yue2ScoreSummary = summary;
    if (isOwn(node)) {
        const edit = plainButton(EDIT_LABEL, EDIT_TOOLTIP, () => openScoreEditor(node));
        const reset = plainButton(RESET_LABEL, sourceWords(node)?.resetTooltip || RESET_TOOLTIP_OWN,
            () => resetScoreEdit(node));
        reset.hidden = true;
        node.__yue2ScoreButton = edit;
        node.__yue2ResetButton = reset;
        panelWidget(node, SCORE_SUMMARY, summary, () => (node.__yue2ScoreRows > 2 ? SUMMARY_H : SONG_SUMMARY_H));
        const watched = isLoadMidi(node) ? [MODE, VOCAL_TRACK, INSTRUMENT_TRACK] : [isTranscribe(node) ? MODE : LYRICS];
        for (const name of watched) repaintOnChange(node, name);
        queueMicrotask(() => {
            try {
                settleBesideTheSongButton(node, [edit, reset]);
            } catch (error) {
                console.error("[YuE2] the score editor buttons could not join the song editor's:", error);
            }
        });
    } else {
        const { buttons } = buttonRow(node, "yue2_score_edit", [
            { label: EDIT_LABEL, tooltip: EDIT_TOOLTIP, onClick: () => openScoreEditor(node) },
            { label: RESET_LABEL, tooltip: RESET_TOOLTIP_PLAN, onClick: () => resetScoreEdit(node) },
        ]);
        node.__yue2ScoreButton = buttons[0];
        node.__yue2ResetButton = buttons[1];
        buttons[1].hidden = true;
        panelWidget(node, SCORE_SUMMARY, summary, () => SUMMARY_H, true);
    }
    showWidget(node, SCORE, false);
    unbindScoreSocket(node);
    const widget = widgetNamed(node, SCORE);
    if (widget) {
        const original = widget.callback;
        widget.callback = function () {
            const result = original?.apply(this, arguments);
            paintScoreSummary(node);
            return result;
        };
    }
    paintScoreSummary(node);
}

function repaintOnChange(node, name) {
    const widget = widgetNamed(node, name);
    if (!widget || widget.__yue2Repaints) return;
    const original = widget.callback;
    widget.callback = function () {
        const result = original?.apply(this, arguments);
        setTimeout(refreshScoreSummaries, 0);
        return result;
    };
    widget.__yue2Repaints = true;
}

function watchForLimits(nodeType, name) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated?.apply(this, arguments);
        repaintOnChange(this, name);
        return result;
    };
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
        const result = onConnectionsChange?.apply(this, arguments);
        setTimeout(refreshScoreSummaries, 0);
        return result;
    };
}

app.registerExtension({
    name: "yue2.score_editor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === OPTIONS_NODE) {
            watchForLimits(nodeType, MAX_SECONDS);
            return;
        }
        const plan = PLAN_NODES.includes(nodeData.name);
        const own = [GENERATE, TRANSCRIBE, LOAD_MIDI].includes(nodeData.name);
        if (plan || own) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const result = onExecuted?.apply(this, arguments);
                const score = message?.[SCORE_UI]?.[0];
                const words = message?.[WORDS_UI]?.[0];
                const auto = message?.[AUTO_SECONDS_UI]?.[0];
                const track = message?.[TRACK_UI]?.[0];
                const marks = message?.[MARKS_UI]?.[0];
                if (typeof score === "string") this.__yue2Score = score;
                if (typeof words === "string") this.__yue2Words = words;
                if (typeof auto === "number") this.__yue2AutoSeconds = auto;
                if (typeof track === "string") {
                    this.__yue2Marks = marks && typeof marks === "object" ? { ...marks } : null;
                }
                if (typeof score === "string" || typeof words === "string") {
                    window.dispatchEvent(new CustomEvent(SCORE_EVENT, { detail: { id: this.id, score, words } }));
                }
                if (typeof score === "string" || typeof words === "string" || typeof auto === "number") {
                    refreshScoreSummaries();
                }
                return result;
            };
        }
        if (plan || nodeData.name === BATCH) {
            watchForLimits(nodeType, LYRICS);
            return;
        }
        if (nodeData.name !== RENDER && !own) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                installScoreEditor(this);
            } catch (error) {
                console.error("[YuE2] the score editor could not be added to this node:", error);
                showWidget(this, SCORE, true);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            unbindScoreSocket(this);
            setTimeout(() => paintScoreSummary(this), 0);
            return result;
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (saved) {
            const result = onSerialize?.apply(this, arguments);
            rebindScoreOnSave(saved);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            setTimeout(() => paintScoreSummary(this), 0);
            return result;
        };
    },
});
