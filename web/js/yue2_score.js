import { app } from "../../scripts/app.js";
import {
    ask, buttonRow, element, frame, installStyle, panelWidget,
    setWidgetValue, showWidget, sourceOf, widgetNamed,
} from "./yue2_controls.js";
import * as roll from "./yue2_roll.js";

const RENDER = "YuE2RenderPlan";
const GENERATE = "YuE2GenerateSong";
const PLAN_NODES = ["YuE2Plan", "YuE2SelectPlan"];
const SCORE = "score_abc";
const PLAN = "plan";
const SCORE_UI = "yue2_score";
const WORDS_UI = "yue2_words";
const SCORE_EVENT = "yue2-score-written";
const READ_ROUTE = "/yue2/score/read";
const WRITE_ROUTE = "/yue2/score/write";
const WRITE_DELAY = 250;
const READ_DELAY = 450;
const SUMMARY_H = 58;
const SONG_SUMMARY_H = 42;
const SCORE_SUMMARY = "yue2_score_summary";
const SONG_SUMMARY = "yue2_song_summary";
const ABCJS_URL = new URL("./vendor/abcjs-basic-min.cjs", import.meta.url).href;

const KEYS_W = 58;
const RULER_H = 36;
const CHORD_H = 24;
const ROW_H = 14;
const EDGE_PX = 7;
const CHANGED_RED = "#C0392B";

const PART_COLORS = { Vocal: "#4E98C4", Ins: "#5FA37A" };
const PART_NAMES = { Vocal: "Voice part", Ins: "Instrument part" };
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
    "new score instead.";

const CHORD_HELP =
    "Chord symbols are a root from A to G with an optional # or b, then one of: nothing (major), " +
    "m, dim, aug, 7, maj7, m7, dim7, m7b5, sus4, sus2, 6, m6, 7sus4, m(maj7) -- and an optional " +
    "bass note after a slash, as in C/E. Leave the box empty to remove the chord.";

const SCORE_STYLE_ID = "yue2-score-style";
const SCORE_STYLE = `
.yue2-panel.yue2-score { width: min(1500px, 98vw); }
.yue2-score [hidden] { display: none !important; }
.yue2-score h3 { font-size: 15px; font-weight: 600; margin: 0; }
.yue2-score .yue2-s-sub { font-size: 11px; color: var(--descrip-text, #999); margin: 2px 0 10px; }
.yue2-score button, .yue2-score select, .yue2-score input[type="text"], .yue2-score textarea {
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
.yue2-s-check { font-size: 12px; display: inline-flex; gap: 4px; align-items: center;
    color: var(--descrip-text, #bbb); user-select: none; }
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
.yue2-s-hint { font-size: 11px; line-height: 1.45; color: var(--descrip-text, #999); margin: 2px 0 0; }
.yue2-s-foot { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
.yue2-s-foot button { font-size: 13px; padding: 6px 14px; }
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
let ABCJS_LOADING = null;

function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
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

function isRender(node) {
    return node?.type === RENDER || node?.comfyClass === RENDER;
}

function scoreSource(node) {
    if (isGenerate(node)) return { node, own: true };
    const origin = planSource(node);
    return origin ? { node: origin, own: false } : null;
}

function graphNodes() {
    return app.graph?._nodes ?? app.graph?.nodes ?? [];
}

function audioContext() {
    if (!AUDIO) AUDIO = new (window.AudioContext || window.webkitAudioContext)();
    if (AUDIO.state === "suspended") AUDIO.resume();
    return AUDIO;
}

function midiHz(pitch) {
    return 440 * Math.pow(2, (pitch - 69) / 12);
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
            this.voices.push(this.tone(context, master, origin + item.at, item.length, item.pitch, item.part));
        }
        this.playing = true;
    }

    tone(context, master, start, length, pitch, part) {
        const oscillator = context.createOscillator();
        const gain = context.createGain();
        const level = part === "Vocal" ? 0.16 : part === "Ins" ? 0.07 : 0.03;
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
    constructor(node) {
        this.node = node;
        const source = scoreSource(node);
        this.origin = source?.node ?? null;
        this.own = Boolean(source?.own);
        this.planScore = this.origin?.__yue2Score || null;
        this.planWords = this.origin?.__yue2Words || null;
        this.baseWords = null;
        this.player = new Player();
        this.sequence = 0;
        this.writeSequence = 0;
        this.isClosed = false;
        this.tab = "roll";
        this.part = "Vocal";
        this.snap = 1;
        this.lastLength = 0;
        this.selection = new Set();
        this.drag = null;
        this.working = null;
        this.playhead = 0;
        this.playTick = null;
        this.chordInput = null;
        this.changed = [];
        this.localChanged = [];
        this.sheet = null;
        this.model = null;
        this.good = null;
        this.history = new roll.History();
        this.tick0 = 0;
        this.pxPerTick = 4;
        this.pitchTop = 79;
        this.width = 0;
        this.height = 0;
        this.ratio = 1;
        installStyle(SCORE_STYLE_ID, SCORE_STYLE);
        this.build();
        this.open();
    }

    build() {
        const { panel, close, handle } = frame({ sticky: true, onClose: () => this.closed() });
        panel.classList.add("yue2-score");
        this.panel = panel;
        this.close = close;
        handle.onEscape = () => this.escape();

        panel.appendChild(element("h3", "", "Score \u2014 " + (this.node.title || "YuE2 Render Plan")));
        panel.appendChild(element("p", "yue2-s-sub",
            "The notes this node sings. Nothing is written to the node until Apply."));

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
        this.playButton.title = "Play from the marker (Space). A plain synth, not YuE2: the song itself is made when the workflow runs.";
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
        tools.append(this.playButton, rewind, this.hearVoice.holder, this.hearIns.holder, this.hearChords.holder,
            element("span", "yue2-s-gap"));

        this.rollTools = element("span", "yue2-s-bar");
        this.rollTools.style.margin = "0";
        this.partSelect = selectControl(Object.entries(PART_NAMES), "Vocal",
            "Which part the mouse edits. The other part is drawn faintly behind it.", (value) => {
                this.part = value;
                this.selection.clear();
                this.draw();
            });
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
        this.rollTools.append(this.partSelect, this.snapHolder, zoomOut, zoomIn, fit, this.undoButton, this.redoButton);
        tools.appendChild(this.rollTools);
        panel.appendChild(tools);

        const body = element("div", "yue2-s-body");
        this.rollBox = element("div", "yue2-s-roll");
        this.rollBox.tabIndex = 0;
        this.canvas = document.createElement("canvas");
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
        panel.appendChild(element("p", "yue2-s-hint", HONEST_NOTE + (this.own ? RETRY_HERE : RETRY_ON_PLAN)));

        const foot = element("div", "yue2-s-foot");
        this.writeButton = element("button", "", "Write the score");
        this.writeButton.title = "Run only the plan node feeding this one: the score is written, nothing is sung.";
        this.writeButton.addEventListener("click", () => this.writeScore());
        this.resetButton = element("button", "", "Back to the model's score");
        this.resetButton.title = this.own
            ? "Throw away the edits and load the score this node wrote on its last run."
            : "Throw away the edits and load the score the plan node wrote.";
        this.resetButton.addEventListener("click", () => this.reset());
        const cancel = element("button", "", "Cancel");
        cancel.addEventListener("click", () => this.escape());
        this.applyButton = element("button", "yue2-s-go", "Apply");
        this.applyButton.addEventListener("click", () => this.apply());
        foot.append(this.writeButton, this.resetButton, element("span", "yue2-s-grow"), cancel, this.applyButton);
        panel.appendChild(foot);

        const chords = document.createElement("datalist");
        chords.id = "yue2-s-chords";
        for (const name of COMMON_CHORDS) {
            const option = document.createElement("option");
            option.value = name;
            chords.appendChild(option);
        }
        panel.appendChild(chords);

        this.canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
        this.canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
        this.canvas.addEventListener("pointerup", (event) => this.pointerUp(event));
        this.canvas.addEventListener("pointercancel", () => this.cancelDrag());
        this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());
        this.canvas.addEventListener("wheel", (event) => this.wheel(event), { passive: false });
        panel.addEventListener("keydown", (event) => this.key(event));
        this.observer = new ResizeObserver(() => this.resize());
        this.observer.observe(this.rollBox);

        this.onScore = (event) => this.scoreArrived(event.detail);
        window.addEventListener(SCORE_EVENT, this.onScore);
    }

    open() {
        const box = roll.splitMark(widgetNamed(this.node, SCORE)?.value ?? "");
        const base = this.node.properties?.yue2_score_base;
        this.fromBox = Boolean(box.score);
        this.opened = box.score || String(this.planScore || "").trim();
        this.baseWords = box.score ? box.words : this.planWords;
        if (box.score && box.words && this.planWords && box.words !== this.planWords) {
            this.pendingNote = "This edit was made for other words than " + (this.own ? "the last run had" : "the plan has now")
                + ", so it is not sung until they come back. 'Back to the model's score' loads the new score.";
            this.pendingBad = true;
        } else if (box.score && this.planScore && base && base !== hashText(String(this.planScore).trim())) {
            this.pendingNote = box.words
                ? "This edit was made on an earlier take of these words, and it is still sung. "
                    + "'Back to the model's score' loads the newer take."
                : "This edit was made on an earlier score; the plan has written a new one since. "
                    + "'Back to the model's score' loads the new one.";
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

    adopt(sheet, fit = true) {
        this.sheet = sheet;
        this.model = roll.modelOf(sheet);
        this.good = this.model;
        this.localChanged = [];
        const choices = roll.snapChoices(sheet.per_quarter);
        const preferred = choices.find((c) => c.name === "Eighth notes") || choices[0];
        this.snap = preferred ? preferred.ticks : 1;
        this.snapHolder.replaceChildren(selectControl(choices.map((c) => [c.ticks, c.name]), this.snap,
            "The grid notes snap to while drawing, moving and stretching.", (value) => {
                this.snap = Number(value);
                this.draw();
            }));
        const bars = sheet.bars.length;
        const key = sheet.bars[0]?.key || "";
        const meter = sheet.bars[0]?.meter || "";
        this.facts.textContent = [key && "Key " + key, meter, sheet.bpm + " BPM",
            bars + " bars", roll.clock(sheet.seconds)].filter(Boolean).join(" \u00B7 ");
        this.playhead = 0;
        if (fit) this.fitView();
        this.showTab(this.tab);
        this.setStatus(this.describe());
        this.refresh();
    }

    describe() {
        if (!this.changed.length) {
            return this.sheet ? "No changes. This is the score as it came in." : "";
        }
        return "Bars " + barList(this.changed) + " will be written again; every other bar stays exactly as it was.";
    }

    fillEmpty(problem) {
        this.empty.replaceChildren();
        if (problem) {
            this.empty.appendChild(element("div", "", "The piano roll cannot draw this score. The ABC tab has its text."));
            return;
        }
        if (this.own) {
            this.empty.appendChild(element("div", "",
                "There is no score yet. Run the workflow once: the score this node writes appears here, "
                + "and every edit after that costs only the singing. Or paste a score into the ABC tab."));
            return;
        }
        if (this.origin) {
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
    }

    refresh() {
        this.undoButton.disabled = !this.history.canUndo;
        this.redoButton.disabled = !this.history.canRedo;
        this.writeButton.hidden = !this.origin || this.own;
        this.resetButton.disabled = !this.planScore && !this.fromBox;
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
        const box = this.rollBox.getBoundingClientRect();
        if (!box.width || !box.height) return;
        const firstTime = !this.width;
        this.ratio = window.devicePixelRatio || 1;
        this.width = box.width;
        this.height = box.height;
        this.canvas.width = Math.round(box.width * this.ratio);
        this.canvas.height = Math.round(box.height * this.ratio);
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
        c.fillStyle = "#1b1b1b";
        c.fillRect(0, 0, W, H);

        c.save();
        c.beginPath();
        c.rect(KEYS_W, top, W - KEYS_W, H - top);
        c.clip();
        const rows = Math.ceil((H - top) / ROW_H) + 1;
        for (let r = 0; r < rows; r++) {
            const pitch = this.pitchTop - r;
            const rowY = top + r * ROW_H;
            c.fillStyle = roll.isBlack(pitch) ? "#181818" : "#212121";
            c.fillRect(KEYS_W, rowY, W - KEYS_W, ROW_H);
            if (((pitch % 12) + 12) % 12 === 0) {
                c.fillStyle = "#303030";
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
                c.fillStyle = "rgba(209, 154, 63, 0.11)";
                c.fillRect(barX, top, barW, H - top);
            }
            if (!bar.editable) {
                c.fillStyle = "rgba(255, 255, 255, 0.05)";
                for (let stripe = -H; stripe < barW; stripe += 12) {
                    c.fillRect(barX + stripe + (H - top) / 2, top, 3, H - top);
                }
            }
            const den = Number(bar.meter.split("/")[1]) || 4;
            const beat = (sheet.per_quarter * 4) / den;
            if (this.snap * px >= 7) {
                c.fillStyle = "#262626";
                for (let t = bar.start + this.snap; t < bar.start + bar.length; t += this.snap) {
                    c.fillRect(Math.round(this.x(t)), top, 1, H - top);
                }
            }
            c.fillStyle = "#353535";
            for (let t = bar.start + beat; t < bar.start + bar.length; t += beat) {
                c.fillRect(Math.round(this.x(t)), top, 1, H - top);
            }
            c.fillStyle = "#5c5c5c";
            c.fillRect(Math.round(barX), top, 1, H - top);
        }
        this.drawPart(c, model, this.part === "Vocal" ? "Ins" : "Vocal", true, first, lastTick);
        this.drawPart(c, model, this.part, false, first, lastTick);
        if (this.drag?.mode === "box") {
            const x0 = this.x(this.drag.tick);
            const x1 = this.x(this.drag.toTick);
            const y0 = this.y(Math.max(this.drag.pitch, this.drag.toPitch));
            const y1 = this.y(Math.min(this.drag.pitch, this.drag.toPitch)) + ROW_H;
            c.strokeStyle = "rgba(255, 255, 255, 0.7)";
            c.setLineDash([4, 3]);
            c.strokeRect(Math.min(x0, x1), y0, Math.abs(x1 - x0), y1 - y0);
            c.setLineDash([]);
        }
        c.restore();

        c.fillStyle = "#262626";
        c.fillRect(0, 0, W, RULER_H);
        c.fillStyle = "#1f1f1f";
        c.fillRect(0, RULER_H, W, CHORD_H);
        c.save();
        c.beginPath();
        c.rect(KEYS_W, 0, W - KEYS_W, top);
        c.clip();
        for (const section of sheet.sections) {
            const startBar = sheet.bars[section.bar];
            const endBar = sheet.bars[section.bar + section.bars - 1];
            const left = this.x(startBar.start);
            const right = this.x(endBar.start + endBar.length);
            if (right < KEYS_W || left > W) continue;
            c.fillStyle = sectionColor(section.name);
            c.globalAlpha = 0.85;
            c.fillRect(left + 1, 1, right - left - 2, 14);
            c.globalAlpha = 1;
            c.fillStyle = "#fff";
            c.fillText(section.name || "section", Math.max(left, KEYS_W) + 4, 12);
        }
        const every = 28 / (px * (sheet.bars[0]?.length || 1)) > 1 ? 4 : 1;
        c.fillStyle = "#bdbdbd";
        index = roll.barAt(sheet, Math.max(0, Math.floor(first)));
        for (; index < sheet.bars.length; index++) {
            const bar = sheet.bars[index];
            if (bar.start > lastTick) break;
            if (index % every === 0) c.fillText(String(index + 1), this.x(bar.start) + 3, 30);
            c.fillStyle = "#4a4a4a";
            c.fillRect(Math.round(this.x(bar.start)), 17, 1, RULER_H - 17);
            c.fillStyle = "#bdbdbd";
        }
        for (const chord of model.chords) {
            const chordX = this.x(chord.start);
            if (chordX > W || chordX + 60 < KEYS_W) continue;
            const width = c.measureText(chord.name).width + 10;
            c.fillStyle = "#3a3f47";
            c.fillRect(chordX, RULER_H + 4, width, CHORD_H - 8);
            c.fillStyle = "#e8e8e8";
            c.font = "11px system-ui, sans-serif";
            c.fillText(chord.name, chordX + 5, RULER_H + CHORD_H - 8);
            c.font = "10px system-ui, sans-serif";
        }
        const marker = this.playTick ?? this.playhead;
        const markerX = this.x(marker);
        if (markerX >= KEYS_W && markerX <= W) {
            c.fillStyle = "#E0A45A";
            c.fillRect(Math.round(markerX) - 1, 0, 2, H);
        }
        c.restore();

        c.fillStyle = "#2a2a2a";
        c.fillRect(0, 0, KEYS_W, H);
        c.fillStyle = "#9a9a9a";
        c.fillText("bar", 6, 30);
        c.fillText("chord", 6, RULER_H + CHORD_H - 8);
        c.save();
        c.beginPath();
        c.rect(0, top, KEYS_W, H - top);
        c.clip();
        for (let r = 0; r < rows; r++) {
            const pitch = this.pitchTop - r;
            const rowY = top + r * ROW_H;
            c.fillStyle = roll.isBlack(pitch) ? "#161616" : "#d9d9d9";
            c.fillRect(roll.isBlack(pitch) ? 0 : 0, rowY, roll.isBlack(pitch) ? KEYS_W * 0.62 : KEYS_W, ROW_H - 1);
            if (((pitch % 12) + 12) % 12 === 0) {
                c.fillStyle = "#333";
                c.fillText(roll.noteName(pitch), KEYS_W - 24, rowY + ROW_H - 3);
            }
        }
        c.restore();
        c.fillStyle = "#444";
        c.fillRect(KEYS_W - 1, 0, 1, H);
        this.undoButton.disabled = !this.history.canUndo;
        this.redoButton.disabled = !this.history.canRedo;
    }

    drawPart(c, model, part, ghost, first, lastTick) {
        const px = this.pxPerTick;
        const top = RULER_H + CHORD_H;
        for (const note of model.notes[part]) {
            if (note.start + note.length < first || note.start > lastTick) continue;
            const noteY = this.y(note.pitch);
            if (noteY + ROW_H < top || noteY > this.height) continue;
            const noteX = this.x(note.start);
            const width = Math.max(2, note.length * px - 1);
            c.globalAlpha = ghost ? 0.26 : 1;
            c.fillStyle = PART_COLORS[part];
            c.fillRect(noteX, noteY + 1, width, ROW_H - 2);
            c.globalAlpha = 1;
            if (ghost) continue;
            if (this.selection.has(note.id)) {
                c.strokeStyle = "#ffffff";
                c.lineWidth = 1.5;
                c.strokeRect(noteX + 0.75, noteY + 1.75, width - 1.5, ROW_H - 3.5);
            }
            if (width > 12) {
                c.fillStyle = "rgba(255, 255, 255, 0.45)";
                c.fillRect(noteX + width - 3, noteY + 4, 2, ROW_H - 8);
            }
            if (width > 36) {
                const bar = this.sheet.bars[roll.barAt(this.sheet, note.start)];
                c.fillStyle = "rgba(0, 0, 0, 0.78)";
                c.fillText(roll.noteName(note.pitch, roll.flatsIn(this.sheet, bar.key)), noteX + 3, noteY + ROW_H - 4);
            }
        }
    }

    local(event) {
        const box = this.canvas.getBoundingClientRect();
        return { px: event.clientX - box.left, py: event.clientY - box.top };
    }

    hitNote(px, py) {
        const pitch = this.pitchAt(py);
        const tick = this.tickAt(px);
        return this.model.notes[this.part].find((note) => note.pitch === pitch && note.start <= tick
            && tick < note.start + note.length + EDGE_PX / this.pxPerTick) || null;
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

    blip(pitch) {
        try {
            this.player.blip(pitch, this.part);
        } catch (error) {
            void error;
        }
    }

    pointerDown(event) {
        if (!this.sheet) return;
        this.rollBox.focus();
        const { px, py } = this.local(event);
        const top = RULER_H + CHORD_H;
        const tick = this.tickAt(px);
        if (px < KEYS_W) {
            if (py >= top) this.blip(this.pitchAt(py));
            return;
        }
        if (py < RULER_H) {
            this.playhead = clamp(roll.snapDown(tick, this.snap), 0, this.sheet.total);
            const wasPlaying = this.player.playing;
            this.stopPlaying();
            this.playhead = clamp(roll.snapDown(tick, this.snap), 0, this.sheet.total);
            if (wasPlaying) this.startPlaying();
            this.draw();
            return;
        }
        if (py < top) {
            this.chordAt(tick, px);
            return;
        }
        const pitch = this.pitchAt(py);
        const hit = this.hitNote(px, py);
        if (hit) {
            if (event.button === 2) {
                const ids = this.selection.has(hit.id) ? [...this.selection] : [hit.id];
                this.selection.clear();
                this.commit(roll.deleteNotes(this.model, this.part, ids));
                return;
            }
            const edge = this.x(hit.start + hit.length) - px <= EDGE_PX;
            if (event.shiftKey && !edge) {
                if (this.selection.has(hit.id)) this.selection.delete(hit.id);
                else this.selection.add(hit.id);
                this.draw();
                return;
            }
            if (this.barLocked(hit.start)) return;
            if (!this.selection.has(hit.id)) this.selection = new Set([hit.id]);
            this.canvas.setPointerCapture(event.pointerId);
            this.blip(hit.pitch);
            this.drag = edge
                ? { mode: "resize", id: hit.id, from: this.model, start: hit.start, length: hit.length }
                : { mode: "move", ids: [...this.selection], from: this.model, base: this.model, tick, pitch, sounded: hit.pitch };
            this.draw();
            return;
        }
        if (event.button !== 0) return;
        this.canvas.setPointerCapture(event.pointerId);
        if (event.shiftKey) {
            this.drag = { mode: "box", tick, pitch, toTick: tick, toPitch: pitch, kept: new Set(this.selection) };
            return;
        }
        if (this.barLocked(tick)) return;
        const start = roll.snapDown(tick, this.snap);
        const placed = roll.addNote(this.model, this.part, start, this.lastLength || this.snap, pitch, this.sheet.total);
        if (!placed) {
            this.setStatus("There is no room for a note there: each part sings one note at a time.", true);
            return;
        }
        this.selection = new Set([placed.id]);
        this.blip(pitch);
        this.working = placed.model;
        this.drag = { mode: "move", ids: [placed.id], from: this.model, base: placed.model, tick: start, pitch, sounded: pitch };
        this.draw();
    }

    pointerMove(event) {
        if (!this.sheet) return;
        const { px, py } = this.local(event);
        if (!this.drag) {
            const hit = py > RULER_H + CHORD_H && px > KEYS_W ? this.hitNote(px, py) : null;
            let cursor = "crosshair";
            if (py <= RULER_H + CHORD_H) cursor = "pointer";
            else if (hit) cursor = this.x(hit.start + hit.length) - px <= EDGE_PX ? "ew-resize" : "move";
            this.canvas.style.cursor = cursor;
            return;
        }
        const tick = this.tickAt(px);
        const pitch = this.pitchAt(py);
        const total = this.sheet.total;
        if (this.drag.mode === "box") {
            this.drag.toTick = tick;
            this.drag.toPitch = pitch;
            const inside = roll.notesIn(this.model, this.part, this.drag.tick, tick, this.drag.pitch, pitch);
            this.selection = new Set([...this.drag.kept, ...inside]);
        } else if (this.drag.mode === "move") {
            const ticks = roll.snapTo(tick - this.drag.tick, this.snap);
            const semitones = pitch - this.drag.pitch;
            const moved = roll.moveNotes(this.drag.base, this.part, this.drag.ids, ticks, semitones, total);
            if (moved) {
                this.working = moved;
                const lead = moved.notes[this.part].find((n) => n.id === this.drag.ids[0]);
                if (lead && lead.pitch !== this.drag.sounded) {
                    this.drag.sounded = lead.pitch;
                    this.blip(lead.pitch);
                }
            }
        } else if (this.drag.mode === "resize") {
            const length = Math.max(1, roll.snapTo(tick - this.drag.start, this.snap)) || this.snap;
            const stretched = roll.resizeNote(this.drag.from, this.part, this.drag.id, length, total);
            if (stretched) {
                this.working = stretched;
                this.lastLength = length;
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
        if ((drag.mode === "move" || drag.mode === "resize") && result && result !== drag.from) {
            if (drag.mode === "move") {
                const lead = result.notes[this.part].find((n) => n.id === drag.ids[0]);
                if (lead && !this.lastLength) this.lastLength = lead.length;
            }
            this.commit(result, drag.from);
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
        if (event.ctrlKey || event.metaKey) {
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
        const tag = event.target?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        const ctrl = event.ctrlKey || event.metaKey;
        const letter = String(event.key || "").toLowerCase();
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
            this.selection = new Set(this.model.notes[this.part].map((n) => n.id));
            this.draw();
            return;
        }
        if (!this.selection.size) return;
        if (event.key === "Delete" || event.key === "Backspace") {
            event.preventDefault();
            const ids = [...this.selection];
            this.selection.clear();
            this.commit(roll.deleteNotes(this.model, this.part, ids));
            return;
        }
        const moves = {
            ArrowUp: [0, event.shiftKey ? 12 : 1], ArrowDown: [0, event.shiftKey ? -12 : -1],
            ArrowLeft: [-this.snap, 0], ArrowRight: [this.snap, 0],
        };
        if (moves[event.key]) {
            event.preventDefault();
            const [ticks, semitones] = moves[event.key];
            const moved = roll.moveNotes(this.model, this.part, [...this.selection], ticks, semitones, this.sheet.total);
            if (!moved) {
                this.setStatus("That move does not fit: it would overlap another note or leave the song.", true);
                return;
            }
            this.commit(moved);
            const lead = moved.notes[this.part].find((n) => this.selection.has(n.id));
            if (lead && semitones) this.blip(lead.pitch);
        }
    }

    chordAt(tick, px) {
        if (this.barLocked(tick)) return;
        const c = this.canvas.getContext("2d");
        c.font = "11px system-ui, sans-serif";
        const found = this.model.chords.find((chord) => {
            const left = this.x(chord.start);
            return px >= left && px <= left + c.measureText(chord.name).width + 10;
        });
        const start = found ? found.start : clamp(roll.snapDown(tick, this.snap), 0, this.sheet.total - 1);
        this.openChordInput(start, found ? found.name : "");
    }

    openChordInput(start, name) {
        this.closeChordInput();
        const input = document.createElement("input");
        input.type = "text";
        input.className = "yue2-s-chord";
        input.value = name;
        input.placeholder = "Am7, F/C \u2014 empty removes";
        input.setAttribute("list", "yue2-s-chords");
        input.style.left = clamp(this.x(start), KEYS_W, Math.max(KEYS_W, this.width - 160)) + "px";
        input.style.top = RULER_H + 1 + "px";
        let done = false;
        const finish = (keep) => {
            if (done) return;
            if (keep) {
                const text = input.value.trim();
                if (!text) {
                    if (name) this.commit(roll.removeChord(this.model, start));
                } else if (text !== name) {
                    const changed = roll.setChord(this.model, start, text);
                    if (!changed) {
                        input.classList.add("yue2-s-bad");
                        this.setStatus(CHORD_HELP, true);
                        return;
                    }
                    this.commit(changed);
                }
            }
            done = true;
            this.closeChordInput();
            this.rollBox.focus();
        };
        input.addEventListener("keydown", (event) => {
            event.stopPropagation();
            if (event.key === "Enter") finish(true);
        });
        input.addEventListener("input", () => {
            input.classList.toggle("yue2-s-bad", Boolean(input.value.trim()) && !roll.isChord(input.value));
        });
        input.addEventListener("blur", () => setTimeout(() => finish(true), 0));
        this.chordInput = { input, finish };
        this.rollBox.appendChild(input);
        input.focus();
        input.select();
    }

    closeChordInput() {
        if (!this.chordInput) return;
        const { input } = this.chordInput;
        this.chordInput = null;
        input.remove();
    }

    commit(next, previous = this.model) {
        if (!next) {
            this.setStatus("That does not fit: each part sings one note at a time, inside the song.", true);
            this.draw();
            return false;
        }
        if (next === previous) return false;
        this.history.push(previous);
        this.model = next;
        this.localChanged = roll.changedBars(this.sheet, this.model);
        this.draw();
        this.scheduleWrite();
        return true;
    }

    undo() {
        if (!this.model) return;
        const previous = this.history.undo(this.model);
        if (!previous) return;
        this.model = previous;
        this.selection.clear();
        this.localChanged = roll.changedBars(this.sheet, this.model);
        this.draw();
        this.scheduleWrite();
    }

    redo() {
        if (!this.model) return;
        const next = this.history.redo(this.model);
        if (!next) return;
        this.model = next;
        this.selection.clear();
        this.localChanged = roll.changedBars(this.sheet, this.model);
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
            this.setStatus(this.describe());
        } else {
            this.setStatus(payload.error || "The edit could not be written into the score.", true);
            if (this.good && this.good !== this.model) {
                this.model = this.good;
                this.localChanged = roll.changedBars(this.sheet, this.model);
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
            this.setStatus(this.own ? "The box is empty: the node writes a new score on its next run."
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
        this.baseWords = typed.words || (text === String(this.planScore || "").trim() ? this.planWords : null);
        this.changed = [];
        this.history = new roll.History();
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

    startPlaying() {
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
            + "'Back to the model's score' loads it; the edit stays until then.");
    }

    reset() {
        if (this.planScore) {
            if (this.edited() && !window.confirm("Throw away the edits and load the score "
                + (this.own ? "this node" : "the plan node") + " wrote?")) return;
            this.fromBox = false;
            this.baseWords = this.planWords;
            this.load(this.planScore);
            return;
        }
        if (!this.fromBox) return;
        if (!window.confirm("Remove the edit kept on this node? The score it was made on has not arrived "
            + "since the page was opened, so the window stays empty until the next run.")) return;
        this.fromBox = false;
        this.baseWords = null;
        this.load("");
        this.setStatus("Apply removes the edit, and the next run " + (this.own
            ? "writes the model's score again." : "sings the plan's own score."));
    }

    escape() {
        if (this.chordInput) {
            this.closeChordInput();
            this.rollBox.focus();
            return;
        }
        if (this.drag) {
            this.cancelDrag();
            return;
        }
        if (this.selection.size && this.tab === "roll") {
            this.selection.clear();
            this.draw();
            return;
        }
        if (this.edited() && !window.confirm("Close the score editor and lose these changes?")) return;
        this.close();
    }

    async apply() {
        if (this.chordInput) this.chordInput.finish(true);
        if (this.readTimer) await this.readTyped();
        if (this.writePending) await this.write();
        if (this.isClosed) return;
        const typed = roll.splitMark(this.textBox.value).score;
        let text = (this.current || "").trim();
        if (this.typedProblem && typed !== text) {
            if (!window.confirm("The ABC text cannot be read note by note:\n\n" + this.typedProblem
                + "\n\nPut it on the node anyway? The node will try to sing it as it is.")) return;
            text = typed;
        }
        const model = String(this.planScore || "").trim();
        const value = text && text !== model ? roll.attachMark(text, this.baseWords) : "";
        setWidgetValue(this.node, SCORE, value);
        this.node.properties = this.node.properties || {};
        if (value) {
            const earlier = this.fromBox && this.base !== model ? this.node.properties.yue2_score_bars || [] : [];
            this.node.properties.yue2_score_bars = [...new Set([...earlier, ...this.changed])].sort((a, b) => a - b);
            this.node.properties.yue2_score_base = model ? hashText(model) : "";
        } else {
            delete this.node.properties.yue2_score_bars;
            delete this.node.properties.yue2_score_base;
        }
        paintScoreSummary(this.node);
        this.close();
    }
}

function openScoreEditor(node) {
    try {
        new ScoreEditor(node);
    } catch (error) {
        console.error("[YuE2] the score editor failed to open:", error);
        window.alert("The score editor could not open: " + (error?.message || error)
            + "\n\nThe score_abc box is still on the node's properties panel.");
    }
}

function resetScoreEdit(node) {
    const box = roll.splitMark(widgetNamed(node, SCORE)?.value ?? "");
    if (!box.score) return;
    const next = isGenerate(node)
        ? "The next run writes the model's score again, and a new seed gives a new tune."
        : "The next run sings the plan's own score.";
    if (!window.confirm("Throw away the edited score kept on this node?\n\n" + next)) return;
    setWidgetValue(node, SCORE, "");
    if (node.properties) {
        delete node.properties.yue2_score_bars;
        delete node.properties.yue2_score_base;
    }
    paintScoreSummary(node);
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
    const row = (...parts) => {
        const line = element("div");
        line.append(...parts);
        holder.appendChild(line);
    };
    const dim = (text) => element("span", "yue2-s-dim", text);
    const strong = (text) => element("span", "yue2-s-key", text);
    const warn = (text) => element("span", "yue2-s-warn", text);
    const own = isGenerate(node);
    const wired = sourceOf(node, SCORE);
    const box = roll.splitMark(widgetNamed(node, SCORE)?.value ?? "");
    if (node.__yue2ResetButton) node.__yue2ResetButton.hidden = Boolean(wired) || !box.score;
    if (wired) {
        row(dim("The score comes in through a wire from "), strong(wired.title || "another node"));
        row(dim(own ? "Sung as it arrives, whatever the words." : "Edit it where it is written."));
        if (node.__yue2ScoreButton) node.__yue2ScoreButton.disabled = true;
        return;
    }
    if (node.__yue2ScoreButton) node.__yue2ScoreButton.disabled = false;
    const origin = scoreSource(node)?.node ?? null;
    const planScore = origin?.__yue2Score || null;
    const planWords = origin?.__yue2Words || null;
    const facts = (abc) => {
        const found = roll.headerFacts(abc);
        return [found.key && "Key " + found.key, found.meter, found.bpm && found.bpm + " BPM",
            found.bars && found.bars + " bars"].filter(Boolean).join(" \u00B7 ");
    };
    if (box.score) {
        row(strong("Edited score"), dim(" \u00B7 " + facts(box.score)));
        const bars = node.properties?.yue2_score_bars || [];
        const base = node.properties?.yue2_score_base;
        const rewritten = bars.length ? "Bars " + barList(bars) + " rewritten" : "Sung as it stands";
        if (box.words && planWords && box.words !== planWords) {
            row(warn(own ? "Made for other words: this node writes a new score instead."
                : "Made for other words: the plan's own score is sung instead."));
        } else if (planScore && base && base !== hashText(String(planScore).trim())) {
            row(box.words ? dim(rewritten + " on an earlier take; still sung.")
                : warn("Edited on an earlier score; the plan has changed since."));
        } else if (own) {
            row(dim(rewritten + ". A new seed keeps these notes."));
        } else {
            row(dim(bars.length ? rewritten + "; the rest as the model wrote it." : "Sung as it stands in this node."));
        }
        return;
    }
    if (planScore) {
        row(strong("The model's score"), dim(" \u00B7 " + facts(planScore)));
        row(dim(own ? "Written by the model on each run. Edit score\u2026 to change notes."
            : "Sung exactly as written. Edit score\u2026 to change notes."));
        return;
    }
    if (own) {
        row(strong("No score yet"));
        row(dim("Run once, and the score this node writes can be edited here."));
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
        if (isRender(node) || isGenerate(node)) paintScoreSummary(node);
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
            { label: RESET_LABEL, tooltip: RESET_TOOLTIP_OWN, onClick: () => resetScoreEdit(node) },
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
    if (isGenerate(node)) {
        const edit = plainButton(EDIT_LABEL, EDIT_TOOLTIP, () => openScoreEditor(node));
        const reset = plainButton(RESET_LABEL, RESET_TOOLTIP_OWN, () => resetScoreEdit(node));
        reset.hidden = true;
        node.__yue2ScoreButton = edit;
        node.__yue2ResetButton = reset;
        panelWidget(node, SCORE_SUMMARY, summary, () => SONG_SUMMARY_H);
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

app.registerExtension({
    name: "yue2.score_editor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const plan = PLAN_NODES.includes(nodeData.name);
        const own = nodeData.name === GENERATE;
        if (plan || own) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const result = onExecuted?.apply(this, arguments);
                const score = message?.[SCORE_UI]?.[0];
                const words = message?.[WORDS_UI]?.[0];
                if (typeof score === "string") this.__yue2Score = score;
                if (typeof words === "string") this.__yue2Words = words;
                if (typeof score === "string" || typeof words === "string") {
                    window.dispatchEvent(new CustomEvent(SCORE_EVENT, { detail: { id: this.id, score, words } }));
                    refreshScoreSummaries();
                }
                return result;
            };
        }
        if (plan) {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function () {
                const result = onConnectionsChange?.apply(this, arguments);
                setTimeout(refreshScoreSummaries, 0);
                return result;
            };
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
