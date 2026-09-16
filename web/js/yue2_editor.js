import { app } from "../../scripts/app.js";
import {
    ask, buttonRow, element, frame, installStyle, panelWidget,
    setWidgetValue, showWidget, sourceOf, widgetNamed,
} from "./yue2_controls.js";
import { editValue, splitMark } from "./yue2_roll.js";
import * as sheet from "./yue2_sheet.js";

const TRANSCRIBE = "YuE2Transcribe";
const LOAD_MIDI = "YuE2LoadMidi";
const NODES = ["YuE2GenerateSong", "YuE2Plan", "YuE2PlanBatch", TRANSCRIBE, LOAD_MIDI];
const LYRICS_UI = "yue2_lyrics";
const TRACK_UI = "yue2_track";
const STYLE = "style";
const LYRICS = "lyrics";
const TOKEN_ROUTE = "/yue2/tokens";
const TOKEN_DELAY = 250;
const SUMMARY_H = 58;

const EDIT_LABEL = "Edit song\u2026";
const EDIT_TOOLTIP =
    "Open the song editor: build the style line from parts, and shape the lyrics " +
    "section by section. Nothing changes on the node until you press Apply.";

const LYRICS_LABEL = "Edit lyrics\u2026";
const LYRICS_TOOLTIP =
    "Open the lyrics editor on the lyrics this node gives: the section tags of the transcription, " +
    "with the recognised words under them when 'lyrics_auto_recognition' is on. Nothing changes on " +
    "the node until you press Apply.";
const RESET_LYRICS_LABEL = "Reset lyrics";
const RESET_LYRICS_TOOLTIP =
    "Throw away the lyrics kept on this node. The next run outputs the node's own lyrics again: " +
    "the section tags, or the recognised words.";
const TAGS_NOTE =
    "The tags are the sections the transcription found, in the order they are sung: one for each " +
    "section with a voice in it. Write the words under each tag. The edit is kept for this recording.";
const RECOGNISED_NOTE =
    "These words were recognised from the recording and laid out under its sections. Recognition " +
    "mishears some words, and a line can land in the section next door: read them through and fix " +
    "what is wrong. The edit is kept for this recording.";
const OTHER_RECORDING_NOTE =
    "These lyrics were written for another recording, so the node outputs its own lyrics for this " +
    "recording instead. Apply keeps them for this recording.";

const SOURCE_TEXTS = {
    [TRANSCRIBE]: {
        lyricsTooltip: LYRICS_TOOLTIP,
        resetTooltip: RESET_LYRICS_TOOLTIP,
        tagsNote: TAGS_NOTE,
        wordsNote: RECOGNISED_NOTE,
        otherNote: OTHER_RECORDING_NOTE,
        back: "Back to the transcription",
        backTitle: "Throw away these lyrics and load what the last run gave: the section tags, "
            + "or the recognised words.",
        empty: "No lyrics and no section tags yet. Run the node once and the tags of the transcription "
            + "appear here, or add a section and write the lyrics now.",
        otherRow: "Written for another recording: this recording's own lyrics are sent on instead.",
        wordsHeading: "Recognised lyrics",
        wordsRow: "Heard in the recording: read them through in Edit lyrics\u2026",
        tagsFound: " found in the transcription",
        noTags: "Run once, and the tags of the transcription appear here.",
    },
    [LOAD_MIDI]: {
        lyricsTooltip: "Open the lyrics editor on the lyrics this node gives: the karaoke words of the file under "
            + "their section tags, or the tags alone. Nothing changes on the node until you press Apply.",
        resetTooltip: "Throw away the lyrics kept on this node. The next run outputs the node's own lyrics again: "
            + "the file's karaoke words, or the section tags.",
        tagsNote: "The tags are the sections of the file: its markers, or one verse when it names none. Write the "
            + "words under each tag, and add sections where the song has them. The edit is kept for this file.",
        wordsNote: "These are the file's own karaoke words, in its lines and sections; a section sung twice is taken "
            + "for the chorus. Check the tags and fix what the file got wrong. The edit is kept for this file.",
        otherNote: "These lyrics were written for another file, so the node outputs its own lyrics for this file "
            + "instead. Apply keeps them for this file.",
        back: "Back to the file's lyrics",
        backTitle: "Throw away these lyrics and load what the last run gave: the file's karaoke words, "
            + "or the section tags.",
        empty: "No lyrics and no section tags yet. Run the node once and the file's words or tags appear here, "
            + "or add a section and write the lyrics now.",
        otherRow: "Written for another file: this file's own lyrics are sent on instead.",
        wordsHeading: "Karaoke lyrics",
        wordsRow: "The file's own words: read them through in Edit lyrics\u2026",
        tagsFound: " found in the file",
        noTags: "Run once, and the tags of the file appear here.",
    },
};

const CASE_NOTE =
    "Click a letter to flip its case; click again to flip it back. A capital inside " +
    "a word splits it where you clicked, and the thin marks under the letters show " +
    "where the YuE2 tokenizer really cuts. That is a change in what the model reads, " +
    "not a stress mark -- and like any change to the words, it gives a different song " +
    "on the same seed.";

const VOICES_NOTE =
    "The voices go into the style line as one part, joined with 'and'. Naming two voices " +
    "asks for both; it does not promise them or decide who sings where. Checked by ear on " +
    "one seed: with plain section tags the male voice never came in, and with " +
    "[Verse - breathy female voice] / [Chorus - male rap vocal], with meaningless tags of " +
    "the same length, and with the roles written into the style, it sang the second verse " +
    "every time -- the same place, whatever had been asked.";

const TAG_COLORS = {
    "Intro": "#8C96A3",
    "Verse": "#4E98C4",
    "Pre-Chorus": "#9B7FD1",
    "Chorus": "#D19A3F",
    "Hook": "#D9774B",
    "Bridge": "#4FA37A",
    "Refrain": "#C98B5B",
    "Outro": "#7C8FA6",
    "Instrumental": "#5FA3A3",
    "Interlude": "#5FA3A3",
};
const UNKNOWN_TAG = "#9A9A9A";

function tagColor(tag) {
    return TAG_COLORS[tag] || UNKNOWN_TAG;
}

function isTranscribe(node) {
    return node?.type === TRANSCRIBE || node?.comfyClass === TRANSCRIBE;
}

function isLoadMidi(node) {
    return node?.type === LOAD_MIDI || node?.comfyClass === LOAD_MIDI;
}

function sourceTexts(node) {
    return isTranscribe(node) ? SOURCE_TEXTS[TRANSCRIBE] : isLoadMidi(node) ? SOURCE_TEXTS[LOAD_MIDI] : null;
}

const EDITOR_STYLE_ID = "yue2-editor-style";
const EDITOR_STYLE = `
.yue2-title { font-size: 15px; font-weight: 600; margin: 0; }
.yue2-sub { font-size: 11px; color: var(--descrip-text, #999); margin: 2px 0 12px; }
.yue2-scroll { flex: 1 1 auto; overflow: auto; padding-right: 6px; }
.yue2-card { border: 1px solid var(--border-color, #4e4e4e); border-radius: 8px;
    padding: 12px 14px; margin-bottom: 14px; background: rgba(0, 0, 0, 0.12); }
.yue2-card h4 { margin: 0 0 10px; font-size: 12px; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--descrip-text, #999); display: flex;
    align-items: center; gap: 10px; }
.yue2-card h4 .yue2-grow { flex: 1 1 auto; }
.yue2-grid { display: grid; grid-template-columns: 92px 1fr; gap: 8px 12px;
    align-items: center; }
.yue2-label { font-size: 12px; color: var(--descrip-text, #999); }
.yue2-panel input[type="text"], .yue2-panel select, .yue2-panel textarea,
.yue2-panel input[type="number"] { box-sizing: border-box; font: inherit; font-size: 13px;
    padding: 5px 8px; border-radius: 6px; color: var(--input-text, #ddd);
    background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-panel input[type="text"], .yue2-panel select { width: 100%; }
.yue2-panel textarea { width: 100%; min-height: 320px; resize: vertical; line-height: 1.5;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.yue2-panel button { font: inherit; font-size: 12px; padding: 4px 10px; border-radius: 6px;
    cursor: pointer; color: var(--input-text, #ddd); background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-panel button:hover { background: var(--comfy-menu-bg, #353535);
    border-color: #6a6a6a; }
.yue2-panel button.yue2-go { background: #3B7DD8; border-color: #3B7DD8; color: #fff; }
.yue2-panel button[disabled] { opacity: 0.45; cursor: not-allowed; }
.yue2-panel button.yue2-icon { padding: 2px 7px; min-width: 26px; }

.yue2-tempo { display: flex; align-items: center; gap: 10px; }
.yue2-tempo input[type="range"] { flex: 1 1 auto; accent-color: #3B7DD8; }
.yue2-tempo input[type="number"] { width: 70px; }
.yue2-tempo .yue2-unit { font-size: 12px; color: var(--descrip-text, #999); }
.yue2-beat { width: 12px; height: 12px; border-radius: 50%; background: #3B7DD8;
    flex: 0 0 auto; animation-name: yue2-beat; animation-iteration-count: infinite;
    animation-timing-function: ease-out; }
.yue2-tempo.yue2-off .yue2-beat { animation: none; opacity: 0.2; }
.yue2-tempo.yue2-off input[type="range"], .yue2-tempo.yue2-off input[type="number"] {
    opacity: 0.45; }
@keyframes yue2-beat {
    0% { opacity: 1; transform: scale(1); }
    25% { opacity: 0.3; transform: scale(0.72); }
    100% { opacity: 0.3; transform: scale(0.72); } }
@media (prefers-reduced-motion: reduce) { .yue2-beat { animation: none; } }

.yue2-parts { display: flex; flex-direction: column; gap: 5px; }
.yue2-part { display: flex; gap: 5px; align-items: center; }
.yue2-part input[type="text"] { flex: 1 1 auto; }
.yue2-add { display: flex; gap: 6px; margin-top: 4px; }
.yue2-add input[type="text"] { flex: 1 1 auto; }
.yue2-line-out { font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 12px; }
.yue2-panel textarea.yue2-line-out { min-height: 0; line-height: 1.45; resize: vertical; }
.yue2-hint { font-size: 11px; line-height: 1.5; color: var(--descrip-text, #999); }
.yue2-tokens { font-size: 11px; color: var(--descrip-text, #999); text-transform: none;
    letter-spacing: 0; font-weight: 400; }
.yue2-tokens.yue2-warn { color: #E0A45A; }

.yue2-block { border-left: 3px solid var(--yue2-tag, #999); padding: 2px 0 6px 10px;
    margin: 10px 0 4px; }
.yue2-block-head { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
.yue2-tag { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 13px;
    font-weight: 600; padding: 2px 9px; border-radius: 999px; cursor: pointer;
    user-select: none; color: #fff; background: var(--yue2-tag, #999); }
.yue2-tag.yue2-none { background: transparent; color: var(--descrip-text, #999);
    border: 1px dashed var(--border-color, #4e4e4e); font-weight: 400; }
.yue2-tag:hover { filter: brightness(1.15); }
.yue2-block-count { font-size: 11px; color: var(--descrip-text, #999); }
.yue2-block-head .yue2-grow { flex: 1 1 auto; }

.yue2-row { display: flex; align-items: center; gap: 8px; min-height: 28px;
    border-radius: 5px; padding: 0 4px; }
.yue2-row:hover { background: rgba(255, 255, 255, 0.04); }
.yue2-row .yue2-acts { visibility: hidden; display: flex; gap: 3px; }
.yue2-row:hover .yue2-acts, .yue2-row.yue2-editing .yue2-acts { visibility: visible; }
.yue2-letters { flex: 1 1 auto; font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 15px; line-height: 24px; white-space: pre-wrap; word-break: break-word;
    cursor: text; }
.yue2-row input.yue2-line-edit { flex: 1 1 auto; font-size: 15px;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.yue2-count { flex: 0 0 auto; width: 44px; text-align: right; font-size: 10px;
    color: var(--descrip-text, #999); }
.yue2-ch { cursor: pointer; border-radius: 2px; padding: 1px 0; }
.yue2-ch.yue2-alt { background: rgba(127, 178, 245, 0.10); }
.yue2-ch.yue2-cut { box-shadow: inset 1.5px 0 0 rgba(127, 178, 245, 0.85); }
.yue2-ch.yue2-cap { color: #F0B35A; font-weight: 700; }
.yue2-ch.yue2-torn { text-decoration: underline dotted #E08A8A; }
.yue2-ch.yue2-flat { cursor: text; }
.yue2-ch:not(.yue2-flat):hover { background: rgba(59, 125, 216, 0.45); color: #fff; }
.yue2-gap { flex: 1 1 auto; border-top: 1px dashed var(--border-color, #4e4e4e);
    margin: 0 6px; height: 0; }
.yue2-gap-label { font-size: 10px; color: var(--descrip-text, #999); }
.yue2-block-foot { display: flex; gap: 6px; margin: 4px 0 0 4px; }
.yue2-empty { padding: 20px 4px; text-align: center; font-size: 12px;
    color: var(--descrip-text, #999); }

.yue2-foot { display: flex; gap: 8px; justify-content: flex-end; align-items: center;
    margin-top: 12px; }
.yue2-foot button { font-size: 13px; padding: 6px 16px; }
.yue2-problem { flex: 1 1 auto; font-size: 11px; color: #E08A8A; }

.yue2-menu { position: fixed; z-index: 1400; min-width: 190px; padding: 4px;
    background: var(--comfy-menu-bg, #353535); color: var(--input-text, #ddd);
    border: 1px solid var(--border-color, #4e4e4e); border-radius: 8px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45); font-family: system-ui, sans-serif;
    font-size: 12px; }
.yue2-menu-item { padding: 5px 10px; border-radius: 5px; cursor: pointer;
    display: flex; gap: 8px; align-items: center; white-space: nowrap; }
.yue2-menu-item:hover { background: #3B7DD8; color: #fff; }
.yue2-menu-item.yue2-danger { color: #E08A8A; }
.yue2-menu-item.yue2-danger:hover { background: #8A3B3B; color: #fff; }
.yue2-menu-item.yue2-disabled { opacity: 0.4; pointer-events: none; }
.yue2-menu-dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 auto; }
.yue2-menu-sep { height: 1px; margin: 4px 2px; background: var(--border-color, #4e4e4e); }

.yue2-summary { width: 100%; height: 100%; box-sizing: border-box; overflow: hidden;
    display: flex; flex-direction: column;
    padding: 5px 8px; border-radius: 6px; cursor: pointer;
    font-family: system-ui, sans-serif; font-size: 11px; line-height: 16px;
    color: var(--input-text, #ddd); background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-summary:hover { border-color: #3B7DD8; }
.yue2-sum-row { flex: 0 0 auto; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.yue2-sum-lyrics { flex: 1 1 auto; min-height: 0; overflow: hidden; margin-top: 6px;
    padding-top: 5px; border-top: 1px solid var(--border-color, #4e4e4e);
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11px;
    line-height: 15px; white-space: pre; color: var(--descrip-text, #999); }
.yue2-sum-lyrics .yue2-sum-tag { display: inline-block; margin-top: 4px; }
.yue2-sum-key { font-weight: 600; }
.yue2-sum-dim { color: var(--descrip-text, #999); }
.yue2-sum-tag { font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-weight: 600; margin-right: 6px; }
.yue2-sum-warn { color: #E0A45A; }
.yue2-hint + .yue2-hint { margin-top: 6px; }
.yue2-hint.yue2-warn-hint { color: #E0A45A; }
.yue2-buttons button[hidden] { display: none !important; }
`;

let MENU = null;

function closeMenu() {
    if (!MENU) return;
    MENU.remove();
    MENU = null;
    document.removeEventListener("pointerdown", menuOutside, true);
}

function menuOutside(event) {
    if (MENU && !MENU.contains(event.target)) closeMenu();
}

function openMenu(x, y, items) {
    closeMenu();
    const menu = element("div", "yue2-menu");
    for (const item of items) {
        if (item === "-") {
            menu.appendChild(element("div", "yue2-menu-sep"));
            continue;
        }
        const row = element("div", "yue2-menu-item"
            + (item.danger ? " yue2-danger" : "") + (item.disabled ? " yue2-disabled" : ""));
        if (item.color) {
            const dot = element("span", "yue2-menu-dot");
            dot.style.background = item.color;
            row.appendChild(dot);
        }
        row.appendChild(element("span", "", item.label));
        row.addEventListener("click", (event) => {
            event.stopPropagation();
            closeMenu();
            item.onClick();
        });
        menu.appendChild(row);
    }
    document.body.appendChild(menu);
    const box = menu.getBoundingClientRect();
    menu.style.left = Math.max(4, Math.min(x, window.innerWidth - box.width - 4)) + "px";
    menu.style.top = Math.max(4, Math.min(y, window.innerHeight - box.height - 4)) + "px";
    MENU = menu;
    setTimeout(() => document.addEventListener("pointerdown", menuOutside, true), 0);
}

function menuAt(button) {
    const box = button.getBoundingClientRect();
    return [box.left, box.bottom + 2];
}

function iconButton(text, title, onClick) {
    const button = element("button", "yue2-icon", text);
    button.title = title;
    button.addEventListener("click", (event) => {
        event.stopPropagation();
        onClick(button);
    });
    return button;
}

class SongEditor {
    constructor(node) {
        this.node = node;
        this.source = sourceTexts(node);
        this.lyricsOnly = Boolean(this.source);
        this.styleFree = !this.lyricsOnly && !sourceOf(node, STYLE);
        this.lyricsFree = !sourceOf(node, LYRICS);

        const kept = splitMark(widgetNamed(node, LYRICS)?.value ?? "");
        this.keptWords = kept.words;
        this.otherRecording = Boolean(this.lyricsOnly && kept.score && kept.words && node.__yue2Track
            && kept.words !== node.__yue2Track);
        this.styleStart = this.lyricsOnly ? "" : String(widgetNamed(node, STYLE)?.value ?? "");
        this.lyricsStart = this.lyricsOnly ? kept.score || String(node.__yue2Lyrics ?? "")
            : String(widgetNamed(node, LYRICS)?.value ?? "");
        this.styleRaw = this.styleStart;
        this.parts = sheet.parseStyle(this.styleRaw);
        this.styleDirty = false;
        this.lyricsRaw = this.lyricsStart;
        this.blocks = sheet.parseLyrics(this.lyricsRaw);
        this.lyricsDirty = false;

        this.asText = false;
        this.editing = null;
        this.tokens = null;
        this.sequence = 0;
        this.pending = null;
        this.closed = false;

        installStyle(EDITOR_STYLE_ID, EDITOR_STYLE);
        this.build();
    }

    styleText() {
        return this.styleDirty ? sheet.formatStyle(this.parts) : this.styleRaw;
    }

    lyricsText() {
        return this.lyricsDirty ? sheet.formatLyrics(this.blocks) : this.lyricsRaw;
    }

    changed() {
        return this.styleText() !== this.styleStart || this.lyricsText() !== this.lyricsStart;
    }

    build() {
        const { panel, close, handle } = frame({
            sticky: true,
            onClose: () => {
                this.closed = true;
                clearTimeout(this.pending);
                closeMenu();
            },
        });
        this.close = close;
        handle.onEscape = () => this.escape();

        panel.appendChild(element("h3", "yue2-title", (this.lyricsOnly ? "Lyrics \u2014 " : "Song \u2014 ")
            + (this.node.title || "YuE2")));
        panel.appendChild(element("p", "yue2-sub", this.lyricsOnly
            ? "The lyrics this node outputs. Nothing is written to it until Apply."
            : "Style and lyrics for this node. Nothing is written to it until Apply."));

        const scroll = element("div", "yue2-scroll");
        panel.appendChild(scroll);
        this.styleCard = element("div", "yue2-card");
        this.lyricsCard = element("div", "yue2-card");
        if (this.lyricsOnly) scroll.append(this.lyricsCard);
        else scroll.append(this.styleCard, this.lyricsCard);

        const foot = element("div", "yue2-foot");
        this.problem = element("div", "yue2-problem");
        const cancel = element("button", "", "Cancel");
        const apply = element("button", "yue2-go", "Apply");
        cancel.addEventListener("click", () => this.escape());
        apply.addEventListener("click", () => this.apply());
        foot.append(this.problem, cancel, apply);
        panel.appendChild(foot);

        panel.addEventListener("contextmenu", (event) => event.preventDefault());

        this.renderStyle();
        this.renderLyrics();
        this.scheduleTokens(0);
    }

    escape() {
        if (MENU) {
            closeMenu();
            return;
        }
        if (this.editing) {
            this.finishEdit(false);
            return;
        }
        if (this.changed() && !window.confirm("Close the song editor and lose these changes?")) {
            return;
        }
        this.close();
    }

    apply() {
        if (this.editing) this.finishEdit(true);
        if (this.lyricsOnly) {
            if (this.lyricsFree) {
                setWidgetValue(this.node, LYRICS, editValue(this.lyricsText(), this.node.__yue2Lyrics,
                    this.node.__yue2Track || this.keptWords));
            }
            paintSummary(this.node);
            this.close();
            return;
        }
        if (this.styleFree) setWidgetValue(this.node, STYLE, this.styleText());
        if (this.lyricsFree) setWidgetValue(this.node, LYRICS, this.lyricsText());
        paintSummary(this.node);
        this.close();
    }

    styleChanged(parts) {
        this.parts = parts;
        this.styleDirty = true;
        this.renderStyle();
        this.scheduleTokens();
    }

    renderStyle() {
        const card = this.styleCard;
        card.replaceChildren();
        const title = element("h4");
        title.append(element("span", "", "Style"));
        card.appendChild(title);

        if (!this.styleFree) {
            card.appendChild(element("div", "yue2-hint",
                "The style comes in through a wire from '" + (sourceOf(this.node, STYLE)?.title || "another node")
                + "', so it is written there, not here."));
            return;
        }

        const grid = element("div", "yue2-grid");
        card.appendChild(grid);

        grid.appendChild(element("div", "yue2-label", "Language"));
        const language = document.createElement("select");
        const named = sheet.languageOf(this.parts);
        for (const name of ["", ...sheet.LANGUAGES]) {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name || "(not named)";
            option.selected = name === named;
            language.appendChild(option);
        }
        language.title = "The language the song is sung in. The writer names it first, and so does this.";
        language.addEventListener("change", () => this.styleChanged(sheet.setLanguage(this.parts, language.value)));
        grid.appendChild(language);

        grid.appendChild(element("div", "yue2-label", "Tempo"));
        grid.appendChild(this.tempoControl());

        grid.appendChild(element("div", "yue2-label", "Voices"));
        grid.appendChild(this.voicesControl());
        grid.appendChild(datalist("yue2-voices", sheet.VOICES));

        grid.appendChild(element("div", "yue2-label", "Sound"));
        grid.appendChild(this.partsControl());

        grid.appendChild(element("div", "yue2-label", "Style line"));
        const line = document.createElement("textarea");
        line.className = "yue2-line-out";
        line.rows = 4;
        line.spellcheck = false;
        line.value = this.styleText();
        line.title = "What the model reads, as one line. Edit it here directly when that is "
            + "quicker; drag the corner for more room.";
        line.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                line.blur();
            }
        });
        line.addEventListener("change", () => {
            line.value = sheet.oneLine(line.value);
            this.styleRaw = line.value;
            this.parts = sheet.parseStyle(line.value);
            this.styleDirty = false;
            this.renderStyle();
            this.scheduleTokens();
        });
        grid.appendChild(line);
    }

    tempoControl() {
        const bpm = sheet.bpmOf(this.parts);
        const on = bpm !== null;
        const shown = on ? bpm : sheet.BPM_DEFAULT;

        const row = element("div", "yue2-tempo" + (on ? "" : " yue2-off"));
        const toggle = document.createElement("input");
        toggle.type = "checkbox";
        toggle.checked = on;
        toggle.title = on ? "Leave the tempo out and let YuE2 pick one" : "Name a tempo in the style line";

        const beat = element("span", "yue2-beat");
        beat.title = "Blinks at the chosen tempo";
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = sheet.BPM_MIN;
        slider.max = sheet.BPM_MAX;
        slider.value = shown;
        const number = document.createElement("input");
        number.type = "number";
        number.min = sheet.BPM_MIN;
        number.max = sheet.BPM_MAX;
        number.value = shown;

        const pace = (value) => {
            beat.style.animationDuration = (60 / Number(value)).toFixed(3) + "s";
        };
        pace(shown);

        slider.addEventListener("input", () => {
            number.value = slider.value;
            pace(slider.value);
        });
        slider.addEventListener("change", () => this.styleChanged(sheet.setBpm(this.parts, slider.value)));
        number.addEventListener("change", () => this.styleChanged(sheet.setBpm(this.parts, number.value)));
        toggle.addEventListener("change", () =>
            this.styleChanged(sheet.setBpm(this.parts, toggle.checked ? slider.value : null)));
        slider.disabled = !on;
        number.disabled = !on;

        row.append(toggle, beat, slider, number, element("span", "yue2-unit", "BPM"));
        return row;
    }

    voicesControl() {
        const holder = element("div", "yue2-parts");
        const voices = sheet.voicesOf(this.parts);
        const shown = voices.length ? [...voices] : [""];
        if (this.pendingVoice && voices.length) shown.push("");

        shown.forEach((voice, index) => {
            const row = element("div", "yue2-part");
            const text = document.createElement("input");
            text.type = "text";
            text.value = voice;
            text.placeholder = index ? "female melodic vocals" : "expressive female voice";
            text.setAttribute("list", "yue2-voices");
            text.addEventListener("change", () => {
                const next = [...shown];
                next[index] = text.value;
                this.pendingVoice = false;
                this.styleChanged(sheet.setVoices(this.parts, next));
            });
            row.appendChild(text);
            if (voice || shown.length > 1) {
                row.appendChild(iconButton("\u2715", "Remove this voice", () => {
                    this.pendingVoice = false;
                    this.styleChanged(sheet.setVoices(this.parts, shown.filter((_, at) => at !== index)));
                }));
            }
            holder.appendChild(row);
            if (this.pendingVoice && !voice && index === shown.length - 1) {
                setTimeout(() => text.focus(), 0);
            }
        });

        const more = element("button", "", "+ Voice");
        more.title = "Name another voice: a rapper and a singer, a duet, a choir behind the lead";
        more.disabled = !voices.length || this.pendingVoice;
        more.addEventListener("click", () => {
            this.pendingVoice = true;
            this.renderStyle();
        });
        const foot = element("div", "yue2-add");
        foot.appendChild(more);
        holder.appendChild(foot);

        if (voices.length > 1) {
            holder.appendChild(element("div", "yue2-hint", VOICES_NOTE));
        }
        return holder;
    }

    partsControl() {
        const holder = element("div", "yue2-parts");
        this.parts.forEach((part, index) => {
            if (part.kind !== "other") return;
            const row = element("div", "yue2-part");
            const text = document.createElement("input");
            text.type = "text";
            text.value = part.text;
            text.addEventListener("change", () => this.styleChanged(sheet.editPart(this.parts, index, text.value)));
            row.append(
                text,
                iconButton("\u2191", "Earlier in the line", () => this.styleChanged(sheet.movePart(this.parts, index, -1))),
                iconButton("\u2193", "Later in the line", () => this.styleChanged(sheet.movePart(this.parts, index, 1))),
                iconButton("\u2715", "Remove", () => this.styleChanged(sheet.removePart(this.parts, index))),
            );
            holder.appendChild(row);
        });

        const add = element("div", "yue2-add");
        const input = document.createElement("input");
        input.type = "text";
        input.placeholder = "staccato phrasing, string section, lo-fi texture\u2026";
        input.setAttribute("list", "yue2-sounds");
        const commit = () => {
            if (!input.value.trim()) return;
            this.styleChanged(sheet.addPart(this.parts, input.value));
            this.styleCard.querySelector(".yue2-add input")?.focus();
        };
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") commit();
        });
        const button = element("button", "", "Add");
        button.addEventListener("click", commit);
        add.append(input, button, datalist("yue2-sounds", sheet.SUGGESTIONS));
        holder.appendChild(add);
        return holder;
    }

    lyricsChanged(blocks, keep) {
        this.blocks = blocks;
        this.lyricsDirty = true;
        this.renderLyrics(keep);
        this.scheduleTokens();
    }

    scheduleTokens(delay = TOKEN_DELAY) {
        clearTimeout(this.pending);
        this.pending = setTimeout(() => this.fetchTokens(), delay);
    }

    async fetchTokens() {
        if (!this.lyricsFree) return;
        const sequence = ++this.sequence;
        const lyrics = sheet.formatLyrics(this.blocks);
        let found;
        if (!lyrics.trim()) {
            found = { available: false, problem: "" };
        } else {
            try {
                const { ok, status, payload } = await ask(TOKEN_ROUTE, { style: this.styleText(), lyrics });
                found = ok ? payload : { available: false, problem: payload.error || "The token route answered " + status + "." };
            } catch (error) {
                found = {
                    available: false,
                    problem: "Token cuts are unavailable until ComfyUI is restarted after installing or updating the pack.",
                };
            }
        }
        if (this.closed || sequence !== this.sequence) return;
        this.tokens = found;
        if (this.editing) this.stale = true;
        else this.renderLyrics();
    }

    lineTokens(start, length) {
        const t = this.tokens;
        if (!t || !t.available) return null;
        const inside = (at) => at > start && at < start + length;
        return {
            cuts: new Set(t.cuts.filter(inside).map((at) => at - start)),
            torn: new Set(t.torn.filter((at) => at >= start && at < start + length).map((at) => at - start)),
        };
    }

    renderLyrics(keep) {
        const card = this.lyricsCard;
        const scroller = card.parentElement;
        const top = scroller?.scrollTop ?? 0;
        card.replaceChildren();
        this.stale = false;

        const title = element("h4");
        title.append(element("span", "", "Lyrics"));
        const status = element("span", "yue2-tokens");
        title.append(status, element("span", "yue2-grow"));
        card.appendChild(title);

        if (!this.lyricsFree) {
            card.appendChild(element("div", "yue2-hint",
                "The lyrics come in through a wire from '" + (sourceOf(this.node, LYRICS)?.title || "another node")
                + "', so they are written there, not here."));
            return;
        }

        const addSection = element("button", "", "+ Section");
        addSection.title = "Add a section at the end";
        addSection.addEventListener("click", (event) => {
            event.stopPropagation();
            const [x, y] = menuAt(addSection);
            openMenu(x, y, sheet.TAGS.map((tag) => ({
                label: "[" + tag + "]",
                color: tagColor(tag),
                onClick: () => this.lyricsChanged(sheet.addBlock(this.blocks, this.blocks.length - 1, tag),
                    { b: this.blocks.length, l: 0 }),
            })));
        });
        const textMode = element("button", "", this.asText ? "Back to sections" : "Edit as text");
        textMode.title = "Switch between the section view and the plain text the model reads";
        textMode.addEventListener("click", () => this.toggleText());
        title.append(addSection, textMode);
        if (this.lyricsOnly) {
            const back = element("button", "", this.source.back);
            back.title = this.source.backTitle;
            back.disabled = !this.node.__yue2Lyrics;
            back.addEventListener("click", () => this.backToTags());
            title.append(back);
        }

        if (this.asText) {
            const area = document.createElement("textarea");
            area.value = this.lyricsText();
            area.spellcheck = false;
            area.addEventListener("input", () => {
                this.lyricsRaw = area.value;
                this.blocks = sheet.parseLyrics(area.value);
                this.lyricsDirty = false;
            });
            this.textArea = area;
            card.appendChild(area);
            status.textContent = "";
            area.focus();
            return;
        }

        if (this.otherRecording) card.appendChild(element("div", "yue2-hint yue2-warn-hint", this.source.otherNote));
        if (this.lyricsOnly) {
            card.appendChild(element("div", "yue2-hint", hasSungLines(this.node.__yue2Lyrics)
                ? this.source.wordsNote : this.source.tagsNote));
        }
        card.appendChild(element("div", "yue2-hint", CASE_NOTE));
        this.paintStatus(status);

        if (!this.blocks.length) {
            card.appendChild(element("div", "yue2-empty", this.lyricsOnly
                ? this.source.empty
                : "No lyrics: the song will be instrumental. Add a section to write some."));
        }

        const lay = sheet.layout(this.blocks);
        const starts = new Map(lay.lines.map((entry) => [entry.block + ":" + entry.line, entry.start]));
        this.blocks.forEach((block, b) => {
            card.appendChild(this.blockView(block, b, starts, keep));
        });
        if (scroller) scroller.scrollTop = top;
    }

    paintStatus(status) {
        const t = this.tokens;
        status.classList.remove("yue2-warn");
        if (!t) {
            status.textContent = "reading the tokenizer\u2026";
        } else if (t.available) {
            status.textContent = t.tokens + " tokens in the lyrics";
        } else if (t.problem) {
            status.textContent = t.problem;
            status.classList.add("yue2-warn");
        } else {
            status.textContent = "";
        }
    }

    blockView(block, b, starts, keep) {
        const view = element("div", "yue2-block");
        view.style.setProperty("--yue2-tag", tagColor(block.tag));

        const head = element("div", "yue2-block-head");
        const chip = element("span", "yue2-tag" + (block.tag === null ? " yue2-none" : ""),
            block.tag === null ? "no section" : sheet.header(block));
        chip.title = block.tag === null
            ? "Lines before the first section header. Click to give them one."
            : "Click for the next section type; right-click for all of them";
        chip.addEventListener("click", (event) => {
            event.stopPropagation();
            this.lyricsChanged(sheet.setTag(this.blocks, b, sheet.cycleTag(block.tag)));
        });
        chip.addEventListener("contextmenu", (event) => {
            event.preventDefault();
            event.stopPropagation();
            this.tagMenu(event.clientX, event.clientY, b);
        });
        const sung = block.lines.filter(Boolean).length;
        head.append(chip, element("span", "yue2-block-count", sung + (sung === 1 ? " line" : " lines")),
            element("span", "yue2-grow"));
        head.append(
            iconButton("\u2191", "Move this section up", () => this.lyricsChanged(sheet.moveBlock(this.blocks, b, -1))),
            iconButton("\u2193", "Move this section down", () => this.lyricsChanged(sheet.moveBlock(this.blocks, b, 1))),
            iconButton("\u22EF", "More", (button) => {
                const [x, y] = menuAt(button);
                this.blockMenu(x, y, b);
            }),
            iconButton("\u2715", "Delete this section", () => this.lyricsChanged(sheet.removeBlock(this.blocks, b))),
        );
        view.appendChild(head);

        block.lines.forEach((line, l) => {
            view.appendChild(this.lineView(line, b, l, starts.get(b + ":" + l), keep));
        });

        const foot = element("div", "yue2-block-foot");
        const addLine = element("button", "", "+ Line");
        addLine.addEventListener("click", () => this.startNewLine(b, block.lines.length));
        foot.appendChild(addLine);
        view.appendChild(foot);
        return view;
    }

    lineView(line, b, l, start, keep) {
        const row = element("div", "yue2-row");
        row.addEventListener("contextmenu", (event) => {
            event.preventDefault();
            event.stopPropagation();
            this.lineMenu(event.clientX, event.clientY, b, l);
        });

        const isEditing = this.editing && this.editing.b === b && this.editing.l === l;
        if (isEditing) {
            row.classList.add("yue2-editing");
            row.appendChild(this.editInput(line));
        } else if (!line) {
            row.append(element("span", "yue2-gap-label", "break"), element("span", "yue2-gap"));
            row.title = "A break between stanzas. Double-click to write a line here instead.";
            row.addEventListener("dblclick", () => this.startEdit(b, l));
        } else {
            row.appendChild(this.letters(line, b, l, start));
        }

        const count = element("span", "yue2-count");
        const chars = sheet.points(line);
        const cuts = line && start !== undefined ? this.lineTokens(start, chars.length) : null;
        if (cuts) count.textContent = cuts.cuts.size + 1 + " tok";
        row.appendChild(count);

        const acts = element("div", "yue2-acts");
        acts.append(
            iconButton("\u270E", "Edit this line (or double-click it)", () => this.startEdit(b, l)),
            iconButton("+", "Add a line below", () => this.startNewLine(b, l + 1)),
            iconButton("\u22EF", "More", (button) => {
                const [x, y] = menuAt(button);
                this.lineMenu(x, y, b, l);
            }),
            iconButton("\u2715", line ? "Delete this line" : "Delete this break",
                () => this.lyricsChanged(sheet.removeLine(this.blocks, b, l))),
        );
        row.appendChild(acts);

        if (keep && keep.b === b && keep.l === l) {
            setTimeout(() => row.scrollIntoView({ block: "nearest" }), 0);
        }
        return row;
    }

    letters(line, b, l, start) {
        const holder = element("span", "yue2-letters");
        holder.title = "Click a letter to flip its case. Double-click the line to edit it.";
        const chars = sheet.points(line);
        const tokens = start === undefined ? null : this.lineTokens(start, chars.length);
        let alternate = false;
        chars.forEach((char, index) => {
            if (tokens && tokens.cuts.has(index)) alternate = !alternate;
            const span = element("span", "yue2-ch", char);
            const cased = char.toUpperCase() !== char.toLowerCase();
            if (!cased) span.classList.add("yue2-flat");
            if (alternate) span.classList.add("yue2-alt");
            if (tokens && tokens.cuts.has(index)) span.classList.add("yue2-cut");
            if (tokens && tokens.torn.has(index)) {
                span.classList.add("yue2-torn");
                span.title = "A token boundary runs through this character";
            }
            if (sheet.isInnerCapital(line, index)) span.classList.add("yue2-cap");
            if (cased) {
                span.addEventListener("click", (event) => {
                    event.stopPropagation();
                    const current = this.blocks[b]?.lines[l] ?? line;
                    if (event.detail === 2) {
                        this.blocks = sheet.setLine(this.blocks, b, l, sheet.toggleCase(current, index));
                        this.startEdit(b, l);
                        return;
                    }
                    if (event.detail > 2) return;
                    this.lyricsChanged(sheet.setLine(this.blocks, b, l, sheet.toggleCase(current, index)));
                });
            }
            holder.appendChild(span);
        });
        holder.addEventListener("dblclick", (event) => {
            event.preventDefault();
            this.startEdit(b, l);
        });
        return holder;
    }

    editInput(line) {
        const input = document.createElement("input");
        input.type = "text";
        input.className = "yue2-line-edit";
        input.value = line;
        input.placeholder = "Type the line; Enter for the next one, Escape to cancel";
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                this.finishEdit(true, true);
            } else if (event.key === "Backspace" && !input.value && this.editing) {
                event.preventDefault();
                const { b, l } = this.editing;
                this.editing = null;
                const blocks = sheet.removeLine(this.blocks, b, l);
                if (l > 0) this.editing = { b, l: l - 1, fresh: false };
                this.lyricsChanged(blocks, { b, l: Math.max(0, l - 1) });
            }
        });
        input.addEventListener("blur", () => {
            setTimeout(() => {
                if (this.editing && this.editing.input === input) this.finishEdit(true);
            }, 0);
        });
        setTimeout(() => {
            input.focus();
            input.setSelectionRange(input.value.length, input.value.length);
        }, 0);
        if (this.editing) this.editing.input = input;
        return input;
    }

    startEdit(b, l) {
        if (this.editing) this.finishEdit(true);
        if (!this.blocks[b] || l >= this.blocks[b].lines.length) return;
        this.editing = { b, l, fresh: false };
        this.renderLyrics({ b, l });
    }

    startNewLine(b, l) {
        if (this.editing) this.finishEdit(true);
        if (!this.blocks[b]) return;
        this.editing = { b, l, fresh: true };
        this.lyricsChanged(sheet.insertLine(this.blocks, b, l, ""), { b, l });
    }

    finishEdit(save, next = false) {
        const editing = this.editing;
        if (!editing) return;
        this.editing = null;
        const typed = editing.input ? editing.input.value : this.blocks[editing.b].lines[editing.l];
        let blocks = this.blocks;
        const empty = !typed.trim();
        if (editing.fresh && (empty || !save)) {
            blocks = sheet.removeLine(blocks, editing.b, editing.l);
            this.lyricsChanged(blocks);
            return;
        }
        if (save) blocks = sheet.setLine(blocks, editing.b, editing.l, typed);
        if (next) {
            this.editing = { b: editing.b, l: editing.l + 1, fresh: true };
            blocks = sheet.insertLine(blocks, editing.b, editing.l + 1, "");
        }
        if (save || next) this.lyricsChanged(blocks, { b: editing.b, l: editing.l + (next ? 1 : 0) });
        else this.renderLyrics();
    }

    backToTags() {
        const tags = String(this.node.__yue2Lyrics || "");
        if (!tags) return;
        if (this.editing) this.finishEdit(true);
        if (this.lyricsText().trim() !== tags.trim()
            && !window.confirm("Throw away these lyrics and load what the last run gave?")) return;
        this.asText = false;
        this.textArea = null;
        this.lyricsRaw = tags;
        this.blocks = sheet.parseLyrics(tags);
        this.lyricsDirty = false;
        this.otherRecording = false;
        this.renderLyrics();
        this.scheduleTokens(0);
    }

    toggleText() {
        if (this.editing) this.finishEdit(true);
        if (this.asText) {
            const value = this.textArea ? this.textArea.value : this.lyricsText();
            if (value !== this.lyricsText()) {
                this.lyricsRaw = value;
                this.blocks = sheet.parseLyrics(value);
                this.lyricsDirty = false;
            }
            this.asText = false;
            this.textArea = null;
            this.renderLyrics();
            this.scheduleTokens(0);
            return;
        }
        this.asText = true;
        this.renderLyrics();
    }

    tagMenu(x, y, b) {
        const current = this.blocks[b]?.tag;
        openMenu(x, y, sheet.TAGS.map((tag) => ({
            label: "[" + tag + "]" + (tag === current ? "  \u2713" : ""),
            color: tagColor(tag),
            onClick: () => this.lyricsChanged(sheet.setTag(this.blocks, b, tag)),
        })));
    }

    blockMenu(x, y, b) {
        openMenu(x, y, [
            { label: "Change section type\u2026", onClick: () => this.tagMenu(x, y, b) },
            { label: "Add a section after", onClick: () => this.addAfterMenu(x, y, b) },
            { label: "Duplicate section", onClick: () => this.lyricsChanged(sheet.duplicateBlock(this.blocks, b), { b: b + 1, l: 0 }) },
            "-",
            { label: "Move up", disabled: b === 0, onClick: () => this.lyricsChanged(sheet.moveBlock(this.blocks, b, -1)) },
            { label: "Move down", disabled: b === this.blocks.length - 1, onClick: () => this.lyricsChanged(sheet.moveBlock(this.blocks, b, 1)) },
            "-",
            { label: "Delete section", danger: true, onClick: () => this.lyricsChanged(sheet.removeBlock(this.blocks, b)) },
        ]);
    }

    addAfterMenu(x, y, b) {
        openMenu(x, y, sheet.TAGS.map((tag) => ({
            label: "[" + tag + "]",
            color: tagColor(tag),
            onClick: () => this.lyricsChanged(sheet.addBlock(this.blocks, b, tag), { b: b + 1, l: 0 }),
        })));
    }

    lineMenu(x, y, b, l) {
        const line = this.blocks[b]?.lines[l] ?? "";
        const moved = (step) => {
            const result = sheet.moveLine(this.blocks, b, l, step);
            this.lyricsChanged(result.blocks, { b: result.b, l: result.l });
        };
        openMenu(x, y, [
            { label: "Edit line", onClick: () => this.startEdit(b, l) },
            { label: "Add a line below", onClick: () => this.startNewLine(b, l + 1) },
            { label: "Duplicate line", onClick: () => this.lyricsChanged(sheet.duplicateLine(this.blocks, b, l), { b, l: l + 1 }) },
            { label: "Add a stanza break below", onClick: () => this.lyricsChanged(sheet.insertLine(this.blocks, b, l + 1, "")) },
            "-",
            { label: "Duplicate stanza", disabled: !line, onClick: () => this.lyricsChanged(sheet.duplicateStanza(this.blocks, b, l)) },
            { label: "Delete stanza", disabled: !line, danger: true, onClick: () => this.lyricsChanged(sheet.removeStanza(this.blocks, b, l)) },
            "-",
            { label: "Move up", onClick: () => moved(-1) },
            { label: "Move down", onClick: () => moved(1) },
            "-",
            { label: line ? "Delete line" : "Delete break", danger: true, onClick: () => this.lyricsChanged(sheet.removeLine(this.blocks, b, l)) },
        ]);
    }
}

function datalist(id, values) {
    const list = document.createElement("datalist");
    list.id = id;
    for (const value of values) {
        const option = document.createElement("option");
        option.value = value;
        list.appendChild(option);
    }
    return list;
}

function openEditor(node) {
    try {
        new SongEditor(node);
    } catch (error) {
        console.error("[YuE2] the song editor failed to open:", error);
        window.alert("The song editor could not open: " + (error?.message || error)
            + "\n\nThe style and lyrics text boxes are still on the node's properties panel.");
    }
}

function paintSummary(node) {
    const holder = node.__yue2Summary;
    if (!holder) return;
    holder.replaceChildren();
    if (sourceTexts(node)) {
        paintLyricsSummary(node, holder);
        return;
    }

    const styleFrom = sourceOf(node, STYLE);
    const lyricsFrom = sourceOf(node, LYRICS);
    if (styleFrom && lyricsFrom) {
        holder.append(
            line([dim("Style and lyrics come in through wires from")]),
            line([key(styleFrom.title === lyricsFrom.title ? styleFrom.title : styleFrom.title + " and " + lyricsFrom.title)]),
            line([dim("Edit them where they are written.")]),
        );
        node.__yue2Button && (node.__yue2Button.disabled = true);
        return;
    }
    node.__yue2Button && (node.__yue2Button.disabled = false);

    const facts = sheet.summarize(String(widgetNamed(node, STYLE)?.value ?? ""),
        String(widgetNamed(node, LYRICS)?.value ?? ""));

    if (styleFrom) {
        holder.appendChild(line([dim("style from "), key(styleFrom.title)]));
    } else {
        const head = [];
        const put = (text) => {
            if (head.length) head.push(dim(" \u00B7 "));
            head.push(key(text));
        };
        if (facts.language) put(facts.language);
        if (facts.bpm !== null) put(facts.bpm + " BPM");
        if (facts.voice) put(facts.voice);
        if (!head.length) head.push(dim("no language, tempo or voice named"));
        holder.appendChild(line(head));
        holder.appendChild(line([dim(facts.others.join(", ") || "no other style parts")]));
    }

    if (lyricsFrom) {
        holder.appendChild(line([dim("lyrics from "), key(lyricsFrom.title)]));
    } else if (!facts.sections.length) {
        holder.appendChild(line([dim("no lyrics: instrumental")]));
    } else {
        const tags = facts.sections.map((section) => {
            const span = element("span", "yue2-sum-tag",
                (section.name ? "[" + section.name + "]" : "[ ]") + " " + section.lines);
            span.style.color = tagColor(section.tag);
            return span;
        });
        holder.appendChild(line(tags));
        holder.appendChild(lyricsPreview(String(widgetNamed(node, LYRICS)?.value ?? "")));
    }

    function line(children) {
        const row = element("div", "yue2-sum-row");
        row.append(...children);
        return row;
    }
    function key(text) {
        return element("span", "yue2-sum-key", text);
    }
    function dim(text) {
        return element("span", "yue2-sum-dim", text);
    }
}

function hasSungLines(text) {
    return sheet.summarize("", String(text || "")).sung > 0;
}

function paintLyricsSummary(node, holder) {
    const texts = sourceTexts(node);
    const row = (...children) => {
        const made = element("div", "yue2-sum-row");
        made.append(...children);
        holder.appendChild(made);
    };
    const strong = (text) => element("span", "yue2-sum-key", text);
    const faint = (text) => element("span", "yue2-sum-dim", text);
    const wired = sourceOf(node, LYRICS);
    const box = splitMark(widgetNamed(node, LYRICS)?.value ?? "");
    if (node.__yue2ResetLyrics) node.__yue2ResetLyrics.hidden = Boolean(wired) || !box.score;
    if (node.__yue2Button) node.__yue2Button.disabled = Boolean(wired);
    if (wired) {
        row(faint("The lyrics come in through a wire from "), strong(wired.title || "another node"));
        row(faint("Sent on as they arrive."));
        return;
    }
    const track = node.__yue2Track || null;
    const shown = box.score || String(node.__yue2Lyrics || "");
    const facts = sheet.summarize("", shown);
    const named = facts.sections.length + (facts.sections.length === 1 ? " section" : " sections");
    const count = named + ", " + facts.sung + (facts.sung === 1 ? " line" : " lines");
    if (box.score) {
        row(strong("Edited lyrics"), faint(" \u00B7 " + count));
        row(box.words && track && box.words !== track
            ? element("span", "yue2-sum-warn", texts.otherRow)
            : faint("Sent on instead of the node's own lyrics."));
    } else if (facts.sung > 0) {
        row(strong(texts.wordsHeading), faint(" \u00b7 " + count));
        row(faint(texts.wordsRow));
    } else if (shown) {
        row(strong("Section tags"), faint(" \u00B7 " + named + texts.tagsFound));
        row(faint("Edit lyrics\u2026 to write the words under them."));
    } else {
        row(strong("No section tags yet"));
        row(faint(texts.noTags));
        return;
    }
    holder.appendChild(lyricsPreview(shown));
}

function resetLyrics(node) {
    const box = splitMark(widgetNamed(node, LYRICS)?.value ?? "");
    if (!box.score) return;
    if (!window.confirm("Throw away the lyrics kept on this node?\n\n"
        + "The next run outputs the node's own lyrics again.")) return;
    setWidgetValue(node, LYRICS, "");
    paintSummary(node);
}

function install(node) {
    installStyle(EDITOR_STYLE_ID, EDITOR_STYLE);
    const texts = sourceTexts(node);
    const lyricsOnly = Boolean(texts);
    const { buttons } = buttonRow(node, "yue2_song_edit", lyricsOnly ? [
        { label: LYRICS_LABEL, tooltip: texts?.lyricsTooltip, onClick: () => openEditor(node) },
        { label: RESET_LYRICS_LABEL, tooltip: texts?.resetTooltip, onClick: () => resetLyrics(node) },
    ] : [
        { label: EDIT_LABEL, tooltip: EDIT_TOOLTIP, onClick: () => openEditor(node) },
    ]);
    node.__yue2Button = buttons[0];
    if (lyricsOnly) {
        node.__yue2ResetLyrics = buttons[1];
        buttons[1].hidden = true;
    }

    const summary = element("div", "yue2-summary");
    summary.title = lyricsOnly ? "Click to open the lyrics editor" : "Click to open the song editor";
    summary.addEventListener("click", () => {
        if (!node.__yue2Button?.disabled) openEditor(node);
    });
    node.__yue2Summary = summary;
    panelWidget(node, "yue2_song_summary", summary, () => SUMMARY_H, true);

    showWidget(node, STYLE, false);
    showWidget(node, LYRICS, false);
    unbindSockets(node);

    for (const name of [STYLE, LYRICS]) {
        const widget = widgetNamed(node, name);
        if (!widget) continue;
        const original = widget.callback;
        widget.callback = function () {
            const result = original?.apply(this, arguments);
            paintSummary(node);
            return result;
        };
    }
    paintSummary(node);
}

function unbindSockets(node) {
    for (const input of node.inputs || []) {
        if ((input.name === STYLE || input.name === LYRICS) && input.widget) {
            input.widget = undefined;
        }
    }
    node.setDirtyCanvas?.(true, true);
}

function rebindOnSave(saved) {
    for (const input of saved?.inputs || []) {
        if ((input.name === STYLE || input.name === LYRICS) && !input.widget) {
            input.widget = { name: input.name };
        }
    }
}

const PROGRESS_CAPTION = "$$node-text-preview";
const PROGRESS_CAPTION_H = 58;

function capProgressCaption(node) {
    const caption = node.widgets?.find((w) => w.name === PROGRESS_CAPTION);
    if (!caption || caption.__yue2Capped) return;
    caption.options = caption.options || {};
    const least = caption.options.getMinHeight;
    caption.options.getMaxHeight = () => (least ? least() : PROGRESS_CAPTION_H);
    caption.__yue2Capped = true;
    node.setDirtyCanvas?.(true, true);
}

function lyricsPreview(lyrics) {
    const holder = element("div", "yue2-sum-lyrics");
    for (const block of sheet.parseLyrics(lyrics)) {
        if (block.tag !== null) {
            const tag = element("span", "yue2-sum-tag", sheet.header(block));
            tag.style.color = tagColor(block.tag);
            holder.append(tag, "\n");
        }
        holder.append(block.lines.join("\n") + "\n");
    }
    return holder;
}

app.registerExtension({
    name: "yue2.song_editor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODES.includes(nodeData.name)) return;

        if (nodeData.name === TRANSCRIBE || nodeData.name === LOAD_MIDI) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const result = onExecuted?.apply(this, arguments);
                const tags = message?.[LYRICS_UI]?.[0];
                const track = message?.[TRACK_UI]?.[0];
                if (typeof tags === "string") this.__yue2Lyrics = tags;
                if (typeof track === "string") this.__yue2Track = track;
                if (typeof tags === "string" || typeof track === "string") paintSummary(this);
                return result;
            };
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                install(this);
            } catch (error) {
                console.error("[YuE2] the song editor could not be added to this node:", error);
                showWidget(this, STYLE, true);
                showWidget(this, LYRICS, true);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            unbindSockets(this);
            setTimeout(() => paintSummary(this), 0);
            return result;
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            capProgressCaption(this);
            return onDrawForeground?.apply(this, arguments);
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (saved) {
            const result = onSerialize?.apply(this, arguments);
            rebindOnSave(saved);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            setTimeout(() => paintSummary(this), 0);
            return result;
        };
    },
});
