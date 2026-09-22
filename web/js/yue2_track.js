import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { buttonRow, element, frame, installStyle, panelWidget, setWidgetValue, showWidget,
    widgetNamed } from "./yue2_controls.js";
import * as list from "./yue2_editlist.js";
import * as roll from "./yue2_roll.js";
import { parseLyrics } from "./yue2_sheet.js";

const NODE = "YuE2EditTrack";
const EDITS = "edits";
const TAKES = "takes";
const TRACK_UI = "yue2_edit_track";
const AUDIO_UI = "audio";
const SUMMARY = "yue2_track_summary";
const BUTTONS = "yue2_track_buttons";
const TRACK_EVENT = "yue2-track-drawn";
const STYLE_ID = "yue2-track-style";

const EDIT_LABEL = "Edit track\u2026";
const EDIT_TOOLTIP =
    "Open the track: the song's wave, its bars and its sections. Select a stretch and sing it "
    + "again or take it out. The node runs when you press Render inside.";
const RESET_LABEL = "Reset track";
const RESET_TOOLTIP =
    "Throw away every edit on this node. The next run hands the song on as it came in; the takes "
    + "already sung stay in memory, so putting an edit back costs nothing.";

const SUMMARY_H = 50;
const STRIP_H = 20;
const RULER_H = 16;
const WAVE_LEAST = 90;
const MIN_BAR_PX = 6;
const MIN_BEAT_PX = 8;
const BAR_NUMBER_PX = 30;
const DRAG_PX = 3;
const GRAB_PX = 5;
const RUN_UP_SECONDS = 3.0;
const STUCK_MS = 3500;
const SOUND_ROUTE = "/yue2/sound";
const SOUND_SUBFOLDER = "yue2_edit";
const PLAYERS_KEPT = 5;
const ZOOM_STEP = 1.25;
const SHORTEST_VIEW = 2.0;
const TIME_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300];
const LABEL_PX = 56;

const SECTION_COLORS = [
    ["pre", "#9B7FD1"], ["intro", "#8C96A3"], ["verse", "#4E98C4"], ["chorus", "#D19A3F"],
    ["hook", "#D9774B"], ["bridge", "#4FA37A"], ["outro", "#7C8FA6"], ["solo", "#C98B5B"],
    ["inter", "#5FA3A3"], ["inst", "#5FA3A3"],
];

const SKIN = {
    back: "#243035",
    wave: "#5C8FA8",
    body: "#9FD2EE",
    bar: "#35464E",
    beat: "#2C3A41",
    seam: "#5E7885",
    strip: "#1C2529",
    ruler: "#1C2529",
    rulerText: "#AFC3CC",
    rulerTick: "#3B4A51",
    pick: "rgba(59, 125, 216, 0.30)",
    pickEdge: "#7FB2F0",
    made: "rgba(240, 176, 88, 0.16)",
    madeEdge: "#F0B058",
    stale: "rgba(190, 190, 190, 0.10)",
    staleEdge: "#8B949B",
    swapped: "rgba(176, 140, 230, 0.28)",
    swappedEdge: "#A88CDC",
    playhead: "#F5A623",
    sung: "rgba(255, 255, 255, 0.35)",
};

const WAITING = "Press Render to hear it.";

const HINT =
    "Drag across the wave to select, and drag either end of a selection to stretch it; hold Alt "
    + "to work in beats instead of whole bars. Click a section to select it, click the wave to "
    + "put the play mark there, the wheel zooms, Shift with the wheel slides, Space plays. "
    + "Escape stops the sound, then puts a message away, then forgets the selection, then "
    + "closes the window.";

const SOUND_BAD = "The song could not be played: ";

const STUCK_SAID =
    "The browser has the song open but no sound is coming out of it. Press Play again, or "
    + "Render to write the file afresh.";

const SOUND_FAULTS = {
    1: "the browser gave up on the file",
    2: "the file did not come through",
    3: "the sound could not be decoded",
    4: "the browser will not play a file of this kind",
};

const NOT_SUNG = "not in this session -- Sing it, then Render";

const GONE_WITH_SESSION =
    "Takes live in the session that sang them, so this one holds the take the list kept and "
    + "nothing else. The rest are still their seeds: singing one again gives back that same take.";

const OTHER_LIST =
    "These takes are of the edit that was drawn. Press Render to catch up with the list as it is "
    + "now -- a take already sung is never sung again.";

const WORDS_GUESSED =
    "The section under the play mark is lit exactly. The line inside it is a guess: the node "
    + "times sections, not lines, so a section's lines are spread evenly across it.";

const WORDS_BY_ORDER =
    " The score names its sections itself, and here those names do not line up with the tags in "
    + "the words, so the two are laid on each other in order.";

const WORDS_BY_SOME =
    " Only the sections the score names as the words' own tags do are lit; what the score calls "
    + "the rest does not line up with the words at all.";

const WORDS_NO_SCORE =
    "This song has no score, so it has no sections to lay the words on: nothing here lights up.";

const RENDER_FIRST_CUT =
    "Press Render first: a cut not yet rendered moves every bar after it, so the track drawn "
    + "here is no longer where this edit would land.";

const RENDER_FIRST_SECONDS =
    "Press Render first: an edit not yet rendered moves the seconds after it, and this "
    + "selection is in seconds.";

const WORDS_UNPAIRED =
    "The words could not be laid on the score's sections: the score names them differently and "
    + "there are not as many of them as there are tagged blocks of words.";

const WORDS_SAID = {
    "name": WORDS_GUESSED,
    "order": WORDS_GUESSED + WORDS_BY_ORDER,
    "some": WORDS_GUESSED + WORDS_BY_SOME,
};

const NOT_DRAWN =
    "This node has not run yet, so there is no track to draw. Render runs this node alone: it "
    + "finds the song in the pack's memory, lays the song's score over its sound and draws it "
    + "here. The first time on a song also separates its voice, which takes a minute of card.";

const NO_SCORE_HERE =
    "This song was sung with 'cot' set to 'off', so it has no score and no bars: select seconds "
    + "instead. A cut here takes exactly the seconds selected, and leaves the words as they are, "
    + "there being no score to say which section it emptied.";

function isTrack(node) {
    return node?.type === NODE || node?.comfyClass === NODE;
}

function nodeText(node) {
    const value = widgetNamed(node, EDITS)?.value;
    return typeof value === "string" ? value : "";
}

function nodeTakes(node) {
    const value = Number(widgetNamed(node, TAKES)?.value);
    return Number.isFinite(value) && value >= 1 ? Math.round(value) : 1;
}

function nodeList(node) {
    return list.readEdits(nodeText(node), nodeTakes(node));
}

function writeList(node, edits) {
    setWidgetValue(node, EDITS, edits.length ? list.writeEdits(edits) : "");
}

function sectionColor(name) {
    const lower = String(name || "").toLowerCase();
    const found = SECTION_COLORS.find(([word]) => lower.includes(word));
    return found ? found[1] : "#9A9A9A";
}

function soundUrl(entry, stamp, viaView) {
    if (!entry || !entry.filename) return "";
    let route;
    if (!viaView && entry.subfolder === SOUND_SUBFOLDER && (entry.type || "temp") === "temp") {
        route = SOUND_ROUTE + "?" + new URLSearchParams({
            name: entry.filename, rand: String(stamp || 0),
        }).toString();
    } else {
        route = "/view?" + new URLSearchParams({
            filename: entry.filename, subfolder: entry.subfolder || "",
            type: entry.type || "temp", rand: String(stamp || 0),
        }).toString();
    }
    return api.apiURL ? api.apiURL(route) : route;
}

function takeFacts(take) {
    if (take.sung === false) return NOT_SUNG;
    const parts = [take.seconds.toFixed(1) + " s sung"];
    if (take.join !== null && take.join !== undefined) {
        parts.push("join " + take.join.toFixed(2)
            + (take.natural === null || take.natural === undefined
                ? "" : " of " + take.natural.toFixed(2)));
    }
    if (typeof take.total === "number") parts.push("song " + roll.clock(take.total));
    if (take.ended) parts.push("ends the song");
    return parts.join(" \u00b7 ");
}

function growNode(node) {
    try {
        const wanted = node.computeSize?.()?.[1];
        if (wanted && node.size && node.size[1] < wanted) node.setSize?.([node.size[0], wanted]);
    } catch (error) {
        console.warn("[YuE2] the track summary could not resize its node:", error);
    }
}

function paintSummary(node) {
    const holder = node.__yue2TrackSummary;
    if (!holder) return;
    holder.replaceChildren();
    const drawn = node.__yue2TrackDrawn;
    const { edits, error } = nodeList(node);
    const line = (className, text) => holder.appendChild(element("div", className, text));
    if (error) {
        line("yue2-t-warn", error);
    } else if (!edits.length) {
        line("yue2-t-dim", "No edits: the song is handed on as it came in.");
    } else {
        const last = list.describeEdit(edits[edits.length - 1], edits.length - 1);
        line("", edits.length + (edits.length === 1 ? " edit, " : " edits, last: ")
            + last.slice(last.indexOf(". ") + 2));
    }
    if (drawn) {
        const bars = list.barCount(drawn.grid);
        line("yue2-t-dim", roll.clock(drawn.seconds) + " long"
            + (bars ? ", " + bars + " bars" : ", no score")
            + (drawn.song ? "" : ", not remembered"));
    } else {
        line("yue2-t-dim", "Not drawn yet: open the track and press Render.");
    }
    node.setDirtyCanvas?.(true, true);
}

function resetTrack(node) {
    const { edits } = nodeList(node);
    if (edits.length && !window.confirm("Throw away " + edits.length
        + (edits.length === 1 ? " edit" : " edits") + " on this node?")) return;
    writeList(node, []);
    paintSummary(node);
    node.__yue2TrackWindow?.refresh();
}

function openTrack(node) {
    if (node.__yue2TrackWindow) return;
    node.__yue2TrackWindow = new TrackWindow(node);
}

class TrackWindow {
    constructor(node) {
        this.node = node;
        this.payload = node.__yue2TrackDrawn || null;
        this.songFile = node.__yue2TrackSong || null;
        this.stamp = node.__yue2TrackStamp || 0;
        this.selection = null;
        this.playhead = 0;
        this.drag = null;
        this.view = null;
        this.working = false;
        this.isClosed = false;
        this.width = 0;
        this.height = 0;
        this.ratio = 1;
        this.sound = null;
        this.player = null;
        this.players = new Map();
        this.timer = null;
        this.asked = 0;
        this.stuck = false;
        this.viaView = false;
        this.until = null;
        this.trailing = false;
        this.soundSaid = false;
        this.take = null;
        this.blocks = [];
        this.paired = [];
        this.pairedBy = "";
        this.marked = "";
        this.hover = null;
        this.look = { stale: false, swapped: false };
        this.promptId = null;
        this.others = new Set();
        this.progress = 0;
        this.said = "";
        installStyle(STYLE_ID, STYLE);
        this.build();
        this.onDrawn = (event) => {
            if (String(event.detail?.id) === String(this.node.id)) this.arrived();
        };
        window.addEventListener(TRACK_EVENT, this.onDrawn);
        this.onRunEnd = (event) => this.finished(event);
        api.addEventListener("execution_error", this.onRunEnd);
        api.addEventListener("execution_interrupted", this.onRunEnd);
        api.addEventListener("execution_success", this.onRunEnd);
        this.onProgress = (event) => this.progressed(event);
        this.onProgressText = (event) => this.progressSaid(event);
        api.addEventListener("progress_state", this.onProgress);
        api.addEventListener("progress_text", this.onProgressText);
        this.box.title = this.payload && !this.hasScore() ? NO_SCORE_HERE : "";
        this.setStatus(this.saidAboutTheRun());
        this.refresh();
    }

    build() {
        const { panel, close, handle } = frame({ sticky: true, onClose: () => this.closed() });
        panel.classList.add("yue2-track");
        this.panel = panel;
        this.close = close;
        panel.appendChild(element("h3", "", "Track \u2014 "
            + (this.node.title || "YuE2 Edit Track")));
        panel.appendChild(element("p", "yue2-t-sub",
            "The song as this node hands it on. Nothing is sung until Render, and every take "
            + "already sung is kept, so undoing an edit or keeping another take costs nothing."));

        const top = element("div", "yue2-t-bar");
        this.playButton = element("button", "", "Play");
        this.playButton.title = "Play the song from the start of the selection (Space).";
        this.playButton.addEventListener("click", () => this.togglePlay());
        top.appendChild(this.playButton);
        this.playPickButton = element("button", "", "Play selected");
        this.playPickButton.title = "Play the selected stretch and stop at the end of it.";
        this.playPickButton.addEventListener("click", () => this.playPick());
        top.appendChild(this.playPickButton);
        this.timeLabel = element("span", "yue2-t-facts", "0:00 / 0:00");
        top.appendChild(this.timeLabel);
        this.fitButton = element("button", "", "Fit");
        this.fitButton.title = "Show the whole song again.";
        this.fitButton.addEventListener("click", () => {
            this.view = null;
            this.draw();
            this.paintButtons();
        });
        top.appendChild(this.fitButton);
        top.appendChild(element("span", "yue2-t-grow"));
        this.placeLabel = element("span", "yue2-t-facts", "");
        top.appendChild(this.placeLabel);
        this.renderButton = element("button", "yue2-t-go", "Render");
        this.renderButton.title = "Run this node alone, with the edits listed below. The song it "
            + "hands on and the track drawn here both come from that run.";
        this.renderButton.addEventListener("click", () => this.render());
        top.appendChild(this.renderButton);
        panel.appendChild(top);

        this.progressBox = element("div", "yue2-t-run");
        const trough = element("div", "yue2-t-progress");
        this.fill = element("div", "yue2-t-fill");
        trough.appendChild(this.fill);
        this.progressBox.appendChild(trough);
        this.progressLabel = element("span", "yue2-t-facts", "");
        this.progressBox.appendChild(this.progressLabel);
        this.cancelButton = element("button", "yue2-t-small", "Cancel");
        this.cancelButton.title = "Stop this run. Everything sung before it stops is kept, so "
            + "asking again goes on from there.";
        this.cancelButton.addEventListener("click", () => this.cancel());
        this.progressBox.appendChild(this.cancelButton);
        this.progressBox.hidden = true;
        panel.appendChild(this.progressBox);

        const body = element("div", "yue2-t-body");
        const left = element("div", "yue2-t-left");
        this.box = element("div", "yue2-t-canvas");
        this.canvas = document.createElement("canvas");
        this.box.appendChild(this.canvas);
        this.empty = element("div", "yue2-t-empty");
        this.box.appendChild(this.empty);
        left.appendChild(this.box);

        this.slider = element("input", "yue2-t-slide");
        this.slider.type = "range";
        this.slider.min = "0";
        this.slider.max = "1000";
        this.slider.value = "0";
        this.slider.title = "Slide along the song.";
        this.slider.addEventListener("input", () => this.slid());
        left.appendChild(this.slider);

        const chosen = element("div", "yue2-t-bar yue2-t-picks");
        this.pickLabel = element("span", "yue2-t-pick", "Nothing selected");
        chosen.appendChild(this.pickLabel);
        this.clearButton = element("button", "yue2-t-small", "Clear");
        this.clearButton.title = "Forget the selection (Escape).";
        this.clearButton.addEventListener("click", () => this.clearPick());
        chosen.appendChild(this.clearButton);
        chosen.appendChild(element("span", "yue2-t-grow"));
        this.retakeButton = element("button", "", "Retake");
        this.retakeButton.title = "Sing the selected stretch again, as many takes as the node's "
            + "'takes' widget asks for, and keep the one that joins the old song best.";
        this.retakeButton.addEventListener("click", () => this.addEdit("retake"));
        chosen.appendChild(this.retakeButton);
        this.cutButton = element("button", "", "Cut");
        this.cutButton.title = "Take the selected bars out and draw the two sides together. The "
            + "score and the words of a section the cut empties go with it.";
        this.cutButton.addEventListener("click", () => this.addEdit("cut"));
        chosen.appendChild(this.cutButton);
        this.undoButton = element("button", "", "Undo last edit");
        this.undoButton.title = "Take the last edit off the list. What it was made on is still in "
            + "memory, so this costs nothing.";
        this.undoButton.addEventListener("click", () => this.undo());
        chosen.appendChild(this.undoButton);
        left.appendChild(chosen);

        this.takesRow = element("div", "yue2-t-takes");
        left.appendChild(this.takesRow);
        body.appendChild(left);
        this.words = element("div", "yue2-t-words");
        body.appendChild(this.words);
        panel.appendChild(body);

        this.status = element("div", "yue2-t-status");
        panel.appendChild(this.status);
        this.listBox = element("div", "yue2-t-list");
        panel.appendChild(this.listBox);
        panel.appendChild(element("p", "yue2-t-hint", HINT));

        const foot = element("div", "yue2-t-foot");
        foot.appendChild(element("span", "yue2-t-grow"));
        const done = element("button", "", "Close");
        done.addEventListener("click", () => this.close());
        foot.appendChild(done);
        panel.appendChild(foot);

        this.canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
        this.canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
        this.canvas.addEventListener("pointerup", (event) => this.pointerUp(event));
        this.canvas.addEventListener("pointercancel", () => {
            this.drag = null;
        });
        this.canvas.addEventListener("pointerleave", () => {
            this.hover = null;
            this.placeLabel.textContent = "";
            this.markWords();
        });
        this.canvas.addEventListener("wheel", (event) => this.wheeled(event), { passive: false });
        panel.addEventListener("keydown", (event) => {
            if (event.key !== " " && event.code !== "Space") return;
            event.preventDefault();
            this.togglePlay();
        });
        handle.onEscape = () => {
            if (this.sound) {
                this.stopSound();
                return;
            }
            if (this.soundSaid) {
                this.hushSound();
                return;
            }
            if (this.selection) {
                this.clearPick();
                return;
            }
            this.close();
        };
        panel.tabIndex = -1;
        this.observer = new ResizeObserver(() => this.resize());
        this.observer.observe(this.box);
        setTimeout(() => this.resize(), 0);
    }

    closed() {
        this.isClosed = true;
        this.dropSound();
        this.observer?.disconnect();
        window.removeEventListener(TRACK_EVENT, this.onDrawn);
        api.removeEventListener("execution_error", this.onRunEnd);
        api.removeEventListener("execution_interrupted", this.onRunEnd);
        api.removeEventListener("execution_success", this.onRunEnd);
        api.removeEventListener("progress_state", this.onProgress);
        api.removeEventListener("progress_text", this.onProgressText);
        if (this.node.__yue2TrackWindow === this) this.node.__yue2TrackWindow = null;
    }

    setStatus(text, bad = false) {
        this.soundSaid = false;
        this.status.textContent = text || "";
        this.status.classList.toggle("yue2-t-bad", Boolean(bad));
    }

    saySound(text, bad = false) {
        this.setStatus(text, bad);
        this.soundSaid = Boolean(text);
    }

    hushSound() {
        if (!this.soundSaid) return;
        this.setStatus("");
    }

    grid() {
        const take = this.shownTake();
        if (take && take.grid) return take.grid;
        return this.payload?.grid || null;
    }

    shownTake() {
        const takes = this.payload?.takes || [];
        if (!takes.length || this.payload.kind !== "retake") return null;
        const at = this.take === null || this.take === undefined
            ? (this.payload.chosen === null || this.payload.chosen === undefined
                ? 0 : this.payload.chosen)
            : this.take;
        const take = takes[Math.max(0, Math.min(takes.length - 1, at))] || null;
        return take && take.sung === false ? null : take;
    }

    wave() {
        const take = this.shownTake();
        if (take && take.peaks && take.peaks.length) return take;
        return this.payload || { peaks: [], rms: [] };
    }

    total() {
        const take = this.shownTake();
        if (take && typeof take.total === "number") return take.total;
        return Number(this.payload?.seconds) || 0;
    }

    songEntry() {
        const take = this.shownTake();
        return take && take.audio ? take.audio : this.songFile;
    }

    hasScore() {
        return list.barCount(this.grid()) > 0;
    }

    arrived() {
        this.payload = this.node.__yue2TrackDrawn || null;
        this.songFile = this.node.__yue2TrackSong || null;
        this.stamp = this.node.__yue2TrackStamp || 0;
        this.dropSound();
        this.playhead = 0;
        this.view = null;
        this.take = null;
        this.selection = null;
        this.setWorking(false);
        this.adoptTakes();
        this.box.title = this.payload && !this.hasScore() ? NO_SCORE_HERE : "";
        this.setStatus(this.saidAboutTheRun());
        this.refresh();
    }

    saidAboutTheRun() {
        if (!this.payload) return "";
        const at = Array.isArray(this.payload.at) ? this.payload.at : null;
        const dropped = (this.payload.dropped || []).map(
            (tag) => tag || "the lines before the first tag");
        if (at && this.payload.kind === "cut") {
            return "The cut took " + list.spanText(at[0], at[1]) + " out of the song."
                + (dropped.length ? " Its words leave with it: " + dropped.join(", ") + "." : "");
        }
        if (at) {
            const made = this.madeSpan();
            return "The retake is in the song, at " + list.spanText(made[0], made[1])
                + ". Listen to the takes, or keep another one.";
        }
        if (!this.hasScore()) return NO_SCORE_HERE;
        return "The track is drawn. Select a stretch and press Retake or Cut.";
    }

    adoptTakes() {
        const answered = list.readEdits(this.payload?.edits || "", nodeTakes(this.node));
        const mine = nodeList(this.node);
        if (answered.error || mine.error) return;
        if (!list.sameEdits(answered.edits, mine.edits)) return;
        if (list.writeEdits(answered.edits) === list.writeEdits(mine.edits)) return;
        writeList(this.node, answered.edits);
        paintSummary(this.node);
    }

    finished(event) {
        const id = event?.detail?.prompt_id;
        if (id && ((this.promptId && id !== this.promptId)
            || (!this.promptId && this.others.has(id)))) return;
        if (this.isClosed) {
            this.working = false;
            return;
        }
        this.setWorking(false);
    }

    setWorking(on) {
        this.working = Boolean(on);
        if (!this.working) {
            this.progress = 0;
            this.said = "";
        }
        this.renderButton.textContent = this.working ? "Rendering\u2026" : "Render";
        this.paintButtons();
        this.paintTakes();
        this.paintProgress();
    }

    progressed(event) {
        const id = event.detail?.prompt_id;
        const mine = (event.detail?.nodes || {})[String(this.node.id)];
        if (!mine) {
            if (id && this.working && !this.promptId) this.others.add(id);
            return;
        }
        if (id) this.promptId = id;
        if (mine.state === "running" && !this.working) this.setWorking(true);
        this.progress = Number(mine.max) ? Number(mine.value) / Number(mine.max) : 0;
        this.paintProgress();
    }

    progressSaid(event) {
        if (String(event.detail?.nodeId) !== String(this.node.id)) return;
        this.said = String(event.detail?.text || "");
        this.paintProgress();
    }

    paintProgress() {
        this.progressBox.hidden = !this.working;
        this.fill.style.width = Math.round(Math.max(0, Math.min(1, this.progress)) * 100) + "%";
        this.progressLabel.textContent = this.said || "Running";
    }

    async cancel() {
        try {
            await api.interrupt();
            this.setStatus("Asked to stop. It stops between stages, so give it a moment; "
                + "everything sung so far is kept.");
        } catch (error) {
            this.setStatus("The run could not be stopped: " + (error?.message || error), true);
        }
    }

    sameEdit() {

        if (!this.payload) return false;
        const answered = list.readEdits(this.payload.edits || "", nodeTakes(this.node));
        const mine = nodeList(this.node);
        if (answered.error || mine.error) return false;
        if (!answered.edits.length || answered.edits.length !== mine.edits.length) return false;
        const there = answered.edits[answered.edits.length - 1];
        const here = mine.edits[mine.edits.length - 1];
        return here.op === there.op && here.seed === there.seed
            && JSON.stringify(here.bars) === JSON.stringify(there.bars)
            && JSON.stringify(here.seconds) === JSON.stringify(there.seconds);
    }

    clearPick() {
        if (!this.selection) return;
        this.selection = null;
        this.paintButtons();
        this.draw();
        this.setStatus("Nothing selected.");
    }

    waiting() {
        const mine = nodeList(this.node);
        if (mine.error || !this.payload) return false;
        const answered = list.readEdits(this.payload.edits || "", nodeTakes(this.node));
        return !answered.error
            && list.writeEdits(answered.edits) !== list.writeEdits(mine.edits);
    }

    refresh() {
        if (this.isClosed) return;
        this.paintWords();
        this.paintTakes();
        this.paintList();
        this.paintButtons();
        this.draw();
    }

    paintButtons() {
        const { edits, error } = nodeList(this.node);
        const drawn = Boolean(this.payload);
        this.empty.hidden = drawn;
        this.canvas.hidden = !drawn;
        if (!drawn) {
            this.empty.replaceChildren();
            this.empty.appendChild(element("div", "", NOT_DRAWN));
            const go = element("button", "yue2-t-go", "Render");
            go.disabled = this.working;
            go.addEventListener("click", () => this.render());
            this.empty.appendChild(go);
        }
        const busy = this.working;
        const why = list.whyNotEdit(this.selection, this.total()) || this.whyNotMore(this.selection);
        this.look = { stale: this.waiting() && !this.onlyTheTake(), swapped: this.swappedTake() };
        this.retakeButton.disabled = busy || !drawn || Boolean(why);
        this.cutButton.disabled = busy || !drawn || Boolean(why)
            || Boolean(list.whyNotCut(this.selection, this.hasScore()));
        this.undoButton.disabled = busy || !edits.length;
        this.renderButton.disabled = busy;
        this.playButton.disabled = !this.songEntry();
        this.fitButton.disabled = !drawn || !this.view;
        this.playPickButton.disabled = !this.selection || !this.songEntry();
        this.slider.hidden = !drawn || !this.view;
        this.pickLabel.textContent = this.selection
            ? "Selected: " + list.describeSelection(this.selection)
            : (drawn ? "Nothing selected" : "");
        this.clearButton.hidden = !this.selection;
        this.renderButton.classList.toggle("yue2-t-wait", !busy && this.waiting());
        if (error) this.setStatus(error, true);
    }

    paintList() {
        const { edits, error } = nodeList(this.node);
        this.listBox.replaceChildren();
        if (error) return;
        if (!edits.length) {
            this.listBox.appendChild(element("div", "yue2-t-dim",
                "No edits yet: the node hands the song on as it came in."));
            return;
        }
        edits.forEach((edit, index) => {
            const row = element("div", "yue2-t-line", list.describeEdit(edit, index));
            if (index === edits.length - 1) row.classList.add("yue2-t-last");
            this.listBox.appendChild(row);
        });
    }

    paintTakes() {
        this.takesRow.replaceChildren();
        const takes = this.payload?.takes || [];
        if (!takes.length || this.payload?.kind !== "retake") return;
        const { edits } = nodeList(this.node);
        const last = edits.length - 1;
        const same = this.sameEdit();
        const mine = same && last >= 0 && edits[last].op === "retake";
        const shown = this.shownTake();
        const head = element("div", "yue2-t-takehead");
        head.appendChild(element("span", "", "Takes of the retake at "
            + (this.payload.at ? list.spanText(this.payload.at[0], this.payload.at[1]) : "")
            + " -- the one chosen is the track above, so playing compares them in place."));
        this.takesRow.appendChild(head);
        if (!same) this.takesRow.appendChild(element("div", "yue2-t-warn", OTHER_LIST));
        if (takes.some((take) => take.sung === false)) {
            this.takesRow.appendChild(element("div", "yue2-t-dim", GONE_WITH_SESSION));
        }
        for (const take of takes) {
            const gone = take.sung === false;
            const row = element(gone ? "div" : "label", "yue2-t-take");
            if (gone) {
                row.appendChild(element("span", "yue2-t-takegap", ""));
            } else {
                const box = document.createElement("input");
                box.type = "radio";
                box.name = "yue2-take-" + this.node.id;
                box.checked = Boolean(shown) && take.index === shown.index;
                box.disabled = this.working;
                box.addEventListener("change", () => this.showTake(take.index));
                row.appendChild(box);
            }
            row.appendChild(element("span", "yue2-t-takename", "Take " + (take.index + 1)));
            row.appendChild(element("span", "yue2-t-dim", takeFacts(take)));
            if (take.kept) {
                const mark = element("span", "yue2-t-dim", "the model's pick");
                mark.title = "The take whose join the model scored best; it is the one the node "
                    + "keeps unless the list says otherwise.";
                row.appendChild(mark);
            }
            if (take.flagged) {
                const flag = element("span", "yue2-t-warn", "may be heard");
                flag.title = "This take joins the old song further below the song's own join here "
                    + "than the takes that sounded right did on the stand. Listen to it, ask for "
                    + "another take, or select a wider stretch.";
                row.appendChild(flag);
            }
            if (gone) {
                const sing = element("button", "yue2-t-small", "Sing it");
                sing.title = "Keep this take instead, and sing it on the next run. It is the only "
                    + "one sung, and its seed has not changed, so it comes back the take it was.";
                sing.disabled = this.working || !mine;
                sing.addEventListener("click", () => this.singTake(take.index));
                row.appendChild(sing);
            }
            this.takesRow.appendChild(row);
        }
        const foot = element("div", "yue2-t-takefoot");
        const more = element("button", "yue2-t-small", "More takes");
        more.title = "Ask for one more take of this edit. The takes already sung are kept, so "
            + "only the new one is sung.";
        more.disabled = this.working || !mine || edits[last].takes >= list.MAX_TAKES;
        more.addEventListener("click", () => this.moreTakes());
        foot.appendChild(more);
        this.takesRow.appendChild(foot);
    }

    paintWords() {
        this.words.replaceChildren();
        this.blocks = [];
        this.paired = [];
        this.marked = "";
        const text = this.payload?.lyrics || "";
        if (!text.trim()) {
            this.words.hidden = true;
            return;
        }
        this.words.hidden = false;
        this.words.appendChild(element("div", "yue2-t-wordhead", "Words"));
        for (const block of parseLyrics(text)) {
            const row = { block, tag: null, lines: [] };
            if (block.tag) {
                row.tag = element("div", "yue2-t-tag",
                    "[" + block.tag + (block.number ? " " + block.number : "") + "]");
                this.words.appendChild(row.tag);
            }
            for (const line of block.lines) {
                const said = element("div", "yue2-t-word", line || " ");
                row.lines.push(said);
                this.words.appendChild(said);
            }
            this.blocks.push(row);
        }
        this.pairWords();
        this.words.title = !(this.grid()?.sections || []).length ? WORDS_NO_SCORE
            : (WORDS_SAID[this.pairedBy] || WORDS_UNPAIRED);
        this.markWords();
    }

    pairWords() {
        const sections = this.grid()?.sections || [];
        this.paired = sections.map(() => null);
        this.pairedBy = "";
        if (!sections.length || !this.blocks.length) return;
        const singing = [];
        this.blocks.forEach((row, index) => {
            if (row.block.lines.some((line) => /\p{L}/u.test(line))) singing.push(index);
        });
        const names = sections.map(list.sectionName);
        const labels = singing.map((index) => list.labelOf(this.blocks[index].block.tag));
        const notes = sections.every((section) => typeof section.notes === "number")
            ? sections.map((section) => section.notes) : null;
        const places = list.pairSections(names, labels, notes);
        if (!places) return;
        places.forEach((where, at) => {
            this.paired[where] = singing[at];
        });
        this.pairedBy = places.length < labels.length ? "some"
            : (places.every((where, at) => names[where] === labels[at]) ? "name" : "order");
    }

    wordAt(second) {
        const sections = this.grid()?.sections || [];
        if (!sections.length || !this.blocks.length) return null;
        let index = -1;
        for (let at = 0; at < sections.length; at += 1) {
            if (sections[at].start <= second) index = at;
        }
        if (index < 0) return null;
        const block = this.paired[index];
        if (block === null || block === undefined) return null;
        const section = sections[index];
        const stop = sections[index + 1] ? sections[index + 1].start
            : Math.max(section.end, this.total());
        const from = typeof section.sung === "number" && section.sung > section.start
            ? section.sung : section.start;
        const lines = this.blocks[block].lines;
        if (!lines.length || stop <= from || second < from) return { block, line: -1 };
        const step = (stop - from) / lines.length;
        return { block, line: Math.max(0, Math.min(lines.length - 1,
            Math.floor((second - from) / step))) };
    }

    markWords() {
        if (!this.blocks.length) return;
        const now = this.wordAt(this.playhead);
        const over = this.hover === null ? null : this.wordAt(this.hover);
        const key = (now ? now.block + ":" + now.line : "-")
            + "/" + (over ? over.block + ":" + over.line : "-");
        if (key === this.marked) return;
        this.marked = key;
        for (let at = 0; at < this.blocks.length; at += 1) {
            const row = this.blocks[at];
            const here = Boolean(now) && now.block === at;
            const under = Boolean(over) && over.block === at;
            row.tag?.classList.toggle("yue2-t-now", here);
            row.tag?.classList.toggle("yue2-t-over", under && !here);
            for (let index = 0; index < row.lines.length; index += 1) {
                const lit = here && now.line === index;
                row.lines[index].classList.toggle("yue2-t-now", lit);
                row.lines[index].classList.toggle("yue2-t-over",
                    under && over.line === index && !lit);
            }
        }
        if (this.sound && now && now.line >= 0) {
            this.keepWordSeen(this.blocks[now.block].lines[now.line]);
        }
    }

    keepWordSeen(said) {
        if (!said || this.words.hidden) return;
        const mine = said.getBoundingClientRect();
        const room = this.words.getBoundingClientRect();
        if (mine.top < room.top) this.words.scrollTop -= room.top - mine.top + 12;
        else if (mine.bottom > room.bottom) {
            this.words.scrollTop += mine.bottom - room.bottom + 12;
        }
    }

    resize() {
        if (this.isClosed) return;
        const rect = this.box.getBoundingClientRect();
        this.ratio = window.devicePixelRatio || 1;
        this.width = Math.max(200, Math.round(rect.width));
        this.height = Math.max(WAVE_LEAST, Math.round(rect.height));
        this.canvas.width = Math.round(this.width * this.ratio);
        this.canvas.height = Math.round(this.height * this.ratio);
        this.canvas.style.width = this.width + "px";
        this.canvas.style.height = this.height + "px";
        this.draw();
    }

    shown() {
        const total = this.total();
        if (!this.view) return { from: 0, span: total || 1 };
        const span = Math.min(total, Math.max(SHORTEST_VIEW, this.view.span));
        const from = Math.max(0, Math.min(total - span, this.view.from));
        return { from, span };
    }

    xOf(second) {
        const view = this.shown();
        return ((second - view.from) / view.span) * this.width;
    }

    atX(x) {
        const view = this.shown();
        return view.from + (x / Math.max(1, this.width)) * view.span;
    }

    slid() {
        this.trailing = false;
        const view = this.shown();
        const room = Math.max(0, this.total() - view.span);
        this.view = { from: (Number(this.slider.value) / 1000) * room, span: view.span };
        this.draw();
    }

    wheeled(event) {
        if (!this.payload) return;
        event.preventDefault();
        this.trailing = false;
        const total = this.total();
        const view = this.shown();
        if (event.shiftKey) {
            this.view = view.span >= total ? null
                : { from: view.from + view.span * 0.15 * Math.sign(event.deltaY || 1),
                    span: view.span };
            this.draw();
            this.paintButtons();
            return;
        }
        const at = this.atX(event.offsetX);
        const span = Math.min(total, Math.max(SHORTEST_VIEW,
            view.span * (event.deltaY > 0 ? ZOOM_STEP : 1 / ZOOM_STEP)));
        if (span >= total) this.view = null;
        else this.view = { from: at - ((at - view.from) / view.span) * span, span };
        this.draw();
        this.paintButtons();
    }

    pointerDown(event) {
        if (!this.payload || event.button !== 0) return;
        this.canvas.setPointerCapture?.(event.pointerId);
        const at = this.atX(event.offsetX);
        if (event.offsetY < STRIP_H && this.hasScore()) {
            const section = list.sectionAt(this.grid(), at);
            this.drag = null;
            if (section) {
                this.select(list.selectBars(this.grid(), section.bar,
                    section.bar + section.bars));
            }
            return;
        }
        const edge = this.edgeAt(event.offsetX);
        if (edge) {
            this.drag = { from: edge === "from" ? this.selection.to : this.selection.from,
                x: event.offsetX, moved: true, alt: roll.modifiersOf(event).alt };
            return;
        }
        this.drag = { from: at, x: event.offsetX, moved: false, alt: roll.modifiersOf(event).alt };
    }

    edgeAt(x) {
        if (!this.selection) return null;
        const left = Math.abs(x - this.xOf(this.selection.from));
        const right = Math.abs(x - this.xOf(this.selection.to));
        if (Math.min(left, right) > GRAB_PX) return null;
        return left <= right ? "from" : "to";
    }

    pointerMove(event) {
        if (!this.payload) return;
        const at = this.atX(event.offsetX);
        this.placeLabel.textContent = this.placeText(at);
        this.hover = at;
        this.markWords();
        if (!this.drag) {
            this.canvas.style.cursor = event.offsetY < STRIP_H && this.hasScore() ? "pointer"
                : (this.edgeAt(event.offsetX) ? "ew-resize" : "");
            return;
        }
        if (Math.abs(event.offsetX - this.drag.x) > DRAG_PX) this.drag.moved = true;
        if (!this.drag.moved) return;
        this.drag.alt = roll.modifiersOf(event).alt;
        this.select(list.selectionBetween(this.grid(), this.drag.from, at, this.drag.alt), true);
    }

    pointerUp(event) {
        this.canvas.releasePointerCapture?.(event.pointerId);
        const drag = this.drag;
        this.drag = null;
        if (!drag) return;
        if (!drag.moved) {
            this.playhead = Math.max(0, Math.min(this.total(), drag.from));
            if (this.sound) {
                this.until = null;
                this.disarmStop();
                this.seekTo(this.playhead);
            }
            this.draw();
            return;
        }
        this.select(list.selectionBetween(this.grid(), drag.from, this.atX(event.offsetX),
            drag.alt));
    }

    placeText(second) {
        const grid = this.grid();
        const where = roll.clock(second);
        if (!this.hasScore()) return where;
        const section = list.sectionAt(grid, second);
        const named = !section ? "" : (second >= section.end - 0.001
            ? ", after " + section.name : ", " + section.name);
        return "bar " + (list.barAt(grid, second) + 1) + named + ", " + where;
    }

    select(selection, dragging = false) {
        this.selection = selection;
        this.paintButtons();
        if (!dragging && selection) {
            const why = list.whyNotEdit(selection, this.total()) || this.whyNotMore(selection);
            this.setStatus(why || ("Selected " + list.describeSelection(selection) + "."),
                Boolean(why));
        }
        this.draw();
    }

    draw() {
        if (this.isClosed || !this.canvas || this.canvas.hidden) return;
        const c = this.canvas.getContext("2d");
        if (!c) return;
        c.setTransform(this.ratio, 0, 0, this.ratio, 0, 0);
        c.clearRect(0, 0, this.width, this.height);
        c.fillStyle = SKIN.back;
        c.fillRect(0, 0, this.width, this.height);
        const top = STRIP_H;
        const tall = Math.max(20, this.height - STRIP_H - RULER_H);
        this.drawWave(c, top, tall);
        this.drawLines(c, top, tall);
        this.drawMade(c, top, tall);
        this.drawPick(c, top, tall);
        this.drawStrip(c);
        this.drawNumbers(c, top);
        this.drawRuler(c, this.height - RULER_H);
        this.drawHead(c);
        this.timeLabel.textContent = roll.clock(this.playhead) + " / " + roll.clock(this.total());
        this.markWords();
        const view = this.shown();
        const room = Math.max(0, this.total() - view.span);
        this.slider.value = String(room > 0 ? Math.round((view.from / room) * 1000) : 0);
    }

    drawWave(c, top, tall) {
        const drawn = this.wave();
        const peaks = drawn.peaks || [];
        const body = drawn.rms || [];
        const total = this.total();
        if (!peaks.length || total <= 0) return;
        const middle = top + tall / 2;
        const half = tall / 2 - 2;
        for (let x = 0; x < this.width; x += 1) {
            const from = this.atX(x);
            const to = this.atX(x + 1);
            if (to <= 0 || from >= total) continue;
            const low = Math.max(0, Math.floor((from / total) * peaks.length));
            const high = Math.min(peaks.length,
                Math.max(low + 1, Math.ceil((to / total) * peaks.length)));
            let loudest = 0;
            let inside = 0;
            for (let index = low; index < high; index += 1) {
                if (peaks[index] > loudest) loudest = peaks[index];
                if ((body[index] || 0) > inside) inside = body[index];
            }
            c.fillStyle = SKIN.wave;
            const size = Math.max(1, loudest * half);
            c.fillRect(x, middle - size, 1, size * 2);
            if (!inside) continue;
            c.fillStyle = SKIN.body;
            const core = Math.max(1, inside * half);
            c.fillRect(x, middle - core, 1, core * 2);
        }
    }

    barPixels() {
        const bars = this.grid()?.bars || [];
        if (bars.length < 2) return 0;
        const view = this.shown();
        const inside = bars.filter((second) => second >= view.from
            && second <= view.from + view.span);
        if (inside.length < 2) return this.width;
        return this.width / (inside.length - 1);
    }

    drawLines(c, top, tall) {
        const grid = this.grid();
        if (!grid) return;
        const barPx = this.barPixels();
        if (barPx >= MIN_BEAT_PX * 2) {
            c.fillStyle = SKIN.beat;
            for (const beat of grid.beats || []) {
                const x = Math.round(this.xOf(beat));
                if (x >= 0 && x <= this.width) c.fillRect(x, top, 1, tall);
            }
        }
        if (!barPx || barPx >= MIN_BAR_PX) {
            c.fillStyle = SKIN.bar;
            for (const bar of grid.bars || []) {
                const x = Math.round(this.xOf(bar));
                if (x >= 0 && x <= this.width) c.fillRect(x, top, 1, tall);
            }
        }
        c.fillStyle = SKIN.seam;
        for (const section of grid.sections || []) {
            const x = Math.round(this.xOf(section.start));
            if (x >= 0 && x <= this.width) c.fillRect(x, top, 1, tall);
        }
    }

    madeSpan() {
        const at = this.payload?.at;
        if (!Array.isArray(at)) return null;
        if (this.payload.kind === "cut") return [at[0], at[0]];
        const take = this.shownTake();
        const length = take && typeof take.seconds === "number" ? take.seconds : at[1] - at[0];
        return [at[0], at[0] + length];
    }

    swappedTake() {
        const take = this.shownTake();
        return Boolean(take) && !take.kept;
    }

    onlyTheTake() {
        const take = this.shownTake();
        if (!take || !this.sameEdit()) return false;
        const mine = nodeList(this.node);
        if (mine.error || !mine.edits.length) return false;
        const last = mine.edits[mine.edits.length - 1];
        if (last.op !== "retake") return false;
        const wanted = last.take === null || last.take === undefined ? 0 : last.take;
        return wanted === take.index;
    }

    drawMade(c, top, tall) {
        const made = this.madeSpan();
        if (!made) return;
        const x0 = this.xOf(made[0]);
        const x1 = this.xOf(made[1]);
        const stale = this.look.stale;
        const swapped = this.look.swapped;
        const fill = stale ? SKIN.stale : (swapped ? SKIN.swapped : SKIN.made);
        const edge = stale ? SKIN.staleEdge : (swapped ? SKIN.swappedEdge : SKIN.madeEdge);
        if (x1 > x0) {
            c.fillStyle = fill;
            c.fillRect(x0, top, x1 - x0, tall);
        }
        c.fillStyle = edge;
        c.fillRect(Math.round(x0), top, 1, tall);
        if (x1 > x0) c.fillRect(Math.round(x1), top, 1, tall);
    }

    drawPick(c, top, tall) {
        if (!this.selection) return;
        const x0 = this.xOf(this.selection.from);
        const x1 = this.xOf(this.selection.to);
        c.fillStyle = SKIN.pick;
        c.fillRect(x0, top, Math.max(1, x1 - x0), tall);
        c.fillStyle = SKIN.pickEdge;
        c.fillRect(Math.round(x0), top, 1, tall);
        c.fillRect(Math.round(x1) - 1, top, 1, tall);
    }

    drawStrip(c) {
        c.fillStyle = SKIN.strip;
        c.fillRect(0, 0, this.width, STRIP_H);
        c.font = "11px system-ui, sans-serif";
        c.textBaseline = "middle";
        const grid = this.grid();
        const sections = grid && Array.isArray(grid.sections) ? grid.sections : [];
        if (!sections.length) {
            c.fillStyle = SKIN.rulerText;
            c.fillText("no score, so no bars: select seconds", 6, STRIP_H / 2);
            return;
        }
        const tall = STRIP_H - 2;
        for (let index = 0; index < sections.length; index += 1) {
            const section = sections[index];
            const next = sections[index + 1];
            const stop = next ? next.start : Math.max(section.end, this.total());
            const x0 = Math.round(this.xOf(section.start));
            const x1 = Math.round(this.xOf(stop));
            if (x1 < 0 || x0 > this.width) continue;
            const own = Math.max(x0, Math.min(x1, Math.round(this.xOf(section.end))));
            c.fillStyle = sectionColor(section.name);
            c.globalAlpha = 0.55;
            c.fillRect(x0, 0, Math.max(1, own - x0), tall);
            if (x1 > own) {
                c.globalAlpha = 0.20;
                c.fillRect(own, 0, x1 - own, tall);
            }
            c.globalAlpha = 1;
            if (index && x0 > 0 && x0 <= this.width) {
                c.fillStyle = SKIN.strip;
                c.fillRect(x0 - 1, 0, 1, tall);
            }
            if (section.sung !== null && section.sung !== undefined
                && section.sung > section.start + 0.01) {
                c.fillStyle = SKIN.sung;
                c.fillRect(Math.round(this.xOf(section.sung)), tall / 2, 1, tall / 2);
            }
            const from = Math.max(x0, 0);
            if (Math.min(x1, this.width) - from > 26) {
                c.save();
                c.beginPath();
                c.rect(from + 3, 0, Math.max(1, Math.min(x1, this.width) - from - 6), tall);
                c.clip();
                c.fillStyle = "#fff";
                c.fillText(section.name, from + 5, STRIP_H / 2 - 1);
                c.restore();
            }
        }
    }

    drawNumbers(c, top) {
        const bars = this.grid()?.bars || [];
        const barPx = this.barPixels();
        if (bars.length < 2 || barPx < MIN_BAR_PX) return;
        const every = Math.max(1, Math.ceil(BAR_NUMBER_PX / barPx));
        c.font = "10px system-ui, sans-serif";
        c.textBaseline = "top";
        c.fillStyle = SKIN.rulerText;
        for (let index = 0; index < bars.length - 1; index += 1) {
            if (index % every) continue;
            const x = Math.round(this.xOf(bars[index]));
            if (x < -10 || x > this.width) continue;
            c.fillText(String(index + 1), x + 2, top + 2);
        }
    }

    drawRuler(c, top) {
        c.fillStyle = SKIN.ruler;
        c.fillRect(0, top, this.width, RULER_H);
        const view = this.shown();
        const perSecond = this.width / view.span;
        const step = TIME_STEPS.find((seconds) => seconds * perSecond >= LABEL_PX)
            || TIME_STEPS[TIME_STEPS.length - 1];
        c.font = "10px system-ui, sans-serif";
        c.textBaseline = "middle";
        for (let second = Math.ceil(view.from / step) * step; second <= view.from + view.span;
            second += step) {
            const x = Math.round(this.xOf(second));
            c.fillStyle = SKIN.rulerTick;
            c.fillRect(x, top, 1, 4);
            const said = roll.clock(second);
            if (x + 3 + c.measureText(said).width > this.width) continue;
            c.fillStyle = SKIN.rulerText;
            c.fillText(said, x + 3, top + RULER_H / 2);
        }
    }

    drawHead(c) {
        const x = Math.round(this.xOf(this.playhead));
        if (x < 0 || x > this.width) return;
        c.fillStyle = SKIN.playhead;
        c.fillRect(x, 0, 1, this.height - RULER_H);
    }

    togglePlay() {
        if (this.sound) {
            this.stopSound();
            return;
        }
        this.playSong(this.startAt());
    }

    startAt() {
        if (this.playhead > 0.05 && this.playhead < this.total() - 0.1) return this.playhead;
        if (this.selection) return this.selection.from;
        const made = this.madeSpan();
        return made ? Math.max(0, made[0] - RUN_UP_SECONDS) : 0;
    }

    sounder(url) {
        const had = this.players.get(url);
        if (had) return had;
        const sound = new Audio();
        sound.preload = "auto";
        sound.__yue2Wanted = null;
        sound.addEventListener("loadedmetadata", () => this.seekWanted(sound));
        sound.addEventListener("playing", () => this.started(sound));
        sound.addEventListener("timeupdate", () => this.ticked(sound));
        sound.addEventListener("ended", () => {
            if (this.player === sound) this.stopSound();
        });
        sound.addEventListener("error", () => this.soundFailed(sound));
        sound.src = url;
        sound.load();
        this.players.set(url, sound);
        for (const old of [...this.players.keys()]) {
            if (this.players.size <= PLAYERS_KEPT || old === url) break;
            if (this.players.get(old) === this.player) continue;
            this.release(this.players.get(old));
            this.players.delete(old);
        }
        return sound;
    }

    release(sound) {
        if (!sound) return;
        sound.pause();
        sound.removeAttribute("src");
        sound.load();
    }

    warmTakes() {
        for (const take of this.payload?.takes || []) {
            if (!take.audio) continue;
            const url = soundUrl(take.audio, this.stamp, this.viaView);
            if (url && !this.players.has(url)) this.sounder(url);
        }
    }

    started(sound) {
        if (this.player !== sound) return;
        this.hushSound();
        this.armStop();
        this.warmTakes();
    }

    ticked(sound) {
        if (this.player !== sound || !this.sound) return;
        this.playhead = sound.currentTime;
        if (this.stopAtEnd()) return;
        this.keepUp();
        this.draw();
    }

    stopAtEnd() {
        const sound = this.player;
        if (this.until === null || !this.sound || !sound) return false;
        if (sound.__yue2Wanted !== null && sound.__yue2Wanted !== undefined) return false;
        if (sound.currentTime < this.until) return false;
        const end = this.until;
        this.stopSound();
        this.playhead = end;
        this.draw();
        return true;
    }

    armStop() {
        this.disarmStop();
        if (this.until === null || !this.sound || !this.player) return;
        if (this.stopAtEnd()) return;
        const rate = this.player.playbackRate || 1;
        const left = (this.until - this.player.currentTime) / (rate > 0 ? rate : 1);
        this.timer = setTimeout(() => {
            this.timer = null;
            this.stopAtEnd();
        }, Math.max(10, left * 1000));
    }

    disarmStop() {
        if (this.timer === null) return;
        clearTimeout(this.timer);
        this.timer = null;
    }

    stopSound() {
        this.asked += 1;
        this.disarmStop();
        this.sound = null;
        this.until = null;
        this.stuck = false;
        if (this.player) {
            this.player.pause();
            this.player.__yue2Wanted = null;
        }
        if (this.playButton) this.playButton.textContent = "Play";
    }

    dropSound() {
        this.stopSound();
        for (const sound of this.players.values()) this.release(sound);
        this.players.clear();
        this.player = null;
    }

    seekTo(second) {
        const sound = this.player;
        if (!sound) return;
        const at = Math.max(0, Math.min(Math.max(0, this.total() - 0.05), second || 0));
        if (sound.__yue2Wanted !== null && sound.__yue2Wanted !== undefined) {
            sound.__yue2Wanted = at;
            return;
        }
        sound.currentTime = at;
        this.armStop();
    }

    seekWanted(sound) {
        const at = sound.__yue2Wanted;
        if (at === null || at === undefined) return;
        sound.__yue2Wanted = null;
        const room = Number.isFinite(sound.duration) ? Math.max(0, sound.duration - 0.05) : at;
        const to = Math.min(at, room);
        if (Math.abs(sound.currentTime - to) > 0.05) sound.currentTime = to;
        if (this.player === sound) this.armStop();
    }

    soundFailed(sound) {
        if (this.isClosed || !this.sound || this.player !== sound) return;
        const code = sound.error?.code;
        if (!this.viaView && (code === 2 || code === 4)
            && String(sound.currentSrc || sound.src).includes(SOUND_ROUTE)) {
            this.viaView = true;
            const at = this.playhead;
            const until = this.until;
            this.dropSound();
            this.playSong(at, until);
            return;
        }
        const why = SOUND_FAULTS[code] || "the browser gave no reason";
        this.stopSound();
        this.saySound(SOUND_BAD + why + ".", true);
    }

    playPick() {
        if (!this.selection) return;
        this.playSong(this.selection.from, this.selection.to);
    }

    playSong(from, until) {
        const url = soundUrl(this.songEntry(), this.stamp, this.viaView);
        if (!url) {
            this.saySound("This run wrote no sound to play: the song could not be remembered.",
                true);
            return;
        }
        const sound = this.sounder(url);
        const at = Math.max(0, Math.min(Math.max(0, this.total() - 0.05), from || 0));
        this.until = typeof until === "number" && until > at ? until : null;
        this.asked += 1;
        const mine = this.asked;
        if (this.player && this.player !== sound) this.player.pause();
        this.player = sound;
        if (sound.readyState >= 1) {
            sound.__yue2Wanted = null;
            if (Math.abs(sound.currentTime - at) > 0.05) sound.currentTime = at;
        } else {
            sound.__yue2Wanted = at;
        }
        this.playhead = at;
        this.sound = sound;
        this.stuck = false;
        this.trailing = true;
        this.hushSound();
        this.playButton.textContent = "Stop";
        this.follow(sound, mine);
        this.armStop();
        const going = sound.play();
        if (!going || !going.catch) return;
        going.catch((error) => {
            if (this.isClosed || mine !== this.asked) return;
            if (error && error.name === "AbortError") return;
            this.stopSound();
            this.saySound(SOUND_BAD + (error?.message || error), true);
        });
    }

    keepUp() {
        if (!this.trailing || !this.view) return;
        const view = this.shown();
        if (this.playhead >= view.from && this.playhead <= view.from + view.span) return;
        this.view = { from: Math.max(0, this.playhead - view.span * 0.1), span: view.span };
    }

    follow(sound, mine) {
        let last = -1;
        let since = performance.now();
        let nudged = false;
        const step = () => {
            if (this.isClosed || this.sound !== sound || mine !== this.asked) return;
            const now = sound.currentTime;
            if (this.stopAtEnd()) return;
            if (now !== last) {
                last = now;
                since = performance.now();
                this.stuck = false;
                nudged = false;
                this.playhead = now;
                this.keepUp();
                this.draw();
            } else if (!this.stuck && performance.now() - since > STUCK_MS) {
                if (!nudged && !sound.paused) {
                    nudged = true;
                    since = performance.now();
                    this.seekTo(now);
                } else {
                    this.stuck = true;
                    this.saySound(STUCK_SAID, true);
                }
            }
            requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    }

    whyNotMore(selection) {
        if (!selection || !this.payload) return "";
        const drawn = list.readEdits(this.payload.edits || "", nodeTakes(this.node));
        const mine = nodeList(this.node);
        if (drawn.error || mine.error) return "";
        const spelled = (edit) => list.writeEdits([{ ...edit, take: null }]);
        let shared = 0;
        while (shared < drawn.edits.length && shared < mine.edits.length
            && spelled(drawn.edits[shared]) === spelled(mine.edits[shared])) shared += 1;
        const apart = drawn.edits.slice(shared).concat(mine.edits.slice(shared));
        if (!apart.length) return "";
        if (apart.some((edit) => edit.op === "cut")) return RENDER_FIRST_CUT;
        if (selection.first === undefined || selection.first === null) return RENDER_FIRST_SECONDS;
        return "";
    }

    addEdit(op) {
        const why = (op === "cut" ? list.whyNotCut(this.selection, this.hasScore())
            : list.whyNotEdit(this.selection, this.total())) || this.whyNotMore(this.selection);
        if (why) {
            this.setStatus(why, true);
            return;
        }
        const { edits, error } = nodeList(this.node);
        if (error) {
            this.setStatus(error, true);
            return;
        }
        const made = list.editFor(this.selection, op, nodeTakes(this.node), list.newSeed());
        writeList(this.node, [...edits, made]);
        paintSummary(this.node);
        this.selection = null;
        this.refresh();
        this.setStatus(list.describeEdit(made, edits.length).slice(3) + " is on the list. "
            + WAITING);
    }

    undo() {
        const { edits, error } = nodeList(this.node);
        if (error || !edits.length) return;
        writeList(this.node, list.dropLast(edits));
        paintSummary(this.node);
        this.refresh();
        this.setStatus("The last edit is off the list. " + WAITING);
    }

    showTake(index) {
        const at = this.sound ? this.playhead : null;
        const until = this.until;
        const fromTakes = this.takesRow.contains(document.activeElement);
        this.take = index;
        this.stopSound();
        const { edits, error } = nodeList(this.node);
        const mine = !error && edits.length && edits[edits.length - 1].op === "retake"
            && this.sameEdit();
        if (mine) {
            writeList(this.node, list.withTake(edits, edits.length - 1, index));
            paintSummary(this.node);
        }
        this.refresh();
        if (fromTakes) this.takesRow.querySelector("input:checked")?.focus();
        this.setStatus("Take " + (index + 1) + " is on the track."
            + (mine ? " " + WAITING + " Nothing is sung again for it." : " " + OTHER_LIST));
        if (at !== null) this.playSong(at, until);
    }

    singTake(index) {
        const { edits, error } = nodeList(this.node);
        if (error || !edits.length) return;
        const last = edits.length - 1;
        if (edits[last].op !== "retake" || !this.sameEdit()) return;
        writeList(this.node, list.withTake(edits, last, index));
        paintSummary(this.node);
        this.refresh();
        this.setStatus("Take " + (index + 1) + " is the one the list keeps. " + WAITING
            + " It is the only take sung, and the song comes back built on it.");
    }

    moreTakes() {
        const { edits, error } = nodeList(this.node);
        if (error || !edits.length) return;
        const last = edits[edits.length - 1];
        if (last.op !== "retake" || last.takes >= list.MAX_TAKES) return;
        writeList(this.node, list.withTakes(edits, edits.length - 1, last.takes + 1));
        paintSummary(this.node);
        this.refresh();
        this.setStatus("Take " + (last.takes + 1) + " is asked for. " + WAITING
            + " Only the new take is sung.");
    }

    async render() {
        if (this.working) return;
        this.promptId = null;
        this.others = new Set();
        this.setWorking(true);
        this.setStatus("Running this node alone. The bar above follows it, and Cancel stops it "
            + "between stages.");
        try {
            await app.queuePrompt(0, 1, { queueNodeIds: [String(this.node.id)] });
        } catch (error) {
            this.finished();
            this.setStatus("This node could not be queued: " + (error?.message || error), true);
        }
    }
}

const STYLE = `
.yue2-panel.yue2-track { width: min(1280px, 97vw); height: auto;
    max-height: calc(100vh - 32px); }
.yue2-track [hidden] { display: none !important; }
.yue2-track h3 { font-size: 15px; font-weight: 600; margin: 0; }
.yue2-track .yue2-t-sub { font-size: 11px; color: var(--descrip-text, #999); margin: 2px 0 10px; }
.yue2-track button, .yue2-track input[type="range"] { box-sizing: border-box; font: inherit;
    font-size: 12px; padding: 4px 9px; border-radius: 6px; color: var(--input-text, #ddd);
    background: var(--comfy-input-bg, #2b2b2b); border: 1px solid var(--border-color, #4e4e4e); }
.yue2-track button { cursor: pointer; }
.yue2-track button:hover { background: var(--comfy-menu-bg, #353535); border-color: #6a6a6a; }
.yue2-track button.yue2-t-go { background: #3B7DD8; border-color: #3B7DD8; color: #fff; }
.yue2-track button.yue2-t-wait { box-shadow: 0 0 0 2px rgba(240, 176, 88, 0.85);
    animation: yue2-t-pulse 1.6s ease-in-out infinite; }
@keyframes yue2-t-pulse {
    0%, 100% { box-shadow: 0 0 0 1px rgba(240, 176, 88, 0.30); }
    50% { box-shadow: 0 0 0 4px rgba(240, 176, 88, 0.85); }
}
@media (prefers-reduced-motion: reduce) {
    .yue2-track button.yue2-t-wait { animation: none; }
}
.yue2-track button[disabled] { opacity: 0.45; cursor: not-allowed; }
.yue2-track button.yue2-t-small { font-size: 11px; padding: 2px 8px; }
.yue2-t-bar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 8px; }
.yue2-t-picks { margin-top: 5px; }
.yue2-t-grow { flex: 1 1 auto; }
.yue2-t-facts { font-size: 12px; color: var(--descrip-text, #999); white-space: nowrap; }
.yue2-t-pick { font-size: 12px; color: var(--input-text, #ddd); white-space: nowrap; }
.yue2-t-body { flex: 1 1 auto; min-height: 0; max-height: 62vh; display: flex; gap: 10px; }
.yue2-t-left { flex: 1 1 auto; min-width: 0; min-height: 0; display: flex; flex-direction: column; }
.yue2-t-canvas { position: relative; flex: 0 0 auto; min-width: 0;
    height: clamp(150px, 30vh, 300px);
    border: 1px solid var(--border-color, #4e4e4e); border-radius: 6px; overflow: hidden; }
.yue2-t-canvas canvas { position: absolute; inset: 0; touch-action: none; cursor: text; }
.yue2-t-empty { position: absolute; inset: 0; display: flex; flex-direction: column; gap: 10px;
    align-items: center; justify-content: center; text-align: center; padding: 14px;
    overflow: auto; font-size: 12px; line-height: 1.5; color: var(--descrip-text, #aaa); }
.yue2-t-empty div { max-width: 620px; }
.yue2-t-words { flex: 0 0 230px; min-height: 0; overflow: auto; font-size: 11px;
    line-height: 1.5;
    color: var(--descrip-text, #aaa); background: rgba(0, 0, 0, 0.14); border-radius: 6px;
    padding: 6px 10px; border: 1px solid var(--border-color, #4e4e4e); }
.yue2-t-wordhead { font-size: 12px; font-weight: 600; color: var(--input-text, #ddd);
    margin-bottom: 4px; }
.yue2-t-tag { color: #9CC4E0; margin-top: 6px; border-left: 2px solid transparent;
    padding-left: 4px; border-radius: 3px; }
.yue2-t-word { overflow-wrap: anywhere; border-left: 2px solid transparent; padding-left: 4px;
    border-radius: 3px; }
.yue2-t-words .yue2-t-now { background: rgba(240, 176, 88, 0.16); border-left-color: #F0B058;
    color: var(--input-text, #eee); }
.yue2-t-words .yue2-t-over { background: rgba(255, 255, 255, 0.05);
    border-left-color: rgba(255, 255, 255, 0.25); }
.yue2-t-slide { width: 100%; margin: 6px 0 2px; accent-color: #3B7DD8; padding: 0;
    flex: 0 0 auto; }
.yue2-t-takes { flex: 0 1 auto; min-height: 0; max-height: 30vh; overflow: auto;
    margin-bottom: 4px; }
.yue2-t-takehead { font-size: 11px; color: var(--descrip-text, #888); margin: 2px 0 4px; }
.yue2-t-takefoot { margin-top: 6px; }
.yue2-t-take { display: flex; gap: 10px; align-items: center; font-size: 12px; padding: 3px 4px;
    border-radius: 5px; cursor: pointer; color: var(--input-text, #ddd); user-select: none; }
.yue2-t-take:hover { background: rgba(255, 255, 255, 0.06); }
.yue2-t-takename { min-width: 54px; }
.yue2-t-takegap { display: inline-block; width: 13px; }
.yue2-t-take input { accent-color: #3B7DD8; margin: 0; }
.yue2-t-run { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
.yue2-t-progress { flex: 1 1 auto; height: 6px; border-radius: 3px; overflow: hidden;
    background: rgba(255, 255, 255, 0.10); }
.yue2-t-fill { height: 100%; width: 0%; background: #3B7DD8; transition: width 0.2s linear; }
.yue2-t-dim { color: var(--descrip-text, #999); }
.yue2-t-warn { color: #E0A45A; }
.yue2-t-status { font-size: 12px; min-height: 18px; margin: 2px 0;
    color: var(--descrip-text, #aaa); }
.yue2-t-status.yue2-t-bad { color: #E08A8A; }
.yue2-t-list { font-size: 11px; line-height: 1.5; color: var(--descrip-text, #999);
    max-height: 50px; overflow: auto; }
.yue2-t-list .yue2-t-last { color: var(--input-text, #ddd); }
.yue2-t-hint { font-size: 11px; line-height: 1.45; color: var(--descrip-text, #888);
    margin: 4px 0 0; }
.yue2-t-foot { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
.yue2-t-foot button { font-size: 13px; padding: 6px 14px; }
.yue2-t-summary { width: 100%; height: 100%; box-sizing: border-box; overflow: hidden;
    padding: 5px 8px; border-radius: 6px; cursor: pointer; font-family: system-ui, sans-serif;
    font-size: 11px; line-height: 16px; color: var(--input-text, #ddd);
    background: var(--comfy-input-bg, #2b2b2b); border: 1px solid var(--border-color, #4e4e4e); }
.yue2-t-summary:hover { border-color: #3B7DD8; }
.yue2-t-summary div { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
`;

function installTrack(node) {
    installStyle(STYLE_ID, STYLE);
    buttonRow(node, BUTTONS, [
        { label: EDIT_LABEL, tooltip: EDIT_TOOLTIP, onClick: () => openTrack(node) },
        { label: RESET_LABEL, tooltip: RESET_TOOLTIP, onClick: () => resetTrack(node) },
    ]);
    const summary = element("div", "yue2-t-summary");
    summary.title = "Click to open the track";
    summary.addEventListener("click", () => openTrack(node));
    node.__yue2TrackSummary = summary;
    panelWidget(node, SUMMARY, summary, () => SUMMARY_H);
    showWidget(node, EDITS, false);
    paintSummary(node);
    growNode(node);
}

app.registerExtension({
    name: "yue2.edit_track",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                installTrack(this);
            } catch (error) {
                console.error("[YuE2] the track could not be added to this node:", error);
                showWidget(this, EDITS, true);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            setTimeout(() => paintSummary(this), 0);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const drawn = message?.[TRACK_UI]?.[0];
            if (drawn && typeof drawn === "object") {
                this.__yue2TrackDrawn = drawn;
                this.__yue2TrackSong = message?.[AUDIO_UI]?.[0] || null;
                this.__yue2TrackStamp = Date.now();
                paintSummary(this);
                window.dispatchEvent(new CustomEvent(TRACK_EVENT, { detail: { id: this.id } }));
            }
            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.__yue2TrackWindow?.close();
            return onRemoved?.apply(this, arguments);
        };
    },
});
