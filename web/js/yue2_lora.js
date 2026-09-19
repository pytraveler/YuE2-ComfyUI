import { app } from "../../scripts/app.js";
import { ask, element, installStyle, widgetNamed } from "./yue2_controls.js";

const NODE = "YuE2LoRA";
const WIDGET = "loras";
const WIDGET_TYPE = "yue2_lora_rows";
const ROUTE = "/yue2/loras";
const STATE = "__yue2Lora";
const STYLE_ID = "yue2-lora-style";
const SUFFIX = ".safetensors";
const HALVES = ["ar", "nar"];
const DOT = " \u00b7 ";

const STEP = 0.05;
const FINE_STEP = 0.01;
const LIMIT = 10;
const DRAG_PX = 4;
const CLICK_SLOP = 3;
const COPIED_MS = 1200;
const LIST_AGE_MS = 5000;
const MIN_WIDTH = 440;

const MARGIN = 6;
const HEAD_H = 16;
const EMPTY_H = 30;
const LINE_H = 22;
const INFO_H = 14;
const GAP = 3;
const FOOT_H = 24;
const GRIP_W = 12;
const TICK_W = 14;
const NUMBER_W = 64;
const KILL_W = 14;
const COLUMN_GAP = 5;
const COLUMNS = `${GRIP_W}px ${TICK_W}px minmax(0, 1fr) ${NUMBER_W}px ${NUMBER_W}px ${KILL_W}px`;
const INFO_INDENT = GRIP_W + TICK_W + 2 * COLUMN_GAP + 2;

const ADD_LABEL = "+ Add LoRA";
const ADD_TOOLTIP =
    "Choose a LoRA file from ComfyUI's loras folders. Only files made for YuE2 are offered; the ones " +
    "that cannot be used are listed greyed, with the reason.";
const EMPTY_TEXT =
    "No adapters yet. Add one below; the 'lora' output takes them to Generate Song, Plan, Plan Batch " +
    "or Render Plan.";
const CHOOSE_TEXT = "Choose a LoRA\u2026";
const SEARCH_PLACEHOLDER = "Search by name or trigger word";
const READING_TEXT = "Reading the LoRA folders\u2026";
const NONE_TEXT =
    "There is no LoRA for YuE2 in ComfyUI's loras folders. Put the .safetensors files into " +
    "models/loras and open this list again.";
const NO_MATCH_TEXT = "Nothing matches.";
const UNCHOSEN_TEXT = "no file chosen yet";
const GONE_TEXT = "not in the LoRA folders any more, or not a LoRA for YuE2";
const UNREAD_TEXT = "the LoRA list could not be read";

const TIPS = {
    ar:
        "AR: the half of YuE2 that writes the score and sings the performance -- melody, structure, " +
        "arrangement. ComfyUI's LoraLoader calls this strength strength_clip for YuE2.",
    nar:
        "NAR: the half that turns the performance into sound -- timbre, mix, production. ComfyUI's " +
        "LoraLoader calls this strength strength_model for YuE2.",
};
const NUMBER_HOW =
    "\n\nDrag sideways or use the arrows to change it, with Shift for steps of 0.01; click the number " +
    "to type one. Below zero turns the adapter's change around.";
const ABSENT = {
    ar: "This file changes nothing in AR, so there is no AR strength to set.",
    nar: "This file changes nothing in NAR, so there is no NAR strength to set.",
};
const ALL_TOOLTIP = "Switch every row on -- or every row off, when all of them are on.";
const ON_TOOLTIP = "On: the song is sung with this adapter. Click to leave it out and keep the row.";
const OFF_TOOLTIP = "Off: left out. Click to sing with it again.";
const GRIP_TOOLTIP =
    "Drag to reorder. The order is only for reading: the adapters are folded in the same order " +
    "however the rows stand.";
const KILL_TOOLTIP = "Remove this row.";
const NAME_TOOLTIP = "Click to choose another file.";
const TRIGGER_TOOLTIP =
    "Click to copy. The trigger word is not added for you: put it into the style where the " +
    "adapter's author says.";
const COT_TOOLTIP =
    "It was trained for this 'cot' on the node that sings; a run with another one says so in a warning.";
const COMPANION_TOOLTIP =
    "Its authors trained it together with this NAR adapter. Add that file as another row for the " +
    "sound it was made with.";

const TICK_SVG =
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 14 14'" +
    "%3E%3Cpolyline points='3.4,7.3 6,10.1 10.9,3.9' fill='none' stroke='%23fff'" +
    " stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E";

const GRIP_SVG =
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 15 15'" +
    " fill='%23bbb'%3E%3Ccircle cx='5.5' cy='2.5' r='1.3'/%3E%3Ccircle cx='9.5' cy='2.5'" +
    " r='1.3'/%3E%3Ccircle cx='5.5' cy='7.5' r='1.3'/%3E%3Ccircle cx='9.5' cy='7.5'" +
    " r='1.3'/%3E%3Ccircle cx='5.5' cy='12.5' r='1.3'/%3E%3Ccircle cx='9.5' cy='12.5'" +
    " r='1.3'/%3E%3C/svg%3E";

const STYLE = `
.yue2-lora { box-sizing: border-box; width: 100%; height: 100%; display: flex; flex-direction: column;
    gap: ${GAP}px; overflow: hidden; font-family: system-ui, sans-serif; font-size: 11px;
    color: var(--input-text, #ddd); user-select: none; }
.yue2-lora-top { flex: 0 0 auto; }
.yue2-lora-head, .yue2-lora-line { display: grid; grid-template-columns: ${COLUMNS};
    column-gap: ${COLUMN_GAP}px; align-items: center; padding: 0 2px; }
.yue2-lora-head { height: ${HEAD_H}px; font-size: 10px; color: var(--descrip-text, #999); }
.yue2-lora-col { text-align: center; cursor: help; }
.yue2-lora-all { cursor: default; }
.yue2-lora-empty { box-sizing: border-box; height: ${EMPTY_H}px; padding: 1px 4px; overflow: hidden;
    line-height: 14px; color: var(--descrip-text, #999); }
.yue2-lora-rows { flex: 0 0 auto; display: flex; flex-direction: column; gap: ${GAP}px; }
.yue2-lora-row { box-sizing: border-box; height: ${LINE_H + INFO_H}px; display: flex;
    flex-direction: column; border-radius: 4px; background: rgba(0, 0, 0, 0.12); }
.yue2-lora-line { flex: 0 0 ${LINE_H}px; }
.yue2-lora-info { flex: 0 0 ${INFO_H}px; box-sizing: border-box; padding: 0 6px 0 ${INFO_INDENT}px;
    font-size: 10px; line-height: ${INFO_H - 1}px; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; color: var(--descrip-text, #999); }
.yue2-lora-bad { color: #E08A8A; }
.yue2-lora-warn { color: #E0A040; }
.yue2-lora-off .yue2-lora-name, .yue2-lora-off .yue2-lora-num, .yue2-lora-off .yue2-lora-info {
    opacity: 0.45; }
.yue2-lora-lifted { opacity: 0.8; outline: 1px dashed #3B7DD8; }
.yue2-lora-grip { height: 16px; cursor: grab; opacity: 0.5; background-image: url("${GRIP_SVG}");
    background-repeat: no-repeat; background-position: center; background-size: ${GRIP_W}px 16px; }
.yue2-lora-grip:hover { opacity: 1; }
.yue2-lora-lifted .yue2-lora-grip { cursor: grabbing; opacity: 1; }
.yue2-lora-tick { width: ${TICK_W}px; height: ${TICK_W}px; box-sizing: border-box; border-radius: 3px;
    cursor: pointer; background: #2B2B2B; border: 1px solid #6A6A6A; }
.yue2-lora-ticked { background-color: #4A9D5B; border-color: #4A9D5B; background-image: url("${TICK_SVG}");
    background-repeat: no-repeat; background-position: center; background-size: ${TICK_W}px ${TICK_W}px; }
.yue2-lora-mixed { border-color: #4A9D5B; background: #4A9D5B linear-gradient(#fff, #fff) center / 8px 2px
    no-repeat; }
.yue2-lora-name { display: flex; align-items: center; min-width: 0; height: 20px; box-sizing: border-box;
    padding: 0 6px; border-radius: 4px; cursor: pointer; white-space: nowrap;
    background: var(--comfy-input-bg, #2b2b2b); border: 1px solid var(--border-color, #4e4e4e); }
.yue2-lora-name:hover { border-color: #3B7DD8; }
.yue2-lora-folder-part { flex: 0 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis;
    direction: rtl; text-align: left; color: var(--descrip-text, #999); }
.yue2-lora-slash { flex: 0 0 auto; color: var(--descrip-text, #999); }
.yue2-lora-stem { flex: 0 0 auto; max-width: 100%; overflow: hidden; text-overflow: ellipsis; }
.yue2-lora-placeholder { color: var(--descrip-text, #999); font-style: italic; }
.yue2-lora-num { display: flex; align-items: stretch; height: 20px; box-sizing: border-box; overflow: hidden;
    border-radius: 4px; background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-lora-step { flex: 0 0 14px; display: flex; align-items: center; justify-content: center;
    cursor: pointer; color: var(--descrip-text, #999); }
.yue2-lora-step:hover { color: var(--input-text, #ddd); background: var(--comfy-menu-bg, #353535); }
.yue2-lora-step::before { content: ""; border-style: solid; border-color: transparent; }
.yue2-lora-less::before { border-width: 4px 5px 4px 0; border-right-color: currentColor; }
.yue2-lora-more::before { border-width: 4px 0 4px 5px; border-left-color: currentColor; }
.yue2-lora-value { flex: 1 1 auto; min-width: 0; display: flex; align-items: center; justify-content: center;
    cursor: ew-resize; font-variant-numeric: tabular-nums; }
.yue2-lora-type { flex: 1 1 auto; width: 100%; min-width: 0; padding: 0; border: 0; outline: none;
    text-align: center; font: inherit; color: var(--input-text, #ddd); background: transparent; }
.yue2-lora-num:focus-within { border-color: #3B7DD8; }
.yue2-lora-absent { align-items: center; justify-content: center; cursor: default; border-style: dashed;
    background: transparent; color: var(--descrip-text, #999); }
.yue2-lora-kill { text-align: center; cursor: pointer; font-size: 13px; line-height: 1;
    color: var(--descrip-text, #999); }
.yue2-lora-kill:hover { color: #E08A8A; }
.yue2-lora-trigger { cursor: copy; color: var(--input-text, #ddd); border-bottom: 1px dotted currentColor; }
.yue2-lora-foot { flex: 0 0 ${FOOT_H}px; display: flex; align-items: center; gap: 8px; min-width: 0; }
.yue2-lora-add { flex: 0 0 auto; height: 22px; padding: 0 12px; font: inherit; font-size: 12px;
    border-radius: 6px; cursor: pointer; color: var(--input-text, #ddd);
    background: var(--comfy-input-bg, #2b2b2b); border: 1px solid var(--border-color, #4e4e4e); }
.yue2-lora-add:hover { background: var(--comfy-menu-bg, #353535); }
.yue2-lora-add:active { background: #3B7DD8; border-color: #3B7DD8; color: #fff; }
.yue2-lora-summary { flex: 1 1 auto; min-width: 0; text-align: right; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; color: var(--descrip-text, #999); }

.yue2-lora-picker { position: fixed; z-index: 1400; width: min(460px, 92vw); display: flex;
    flex-direction: column; gap: 4px; padding: 6px; box-sizing: border-box; border-radius: 8px;
    font-family: system-ui, sans-serif; font-size: 12px; color: var(--input-text, #ddd);
    background: var(--comfy-menu-bg, #353535); border: 1px solid var(--border-color, #4e4e4e);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5); }
.yue2-lora-search { height: 26px; padding: 0 8px; box-sizing: border-box; font: inherit; border-radius: 5px;
    color: var(--input-text, #ddd); background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); outline: none; }
.yue2-lora-search:focus { border-color: #3B7DD8; }
.yue2-lora-list { max-height: min(360px, 50vh); overflow-y: auto; }
.yue2-lora-folder { padding: 6px 8px 2px; font-size: 10px; color: var(--descrip-text, #999);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.yue2-lora-choice { display: flex; flex-wrap: wrap; align-items: baseline; column-gap: 8px; padding: 4px 8px;
    border-radius: 5px; cursor: pointer; }
.yue2-lora-choice.yue2-lora-active { background: var(--comfy-input-bg, #2b2b2b); }
.yue2-lora-choice.yue2-lora-now { box-shadow: inset 2px 0 0 #3B7DD8; }
.yue2-lora-choice-name { flex: 1 1 0; min-width: 0; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; }
.yue2-lora-choice-tags { flex: 0 0 auto; font-size: 10px; color: var(--descrip-text, #999); }
.yue2-lora-choice-more, .yue2-lora-choice-problem { flex: 1 1 100%; font-size: 10px; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; color: var(--descrip-text, #999); }
.yue2-lora-choice-problem { white-space: normal; color: #E08A8A; }
.yue2-lora-unusable { cursor: not-allowed; opacity: 0.6; }
.yue2-lora-note { padding: 4px 8px; font-size: 11px; line-height: 15px; color: var(--descrip-text, #999); }
`;

const LISTED = { items: null, failed: "", when: 0, asking: null, byName: new Map() };
const LIVE = new Set();
let PICKER = null;

function clean(name) {
    return slashed(name).trim().toLowerCase();
}

function same(one, other) {
    return !!clean(one) && clean(one) === clean(other);
}

function known(name) {
    return LISTED.byName.get(clean(name));
}

function slashed(name) {
    return String(name ?? "").replace(/\\/g, "/");
}

function folderOf(name) {
    const path = slashed(name);
    const at = path.lastIndexOf("/");
    return at < 0 ? "" : path.slice(0, at);
}

function stemOf(name) {
    const path = slashed(name);
    const base = path.slice(path.lastIndexOf("/") + 1);
    return base.toLowerCase().endsWith(SUFFIX) ? base.slice(0, -SUFFIX.length) : base;
}

function halvesText(item) {
    const halves = item?.halves || {};
    return HALVES.filter((half) => halves[half])
        .map((half) => half.toUpperCase() + (halves[half].rank ? " rank " + halves[half].rank : ""))
        .join(" + ");
}

function matrices(item) {
    const halves = item?.halves || {};
    return HALVES.reduce((sum, half) => sum + (Number(halves[half]?.tensors) || 0), 0);
}

function triggersText(item) {
    const words = item?.triggers || [];
    return words.length ? (words.length > 1 ? "triggers: " : "trigger: ") + words.join(", ") : "";
}

function details(item) {
    return [
        item.name,
        [halvesText(item), matrices(item) + " matrices"].join(DOT),
        item.layout ? "layout: " + item.layout + (item.scale ? DOT + "scale from " + item.scale : "") : "",
        triggersText(item),
        item.intended_cot ? "made for cot '" + item.intended_cot + "'" : "",
        item.companion ? "made with " + item.companion + " on NAR" : "",
        item.problem || "",
    ].filter(Boolean).join("\n");
}

function tidy(value, places = 4) {
    const scale = 10 ** places;
    return Math.round(Math.max(-LIMIT, Math.min(LIMIT, value)) * scale) / scale;
}

function format(value) {
    const cents = value * 100;
    return Math.abs(cents - Math.round(cents)) < 1e-6 ? value.toFixed(2) : String(value);
}

function strengthOf(value) {
    const number = Number(value ?? 1);
    return Number.isFinite(number) ? tidy(number) : 1;
}

function normalise(item) {
    if (!item || typeof item !== "object" || Array.isArray(item)) return null;
    return { name: String(item.name ?? "").trim(), ar: strengthOf(item.ar), nar: strengthOf(item.nar),
        on: item.on !== false };
}

function readRows(node) {
    const raw = widgetNamed(node, WIDGET)?.value;
    let parsed;
    try {
        parsed = JSON.parse(typeof raw === "string" && raw.trim() ? raw : "[]");
    } catch (error) {
        return { rows: [], broken: String(error?.message || error) };
    }
    if (!Array.isArray(parsed)) return { rows: [], broken: "they are not a list" };
    return { rows: parsed.map(normalise).filter(Boolean), broken: "" };
}

function remember() {
    try {
        const tracker = app.extensionManager?.workflow?.activeWorkflow?.changeTracker;
        (tracker?.captureCanvasState ?? tracker?.checkState)?.call(tracker);
    } catch (error) {
        console.debug("[YuE2] the LoRA change could not be recorded for undo:", error);
    }
}

function writeRows(node, rows, quiet) {
    const widget = widgetNamed(node, WIDGET);
    if (!widget) return;
    const state = node[STATE];
    const text = JSON.stringify(rows.map(({ name, ar, nar, on }) => ({ name, ar, nar, on })));
    if (state) state.quiet = true;
    try {
        widget.value = text;
    } finally {
        if (state) state.quiet = false;
    }
    if (quiet) return;
    redraw(node);
    remember();
}

function change(node, index, edit) {
    const { rows } = readRows(node);
    if (!rows[index]) return;
    edit(rows[index]);
    writeRows(node, rows);
}

function refreshListing() {
    if (LISTED.asking) return LISTED.asking;
    LISTED.asking = (async () => {
        let items = null;
        let failed = "";
        try {
            const { ok, payload } = await ask(ROUTE);
            if (ok && payload?.ok !== false && Array.isArray(payload?.loras)) items = payload.loras;
            else failed = payload?.error || "The LoRA list could not be read.";
        } catch (error) {
            failed = "The LoRA list could not be read: " + (error?.message || error);
        }
        LISTED.failed = failed;
        LISTED.when = Date.now();
        if (items) {
            LISTED.items = items;
            LISTED.byName = new Map(items.map((item) => [clean(item.name), item]));
        } else if (!LISTED.items) {
            LISTED.items = [];
        }
        LISTED.asking = null;
        for (const node of [...LIVE]) {
            if (node.graph) redraw(node);
            else LIVE.delete(node);
        }
        PICKER?.paint();
        return LISTED;
    })();
    return LISTED.asking;
}

function freshen() {
    if (LISTED.asking) return LISTED.asking;
    if (LISTED.items && Date.now() - LISTED.when < LIST_AGE_MS) return Promise.resolve(LISTED);
    return refreshListing();
}

function hold(event) {
    event.stopPropagation();
    if (event.button === 0) event.preventDefault();
}

function shutPicker() {
    const open = PICKER;
    PICKER = null;
    open?.dispose();
}

function matches(item, terms) {
    const text = (item.name + " " + (item.triggers || []).join(" ")).toLowerCase();
    return terms.every((term) => text.includes(term));
}

function arranged(items) {
    const key = (item) => [folderOf(item.name) ? 1 : 0, folderOf(item.name).toLowerCase(),
        stemOf(item.name).toLowerCase()];
    return [...items].sort((one, other) => {
        const [a, b] = [key(one), key(other)];
        return a[0] - b[0] || a[1].localeCompare(b[1]) || a[2].localeCompare(b[2]);
    });
}

function choiceRow(item, now) {
    const row = element("div", "yue2-lora-choice" + (item.problem ? " yue2-lora-unusable" : "")
        + (now ? " yue2-lora-now" : ""));
    row.append(element("span", "yue2-lora-choice-name", stemOf(item.name)),
        element("span", "yue2-lora-choice-tags", halvesText(item)));
    if (item.problem) row.appendChild(element("div", "yue2-lora-choice-problem", item.problem));
    else if (triggersText(item)) row.appendChild(element("div", "yue2-lora-choice-more", triggersText(item)));
    row.title = details(item);
    return row;
}

function openPicker(node, anchor, current, onPick) {
    shutPicker();
    installStyle(STYLE_ID, STYLE);
    const panel = element("div", "yue2-lora-picker");
    const search = element("input", "yue2-lora-search");
    search.type = "text";
    search.placeholder = SEARCH_PLACEHOLDER;
    search.spellcheck = false;
    search.autocomplete = "off";
    const list = element("div", "yue2-lora-list");
    const note = element("div", "yue2-lora-note");
    panel.append(search, list, note);
    document.body.appendChild(panel);

    let offered = [];
    let active = -1;

    function highlight(at, scroll) {
        offered[active]?.row.classList.remove("yue2-lora-active");
        active = offered.length ? Math.max(0, Math.min(offered.length - 1, at)) : -1;
        const chosen = offered[active];
        if (!chosen) return;
        chosen.row.classList.add("yue2-lora-active");
        if (scroll) chosen.row.scrollIntoView({ block: "nearest" });
    }

    function pick(item) {
        if (!item || item.problem) return;
        shutPicker();
        onPick(item.name);
    }

    function place() {
        const box = anchor.getBoundingClientRect();
        const size = panel.getBoundingClientRect();
        const left = Math.max(4, Math.min(box.left, window.innerWidth - size.width - 4));
        const below = box.bottom + 4;
        const top = below + size.height > window.innerHeight - 4
            ? Math.max(4, box.top - size.height - 4)
            : below;
        panel.style.left = `${left}px`;
        panel.style.top = `${top}px`;
    }

    function paint() {
        const terms = search.value.toLowerCase().split(/\s+/).filter(Boolean);
        const items = arranged(LISTED.items || []).filter((item) => matches(item, terms));
        list.replaceChildren();
        offered = [];
        active = -1;
        let folder = null;
        for (const item of items) {
            const where = folderOf(item.name);
            if (where !== folder) {
                folder = where;
                if (where) list.appendChild(element("div", "yue2-lora-folder", where + "/"));
            }
            const row = choiceRow(item, same(item.name, current));
            list.appendChild(row);
            if (item.problem) continue;
            const at = offered.length;
            offered.push({ item, row });
            row.addEventListener("click", () => pick(item));
            row.addEventListener("pointermove", () => {
                if (active !== at) highlight(at, false);
            });
        }
        note.textContent = LISTED.failed
            || (LISTED.items === null ? READING_TEXT
                : !LISTED.items.length ? NONE_TEXT
                    : !items.length ? NO_MATCH_TEXT : "");
        note.classList.toggle("yue2-lora-warn", !!LISTED.failed);
        note.hidden = !note.textContent;
        const now = offered.findIndex((entry) => same(entry.item.name, current));
        highlight(now >= 0 ? now : 0, true);
        place();
    }

    search.addEventListener("input", paint);
    panel.addEventListener("pointerdown", (event) => event.stopPropagation());
    panel.addEventListener("keydown", (event) => {
        event.stopPropagation();
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            highlight(active + (event.key === "ArrowDown" ? 1 : -1), true);
        } else if (event.key === "Enter") {
            event.preventDefault();
            pick(offered[active]?.item);
        }
    });
    const away = (event) => {
        if (!panel.contains(event.target)) shutPicker();
    };
    const escape = (event) => {
        if (event.key !== "Escape") return;
        event.stopPropagation();
        event.preventDefault();
        shutPicker();
    };
    document.addEventListener("pointerdown", away, true);
    document.addEventListener("keydown", escape, true);
    PICKER = {
        node,
        paint,
        dispose() {
            document.removeEventListener("pointerdown", away, true);
            document.removeEventListener("keydown", escape, true);
            panel.remove();
        },
    };
    paint();
    search.focus();
    refreshListing();
}

function tickBox(state, tooltip) {
    const box = element("div", "yue2-lora-tick" + (state === true ? " yue2-lora-ticked"
        : state === null ? " yue2-lora-mixed" : ""));
    box.title = tooltip;
    return box;
}

function release(node, redrawn) {
    const state = node[STATE];
    if (!state) return;
    state.busy = false;
    if (redrawn || state.pending) redraw(node);
}

function typeIn(node, index, key, shown) {
    const state = node[STATE];
    const input = element("input", "yue2-lora-type");
    input.type = "text";
    input.inputMode = "decimal";
    input.spellcheck = false;
    input.value = shown.textContent;
    shown.replaceWith(input);
    if (state) {
        state.busy = true;
        state.typing = true;
    }
    let finished = false;
    const finish = (keep) => {
        if (finished) return;
        finished = true;
        if (state) {
            state.typing = false;
            state.busy = false;
        }
        const value = Number(String(input.value).trim().replace(",", "."));
        const { rows } = readRows(node);
        if (keep && input.value.trim() && Number.isFinite(value) && rows[index]) {
            rows[index][key] = tidy(value);
            writeRows(node, rows);
        } else {
            redraw(node);
        }
    };
    input.addEventListener("pointerdown", (event) => event.stopPropagation());
    input.addEventListener("keydown", (event) => {
        event.stopPropagation();
        if (event.key === "Enter") {
            event.preventDefault();
            finish(true);
        } else if (event.key === "Escape") {
            event.preventDefault();
            finish(false);
        }
    });
    input.addEventListener("blur", () => finish(true));
    input.focus();
    input.select();
}

function strength(node, index, key, entry, item) {
    const box = element("div", "yue2-lora-num");
    if (item && !item.halves?.[key]) {
        box.classList.add("yue2-lora-absent");
        box.textContent = "-";
        box.title = ABSENT[key];
        return box;
    }
    const less = element("span", "yue2-lora-step yue2-lora-less");
    const shown = element("span", "yue2-lora-value", format(entry[key]));
    const more = element("span", "yue2-lora-step yue2-lora-more");
    box.append(less, shown, more);
    box.title = TIPS[key] + NUMBER_HOW;

    for (const [part, direction] of [[less, -1], [more, 1]]) {
        part.addEventListener("pointerdown", hold);
        part.addEventListener("click", (event) => {
            event.stopPropagation();
            const unit = event.shiftKey ? FINE_STEP : STEP;
            change(node, index, (row) => {
                row[key] = tidy(row[key] + direction * unit, 2);
            });
        });
    }

    let dragged = false;
    shown.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        hold(event);
        const state = node[STATE];
        if (state) state.busy = true;
        const startX = event.clientX;
        const start = entry[key];
        dragged = false;
        const moving = (move) => {
            const dx = move.clientX - startX;
            if (!dragged && Math.abs(dx) < CLICK_SLOP) return;
            dragged = true;
            const unit = move.shiftKey ? FINE_STEP : STEP;
            const value = tidy(start + Math.round(dx / DRAG_PX) * unit, 2);
            shown.textContent = format(value);
            const { rows } = readRows(node);
            if (rows[index] && rows[index][key] !== value) {
                rows[index][key] = value;
                writeRows(node, rows, true);
            }
        };
        const done = () => {
            window.removeEventListener("pointermove", moving, true);
            window.removeEventListener("pointerup", done, true);
            window.removeEventListener("pointercancel", done, true);
            setTimeout(() => {
                if (node[STATE]?.typing) return;
                release(node, dragged);
                if (dragged) remember();
            }, 0);
        };
        window.addEventListener("pointermove", moving, true);
        window.addEventListener("pointerup", done, true);
        window.addEventListener("pointercancel", done, true);
    });
    shown.addEventListener("click", (event) => {
        event.stopPropagation();
        if (dragged) return;
        typeIn(node, index, key, shown);
    });
    return box;
}

function triggerWord(word) {
    const span = element("span", "yue2-lora-trigger", word);
    span.title = TRIGGER_TOOLTIP;
    span.addEventListener("pointerdown", hold);
    span.addEventListener("click", async (event) => {
        event.stopPropagation();
        try {
            await navigator.clipboard.writeText(word);
            span.textContent = "copied";
        } catch (error) {
            span.textContent = "not copied";
        }
        setTimeout(() => {
            span.textContent = word;
        }, COPIED_MS);
    });
    return span;
}

function infoLine(entry, item) {
    const line = element("div", "yue2-lora-info");
    if (!entry.name) {
        line.textContent = UNCHOSEN_TEXT;
        line.classList.add("yue2-lora-warn");
        return line;
    }
    if (!item) {
        if (LISTED.items === null) {
            line.textContent = "\u2026";
        } else if (LISTED.failed && !LISTED.byName.size) {
            line.textContent = UNREAD_TEXT;
            line.classList.add("yue2-lora-warn");
        } else {
            line.textContent = GONE_TEXT;
            line.classList.add("yue2-lora-bad");
        }
        line.title = entry.name + (LISTED.failed ? "\n\n" + LISTED.failed : "");
        return line;
    }
    line.title = details(item);
    if (item.problem) {
        line.textContent = item.problem;
        line.classList.add("yue2-lora-bad");
        return line;
    }
    const add = (text, tooltip) => {
        if (line.childNodes.length) line.append(DOT);
        const span = element("span", "", text);
        if (tooltip) span.title = tooltip;
        line.appendChild(span);
    };
    add(halvesText(item));
    add(matrices(item) + " matrices");
    const words = item.triggers || [];
    if (words.length) {
        line.append(DOT + (words.length > 1 ? "triggers: " : "trigger: "));
        words.forEach((word, at) => {
            if (at) line.append(", ");
            line.appendChild(triggerWord(word));
        });
    }
    if (item.intended_cot) add("made for cot '" + item.intended_cot + "'", COT_TOOLTIP);
    if (item.companion) add("made with " + item.companion + " on NAR", COMPANION_TOOLTIP);
    return line;
}

function nameButton(node, index, entry) {
    const button = element("div", "yue2-lora-name");
    if (entry.name) {
        const folder = folderOf(entry.name);
        if (folder) {
            const part = element("span", "yue2-lora-folder-part");
            part.appendChild(element("bdi", "", folder));
            button.append(part, element("span", "yue2-lora-slash", "/"));
        }
        button.appendChild(element("span", "yue2-lora-stem", stemOf(entry.name)));
    } else {
        button.appendChild(element("span", "yue2-lora-placeholder", CHOOSE_TEXT));
    }
    button.title = (entry.name ? entry.name + "\n\n" : "") + NAME_TOOLTIP;
    button.addEventListener("pointerdown", hold);
    button.addEventListener("click", (event) => {
        event.stopPropagation();
        openPicker(node, button, entry.name, (name) => change(node, index, (row) => {
            row.name = name;
        }));
    });
    return button;
}

function dropBefore(holder, y) {
    let closest = null;
    let nearest = Number.NEGATIVE_INFINITY;
    for (const child of holder.querySelectorAll(".yue2-lora-row:not(.yue2-lora-lifted)")) {
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;
        if (offset < 0 && offset > nearest) {
            nearest = offset;
            closest = child;
        }
    }
    return closest;
}

function commitOrder(node) {
    const state = node[STATE];
    if (!state) return;
    const { rows } = readRows(node);
    const order = [...state.rows.children]
        .map((child) => Number(child.dataset.yue2Row))
        .filter((index) => Number.isInteger(index) && rows[index]);
    if (order.length !== rows.length || order.every((index, at) => index === at)) {
        redraw(node);
        return;
    }
    writeRows(node, order.map((index) => rows[index]));
}

function beginDrag(node, row) {
    const state = node[STATE];
    if (!state) return;
    state.busy = true;
    row.classList.add("yue2-lora-lifted");
    const moved = (event) => {
        const before = dropBefore(state.rows, event.clientY);
        if (before) state.rows.insertBefore(row, before);
        else state.rows.appendChild(row);
    };
    const done = () => {
        window.removeEventListener("pointermove", moved, true);
        window.removeEventListener("pointerup", done, true);
        window.removeEventListener("pointercancel", done, true);
        row.classList.remove("yue2-lora-lifted");
        state.busy = false;
        commitOrder(node);
    };
    window.addEventListener("pointermove", moved, true);
    window.addEventListener("pointerup", done, true);
    window.addEventListener("pointercancel", done, true);
}

function removeRow(node, index) {
    const { rows } = readRows(node);
    if (index < 0 || index >= rows.length) return;
    rows.splice(index, 1);
    writeRows(node, rows);
    fit(node, true);
}

function buildRow(node, entry, index) {
    const item = known(entry.name);
    const row = element("div", "yue2-lora-row" + (entry.on ? "" : " yue2-lora-off"));
    row.dataset.yue2Row = String(index);
    const line = element("div", "yue2-lora-line");

    const grip = element("div", "yue2-lora-grip");
    grip.title = GRIP_TOOLTIP;
    grip.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        hold(event);
        beginDrag(node, row);
    });

    const tick = tickBox(entry.on, entry.on ? ON_TOOLTIP : OFF_TOOLTIP);
    tick.addEventListener("pointerdown", hold);
    tick.addEventListener("click", (event) => {
        event.stopPropagation();
        change(node, index, (target) => {
            target.on = !target.on;
        });
    });

    const kill = element("div", "yue2-lora-kill", "\u00d7");
    kill.title = KILL_TOOLTIP;
    kill.addEventListener("pointerdown", hold);
    kill.addEventListener("click", (event) => {
        event.stopPropagation();
        removeRow(node, index);
    });

    line.append(grip, tick, nameButton(node, index, entry), strength(node, index, "ar", entry, item),
        strength(node, index, "nar", entry, item), kill);
    row.append(line, infoLine(entry, item));
    return row;
}

function header(node, rows) {
    if (!rows.length) return element("div", "yue2-lora-empty", EMPTY_TEXT);
    const bar = element("div", "yue2-lora-head");
    const all = rows.every((row) => row.on);
    const none = rows.every((row) => !row.on);
    const tick = tickBox(all ? true : none ? false : null, ALL_TOOLTIP);
    tick.addEventListener("pointerdown", hold);
    tick.addEventListener("click", (event) => {
        event.stopPropagation();
        const { rows: now } = readRows(node);
        const on = !now.every((row) => row.on);
        writeRows(node, now.map((row) => ({ ...row, on })));
    });
    const ar = element("div", "yue2-lora-col", "AR");
    ar.title = TIPS.ar;
    const nar = element("div", "yue2-lora-col", "NAR");
    nar.title = TIPS.nar;
    bar.append(element("span"), tick, element("span", "yue2-lora-all", "All"), ar, nar, element("span"));
    return bar;
}

function sounding(row) {
    const item = known(row.name);
    return HALVES.some((half) => row[half] !== 0 && (!item || item.halves?.[half]));
}

function refused(row) {
    if (LISTED.items === null || !row.on || !row.name || !sounding(row)) return false;
    const item = known(row.name);
    return !item ? !(LISTED.failed && !LISTED.byName.size) : !!item.problem;
}

function summaryText(rows, broken) {
    if (broken) return "The saved rows could not be read (" + broken + "); adding a row starts them again.";
    if (!rows.length) return "";
    const stopping = rows.filter(refused).length;
    const live = rows.filter((row) => row.on && row.name && sounding(row) && !refused(row)).length;
    return live + " of " + rows.length + " on" + (stopping ? DOT + stopping + " cannot be used" : "");
}

function heightOf(node) {
    const count = node[STATE]?.count ?? 0;
    const rows = count ? count * (LINE_H + INFO_H) + (count - 1) * GAP : 0;
    return (count ? HEAD_H : EMPTY_H) + rows + FOOT_H + 2 * GAP;
}

function fit(node, snug) {
    try {
        const wanted = node.computeSize?.()?.[1];
        if (wanted && node.size && (node.size[1] < wanted || (snug && node.size[1] > wanted))) {
            node.setSize?.([node.size[0], wanted]);
        }
    } catch (error) {
        console.warn("[YuE2] the LoRA rows could not resize their node:", error);
    }
    node.setDirtyCanvas?.(true, true);
}

function redraw(node) {
    const state = node[STATE];
    if (!state || state.quiet) return;
    if (state.busy) {
        state.pending = true;
        return;
    }
    state.pending = false;
    const { rows, broken } = readRows(node);
    state.count = rows.length;
    state.top.replaceChildren(header(node, rows));
    state.rows.replaceChildren(...rows.map((entry, index) => buildRow(node, entry, index)));
    state.summary.textContent = summaryText(rows, broken);
    state.summary.title = state.summary.textContent;
    state.summary.classList.toggle("yue2-lora-warn", !!broken || rows.some(refused));
    fit(node, false);
}

function replaceWidget(node, holder) {
    const index = node.widgets?.findIndex((widget) => widget.name === WIDGET) ?? -1;
    if (index < 0) return null;
    const original = node.widgets[index];
    const held = { value: typeof original.value === "string" && original.value.trim() ? original.value : "[]" };
    const widget = node.addDOMWidget(WIDGET, WIDGET_TYPE, holder, {
        hideOnZoom: false,
        margin: MARGIN,
        hideInPanel: true,
        getValue: () => held.value,
        setValue: (value) => {
            held.value = typeof value === "string" ? value : JSON.stringify(value ?? []);
            redraw(node);
        },
        getMinHeight: () => heightOf(node) + 2 * MARGIN,
        getMaxHeight: () => heightOf(node) + 2 * MARGIN,
    });
    const appended = node.widgets.indexOf(widget);
    if (appended >= 0) node.widgets.splice(appended, 1);
    node.widgets.splice(index, 1, widget);
    return widget;
}

function build(node) {
    installStyle(STYLE_ID, STYLE);
    const holder = element("div", "yue2-lora");
    const top = element("div", "yue2-lora-top");
    const rows = element("div", "yue2-lora-rows");
    const foot = element("div", "yue2-lora-foot");
    const add = element("button", "yue2-lora-add", ADD_LABEL);
    add.type = "button";
    add.title = ADD_TOOLTIP;
    const summary = element("span", "yue2-lora-summary");
    foot.append(add, summary);
    holder.append(top, rows, foot);
    holder.addEventListener("dblclick", (event) => event.stopPropagation());
    node[STATE] = { top, rows, summary, count: 0, quiet: false, busy: false, typing: false, pending: false };

    add.addEventListener("pointerdown", hold);
    add.addEventListener("click", (event) => {
        event.stopPropagation();
        openPicker(node, add, "", (name) => {
            const { rows: now } = readRows(node);
            now.push({ name, ar: 1, nar: 1, on: true });
            writeRows(node, now);
        });
    });

    if (!replaceWidget(node, holder)) {
        delete node[STATE];
        return;
    }
    LIVE.add(node);
    redraw(node);
    if (node.size) node.setSize?.([Math.max(node.size[0], MIN_WIDTH), node.computeSize?.()?.[1] ?? node.size[1]]);
    freshen();
}

app.registerExtension({
    name: "yue2.lora",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                build(this);
            } catch (error) {
                console.error("[YuE2] the LoRA rows could not be added to this node:", error);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            redraw(this);
            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            LIVE.delete(this);
            if (PICKER?.node === this) shutPicker();
            return onRemoved?.apply(this, arguments);
        };

        const refreshComboInNode = nodeType.prototype.refreshComboInNode;
        nodeType.prototype.refreshComboInNode = function () {
            const result = refreshComboInNode?.apply(this, arguments);
            refreshListing();
            return result;
        };
    },
});
