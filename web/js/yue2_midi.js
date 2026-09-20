import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ask, buttonRow, element, graphChanged, installStyle, panelWidget, widgetNamed } from "./yue2_controls.js";
import * as roll from "./yue2_roll.js";

const LOAD_MIDI = "YuE2LoadMidi";
const MIDI = "midi";
const MODE = "mode";
const VOCAL = "vocal_track";
const INSTRUMENT = "instrument_track";
const MIDI_UI = "yue2_midi";
const TRACKS_ROUTE = "/yue2/midi/tracks";
const UPLOAD_ROUTE = "/upload/image";
const BUTTONS = "yue2_midi_buttons";
const LIST = "yue2_midi_list";
const FIRST_EDITOR_WIDGET = "yue2_song_edit";
const ASK_DELAY = 200;
const ROW_H = 16;
const LIST_PAD = 10;
const WRAP_CHARS = 64;

const CHOOSE_LABEL = "Choose MIDI file\u2026";
const CHOOSE_TOOLTIP =
    "Upload a MIDI file -- .mid, .midi, .kar or .rmi -- into ComfyUI's input folder and choose it. " +
    "Dropping the file on the node does the same.";
const NO_FILE = "No MIDI file yet: press 'Choose MIDI file\u2026' or drop a file on the node.";

const STYLE_ID = "yue2-midi-style";
const STYLE = `
.yue2-midi { box-sizing: border-box; width: 100%; height: 100%; overflow: hidden; padding: 4px 6px;
    font-family: system-ui, sans-serif; font-size: 11px; line-height: 16px; cursor: default;
    color: var(--descrip-text, #999); background: rgba(0, 0, 0, 0.12);
    border: 1px solid var(--border-color, #4e4e4e); border-radius: 6px; }
.yue2-midi-row { display: flex; gap: 8px; white-space: nowrap; overflow: hidden; }
.yue2-midi-part { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.yue2-midi-role { flex: 0 0 auto; }
.yue2-midi-facts, .yue2-midi-sung .yue2-midi-part, .yue2-midi-sung .yue2-midi-role {
    color: var(--input-text, #ddd); }
.yue2-midi-warn { color: #E0A040; white-space: normal; }
`;

function isLoadMidi(node) {
    return node?.type === LOAD_MIDI || node?.comfyClass === LOAD_MIDI;
}

function grow(node) {
    try {
        const wanted = node.computeSize?.()?.[1];
        if (wanted && node.size && node.size[1] < wanted) node.setSize?.([node.size[0], wanted]);
    } catch (error) {
        console.warn("[YuE2] the MIDI list could not resize its node:", error);
    }
}

function paint(node, listing) {
    const holder = node.__yue2MidiList;
    if (!holder) return;
    holder.replaceChildren();
    let rows = 0;
    const line = (className, text) => {
        const made = element("div", className, text);
        holder.appendChild(made);
        rows += 1;
        return made;
    };
    if (listing.error) {
        line("yue2-midi-row yue2-midi-warn", listing.error);
        rows += Math.floor(String(listing.error).length / WRAP_CHARS);
    } else if (listing.message) {
        line("yue2-midi-row yue2-midi-facts", listing.message);
    } else if (listing.facts) {
        line("yue2-midi-row yue2-midi-facts", roll.midiFacts(listing.facts));
    }
    for (const part of listing.parts || []) {
        const row = line("yue2-midi-row" + (part.role ? " yue2-midi-sung" : ""));
        row.append(element("span", "yue2-midi-part", roll.partLine(part)),
            element("span", "yue2-midi-role", roll.partRole(part)));
        row.title = roll.partLine(part);
    }
    const count = Math.max(2, rows);
    if (node.__yue2MidiRows !== count) {
        node.__yue2MidiRows = count;
        grow(node);
    }
    node.setDirtyCanvas?.(true, true);
}

function valueOf(node, name, fallback) {
    const value = widgetNamed(node, name)?.value;
    return value === undefined || value === null || value === "" ? fallback : String(value);
}

function describe(node) {
    clearTimeout(node.__yue2MidiTimer);
    node.__yue2MidiTimer = setTimeout(async () => {
        const asked = (node.__yue2MidiAsked || 0) + 1;
        node.__yue2MidiAsked = asked;
        const name = valueOf(node, MIDI, "");
        if (!name) {
            paint(node, { error: NO_FILE, parts: [] });
            return;
        }
        try {
            const { ok, payload } = await ask(TRACKS_ROUTE, {
                name, mode: valueOf(node, MODE, "melody"), vocal_track: valueOf(node, VOCAL, "auto"),
                instrument_track: valueOf(node, INSTRUMENT, "auto"),
            });
            if (node.__yue2MidiAsked !== asked) return;
            paint(node, ok ? payload : { error: payload.error || "The file's tracks could not be read.", parts: [] });
        } catch (error) {
            if (node.__yue2MidiAsked !== asked) return;
            paint(node, { error: "The file's tracks could not be read: " + (error?.message || error), parts: [] });
        }
    }, ASK_DELAY);
}

function choose(node, name) {
    const widget = widgetNamed(node, MIDI);
    if (!widget) return;
    const values = widget.options?.values;
    if (Array.isArray(values) && !values.includes(name)) {
        values.push(name);
        values.sort((a, b) => String(a).toLowerCase().localeCompare(String(b).toLowerCase()));
    }
    widget.value = name;
    widget.callback?.(name);
    graphChanged();
    describe(node);
}

async function upload(node, file) {
    if (!roll.isMidiFile(file?.name)) {
        paint(node, { error: "'" + (file?.name || "") + "' is not a MIDI file: choose a .mid, .midi, .kar or .rmi file.",
            parts: [] });
        return false;
    }
    node.__yue2MidiAsked = (node.__yue2MidiAsked || 0) + 1;
    paint(node, { message: "Uploading " + file.name + "\u2026", parts: [] });
    try {
        const form = new FormData();
        form.append("image", file, file.name);
        form.append("type", "input");
        const response = await api.fetchApi(UPLOAD_ROUTE, { method: "POST", body: form });
        if (!response.ok) throw new Error(response.status + " " + response.statusText);
        const answer = await response.json();
        choose(node, answer.subfolder ? answer.subfolder + "/" + answer.name : answer.name);
        return true;
    } catch (error) {
        paint(node, { error: "The file could not be uploaded: " + (error?.message || error), parts: [] });
        return false;
    }
}

function pickFile(node) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = roll.MIDI_EXTENSIONS.join(",");
    input.style.display = "none";
    input.addEventListener("change", () => {
        const file = input.files?.[0];
        input.remove();
        if (file) upload(node, file);
    });
    document.body.appendChild(input);
    input.click();
}

function placeBeforeEditors(node, mine) {
    const widgets = node.widgets || [];
    for (const widget of mine) {
        const at = widgets.indexOf(widget);
        if (at >= 0) widgets.splice(at, 1);
    }
    const editors = widgets.findIndex((widget) => widget.name === FIRST_EDITOR_WIDGET);
    widgets.splice(editors >= 0 ? editors : widgets.length, 0, ...mine);
}

function watch(node, name) {
    const widget = widgetNamed(node, name);
    if (!widget || widget.__yue2MidiWatch) return;
    const original = widget.callback;
    widget.callback = function () {
        const result = original?.apply(this, arguments);
        describe(node);
        return result;
    };
    widget.__yue2MidiWatch = true;
}

function install(node) {
    installStyle(STYLE_ID, STYLE);
    const { widget: buttons } = buttonRow(node, BUTTONS, [
        { label: CHOOSE_LABEL, tooltip: CHOOSE_TOOLTIP, onClick: () => pickFile(node) },
    ]);
    const holder = element("div", "yue2-midi");
    holder.title = "The file's tracks, and which of them the score takes";
    node.__yue2MidiList = holder;
    const list = panelWidget(node, LIST, holder, () => Math.max(2, node.__yue2MidiRows || 2) * ROW_H + LIST_PAD, true);
    placeBeforeEditors(node, [buttons, list]);
    for (const name of [MIDI, MODE, VOCAL, INSTRUMENT]) watch(node, name);
    describe(node);
}

app.registerExtension({
    name: "yue2.load_midi",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== LOAD_MIDI) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                install(this);
            } catch (error) {
                console.error("[YuE2] the MIDI list could not be added to this node:", error);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            setTimeout(() => describe(this), 0);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const listed = message?.[MIDI_UI]?.[0];
            if (listed && typeof listed === "object" && isLoadMidi(this)) {
                clearTimeout(this.__yue2MidiTimer);
                this.__yue2MidiAsked = (this.__yue2MidiAsked || 0) + 1;
                paint(this, { parts: listed.parts || [], facts: listed.facts });
            }
            return result;
        };

        const onDragOver = nodeType.prototype.onDragOver;
        nodeType.prototype.onDragOver = function (event) {
            const items = Array.from(event?.dataTransfer?.items || []);
            if (items.some((item) => item.kind === "file")) return true;
            return onDragOver?.apply(this, arguments) ?? false;
        };

        const onDragDrop = nodeType.prototype.onDragDrop;
        nodeType.prototype.onDragDrop = function (event) {
            const file = Array.from(event?.dataTransfer?.files || []).find((item) => roll.isMidiFile(item.name));
            if (file) {
                upload(this, file);
                return true;
            }
            return onDragDrop?.apply(this, arguments) ?? false;
        };
    },
});
