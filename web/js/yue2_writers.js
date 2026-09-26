import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    ask, buttonRow, confirmed, element, frame, graphChanged, installStyle, warned, widgetNamed,
} from "./yue2_controls.js";

export const NODES = ["YuE2WriteSong", "YuE2Transcribe"];
const WIDGET = "model";
const PROBLEM_PREFIX = "!! ";
const TAIL = / \([^()]*\)$/;
const RUN_ENDS = ["execution_success", "execution_error", "execution_interrupted"];
const ROUTE = "/yue2/writers";
const BUTTONS = "yue2_model_list";
const STYLE_ID = "yue2-models-style";
const DOT = " \u00b7 ";

const LIST_LABEL = "Model list\u2026";
const LIST_TOOLTIP =
    "Add, edit and delete the models the 'model' list offers, and check one before it downloads. " +
    "Any GGUF with a chat template put into models/LLM is offered without being added.";
const INTRO =
    "The language models YuE2 Write Song and YuE2 Transcribe can use. A model from Hugging Face " +
    "downloads into models/LLM the first time a run picks it. The list is kept in writers.json in " +
    "ComfyUI's user folder, so it outlives an update of the pack.";
const FOUND_TEXT =
    "Found in your model folders and in Ollama. These are offered too, and there is nothing to edit: " +
    "they are files, not entries. Put any GGUF with a chat template into models/LLM and it appears here.";
const NONE_FOUND = "No other GGUF that can write was found in your model folders.";
const EMPTY = "Nothing in the list. 'Add a model' puts one here; the files found below are offered anyway.";
const LOCKED = "The list file cannot be written right now; the message at the top says why.";
const FORM_NOTE =
    "A model on Hugging Face: its repository and the .gguf file in it, downloaded into models/LLM the " +
    "first time a run picks it. A file already on this machine: leave the repository empty and give its " +
    "full path. Check it says what the model is before you save it.";
const BROKEN = "cannot be used: an entry needs a name and a .gguf file";
const FOLDER_LABEL = "Open the models folder";
const FOLDER_TOOLTIP = "Opens models/LLM on the machine ComfyUI runs on. A GGUF put there is offered at once.";
const FILE_TOOLTIP =
    "Opens writers.json itself, for what this window does not cover, such as a path to a network share. " +
    "It is written from the pack's list first if it does not exist yet.";
const REFRESH_TOOLTIP =
    "Reads the list and the model folders again and updates every 'model' list in this graph -- after a " +
    "model was copied in by hand, or writers.json was edited outside this window.";

const STYLE = `
.yue2-models { width: min(900px, 95vw); height: min(86vh, 860px); gap: 8px; }
.yue2-models.yue2-m-form { width: min(680px, 94vw); height: auto; max-height: calc(100vh - 48px); }
.yue2-m-title { font-size: 15px; font-weight: 600; }
.yue2-m-note { font-size: 12px; line-height: 1.45; color: var(--descrip-text, #999); }
.yue2-m-banner { font-size: 12px; line-height: 1.45; color: #E0A33E; padding: 6px 9px; border-radius: 6px;
    border: 1px solid #8a6a2a; background: rgba(224, 163, 62, 0.1); white-space: pre-line; }
.yue2-m-list { flex: 1 1 auto; min-height: 0; overflow: auto; display: flex; flex-direction: column;
    gap: 6px; padding-right: 4px; }
.yue2-m-card { display: flex; gap: 10px; align-items: flex-start; padding: 8px 10px; border-radius: 7px;
    border: 1px solid var(--border-color, #4e4e4e); background: var(--comfy-input-bg, #2b2b2b); }
.yue2-m-broken { border-style: dashed; }
.yue2-m-body { flex: 1 1 auto; min-width: 0; }
.yue2-m-head { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.yue2-m-name { font-size: 13px; font-weight: 600; word-break: break-word; }
.yue2-m-tag { flex: 0 0 auto; font-size: 10px; padding: 0 6px; border-radius: 999px;
    color: var(--descrip-text, #999); border: 1px solid var(--border-color, #4e4e4e); }
.yue2-m-tag.yue2-m-here { color: #7FB77F; border-color: #5f8f5f; }
.yue2-m-tag.yue2-m-fetch { color: #A9C7F0; border-color: #3B7DD8; }
.yue2-m-tag.yue2-m-gone { color: #E08A8A; border-color: #a05050; }
.yue2-m-facts { margin-top: 3px; font-size: 12px; color: var(--descrip-text, #bbb); }
.yue2-m-where { margin-top: 3px; font-size: 11px; color: var(--descrip-text, #999); word-break: break-all;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.yue2-m-acts { flex: 0 0 auto; display: flex; gap: 6px; }
.yue2-m-heading { display: flex; gap: 10px; align-items: flex-start; margin-top: 10px; padding-top: 10px;
    border-top: 1px solid var(--border-color, #4e4e4e); font-size: 12px; line-height: 1.45;
    color: var(--descrip-text, #999); }
.yue2-m-heading > div { flex: 1 1 auto; min-width: 0; }
.yue2-m-heading button { flex: 0 0 auto; white-space: nowrap; }
.yue2-m-path { margin-top: 3px; font-size: 11px; word-break: break-all;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.yue2-m-foundlist { display: flex; flex-direction: column; }
.yue2-m-found { padding: 2px 10px; font-size: 12px; line-height: 1.5; color: var(--descrip-text, #aaa);
    word-break: break-all; }
.yue2-m-foot { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.yue2-m-said { flex: 1 1 200px; min-width: 0; font-size: 12px; color: var(--descrip-text, #999); }
.yue2-m-failed { color: #E06C4E; }
.yue2-m-field { display: flex; flex-direction: column; gap: 3px; font-size: 11px;
    color: var(--descrip-text, #999); }
.yue2-m-field input { box-sizing: border-box; width: 100%; font: inherit; font-size: 13px; padding: 5px 8px;
    border-radius: 6px; color: var(--input-text, #ddd); background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-m-field input:focus { outline: none; border-color: #3B7DD8; }
.yue2-m-grid { display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 8px; }
.yue2-m-shows { padding: 7px 10px; border-radius: 6px; font-size: 13px; word-break: break-word;
    background: var(--comfy-input-bg, #2b2b2b); border: 1px solid var(--border-color, #4e4e4e); }
.yue2-m-shows span { display: block; margin-bottom: 3px; font-size: 10px; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--descrip-text, #999); }
.yue2-m-lines { display: flex; flex-direction: column; gap: 3px; font-size: 12px; line-height: 1.45; }
.yue2-m-good { color: #7FB77F; }
.yue2-m-warn { color: #E0A33E; }
.yue2-m-bad { color: #E08A8A; }
.yue2-m-note-line, .yue2-m-busy { color: var(--descrip-text, #999); }
.yue2-m-picks { display: flex; flex-wrap: wrap; gap: 5px; max-height: 160px; overflow: auto; }
.yue2-models button { font: inherit; font-size: 12px; padding: 4px 10px; border-radius: 6px; cursor: pointer;
    color: var(--input-text, #ddd); background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-models button:hover { background: var(--comfy-menu-bg, #353535); }
.yue2-models button.yue2-go { background: #3B7DD8; border-color: #3B7DD8; color: #fff; }
.yue2-models button.yue2-m-drop:hover { background: #7A2F27; border-color: #E06C4E; color: #fff; }
.yue2-models button.yue2-m-pick { font-size: 11px; padding: 2px 8px; }
.yue2-models button[disabled] { opacity: 0.45; cursor: not-allowed; }
`;

function stem(label) {
    return String(label ?? "").replace(TAIL, "").trim().toLowerCase();
}

function successor(value, offered) {
    if (typeof value !== "string" || !Array.isArray(offered) || offered.includes(value)) return value;
    if (value.startsWith(PROBLEM_PREFIX)) return value;
    const wanted = stem(value);
    const same = offered.filter((choice) => stem(choice) === wanted);
    return wanted && same.length === 1 ? same[0] : value;
}

function graphs() {
    const root = app.rootGraph ?? app.graph;
    if (!root) return [];
    return [root, ...(root.subgraphs?.values?.() ?? [])];
}

function modelNodes() {
    return graphs().flatMap((graph) => [...(graph.nodes ?? graph._nodes ?? [])])
        .filter((node) => NODES.includes(node.type));
}

function mend(node, offered) {
    const widget = widgetNamed(node, WIDGET);
    if (!widget) return false;
    if (offered) {
        widget.options = widget.options || {};
        widget.options.values = offered;
    }
    const next = successor(widget.value, widget.options?.values);
    if (next === widget.value) return false;
    widget.value = next;
    node.setDirtyCanvas?.(true, true);
    return true;
}

function applyChoices(offered, moves = []) {
    let changed = false;
    for (const node of modelNodes()) {
        const widget = widgetNamed(node, WIDGET);
        if (!widget) continue;
        for (const [from, to] of moves) {
            if (!from || !to || widget.value === to || stem(widget.value) !== from.toLowerCase()) continue;
            widget.value = to;
            node.setDirtyCanvas?.(true, true);
            changed = true;
        }
        changed = mend(node, Array.isArray(offered) ? offered : undefined) || changed;
    }
    if (changed) graphChanged();
    return changed;
}

function holding(name) {
    const wanted = String(name ?? "").toLowerCase();
    return modelNodes().filter((node) => stem(widgetNamed(node, WIDGET)?.value) === wanted).length;
}

function choicesIn(info, type) {
    const input = info?.[type]?.input;
    const spec = input?.required?.[WIDGET] ?? input?.optional?.[WIDGET];
    if (Array.isArray(spec?.[0])) return spec[0];
    if (spec?.[0] === "COMBO" && Array.isArray(spec[1]?.options)) return spec[1].options;
    return null;
}

async function freshen() {
    const nodes = modelNodes();
    if (!nodes.length) return;
    const type = nodes[0].type;
    let offered = null;
    try {
        const response = await api.fetchApi("/object_info/" + encodeURIComponent(type));
        if (response.ok) offered = choicesIn(await response.json(), type);
    } catch (error) {
        console.warn("[YuE2] the model list could not be read again:", error);
    }
    if (offered) applyChoices(offered);
}

function mendAll() {
    let changed = false;
    for (const node of modelNodes()) changed = mend(node) || changed;
    if (changed) graphChanged();
}

function button(text, className, title) {
    const made = element("button", className || "", text);
    made.type = "button";
    if (title) made.title = title;
    return made;
}

function gigabytes(value) {
    const number = Number(String(value ?? "").trim().replace(",", "."));
    return Number.isFinite(number) && number > 0 ? Math.round(number * 100) / 100 : 0;
}

function shownLabel(values, here) {
    const size = gigabytes(values.download_gb);
    const where = here ? "on disk" : !values.repo ? "not found" : "download" + (size ? " " + size + " GB" : "");
    return (values.name || "(no name yet)") + " (" + [where, values.vram].filter(Boolean).join(", ") + ")";
}

async function call(path, body) {
    try {
        const { ok, payload } = await ask(ROUTE + path, body);
        if (ok && payload?.ok !== false) return { payload, error: "" };
        return { payload, error: payload?.error || "That did not work." };
    } catch (error) {
        return { payload: null, error: "The ComfyUI server could not be reached: " + (error?.message || error) };
    }
}

async function reveal(what, said) {
    const { payload, error } = await call("/open", { what });
    if (error) {
        await warned(error + (payload?.path ? "\n\nOpen it by hand instead:\n" + payload.path : ""),
            { title: "It could not be opened" });
        return;
    }
    said?.(("Opened " + (payload?.path || "")).trim());
}

function field(holder, caption, value, hint) {
    const wrap = element("label", "yue2-m-field");
    wrap.appendChild(element("span", "", caption));
    const box = document.createElement("input");
    box.type = "text";
    box.spellcheck = false;
    box.autocomplete = "off";
    box.value = value === undefined || value === null ? "" : String(value);
    if (hint) box.placeholder = hint;
    wrap.appendChild(box);
    holder.appendChild(wrap);
    return box;
}

function openForm(held, onSaved) {
    installStyle(STYLE_ID, STYLE);
    const made = frame({ sticky: true });
    const panel = made.panel;
    panel.classList.add("yue2-models", "yue2-m-form");
    panel.appendChild(element("div", "yue2-m-title", held ? "Edit '" + (held.name || held.label) + "'" : "Add a model"));
    panel.appendChild(element("div", "yue2-m-note", FORM_NOTE));

    const name = field(panel, "Name -- what the list shows", held?.name, "Qwen3.5-9B Q4_K_M");
    const repo = field(panel, "Repository on Hugging Face -- empty for a file on this machine", held?.repo,
        "unsloth/Qwen3.5-9B-GGUF");
    const file = field(panel, "File -- the .gguf in the repository, or a full path on this machine", held?.file,
        "Qwen3.5-9B-Q4_K_M.gguf");
    const grid = element("div", "yue2-m-grid");
    const size = field(grid, "Download, GB", held?.download_gb, "5.29");
    const vram = field(grid, "VRAM note", held?.vram, "~7 GB VRAM");
    const note = field(grid, "Note", held?.note, "what it is good for");
    panel.appendChild(grid);

    const shows = element("div", "yue2-m-shows");
    shows.appendChild(element("span", "", "Shows in the list as"));
    const shown = element("div", "", "");
    shows.appendChild(shown);
    panel.appendChild(shows);

    const moved = element("div", "yue2-m-lines");
    const lines = element("div", "yue2-m-lines");
    const picks = element("div", "yue2-m-picks");
    panel.append(moved, lines, picks);

    const foot = element("div", "yue2-m-foot");
    const trouble = element("div", "yue2-m-said yue2-m-failed", "");
    const probe = button("Check it", "", "Reads the file if it is on this machine, or asks Hugging Face "
        + "whether the repository has it and how big it is. With a repository and no file, lists its models.");
    const cancel = button("Cancel");
    const keep = button("Save", "yue2-go");
    foot.append(trouble, probe, cancel, keep);
    panel.appendChild(foot);

    let checked = null;

    function values() {
        return {
            name: name.value.trim(), repo: repo.value.trim(), file: file.value.trim(),
            download_gb: size.value.trim(), vram: vram.value.trim(), note: note.value.trim(),
        };
    }

    function here(now) {
        if (checked && checked.repo === now.repo && checked.file === now.file) return checked.here;
        if (held && (held.repo || "") === now.repo && (held.file || "") === now.file) return Boolean(held.on_disk);
        return !now.repo;
    }

    function redraw() {
        const now = values();
        shown.textContent = shownLabel(now, here(now));
        moved.replaceChildren();
        if (held?.name && now.name && now.name.toLowerCase() !== held.name.toLowerCase()) {
            moved.appendChild(element("div", "yue2-m-warn",
                "The list reads '" + held.label + "' today. Saving renames it: workflows saved with the old "
                + "name no longer find it. The graph open here moves to the new name; saved workflows do not."));
        }
    }

    function paint(payload) {
        lines.replaceChildren();
        picks.replaceChildren();
        for (const line of payload?.lines || []) {
            const level = line.level === "note" ? "note-line" : line.level;
            lines.appendChild(element("div", "yue2-m-" + level, line.text));
        }
        const said = gigabytes(size.value);
        if (payload?.download_gb) {
            if (!said) size.value = String(payload.download_gb);
            else if (Math.abs(said - payload.download_gb) > 0.01) {
                lines.appendChild(element("div", "yue2-m-warn", "The file is " + payload.download_gb
                    + " GB and the form says " + said + ". Saving keeps what the form says."));
            }
        }
        for (const one of payload?.files || []) {
            const chip = button(one.file + " (" + one.gb + " GB)", "yue2-m-pick", "Use this file");
            chip.addEventListener("click", () => {
                file.value = one.file;
                size.value = String(one.gb);
                runCheck();
            });
            picks.appendChild(chip);
        }
    }

    async function runCheck() {
        const now = values();
        probe.disabled = true;
        picks.replaceChildren();
        lines.replaceChildren(element("div", "yue2-m-busy", "Looking\u2026"));
        const { payload, error } = await call("/check", { entry: now });
        probe.disabled = false;
        if (error) {
            lines.replaceChildren(element("div", "yue2-m-bad", error));
            return;
        }
        checked = { repo: now.repo, file: now.file, here: Boolean(payload.here) };
        paint(payload);
        redraw();
    }

    for (const box of [name, repo, file, size, vram, note]) box.addEventListener("input", redraw);
    probe.addEventListener("click", runCheck);
    cancel.addEventListener("click", () => made.close());
    keep.addEventListener("click", async () => {
        keep.disabled = true;
        trouble.textContent = "";
        const { payload, error } = await call("/save", { was: held?.name || "", entry: values() });
        keep.disabled = false;
        if (error) {
            trouble.textContent = error;
            return;
        }
        applyChoices(payload.choices, payload.renamed ? [[payload.renamed, payload.label]] : []);
        made.close();
        onSaved?.(payload);
    });
    redraw();
    name.focus();
    return made;
}

export function openModelList() {
    installStyle(STYLE_ID, STYLE);
    const made = frame({});
    const panel = made.panel;
    panel.classList.add("yue2-models");
    panel.appendChild(element("div", "yue2-m-title", "Model list"));
    panel.appendChild(element("div", "yue2-m-note", INTRO));
    const banner = element("div", "yue2-m-banner");
    banner.hidden = true;
    panel.appendChild(banner);
    const list = element("div", "yue2-m-list");
    panel.appendChild(list);

    const foot = element("div", "yue2-m-foot");
    const said = element("div", "yue2-m-said", "Reading the list\u2026");
    const adding = button("Add a model", "yue2-go");
    const restoring = button("Restore the packaged entries");
    const opening = button("Open writers.json", "", FILE_TOOLTIP);
    const refreshing = button("Refresh", "", REFRESH_TOOLTIP);
    const shutting = button("Close");
    foot.append(said, adding, restoring, opening, refreshing, shutting);
    panel.appendChild(foot);

    let state = null;
    let message = "";
    let failed = false;

    function tell(text, bad = false) {
        message = text;
        failed = bad;
        paintSaid();
    }

    function paintSaid() {
        said.classList.toggle("yue2-m-failed", failed);
        if (message) {
            said.textContent = message;
            return;
        }
        if (!state) return;
        const entries = state.entries.length;
        said.textContent = entries + (entries === 1 ? " entry" : " entries") + DOT + state.found.length
            + (state.found.length === 1 ? " file found" : " files found");
    }

    function statusTag(row) {
        if (!row.usable) return element("span", "yue2-m-tag yue2-m-gone", BROKEN);
        if (row.on_disk) {
            const tag = element("span", "yue2-m-tag yue2-m-here", "on disk");
            tag.title = (row.where || "") + "\n\nAlready here, so picking it downloads nothing.";
            return tag;
        }
        if (row.local) return element("span", "yue2-m-tag yue2-m-gone", "not found");
        const size = gigabytes(row.download_gb);
        const tag = element("span", "yue2-m-tag yue2-m-fetch", "download" + (size ? " " + size + " GB" : ""));
        tag.title = "Downloads into " + (state?.folder || "models/LLM") + " the first time a run picks it.";
        return tag;
    }

    function card(row) {
        const box = element("div", "yue2-m-card" + (row.usable ? "" : " yue2-m-broken"));
        const body = element("div", "yue2-m-body");
        const head = element("div", "yue2-m-head");
        if (row.from_pack) head.appendChild(element("span", "yue2-m-tag", "from the pack"));
        head.appendChild(statusTag(row));
        head.appendChild(element("span", "yue2-m-name", row.name || row.label));
        body.appendChild(head);
        const facts = [row.vram, row.note].filter(Boolean).join(DOT);
        if (facts) body.appendChild(element("div", "yue2-m-facts", facts));
        const where = row.local ? row.file : [row.repo, row.file].filter(Boolean).join("  ");
        if (where) body.appendChild(element("div", "yue2-m-where", where));
        box.appendChild(body);
        const acts = element("div", "yue2-m-acts");
        const change = button("Edit");
        const drop = button("Delete", "yue2-m-drop");
        change.addEventListener("click", () => openForm(row, (payload) => {
            tell("Saved '" + payload.label + "'.");
            load();
        }));
        drop.addEventListener("click", () => erase(row, drop));
        if (!state.writable) {
            for (const one of [change, drop]) {
                one.disabled = true;
                one.title = LOCKED;
            }
        }
        acts.append(change, drop);
        box.appendChild(acts);
        return box;
    }

    function draw() {
        list.replaceChildren();
        banner.hidden = !state?.problem;
        if (state?.problem) {
            banner.textContent = state.problem + ".\n\nThe pack's own list is shown and nothing here can be "
                + "changed until the file parses again. Open writers.json to mend it, or delete the file to "
                + "start again from the pack's list.";
        }
        if (!state) {
            paintSaid();
            return;
        }
        if (!state.entries.length) list.appendChild(element("div", "yue2-m-note", EMPTY));
        for (const row of state.entries) list.appendChild(card(row));
        const heading = element("div", "yue2-m-heading");
        const words = element("div", "", FOUND_TEXT);
        if (state.folder) words.appendChild(element("div", "yue2-m-path", state.folder));
        heading.appendChild(words);
        const folder = button(FOLDER_LABEL, "", FOLDER_TOOLTIP);
        folder.addEventListener("click", () => reveal("folder", tell));
        heading.appendChild(folder);
        list.appendChild(heading);
        const found = element("div", "yue2-m-foundlist");
        if (!state.found.length) found.appendChild(element("div", "yue2-m-found", NONE_FOUND));
        for (const row of state.found) {
            const line = element("div", "yue2-m-found", row.label);
            line.title = row.path;
            found.appendChild(line);
        }
        list.appendChild(found);
        const coming = state.restorable || [];
        restoring.disabled = !coming.length || !state.writable;
        restoring.title = !state.writable ? LOCKED : coming.length
            ? "Puts back: " + coming.join(", ") : "Every entry the pack ships is already in the list.";
        adding.disabled = !state.writable;
        adding.title = state.writable ? "" : LOCKED;
        paintSaid();
    }

    async function load() {
        const { payload, error } = await call("");
        if (error) {
            tell(error, true);
            return;
        }
        state = payload;
        applyChoices(payload.choices);
        draw();
    }

    async function erase(row, drop) {
        const name = row.name || row.label;
        const back = row.from_pack
            ? "It is one of the pack's own, so 'Restore the packaged entries' brings it back."
            : "It is yours: only typing it again brings it back.";
        const after = row.on_disk
            ? "Its file stays on disk. Where the file is in a model folder, it is offered under its own "
              + "name again, and nodes here that use this entry move to it."
            : "Nodes that use it keep it until another model is picked there, and a run before then stops "
              + "and says the model is gone.";
        if (!(await confirmed(back + "\n\n" + after, { title: "Delete '" + name + "' from the list?",
            ok: "Delete", danger: true }))) return;
        drop.disabled = true;
        const { payload, error } = await call("/delete", { name });
        if (error) {
            drop.disabled = false;
            tell(error, true);
            return;
        }
        applyChoices(payload.choices, payload.moved_to ? [[name, payload.moved_to]] : []);
        const left = holding(name);
        tell(left ? left + (left === 1 ? " node here still names it" : " nodes here still name it")
            + "; pick another model there before the next run." : "Deleted '" + name + "'.", Boolean(left));
        load();
    }

    adding.addEventListener("click", () => openForm(null, (payload) => {
        tell("Added '" + payload.label + "'.");
        load();
    }));
    restoring.addEventListener("click", async () => {
        restoring.disabled = true;
        const { payload, error } = await call("/restore", {});
        if (error) {
            tell(error, true);
            restoring.disabled = false;
            return;
        }
        applyChoices(payload.choices);
        tell(payload.restored?.length ? "Put back: " + payload.restored.join(", ") + "." : "Nothing was missing.");
        load();
    });
    opening.addEventListener("click", () => reveal("list", tell));
    refreshing.addEventListener("click", async () => {
        refreshing.disabled = true;
        message = "";
        await load();
        refreshing.disabled = false;
    });
    shutting.addEventListener("click", () => made.close());
    load();
    return made;
}

app.registerExtension({
    name: "yue2.writers",
    setup() {
        for (const name of RUN_ENDS) {
            api.addEventListener(name, () => {
                freshen().catch((error) => console.warn("[YuE2] the model list was not refreshed:", error));
            });
        }
    },
    async refreshComboInNodes() {
        mendAll();
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODES.includes(nodeData.name)) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                buttonRow(this, BUTTONS, [{ label: LIST_LABEL, tooltip: LIST_TOOLTIP, onClick: () => openModelList() }]);
            } catch (error) {
                console.error("[YuE2] the Model list button could not be added to this node:", error);
            }
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            try {
                mend(this);
            } catch (error) {
                console.warn("[YuE2] the saved model could not be matched to the list:", error);
            }
            return result;
        };
    },
});
