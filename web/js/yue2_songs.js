import { ask, confirmed, element, frame, installStyle } from "./yue2_controls.js";
import * as roll from "./yue2_roll.js";

const ROUTE = "/yue2/songs";
const NOTE_ROUTE = "/yue2/songs/note";
const DROP_ROUTE = "/yue2/songs/drop";
const SOUNDS_ROUTE = "/yue2/songs/sounds";
const STYLE_ID = "yue2-songs-style";
const SVG = "http://www.w3.org/2000/svg";
const DOT = " \u00b7 ";
const WIDE = 1000;
const TWICE = 400;
const NOTE_LETTERS = 80;

const HELD =
    "Every song this install sings is remembered with what an edit needs: the performance, its "
    + "score and its latents. Opening one here brings its sound back without singing it again, "
    + "which is a second or two rather than the minutes a song takes. Double-click a row to "
    + "open it at once.";

const EMPTY =
    "Nothing is remembered yet. Sing a song with 'YuE2 Generate Song', or load a FLAC this pack "
    + "saved with 'Save Audio', and it will be in this list.";

const NO_MATCH = "No song here says that.";

const JOINED =
    "A song is joined to this node's 'audio', and that one wins: the graph is what a run is "
    + "about. So nothing here can be opened while it is plugged in -- untick the square beside "
    + "'audio' on the node, or unplug it, to edit a song chosen here. Everything else on this "
    + "list still works: what is remembered can be read, labelled and deleted from here.";

const VOICE_ONLY = "voice only";
const VOICE_WHY =
    "This song was made with 'vocals_only': what it holds is the whole mix while what you hear "
    + "is the voice, so an edit would lay mixed sound into a voice-only track. Edit the song it "
    + "came from and take the voice afterwards.";

const NO_BARS = "no score";
const NO_BARS_WHY =
    "Sung with 'cot' set to 'off', so it has no bars. It can still be edited, by selecting "
    + "seconds instead of bars.";

const ON_THE_NODE = "on this node now";

const FOLDED = "The edits made of this song, oldest first. The bar is the song, and the stretch "
    + "the edit changed is marked on it.";

const NEW_WORDS = "text replace";

const WAS_SUNG = "These words were sung here:";

const IS_SUNG = "and these were put in their place:";

const OPEN_IT = "Show the edits made of this song";
const SHUT_IT = "Fold the edits away";

const NOTE_NEW = "Write a word on this one";
const NOTE_AGAIN = "Change the word written on this one";
const NOTE_ASK = "a word about this one";
const NOTE_WHY =
    "What you wrote on this song. It is kept beside the song, so it costs nothing to change "
    + "and it is found by the search above.";

const DROP_ONE = "Delete this edit, and the files kept with it";
const DROP_ALL = "Delete this song, its edits, and the files kept with them";
const DROP_WHAT =
    "The song's own file goes, and so do its sound, its note and the place the score was "
    + "measured to sit on it. This cannot be undone.";

const SOUNDS_LABEL = "Keep each song's sound beside it";
const SOUNDS_WHY =
    "A song is remembered as its latents, and opening one turns them back into sound, which "
    + "takes a second or two of card. With this on, the sound itself is kept beside the song as "
    + "a FLAC -- half the size of the samples, and read back in a fifth of a second -- so "
    + "opening a song is a read and no card at all. It is also the file the node handed on, to "
    + "the bit.\n\nIt is written for every song sung or edited from now on, and for a song "
    + "already remembered the first time it is opened, which pays that one decode once. A sound "
    + "can always be made again from the song, so deleting them loses nothing.";

const NO_SWITCH = {
    can: false,
    why: "This ComfyUI was started before the pack learned to keep sounds. Restart it.",
};

const NOTHING_KEPT = "nothing kept yet";
const SWEEP_LABEL = "Delete them";
const SWEEP_WHY =
    "Delete every sound kept beside a song. The songs stay: a sound is a decode away from being "
    + "back.";

function when(used) {
    const at = new Date(Number(used || 0) * 1000);
    if (!Number.isFinite(at.getTime())) return "";
    const day = at.toLocaleDateString(undefined, { day: "numeric", month: "short" });
    const hour = at.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
    return day + " " + hour;
}

function born(row) {
    return Number(row.created) || Number(row.used) || 0;
}

function size(bytes) {
    const held = Number(bytes) || 0;
    if (held >= (1 << 30)) return (held / (1 << 30)).toFixed(1) + " GB";
    if (held >= (1 << 20)) return Math.round(held / (1 << 20)) + " MB";
    return Math.max(1, Math.round(held / 1024)) + " KB";
}

function words(row) {
    const lines = String(row.lyrics || "").split("\n")
        .map((line) => line.trim())
        .filter((line) => line && !(line.startsWith("[") && line.endsWith("]")));
    if (!lines.length) return "no words";
    return lines.slice(0, 3).join(" / ") + (lines.length > 3 || row.whole === false ? " ..." : "");
}

function facts(row) {
    return [row.origin || "somewhere", roll.clock(row.seconds || 0),
            "seed " + row.seed].join(DOT);
}

function where(mark) {
    if (Array.isArray(mark.bars)) return "bars " + mark.bars[0] + "-" + mark.bars[1];
    const at = Array.isArray(mark.at) ? mark.at : [0, 0];
    return roll.clock(at[0]) + (mark.op === "cut" ? "" : "-" + roll.clock(at[1]));
}

function what(row) {
    const marks = Array.isArray(row.edit) ? row.edit : [];
    if (!marks.length) return "as it was sung";
    return marks.map((mark) => {
        const at = Array.isArray(mark.at) ? mark.at : [0, 0];
        if (mark.op === "cut") return "cut " + where(mark) + DOT + "-" + roll.clock(mark.took || 0);
        const did = mark.op === "words" ? "new words " : "retake ";
        return did + where(mark) + DOT + roll.clock(Math.max(0, at[1] - at[0]));
    }).join(", ");
}

function swapSaid(mark) {
    return [WAS_SUNG, String(mark.was || "").trim() || "nothing",
            IS_SUNG, String(mark.now || "").trim() || "nothing"].join("\n");
}

function shape(tag, cls) {
    const drawn = document.createElementNS(SVG, tag);
    drawn.setAttribute("class", cls);
    return drawn;
}

function bytes(text) {
    if (!text) return [];
    let raw = "";
    try {
        raw = atob(String(text));
    } catch (error) {
        return [];
    }
    const out = new Array(raw.length);
    for (let index = 0; index < raw.length; index += 1) out[index] = raw.charCodeAt(index);
    return out;
}

function envelope(values) {
    const step = WIDE / values.length;
    const top = [];
    const under = [];
    for (let index = 0; index < values.length; index += 1) {
        const half = Math.max(0.5, (Number(values[index]) / 255) * 49);
        const x = (index * step).toFixed(2);
        top.push(x + " " + (50 - half).toFixed(2));
        under.push(x + " " + (50 + half).toFixed(2));
    }
    under.reverse();
    return "M " + top.join(" L ") + " L " + under.join(" L ") + " Z";
}

function band(mark, seconds) {
    const at = Array.isArray(mark.at) ? mark.at : null;
    if (!at || !seconds) return null;
    const from = Math.max(0, Math.min(WIDE, (Number(at[0]) / seconds) * WIDE));
    const to = Math.max(from, Math.min(WIDE, (Number(at[1]) / seconds) * WIDE));
    const cut = mark.op === "cut";
    const drawn = shape("rect", cut ? "yue2-s-seam" : "yue2-s-span");
    drawn.setAttribute("x", (cut ? Math.max(0, from - 2) : from).toFixed(2));
    drawn.setAttribute("y", "0");
    drawn.setAttribute("width", (cut ? 4 : Math.max(3, to - from)).toFixed(2));
    drawn.setAttribute("height", "100");
    return drawn;
}

function picture(row) {
    const box = shape("svg", "yue2-s-strip");
    box.setAttribute("viewBox", "0 0 " + WIDE + " 100");
    box.setAttribute("preserveAspectRatio", "none");
    const seconds = Number(row.seconds) || 0;
    for (const mark of Array.isArray(row.edit) ? row.edit : []) {
        const drawn = band(mark, seconds);
        if (drawn) box.appendChild(drawn);
    }
    const peaks = bytes(row.peaks);
    const body = bytes(row.body);
    if (!peaks.length) {
        const flat = shape("rect", "yue2-s-flat");
        flat.setAttribute("x", "0");
        flat.setAttribute("y", "49");
        flat.setAttribute("width", String(WIDE));
        flat.setAttribute("height", "2");
        box.appendChild(flat);
        return box;
    }
    const outer = shape("path", "yue2-s-peaksline");
    outer.setAttribute("d", envelope(peaks));
    box.appendChild(outer);
    if (body.length) {
        const inner = shape("path", "yue2-s-bodyline");
        inner.setAttribute("d", envelope(body));
        box.appendChild(inner);
    }
    return box;
}

function badges(row, chosen) {
    const said = [];
    const swap = (Array.isArray(row.edit) ? row.edit : [])
        .filter((mark) => mark.op === "words" && (mark.was || mark.now));
    if (swap.length) {
        said.push([NEW_WORDS, "yue2-s-swap-mark", swap.map(swapSaid).join("\n\n")]);
    }
    if (row.key === chosen) said.push([ON_THE_NODE, "yue2-s-mark", ""]);
    if (row.voice) said.push([VOICE_ONLY, "yue2-s-warn-mark", VOICE_WHY]);
    else if (!row.bars) said.push([NO_BARS, "yue2-s-mark", NO_BARS_WHY]);
    return said.map(([text, cls, why]) => {
        const tag = element("span", cls, text);
        if (why) tag.title = why;
        return tag;
    });
}

function matches(row, wanted) {
    if (!wanted) return true;
    const swaps = (Array.isArray(row.edit) ? row.edit : [])
        .map((mark) => [mark.was || "", mark.now || ""].join(" ")).join(" ");
    const hay = [row.style, row.lyrics, row.origin, String(row.seed), row.note, what(row), swaps]
        .join(" ").toLowerCase();
    return wanted.split(/\s+/).every((word) => hay.includes(word));
}

function iconButton(text, title, onClick) {
    const button = element("button", "yue2-s-icon", text);
    button.title = title;
    button.addEventListener("click", (event) => {
        event.stopPropagation();
        onClick();
    });
    return button;
}

function families(shown) {
    const held = new Map();
    const order = [];
    for (const row of shown) {
        const id = row.root || row.key;
        let line = held.get(id);
        if (!line) {
            line = { id, head: null, kids: [], used: 0 };
            held.set(id, line);
            order.push(line);
        }
        if (row.key === id) line.head = row;
        else line.kids.push(row);
        line.used = Math.max(line.used, Number(row.used) || 0);
    }
    for (const line of order) {
        line.kids.sort((one, two) => born(one) - born(two));
        if (!line.head) line.head = line.kids.shift() || null;
    }
    order.sort((one, two) => two.used - one.used);
    return order.filter((line) => line.head);
}

export function openSongs(node, chosen, joined, onPick, onDrop) {
    installStyle(STYLE_ID, STYLE);
    const made = frame({});
    const panel = made.panel;
    panel.classList.add("yue2-songs");

    panel.appendChild(element("div", "yue2-s-title", "Saved songs"));
    panel.appendChild(element("div", "yue2-s-note", HELD));
    if (joined) panel.appendChild(element("div", "yue2-s-warn", JOINED));

    const find = document.createElement("input");
    find.type = "text";
    find.className = "yue2-s-find";
    find.placeholder = "Find by style, words, seed, note or where it came from";
    panel.appendChild(find);

    const keepRow = element("div", "yue2-s-keep");
    const keepLabel = element("label", "yue2-s-keeplabel");
    const keepBox = document.createElement("input");
    keepBox.type = "checkbox";
    keepLabel.appendChild(keepBox);
    keepLabel.appendChild(element("span", "", SOUNDS_LABEL));
    keepLabel.title = SOUNDS_WHY;
    const keepSaid = element("span", "yue2-s-dim", "");
    const sweep = element("button", "", SWEEP_LABEL);
    sweep.title = SWEEP_WHY;
    keepRow.appendChild(keepLabel);
    keepRow.appendChild(keepSaid);
    keepRow.appendChild(sweep);
    panel.appendChild(keepRow);

    const listBox = element("div", "yue2-s-list");
    panel.appendChild(listBox);

    const foot = element("div", "yue2-s-foot");
    const said = element("div", "yue2-s-said", "Reading the songs...");
    const open = element("button", "yue2-go", "Open this song");
    const close = element("button", "", "Close");
    open.disabled = true;
    if (joined) open.title = JOINED;
    open.addEventListener("click", () => take());
    close.addEventListener("click", () => made.close());
    foot.appendChild(said);
    foot.appendChild(open);
    foot.appendChild(close);
    panel.appendChild(foot);

    let rows = [];
    let picked = chosen || "";
    let mine = joined ? "" : (chosen || "");
    let unfolded = "";
    let trouble = "";
    let keeping = {};
    let writing = "";
    let busy = false;
    let struck = { key: "", at: 0 };

    function tell(shown) {
        const row = shown.find((one) => one.key === picked);
        said.classList.toggle("yue2-s-failed", Boolean(trouble));
        if (trouble) {
            said.textContent = trouble;
        } else if (busy) {
            said.textContent = "Working...";
        } else if (row && row.voice) {
            said.textContent = VOICE_WHY;
        } else if (shown.length !== rows.length) {
            said.textContent = shown.length + " of " + rows.length + " shown";
        } else {
            said.textContent = rows.length === 1 ? "1 song remembered"
                : rows.length + " songs remembered";
        }
    }

    function openable(row) {
        return Boolean(row) && !row.voice && !joined;
    }

    function take() {
        const row = rows.find((one) => one.key === picked);
        if (!openable(row)) return;
        made.close();
        onPick(row);
    }

    function pick(row) {
        picked = row.key;
        open.disabled = !openable(row);
        draw();
    }

    function struckTwice(row) {
        const now = Date.now();
        const again = struck.key === row.key && now - struck.at < TWICE;
        struck = { key: row.key, at: again ? 0 : now };
        if (!again) return false;
        picked = row.key;
        take();
        return true;
    }

    function noteBox(row) {
        const box = document.createElement("input");
        box.type = "text";
        box.className = "yue2-s-notebox";
        box.value = row.note || "";
        box.maxLength = NOTE_LETTERS;
        box.placeholder = NOTE_ASK;
        let over = false;
        const end = (keepIt) => {
            if (over) return;
            over = true;
            const typed = box.value;
            setTimeout(() => {
                if (keepIt) writeNote(row, typed);
                else {
                    writing = "";
                    draw();
                }
            }, 0);
        };
        box.addEventListener("click", (event) => event.stopPropagation());
        box.addEventListener("keydown", (event) => {
            event.stopPropagation();
            if (event.key === "Enter") end(true);
            else if (event.key === "Escape") end(false);
        });
        box.addEventListener("blur", () => end(true));
        return box;
    }

    function noted(row) {
        if (writing === row.key) return noteBox(row);
        if (!row.note) return null;
        const tag = element("span", "yue2-s-said-note", row.note);
        tag.title = NOTE_WHY;
        return tag;
    }

    function acts(row, whole) {
        const box = element("span", "yue2-s-acts");
        box.appendChild(iconButton("\u270e", row.note ? NOTE_AGAIN : NOTE_NEW, () => {
            writing = row.key;
            draw();
        }));
        const bin = iconButton("\u2715", whole ? DROP_ALL : DROP_ONE, () => erase(row, whole));
        bin.classList.add("yue2-s-bin");
        box.appendChild(bin);
        return box;
    }

    function headRow(line, shownKids, forced) {
        const row = line.head;
        const box = element("div", "yue2-s-row" + (row.key === picked ? " yue2-s-picked" : "")
            + (row.voice ? " yue2-s-barred" : ""));
        const head = element("div", "yue2-s-head");
        if (shownKids.length) {
            const turn = element("button", "yue2-s-turn",
                forced || unfolded === line.id ? "\u25be" : "\u25b8");
            turn.title = forced || unfolded === line.id ? SHUT_IT : OPEN_IT;
            turn.addEventListener("click", (event) => {
                event.stopPropagation();
                unfolded = unfolded === line.id ? "" : line.id;
                draw();
            });
            head.appendChild(turn);
        }
        head.appendChild(element("span", "yue2-s-when", when(born(row))));
        const word = noted(row);
        if (word) head.appendChild(word);
        for (const tag of badges(row, mine)) head.appendChild(tag);
        if (shownKids.length) {
            head.appendChild(element("span", "yue2-s-count", shownKids.length === 1
                ? "1 edit" : shownKids.length + " edits"));
        }
        head.appendChild(acts(row, true));
        box.appendChild(head);
        box.appendChild(element("div", "yue2-s-style", row.style || "no style given"));
        const sung = element("div", "yue2-s-words", words(row));
        sung.title = row.lyrics || "";
        box.appendChild(sung);
        box.appendChild(element("div", "yue2-s-facts", facts(row)));
        box.addEventListener("click", () => {
            if (struckTwice(row)) return;
            unfolded = line.id;
            pick(row);
        });
        return box;
    }

    function kidRow(row) {
        const box = element("div", "yue2-s-kid" + (row.key === picked ? " yue2-s-picked" : "")
            + (row.voice ? " yue2-s-barred" : ""));
        const head = element("div", "yue2-s-head");
        head.appendChild(element("span", "yue2-s-when", when(born(row))));
        head.appendChild(element("span", "yue2-s-did", what(row)));
        const word = noted(row);
        if (word) head.appendChild(word);
        head.appendChild(element("span", "yue2-s-long", roll.clock(row.seconds || 0)));
        for (const tag of badges(row, mine)) head.appendChild(tag);
        head.appendChild(acts(row, false));
        box.appendChild(head);
        box.appendChild(picture(row));
        box.addEventListener("click", () => {
            if (struckTwice(row)) return;
            pick(row);
        });
        return box;
    }

    function draw() {
        listBox.replaceChildren();
        const wanted = find.value.trim().toLowerCase();
        const shown = rows.filter((row) => matches(row, wanted));
        if (!rows.length) {
            listBox.appendChild(element("div", "yue2-s-empty", EMPTY));
        } else if (!shown.length) {
            listBox.appendChild(element("div", "yue2-s-empty", NO_MATCH));
        }
        for (const line of families(shown)) {
            const group = element("div", "yue2-s-family");
            group.appendChild(headRow(line, line.kids, Boolean(wanted)));
            if (line.kids.length && (wanted || unfolded === line.id)) {
                const under = element("div", "yue2-s-kids");
                under.title = FOLDED;
                for (const row of line.kids) under.appendChild(kidRow(row));
                group.appendChild(under);
            }
            listBox.appendChild(group);
        }
        paintKeep();
        tell(shown);
        if (writing) {
            const box = listBox.querySelector(".yue2-s-notebox");
            if (box) {
                box.focus();
                box.select();
            }
        }
    }

    function paintKeep() {
        const can = keeping.can !== false;
        keepBox.checked = Boolean(keeping.on);
        keepBox.disabled = busy || !can;
        const count = Number(keeping.count) || 0;
        if (!can) keepSaid.textContent = keeping.why || "";
        else if (!count) keepSaid.textContent = NOTHING_KEPT;
        else {
            keepSaid.textContent = count + (count === 1 ? " song" : " songs") + DOT
                + size(keeping.bytes) + " of " + size(keeping.budget);
        }
        sweep.hidden = !count;
        sweep.disabled = busy;
    }

    async function writeNote(row, text) {
        writing = "";
        const was = row.note || "";
        const wanted = String(text || "").trim().slice(0, NOTE_LETTERS);
        row.note = wanted;
        draw();
        if (wanted === was) return;
        try {
            const { ok, payload } = await ask(NOTE_ROUTE, { key: row.key, note: wanted });
            if (ok && payload?.ok !== false) row.note = String(payload.note || "");
            else {
                row.note = was;
                trouble = payload?.error || "The note could not be written.";
            }
        } catch (error) {
            row.note = was;
            trouble = "The note could not be written: " + (error?.message || error);
        }
        draw();
    }

    async function erase(row, whole) {
        const line = families(rows).find((one) => one.head.key === row.key);
        const kids = whole && line ? line.kids.length : 0;
        const asked = whole
            ? "Delete this song" + (kids ? " and the " + kids + (kids === 1 ? " edit" : " edits")
                + " made of it" : "") + "?"
            : "Delete this edit?";
        const said = (whole ? row.style || "no style given" : what(row)) + "\n\n" + DROP_WHAT;
        if (!(await confirmed(said, { title: asked, ok: "Delete", danger: true }))) return;
        busy = true;
        draw();
        let gone = [];
        let failed = "";
        try {
            const { ok, payload } = await ask(DROP_ROUTE,
                { key: row.key, family: Boolean(whole) });
            if (ok && payload?.ok !== false) {
                gone = Array.isArray(payload.keys) ? payload.keys : [];
                keeping = payload.sounds || keeping;
            } else failed = payload?.error || "The song could not be deleted.";
        } catch (error) {
            failed = "The song could not be deleted: " + (error?.message || error);
        }
        busy = false;
        await load();
        if (failed) trouble = failed;
        if (gone.includes(picked)) picked = "";
        if (gone.includes(mine)) mine = "";
        if (gone.length) onDrop?.(gone);
        open.disabled = !openable(rows.find((one) => one.key === picked));
        draw();
    }

    async function setSounds(body) {
        busy = true;
        if (typeof body.on === "boolean") keeping = Object.assign({}, keeping, { on: body.on });
        draw();
        try {
            const { ok, payload } = await ask(SOUNDS_ROUTE, body);
            if (ok && payload?.ok !== false) {
                keeping = payload;
                trouble = "";
            } else trouble = payload?.error || "That could not be done.";
        } catch (error) {
            trouble = "That could not be done: " + (error?.message || error);
        }
        busy = false;
        if (body.sweep) await load();
        draw();
    }

    async function load() {
        let failed = "";
        try {
            const { ok, payload } = await ask(ROUTE);
            if (ok && payload?.ok !== false && Array.isArray(payload?.songs)) {
                rows = payload.songs;
                keeping = payload.sounds || NO_SWITCH;
            } else failed = payload?.error || "The songs could not be read.";
        } catch (error) {
            failed = "The songs could not be read: " + (error?.message || error);
        }
        trouble = failed;
        return !failed;
    }

    find.addEventListener("input", draw);
    find.addEventListener("keydown", (event) => {
        if (event.key === "Enter") take();
    });
    keepBox.addEventListener("change", () => setSounds({ on: keepBox.checked }));
    sweep.addEventListener("click", async () => {
        if (await confirmed("The songs stay as they are. A sound is a decode away from being "
            + "back.", { title: "Delete the " + size(keeping.bytes) + " of sound kept beside "
                                + "these songs?", ok: "Delete them", danger: true })) {
            setSounds({ sweep: true });
        }
    });
    made.handle.onEscape = () => {
        if (!writing) {
            made.close();
            return;
        }
        writing = "";
        draw();
    };

    (async () => {
        await load();
        const held = rows.find((row) => row.key === picked);
        unfolded = held ? (held.root || held.key) : "";
        open.disabled = Boolean(trouble) || !openable(held);
        draw();
        listBox.querySelector(".yue2-s-picked")?.scrollIntoView({ block: "nearest" });
        find.focus();
    })();

    return made;
}

const STYLE = `
.yue2-songs { width: min(980px, 96vw); height: min(84vh, 820px); gap: 8px; }
.yue2-s-title { font-size: 15px; font-weight: 600; }
.yue2-s-note, .yue2-s-facts, .yue2-s-words, .yue2-s-empty, .yue2-s-said, .yue2-s-dim {
    color: var(--descrip-text, #999); font-size: 12px; }
.yue2-s-note { line-height: 1.45; }
.yue2-s-warn { color: #E0A33E; font-size: 12px; line-height: 1.45; }
.yue2-s-find { box-sizing: border-box; width: 100%; font: inherit; font-size: 13px;
    padding: 5px 8px; border-radius: 6px; color: var(--input-text, #ddd);
    background: var(--comfy-input-bg, #2b2b2b); border: 1px solid var(--border-color, #4e4e4e); }
.yue2-s-keep { display: flex; gap: 8px; align-items: center; font-size: 12px; }
.yue2-s-keeplabel { display: flex; gap: 6px; align-items: center; cursor: pointer; }
.yue2-s-keeplabel input { accent-color: #3B7DD8; margin: 0; }
.yue2-s-keep .yue2-s-dim { flex: 1 1 auto; min-width: 0; }
.yue2-s-list { flex: 1 1 auto; min-height: 0; overflow: auto; display: flex;
    flex-direction: column; gap: 6px; padding-right: 4px; }
.yue2-s-family { display: flex; flex-direction: column; gap: 3px; }
.yue2-s-row { padding: 7px 9px; border-radius: 7px; cursor: pointer;
    border: 1px solid var(--border-color, #4e4e4e); background: var(--comfy-input-bg, #2b2b2b); }
.yue2-s-row:hover, .yue2-s-kid:hover { background: var(--comfy-menu-bg, #353535); }
.yue2-s-picked { border-color: #3B7DD8; background: rgba(59, 125, 216, 0.24);
    box-shadow: inset 4px 0 0 #3B7DD8; }
.yue2-s-picked:hover { background: rgba(59, 125, 216, 0.3); }
.yue2-s-barred { opacity: 0.6; }
.yue2-s-kids { display: flex; flex-direction: column; gap: 3px;
    margin: 0 0 2px 22px; border-left: 1px solid var(--border-color, #4e4e4e); padding-left: 8px; }
.yue2-s-kid { padding: 5px 8px; border-radius: 6px; cursor: pointer;
    border: 1px solid transparent; background: rgba(255, 255, 255, 0.03); }
.yue2-s-head { display: flex; gap: 8px; align-items: center; font-size: 12px; }
.yue2-s-when { flex: 0 0 auto; font-variant-numeric: tabular-nums;
    color: var(--descrip-text, #999); }
.yue2-s-did { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; color: var(--input-text, #ddd); }
.yue2-s-long { flex: 0 0 auto; font-variant-numeric: tabular-nums;
    color: var(--descrip-text, #999); }
.yue2-s-row .yue2-s-when { flex: 1 1 auto; }
.yue2-s-style { margin-top: 2px; font-size: 13px; font-weight: 600; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
.yue2-s-mark, .yue2-s-count { flex: 0 0 auto; font-size: 11px;
    color: var(--descrip-text, #999); border: 1px solid var(--border-color, #4e4e4e);
    border-radius: 5px; padding: 0 5px; }
.yue2-s-swap-mark { flex: 0 0 auto; font-size: 11px; color: #A9D3F0; border: 1px solid #3B7DD8;
    border-radius: 5px; padding: 0 5px; background: rgba(59, 125, 216, 0.12); cursor: help; }
.yue2-s-warn-mark { flex: 0 0 auto; font-size: 11px; color: #E0A33E; border: 1px solid #E0A33E;
    border-radius: 5px; padding: 0 5px; }
.yue2-s-said-note { flex: 0 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; font-size: 11px; color: #D8C98A; border: 1px solid #8a7a3a;
    border-radius: 5px; padding: 0 5px; background: rgba(216, 201, 138, 0.1); }
.yue2-s-notebox { flex: 1 1 140px; min-width: 90px; font: inherit; font-size: 11px;
    padding: 1px 5px; border-radius: 5px; color: var(--input-text, #ddd);
    background: var(--comfy-input-bg, #2b2b2b); border: 1px solid #3B7DD8; }
.yue2-s-acts { flex: 0 0 auto; display: flex; gap: 4px; opacity: 0; pointer-events: none; }
.yue2-s-row:hover .yue2-s-acts, .yue2-s-kid:hover .yue2-s-acts,
.yue2-s-picked .yue2-s-acts { opacity: 1; pointer-events: auto; }
.yue2-s-turn { flex: 0 0 auto; font: inherit; font-size: 11px; line-height: 1; padding: 1px 4px;
    border-radius: 4px; cursor: pointer; color: var(--input-text, #ddd);
    background: transparent; border: 1px solid var(--border-color, #4e4e4e); }
.yue2-s-facts { margin-top: 2px; }
.yue2-s-words { margin-top: 3px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; line-height: 1.35; max-height: 2.7em; }
.yue2-s-strip { display: block; width: 100%; height: 22px; margin-top: 4px; }
.yue2-s-peaksline { fill: rgba(150, 175, 205, 0.45); }
.yue2-s-bodyline { fill: rgba(190, 210, 235, 0.75); }
.yue2-s-flat { fill: rgba(150, 175, 205, 0.35); }
.yue2-s-span { fill: rgba(59, 125, 216, 0.38); }
.yue2-s-seam { fill: rgba(224, 163, 62, 0.75); }
.yue2-s-empty { padding: 14px 4px; line-height: 1.5; }
.yue2-s-foot { display: flex; gap: 8px; align-items: center; }
.yue2-s-said { flex: 1 1 auto; min-width: 0; }
.yue2-s-failed { color: #E06C4E; }
.yue2-songs button { font: inherit; font-size: 12px; padding: 4px 10px; border-radius: 6px;
    cursor: pointer; color: var(--input-text, #ddd); background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-songs button:hover { background: var(--comfy-menu-bg, #353535); }
.yue2-songs button.yue2-go { background: #3B7DD8; border-color: #3B7DD8; color: #fff; }
.yue2-songs button.yue2-s-icon { font-size: 11px; line-height: 1; padding: 2px 6px;
    border-radius: 4px; background: transparent; }
.yue2-songs button.yue2-s-bin:hover { background: #7A2F27; border-color: #E06C4E; color: #fff; }
.yue2-songs button[disabled] { opacity: 0.45; cursor: not-allowed; }
`;
