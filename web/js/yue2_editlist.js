import { clock } from "./yue2_roll.js";

export const MAX_TAKES = 4;

export const SEED_CEILING = 2147483648;

export const OPS = ["retake", "cut", "words", "notes", "extend"];

export const LONGEST_SCORE = 200000;

export const LONGEST_WORDS = 4000;

export const NOTES_APART = 2;

export const NOT_A_LIST =
    "The edits field does not hold a JSON list. 'Reset track' on the node clears it.";

export const SECONDS_CUT =
    "A cut of a song that has a score goes by whole bars, so that its score and its words can be "
    + "cut with it. Let go of Alt and select bars.";

export const PAST_THE_END = "The song stops before that, so there is nothing there to edit.";

const PY_SPACE = /^[\t\n\v\f\r \x1c-\x1f\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/u;

function whole(value) {
    return typeof value === "number" && Number.isInteger(value);
}

function seedly(value) {
    return whole(value) && Number.isSafeInteger(value) && value >= 0;
}

const VARY = [0, 5];
const GUIDE = [1, 10];
const FADE = [0, 6];

function measure(item, key, where, what, span) {
    const value = given(item, key);
    if (value === null) return null;
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(where + ": " + what + " is not a number.");
    }
    if (value < span[0] || value > span[1]) {
        throw new Error(where + ": " + what + " is " + value + ", and it is between " + span[0]
            + " and " + span[1] + ".");
    }
    return value;
}

function pair(value) {
    return Array.isArray(value) && value.length === 2;
}

function given(item, name) {
    const value = item[name];
    return value === undefined ? null : value;
}

function asked(item, name, fallback) {
    return Object.prototype.hasOwnProperty.call(item, name) ? item[name] : fallback;
}

function rangeOf(item, where, op) {
    const bars = given(item, "bars");
    const seconds = given(item, "seconds");
    if (op === "extend") {
        if (bars !== null || seconds !== null) {
            throw new Error(where + " makes the song go on, which it does from where the singing "
                + "ends, so it selects no bars and no seconds.");
        }
        return { bars: null, seconds: null };
    }
    if ((op === "words" || op === "notes") && bars === null && seconds === null) {
        return { bars: null, seconds: null };
    }
    if (op === "notes" && seconds !== null) {
        throw new Error(where + " changes notes, which go by bars, and it selects seconds. Select "
            + "bars, or none: the bars whose notes changed are the ones sung again.");
    }
    if ((bars === null) === (seconds === null)) {
        throw new Error(where + " selects neither bars nor seconds, or both at once. It takes one "
            + "of them.");
    }
    const chosen = seconds === null ? bars : seconds;
    if (!pair(chosen)) {
        throw new Error(where + " selects " + (seconds === null ? "bars" : "seconds")
            + ", which is not a pair.");
    }
    if (seconds === null) {
        if (!whole(chosen[0]) || !whole(chosen[1])) {
            throw new Error(where + ": a bar is not a whole number.");
        }
        if (!(chosen[0] >= 0 && chosen[0] < chosen[1])) {
            throw new Error(where + " selects bars " + (chosen[0] + 1) + " to " + chosen[1]
                + ", which is not a stretch of a score.");
        }
        return { bars: [chosen[0], chosen[1]], seconds: null };
    }
    for (const value of chosen) {
        if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new Error(where + ": a second is not a number.");
        }
    }
    if (!(chosen[0] >= 0 && chosen[0] < chosen[1])) {
        throw new Error(where + " selects " + chosen[0].toFixed(2) + " to " + chosen[1].toFixed(2)
            + " seconds, which is not a stretch of a song.");
    }
    return { bars: null, seconds: [chosen[0], chosen[1]] };
}

function editOf(item, index, takes) {
    const where = "Edit " + (index + 1);
    if (!item || typeof item !== "object" || Array.isArray(item)) {
        throw new Error(where + " is not an object with an 'op' in it.");
    }
    const op = given(item, "op");
    if (!OPS.includes(op)) {
        throw new Error(where + " asks for '" + op + "', which is not something an edit does. It "
            + "is 'retake', 'cut', 'words', 'notes' or 'extend'.");
    }
    const span = rangeOf(item, where, op);
    if (op === "cut") {
        return { op, bars: span.bars, seconds: span.seconds, seed: 0, takes: 1, take: null,
                 vary: null, guide: null, lines: null, text: null, score: null,
                 fade: measure(item, "fade", where, "the fade", FADE) };
    }
    const seed = asked(item, "seed", 0);
    if (!whole(seed)) throw new Error(where + ": the seed is not a whole number.");
    if (!seedly(seed)) {
        throw new Error(where + ": the seed is not between 0 and 2**53 - 1, the numbers the "
            + "track window can write back without changing them.");
    }
    const wanted = asked(item, "takes", takes);
    if (!whole(wanted) || wanted < 1 || wanted > MAX_TAKES) {
        throw new Error(where + " asks for " + wanted + " takes; between 1 and " + MAX_TAKES
            + " can be sung.");
    }
    const take = given(item, "take");
    if (take !== null) {
        if (!whole(take)) throw new Error(where + ": the take kept is not a whole number.");
        if (take < 0 || take >= wanted) {
            throw new Error(where + " keeps take " + (take + 1) + " of " + wanted + ".");
        }
    }
    let lines = null;
    let said = null;
    if (op === "words") {
        const chosen = given(item, "lines");
        if (!pair(chosen)) {
            throw new Error(where + " does not say which lines of the words it rewrites.");
        }
        if (!whole(chosen[0])) throw new Error(where + ": the first line is not a whole number.");
        if (!whole(chosen[1])) throw new Error(where + ": the last line is not a whole number.");
        if (!(chosen[0] >= 0 && chosen[0] < chosen[1])) {
            throw new Error(where + " rewrites lines " + (chosen[0] + 1) + " to " + chosen[1]
                + ", which is not a stretch of words.");
        }
        const words = given(item, "text");
        if (typeof words !== "string") {
            throw new Error(where + " holds no new words to sing.");
        }
        lines = [chosen[0], chosen[1]];
        said = words;
    }
    let score = null;
    if (op === "notes") {
        score = given(item, "score");
        if (typeof score !== "string" || PY_SPACE.test(score)) {
            throw new Error(where + " holds no score to sing.");
        }
        if (score.length > LONGEST_SCORE && [...score].length > LONGEST_SCORE) {
            throw new Error(where + " holds a score far longer than any song.");
        }
    }
    if (op === "extend") {
        const words = given(item, "text");
        said = words === null ? "" : words;
        if (typeof said !== "string") {
            throw new Error(where + " holds new words that are not text.");
        }
        if (said.length > LONGEST_WORDS && [...said].length > LONGEST_WORDS) {
            throw new Error(where + " goes on with more than " + LONGEST_WORDS
                + " characters of words.");
        }
    }
    return { op, bars: span.bars, seconds: span.seconds, seed, takes: wanted, take,
             vary: measure(item, "vary", where, "the variety", VARY),
             guide: measure(item, "guide", where, "the guide", GUIDE), fade: null,
             lines, text: said, score };
}

export function readEdits(text, takes = 1) {
    if (text === null || text === undefined) return { edits: [], error: "" };
    if (typeof text !== "string") return { edits: [], error: NOT_A_LIST };
    const body = text.trim();
    if (!body) return { edits: [], error: "" };
    let found = null;
    try {
        found = JSON.parse(body);
    } catch (error) {
        return { edits: [], error: NOT_A_LIST };
    }
    if (!Array.isArray(found)) return { edits: [], error: NOT_A_LIST };
    const edits = [];
    for (let index = 0; index < found.length; index += 1) {
        try {
            edits.push(editOf(found[index], index, takes));
        } catch (error) {
            return { edits: [], error: error.message };
        }
    }
    return { edits, error: "" };
}

export function writeEdits(edits) {
    return JSON.stringify((edits || []).map((edit) => {
        const item = { op: edit.op };
        if (edit.bars) item.bars = [edit.bars[0], edit.bars[1]];
        else if (edit.seconds) item.seconds = [edit.seconds[0], edit.seconds[1]];
        if (edit.op === "words") {
            item.lines = [edit.lines[0], edit.lines[1]];
            item.text = edit.text;
        }
        if (edit.op === "notes") item.score = edit.score;
        if (edit.op === "extend") item.text = edit.text;
        if (edit.op === "retake" || edit.op === "words" || edit.op === "notes"
            || edit.op === "extend") {
            item.seed = edit.seed;
            item.takes = edit.takes;
            if (edit.take !== null && edit.take !== undefined) item.take = edit.take;
            if (typeof edit.vary === "number") item.vary = edit.vary;
            if (typeof edit.guide === "number") item.guide = edit.guide;
        } else if (typeof edit.fade === "number") {
            item.fade = edit.fade;
        }
        return item;
    }));
}

export function newSeed() {
    return Math.floor(Math.random() * SEED_CEILING);
}

export function barCount(grid) {
    const lines = grid && Array.isArray(grid.bars) ? grid.bars : [];
    return Math.max(0, lines.length - 1);
}

export function lineAt(grid, bar) {
    const lines = grid && Array.isArray(grid.bars) ? grid.bars : [];
    if (!lines.length) return 0;
    const at = Math.max(0, Math.min(lines.length - 1, Math.round(bar)));
    return lines[at];
}

export function nearestIn(values, second) {
    let best = -1;
    let gap = Infinity;
    for (let index = 0; index < values.length; index += 1) {
        const away = Math.abs(values[index] - second);
        if (away < gap) {
            gap = away;
            best = index;
        }
    }
    return best;
}

export function barAt(grid, second) {
    const lines = grid && Array.isArray(grid.bars) ? grid.bars : [];
    let found = 0;
    for (let index = 0; index < lines.length - 1; index += 1) {
        if (lines[index] <= second) found = index;
    }
    return found;
}

export function snapped(grid, second, toBeats) {
    if (!grid) return { second, bar: null };
    const lines = toBeats ? grid.beats || [] : grid.bars || [];
    const at = nearestIn(lines, second);
    if (at < 0) return { second, bar: null };
    return { second: lines[at], bar: toBeats ? null : at };
}

export function selectSeconds(from, to) {
    return { from: Math.min(from, to), to: Math.max(from, to), first: null, stop: null };
}

export function selectBars(grid, first, stop) {
    const count = barCount(grid);
    if (!count) return null;
    const low = Math.max(0, Math.min(count - 1, Math.min(first, stop)));
    const high = Math.max(low + 1, Math.min(count, Math.max(first, stop)));
    return { from: lineAt(grid, low), to: lineAt(grid, high), first: low, stop: high };
}

export function selectionBetween(grid, from, to, toBeats) {
    if (!grid || toBeats) {
        const low = snapped(grid, Math.min(from, to), true);
        const high = snapped(grid, Math.max(from, to), true);
        if (low.second === high.second) return null;
        return selectSeconds(low.second, high.second);
    }
    const low = snapped(grid, Math.min(from, to), false);
    const high = snapped(grid, Math.max(from, to), false);
    return selectBars(grid, low.bar, high.bar === low.bar ? low.bar + 1 : high.bar);
}

export function sectionsBetween(grid, from, to) {
    const sections = grid && Array.isArray(grid.sections) ? grid.sections : [];
    return sections.filter((section) => section.end > from + 0.001 && section.start < to - 0.001);
}

export function sectionAt(grid, second) {
    const sections = grid && Array.isArray(grid.sections) ? grid.sections : [];
    let found = null;
    for (const section of sections) {
        if (section.start <= second) found = section;
    }
    return found || sections[0] || null;
}

export function spanText(from, to) {
    return clock(from) + "-" + clock(to);
}

export function barsText(first, stop) {
    return stop - first === 1 ? "bar " + (first + 1) : "bars " + (first + 1) + "-" + stop;
}

export function describeSelection(selection) {
    if (!selection) return "";
    const when = spanText(selection.from, selection.to);
    if (selection.first === null) return when;
    return barsText(selection.first, selection.stop) + ", " + when;
}

export function linesText(first, stop) {
    return stop - first === 1 ? "line " + (first + 1) : "lines " + (first + 1) + "-" + stop;
}

export function sungLines(text) {
    return String(text === null || text === undefined ? "" : text).split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => /\p{L}|\p{N}/u.test(line) && !/^\[[^\]]*\]$/.test(line));
}

function goingOn(edit) {
    const sung = sungLines(edit.text);
    if (!sung.length) return "A new ending";
    const more = sung.length - 1;
    return "Go on with \"" + sung[0] + "\""
        + (more ? " and " + more + (more === 1 ? " more line" : " more lines") : "");
}

export function describeEdit(edit, index) {
    const named = { cut: "Cut", words: "New words", retake: "Retake", notes: "New notes" };
    const what = edit.op === "extend" ? goingOn(edit) : named[edit.op] || "Retake";
    let where = "";
    if (edit.bars) where = barsText(edit.bars[0], edit.bars[1]);
    else if (edit.seconds) where = spanText(edit.seconds[0], edit.seconds[1]);
    else if (edit.lines) where = linesText(edit.lines[0], edit.lines[1]);
    else if (edit.op !== "extend") where = "the bars they change";
    const joiner = edit.op === "extend" ? "" : edit.op === "words" ? " for "
        : edit.op === "notes" ? " in " : " of ";
    const kept = edit.op !== "cut" && edit.take !== null && edit.take !== undefined
        ? ", take " + (edit.take + 1) + " of " + edit.takes : "";
    const extra = [];
    if (typeof edit.vary === "number") extra.push("variety " + edit.vary.toFixed(2));
    if (typeof edit.guide === "number") extra.push("guide " + edit.guide.toFixed(1));
    if (typeof edit.fade === "number") extra.push("fade " + edit.fade.toFixed(1) + " s");
    return (index + 1) + ". " + what + joiner + where + kept
        + (extra.length ? ", " + extra.join(", ") : "");
}

export function wordsFor(lines, said, takes, seed, selection = null, knobs = {}) {
    const edit = { op: "words", bars: null, seconds: null, seed,
                   takes: Math.max(1, Math.min(MAX_TAKES, takes)), take: null,
                   vary: null, guide: null, fade: null,
                   lines: [lines[0], lines[1]], text: String(said), score: null };
    if (selection && selection.first !== null && selection.first !== undefined) {
        edit.bars = [selection.first, selection.stop];
    } else if (selection) {
        edit.seconds = [Number(selection.from.toFixed(3)), Number(selection.to.toFixed(3))];
    }
    if (typeof knobs.vary === "number") edit.vary = knobs.vary;
    if (typeof knobs.guide === "number") edit.guide = knobs.guide;
    return edit;
}

export function extendFor(text, takes, seed, knobs = {}) {
    const edit = { op: "extend", bars: null, seconds: null, seed,
                   takes: Math.max(1, Math.min(MAX_TAKES, takes)), take: null,
                   vary: null, guide: null, fade: null, lines: null,
                   text: String(text === null || text === undefined ? "" : text).trim(),
                   score: null };
    if (typeof knobs.vary === "number") edit.vary = knobs.vary;
    if (typeof knobs.guide === "number") edit.guide = knobs.guide;
    return edit;
}

export function lastSung(lyrics) {
    const blocks = [];
    let current = null;
    for (const raw of String(lyrics || "").replace(/\r\n?/g, "\n").split("\n")) {
        const line = raw.trim();
        if (/^\[[^\]]*\]$/.test(line)) {
            current = { tag: line, lines: [] };
            blocks.push(current);
        } else if (line) {
            if (!current) {
                current = { tag: "", lines: [] };
                blocks.push(current);
            }
            current.lines.push(line);
        }
    }
    const sung = blocks.filter((block) => block.lines.some((line) => /\p{L}|\p{N}/u.test(line)));
    const last = sung[sung.length - 1];
    if (!last) return "";
    return (last.tag ? [last.tag] : []).concat(last.lines).join("\n");
}

export function stretchesOf(changed, apart = NOTES_APART) {
    const bars = [...new Set((changed || []).filter((bar) => whole(bar) && bar >= 0))]
        .sort((a, b) => a - b);
    const found = [];
    for (const bar of bars) {
        const last = found[found.length - 1];
        if (last && bar - last[1] < apart) last[1] = bar + 1;
        else found.push([bar, bar + 1]);
    }
    return found;
}

export function notesFor(score, changed, takes, seeds, selection = null, knobs = {}) {
    const stretches = stretchesOf(changed);
    const inside = selection && selection.first !== null && selection.first !== undefined
        && stretches.length
        && stretches.every(([first, stop]) => first >= selection.first && stop <= selection.stop);
    const wanted = inside ? [[selection.first, selection.stop]]
        : stretches.length ? stretches : [null];
    return wanted.map((bars, index) => {
        const edit = { op: "notes", bars: bars ? [bars[0], bars[1]] : null, seconds: null,
                       seed: seeds[index],
                       takes: Math.max(1, Math.min(MAX_TAKES, takes)), take: null,
                       vary: null, guide: null, fade: null, lines: null, text: null,
                       score: String(score) };
        if (typeof knobs.vary === "number") edit.vary = knobs.vary;
        if (typeof knobs.guide === "number") edit.guide = knobs.guide;
        return edit;
    });
}

export function editFor(selection, op, takes, seed, knobs = {}) {
    const edit = { op, bars: null, seconds: null, seed: 0, takes: 1, take: null,
                   vary: null, guide: null, fade: null, lines: null, text: null, score: null };
    if (selection.first === null) {
        edit.seconds = [Number(selection.from.toFixed(3)), Number(selection.to.toFixed(3))];
    } else {
        edit.bars = [selection.first, selection.stop];
    }
    if (op === "retake") {
        edit.seed = seed;
        edit.takes = Math.max(1, Math.min(MAX_TAKES, takes));
        if (typeof knobs.vary === "number") edit.vary = knobs.vary;
        if (typeof knobs.guide === "number") edit.guide = knobs.guide;
    } else if (typeof knobs.fade === "number") {
        edit.fade = knobs.fade;
    }
    return edit;
}

export function cutEdge(selection, grid, seconds) {
    const bars = barCount(grid);
    if (!selection) return "";
    const head = selection.first === null ? selection.from <= 0.02 : selection.first === 0;
    const tail = selection.first === null ? selection.to >= seconds - 0.02
        : bars > 0 && selection.stop >= bars;
    if (head === tail) return "";
    return head ? "head" : "tail";
}

export function whyNotCut(selection, hasScore) {
    if (!selection) return "Select a stretch of the song first.";
    if (hasScore && selection.first === null) return SECONDS_CUT;
    return "";
}

export function whyNotEdit(selection, seconds) {
    if (!selection) return "Select a stretch of the song first.";
    if (selection.from >= seconds - 0.001) return PAST_THE_END;
    return "";
}

export function withTake(edits, index, take) {
    return edits.map((edit, at) => (at === index ? { ...edit, take } : edit));
}

export function withTakes(edits, index, takes) {
    return edits.map((edit, at) => (at === index
        ? { ...edit, takes, take: edit.take !== null && edit.take < takes ? edit.take : null }
        : edit));
}

export function dropLast(edits) {
    return edits.slice(0, Math.max(0, edits.length - 1));
}

export function sameEdits(one, other) {
    const bare = (edits) => (edits || []).map((edit) => ({ ...edit, take: null }));
    return writeEdits(bare(one)) === writeEdits(bare(other));
}

export const SECTION_TAGS = ["intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge",
    "outro", "rap", "interlude", "instrumental", "solo", "fade-out", "pre-outro", "loop",
    "intro and verse", "pre-chorus and chorus", "verse and pre-chorus", "theme", "development",
    "variation", "irregular", "preshot", "silence"];

export const TAG_ALIASES = {
    "prechorus": "pre-chorus", "pre chorus": "pre-chorus", "post chorus": "post-chorus",
    "hook": "chorus", "refrain": "chorus",
};

export const QUIET_SECTIONS = ["intro", "interlude", "instrumental", "solo", "fade-out",
    "silence", "preshot"];

export function labelOf(tag) {
    let clean = String(tag === null || tag === undefined ? "" : tag)
        .toLowerCase().replace(/_/g, " ").replace(/^[\s\x1c-\x1f]+|[\s\x1c-\x1f]+$/gu, "")
        .replace(/[\s\x1c-\x1f]+/gu, " ");
    clean = clean.replace(/[\s\x1c-\x1f\p{Nd}:.#-]+$/u, "");
    clean = TAG_ALIASES[clean] || clean;
    return SECTION_TAGS.includes(clean) ? clean : "verse";
}

export function sectionName(section) {
    return String(section && section.name !== null && section.name !== undefined
        ? section.name : "").toLowerCase().trim().replace(/\s+/g, " ");
}

export function pairSections(names, labels, notes) {
    if (!labels.length) return [];
    const found = [];
    let at = 0;
    for (const label of labels) {
        while (at < names.length && names[at] !== label) at += 1;
        if (at >= names.length) break;
        found.push(at);
        at += 1;
    }
    if (found.length === labels.length) return found;
    if (Array.isArray(notes) && notes.length === names.length) {
        const sung = [];
        for (let index = 0; index < names.length; index += 1) {
            if (Number(notes[index]) > 0) sung.push(index);
        }
        if (sung.length === labels.length) return sung;
    }
    for (const quiet of [QUIET_SECTIONS, QUIET_SECTIONS.concat(["outro"]), []]) {
        const left = [];
        for (let index = 0; index < names.length; index += 1) {
            if (!quiet.includes(names[index])) left.push(index);
        }
        if (left.length === labels.length) return left;
    }
    return found.length ? found : null;
}
