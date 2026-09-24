export const LANGUAGES = [
    "English", "Russian", "Chinese", "Japanese", "Korean",
    "Spanish", "French", "German", "Italian", "Portuguese",
];

export const TAGS = [
    "Intro", "Verse", "Pre-Chorus", "Chorus", "Hook", "Bridge",
    "Refrain", "Outro", "Instrumental", "Interlude",
];

export const CYCLE = ["Verse", "Pre-Chorus", "Chorus", "Bridge", "Outro", "Intro"];

export const VOICES = [
    "expressive female voice", "soft female voice", "powerful female voice",
    "breathy female voice", "female melodic vocals", "warm expressive female lead vocal",
    "intimate female jazz vocalist", "1960s female vocal", "warm male voice",
    "deep male voice", "raspy male voice", "smooth male vocals", "gritty male lead vocal",
    "harsh screamed vocals", "male rap vocals", "playful duet",
    "child-and-adult unison vocals", "layered vocal harmonies", "backing choir",
];

export const SUGGESTIONS = [
    "acoustic piano", "electric piano", "honky-tonk piano", "toy piano",
    "jazz piano with rich extended chord voicings", "sparse piano comping",
    "acoustic guitar", "electric guitar", "clean electric guitar", "distorted guitars",
    "palm-muted riffs", "shimmering guitars", "fingerstyle jazz guitar", "steel guitar",
    "banjo and fiddle", "pizzicato strings", "string section", "lush orchestral strings",
    "brass section", "tenor saxophone", "honking sax section", "muted trumpet", "sousaphone",
    "synth bass", "funky bass", "rounded bass and light drums", "walking upright double bass",
    "upright bass and brushed drums", "punchy drums", "second-line drums", "double-kick drums",
    "congas and shakers", "sleigh bells", "synth pads", "driving rhythm section",
    "medium swing", "loose triplet feel", "driving rhythm", "unhurried phrasing",
    "staccato phrasing", "legato phrasing", "lyrical memorable melody", "catchy hook",
    "lo-fi texture", "wide reverb", "no guitars", "vintage 1950s mix",
    "warm analog production", "polished studio recording", "warm natural jazz-club recording",
    "4/4",
];

export const BPM_MIN = 40;
export const BPM_MAX = 200;
export const BPM_DEFAULT = 88;

const BPM_RE = /^(?:(\d{2,3})\s*bpm|bpm\s*(\d{2,3}))$/i;
const VOICE_RE =
    /\b(voice|voices|vocal|vocals|vocalist|vocalists|singer|singers|choir|duet|rapper|rappers)\b/i;
const TAG_RE = /^\s*\[\s*([A-Za-z][A-Za-z -]*?)\s*(\d+)?\s*\]\s*$/;

export function parseStyle(line) {
    const parts = [];
    const seen = new Set();
    for (const piece of String(line || "").split(/[,\n]/)) {
        const text = piece.trim();
        if (!text) continue;
        let kind = "other";
        if (!seen.has("language") && LANGUAGES.some((l) => l.toLowerCase() === text.toLowerCase())) {
            kind = "language";
        } else if (!seen.has("bpm") && BPM_RE.test(text)) {
            kind = "bpm";
        } else if (!seen.has("voice") && VOICE_RE.test(text)) {
            kind = "voice";
        }
        seen.add(kind);
        parts.push({ kind, text });
    }
    return parts;
}

export function oneLine(text) {
    return String(text || "")
        .split(/\r?\n/)
        .map((piece) => piece.trim().replace(/,$/, "").trim())
        .filter(Boolean)
        .join(", ");
}

export function formatStyle(parts) {
    return parts.map((p) => p.text.trim()).filter(Boolean).join(", ");
}

function find(parts, kind) {
    return parts.findIndex((p) => p.kind === kind);
}

export function languageOf(parts) {
    const at = find(parts, "language");
    if (at < 0) return "";
    return LANGUAGES.find((l) => l.toLowerCase() === parts[at].text.toLowerCase()) || "";
}

export function bpmOf(parts) {
    const at = find(parts, "bpm");
    if (at < 0) return null;
    const match = BPM_RE.exec(parts[at].text);
    return Number(match[1] || match[2]);
}

export function voiceOf(parts) {
    const at = find(parts, "voice");
    return at < 0 ? "" : parts[at].text;
}

function place(parts, kind, text) {
    const next = parts.map((p) => ({ ...p }));
    const at = find(next, kind);
    if (!text) {
        if (at >= 0) next.splice(at, 1);
        return next;
    }
    if (at >= 0) {
        next[at].text = text;
        return next;
    }
    const made = { kind, text };
    if (kind === "language") {
        next.unshift(made);
    } else if (kind === "bpm") {
        next.push(made);
    } else {
        const genre = find(next, "other");
        const language = find(next, "language");
        next.splice(genre >= 0 ? genre + 1 : language + 1, 0, made);
    }
    return next;
}

export function setLanguage(parts, name) {
    return place(parts, "language", name);
}

export function setBpm(parts, bpm) {
    if (bpm === null || bpm === undefined || bpm === "") return place(parts, "bpm", "");
    const clamped = Math.min(BPM_MAX, Math.max(BPM_MIN, Math.round(Number(bpm))));
    return place(parts, "bpm", clamped + " BPM");
}

export function setVoice(parts, text) {
    return place(parts, "voice", String(text || "").trim());
}

const VOICE_JOIN = /\s+and\s+|\s*&\s*/i;

export function splitVoices(text) {
    const clean = String(text || "").trim();
    if (!clean) return [];
    const pieces = clean.split(VOICE_JOIN).map((piece) => piece.trim()).filter(Boolean);
    return pieces.length > 1 && pieces.every((piece) => VOICE_RE.test(piece)) ? pieces : [clean];
}

export function voicesOf(parts) {
    return splitVoices(voiceOf(parts));
}

export function setVoices(parts, voices) {
    const clean = (voices || []).map(unsplit).filter(Boolean);
    return place(parts, "voice", clean.join(" and "));
}

function unsplit(text) {
    return String(text || "").replace(/\s*,\s*/g, " ").replace(/\s+/g, " ").trim();
}

export function addPart(parts, text) {
    const clean = unsplit(text);
    const next = parts.map((p) => ({ ...p }));
    if (!clean) return next;
    const bpm = find(next, "bpm");
    const at = bpm >= 0 && bpm === next.length - 1 ? bpm : next.length;
    next.splice(at, 0, { kind: "other", text: clean });
    return next;
}

export function editPart(parts, index, text) {
    const next = parts.map((p) => ({ ...p }));
    if (!next[index]) return next;
    const clean = unsplit(text);
    if (!clean) next.splice(index, 1);
    else next[index].text = clean;
    return next;
}

export function removePart(parts, index) {
    return parts.filter((_, at) => at !== index);
}

export function movePart(parts, index, step) {
    const next = parts.map((p) => ({ ...p }));
    let other = index + step;
    while (other >= 0 && other < next.length && next[other].kind !== "other") other += step;
    if (other < 0 || other >= next.length || !next[index]) return next;
    [next[index], next[other]] = [next[other], next[index]];
    return next;
}

export function parseLyrics(text) {
    const blocks = [];
    let current = null;
    for (const raw of String(text || "").replace(/\r\n?/g, "\n").split("\n")) {
        const match = TAG_RE.exec(raw);
        if (match) {
            current = { tag: tagName(match[1]), number: match[2] || "", lines: [] };
            blocks.push(current);
            continue;
        }
        if (!current) {
            current = { tag: null, number: "", lines: [] };
            blocks.push(current);
        }
        current.lines.push(raw.trim());
    }
    for (const block of blocks) {
        while (block.lines.length && !block.lines[block.lines.length - 1]) block.lines.pop();
        while (block.lines.length && !block.lines[0]) block.lines.shift();
    }
    return blocks.filter((b) => b.tag !== null || b.lines.length);
}

export function tagName(word) {
    const clean = String(word || "").trim().replace(/\s+/g, " ");
    const key = clean.toLowerCase().replace(/[ -]/g, "");
    const known = TAGS.find((t) => t.toLowerCase().replace(/-/g, "") === key);
    return known || clean;
}

export function header(block) {
    return "[" + block.tag + (block.number ? " " + block.number : "") + "]";
}

export function points(text) {
    return Array.from(String(text || ""));
}

export function layout(blocks) {
    const out = [];
    const lines = [];
    const headers = [];
    let position = 0;
    function put(text) {
        if (out.length) position += 1;
        out.push(text);
        const start = position;
        position += points(text).length;
        return start;
    }
    blocks.forEach((block, b) => {
        if (b > 0) put("");
        if (block.tag !== null) headers.push({ block: b, start: put(header(block)) });
        block.lines.forEach((line, l) => {
            lines.push({ block: b, line: l, start: put(line) });
        });
    });
    return { text: out.join("\n"), lines, headers };
}

export function formatLyrics(blocks) {
    return layout(blocks).text;
}

export function excerpt(blocks, picks) {
    const kept = [];
    blocks.forEach((block, b) => {
        const pick = picks[b];
        if (!pick) return;
        const lines = [];
        block.lines.forEach((line, l) => {
            const span = pick.lines[l];
            if (span) lines.push(points(line).slice(span[0], span[1]).join(""));
        });
        while (lines.length && !lines[lines.length - 1]) lines.pop();
        while (lines.length && !lines[0]) lines.shift();
        const tag = pick.header ? block.tag : null;
        if (tag !== null || lines.length) kept.push({ tag, number: block.number, lines });
    });
    return formatLyrics(kept);
}

function copy(blocks) {
    return blocks.map((b) => ({ ...b, lines: [...b.lines] }));
}

export function cycleTag(tag) {
    const at = CYCLE.indexOf(tag);
    return CYCLE[(at + 1) % CYCLE.length];
}

export function setTag(blocks, b, tag) {
    const next = copy(blocks);
    if (next[b]) next[b].tag = tag;
    return next;
}

export function insertLine(blocks, b, l, text = "") {
    const next = copy(blocks);
    if (next[b]) next[b].lines.splice(l, 0, text);
    return next;
}

export function setLine(blocks, b, l, text) {
    const next = copy(blocks);
    if (next[b] && l < next[b].lines.length) next[b].lines[l] = String(text).replace(/\n/g, " ").trim();
    return next;
}

export function duplicateLine(blocks, b, l) {
    return insertLine(blocks, b, l + 1, blocks[b]?.lines[l] ?? "");
}

export function removeLine(blocks, b, l) {
    const next = copy(blocks);
    if (next[b]) next[b].lines.splice(l, 1);
    return next;
}

export function moveLine(blocks, b, l, step) {
    const next = copy(blocks);
    const here = next[b];
    if (!here || l < 0 || l >= here.lines.length) return { blocks: next, b, l };
    const target = l + step;
    if (target >= 0 && target < here.lines.length) {
        [here.lines[l], here.lines[target]] = [here.lines[target], here.lines[l]];
        return { blocks: next, b, l: target };
    }
    const other = next[b + step];
    if (!other) return { blocks: next, b, l };
    const [line] = here.lines.splice(l, 1);
    if (step < 0) {
        other.lines.push(line);
        return { blocks: next, b: b + step, l: other.lines.length - 1 };
    }
    other.lines.unshift(line);
    return { blocks: next, b: b + step, l: 0 };
}

export function stanzaOf(block, l) {
    let first = l;
    let last = l;
    while (first > 0 && block.lines[first - 1]) first -= 1;
    while (last < block.lines.length - 1 && block.lines[last + 1]) last += 1;
    return [first, last];
}

export function duplicateStanza(blocks, b, l) {
    const next = copy(blocks);
    const block = next[b];
    if (!block || !block.lines[l]) return next;
    const [first, last] = stanzaOf(block, l);
    block.lines.splice(last + 1, 0, "", ...block.lines.slice(first, last + 1));
    return next;
}

export function removeStanza(blocks, b, l) {
    const next = copy(blocks);
    const block = next[b];
    if (!block || !block.lines[l]) return next;
    const [first, last] = stanzaOf(block, l);
    const blankAfter = block.lines[last + 1] === "" ? 1 : 0;
    const blankBefore = !blankAfter && block.lines[first - 1] === "" ? 1 : 0;
    block.lines.splice(first - blankBefore, last - first + 1 + blankAfter + blankBefore);
    return next;
}

export function addBlock(blocks, after, tag) {
    const next = copy(blocks);
    next.splice(after + 1, 0, { tag, number: "", lines: [""] });
    return next;
}

export function duplicateBlock(blocks, b) {
    const next = copy(blocks);
    if (next[b]) next.splice(b + 1, 0, { ...next[b], lines: [...next[b].lines] });
    return next;
}

export function removeBlock(blocks, b) {
    return copy(blocks).filter((_, at) => at !== b);
}

export function moveBlock(blocks, b, step) {
    const next = copy(blocks);
    const other = b + step;
    if (!next[b] || !next[other]) return next;
    [next[b], next[other]] = [next[other], next[b]];
    return next;
}

export function toggleCase(line, index) {
    const chars = points(line);
    const char = chars[index];
    if (char === undefined) return chars.join("");
    const upper = char.toUpperCase();
    const lower = char.toLowerCase();
    if (upper === lower) return chars.join("");
    const flipped = char === upper ? lower : upper;
    if (points(flipped).length !== 1) return chars.join("");
    chars[index] = flipped;
    return chars.join("");
}

function cased(char) {
    return !!char && char.toUpperCase() !== char.toLowerCase();
}

export function isInnerCapital(line, index) {
    const chars = points(line);
    const char = chars[index];
    if (!cased(char) || char === char.toLowerCase()) return false;
    return cased(chars[index - 1]);
}

export function summarize(style, lyrics) {
    const parts = parseStyle(style);
    const blocks = parseLyrics(lyrics);
    return {
        language: languageOf(parts),
        bpm: bpmOf(parts),
        voice: voiceOf(parts),
        others: parts.filter((p) => p.kind === "other").map((p) => p.text),
        sections: blocks.map((b) => ({
            name: b.tag === null ? "" : header(b).slice(1, -1),
            tag: b.tag,
            lines: b.lines.filter(Boolean).length,
        })),
        sung: blocks.reduce((sum, b) => sum + b.lines.filter(Boolean).length, 0),
    };
}
