export const PARTS = ["Vocal", "Ins"];

export const QUALITIES = [
    "", "m", "dim", "aug", "7", "maj7", "m7", "dim7", "m7b5",
    "sus4", "sus2", "6", "m6", "7sus4", "m(maj7)",
];

export const CHORD_TONES = {
    "": [0, 4, 7], "m": [0, 3, 7], "dim": [0, 3, 6], "aug": [0, 4, 8],
    "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11], "m7": [0, 3, 7, 10], "dim7": [0, 3, 6, 9],
    "m7b5": [0, 3, 6, 10], "sus4": [0, 5, 7], "sus2": [0, 2, 7], "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9], "7sus4": [0, 5, 7, 10], "m(maj7)": [0, 3, 7, 11],
};

export const SNAPS = [
    { name: "Quarter notes", quarters: 1 },
    { name: "Eighth notes", quarters: 0.5 },
    { name: "Sixteenth notes", quarters: 0.25 },
    { name: "Thirty-second notes", quarters: 0.125 },
];

export const SECTION_NAMES = [
    "intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge", "interlude",
    "instrumental", "solo", "rap", "theme", "variation", "development", "loop",
    "intro and verse", "verse and pre-chorus", "pre-chorus and chorus",
    "pre-outro", "outro", "fade-out", "preshot", "irregular", "silence",
];

export const SECTION_LONGEST = 40;

export const FINEST = 32;

export const LOWEST = 21;
export const HIGHEST = 108;

const SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
const NATURAL = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const SHIFT = { "": 0, "#": 1, "##": 2, "b": -1, "bb": -2 };
const PITCH_NAME = "([A-G])(bb|##|b|#)?";
const ESCAPED = QUALITIES.map((q) => q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
const CHORD_RE = new RegExp("^" + PITCH_NAME + "(" + ESCAPED.join("|") + ")(?:/" + PITCH_NAME + ")?$");
const FULL_REST_RE = /^\s*Z([2-4])?\s*$/;

export function isChord(name) {
    return CHORD_RE.test(String(name ?? "").trim());
}

export function chordPitches(name) {
    const match = CHORD_RE.exec(String(name ?? "").trim());
    if (!match) return [];
    const root = 48 + ((NATURAL[match[1]] + SHIFT[match[2] || ""] + 12) % 12);
    const tones = CHORD_TONES[match[3]].map((step) => root + step);
    if (match[4]) tones.unshift(36 + ((NATURAL[match[4]] + SHIFT[match[5] || ""] + 12) % 12));
    return tones;
}

const GUESSED = { 3: "m", 4: "", 6: "dim", 7: "", 10: "7", 11: "maj7" };

export function chordGuess(one, other, flats = false) {
    const low = Math.min(one, other);
    const step = (((Math.max(one, other) - low) % 12) + 12) % 12;
    if (!(step in GUESSED)) return "";
    return (flats ? FLAT_NAMES : SHARP_NAMES)[((low % 12) + 12) % 12] + GUESSED[step];
}

export function noteName(pitch, flats = false) {
    return (flats ? FLAT_NAMES : SHARP_NAMES)[((pitch % 12) + 12) % 12] + (Math.floor(pitch / 12) - 1);
}

export function isBlack(pitch) {
    return [1, 3, 6, 8, 10].includes(((pitch % 12) + 12) % 12);
}

export function flatsIn(sheet, key) {
    return (sheet?.signatures?.[key] ?? 0) < 0;
}

export function snapChoices(perQuarter) {
    return SNAPS
        .map((snap) => ({ name: snap.name, ticks: snap.quarters * perQuarter }))
        .filter((snap) => Number.isInteger(snap.ticks) && snap.ticks >= 1);
}

export function snapTo(tick, step) {
    return Math.round(tick / step) * step;
}

export function snapDown(tick, step) {
    return Math.floor(tick / step) * step;
}

export function modifiersOf(event) {
    const alt = Boolean(event.altKey || event.getModifierState?.("AltGraph"));
    return { alt, ctrl: Boolean(event.ctrlKey || event.metaKey) && !alt, shift: Boolean(event.shiftKey) };
}

export const TEMPO_LOW = 40;
export const TEMPO_HIGH = 200;

export function tempoRange(bpm) {
    const own = Math.round(Number(bpm)) || TEMPO_LOW;
    return { low: Math.min(TEMPO_LOW, own), high: Math.max(TEMPO_HIGH, own) };
}

export function tempoOf(value, own) {
    const asked = Math.round(Number(value));
    if (!Number.isFinite(asked)) return null;
    const range = tempoRange(own);
    return Math.min(range.high, Math.max(range.low, asked));
}

export function secondsAt(sheet, tick) {
    return (tick * 60) / (sheet.bpm * sheet.per_quarter);
}

export function clock(seconds) {
    const whole = Math.max(0, Math.floor(seconds));
    return Math.floor(whole / 60) + ":" + String(whole % 60).padStart(2, "0");
}

export const AUTO_BASE_SECONDS = 12;
export const AUTO_SECONDS_PER_LINE = 12;
export const AUTO_MIN_SECONDS = 40;
export const AUTO_INSTRUMENTAL_SECONDS = 180;
export const MAX_SECONDS = 360;
export const AUTO_WORDS_PER_LINE = 8;

const LINE_BREAKS = /\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]/;
const DIRECTION = /\[[^\]]*\]/g;
const LETTER = /[\p{L}\p{N}]/u;

export function sungLines(lyrics) {
    let count = 0;
    for (const piece of String(lyrics ?? "").split(LINE_BREAKS)) {
        const words = piece.replace(DIRECTION, " ").split(/\s+/).filter(Boolean);
        if (!words.some((word) => LETTER.test(word))) continue;
        count += Math.max(1, Math.floor((words.length + Math.floor(AUTO_WORDS_PER_LINE / 2)) / AUTO_WORDS_PER_LINE));
    }
    return count;
}

export function autoSeconds(lyrics) {
    const lines = sungLines(lyrics);
    if (lines === 0) return AUTO_INSTRUMENTAL_SECONDS;
    return Math.min(MAX_SECONDS, Math.max(AUTO_MIN_SECONDS, AUTO_BASE_SECONDS + AUTO_SECONDS_PER_LINE * lines));
}

export function lengthLimit(maxSeconds, lyrics, reported) {
    if (maxSeconds === null || maxSeconds === undefined) return null;
    const requested = Number(maxSeconds) || 0;
    if (requested > 0) return { seconds: requested, auto: false };
    if (typeof lyrics === "string") return { seconds: autoSeconds(lyrics), auto: true };
    const known = Number(reported);
    return known > 0 ? { seconds: known, auto: true } : null;
}

export function limitTick(sheet, seconds) {
    return (seconds * sheet.bpm * sheet.per_quarter) / 60;
}

export function barsAfter(sheet, seconds) {
    const edge = limitTick(sheet, seconds) - 1e-6;
    const after = [];
    sheet.bars.forEach((bar, index) => {
        if (bar.start >= edge) after.push(index);
    });
    return after;
}

export function barAt(sheet, tick) {
    const bars = sheet.bars;
    let low = 0;
    let high = bars.length - 1;
    while (low < high) {
        const middle = (low + high + 1) >> 1;
        if (bars[middle].start <= tick) low = middle;
        else high = middle - 1;
    }
    return low;
}

export function sectionAt(sheet, bar) {
    return sheet.sections.find((s) => bar >= s.bar && bar < s.bar + s.bars) || null;
}

export function modelOf(sheet) {
    let next = 1;
    const notes = {};
    for (const part of PARTS) {
        notes[part] = (sheet.notes[part] || [])
            .map((n) => ({ id: next++, start: n.start, length: n.length, pitch: n.pitch }))
            .sort((a, b) => a.start - b.start);
    }
    const chords = (sheet.chords || [])
        .map((c) => ({ start: c.start, name: c.name }))
        .sort((a, b) => a.start - b.start);
    return { notes, chords, next, bpm: sheet.bpm, unit: sheet.unit, sections: sectionsOf(sheet) };
}

export function sectionsOf(sheet) {
    return (sheet.sections || [])
        .filter((section) => section.name)
        .map((section) => ({ bar: section.bar, name: section.name }))
        .sort((a, b) => a.bar - b.bar);
}

export function sectionName(name) {
    const clean = String(name ?? "").trim().replace(/\s+/g, " ");
    return clean.length && clean.length <= SECTION_LONGEST ? clean : "";
}

export function setSection(model, bar, name) {
    const clean = sectionName(name);
    if (!clean || !Number.isInteger(bar) || bar < 0) return null;
    const sections = model.sections.filter((section) => section.bar !== bar);
    sections.push({ bar, name: clean });
    sections.sort((a, b) => a.bar - b.bar);
    return { ...model, sections };
}

export function removeSection(model, bar) {
    if (!model.sections.some((section) => section.bar === bar)) return null;
    return { ...model, sections: model.sections.filter((section) => section.bar !== bar) };
}

export function moveSection(model, from, to) {
    const found = model.sections.find((section) => section.bar === from);
    if (!found || !Number.isInteger(to) || to < 0) return null;
    if (to === from) return model;
    if (model.sections.some((section) => section.bar === to)) return null;
    return setSection(removeSection(model, from), to, found.name);
}

export function scaledModel(model, factor, unit) {
    const notes = {};
    for (const part of PARTS) {
        notes[part] = model.notes[part]
            .map((note) => ({ ...note, start: note.start * factor, length: note.length * factor }));
    }
    const chords = model.chords.map((chord) => ({ ...chord, start: chord.start * factor }));
    return { ...model, notes, chords, unit };
}

export function sectionStarting(model, bar) {
    return model.sections.find((section) => section.bar === bar) || null;
}

export function sectionSpans(model, bars) {
    const spans = [];
    for (const [index, section] of model.sections.entries()) {
        const next = model.sections[index + 1];
        spans.push({ name: section.name, bar: section.bar,
                     bars: (next ? next.bar : bars) - section.bar });
    }
    return spans;
}

export function sheetOf(model) {
    const notes = {};
    for (const part of PARTS) {
        notes[part] = model.notes[part]
            .map((n) => ({ start: n.start, length: n.length, pitch: n.pitch }))
            .sort((a, b) => a.start - b.start || a.pitch - b.pitch);
    }
    const chords = model.chords
        .map((c) => ({ start: c.start, name: c.name }))
        .sort((a, b) => a.start - b.start);
    return { notes, chords, bpm: model.bpm, unit: model.unit,
             sections: model.sections.map((section) => ({ bar: section.bar, name: section.name })) };
}

function withPart(model, part, notes) {
    return { ...model, notes: { ...model.notes, [part]: notes.slice().sort((a, b) => a.start - b.start) } };
}

export function fits(notes, candidate, ignore, total) {
    if (!Number.isInteger(candidate.start) || !Number.isInteger(candidate.length)) return false;
    if (candidate.start < 0 || candidate.length < 1 || candidate.start + candidate.length > total) return false;
    if (candidate.pitch < 0 || candidate.pitch > 127) return false;
    const end = candidate.start + candidate.length;
    return !notes.some((n) => !ignore.has(n.id) && n.start < end && n.start + n.length > candidate.start);
}

export function addNote(model, part, start, length, pitch, total) {
    const note = { id: model.next, start, length, pitch };
    if (!fits(model.notes[part], note, new Set(), total)) return null;
    const changed = withPart(model, part, [...model.notes[part], note]);
    changed.next = model.next + 1;
    return { model: changed, id: note.id };
}

export function roomAt(model, part, tick, total) {
    if (!Number.isInteger(tick) || tick < 0 || tick >= total) return 0;
    let end = total;
    for (const n of model.notes[part]) {
        if (n.start <= tick && tick < n.start + n.length) return 0;
        if (n.start > tick) end = Math.min(end, n.start);
    }
    return end - tick;
}

export function moveNotes(model, part, ids, ticks, semitones, total) {
    const moving = new Set(ids);
    const moved = model.notes[part].map((n) => (moving.has(n.id)
        ? { ...n, start: n.start + ticks, pitch: n.pitch + semitones } : n));
    for (const note of moved) {
        if (!moving.has(note.id)) continue;
        if (!fits(moved, note, moving, total)) return null;
    }
    return withPart(model, part, moved);
}

export function stretchNote(model, part, id, length, total) {
    const notes = model.notes[part];
    const note = notes.find((n) => n.id === id);
    if (!note || !Number.isInteger(length)) return null;
    const next = notes.find((n) => n.start > note.start) || null;
    const farthest = next ? next.start + next.length - 1 : total;
    const end = Math.max(note.start + 1, Math.min(note.start + length, farthest, total));
    if (end === note.start + note.length) return model;
    return withPart(model, part, notes.map((n) => {
        if (n.id === id) return { ...n, length: end - n.start };
        if (next && n.id === next.id && end > n.start) return { ...n, start: end, length: n.start + n.length - end };
        return n;
    }));
}

export function deleteNotes(model, part, ids) {
    const gone = new Set(ids);
    return withPart(model, part, model.notes[part].filter((n) => !gone.has(n.id)));
}

export function noteAt(model, part, tick, pitch) {
    return model.notes[part].find((n) => n.pitch === pitch && n.start <= tick && tick < n.start + n.length) || null;
}

export function notesIn(model, part, fromTick, toTick, lowPitch, highPitch) {
    const [t0, t1] = [Math.min(fromTick, toTick), Math.max(fromTick, toTick)];
    const [p0, p1] = [Math.min(lowPitch, highPitch), Math.max(lowPitch, highPitch)];
    return model.notes[part]
        .filter((n) => n.start < t1 && n.start + n.length > t0 && n.pitch >= p0 && n.pitch <= p1)
        .map((n) => n.id);
}

export function setChord(model, start, name) {
    const clean = String(name ?? "").trim();
    if (!isChord(clean)) return null;
    const chords = model.chords.filter((c) => c.start !== start);
    chords.push({ start, name: clean });
    chords.sort((a, b) => a.start - b.start);
    return { ...model, chords };
}

export function removeChord(model, start) {
    return { ...model, chords: model.chords.filter((c) => c.start !== start) };
}

function barSignature(notes, chords, start, end) {
    const inside = notes
        .filter((n) => n.start < end && n.start + n.length > start)
        .map((n) => n.start + ":" + n.length + ":" + n.pitch)
        .sort();
    const named = chords.filter((c) => c.start >= start && c.start < end).map((c) => c.start + ":" + c.name);
    return inside.join(",") + "|" + named.join(",");
}

export function changedBars(sheet, model) {
    const original = modelOf(sheet);
    const changed = [];
    sheet.bars.forEach((bar, index) => {
        const end = bar.start + bar.length;
        for (const part of PARTS) {
            const before = barSignature(original.notes[part], part === "Vocal" ? original.chords : [], bar.start, end);
            const after = barSignature(model.notes[part], part === "Vocal" ? model.chords : [], bar.start, end);
            if (before !== after) {
                changed.push(index);
                return;
            }
        }
    });
    return changed;
}

export function pitchSpan(model) {
    let low = Infinity;
    let high = -Infinity;
    for (const part of PARTS) {
        for (const n of model.notes[part]) {
            low = Math.min(low, n.pitch);
            high = Math.max(high, n.pitch);
        }
    }
    if (low === Infinity) return { low: 55, high: 79 };
    return { low: Math.max(LOWEST, low - 5), high: Math.min(HIGHEST, high + 5) };
}

export function events(sheet, model, fromTick, parts = { Vocal: true, Ins: true, chords: true }) {
    const tick = 60 / (sheet.bpm * sheet.per_quarter);
    const out = [];
    for (const part of PARTS) {
        if (!parts[part]) continue;
        for (const n of model.notes[part]) {
            const end = n.start + n.length;
            if (end <= fromTick) continue;
            const begin = Math.max(n.start, fromTick);
            out.push({ at: (begin - fromTick) * tick, length: (end - begin) * tick, pitch: n.pitch, part });
        }
    }
    if (parts.chords) {
        model.chords.forEach((chord, index) => {
            const end = index + 1 < model.chords.length ? model.chords[index + 1].start : sheet.total;
            if (end <= fromTick) return;
            const begin = Math.max(chord.start, fromTick);
            for (const pitch of chordPitches(chord.name)) {
                out.push({ at: (begin - fromTick) * tick, length: (end - begin) * tick, pitch, part: "chords" });
            }
        });
    }
    return out.sort((a, b) => a.at - b.at || a.pitch - b.pitch);
}

function isMusicLine(line) {
    const body = line.trim();
    return body.endsWith("|") && !/^(%|[A-Za-z]:)/.test(body);
}

export function forNotation(abc) {
    return String(abc ?? "").split(/\r?\n/).map((line) => {
        if (!isMusicLine(line)) return line;
        const pieces = line.trim().slice(0, -1).split("|").flatMap((piece) => {
            const rest = FULL_REST_RE.exec(piece);
            return rest ? Array(Number(rest[1] || 1)).fill("Z") : [piece];
        });
        return pieces.join("|") + "|";
    }).join("\n");
}

export function headerFacts(abc) {
    const lines = String(abc ?? "").trim().split(/\r?\n/);
    const field = (name) => {
        const line = lines.slice(0, 8).find((l) => l.startsWith(name + ":"));
        return line ? line.slice(name.length + 1).trim() : "";
    };
    let bars = 0;
    let vocal = false;
    for (const line of lines.slice(8)) {
        if (line.startsWith("V:")) vocal = line.slice(2).trim() === "Vocal";
        else if (vocal && isMusicLine(line)) bars += forNotation(line).trim().slice(0, -1).split("|").length;
    }
    const tempo = /=(\d+)/.exec(field("Q"));
    return { key: field("K"), meter: field("M"), bpm: tempo ? Number(tempo[1]) : null, bars };
}

export const MARK_PREFIX = "%yue2-words ";
const MARK_LINE = /^%yue2-words ([0-9a-f]{16})[ \t\r]*$/;

export function splitMark(text) {
    let words = null;
    const kept = [];
    for (const line of String(text ?? "").split("\n")) {
        const found = MARK_LINE.exec(line);
        if (found) words = found[1];
        else kept.push(line);
    }
    return { score: kept.join("\n").trim(), words };
}

export function attachMark(score, words) {
    const clean = String(score ?? "").replace(/\s+$/, "");
    return words ? clean + "\n" + MARK_PREFIX + words : clean;
}

export function editValue(text, base, words) {
    const clean = String(text ?? "").trim();
    return clean && clean !== String(base ?? "").trim() ? attachMark(clean, words) : "";
}

export class History {
    constructor(limit = 200) {
        this.limit = limit;
        this.done = [];
        this.undone = [];
    }

    push(state) {
        this.done.push(state);
        if (this.done.length > this.limit) this.done.shift();
        this.undone = [];
    }

    undo(current) {
        if (!this.done.length) return null;
        this.undone.push(current);
        return this.done.pop();
    }

    redo(current) {
        if (!this.undone.length) return null;
        this.done.push(current);
        return this.undone.pop();
    }

    get canUndo() {
        return this.done.length > 0;
    }

    get canRedo() {
        return this.undone.length > 0;
    }
}

export const MIDI_EXTENSIONS = [".mid", ".midi", ".kar", ".rmi"];

export const PART_REASONS = {
    karaoke: "karaoke words", name: "by name", highest: "highest line", busiest: "busiest track",
};

export function isMidiFile(name) {
    const lower = String(name ?? "").toLowerCase();
    return MIDI_EXTENSIONS.some((ending) => lower.endsWith(ending));
}

export function midiFileName(title) {
    const clean = String(title ?? "").replace(/[^\p{L}\p{N} _().-]+/gu, "_").replace(/\s+/g, " ").trim()
        .replace(/^[._ ]+|[._ ]+$/g, "").slice(0, 80);
    return (clean || "YuE2 score") + ".mid";
}

export function partLine(part) {
    const pieces = [part.number + " " + (part.name || part.family)];
    if (part.name && part.family) pieces.push(part.family);
    pieces.push(part.notes + (part.notes === 1 ? " note" : " notes"));
    if (!part.drums) pieces.push(noteName(part.low) + "\u2013" + noteName(part.high));
    return pieces.join(" \u00b7 ");
}

export function partRole(part) {
    if (part.drums) return "drums, not sung";
    if (!part.role) return "";
    const why = PART_REASONS[part.why];
    return part.role + (why ? " (auto: " + why + ")" : "");
}

export function octaveMove(semitones) {
    const octaves = Math.abs(Math.round(Number(semitones) / 12));
    if (!octaves) return "";
    return (semitones > 0 ? "up " : "down ") + (octaves === 1 ? "an octave" : octaves + " octaves");
}

export function midiFacts(facts) {
    if (!facts) return "";
    const moved = octaveMove(facts.voice_shift);
    return [facts.bars + (facts.bars === 1 ? " bar" : " bars"), facts.bpm + " BPM", facts.meter,
        facts.key && "Key " + facts.key, clock(facts.seconds), moved && "voice " + moved,
        facts.karaoke && "karaoke words"].filter(Boolean).join(" \u00b7 ");
}
