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
export const MAJOR_ROOTS = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
export const MINOR_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"];
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

export function movedChord(name, semitones) {
    const match = CHORD_RE.exec(String(name ?? "").trim());
    if (!match || !Number.isInteger(semitones)) return null;
    if (semitones % 12 === 0) return match[0];
    const quality = match[3];
    const minor = (quality.startsWith("m") && !quality.startsWith("maj")) || quality.startsWith("dim");
    const spell = (letter, shift, names) => names[(((NATURAL[letter] + SHIFT[shift || ""] + semitones) % 12) + 12) % 12];
    const root = spell(match[1], match[2], minor ? MINOR_ROOTS : MAJOR_ROOTS);
    return root + quality + (match[4] ? "/" + spell(match[4], match[5], MAJOR_ROOTS) : "");
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

export const DOUBLE_CLICK_MS = 400;
export const DOUBLE_CLICK_PX = 5;

export function isDoubleClick(last, at, px, py) {
    if (!last) return false;
    return at - last.at <= DOUBLE_CLICK_MS
        && Math.abs(px - last.px) <= DOUBLE_CLICK_PX
        && Math.abs(py - last.py) <= DOUBLE_CLICK_PX;
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

export function chordsIn(model, fromTick, toTick) {
    const [t0, t1] = [Math.min(fromTick, toTick), Math.max(fromTick, toTick)];
    const starts = model.chords.map((c) => c.start).sort((a, b) => a - b);
    return starts.filter((start, index) => start < t1 && (index + 1 < starts.length ? starts[index + 1] : Infinity) > t0);
}

export function partOf(model, id) {
    return PARTS.find((part) => model.notes[part].some((n) => n.id === id)) || null;
}

export function moveTogether(model, ids, starts, ticks, semitones, total) {
    if (!ticks && !semitones) return model;
    const moving = new Set(ids);
    let changed = model;
    for (const part of PARTS) {
        const mine = model.notes[part].filter((n) => moving.has(n.id)).map((n) => n.id);
        if (!mine.length) continue;
        changed = moveNotes(changed, part, mine, ticks, semitones, total);
        if (!changed) return null;
    }
    const chosen = new Set(starts);
    const kept = model.chords.filter((c) => !chosen.has(c.start));
    if (kept.length === model.chords.length) return changed;
    const taken = new Set(kept.map((c) => c.start));
    const moved = [];
    for (const chord of model.chords) {
        if (!chosen.has(chord.start)) continue;
        const start = chord.start + ticks;
        const name = movedChord(chord.name, semitones);
        if (!name || !Number.isInteger(start) || start < 0 || start >= total || taken.has(start)) return null;
        moved.push({ start, name });
    }
    return { ...changed, chords: [...kept, ...moved].sort((a, b) => a.start - b.start) };
}

export function deleteTogether(model, ids, starts) {
    const gone = new Set(ids);
    const dropped = new Set(starts);
    const notes = {};
    for (const part of PARTS) notes[part] = model.notes[part].filter((n) => !gone.has(n.id));
    return { ...model, notes, chords: model.chords.filter((c) => !dropped.has(c.start)) };
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

const LETTERS = "CDEFGAB";
const LETTER_STEPS = [0, 2, 4, 5, 7, 9, 11];
const LETTER_FIFTHS = [0, 2, 4, -1, 1, 3, 5];
const FIFTH_TONES = {
    "": [0, 4, 1], "m": [0, -3, 1], "dim": [0, -3, -6], "aug": [0, 4, 8],
    "maj7": [0, 4, 1, 5], "m7": [0, -3, 1, -2], "7": [0, 4, 1, -2], "m7b5": [0, -3, -6, -2],
    "dim7": [0, -3, -6, -9], "m(maj7)": [0, -3, 1, 5], "sus2": [0, 2, 1], "sus4": [0, -1, 1],
    "7sus4": [0, -1, 1, -2], "6": [0, 4, 1, 3], "m6": [0, -3, 1, 3],
};
const DEGREES = ["1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7"];
const TONE_DEGREES = { "dim": { 6: "b5" }, "m7b5": { 6: "b5" }, "dim7": { 6: "b5", 9: "bb7" }, "aug": { 8: "#5" } };
const DEGREE_RE = /^(#{0,2}|b{0,2})([1-9]|1[0-3])$/;
const ACCIDENTALS = { "-2": "bb", "-1": "b", "0": "", "1": "#", "2": "##" };
export const CHORD_ROOT = 48;

function mod12(value) {
    return ((value % 12) + 12) % 12;
}

export function keySharps(key) {
    const found = KEY_NAME_RE.exec(String(key ?? "").trim());
    if (!found) return null;
    const tonic = NATURAL[found[1]] + SHIFT[found[2] || ""] + (found[3] ? 3 : 0);
    const sharps = mod12(7 * tonic);
    return sharps > 6 ? sharps - 12 : sharps;
}

export function chordParts(name) {
    const match = CHORD_RE.exec(String(name ?? "").trim());
    if (!match) return null;
    return {
        root: mod12(NATURAL[match[1]] + SHIFT[match[2] || ""]),
        quality: match[3],
        bass: match[4] ? mod12(NATURAL[match[4]] + SHIFT[match[5] || ""]) : null,
    };
}

function spelledRoot(root, quality, sharps) {
    const centre = sharps + 2;
    let best = null;
    for (let letter = 0; letter < 7; letter++) {
        const alteration = mod12(root - LETTER_STEPS[letter] + 6) - 6;
        const place = LETTER_FIFTHS[letter] + 7 * alteration;
        const places = FIFTH_TONES[quality].map((step) => place + step);
        const cost = [...places, places[0]].reduce((sum, at) => sum + Math.max(0, Math.abs(at - centre) - 3), 0);
        if (!best || cost < best.cost) best = { cost, letter, alteration };
    }
    return LETTERS[best.letter] + (best.alteration > 0 ? "#".repeat(best.alteration) : "b".repeat(-best.alteration));
}

function bassName(root, degreeText) {
    const found = DEGREE_RE.exec(degreeText);
    const rootLetter = LETTERS.indexOf(root[0]);
    const rootShift = [...root.slice(1)].reduce((sum, sign) => sum + (sign === "#" ? 1 : -1), 0);
    const degree = Number(found[2]);
    const accidental = found[1];
    const interval = LETTER_STEPS[(degree - 1) % 7] + 12 * Math.floor((degree - 1) / 7)
        + [...accidental].reduce((sum, sign) => sum + (sign === "#" ? 1 : -1), 0);
    const target = mod12(LETTER_STEPS[rootLetter] + rootShift + interval);
    const letter = (rootLetter + degree - 1) % 7;
    const difference = mod12(target - LETTER_STEPS[letter] + 6) - 6;
    if (Math.abs(difference) <= 2) return LETTERS[letter] + ACCIDENTALS[difference];
    return ((root + accidental).includes("#") ? SHARP_NAMES : FLAT_NAMES)[target];
}

export function spellChord(root, quality, bass, key) {
    const sharps = keySharps(key) ?? 0;
    const name = spelledRoot(mod12(root), quality, sharps);
    if (bass === null || bass === undefined || mod12(bass - root) === 0) return name + quality;
    const step = mod12(bass - root);
    return name + quality + "/" + bassName(name, TONE_DEGREES[quality]?.[step] ?? DEGREES[step]);
}

function shapeOf(classes, prefer) {
    const found = [];
    for (const quality of QUALITIES) {
        for (let root = 0; root < 12; root++) {
            const tones = new Set(CHORD_TONES[quality].map((step) => (root + step) % 12));
            if (tones.size === classes.size && [...tones].every((tone) => classes.has(tone))) found.push({ root, quality });
        }
    }
    for (const root of prefer) {
        const hit = found.find((shape) => shape.root === root);
        if (hit) return hit;
    }
    return found[0] || null;
}

export function namedChord(pitches, key) {
    const sorted = [...new Set(pitches.map(Number))].sort((a, b) => a - b);
    const classes = new Set(sorted.map(mod12));
    if (classes.size < 3) return null;
    const [bass, ...upper] = sorted;
    const above = new Set(upper.map(mod12));
    const tries = [];
    if (bass < CHORD_ROOT && CHORD_ROOT <= upper[0]) tries.push([above, [mod12(upper[0]), mod12(bass)]]);
    tries.push([classes, [mod12(bass), mod12(upper[0])]]);
    if (!above.has(mod12(bass))) tries.push([above, [mod12(upper[0])]]);
    for (const [wanted, prefer] of tries) {
        const found = shapeOf(wanted, prefer);
        if (found) return spellChord(found.root, found.quality, mod12(bass), key);
    }
    return null;
}

export function transposedChord(name, semitones, key) {
    const parts = chordParts(name);
    if (!parts || !Number.isInteger(semitones)) return null;
    if (mod12(semitones) === 0) return String(name).trim();
    return spellChord(parts.root + semitones, parts.quality, parts.bass === null ? null : parts.bass + semitones, key);
}

export function chordByDegree(pitch, key) {
    const tonic = mod12(7 * (keySharps(key) ?? 0));
    const scale = LETTER_STEPS.map((step) => mod12(tonic + step));
    const root = mod12(pitch);
    const at = scale.indexOf(root);
    if (at < 0) return spellChord(root, "", null, key);
    const third = mod12(scale[(at + 2) % 7] - root);
    const fifth = mod12(scale[(at + 4) % 7] - root);
    const quality = fifth === 6 ? "dim" : third === 3 ? "m" : "";
    return spellChord(root, quality, null, key);
}

export function rootPitch(name) {
    const parts = chordParts(name);
    return parts ? CHORD_ROOT + parts.root : null;
}

export function keyAt(sheet, tick) {
    return sheet.bars[barAt(sheet, Math.max(0, Math.floor(tick)))]?.key || "";
}

export function chordSpans(model, total) {
    const sorted = model.chords.slice().sort((a, b) => a.start - b.start);
    return sorted.map((chord, index) => ({
        start: chord.start, name: chord.name, end: index + 1 < sorted.length ? sorted[index + 1].start : total,
    }));
}

export function chordRoom(model, start, total) {
    const starts = model.chords.map((c) => c.start).sort((a, b) => a - b);
    const at = starts.indexOf(start);
    if (at < 0) return null;
    return { low: at > 0 ? starts[at - 1] + 1 : 0, high: at + 1 < starts.length ? starts[at + 1] - 1 : total - 1 };
}

export function moveChord(model, sheet, start, ticks, semitones) {
    const chord = model.chords.find((c) => c.start === start);
    const room = chordRoom(model, start, sheet.total);
    if (!chord || !room || !Number.isInteger(ticks) || !Number.isInteger(semitones)) return null;
    const to = start + ticks;
    if (to < room.low || to > room.high) return null;
    if (!ticks && mod12(semitones) === 0) return model;
    const name = transposedChord(chord.name, semitones, keyAt(sheet, to));
    if (!name) return null;
    const chords = model.chords.map((c) => (c.start === start ? { start: to, name } : c));
    return { ...model, chords: chords.sort((a, b) => a.start - b.start) };
}

export function moveChords(model, sheet, starts, ticks, semitones) {
    if (!ticks && mod12(semitones) === 0) return model;
    const chosen = new Set(starts);
    const kept = model.chords.filter((c) => !chosen.has(c.start));
    const taken = new Set(kept.map((c) => c.start));
    const moved = [];
    for (const chord of model.chords) {
        if (!chosen.has(chord.start)) continue;
        const start = chord.start + ticks;
        const name = transposedChord(chord.name, semitones, keyAt(sheet, start));
        if (!name || !Number.isInteger(start) || start < 0 || start >= sheet.total || taken.has(start)) return null;
        taken.add(start);
        moved.push({ start, name });
    }
    return { ...model, chords: [...kept, ...moved].sort((a, b) => a.start - b.start) };
}

export function sectionKind(name) {
    return String(name ?? "").trim().toLowerCase().replace(/\s+/g, " ").replace(/\s*\d+$/, "");
}

function carriedAt(chords, tick) {
    let name = null;
    for (const chord of chords) if (chord.start < tick) name = chord.name;
    return name;
}

function soundingAt(chords, tick) {
    let name = null;
    for (const chord of chords) if (chord.start <= tick) name = chord.name;
    return name;
}

function patternIn(chords, from, to) {
    const sorted = chords.slice().sort((a, b) => a.start - b.start);
    const inside = sorted.filter((c) => c.start >= from && c.start < to).map((c) => [c.start - from, c.name]);
    if (!inside.length || inside[0][0] !== 0) inside.unshift([0, carriedAt(sorted, from)]);
    return inside;
}

function samePattern(one, other) {
    return one.length === other.length && one.every(([at, name], index) => other[index][0] === at && other[index][1] === name);
}

function writePattern(chords, from, to, pattern, total) {
    const sorted = chords.slice().sort((a, b) => a.start - b.start);
    const had = sorted.some((c) => c.start === from);
    const heldOn = to < total && !sorted.some((c) => c.start === to) ? soundingAt(sorted, to) : undefined;
    const out = sorted.filter((c) => c.start < from || c.start >= to);
    for (const [at, name] of pattern) {
        if (at > 0) {
            out.push({ start: from + at, name });
            continue;
        }
        if (name === null) {
            if (carriedAt(out, from) !== null) return null;
            continue;
        }
        if (had || carriedAt(out, from) !== name) out.push({ start: from, name });
    }
    out.sort((a, b) => a.start - b.start);
    if (heldOn !== undefined && soundingAt(out, to) !== heldOn) {
        if (heldOn === null) return null;
        out.push({ start: to, name: heldOn });
        out.sort((a, b) => a.start - b.start);
    }
    return out;
}

export function mirrorChords(sheet, before, after) {
    const bars = sheet.bars;
    const spans = sectionSpans(after, bars.length);
    const range = (bar, count) => [bars[bar].start, bar + count < bars.length ? bars[bar + count].start : sheet.total];
    const edited = spans.filter((span) => {
        const [from, to] = range(span.bar, span.bars);
        return !samePattern(patternIn(before.chords, from, to), patternIn(after.chords, from, to));
    });
    const touched = new Set(edited.map((span) => span.bar));
    const done = [];
    const skipped = [];
    let chords = after.chords;
    for (const span of edited) {
        for (const other of spans) {
            if (touched.has(other.bar) || sectionKind(other.name) !== sectionKind(span.name)) continue;
            const count = Math.min(span.bars, other.bars);
            const [from, to] = range(span.bar, count);
            const was = patternIn(before.chords, from, to);
            const now = patternIn(after.chords, from, to);
            if (samePattern(was, now)) continue;
            const [start, end] = range(other.bar, count);
            const alike = bars.slice(other.bar, other.bar + count).every((bar, index) =>
                bar.editable !== false && bar.length === bars[span.bar + index].length);
            const written = alike && samePattern(patternIn(chords, start, end), was)
                ? writePattern(chords, start, end, now, sheet.total) : null;
            if (!written) {
                if (!skipped.some((one) => one.bar === other.bar)) skipped.push(other);
                continue;
            }
            chords = written;
            touched.add(other.bar);
            done.push(other);
        }
    }
    const kept = skipped.filter((span) => !touched.has(span.bar));
    const brief = (list) => list.map((span) => ({ bar: span.bar, bars: span.bars, name: span.name }))
        .sort((a, b) => a.bar - b.bar);
    return { model: done.length ? { ...after, chords } : after, done: brief(done), skipped: brief(kept) };
}

export function clipOf(model, sheet, ids, starts) {
    const chosen = new Set(ids);
    const total = sheet.total;
    const notes = {};
    let first = Infinity;
    let last = -Infinity;
    for (const part of PARTS) {
        notes[part] = model.notes[part].filter((n) => chosen.has(n.id));
        for (const n of notes[part]) {
            first = Math.min(first, n.start);
            last = Math.max(last, n.start + n.length);
        }
    }
    const picked = new Set(starts);
    const spans = chordSpans(model, total).filter((c) => picked.has(c.start));
    const bare = first === Infinity;
    for (const c of bare ? spans : []) {
        first = Math.min(first, c.start);
        last = Math.max(last, c.end);
    }
    if (first === Infinity) return null;
    const origin = sheet.bars[barAt(sheet, first)].start;
    const endBar = barAt(sheet, Math.max(first, last - 1));
    const after = endBar + 1 < sheet.bars.length ? sheet.bars[endBar + 1].start : total;
    const chords = spans
        .map((c) => ({ at: Math.max(c.start, bare ? c.start : origin) - origin, name: c.name,
                       end: Math.min(c.end, bare ? c.end : after) - origin }))
        .filter((c) => c.at < c.end);
    const clipped = {};
    for (const part of PARTS) {
        clipped[part] = notes[part].map((n) => ({ at: n.start - origin, length: n.length, pitch: n.pitch }));
    }
    return { origin, span: after - origin, notes: clipped, chords };
}

export function clipReach(clip, routes, withChords) {
    const pieces = routes.flatMap(([from]) => clip.notes[from].map((n) => [n.at, n.at + n.length]));
    if (withChords) pieces.push(...clip.chords.map((c) => [c.at, c.end]));
    if (!pieces.length) return null;
    return { from: Math.min(...pieces.map((p) => p[0])), to: Math.max(...pieces.map((p) => p[1])) };
}

export function pasteClip(model, sheet, clip, at, routes, withChords) {
    const total = sheet.total;
    let changed = model;
    let next = model.next;
    const ids = [];
    for (const [from, to] of routes) {
        const placed = clip.notes[from].map((n) => ({ start: at + n.at, length: n.length, pitch: n.pitch }));
        if (!placed.length) continue;
        const low = Math.min(...placed.map((n) => n.start));
        const high = Math.max(...placed.map((n) => n.start + n.length));
        if (low < 0 || high > total) return null;
        const kept = [];
        for (const n of changed.notes[to]) {
            if (n.start + n.length <= low || n.start >= high) kept.push(n);
            else if (n.start < low) kept.push({ ...n, length: low - n.start });
        }
        for (const n of placed) {
            kept.push({ id: next, ...n });
            ids.push(next);
            next += 1;
        }
        changed = { ...withPart(changed, to, kept), next };
    }
    let starts = [];
    if (withChords && clip.chords.length) {
        const from = at + clip.chords[0].at;
        const to = at + Math.max(...clip.chords.map((c) => c.end));
        if (from < 0 || to > total) return null;
        const pattern = clip.chords.map((c) => [c.at - clip.chords[0].at, c.name]);
        const chords = writePattern(changed.chords, from, to, pattern, total);
        if (!chords) return null;
        changed = { ...changed, chords };
        starts = clip.chords.map((c) => at + c.at);
    }
    if (!ids.length && !starts.length) return null;
    return { model: changed, ids, starts };
}

export function shortcutLetter(event) {
    const key = String(event.key || "").toLowerCase();
    if (/^[a-z]$/.test(key)) return key;
    const code = String(event.code || "");
    return /^Key[A-Z]$/.test(code) ? code.slice(3).toLowerCase() : key;
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

export const VOICE_WINDOW = [60, 82];

export function voiceMiddle(model) {
    const pitches = model.notes.Vocal.map((n) => n.pitch).sort((a, b) => a - b);
    if (!pitches.length) return null;
    return (pitches[Math.floor((pitches.length - 1) / 2)] + pitches[Math.floor(pitches.length / 2)]) / 2;
}

const KEY_NAME_RE = /^([A-G])(#|b)?(m?)$/;

export function keyChoices(key, middle = null) {
    const found = KEY_NAME_RE.exec(String(key ?? "").trim());
    if (!found) return [];
    const minor = found[3] === "m";
    const tonic = NATURAL[found[1]] + SHIFT[found[2] || ""];
    const roots = minor ? MINOR_ROOTS : MAJOR_ROOTS;
    const [low, high] = VOICE_WINDOW;
    const centre = (low + high) / 2;
    const known = typeof middle === "number" && Number.isFinite(middle);
    const inside = (shift) => !known || (middle + shift >= low && middle + shift <= high);
    const choices = [];
    for (let step = -5; step <= 6; step++) {
        let shift = step;
        if (step === 6 && known && Math.abs(middle - 6 - centre) < Math.abs(middle + 6 - centre)) shift = -6;
        const other = shift > 0 ? shift - 12 : shift + 12;
        if (shift !== 0 && !inside(shift) && inside(other)) shift = other;
        const name = step ? roots[(((tonic + step) % 12) + 12) % 12] + (minor ? "m" : "") : found[0];
        choices.push({ name, shift });
    }
    return choices;
}

export function shiftWords(shift) {
    const step = Math.trunc(Number(shift)) || 0;
    if (!step) return "nowhere";
    return (step > 0 ? "up " : "down ") + Math.abs(step) + (Math.abs(step) === 1 ? " semitone" : " semitones");
}

export function keyMove(shift) {
    if (!shift) return "";
    return (shift > 0 ? "+" : "\u2212") + Math.abs(shift);
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
const MARK_LINE = /^%yue2-words ([0-9a-f]{16})( keep)?[ \t\r]*$/;
const KEEP = " keep";

export function splitMark(text) {
    let words = null;
    let keep = false;
    const kept = [];
    for (const line of String(text ?? "").split("\n")) {
        const found = MARK_LINE.exec(line);
        if (found) {
            words = found[1];
            keep = Boolean(found[2]);
        } else {
            kept.push(line);
        }
    }
    return { score: kept.join("\n").trim(), words, keep };
}

export function attachMark(score, words, keep = false) {
    const clean = String(score ?? "").replace(/\s+$/, "");
    return words ? clean + "\n" + MARK_PREFIX + words + (keep ? KEEP : "") : clean;
}

export function editValue(text, base, words, keep = false) {
    const clean = String(text ?? "").trim();
    if (!clean) return "";
    if (clean !== String(base ?? "").trim()) return attachMark(clean, words, keep);
    return keep && words ? attachMark(clean, words, keep) : "";
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
        const dropped = this.undone;
        this.undone = [];
        return dropped;
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

    take(undone = null) {
        if (!this.done.length) return null;
        if (undone) this.undone = undone;
        return this.done.pop();
    }

    get lastDone() {
        return this.done.length ? this.done[this.done.length - 1] : null;
    }

    get lastUndone() {
        return this.undone.length ? this.undone[this.undone.length - 1] : null;
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
