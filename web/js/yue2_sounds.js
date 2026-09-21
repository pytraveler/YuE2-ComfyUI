export const DRUMS = "drums";

export const KIT = [
    "Kick", "Stick", "Snare", "Clap", "Tom low", "Tom mid",
    "Hat", "Tom high", "Hat open", "Crash", "Ride", "Shaker",
];

export const CHOICES = {
    Vocal: [["piano", "piano"], ["synth", "synth"], ["pluck", "pluck"]],
    Ins: [["piano", "piano"], ["bass", "bass"], ["pluck", "pluck"], [DRUMS, "drums"]],
    chords: [["piano", "piano"], ["pad", "pad"], ["pluck", "pluck"]],
};

export const DEFAULTS = { Vocal: "piano", Ins: "piano", chords: "piano" };

const FLOOR = 0.0005;
const NOISE = new WeakMap();

export function drumName(pitch) {
    return KIT[((Math.round(pitch) % 12) + 12) % 12];
}

export function known(part, kind) {
    const offered = CHOICES[part] || [];
    return offered.some(([id]) => id === kind) ? kind : (DEFAULTS[part] || "piano");
}

export function hertz(pitch) {
    return 440 * Math.pow(2, (pitch - 69) / 12);
}

function noiseBuffer(context) {
    let buffer = NOISE.get(context);
    if (!buffer) {
        buffer = context.createBuffer(1, Math.ceil(context.sampleRate), context.sampleRate);
        const data = buffer.getChannelData(0);
        for (let i = 0; i < data.length; i += 1) data[i] = Math.random() * 2 - 1;
        NOISE.set(context, buffer);
    }
    return buffer;
}

function hiss(context, master, start, seconds, level, type, frequency, q = 1) {
    const source = context.createBufferSource();
    source.buffer = noiseBuffer(context);
    source.loop = true;
    const filter = context.createBiquadFilter();
    filter.type = type;
    filter.frequency.value = frequency;
    filter.Q.value = q;
    const gain = context.createGain();
    gain.gain.setValueAtTime(Math.max(level, FLOOR), start);
    gain.gain.exponentialRampToValueAtTime(FLOOR, start + seconds);
    source.connect(filter).connect(gain).connect(master);
    source.start(start);
    source.stop(start + seconds + 0.02);
    return source;
}

function boom(context, master, start, seconds, level, from, to) {
    const osc = context.createOscillator();
    osc.type = "sine";
    osc.frequency.setValueAtTime(from, start);
    osc.frequency.exponentialRampToValueAtTime(Math.max(20, to), start + seconds);
    const gain = context.createGain();
    gain.gain.setValueAtTime(Math.max(level, FLOOR), start);
    gain.gain.exponentialRampToValueAtTime(FLOOR, start + seconds);
    osc.connect(gain).connect(master);
    osc.start(start);
    osc.stop(start + seconds + 0.02);
    return osc;
}

function sustained(context, master, start, length, level, type, frequency, attack, release, detune = 0) {
    const osc = context.createOscillator();
    osc.type = type;
    osc.frequency.value = frequency;
    if (detune) osc.detune.value = detune;
    const gain = context.createGain();
    const stop = start + Math.max(0.06, length);
    gain.gain.setValueAtTime(0, start);
    gain.gain.linearRampToValueAtTime(level, start + attack);
    gain.gain.setValueAtTime(level, Math.max(start + attack, stop - release));
    gain.gain.linearRampToValueAtTime(0, stop + release);
    osc.connect(gain).connect(master);
    osc.start(start);
    osc.stop(stop + release + 0.02);
    return osc;
}

function plucked(context, master, start, length, level, frequency) {
    const osc = context.createOscillator();
    osc.type = "sawtooth";
    osc.frequency.value = frequency;
    const filter = context.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(Math.min(9000, frequency * 9), start);
    filter.frequency.exponentialRampToValueAtTime(Math.max(180, frequency * 1.4), start + 0.3);
    const gain = context.createGain();
    const stop = start + Math.min(Math.max(0.14, length), 1.4);
    gain.gain.setValueAtTime(Math.max(level, FLOOR), start);
    gain.gain.exponentialRampToValueAtTime(FLOOR, stop);
    osc.connect(filter).connect(gain).connect(master);
    osc.start(start);
    osc.stop(stop + 0.02);
    return osc;
}

const PIECES = [
    (context, master, start) => [boom(context, master, start, 0.3, 0.30, 125, 45)],
    (context, master, start) => [hiss(context, master, start, 0.05, 0.10, "bandpass", 1900, 6)],
    (context, master, start) => [hiss(context, master, start, 0.18, 0.13, "highpass", 1500),
        boom(context, master, start, 0.12, 0.10, 220, 140)],
    (context, master, start) => [hiss(context, master, start, 0.09, 0.09, "bandpass", 1300, 1.5),
        hiss(context, master, start + 0.014, 0.09, 0.08, "bandpass", 1300, 1.5),
        hiss(context, master, start + 0.03, 0.12, 0.08, "bandpass", 1300, 1.5)],
    (context, master, start) => [boom(context, master, start, 0.32, 0.20, 165, 85)],
    (context, master, start) => [boom(context, master, start, 0.28, 0.19, 235, 120)],
    (context, master, start) => [hiss(context, master, start, 0.045, 0.07, "highpass", 6500)],
    (context, master, start) => [boom(context, master, start, 0.24, 0.18, 320, 170)],
    (context, master, start) => [hiss(context, master, start, 0.32, 0.06, "highpass", 6000)],
    (context, master, start) => [hiss(context, master, start, 1.0, 0.07, "highpass", 4200)],
    (context, master, start) => [hiss(context, master, start, 0.5, 0.05, "bandpass", 5400, 0.7)],
    (context, master, start) => [hiss(context, master, start, 0.06, 0.05, "highpass", 7500)],
];

export function play(context, master, start, length, pitch, kind, level) {
    if (kind === DRUMS) {
        return PIECES[((Math.round(pitch) % 12) + 12) % 12](context, master, start);
    }
    const frequency = hertz(pitch);
    if (kind === "bass") {
        return [sustained(context, master, start, length, level * 1.2, "sine", frequency, 0.014, 0.05)];
    }
    if (kind === "pluck") {
        return [plucked(context, master, start, length, level, frequency)];
    }
    if (kind === "pad") {
        return [sustained(context, master, start, length, level * 0.9, "triangle", frequency, 0.16, 0.2),
            sustained(context, master, start, length, level * 0.55, "sine", frequency, 0.2, 0.2, 8)];
    }
    return [sustained(context, master, start, length, level, "triangle", frequency, 0.014, 0.05)];
}
