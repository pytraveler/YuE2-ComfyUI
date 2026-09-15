"""SheetSage2's decoder vocabulary: what each of its 31678 token ids means.

After four specials the ids sit in fixed blocks: 256 prompt slots (eight are
used), 257 subbeat shifts, 30000 time stamps at 100 per second across a
300-second window, 192 meters, 256 eighth-note positions, 23 section labels,
24 keys, 25 major/minor chords, 361 full chords, 256 pitches (two melody tracks
of 128 each) and 24 note lengths. The layout is a fact of the released weights
-- the embedding has exactly this many rows -- so the total is checked when the
module loads rather than trusted.

A sequence is a prompt prefix -- start, the chosen tasks, out -- followed by
events. Each event is one or more subbeat shifts and then at most one value
per field, in the fixed order timestamp, rhythm, structure, key, chord,
melody. A subbeat is an eighth of a beat's quarter: the positions the model
counts in.
"""

from __future__ import annotations

PAD, SOS, EOS, OUT = 0, 1, 2, 3

PROMPTS = ("timestamp", "downbeat_meter", "structure", "key", "chord_majmin",
           "chord_full", "melody_vocal", "melody_full")
FULL_PROMPTS = ("timestamp", "downbeat_meter", "structure", "key", "chord_full",
                "melody_full")
PROMPT_FIELDS = {"timestamp": "timestamp", "downbeat_meter": "rhythm",
                 "structure": "structure", "key": "key", "chord_majmin": "chord",
                 "chord_full": "chord", "melody_vocal": "melody", "melody_full": "melody"}
FIELDS = ("timestamp", "rhythm", "structure", "key", "chord", "melody")

TIME_HZ = 100
WINDOW_SECONDS = 300.0
MAX_SHIFT = 256
MAX_TOKENS = 5120

SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
QUALITIES = ("maj", "min", "dim", "aug", "maj7", "min7", "7", "hdim7", "dim7",
             "minmaj7", "sus2", "sus4", "sus4(b7)", "maj6", "min6")
INVERSIONS = {"maj": ("/2", "/3", "/5"), "min": ("/2", "/b3", "/5"),
              "maj7": ("/3", "/5", "/7"), "min7": ("/b3", "/5", "/b7"),
              "7": ("/3", "/5", "/b7")}
STRUCTURE_LABELS = ("silence", "intro", "outro", "verse", "chorus", "bridge",
                    "pre-chorus", "post-chorus", "interlude", "fade-out", "loop", "rap",
                    "preshot", "irregular", "instrumental", "intro and verse",
                    "pre-chorus and chorus", "verse and pre-chorus", "solo", "theme",
                    "development", "variation", "pre-outro")
DURATION_STEPS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384,
                  512, 768, 1024, 1536, 2048, 3072, 4096)
METERS = tuple((numerator, denominator) for numerator in range(1, 33)
               for denominator in (1, 2, 4, 8, 16, 32))


def _full_chords() -> tuple:
    labels = ["N"]
    for quality in QUALITIES:
        for root in SHARPS:
            for inversion in INVERSIONS.get(quality, ()) + ("",):
                labels.append(root + ":" + quality + inversion)
    return tuple(labels)


FULL_CHORDS = _full_chords()
MAJMIN_CHORDS = (("N",) + tuple(root + ":maj" for root in SHARPS)
                 + tuple(root + ":min" for root in SHARPS))

_SIZES = (("prompt", 256), ("shift", MAX_SHIFT + 1), ("time", int(WINDOW_SECONDS) * TIME_HZ),
          ("meter", len(METERS)), ("eighth", 256), ("structure", len(STRUCTURE_LABELS)),
          ("key", 24), ("chord_majmin", len(MAJMIN_CHORDS)), ("chord_full", len(FULL_CHORDS)),
          ("pitch", 256), ("duration", len(DURATION_STEPS)))


def _blocks() -> tuple:
    blocks = {}
    start = 4
    for name, size in _SIZES:
        blocks[name] = (start, start + size)
        start += size
    return blocks, start


BLOCKS, SIZE = _blocks()
"""Each block's ``[start, end)`` ids, and the vocabulary size."""

if SIZE != 31678:
    raise ValueError("the SheetSage2 vocabulary must have 31678 tokens, this layout has {}".format(SIZE))

KIND_FIELDS = {"time": "timestamp", "meter": "rhythm", "eighth": "rhythm",
               "structure": "structure", "key": "key", "chord_majmin": "chord",
               "chord_full": "chord", "pitch": "melody", "duration": "melody"}


def kind(token) -> str:
    """The block a token id belongs to, or 'pad', 'sos', 'eos', 'out', 'prompt'."""
    token = int(token)
    for name, (start, end) in BLOCKS.items():
        if name != "prompt" and start <= token < end:
            return name
    if BLOCKS["prompt"][0] <= token < BLOCKS["prompt"][0] + len(PROMPTS):
        return "prompt"
    specials = {PAD: "pad", SOS: "sos", EOS: "eos", OUT: "out"}
    if token in specials:
        return specials[token]
    raise ValueError("token {} is outside the SheetSage2 vocabulary".format(token))


def first(name: str) -> int:
    """The first id of a block."""
    return BLOCKS[name][0]


def canonical_prompts(prompts) -> tuple:
    """Prompts in schema order, one per field, refusing unknown or clashing ones."""
    names = []
    for prompt in prompts:
        name = str(prompt).strip()
        if name.startswith("<|") and name.endswith("|>"):
            name = name[2:-2]
        if name not in PROMPTS:
            raise ValueError("unknown prompt: {!r}".format(prompt))
        if name not in names:
            names.append(name)
    names.sort(key=PROMPTS.index)
    fields = [PROMPT_FIELDS[name] for name in names]
    if len(set(fields)) != len(fields):
        raise ValueError("prompts {} ask for the same field twice".format(names))
    if not names:
        raise ValueError("at least one task prompt is required")
    return tuple(names)


def prompt_prefix(prompts=FULL_PROMPTS) -> list:
    """The tokens that open a sequence for these tasks."""
    return [SOS] + [first("prompt") + PROMPTS.index(name)
                    for name in canonical_prompts(prompts)] + [OUT]


def shift_tokens(shift: int) -> list:
    """A subbeat distance as shift tokens, 256 at most apiece."""
    shift = int(shift)
    if shift < 0:
        raise ValueError("a subbeat shift cannot be negative")
    tokens = []
    while shift > MAX_SHIFT:
        tokens.append(first("shift") + MAX_SHIFT)
        shift -= MAX_SHIFT
    tokens.append(first("shift") + shift)
    return tokens


def time_token(seconds: float) -> int:
    """The time stamp token nearest a time inside the window, clamped to it."""
    time_id = int(round(float(seconds) * TIME_HZ))
    time_id = max(0, min(time_id, int(WINDOW_SECONDS) * TIME_HZ - 1))
    return first("time") + time_id


def field_value(field: str, tokens: list):
    """What one field's tokens say, in the shapes the rest of the package reads."""
    kinds = [kind(token) for token in tokens]
    if field == "timestamp":
        return (tokens[0] - first("time")) / TIME_HZ
    if field == "rhythm":
        rhythm = {}
        for token, token_kind in zip(tokens, kinds):
            if token_kind == "meter":
                rhythm["meter"] = METERS[token - first("meter")]
            elif token_kind == "eighth":
                rhythm["eighth_position"] = token - first("eighth")
        return rhythm
    if field == "structure":
        return STRUCTURE_LABELS[tokens[0] - first("structure")]
    if field == "key":
        key_id = tokens[0] - first("key")
        return SHARPS[key_id % 12] + (":minor" if key_id >= 12 else ":major")
    if field == "chord":
        if kinds[0] == "chord_majmin":
            return MAJMIN_CHORDS[tokens[0] - first("chord_majmin")]
        return FULL_CHORDS[tokens[0] - first("chord_full")]
    notes = []
    index = 0
    while index < len(tokens):
        pitch_id = tokens[index] - first("pitch")
        duration_bin = 0
        if index + 1 < len(tokens) and kinds[index + 1] == "duration":
            duration_bin = tokens[index + 1] - first("duration")
            index += 2
        else:
            index += 1
        notes.append({"pitch": pitch_id % 128, "track": int(pitch_id >= 128),
                      "duration_bin": duration_bin, "duration_steps": DURATION_STEPS[duration_bin]})
    return notes


def refresh_values(event: dict) -> None:
    """Recompute an event's values from its tokens after the tokens were changed."""
    event["values"] = {field: field_value(field, tokens)
                       for field, tokens in event["tokens"].items() if tokens}


def _check_event(tokens_by_field: dict) -> None:
    for field, tokens in tokens_by_field.items():
        kinds = [kind(token) for token in tokens]
        if field == "timestamp" and kinds != ["time"]:
            raise ValueError("timestamp event must contain exactly one time token")
        if field == "rhythm" and kinds not in (["eighth"], ["meter", "eighth"], ["meter"]):
            raise ValueError("invalid rhythm payload: {}".format(kinds))
        if field in ("structure", "key", "chord") and len(tokens) != 1:
            raise ValueError("field {!r} must contain exactly one token".format(field))
        if field == "melody":
            index = 0
            while index < len(kinds):
                if kinds[index] != "pitch":
                    raise ValueError("melody payload must contain pitch tokens with optional duration")
                index += 2 if index + 1 < len(kinds) and kinds[index + 1] == "duration" else 1


def decode(tokens, strict: bool = True) -> dict:
    """A token sequence as prompts and timed events.

    Strict decoding refuses an event with no values, a value for a field no
    prompt asked for, and a sequence that does not end in eos. The window loop
    retries the two recoverable ones leniently, as the model's authors do. A
    meter with no eighth-note position after it, which the grammar lets the
    model end an event on, is a meter change that places no beat.
    """
    tokens = [int(token) for token in tokens]
    while tokens and tokens[-1] == PAD:
        tokens.pop()
    if not tokens or tokens[0] != SOS:
        raise ValueError("sequence must begin with <|sos|>")
    try:
        out_index = tokens.index(OUT, 1)
    except ValueError:
        raise ValueError("sequence is missing <|out|>") from None
    prompts = tuple(PROMPTS[token - first("prompt")] for token in tokens[1:out_index]
                    if kind(token) == "prompt")
    if len(prompts) != out_index - 1:
        raise ValueError("the prefix holds something other than prompts")
    if strict and canonical_prompts(prompts) != prompts:
        raise ValueError("prompt tokens are not in canonical schema order")
    active = {PROMPT_FIELDS[name] for name in prompts}

    events = []
    position = out_index + 1
    step = 0
    saw_eos = False
    while position < len(tokens):
        token = tokens[position]
        if token == EOS:
            saw_eos = True
            position += 1
            break
        if kind(token) != "shift":
            raise ValueError("event at token index {} has no subbeat shift".format(position))
        while position < len(tokens) and kind(tokens[position]) == "shift":
            step += tokens[position] - first("shift")
            position += 1
        by_field = {field: [] for field in FIELDS}
        while position < len(tokens):
            token = tokens[position]
            token_kind = kind(token)
            if token_kind == "shift" or token == EOS:
                break
            field = KIND_FIELDS.get(token_kind)
            if field is None:
                raise ValueError("token {} ({}) has no field".format(token, token_kind))
            if strict and field not in active:
                raise ValueError("token {} belongs to inactive output field {!r}".format(token, field))
            by_field[field].append(token)
            position += 1
        by_field = {field: values for field, values in by_field.items() if values}
        if not by_field:
            if strict:
                raise ValueError("empty event at subbeat {}".format(step))
            continue
        if strict:
            _check_event(by_field)
        event = {"subbeat": step, "tokens": by_field}
        refresh_values(event)
        events.append(event)
    if strict and not saw_eos:
        raise ValueError("sequence is missing <|eos|>")
    if strict and position != len(tokens):
        raise ValueError("non-padding tokens follow <|eos|>")
    return {"prompts": prompts, "events": events, "has_eos": saw_eos}


RECOVERABLE = ("empty event at subbeat", "belongs to inactive output field")


def decode_window(tokens) -> tuple:
    """A window's tokens decoded strictly, or leniently with a warning when that is allowed."""
    try:
        return decode(tokens, strict=True), None
    except ValueError as error:
        if not any(message in str(error) for message in RECOVERABLE):
            raise
        return decode(tokens, strict=False), str(error)


def encode(prompts, events, has_eos: bool = True) -> list:
    """Events back into tokens: the inverse of ``decode`` for sorted events."""
    tokens = prompt_prefix(prompts)
    previous = 0
    for event in events:
        step = int(event["subbeat"])
        if step < previous:
            raise ValueError("events must be sorted by subbeat")
        tokens.extend(shift_tokens(step - previous))
        previous = step
        for field in FIELDS:
            tokens.extend(int(token) for token in event["tokens"].get(field, ()))
    if has_eos:
        tokens.append(EOS)
    return tokens
