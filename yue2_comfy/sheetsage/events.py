"""A whole song from 300-second windows, and the rows a score is built from.

The model hears at most 300 seconds at a time. A longer song is read in
windows that start 100 seconds apart: each window keeps what it says about its
first 200 seconds and leaves the rest to the next one, which starts with the
events already kept for the overlap as a prefix, so it continues them rather
than beginning again. Inside a window, events are counted in subbeats and only
some carry a time stamp; the rest are placed by interpolating between stamps.

A window can run out of the decoder's tokens before the end of the part it
answers for: a dense recording spends all 5120 of them a few seconds early. The
next window heard that music too, so ``resume_point`` hands it the seconds the
one before it never reached, and the song keeps its beats across the seam.

What comes out is a list of timed events. ``score_rows`` turns them into the
beats, chords, keys, sections and notes that ``abc_rebuild`` writes a score
from, applying the same clean-up the model's authors apply: the beat grid is
extended to the end of the song, and a note running into the next one in its
track is cut at that onset.
"""

from __future__ import annotations

import bisect
import math
from fractions import Fraction

from . import vocab

EPS = 1e-4

MINUTE = 60.0
"""The short window: a minute of a recording at a time, for a song heard at the wrong tempo.

The model decides a window's beat once and writes everything else against it,
so a pulse it settles on wrongly early in a long window is carried to the end
of that window. Measured on the recording of ``beat``, whose beat is a steady
130.1 BPM: heard whole, the score came out at 147 BPM with 23 percent of its
sung notes on the recording's own beat; heard a minute at a time, at 130 BPM
with 40 percent, and no slower, because a shorter window decodes less.

A window still starts from the events kept for its overlap. Letting every
minute start from nothing was measured as well: no closer to the beat, worse
bars, twice the time.
"""


def plan_for(duration: float, window: float = vocab.WINDOW_SECONDS) -> list:
    """The windows of a recording at a chosen window length, overlapping as they always have.

    Two thirds of a window overlap the one before it and a third of it is kept,
    which at the full 300 seconds is the 200 and 100 the plan has always used.
    """
    return window_plan(duration, window=window, overlap=window * 2.0 / 3.0, lookahead=window / 3.0)


def window_plan(duration: float, window: float = vocab.WINDOW_SECONDS,
                overlap: float = 200.0, lookahead: float = 100.0) -> list:
    """Where each window starts and ends, which part of it is kept, and where decoding stops."""
    if not duration > 0:
        raise ValueError("a song needs a positive length")
    if not 0 <= lookahead <= overlap < window:
        raise ValueError("require 0 <= lookahead <= overlap < window")
    hop = window - overlap
    start = 0.0
    accepted = 0.0
    plan = []
    while True:
        last = start + window >= duration - 1e-6
        accept_end = duration if last else start + window - lookahead
        plan.append({"start": start, "end": min(duration, start + window),
                     "accept_start": accepted, "accept_end": accept_end, "prefix_end": accepted,
                     "stop": None if last else window - lookahead})
        if last:
            return plan
        accepted = accept_end
        start = min(start + hop, duration - window)


def stop_seconds(window: dict, duration: float, length: float = vocab.WINDOW_SECONDS) -> float:
    """The window time past which decoding ends, as soon as a time stamp reaches it."""
    if window["stop"] is not None:
        return window["stop"]
    return min(duration - window["start"], length)


def _clip(value, low, high):
    return min(max(value, low), high)


def _median(values: list) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _interp(x: float, xs: list, ys: list) -> float:
    """Linear interpolation inside ``xs``, in the order of operations numpy uses."""
    index = bisect.bisect_right(xs, x) - 1
    if xs[index] == x:
        return ys[index]
    slope = (ys[index + 1] - ys[index]) / (xs[index + 1] - xs[index])
    return slope * (x - xs[index]) + ys[index]


def time_map(decoded: dict, target: float = vocab.WINDOW_SECONDS):
    """A function from a window's subbeat to seconds in that window.

    Stamped events are the anchors; between them time is interpolated, and
    past them it runs on at the median seconds per subbeat. A window with no
    stamp at all assumes an eighth of a second per subbeat.
    """
    anchors = {}
    for event in decoded["events"]:
        value = event["values"].get("timestamp")
        if value is not None:
            anchors[int(event["subbeat"])] = float(value)
    target = float(target)
    if not anchors:
        return lambda step: min(target, max(0.0, float(step) * 0.125))
    items = sorted(anchors.items())
    steps = [float(step) for step, _ in items]
    times = [time for _, time in items]
    per_step = 0.125
    if len(items) >= 2:
        ratios = [(times[i + 1] - times[i]) / max(steps[i + 1] - steps[i], 1.0)
                  for i in range(len(items) - 1)]
        median = _median(ratios)
        if math.isfinite(median) and median > 0:
            per_step = median

    def lookup(step):
        step = float(step)
        if step <= steps[0]:
            return _clip(times[0] + (step - steps[0]) * per_step, 0.0, target)
        if step >= steps[-1]:
            return _clip(times[-1] + (step - steps[-1]) * per_step, 0.0, target)
        return _interp(step, steps, times)

    return lookup


def _copy(event: dict) -> dict:
    return {"subbeat": int(event["subbeat"]),
            "tokens": {field: [int(token) for token in tokens]
                       for field, tokens in event["tokens"].items()},
            "values": {field: ([dict(note) for note in value] if field == "melody" else
                               dict(value) if isinstance(value, dict) else value)
                       for field, value in event["values"].items()}}


def stitch(decoded: dict, lookup, window: dict, duration: float, index: int, base: int = 0) -> list:
    """A window's events that fall in its kept span, in song time, with note ends placed."""
    kept = []
    for event in decoded["events"]:
        local = float(lookup(event["subbeat"]))
        absolute = float(window["start"]) + local
        if absolute < float(window["accept_start"]) - EPS:
            continue
        if absolute >= float(window["accept_end"]) - EPS or absolute >= float(duration) - EPS:
            continue
        output = _copy(event)
        output["time"] = _clip(absolute, 0.0, float(duration))
        output["window_index"] = int(index)
        output["window_start"] = float(window["start"])
        output["source_subbeat"] = int(event["subbeat"])
        output["global_subbeat"] = int(base) + int(event["subbeat"])
        if "timestamp" in output["values"]:
            output["values"]["timestamp"] = output["time"]
        for note in output["values"].get("melody", ()):
            local_end = float(lookup(int(event["subbeat"]) + int(note["duration_steps"])))
            end = float(window["start"]) + local_end
            note["end_time"] = min(float(duration), max(output["time"] + 0.04, end))
        kept.append(output)
    return kept


def resume_point(window: dict, kept: list, gap_beats: float = 1.5, subbeats: int = 4) -> float:
    """The time the next window should take over from, or 0.0 when this one reached its end.

    A window that has spent the decoder's tokens stops before the end of the
    span it answers for, and what it never said is lost: the next window drops
    everything before the planned boundary. The seconds between the two are a
    hole in the beat grid, which skips beats and stretches the subbeats around
    them until a short note no longer fits. The next window has heard that music
    as well, so it takes over at the last event of this one instead.

    A window normally stops a fraction of a beat early, and a hole shorter than
    ``gap_beats`` beats holds no beat of its own: those are left where they are,
    so an ordinary song is stitched exactly where ComfyUI stitches it. The beat
    is this window's own, measured over its last beats; a window that placed no
    beat at all counts a second instead.

    The point sits half a subbeat past the last event, the closest the score's
    grid can tell two moments apart: what the next window says in that half
    subbeat is what the window before it has already said.
    """
    if not kept:
        return 0.0
    last = max(float(event["time"]) for event in kept)
    times = [row[0] for row in beats(kept)]
    steps = [b - a for a, b in zip(times[-9:], times[-8:])]
    period = _median(steps) if steps else 1.0
    if float(window["accept_end"]) - last <= gap_beats * period:
        return 0.0
    return last + max(2 * EPS, period / (2.0 * subbeats))


def _context_before(events: list, moment: float) -> dict:
    context = {}
    for event in events:
        time = event.get("time")
        if time is None or float(time) > float(moment) + 1e-6:
            continue
        for field in ("structure", "key", "chord"):
            if event["tokens"].get(field):
                context[field] = [int(token) for token in event["tokens"][field]]
        meters = [int(token) for token in event["tokens"].get("rhythm", ())
                  if vocab.kind(token) == "meter"]
        if meters:
            context["meter"] = meters[:1]
    return context


def carried_prefix(stitched: list, prompts, window: dict) -> tuple:
    """The tokens a window starts from: what is already kept of its overlap.

    Returns ``(tokens, base)``, where base is the song subbeat the prefix's
    first event sits at, or ``(None, None)`` when the overlap holds no beat.
    The first event carries the section, key, chord and meter in force there,
    so the window continues the song's context instead of guessing it.
    """
    start = float(window["start"])
    source = [event for event in stitched
              if start - EPS <= float(event.get("time", -1.0)) < float(window["prefix_end"]) - EPS]
    source.sort(key=lambda event: (int(event.get("global_subbeat", event.get("source_subbeat", event["subbeat"]))),
                                   float(event.get("time", 0.0))))
    first_beat = next((i for i, event in enumerate(source)
                       if "timestamp" in event["values"] or "rhythm" in event["values"]), None)
    if first_beat is None:
        return None, None
    source = source[first_beat:]

    def song_subbeat(event):
        return int(event.get("global_subbeat", event.get("source_subbeat", event["subbeat"])))

    base = song_subbeat(source[0])
    context = _context_before(stitched, source[0]["time"])
    prefix = []
    for original in source:
        event = _copy(original)
        event["subbeat"] = max(0, song_subbeat(original) - base)
        if "timestamp" in event["tokens"]:
            event["tokens"]["timestamp"] = [vocab.time_token(float(original["time"]) - start)]
        vocab.refresh_values(event)
        prefix.append(event)
    head = prefix[0]
    for field in ("structure", "key", "chord"):
        if field not in head["tokens"] and field in context:
            head["tokens"][field] = list(context[field])
    rhythm = list(head["tokens"].get("rhythm", ()))
    kinds = [vocab.kind(token) for token in rhythm]
    if "eighth" in kinds and "meter" not in kinds and "meter" in context:
        head["tokens"]["rhythm"] = list(context["meter"]) + rhythm
    vocab.refresh_values(head)
    return vocab.encode(prompts, prefix, has_eos=False), base


def sort_song(events: list) -> list:
    """Kept events from every window in song order."""
    return sorted(events, key=lambda event: (event["time"], event["global_subbeat"]))


def _intervals(events: list, field: str, duration: float) -> list:
    rows = [[event["time"], 0, event["values"][field]] for event in events if field in event["values"]]
    for i, row in enumerate(rows):
        row[1] = rows[i + 1][0] if i + 1 < len(rows) else duration
    return [row for row in rows if row[1] > row[0]]


def beats(events: list) -> list:
    """``[time, beat number, numerator, denominator]`` for every event that places a beat."""
    rows = []
    meter = None
    for event in events:
        rhythm = event["values"].get("rhythm", {})
        meter = rhythm.get("meter", meter)
        eighth = rhythm.get("eighth_position")
        if eighth is not None and meter is not None:
            position = Fraction(int(eighth) * int(meter[1]), 8)
            if position.denominator != 1:
                raise ValueError("Eighth position {} is off the {}/{} beat grid".format(eighth, meter[0], meter[1]))
            if not 0 <= position < meter[0]:
                raise ValueError("Eighth position {} is outside meter {}".format(eighth, meter))
            rows.append([float(event["time"]), int(position) + 1, int(meter[0]), int(meter[1])])
    return rows


def _period_near(rows: list, index: int, span: int = 8) -> float:
    """The beat period around a row, taken from the gaps on both sides of it."""
    low, high = max(0, index - span), min(len(rows), index + span + 1)
    gaps = sorted(float(b[0]) - float(a[0]) for a, b in zip(rows[low:high - 1], rows[low + 1:high]))
    return gaps[len(gaps) // 2] if gaps else 0.0


def filled_beats(rows: list) -> list:
    """Beat rows with the beats a window seam dropped put back where they belonged.

    A window that has spent its tokens stops before the end of its span and the
    next one takes over from there (see ``resume_point``); the beat neither of
    them placed is simply gone. The bar it belonged to is then written a beat
    short, and since every bar of this dialect has to be as long as its meter
    says, the only way to write it is a meter change there and another one
    back -- four lines of ``M:`` around one hole, in both voices.

    Two witnesses are wanted before a beat is invented, because a gap in the
    beat is not always a lost beat: the clock has room for whole beats at the
    period around it, and the numbering steps over exactly that many. Measured
    on the eight recordings of ``MINUTE``, heard a minute at a time: seven
    holes, and with them filled two scores come out with no meter change left
    at all (9 to 1 and 5 to 1) while four others lose the pair each hole cost
    (49 to 45, 45 to 41, 29 to 25, 19 to 15). Not one beat is added to any of
    the same recordings heard whole, whose seams are a hundred seconds apart,
    so a score that is written well today is written the same way.

    What is left after this is not a seam: the model re-counts the bar inside a
    window and writes a one-beat bar where it starts the count again. Those
    bars are its own reading and not an accident -- 87 to 99 percent of the
    chord changes of a melodic recording land on its downbeats, against 48 to
    76 percent when one bar phase is forced on the whole song -- so they stay.
    """
    out = []
    for index, row in enumerate(rows):
        if out:
            previous = out[-1]
            period = _period_near(rows, index)
            gap = float(row[0]) - float(previous[0])
            missing = int(round(gap / period)) - 1 if period > 0 else 0
            skipped = (int(row[1]) - int(previous[1]) - 1) % int(previous[2])
            if 1 <= missing <= 3 and gap > 1.6 * period and skipped == missing:
                for step in range(1, missing + 1):
                    out.append([float(previous[0]) + gap * step / (missing + 1),
                                (int(previous[1]) - 1 + step) % int(previous[2]) + 1,
                                int(previous[2]), int(previous[3])])
        out.append(list(row))
    return out


def notation_notes(notes: list) -> list:
    """One note at a time per track: a note sounding into the next onset is cut there."""
    result = []
    for track in (0, 1):
        ordered = sorted((list(note) for note in notes if note[3] == track),
                         key=lambda note: (note[0], note[2], note[1]))
        for i, note in enumerate(ordered):
            if i + 1 < len(ordered) and note[1] > ordered[i + 1][0] + 1e-6:
                note[1] = ordered[i + 1][0]
            if note[1] > note[0] + 1e-6:
                result.append(note)
    return sorted(result)


def score_rows(events: list, duration: float) -> dict:
    """Everything ``abc_rebuild.build`` needs, from a song's sorted events.

    The beats a window seam dropped are put back first (see ``filled_beats``),
    because every later step counts bars by the beats it is given.

    Raises ValueError with the reason a score cannot be written: fewer than two
    beats, beats that do not move forward, or no key at all.
    """
    notes = []
    for event in events:
        start = float(event["time"])
        for note in event["values"].get("melody", ()):
            end = min(duration, float(note["end_time"]))
            if end > start:
                notes.append([start, end, int(note["pitch"]), int(note["track"])])
    notes.sort()
    rows = filled_beats(beats(events))
    if len(rows) < 2:
        raise ValueError("At least two decoded beats are required for ABC")
    grid = [list(row) for row in rows]
    recent = [row[0] for row in rows[-9:]]
    period = _median([b - a for a, b in zip(recent, recent[1:])])
    if period <= 0:
        raise ValueError("Decoded beats must increase in time")
    end = max(duration, max((note[1] for note in notes), default=0))
    while grid[-1][0] < end - 1e-6:
        previous = grid[-1]
        grid.append([previous[0] + period, previous[1] % previous[2] + 1, previous[2], previous[3]])
    clipped = {}
    for field in ("chord", "key", "structure"):
        clipped[field] = [[max(grid[0][0], a), min(grid[-1][0], b), value]
                          for a, b, value in _intervals(events, field, duration)
                          if b > grid[0][0] and a < grid[-1][0]]
    if not _intervals(events, "key", duration):
        raise ValueError("No key was decoded; cannot construct a keyed ABC score")
    return {"beats": grid, "chords": clipped["chord"], "keys": clipped["key"],
            "structures": clipped["structure"], "notes": notation_notes(notes)}
