"""Where a remembered song's bars sit in its sound, and where an edit of some of them opens and closes.

A selection in the track window is a stretch of bars. Two things turn it into
frames of the song.

Where the bars are. The score says how long every bar is, and the sound says
where the first one starts and how fast they go by: YuE2 sings close to the
tempo of its score but not on it, and does not start its first bar where the
score does. The song's start comes from its voice: the loudness of the
separated voice laid against the notes the score sings, which is the one signal
that survives a song whose instrumental parts came out another length than the
score asked for. A rock song of the seven this was measured on sang a 33-bar
intro nine bars short, and since its melody repeats every eight bars, both the
pitch classes of the mix and the pitch of the voice locked eight bars off,
fourteen seconds from where the singing is; the loudness placed all seven.
Then the pitch classes of the mix are laid against the score's vocal notes and
chords, section by section around that start, and a line through the sections'
offsets gives the start and the speed -- 0.65 % slow on one song of the seven
and 2.7 % fast on another. Without the voice only the pitch classes place the
song, which was right on six of the seven, and the grid says which of the two
placed it.

Where the singing can be opened. A bar line is rarely where a phrase starts.
Every section of the stand's rap song begins a beat and a half before its
downbeat, and a cut from downbeat to downbeat took the first words of the next
chorus with it; a retake of that chorus afterwards sang its first line from the
top and ran a line late. So an edit opens at the phrase that sounds across the
bar line: in the rest before it when there is one, or, when the singing runs on
without a rest, where the section's own phrases usually begin before their bars.
Everything but ``measured`` is arithmetic on the score and needs no torch.
"""

from __future__ import annotations

import collections
import dataclasses
import math

from ..constants import FRAME_SECONDS

HOP_SECONDS = 0.01
"""The step the song's pitch classes are taken at."""

WINDOW_SECONDS = 0.04
"""The shortest window the pitch classes are taken over; the next power of two above it in samples."""

LOWEST_HZ, HIGHEST_HZ = 80.0, 2000.0
"""The band the pitch classes are read from: the voice and the chords, above the bass drum."""

CHORD_WEIGHT = 0.5
"""What a chord tone counts for in the score's picture against a sung note."""

SEARCH_SECONDS = 10.0
"""How far the song's start may be from the score's when the pitch classes place it alone."""

VOICE_SECONDS = 60.0
"""How far the song's start may be from the score's when the voice places it.

The voice can be trusted this far out: what moves a song this much is an
instrumental stretch the model sang at another length, and that stretch is as
long as a minute in the pack's own songs."""

VOICE_TOP = 95.0
"""The percentile of the voice's loudness taken as its full level, so one shout does not set the scale."""

SECTION_SECONDS = 1.0
"""How far a section may sit from where the pass before it put the song."""

PASSES = 4
"""How many times the sections are placed again with the speed found by the pass before.

A song whose tempo is a few percent off its score moves further than one pass
can search by the end, so each pass stretches the score by the speed found so
far and places the sections again. On the seven songs measured, the fourth
pass moved nothing by more than a millisecond."""

OUTLIER_SECONDS = 0.25
"""A section further than this from the line through the sections is left out of the line."""

STEEPEST = 0.03
"""The most a pass may stretch the score; a steeper line is held at this slope, and the next pass takes up the rest."""

OUTSIDE = "Bars {} to {} of the score are not a stretch of the song as it was sung."

MARGIN_SECONDS = 0.08
"""How far a seam in continuous singing is kept before the next phrase's first note: two frames."""


@dataclasses.dataclass(frozen=True)
class Seam:
    """Where an edit can open at one bar line, in score units before that bar's downbeat.

    ``pickup`` is where the phrase sounding across the bar line begins, before
    the downbeat; negative when the voice comes in after it. ``low`` and ``high``
    bound where a seam may go: from the phrase's start back to the end of the
    phrase before it, which is one point when the singing runs on without a rest.
    """

    bar: int
    pickup: float
    low: float
    high: float

    @property
    def room(self) -> float:
        return self.high - self.low


@dataclasses.dataclass(frozen=True)
class Grid:
    """The score's clock laid on the song's: audio seconds = ``offset`` + ``rate`` * score seconds.

    ``tick`` is the score seconds of one unit of L:, and ``starts`` the start
    of every bar in those units, with the end of the score after the last.
    ``by_voice`` says the separated voice placed the song's start, rather than
    the pitch classes of the mix alone.

    ``lines`` is the second each bar line really falls on, and is empty until
    an edit moves them off the clock: a cut takes frames out of the middle of
    a song without touching its tempo, and a retake may sing a second more or
    less than it replaced, so from there on no single line through the score
    reaches the bars after the edit. Everything that asks where a bar is goes
    through ``at`` and ``moment``, which answer from the clock while the lines
    are empty and from the lines once they are not.
    """

    offset: float
    rate: float
    tick: float
    starts: tuple
    by_voice: bool = False
    lines: tuple = ()

    def seconds(self, ticks: float) -> float:
        return self.offset + self.rate * self.tick * float(ticks)

    def frame(self, ticks: float) -> int:
        return int(round(self.seconds(ticks) / FRAME_SECONDS))

    def frames(self, ticks: float) -> int:
        """A length in score units as a whole number of frames."""
        return int(round(self.rate * self.tick * float(ticks) / FRAME_SECONDS))

    def at(self, bar: int) -> float:
        """The second the bar line ``bar`` falls on."""
        return self.lines[bar] if self.lines else self.seconds(self.starts[bar])

    def moment(self, bar: int, before: float = 0.0) -> int:
        """The frame that falls ``before`` score units ahead of the bar line ``bar``."""
        if self.lines:
            return int(round((self.lines[bar] - float(before) * self.rate * self.tick) / FRAME_SECONDS))
        return self.frame(self.starts[bar] - float(before))

    def bar_seconds(self) -> tuple:
        """The second of every bar line, the end of the last bar after them."""
        return self.lines if self.lines else tuple(self.seconds(start) for start in self.starts)


def tick_seconds(sheet) -> float:
    """The score seconds of one unit of L: in a sheet from ``notation.read``."""
    return 60.0 / (float(sheet["bpm"]) * sheet["per_quarter"])


def starts_of(sheet) -> tuple:
    """Every bar's start in units of L:, and the end of the score after the last."""
    return tuple(bar["start"] for bar in sheet["bars"]) + (sheet["total"],)


def phrases(notes) -> list:
    """The sung notes as runs that touch end to start: ``[(start, end, [note starts])]``, a rest of any length ending one."""
    out = []
    for note in sorted(notes, key=lambda item: item["start"]):
        end = note["start"] + note["length"]
        if out and note["start"] <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end), out[-1][2] + [note["start"]])
        else:
            out.append((note["start"], end, [note["start"]]))
    return out


def _section_bounds(sheet, bar: int):
    """The first bar line and the last of the section ``bar`` belongs to, in units of L:."""
    starts = starts_of(sheet)
    for section in sheet["sections"]:
        if section["bar"] <= bar < section["bar"] + section["bars"]:
            return starts[section["bar"]], starts[section["bar"] + section["bars"]]
    return starts[0], starts[-1]


def usual_pickup(sheet, runs, bar: int):
    """How far before a bar line the phrases of ``bar``'s section most often begin, or None when none cross one.

    Every phrase that begins after a rest inside the section counts: one that
    starts on a bar line with nothing before it, and one that starts in a bar
    and runs across the next line with the distance to that line. A tie
    between two lengths goes to the shorter.
    """
    first, last = _section_bounds(sheet, bar)
    starts = starts_of(sheet)
    leads = []
    for start, end, _notes in runs:
        if not first <= start < last:
            continue
        line = next((value for value in starts if value >= start), None)
        if line == start:
            leads.append(0)
        elif line is not None and end > line:
            leads.append(line - start)
    if not leads:
        return None
    counted = collections.Counter(leads)
    most = max(counted.values())
    return min(lead for lead, count in counted.items() if count == most)


def seam_at(sheet, runs, bar: int) -> Seam:
    """The Seam of one bar line of the score; bar 0 is the song's start and the last line its end.

    What decides it is the phrase sounding across the line, or the first one
    after it. When that phrase begins after the line, or within the bar before
    it, it is the new section's own, and a seam may go anywhere in the rest
    before it. When the singing runs on from further back, the note on the
    line tells the cases apart. A note held across the line ends the phrase
    before, and the section begins with the next phrase, after the rest; so
    does a note struck on the line of a section with no phrase of its own, an
    interlude after the last word of a chorus. Otherwise a note struck on the
    line means the phrase runs through it, and the section begins where its
    own phrases usually begin before their bars, if a note starts there, or
    else on the line itself.
    """
    starts = starts_of(sheet)
    if bar <= 0:
        return Seam(0, 0.0, 0.0, 0.0)
    if bar >= len(starts) - 1:
        return Seam(len(starts) - 1, 0.0, 0.0, 0.0)
    down = starts[bar]
    index = next((i for i, run in enumerate(runs) if run[1] > down), len(runs))
    before = runs[index - 1][1] if index > 0 else starts[0]
    if index == len(runs):
        return Seam(bar, float(down - starts[-1]), float(down - starts[-1]), float(down - before))
    start, end, onsets = runs[index]
    if start >= starts[bar - 1]:
        return Seam(bar, float(down - start), float(down - start), float(down - before))
    if down not in onsets and index + 1 < len(runs) and end <= starts[bar + 1]:
        following = runs[index + 1][0]
        return Seam(bar, float(down - following), float(down - following), float(down - end))
    lead = usual_pickup(sheet, runs, bar)
    if lead is None:
        following = runs[index + 1][0] if index + 1 < len(runs) else starts[-1]
        return Seam(bar, float(down - following), float(down - following), float(down - end))
    if down - lead in onsets:
        return Seam(bar, float(lead), float(lead), float(lead))
    return Seam(bar, 0.0, 0.0, 0.0)


def seams(sheet) -> list:
    """A Seam for every bar line of a sheet from ``notation.read``, from the song's start to its end."""
    runs = phrases(sheet["notes"]["Vocal"])
    return [seam_at(sheet, runs, bar) for bar in range(len(sheet["bars"]) + 1)]


def opening(seam: Seam, beat: float, margin: float) -> float:
    """Units of L: before the downbeat where a retake opens, or where one ends and the old song comes back.

    In a rest, a beat before the phrase at most and half the rest at most,
    and on the downbeat itself when the rest spans it; in continuous singing,
    ``margin`` before the phrase, so its first note is not cut.
    """
    if seam.room >= 2 * margin:
        near = seam.low + min(max(margin, beat), seam.room / 2)
        return min(max(0.0, near), seam.high - margin)
    return seam.low + margin


def cut_shift(first: Seam, second: Seam, beat: float, margin: float) -> float:
    """Units of L: that both ends of a cut move back by, so the cut stays whole bars.

    Both seams want the same shift: after the last word before the cut and
    before the first word cut, at the start; after the last word cut and before
    the first word kept, at the end. Where the two rests overlap by more than
    two margins the shift keeps a beat, or half the overlap, clear of the
    phrases, and stays on the downbeat if it can. Where they barely meet, the
    first word kept is what is protected.
    """
    low, high = max(first.low, second.low), min(first.high, second.high)
    if high - low >= 2 * margin:
        near = low + min(max(margin, beat), (high - low) / 2)
        return min(max(0.0, near), high - margin)
    if first.high >= second.low:
        return min(first.high, max(low, second.low + margin))
    return second.low


def _beat_and_margin(sheet, grid: Grid):
    beat = float(sheet["per_quarter"])
    margin = MARGIN_SECONDS / (grid.rate * grid.tick)
    return beat, margin


def _stretch(first: int, stop: int, start: int, end: int, frames: int):
    """Frames ``(start, end)`` of bars ``first`` to ``stop`` clipped to the song, or a ValueError when none of them is there.

    A song can come out shorter than its score, and an edit of bars the
    singing never reached has nothing to work on.
    """
    start, end = max(0, min(frames, start)), max(0, min(frames, end))
    if not start < end:
        raise ValueError(OUTSIDE.format(first + 1, stop))
    return start, end


def bars_of(sheet, first: int, stop: int) -> None:
    """Refuse bars ``first`` to ``stop`` that are not bars of ``sheet``, with a ValueError saying so."""
    count = len(sheet["bars"])
    if not 0 <= first < stop <= count:
        raise ValueError("Bars {} to {} are not bars of this score, which has {}.".format(first + 1, stop, count))


def cut_frames(sheet, grid: Grid, marks, first: int, stop: int, frames: int):
    """Frames ``(start, stop)`` of the song a cut of bars ``first`` to ``stop`` takes out.

    A cut that reaches the song's end takes the rest of it, and one from the
    start begins at frame 0; otherwise both ends move back by the same shift,
    so the stretch taken out is the bars' own length to within the rounding
    of each end to a frame. Bars the song never reached raise a ValueError.
    """
    bars_of(sheet, first, stop)
    beat, margin = _beat_and_margin(sheet, grid)
    if first <= 0:
        return _stretch(first, stop, 0, grid.moment(stop, opening(marks[stop], beat, margin)), frames)
    shift = cut_shift(marks[first], marks[stop], beat, margin)
    start = max(0, grid.moment(first, shift))
    if stop >= len(grid.starts) - 1:
        return _stretch(first, stop, start, frames, frames)
    return _stretch(first, stop, start, grid.moment(stop, shift), frames)


def retake_frames(sheet, grid: Grid, marks, first: int, stop: int, frames: int):
    """Frames ``(start, stop)`` of the song a retake of bars ``first`` to ``stop`` sings again.

    Each end opens where ``opening`` puts it; a retake reaching the song's end
    takes the rest of it, and one from the start begins at frame 0. Bars the
    song never reached raise a ValueError.
    """
    bars_of(sheet, first, stop)
    beat, margin = _beat_and_margin(sheet, grid)
    start = 0 if first <= 0 else grid.moment(first, opening(marks[first], beat, margin))
    end = frames if stop >= len(grid.starts) - 1 else grid.moment(stop, opening(marks[stop], beat, margin))
    return _stretch(first, stop, start, end, frames)


def after_cut(grid: "Grid", starts, first: int, stop: int, start: int, removed: int) -> "Grid":
    """The grid of the song a cut of bars ``first`` to ``stop`` leaves: ``removed`` frames from ``start`` are gone.

    ``starts`` are the bar starts of the score the cut leaves, which
    ``notation.without`` writes and ``starts_of`` reads. The bars before the
    cut stay where they were; the ones after it come back by exactly what came
    out, and the two sides meet on one line. A cut that ran into the end of
    the song took less than its bars' own length, and no line can then stand
    before where the cut began: what is left of the song ends there.
    """
    lines = grid.bar_seconds()
    gone = removed * FRAME_SECONDS
    floor = start * FRAME_SECONDS
    kept = lines[:first] + tuple(max(second - gone, floor) for second in lines[stop:])
    return dataclasses.replace(grid, starts=tuple(starts), lines=kept)


def after_retake(grid: "Grid", start: int, stop: int, count: int) -> "Grid":
    """The grid of the song after frames ``start`` to ``stop`` were sung again as ``count`` of them.

    A retake may come out a second or so from the length it replaced, and from
    there on the score no longer lies on the song by one line. The bars after
    it move by the difference; the bar lines inside it are spread through the
    new singing in the proportions the score has them, which is a guess -- the
    new take alone knows where it put its bars.
    """
    moved = (count - (stop - start)) * FRAME_SECONDS
    if not moved:
        return grid
    head, tail = start * FRAME_SECONDS, stop * FRAME_SECONDS
    span = tail - head
    lines = []
    for second in grid.bar_seconds():
        if second >= tail:
            lines.append(second + moved)
        elif second > head and span > 0:
            lines.append(head + (second - head) * (span + moved) / span)
        else:
            lines.append(second)
    return dataclasses.replace(grid, lines=tuple(lines))


def after_extend(grid: "Grid", starts, bar: int) -> "Grid":
    """The grid of a song that went on from bar line ``bar`` under a score whose bars start at ``starts``.

    The bars before that line are the old score's own and stay where they
    were; the ones after it are new, and fall at the song's own tempo from
    there, which is what the model sings a score it wrote at.
    """
    starts = tuple(starts)
    if not grid.lines:
        return dataclasses.replace(grid, starts=starts)
    line = grid.at(bar)
    step = grid.rate * grid.tick
    lines = tuple(grid.lines[:bar + 1]) + tuple(line + (start - starts[bar]) * step
                                                 for start in starts[bar + 1:])
    return dataclasses.replace(grid, starts=starts, lines=lines)


def _span(sheet, clock: Grid, bar: int) -> float:
    """Seconds of one unit of L: within bar ``bar``: the clock's own, or what the lines have left it."""
    length = float(sheet["bars"][bar]["length"]) if 0 <= bar < len(sheet["bars"]) else 0.0
    if not clock.lines or length <= 0:
        return clock.rate * clock.tick
    return (clock.at(bar + 1) - clock.at(bar)) / length


def layout(sheet, clock: Grid, frames: int) -> dict:
    """The grid as the track window draws it: every bar line and beat in seconds, and the sections.

    A section carries the second its singing begins as well as its first bar
    line, because that is where an edit of it opens, and the two are not the
    same wherever the words come in before the downbeat. A section that sings
    nothing before its end carries None there. ``notes`` is how many vocal
    notes start inside the section, which is what tells a sung section from
    an instrumental one when the words are laid on the score by count. After an edit a bar is as long
    as its lines say, and its beats and pickups are measured by that length.
    """
    marks = seams(sheet)
    beats = []
    for index, bar in enumerate(sheet["bars"]):
        span = _span(sheet, clock, index)
        for beat in range(max(1, int(round(bar["length"] / sheet["per_quarter"])))):
            beats.append(round(clock.at(index) + beat * sheet["per_quarter"] * span, 3))
    onsets = [note["start"] for note in sheet.get("notes", {}).get("Vocal", [])]
    sections = []
    for section in sheet["sections"]:
        stop = section["bar"] + section["bars"]
        pickup = marks[section["bar"]].pickup
        sung = clock.at(section["bar"]) - pickup * _span(
            sheet, clock, section["bar"] - 1 if pickup > 0 else section["bar"])
        low = sheet["bars"][section["bar"]]["start"]
        high = sheet["bars"][stop - 1]["start"] + sheet["bars"][stop - 1]["length"]
        sections.append({"name": section["name"], "bar": section["bar"], "bars": section["bars"],
                         "start": round(clock.at(section["bar"]), 3),
                         "end": round(clock.at(stop), 3),
                         "sung": None if sung >= clock.at(stop) else round(sung, 3),
                         "notes": sum(1 for tick in onsets if low <= tick < high)})
    return {"seconds": round(frames * FRAME_SECONDS, 3), "offset": round(clock.offset, 3),
            "rate": round(clock.rate, 6), "by_voice": bool(clock.by_voice),
            "bars": [round(clock.at(bar), 3) for bar in range(len(clock.starts))],
            "beats": beats, "sections": sections}


def line(points):
    """``(offset, drift)`` of the line through ``[(score seconds, offset, weight)]``, weighted, by least squares.

    One point, or points at one time, give a flat line through their weighted
    mean. While the point furthest from the line is more than
    ``OUTLIER_SECONDS`` off it and more than two are left, it is dropped and
    the line drawn again. A line steeper than ``STEEPEST`` is held at that
    slope, through the points' weighted mean, so that one pass of ``measured``
    can stretch the score no further and the next pass takes up the rest.
    """
    def fitted(chosen):
        total = sum(w for _t, _o, w in chosen)
        mean_t = sum(t * w for t, _o, w in chosen) / total
        mean_o = sum(o * w for _t, o, w in chosen) / total
        spread = sum(w * (t - mean_t) ** 2 for t, _o, w in chosen)
        if spread <= 1e-9:
            return mean_o, 0.0
        drift = sum(w * (t - mean_t) * (o - mean_o) for t, o, w in chosen) / spread
        if abs(drift) > STEEPEST:
            drift = math.copysign(STEEPEST, drift)
        return mean_o - drift * mean_t, drift

    kept = [point for point in points if point[2] > 0]
    if not kept:
        return 0.0, 0.0
    offset, drift = fitted(kept)
    while len(kept) > 2:
        worst = max(kept, key=lambda p: abs(p[1] - (offset + drift * p[0])))
        if abs(worst[1] - (offset + drift * worst[0])) <= OUTLIER_SECONDS:
            break
        kept = [p for p in kept if p is not worst]
        offset, drift = fitted(kept)
    return offset, drift


def _pitch_classes(waveform, sample_rate: int):
    """The song's pitch classes every ``HOP_SECONDS``: a 12-row tensor, each column of length one."""
    import torch

    mono = waveform.detach().float().cpu().reshape(-1, waveform.shape[-1]).mean(0)
    hop = int(round(sample_rate * HOP_SECONDS))
    size = 1 << int(math.ceil(math.log2(sample_rate * WINDOW_SECONDS)))
    freqs = torch.fft.rfftfreq(size, 1.0 / sample_rate)
    keep = (freqs > LOWEST_HZ) & (freqs < HIGHEST_HZ)
    classes = torch.round(69 + 12 * torch.log2(freqs[keep] / 440.0)).long() % 12
    padded = torch.nn.functional.pad(mono, (size // 2, size // 2))
    count = mono.shape[-1] // hop + 1
    window = torch.hann_window(size)
    out = torch.zeros(12, count)
    step = 3000
    for first in range(0, count, step):
        last = min(count, first + step)
        piece = padded[first * hop:(last - 1) * hop + size]
        spectrum = torch.stft(piece, size, hop, window=window, center=False, return_complex=True).abs()
        out[:, first:first + spectrum.shape[1]] = torch.zeros(12, spectrum.shape[1]).index_add_(
            0, classes, spectrum[keep].sqrt())
    return out / out.norm(dim=0, keepdim=True).clamp_min(1e-9)


def _picture(sheet, rate: float, count: int):
    """The score as pitch classes every ``HOP_SECONDS`` of its clock stretched by ``rate``: sung notes and chord tones."""
    import torch

    from ..midi.export import chord_pitches

    step = tick_seconds(sheet) * rate / HOP_SECONDS
    out = torch.zeros(12, count)
    for note in sheet["notes"]["Vocal"]:
        a = int(round(note["start"] * step))
        b = int(round((note["start"] + note["length"]) * step))
        if a < count:
            out[note["pitch"] % 12, max(0, a):max(0, min(count, b))] += 1.0
    chords = sheet["chords"]
    for index, chord in enumerate(chords):
        end = chords[index + 1]["start"] if index + 1 < len(chords) else sheet["total"]
        a, b = int(round(chord["start"] * step)), int(round(end * step))
        try:
            tones = {pitch % 12 for pitch in chord_pitches(chord["name"])}
        except ValueError:
            continue
        for tone in tones:
            out[tone, max(0, a):max(0, min(count, b))] += CHORD_WEIGHT
    return out


def _correlation(signal, kernel, first: int, last: int, start: int):
    """For every shift s from ``first`` to ``last``: the sum of ``signal[:, start + s + m] * kernel[:, m]``."""
    import torch

    width = kernel.shape[1]
    low = start + first
    high = start + last + width
    before = max(0, -low)
    after = max(0, high - signal.shape[1])
    window = torch.nn.functional.pad(signal, (before, after))[:, low + before:high + before]
    size = window.shape[1] + width
    spectrum = (torch.fft.rfft(window.double(), size) * torch.fft.rfft(kernel.double(), size).conj()).sum(0)
    return torch.fft.irfft(spectrum, size)[:last - first + 1]


def sung_ticks(sheet) -> list:
    """How many units of L: each section sings, which is how much its place in the song can be trusted."""
    starts = starts_of(sheet)
    out = []
    for section in sheet["sections"]:
        first, last = starts[section["bar"]], starts[section["bar"] + section["bars"]]
        out.append(sum(min(note["start"] + note["length"], last) - max(note["start"], first)
                       for note in sheet["notes"]["Vocal"]
                       if note["start"] < last and note["start"] + note["length"] > first))
    return out


def _offsets(sheet, signal, picture, rate: float, center: int, reach: int, weights):
    """``[(seconds, offset, weight)]`` for every section that carries weight.

    ``seconds`` is the middle of the section on the picture's clock, the
    score's stretched by ``rate``, and ``offset`` where the section sits best
    within ``reach`` hops of ``center``. A section that sings nothing is left
    out: an intro of chords alone sits where the song's own chords let it,
    which on one of the seven songs measured was 0.8 s from where its singing
    said the song was.
    """
    per_hop = tick_seconds(sheet) * rate / HOP_SECONDS
    starts = starts_of(sheet)
    out = []
    for section, weight in zip(sheet["sections"], weights):
        a = int(round(starts[section["bar"]] * per_hop))
        b = int(round(starts[section["bar"] + section["bars"]] * per_hop))
        if b <= a or weight <= 0 or not float(picture[:, a:b].sum()):
            continue
        values = _correlation(signal, picture[:, a:b], center - reach, center + reach, a)
        best = int(values.argmax()) + center - reach
        out.append(((a + b) / 2 * HOP_SECONDS, best * HOP_SECONDS, float(weight)))
    return out


def _loudness(voice, sample_rate: int, count: int):
    """The separated voice as one loudness value every ``HOP_SECONDS``, from silent at 0 to full at 1."""
    import torch

    mono = voice.detach().float().cpu().reshape(-1, voice.shape[-1]).mean(0)
    hop = int(round(sample_rate * HOP_SECONDS))
    frames = min(count, mono.shape[-1] // hop)
    level = torch.zeros(1, count)
    if frames <= 0:
        return level
    blocks = mono[:frames * hop].reshape(frames, hop)
    loud = blocks.pow(2).mean(dim=1).sqrt()
    top = float(torch.quantile(loud, VOICE_TOP / 100.0))
    level[0, :frames] = (loud / max(top, 1e-9)).clamp(0.0, 1.0)
    return level


def _sung(sheet, count: int):
    """The score's singing as one row: 1 while a vocal note sounds, 0 in the rests, with its mean taken off."""
    import torch

    step = tick_seconds(sheet) / HOP_SECONDS
    mask = torch.zeros(1, count)
    for note in sheet["notes"]["Vocal"]:
        a = int(round(note["start"] * step))
        b = int(round((note["start"] + note["length"]) * step))
        if a < count:
            mask[0, max(0, a):max(0, min(count, b))] = 1.0
    if float(mask.sum()) <= 0:
        return None
    return mask - mask.mean()


def _voice_start(sheet, voice, sample_rate: int, count: int):
    """Hops between the score's clock and the song's, from the voice, or None when the score sings nothing."""
    mask = _sung(sheet, count)
    if mask is None:
        return None
    level = _loudness(voice, sample_rate, count)
    reach = int(round(VOICE_SECONDS / HOP_SECONDS))
    values = _correlation(level, mask, -reach, reach, 0)
    return int(values.argmax()) - reach


def measured(sheet, waveform, sample_rate: int, voice=None) -> Grid:
    """The grid of a song from its sound and its score: where its bars start and how fast they go by.

    ``voice`` is the song's separated voice at the same rate, from the pack's
    own separator (see ``vocals_only``); with it the song's start is where its
    singing lines up with the score's, which holds however the instrumental
    parts came out. Without it the whole score's pitch classes are slid along
    the song's to find the start, which is right as long as the song follows
    its score's lengths.

    From that start, every section that sings is slid within a second of it
    and a line is drawn through them, weighted by how much each one sings. The
    score is then stretched by the speed that line found and the sections are
    placed again, ``PASSES`` times over, because a song a few percent off its
    score's tempo ends further out than one pass can search.
    """
    signal = _pitch_classes(waveform, sample_rate)
    count = signal.shape[1]
    reach = int(round(SEARCH_SECONDS / HOP_SECONDS))
    near = int(round(SECTION_SECONDS / HOP_SECONDS))
    start = None if voice is None else _voice_start(sheet, voice, sample_rate, count)
    by_voice = start is not None
    if start is None:
        start = int(_correlation(signal, _picture(sheet, 1.0, count), -reach, reach, 0).argmax()) - reach
    weights = sung_ticks(sheet)
    if not any(weights):
        weights = [1.0] * len(sheet["sections"])
    offset, rate = start * HOP_SECONDS, 1.0
    for _pass in range(PASSES):
        picture = _picture(sheet, rate, count + reach)
        points = _offsets(sheet, signal, picture, rate, int(round(offset / HOP_SECONDS)), near, weights)
        if not points:
            break
        offset, drift = line(points)
        rate *= 1.0 + drift
    return Grid(offset=offset, rate=rate, tick=tick_seconds(sheet),
                starts=starts_of(sheet), by_voice=by_voice)
