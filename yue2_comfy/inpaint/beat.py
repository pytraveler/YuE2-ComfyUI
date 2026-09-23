"""Where the beat falls in a stretch of sound, and how far it jumps across a seam.

A move lays pieces of a song one after another, cut where the grid says their
bar lines are. The grid is one line through the whole song, and a song can
stray from it: on a user's song of 2026-09-23 the model sang some sections
about 0.15 s off that line, and every seam of a move there stumbled by 76 to
146 ms, where the song itself never moves its beat by more than 18. The ear
hears that as the rhythm tripping.

The beat is read off the sound's onsets. The onset envelope -- how much the
spectrum rises from one 5 ms step to the next, above its own running mean --
is laid against a comb of teeth one pulse apart, and the phase where the teeth
catch the most onsets is where the pulse falls. The pulse is an eighth note:
on a comb of quarter notes a section that accents its off-beats reads as a
jump of half a beat where there is none. So a jump is known to within an
eighth, and the model, which hears the whole song, is left to say which eighth
(see ``core.moved``).

Nothing here needs a model; the arithmetic runs on the CPU.
"""

from __future__ import annotations

import math

import torch

HOP_SECONDS = 0.005
"""The step of the onset envelope."""

WINDOW = 2048
"""The spectrum's window, in samples at the song's rate."""

MEAN_SECONDS = 0.4
"""The running mean the envelope is taken above, so a loud stretch does not outweigh a quiet one."""

SMOOTH_SECONDS = 0.01
"""How far an onset is smeared either way, so a tooth a few milliseconds off still catches it."""

REACH_SECONDS = 6.0
"""How much sound on each side of a seam its beat is read from."""

GAP_SECONDS = 0.3
"""How much sound next to a seam is left out, the seam itself being drawn again."""

STRETCH = 0.03
"""How far the pulse may be from the grid's, either way: a song's tempo drifts from its line."""

STRETCH_STEPS = 25
"""How many pulses between those bounds are tried."""

PHASE_SECONDS = 0.002
"""The step the phase of the comb is tried at."""

WANDER = 0.05
"""How far, in seconds, the beat may drift between the halves of either side of a seam for its jump to be trusted.

Measured on 2026-09-23. At every seam of the stand's pop and rap moves, and
of a user's first move, the halves of each side agreed within 6 ms. At the
user's second move they agreed within 14 to 40 ms. The stand's Russian
ballad has no steady beat: its seams wandered 28 to 170 ms, and eight of its
own bar lines wandered more than 100 ms. A jump read there says nothing about
where the beat is."""


def envelope(sound, rate: int):
    """The onset envelope of ``sound`` ([channels, samples] or [samples]), one value every ``HOP_SECONDS``."""
    samples = torch.as_tensor(sound).detach().float().cpu()
    if samples.dim() > 1:
        samples = samples.reshape(-1, samples.shape[-1]).mean(0)
    hop = max(1, int(round(rate * HOP_SECONDS)))
    if samples.shape[-1] < WINDOW:
        samples = torch.nn.functional.pad(samples, (0, WINDOW - samples.shape[-1]))
    spectrum = torch.stft(samples, WINDOW, hop, window=torch.hann_window(WINDOW),
                          return_complex=True).abs()
    level = torch.log1p(1000.0 * spectrum)
    rise = torch.clamp(level[:, 1:] - level[:, :-1], min=0.0).sum(0)
    rise = torch.cat([torch.zeros(1), rise])
    width = max(1, int(round(MEAN_SECONDS / HOP_SECONDS)))
    mean = torch.nn.functional.avg_pool1d(rise[None, None], width, 1, width // 2,
                                          count_include_pad=False)[0, 0, :rise.shape[0]]
    onsets = torch.clamp(rise - mean, min=0.0)
    spread = max(1, int(round(SMOOTH_SECONDS / HOP_SECONDS)))
    taps = torch.arange(-3 * spread, 3 * spread + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (taps / spread) ** 2)
    kernel = kernel / kernel.sum()
    smooth = torch.nn.functional.conv1d(onsets[None, None], kernel[None, None],
                                        padding=3 * spread)[0, 0]
    return smooth.double()


def _comb(values, anchor: float, low: float, high: float, pulse: float):
    """``(phases, catch)``: for every phase of a comb of ``pulse`` through ``anchor``, the mean of ``values`` under its teeth in ``low`` to ``high`` seconds."""
    phases = torch.arange(0.0, pulse, PHASE_SECONDS, dtype=torch.float64)
    first = math.ceil((low - anchor - pulse) / pulse)
    last = math.floor((high - anchor) / pulse)
    teeth = anchor + phases[:, None] + pulse * torch.arange(first, last + 1, dtype=torch.float64)[None]
    inside = (teeth >= low) & (teeth <= high)
    where = teeth / HOP_SECONDS
    below = where.floor().clamp(0, values.shape[0] - 2)
    part = (where - below).clamp(0.0, 1.0)
    index = below.long()
    caught = values[index] * (1 - part) + values[index + 1] * part
    caught = torch.where(inside, caught, torch.zeros_like(caught))
    counts = inside.sum(1).clamp_min(1)
    return phases, caught.sum(1) / counts


def read(sound, rate: int, second: float, pulse: float):
    """``(seconds, wander)``: the jump of the beat across ``second``, and how steadily both sides keep it.

    ``pulse`` is the grid's eighth note in seconds. The comb's pulse is tried
    ``STRETCH`` either way of it and shared by both sides, each side read
    over ``REACH_SECONDS`` of sound, ``GAP_SECONDS`` clear of the seam; the
    jump is the phase after less the phase before, folded to within half a
    pulse. ``wander`` is how far the beat of either side's first half sits
    from its second half's, the most of the two, which is small where the
    song keeps time; None when there is not enough sound on both sides to read.
    """
    samples = torch.as_tensor(sound)
    length = samples.shape[-1] / float(rate)
    low, high = second - REACH_SECONDS, second + REACH_SECONDS
    if low < 0 or high > length or pulse <= 4 * PHASE_SECONDS:
        return 0.0, None
    first = int(math.floor(low * rate))
    values = envelope(samples[..., first:int(math.ceil(high * rate))], rate)
    offset = first / float(rate)
    anchor = second - offset
    best = None
    for pulse_tried in torch.linspace(pulse * (1 - STRETCH), pulse * (1 + STRETCH), STRETCH_STEPS).tolist():
        phases, before = _comb(values, anchor, 0.0, anchor - GAP_SECONDS, pulse_tried)
        _phases, after = _comb(values, anchor, anchor + GAP_SECONDS, 2 * REACH_SECONDS, pulse_tried)
        score = float(before.max() + after.max())
        if best is None or score > best[0]:
            best = (score, pulse_tried, phases, before, after)
    _score, pulse_found, phases, before, after = best
    one = float(phases[int(before.argmax())])
    two = float(phases[int(after.argmax())])
    moved = _folded(two - one, pulse_found)
    wander = 0.0
    for low_side, high_side in ((0.0, anchor - GAP_SECONDS), (anchor + GAP_SECONDS, 2 * REACH_SECONDS)):
        middle = (low_side + high_side) / 2
        _p, early = _comb(values, anchor, low_side, middle, pulse_found)
        _p, late = _comb(values, anchor, middle, high_side, pulse_found)
        wander = max(wander, abs(_folded(float(phases[int(late.argmax())])
                                         - float(phases[int(early.argmax())]), pulse_found)))
    return moved, wander


def _folded(seconds: float, pulse: float) -> float:
    """``seconds`` folded to within half a ``pulse`` either way."""
    return (seconds + pulse / 2) % pulse - pulse / 2


def jump(sound, rate: int, second: float, pulse: float):
    """The jump of the beat across ``second`` in seconds, as ``read`` finds it; None where it cannot be trusted (``WANDER``)."""
    moved, wander = read(sound, rate, second, pulse)
    if wander is None or wander > WANDER:
        return None
    return moved


def counts_on_beat(length: int, shortest: int, longest: int, moved: float, pulse: float,
                   frame: float):
    """The lengths of a new stretch between ``shortest`` and ``longest`` frames that put the beat back.

    The new frames go on from the beat before the seam, so what comes after
    them jumps by ``moved`` plus a frame's worth for every frame more than
    ``length``; the lengths kept are the ones nearest to a jump of none, one
    for every eighth either way, which the model chooses among. Empty when
    none lies in the bounds.
    """
    found = set()
    reach = int(math.ceil((longest - shortest + 1) * frame / pulse)) + 1
    for eighths in range(-reach, reach + 1):
        count = length + int(round((eighths * pulse - moved) / frame))
        if shortest <= count <= longest:
            found.add(count)
    return sorted(found)
