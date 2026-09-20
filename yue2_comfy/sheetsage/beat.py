"""The pulse of a recording, heard without a model.

SheetSage2 decides the beat itself, and on a long recording it can settle on a
pulse the music does not have. Measured 2026-09-20 on a five-minute song whose
beat is a steady 130 BPM: the transcription was written at 147, and the notes of
that score, laid against the recording's real beat, fall on it at chance -- 12%
on a beat where chance is 12%. The melody and the chords survive such a miss,
because they move slowly; the rhythm does not, and a cover sung from that score
does not sit in the beat. Nothing in the pack noticed, so this module hears the
pulse the cheap way and the node says so when the two numbers disagree.

It is not a beat tracker and does not try to be one. It answers how fast and how
sure, which is all a warning needs, and it answers in torch because the node
that asks is about to run a model on the same audio anyway.
"""

from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)

HOME = 120.0
"""The tempo a pulse is read towards when its half and its double both fit.

Every autocorrelation answers in octaves -- a beat is also a half-beat and a
double one -- and the peaks are often within a hair of each other. Measured on
the track that prompted this module, the true 130 BPM and its 65 scored 0.59
and 0.55. Reading towards the middle of what people call a tempo picks the one
a person would name.
"""

FRAMES_PER_SECOND = 86.0
"""Onset frames a second. Fine enough for a beat, coarse enough to be free."""

SLOWEST = 60.0
FASTEST = 200.0
"""The tempi looked for. Outside them a song is counted in halves or doubles."""

SURE_ENOUGH = 0.15
"""How strong the pulse must be before anything is said about it.

The strength is the autocorrelation of the onset track at the beat, against
itself at zero, so a steady drum machine scores around 0.6 and a recording with
no pulse at all scores near nothing. Below this the recording is not telling us
a tempo and a warning would be noise.
"""

SHORTEST = 20.0
"""Seconds of recording below which no tempo is offered."""


def heard(waveform, rate: int):
    """``{"bpm", "strength"}`` for the steadiest pulse, or None when there is none.

    *waveform* is ``[channels, samples]`` or ``[samples]`` at *rate*, as the node
    holds it. Nothing is resampled: the hop is chosen from the rate instead.
    """
    import torch

    mono = waveform.detach().float().cpu()
    if mono.dim() > 1:
        mono = mono.mean(dim=tuple(range(mono.dim() - 1)))
    seconds = mono.numel() / float(rate)
    if seconds < SHORTEST:
        return None
    hop = max(1, int(round(rate / FRAMES_PER_SECOND)))
    window = 4 * hop
    if mono.numel() < window * 4:
        return None
    spectrum = torch.stft(mono, n_fft=window, hop_length=hop, win_length=window,
                          window=torch.hann_window(window), center=True,
                          return_complex=True).abs()
    rise = torch.log1p(spectrum * 100.0).diff(dim=-1).clamp(min=0.0).sum(dim=0)
    rise = rise - rise.mean()
    if not torch.any(rise > 0):
        return None
    padded = torch.fft.rfft(rise, n=2 * rise.numel())
    both = torch.fft.irfft(padded * padded.conj())[:rise.numel()]
    if both[0] <= 0:
        return None
    both = both / both[0]
    per_second = float(rate) / hop
    first = max(1, int(round(60.0 / FASTEST * per_second)))
    last = min(both.numel() - 2, int(round(60.0 / SLOWEST * per_second)))
    if last <= first:
        return None
    best = int(torch.argmax(both[first:last + 1])) + first
    found = []
    for factor in (1.0, 0.5, 2.0):
        peak = _peak(both, int(round(best * factor)), first, last, per_second)
        if peak is not None:
            found.append(peak)
    if not found:
        return None
    strongest = max(value for _lag, value in found)
    chosen = min((lag for lag, value in found if value >= 0.7 * strongest),
                 key=lambda lag: abs(math.log(60.0 / lag / HOME)))
    return {"bpm": round(60.0 / chosen, 1),
            "strength": round(max(value for lag, value in found if lag == chosen), 3)}


def _peak(both, at: int, first: int, last: int, per_second: float):
    """``(lag in seconds, height)`` of the peak around a lag, read between the samples."""
    if not first < at < last:
        return None
    left, middle, right = float(both[at - 1]), float(both[at]), float(both[at + 1])
    bend = left + right - 2 * middle
    shift = 0.5 * (left - right) / bend if bend else 0.0
    lag = (at + max(-0.5, min(0.5, shift))) / per_second
    return (lag, middle) if lag > 0 else None


def agrees(measured: float, written: float, tolerance: float = 0.05) -> bool:
    """Whether two tempi are the same beat, counting halves and doubles as agreement.

    A pulse heard an octave out is the ordinary failure of every tempo estimate
    ever written, and a warning that fires on it would be wrong more often than
    right.
    """
    if not measured or not written:
        return True
    ratio = float(measured) / float(written)
    return any(abs(ratio * factor - 1.0) <= tolerance for factor in (1.0, 0.5, 2.0, 0.25, 4.0))


def disagreement(waveform, rate: int, written: float):
    """``(heard BPM, strength)`` when the recording's beat is not the score's, else None."""
    try:
        found = heard(waveform, rate)
    except Exception as error:
        log.info("[yue2_comfy.sheetsage.beat] the beat could not be measured: %s", error)
        return None
    if found is None or found["strength"] < SURE_ENOUGH:
        return None
    if agrees(found["bpm"], written):
        return None
    return found["bpm"], found["strength"]
