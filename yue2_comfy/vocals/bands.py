"""Which spectrogram bins each band of Mel-Band RoFormer reads.

The model cuts a 2048-sample STFT at 44.1 kHz (1025 bins) into sixty
overlapping bands: the bins under each triangle of a Slaney-scale mel filter
bank from 0 Hz to Nyquist, taken as a yes or no rather than as weights. A bin
under two triangles belongs to both, and the masks the model estimates for it
are averaged afterwards.

The triangles are computed with the same double-precision arithmetic in the
same order as librosa's ``filters.mel`` with its defaults, so the memberships
match the ones the released weights were trained on to the bin. Two entries
are forced, as the training code forced them: the 0 Hz bin, which sits exactly
on the first triangle's foot, and the Nyquist bin of the last band, which
rounding puts on one side of zero or the other depending on the machine.

Pure Python, so the band plan can be checked where torch is not installed.
"""

from __future__ import annotations

import functools
import math

SAMPLE_RATE = 44100
N_FFT = 2048
BANDS = 60
CHANNELS = 2

_SPACING_HZ = 200.0 / 3.0
_LOG_START_HZ = 1000.0
_LOG_START_MEL = _LOG_START_HZ / _SPACING_HZ
_LOG_STEP = math.log(6.4) / 27.0


def hz_to_mel(hz: float) -> float:
    """Slaney's mel scale: linear below 1 kHz, logarithmic above."""
    if hz >= _LOG_START_HZ:
        return _LOG_START_MEL + math.log(hz / _LOG_START_HZ) / _LOG_STEP
    return hz / _SPACING_HZ


def mel_to_hz(mel: float) -> float:
    """The inverse of ``hz_to_mel``."""
    if mel >= _LOG_START_MEL:
        return _LOG_START_HZ * math.exp(_LOG_STEP * (mel - _LOG_START_MEL))
    return _SPACING_HZ * mel


def bin_frequencies(sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT) -> list:
    """The centre frequency of every STFT bin, as ``numpy.fft.rfftfreq`` computes it."""
    step = 1.0 / (n_fft * (1.0 / sample_rate))
    return [index * step for index in range(n_fft // 2 + 1)]


def mel_edges(bands: int = BANDS, sample_rate: int = SAMPLE_RATE) -> list:
    """``bands + 2`` frequencies evenly spaced in mels from 0 Hz to Nyquist: every triangle's foot and peak."""
    low, high = hz_to_mel(0.0), hz_to_mel(sample_rate / 2.0)
    count = bands + 2
    step = (high - low) / (count - 1)
    mels = [index * step + low for index in range(count)]
    mels[-1] = high
    return [mel_to_hz(mel) for mel in mels]


@functools.lru_cache(maxsize=None)
def memberships(bands: int = BANDS, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT) -> tuple:
    """For each band, the ascending bins under its triangle, as a tuple of tuples.

    Raises ``ValueError`` if some bin falls under no triangle, which the model
    was never trained to handle.
    """
    freqs = bin_frequencies(sample_rate, n_fft)
    edges = mel_edges(bands, sample_rate)
    found = []
    for band in range(bands):
        rise = edges[band + 1] - edges[band]
        fall = edges[band + 2] - edges[band + 1]
        inside = []
        for index, freq in enumerate(freqs):
            lower = -(edges[band] - freq) / rise
            upper = (edges[band + 2] - freq) / fall
            if max(0.0, min(lower, upper)) > 0.0:
                inside.append(index)
        found.append(inside)
    if 0 not in found[0]:
        found[0].insert(0, 0)
    if len(freqs) - 1 not in found[-1]:
        found[-1].append(len(freqs) - 1)
    covered = set()
    for inside in found:
        covered.update(inside)
    if len(covered) != len(freqs):
        raise ValueError("the mel bands leave {} spectrogram bins unread".format(len(freqs) - len(covered)))
    return tuple(tuple(inside) for inside in found)


def band_widths(bands: int = BANDS, channels: int = CHANNELS) -> list:
    """How many numbers each band hands the network: bins times channels times the two halves of a complex value."""
    return [2 * channels * len(inside) for inside in memberships(bands)]


def gather_order(bands: int = BANDS, channels: int = CHANNELS) -> list:
    """Every band's bins, band after band, each bin once per channel: indices into bins interleaved by channel."""
    order = []
    for inside in memberships(bands):
        for index in inside:
            order.extend(index * channels + channel for channel in range(channels))
    return order


def bands_per_bin(bands: int = BANDS, n_fft: int = N_FFT) -> list:
    """For each bin, how many bands read it: what an averaged mask is divided by."""
    counts = [0] * (n_fft // 2 + 1)
    for inside in memberships(bands):
        for index in inside:
            counts[index] += 1
    return counts
