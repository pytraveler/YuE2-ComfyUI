"""Mel-Band RoFormer's network in plain torch, shaped like Kimberley Jensen's vocal model.

The shapes are those of the released weights, and module names follow the
file's keys, so the weights load without a rename table.

A stereo 44.1 kHz chunk goes through a 2048-sample STFT with a 441-sample
hop. The two channels are laid side by side along frequency, and the sixty
mel bands of ``bands`` pick their bins, real and imaginary halves side by
side. Each band is normalised and projected to 384 numbers. Six layers then
attend across time within each band and across bands within each frame,
every attention a pre-norm block of eight heads of 64 with rotary positions
and a sigmoid gate per head, followed by a four-times feed-forward, and each
direction ending in its own norm.

A two-hidden-layer tanh network per band, closed by a GLU, turns the result
back into a complex mask for every bin the band reads. Bins read by several
bands get the mean of their masks. The masked spectrogram goes back through
the inverse STFT: the vocals of the chunk.

The rotary angles and the spectrogram arithmetic always run in float32. The
weights may be half precision, and an angle as large as 800 radians does not
survive float16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import bands

WIDTH = 384
LAYERS = 6
HEADS = 8
HEAD = 64
FEED_FORWARD = 4 * WIDTH
MASK_HIDDEN = 4 * WIDTH
HOP = 441
WINDOW = bands.N_FFT
ROTARY_BASE = 10000.0


class RMSNorm(nn.Module):
    """Unit length times the square root of the width, then a learned gain."""

    def __init__(self, width: int):
        super().__init__()
        self.scale = width ** 0.5
        self.gamma = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class Rotary(nn.Module):
    """Rotary positions over neighbouring pairs of a head's numbers, with the frequencies the weights carry."""

    def __init__(self):
        super().__init__()
        self.freqs = nn.Parameter(1.0 / (ROTARY_BASE ** (torch.arange(0, HEAD, 2).float() / HEAD)),
                                  requires_grad=False)

    def angles(self, length: int, device) -> tuple:
        positions = torch.arange(length, device=device, dtype=torch.float32)
        turns = positions[:, None] * self.freqs.float()[None, :]
        turns = turns.repeat_interleave(2, dim=-1)
        return turns.cos(), turns.sin()

    @staticmethod
    def rotate(x, cos, sin):
        pairs = x.float().unflatten(-1, (-1, 2))
        turned = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return (x.float() * cos + turned * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(WIDTH)
        self.rotary_embed = Rotary()
        self.to_qkv = nn.Linear(WIDTH, 3 * HEADS * HEAD, bias=False)
        self.to_gates = nn.Linear(WIDTH, HEADS)
        self.to_out = nn.Sequential(nn.Linear(HEADS * HEAD, WIDTH, bias=False))

    def forward(self, x):
        x = self.norm(x)
        batch, length, _ = x.shape
        q, k, v = self.to_qkv(x).reshape(batch, length, 3, HEADS, HEAD).permute(2, 0, 3, 1, 4)
        cos, sin = self.rotary_embed.angles(length, x.device)
        q = Rotary.rotate(q, cos, sin)
        k = Rotary.rotate(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)
        gates = self.to_gates(x).sigmoid().transpose(1, 2).unsqueeze(-1)
        out = (out * gates).transpose(1, 2).reshape(batch, length, HEADS * HEAD)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(WIDTH), nn.Linear(WIDTH, FEED_FORWARD), nn.GELU(), nn.Identity(),
                                 nn.Linear(FEED_FORWARD, WIDTH), nn.Identity())

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    """One attention block and one feed-forward, both residual, then a norm."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([Attention(), FeedForward()])])
        self.norm = RMSNorm(WIDTH)

    def forward(self, x):
        for attention, feed_forward in self.layers:
            x = attention(x) + x
            x = feed_forward(x) + x
        return self.norm(x)


class BandSplit(nn.Module):
    def __init__(self, widths: list):
        super().__init__()
        self.widths = list(widths)
        self.to_features = nn.ModuleList([nn.Sequential(RMSNorm(width), nn.Linear(width, WIDTH))
                                          for width in self.widths])

    def forward(self, x):
        parts = x.split(self.widths, dim=-1)
        return torch.stack([project(part) for part, project in zip(parts, self.to_features)], dim=-2)


class GLU(nn.Module):
    """The second half of the last axis, through a sigmoid, gates the first half."""

    def forward(self, x):
        value, gate = x.chunk(2, dim=-1)
        return value * gate.sigmoid()


class MaskEstimator(nn.Module):
    def __init__(self, widths: list):
        super().__init__()
        self.to_freqs = nn.ModuleList([
            nn.Sequential(
                nn.Sequential(nn.Linear(WIDTH, MASK_HIDDEN), nn.Tanh(), nn.Linear(MASK_HIDDEN, MASK_HIDDEN), nn.Tanh(),
                              nn.Linear(MASK_HIDDEN, 2 * width)),
                GLU())
            for width in widths])

    def forward(self, x):
        return torch.cat([estimate(band) for band, estimate in zip(x.unbind(dim=-2), self.to_freqs)], dim=-1)


class Network(nn.Module):
    """Stereo audio at 44.1 kHz in, the vocals of the same length out."""

    def __init__(self):
        super().__init__()
        widths = bands.band_widths()
        self.layers = nn.ModuleList([nn.ModuleList([Transformer(), Transformer()]) for _ in range(LAYERS)])
        self.band_split = BandSplit(widths)
        self.mask_estimators = nn.ModuleList([MaskEstimator(widths)])
        self.gather = None
        self.bands_per_bin = None

    def index_tables(self, device) -> None:
        """Build the bin tables on ``device``; they are not in the weights, so a network made on 'meta' needs this."""
        self.gather = torch.tensor(bands.gather_order(), dtype=torch.long, device=device)
        counts = torch.tensor(bands.bands_per_bin(), dtype=torch.float32, device=device)
        self.bands_per_bin = counts.repeat_interleave(bands.CHANNELS)[:, None]

    def forward(self, audio):
        batch, channels, length = audio.shape
        if channels != bands.CHANNELS:
            raise ValueError("the network takes stereo audio, not {} channels".format(channels))
        if self.gather is None or self.gather.device != audio.device:
            self.index_tables(audio.device)
        dtype = next(self.parameters()).dtype
        window = torch.hann_window(WINDOW, device=audio.device)
        spectrum = torch.stft(audio.float().reshape(batch * channels, length), bands.N_FFT, HOP, WINDOW,
                              window=window, return_complex=True)
        bins, frames = spectrum.shape[-2:]
        spectrum = spectrum.reshape(batch, channels, bins, frames).transpose(1, 2).reshape(batch, bins * channels, frames)
        halves = torch.view_as_real(spectrum[:, self.gather])
        x = halves.permute(0, 2, 1, 3).reshape(batch, frames, -1)
        x = self.band_split(x.to(dtype))
        count = x.shape[2]
        for across_time, across_bands in self.layers:
            x = x.transpose(1, 2).reshape(batch * count, frames, WIDTH)
            x = across_time(x)
            x = x.reshape(batch, count, frames, WIDTH).transpose(1, 2).reshape(batch * frames, count, WIDTH)
            x = across_bands(x)
            x = x.reshape(batch, frames, count, WIDTH)
        masks = self.mask_estimators[0](x).float()
        masks = torch.view_as_complex(masks.reshape(batch, frames, -1, 2).transpose(1, 2).contiguous())
        summed = torch.zeros_like(spectrum).scatter_add_(1, self.gather[None, :, None].expand(batch, -1, frames), masks)
        spectrum = spectrum * (summed / self.bands_per_bin.clamp(min=1e-8))
        spectrum = spectrum.reshape(batch, bins, channels, frames).transpose(1, 2).reshape(batch * channels, bins, frames)
        vocals = torch.istft(spectrum, bands.N_FFT, HOP, WINDOW, window=window, length=length)
        return vocals.reshape(batch, channels, length)
