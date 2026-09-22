"""An edit on the model: one stretch of a remembered song sung again, or cut out.

Old frames ``start`` to ``stop`` of the song are replaced by new ones in four
stages. They run on the same wrappers the song was sung with -- ``placement``,
``runtime``, ``vocabulary``, ``quantized`` -- so 'offload' and 'low_vram' do here
what they do there.

1. The performance goes on from ``start``. The song's prompt and its codec
   tokens before ``start`` are the context, and the model writes new tokens with
   the edit's own seed, the end of the song barred. The repetition penalty
   starts from the tokens the song had sung just before, as it would have at
   that frame.
2. The join chooses how many of them to keep. For every count within the
   region's width of the length selected, the model scores how well the next
   three seconds of the old song follow, and the best count wins. An edit that
   reaches the end of the song has nothing after it to join, so there the model
   may end the song itself once it has sung the shortest length allowed.
3. The acoustic stage runs over the whole song with every old frame held on
   the line from its own noise to its own latent (RePaint), so that those
   frames end as the old latents to the bit and only the new frames and
   ``ops.MARGIN`` of the old ones on each side are drawn again.
4. A window around the edit is decoded and laid into the old sound, with a
   0.1-second crossfade at each end in a stretch where the old and the new
   latents are the same. Outside the window the new sound is the old sound,
   sample for sample, moved by what the edit added or took away.

A cut is stages 3 and 4 alone: nothing new is sung, and the two sides meet at
the seam, under the prompt with the cut's score and lyrics.

This is the inpainting stand of 2026-09-19 carried into the pack. On its three
songs by three seeds, retakes sang their words every time, the join landed
within two frames of the length the bars asked for, and a retake took 6.5-14 s
on an RTX 5090 against 34-43 s for the whole song -- the acoustic stage over
the whole song being most of it.
"""

from __future__ import annotations

import array
import contextlib
import dataclasses
import logging
import math
import sys
import time

import torch

from .. import generate, placement, quantized, runtime, songs, vocabulary
from ..constants import CONTEXT, FRAME_SECONDS, LATENT_DIM, normalize_seed
from . import ops

log = logging.getLogger(__name__)

ORIGIN = "YuE2 Edit Track"

PREFILL_AREA = 3072 * 3072
"""Queries times keys the prefill of a context may attend in one pass.

The AR half pays for attention in whole matrices (see ``placement``), and a
context here is the song's prompt and every frame before the edit: ten thousand
tokens for a retake near the end of a six-minute song, whose prefill in one
pass would ask for about 9 GiB of matrices. So a context longer than 3072
tokens is fed in pieces of this many queries against it, which caps the
matrices at about 0.9 GiB, less than a long prompt's own prefill takes when the
song is sung. The pieces depend on the length of the context alone, never on
the memory free, so an edit comes out the same on every card. A context up to
3072 tokens goes in one piece, exactly as upstream's prefill does."""

PIECE_FLOOR = 256
"""The fewest queries a piece of prefill holds, however long the context."""


class Stages:
    """Where each stage of an edit sits on a 0..100 bar, and what it is called.

    The acoustic stage runs over the whole song and is most of the time; the
    shares are the ones the stand measured on a 100-second song.
    """

    NATURAL = (0.0, 3.0, "Scoring the song's own join")
    PERFORM = (3.0, 35.0, "Singing the new part")
    JOIN = (35.0, 45.0, "Choosing the join")
    ACOUSTIC = (45.0, 92.0, "Drawing the audio around the edit")
    DECODE = (92.0, 100.0, "Decoding the new part")


def _part(band, index: int, count: int):
    """The ``index``-th of ``count`` equal slices of one band, titled with the take."""
    low, high, title = band
    step = (high - low) / max(1, count)
    if count > 1:
        title = "{} (take {} of {})".format(title, index + 1, count)
    return (low + step * index, low + step * (index + 1), title)


def _report(progress, band, done, total) -> None:
    """One band filled to ``done`` of ``total``."""
    if progress is None or band is None:
        return
    low, high, title = band
    share = 0.0 if total <= 0 else max(0.0, min(1.0, float(done) / float(total)))
    progress.ratio((low + (high - low) * share) / 100.0, title)


@dataclasses.dataclass
class Take:
    """One way an edit came out: its sound, the song that sound is, and how the join was chosen.

    ``count`` is how many new frames it sang in place of the old ones, 0 for a
    cut. ``join`` is the mean log-probability of the old frames after the edit
    given the new part, the score the join was chosen by: None when there was
    no choosing, for a cut or for an edit that reaches the end of the song.
    ``joins`` holds that score for every count the search tried. ``ended`` says
    that the model ended the song itself. ``natural`` is the same score for the
    old song at the same place, the one this take has to stand against; it is
    None unless it was asked for.
    """

    seed: int
    waveform: object
    song: songs.Song
    count: int
    join: float | None
    joins: dict
    ended: bool
    timing: dict
    natural: float | None = None


def best(takes) -> int:
    """The index of the take whose join scored best; the first take when none was scored."""
    scored = [(take.join, -index) for index, take in enumerate(takes) if take.join is not None]
    return -max(scored)[1] if scored else 0


def _settings(settings) -> dict:
    """DEFAULT_OPTIONS under the settings given, so a song remembered before a setting existed still edits."""
    from ..constants import DEFAULT_OPTIONS

    merged = dict(DEFAULT_OPTIONS)
    merged.update(settings or {})
    return merged


def _editable(song, waveform, region) -> torch.Tensor:
    """The song's audio as [channels, samples] float32, or a ValueError saying why it cannot be edited."""
    if not 0 <= region.start < region.stop <= song.frames:
        raise ValueError("Frames {} to {} are not a stretch of this song, which has {} frames."
                         .format(region.start, region.stop, song.frames))
    if song.settings.get("vocals_only"):
        raise ValueError(
            "This song was sung with 'vocals_only', so its audio is the voice alone while its "
            "latents are the whole song, and an edit would lay the band back into the voice. "
            "Edit the song as it came out of the singing node, then separate the voice again.")
    samples = waveform.detach()
    if samples.dim() == 3:
        samples = samples[0]
    samples = samples.to(device="cpu", dtype=torch.float32)
    if tuple(samples.shape) != (song.channels, song.samples):
        raise ValueError(
            "This audio is {} channels by {} samples, and the song it was recognised as is {} by {}."
            .format(samples.shape[0], samples.shape[-1], song.channels, song.samples))
    return samples


def latents_of(song) -> torch.Tensor:
    """A song's latents as a [frames, 64] float32 tensor on the CPU, from its little-endian bytes."""
    values = array.array("f")
    values.frombytes(song.latents)
    if sys.byteorder == "big":
        values.byteswap()
    return torch.frombuffer(values, dtype=torch.float32).clone().reshape(song.frames, LATENT_DIM)


def noise_of(runs) -> torch.Tensor:
    """The acoustic noise of every frame, [frames, 64], drawn the way ``nar.song_chunks`` draws it.

    Each seed is drawn once, as many rows as its runs reach, on the CPU in
    float32. Rows of that draw do not depend on how many are drawn: the
    generator fills normals sixteen at a time and a row is sixty-four wide, so
    the first rows of a long draw are the rows of a short one -- which is what
    lets a run name rows of the song's own draw without drawing the song.
    """
    reach = {}
    for seed, offset, count in runs:
        reach[int(seed)] = max(reach.get(int(seed), 0), int(offset) + int(count))
    draws = {}
    for seed, rows in reach.items():
        generator = torch.Generator(device="cpu").manual_seed(seed)
        draws[seed] = torch.randn((rows, LATENT_DIM), dtype=torch.float32, device="cpu",
                                  generator=generator)
    return torch.cat([draws[int(seed)][int(offset):int(offset) + int(count)]
                      for seed, offset, count in runs], dim=0)


def piece_for(tokens: int) -> int:
    """How many queries each piece of a prefill of ``tokens`` holds: all of them, in one piece, up to 3072."""
    return max(PIECE_FLOOR, PREFILL_AREA // max(1, int(tokens)))


@contextlib.contextmanager
def _singing(models, settings):
    """The wrappers the song's own AR and acoustic stages ran under, for the length of an edit."""
    with runtime.deterministic_math(), \
            runtime.pinned_attention(settings["attention_backend"]), \
            vocabulary.narrowed(models, placement.narrows(settings["offload"])), \
            quantized.captured(models):
        yield


@contextlib.contextmanager
def _remembering(history):
    """Upstream's sampler, told the tokens the song had sung before the edit.

    ``sampling.generate_tokens`` starts its repetition penalty from nothing,
    which is right at the start of a song. Going on from a frame in the middle,
    the song had the tokens before that frame in its penalty window, so they are
    put in front of the history the sampler hands its distribution. The function
    is looked up in its module on every step, so replacing it there for one call
    is enough, the way ``runtime.pinned_attention`` replaces GraphAR.
    """
    from ..vendor.yue2 import sampling

    original = sampling.distribution
    earlier = [int(token) for token in history]

    def distribution(logits, config, history, step, phase, legacy_off=False):
        return original(logits, config, earlier + list(history), step, phase, legacy_off)

    sampling.distribution = distribution
    try:
        yield
    finally:
        sampling.distribution = original


def _fill(lm, tokens, cache, piece: int, device, keep_last: bool = False):
    """Feed ``tokens`` into ``cache`` ``piece`` at a time; the last position's logits when asked for.

    A piece before the last asks for no logits at all -- an empty index -- so
    the head does not run for positions nobody reads, which with the vocabulary
    narrowed would be a multiply by the whole table on the processor.
    """
    nothing = torch.zeros(0, dtype=torch.long, device=device)
    result = None
    for begin in range(0, len(tokens), piece):
        last = begin + piece >= len(tokens)
        keep = 1 if last and keep_last else nothing
        result = lm(torch.tensor([list(tokens[begin:begin + piece])], dtype=torch.long,
                                 device=device),
                    past_key_values=cache, use_cache=True, logits_to_keep=keep)
    return result.logits[:, -1] if keep_last else None


@contextlib.contextmanager
def _in_pieces(piece: int):
    """GraphAR with a prefill that goes ``piece`` tokens at a time, for the length of one call.

    When every branch fits in one piece, upstream's own prefill runs untouched.
    Entered inside the other wrappers, so the class it extends is the one they
    made and their parts of the graph -- the windows, the attention backend, the
    packed layers -- stay as they are.
    """
    from ..vendor.yue2 import cuda_graph

    original = cuda_graph.GraphAR

    class PiecewiseGraphAR(original):
        @torch.inference_mode()
        def prefill(self):
            if self.closed or self.ready or max(map(len, self.prefixes)) <= piece:
                return super().prefill()
            logits = []
            for branch, prefix in enumerate(self.prefixes):
                cache = cuda_graph._PrefixCache(self.keys, self.values, branch)
                logits.append(_fill(self.model, prefix, cache, piece, self.device, keep_last=True))
            result = torch.cat(logits, dim=0)
            if self.capture and self.max_tokens > 1:
                self._capture()
            self.ready = True
            return result

    cuda_graph.GraphAR = PiecewiseGraphAR
    try:
        yield
    finally:
        cuda_graph.GraphAR = original


def _sampling(settings, earliest: int, count: int):
    """The song's own sampling, for ``count`` tokens with the end barred before ``earliest``."""
    from ..vendor.yue2.protocol import GenerationConfig

    return dataclasses.replace(
        GenerationConfig().semantic, temperature=float(settings["temperature"]),
        top_p=float(settings["top_p"]), top_k=int(settings["top_k"]),
        repetition_penalty=float(settings["repetition_penalty"]),
        min_tokens=int(earliest), max_tokens=int(count))


def guidance(settings) -> float:
    """The CFG scale the song was sung with, by upstream's own rule for what an unset scale means."""
    from ..vendor.yue2.protocol import SongRequest

    scale = float(settings.get("cfg_scale") or 0) or None
    return SongRequest(style="", lyrics="", cot=settings["cot"], cfg_scale=scale).guidance


def _ar_bytes(width: int, count: int, branches: int, piece: int) -> int:
    """What singing on from a context of ``width`` tokens allocates beyond the weights."""
    return (placement.kv_bytes(width + count, branches)
            + placement.attention_bytes(min(piece, width), width) + placement.WORK_BASE)


def perform(models, context, negative, count: int, earliest: int, seed: int, history, settings,
            progress=None, band=None, cancelled=None):
    """``count`` new codec tokens after ``context``, as vocabulary ids; and whether the model ended the song.

    Sampled with the song's own settings by upstream's sampler, on the graph
    the song was sung on. ``negative`` is the unconditional branch's context when
    the song ran CFG. The end of the song is barred before ``earliest`` tokens.
    """
    from ..vendor.yue2 import sampling

    config = _sampling(settings, earliest, count)
    scale = guidance(settings)
    branches = 1 if scale == 1 else 2
    if branches == 2 and negative is None:
        raise ValueError("This song was sung with CFG, and the memory of it has no negative prompt.")
    width = max(len(context), len(negative or ()))
    piece = piece_for(width)
    done = [0]

    def on_token(_phase, _token):
        done[0] += 1
        _report(progress, band, done[0], count)

    def sing(mode):
        done[0] = 0
        placement.arrange(models, placement.AR, _ar_bytes(width, count, branches, piece),
                          "the edit", mode)
        vocabulary.tune(models, vocabulary.MUSIC, context, negative if branches == 2 else None)
        with _remembering(history), _in_pieces(piece):
            return sampling.generate_tokens(
                models.lm, list(context), config, normalize_seed(seed), "semantic",
                negative=list(negative) if branches == 2 else None, cfg_scale=scale,
                legacy_off=settings["cot"] == "off", cancelled=cancelled, on_token=on_token)

    tokens, timing, truncated = placement.guarded(models, "the edit", sing)
    return [int(token) for token in tokens], timing, not truncated


def _join_bytes(tokens: int, capacity: int, piece: int, queries: int) -> int:
    """What scoring the joins allocates beyond the weights: the cache, the attention, the logits."""
    from ..vendor.yue2.protocol import CODEC_SIZE

    return (placement.kv_bytes(capacity)
            + max(placement.attention_bytes(min(piece, tokens), tokens),
                  placement.attention_bytes(queries, capacity))
            + 4 * queries * CODEC_SIZE * placement.ELEMENT_BYTES + placement.WORK_BASE)


class _Rows(torch.nn.Module):
    """A head that answers for a few rows of the vocabulary: one multiply by those rows alone."""

    def __init__(self, rows):
        super().__init__()
        self.held = [rows]

    def forward(self, hidden):
        return torch.nn.functional.linear(hidden, self.held[0])


@contextlib.contextmanager
def _codec_head(lm):
    """The head answering for the codec rows alone, by the same multiply wherever the table is kept.

    The join reads codec logits and nothing else. Measured on 2026-09-19 on the
    ballad of the stand, the whole head of 'off' and the window ``vocabulary``
    keeps on the card under 'auto' gave the same tokens but join scores 0.005
    apart: a matrix of another width is multiplied by another kernel, which
    rounds another way once there are a few hundred positions. Multiplying by
    the 32768 codec rows in every mode -- a view of the head's own weight, or of
    the window -- makes the join the same whichever way 'offload' is set.
    """
    from ..vendor.yue2.protocol import CODEC_OFFSET, CODEC_SIZE

    head = lm.lm_head
    if isinstance(head, vocabulary.Sliced):
        low = CODEC_OFFSET - head.start
        rows = head.rows[low:low + CODEC_SIZE]
    else:
        rows = head.weight[CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE]
    if tuple(rows.shape[:1]) != (CODEC_SIZE,) or getattr(head, "bias", None) is not None:
        raise ValueError("This model's head does not hold the codec rows where YuE2's does.")
    lm.lm_head = _Rows(rows)
    try:
        yield
    finally:
        lm.lm_head = head


def joins(models, context, tokens, lowest: int, highest: int, suffix, progress=None, band=None,
          cancelled=None) -> dict:
    """``{count: score}``: how well the old ``suffix`` follows the first ``count`` new tokens.

    The score is the mean log-probability of the suffix's tokens under the
    model, among codec tokens only, for every count from ``lowest`` to
    ``highest``. The context and the first ``lowest`` new tokens go into the
    cache once; each count then runs the tokens after them and the suffix, and
    rewinds the cache to where it was.
    """
    from ..vendor.yue2.modeling_yue2 import StaticKVCache
    from ..vendor.yue2.protocol import CODEC_OFFSET

    lm = models.lm
    config = lm.config
    base = list(context) + list(tokens[:lowest])
    size = len(suffix)
    queries = 1 + (highest - lowest) + size
    capacity = len(base) + queries + 1
    piece = piece_for(len(base))

    @torch.inference_mode()
    def score(mode):
        placement.arrange(models, placement.AR, _join_bytes(len(base), capacity, piece, queries),
                          "the join", mode)
        vocabulary.tune(models, vocabulary.MUSIC, base)
        weight = next(lm.parameters())
        device, dtype = weight.device, weight.dtype
        cache = StaticKVCache(config.num_hidden_layers, 1, config.num_key_value_heads, capacity,
                              config.head_dim, dtype, device)
        _fill(lm, base[:-1], cache, piece, device)
        start = cache.get_seq_length()
        target = torch.tensor([int(token) - CODEC_OFFSET for token in suffix], device=device)
        found = {}
        with vocabulary.windowed(models), _codec_head(lm):
            for count in range(lowest, highest + 1):
                if cancelled is not None and cancelled():
                    raise InterruptedError("Cancelled while choosing the join")
                cache._seen_tokens = start
                chunk = [base[-1]] + list(tokens[lowest:count]) + list(suffix)
                logits = lm(torch.tensor([chunk], device=device), past_key_values=cache,
                            use_cache=True, logits_to_keep=0).logits[0]
                span = logits[count - lowest:count - lowest + size].float()
                found[count] = float(torch.log_softmax(span, -1).gather(1, target[:, None])[:, 0].mean())
                del logits, span
                _report(progress, band, count - lowest + 1, highest - lowest + 1)
        return found

    return placement.guarded(models, "the join", score)


def _logit(t: float) -> float:
    """Upstream's time argument: the logit of ``t`` in float64 on the CPU, clamped as ``CachedNAR.solve`` does."""
    return torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()


@torch.inference_mode()
def _solved(engine, steps: int, noise, known, held, cancelled=None, on_step=None):
    """``CachedNAR.solve`` with the ``held`` frames put back on their line after every half step.

    The line of a held frame runs from its noise at t = 1 to its known latent at
    t = 0, so the frame is where the song's own solve would have had it if the
    solver had gone straight, and it ends as the known latent exactly. The rest
    of the arithmetic is upstream's, operation for operation: with nothing held
    this returns what ``solve`` returns, to the bit.
    """
    device, dtype = engine.device, engine.dtype
    eps = noise.to(device=device, dtype=dtype)
    z0 = known.to(device=device, dtype=dtype)
    keep = held.to(device)[:, None]
    state = eps.clone()
    dt = 1.0 / steps
    for step in range(steps):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during acoustic flow matching")
        t = 1.0 - step * dt
        first = engine.velocity(state, _logit(t))
        middle = t - dt / 2
        mid = state - first * (dt / 2)
        mid = torch.where(keep, (middle * eps + (1 - middle) * z0).to(dtype), mid)
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during acoustic flow matching")
        state = state - engine.velocity(mid, _logit(middle)) * dt
        after = t - dt
        state = torch.where(keep, (after * eps + (1 - after) * z0).to(dtype), state)
        if on_step is not None:
            on_step(step + 1, steps)
    state = torch.where(keep, z0, state)
    result = state.float().cpu()
    if not torch.isfinite(result).all():
        raise FloatingPointError("Acoustic flow matching produced non-finite latents")
    return result


def repaint(models, prefix, codec, noise, known, held, steps: int, progress=None, band=None,
            cancelled=None) -> torch.Tensor:
    """Latents for ``codec`` under ``prefix``, with the ``held`` frames kept on ``known``.

    The song is cut into upstream's acoustic chunks as the song itself was
    (``protocol.chunk_ranges``). A chunk with every frame held is the known
    latents already and is not run at all; a song this pack sings fits one
    chunk unless its prompt is very long. An edit whose redrawn frames reach
    across two chunks is refused: each chunk would draw its side of the seam
    without hearing the other.

    Each chunk runs the way ``nar.synthesize`` runs one -- its prefill with the
    AR half, its flow matching with the NAR half -- inside
    ``placement.acoustic``, which is what moves the halves.
    """
    from ..vendor.yue2 import nar
    from ..vendor.yue2.protocol import CODEC_OFFSET, MUSIC_END, chunk_ranges

    frames = len(codec)
    ranges = chunk_ranges(frames, len(prefix), CONTEXT)
    loose = [(low, high) for low, high in ranges if not bool(held[low:high].all())]
    if len(loose) > 1:
        raise ValueError(
            "This edit falls across the boundary between two parts the acoustic stage draws "
            "one at a time, at {:.1f} s. Edit up to that point, or from it."
            .format(loose[0][1] * FRAME_SECONDS))
    result = known.clone()
    total = len(loose) * steps
    for position, (low, high) in enumerate(loose):
        chunk = nar.Chunk(list(prefix) + [int(value) + CODEC_OFFSET for value in codec[low:high]]
                          + [MUSIC_END], noise[low:high].contiguous())

        def on_step(done, _steps, position=position):
            _report(progress, band, position * steps + done, total)

        def solve(mode, chunk=chunk, low=low, high=high, on_step=on_step):
            with placement.acoustic(models, mode), runtime.fused_attention():
                engine = nar.CachedNAR(models.lm, chunk, "sdpa", None)
                try:
                    with nar._offload_ar(models.lm, False):
                        return _solved(engine, int(steps), noise[low:high], known[low:high],
                                       held[low:high], cancelled, on_step)
                finally:
                    engine.close()

        result[low:high] = placement.guarded(models, "the edit's audio", solve)
    return result


def _window(frames: int, changed, halo: int, hop: int, samples: int):
    """Where the new sound is laid in: ``(first frame, end frame, left fade, right fade)``.

    The decoder reaches ``halo`` frames either way, so the sound differs from the
    old only within ``halo`` frames of the latents that changed. Each fade is
    the ``ops.FADE_SAMPLES`` just outside that, as ``(first sample, end sample)``,
    or None where the window runs into the start or the end of the song; the
    window is the frames that cover both fades.
    """
    low, high = changed
    fade = ops.FADE_SAMPLES
    left = (low - halo) * hop
    left = (left - fade, left) if left - fade >= 0 else None
    right = (high + halo) * hop
    right = (right, right + fade) if right + fade <= samples else None
    first = 0 if left is None else left[0] // hop
    end = frames if right is None else min(frames, -(-right[1] // hop))
    return first, end, left, right


def splice(models, old, latents, region, count: int, progress=None, band=None, cancelled=None):
    """The edited song's sound: the old sound, with a decoded window of the new one laid in.

    ``old`` is the song's sound before the edit, [channels, samples]. ``latents``
    are the whole edited song's, and ``region`` and ``count`` say which old
    frames went and how many new ones came. Returns [1, channels, samples]
    float32 on the CPU, and the decode's timing.
    """
    vae = models.vae
    hop = int(vae.config.downsampling_ratio)
    halo = int(vae.config.decode_halo_frames)
    frames = int(latents.shape[0])
    shift = (region.removed - count) * hop
    samples = old.shape[-1] - shift
    changed = (max(0, region.start - ops.MARGIN), min(frames, region.start + count + ops.MARGIN))
    first, end, left, right = _window(frames, changed, halo, hop, samples)
    low, high = max(0, first - halo), min(frames, end + halo)
    decoded, timing = generate.decode(models, latents[low:high], progress, cancelled,
                                      stages=None if band is None else (band,))
    offset = (first - low) * hop
    begin = first * hop
    stop = samples if end == frames else end * hop
    window = decoded[0, :, offset:offset + stop - begin]
    if window.shape[-1] != stop - begin:
        raise RuntimeError("The decoded window is {} samples short".format(stop - begin - window.shape[-1]))

    result = torch.empty((old.shape[0], samples), dtype=torch.float32)
    inner_start = 0 if left is None else left[1]
    inner_stop = samples if right is None else right[0]
    result[:, inner_start:inner_stop] = window[:, inner_start - begin:inner_stop - begin]
    ramp = 0.5 - 0.5 * torch.cos(math.pi * (torch.arange(ops.FADE_SAMPLES, dtype=torch.float64) + 0.5)
                                 / ops.FADE_SAMPLES)
    ramp = ramp.to(torch.float32)
    if left is not None:
        a, b = left
        result[:, :a] = old[:, :a]
        result[:, a:b] = old[:, a:b] * (1 - ramp) + window[:, a - begin:b - begin] * ramp
    if right is not None:
        a, b = right
        result[:, a:b] = window[:, a - begin:b - begin] * (1 - ramp) + old[:, a + shift:b + shift] * ramp
        result[:, b:] = old[:, b + shift:]
    return result.unsqueeze(0), timing


def _held(song, region, count: int):
    """The known latents and the held frames of the edited song, frames x 64 and frames."""
    old = latents_of(song)
    frames = song.frames - region.removed + count
    known = torch.zeros((frames, LATENT_DIM), dtype=torch.float32)
    held = torch.zeros(frames, dtype=torch.bool)
    known[:region.start] = old[:region.start]
    known[region.start + count:] = old[region.stop:]
    held[:region.start] = True
    held[region.start + count:] = True
    held[max(0, region.start - ops.MARGIN):min(frames, region.start + count + ops.MARGIN)] = False
    return known, held


def _prompts(models, song, lyrics, score, settings):
    """The prompt and the negative prompt of the song with other lyrics and another score.

    The unconditional branch is built whenever this run needs one, which is
    not only when the song had one: an edit may be sung under a guide of its
    own, see ``track.GUIDE``.
    """
    from ..vendor.yue2.protocol import SongRequest, negative_prefix, token_prefixes

    scale = float(settings.get("cfg_scale") or 0) or None
    request = SongRequest(style=song.style, lyrics=lyrics, cot=settings["cot"],
                          seed=normalize_seed(song.seed), cfg_scale=scale)
    ids = None if request.cot == "off" else list(models.tokenizer.encode(score))
    prefix = token_prefixes(request, models.tokenizer, ids)
    negative = None
    if song.negative is not None or request.guidance != 1:
        negative = negative_prefix(request, models.tokenizer, ids)
    return prefix, negative


def _made(song, waveform, prefix, negative, codec, noise, latents, lyrics=None, score=None):
    """The edited song, for the song memory: the old one with what the edit changed."""
    return songs.made(ORIGIN, song.style, song.lyrics if lyrics is None else lyrics, song.seed,
                      song.settings, song.score if score is None else score, prefix, negative,
                      codec, noise, latents, song.sample_rate, waveform.shape[-2],
                      waveform.shape[-1])


def retakes(models, song, waveform, region, seeds, settings, progress=None, cancelled=None,
            noise_seeds=None, natural: bool = False, lyrics=None, score=None) -> list:
    """The stretch ``region`` of ``song`` sung again, once for every seed: one Take each, in seed order.

    ``waveform`` is the song's sound, which the new part is laid into.
    ``settings`` are the ones to sing with -- the song's own sampling, and the
    run's own device and offload. The new frames' noise comes from each take's
    seed, or from ``noise_seeds`` when given.

    Every take is sung and joined first, while the AR half is on the card, and
    then drawn and decoded, so a card that holds one half at a time swaps twice
    rather than twice a take.

    With ``natural`` the old song's own join is scored at the same place too,
    one pass before the singing, and every take carries it. A join score means
    nothing on its own -- it is the model's opinion of the words that follow,
    which differs from song to song and place to place -- so the number to
    compare a take with is what the song itself scored there. That pass is
    made under the song's own prompt whatever this edit sings, because the
    song as it was is what a take is held against.

    ``lyrics`` and ``score`` change the words this stretch is sung to: given
    either, the prompt is built again from them and the takes are sung, drawn
    and remembered under it, which is what a change of words is. The frames
    before the edit are the song's own either way -- they are what the model
    hears itself having sung.
    """
    from ..vendor.yue2.protocol import CODEC_OFFSET

    settings = _settings(settings)
    old = _editable(song, waveform, region)
    seeds = [normalize_seed(seed) for seed in seeds]
    noise_seeds = seeds if noise_seeds is None else [normalize_seed(seed) for seed in noise_seeds]
    ids = [int(value) + CODEC_OFFSET for value in song.codec]
    prefix, unconditional = (song.prefix, song.negative) if lyrics is None and score is None         else _prompts(models, song, song.lyrics if lyrics is None else lyrics,
                      song.score if score is None else score, settings)
    context = list(prefix) + ids[:region.start]
    negative = None if unconditional is None else list(unconditional) + ids[:region.start]
    window = _sampling(settings, 1, 1).penalty_window
    history = ids[max(0, region.start - window):region.start]
    at_end = region.stop >= song.frames
    suffix = ids[region.stop:region.stop + ops.JOIN_FRAMES]
    count = region.longest
    earliest = region.shortest if at_end else count

    sung = []
    with _singing(models, settings):
        own = None
        if natural and not at_end and suffix:
            heard = list(song.prefix) + ids[:region.start]
            own = joins(models, heard, ids[region.start:region.stop], region.length,
                        region.length, suffix, progress, Stages.NATURAL, cancelled)[region.length]
        for index, seed in enumerate(seeds):
            timing = {}
            began = time.perf_counter()
            tokens, _spent, ended = perform(
                models, context, negative, count, earliest, seed, history, settings, progress,
                _part(Stages.PERFORM, index, len(seeds)), cancelled)
            timing["perform"] = time.perf_counter() - began
            scores = {}
            if ended:
                chosen = len(tokens)
            elif at_end or not suffix:
                chosen = region.length
            else:
                began = time.perf_counter()
                scores = joins(models, context, tokens, region.shortest, region.longest, suffix,
                               progress, _part(Stages.JOIN, index, len(seeds)), cancelled)
                timing["join"] = time.perf_counter() - began
                chosen = max(scores, key=scores.get)
            sung.append((seed, tokens, chosen, scores, ended, timing))

        drawn = []
        for index, (seed, tokens, chosen, scores, ended, timing) in enumerate(sung):
            codec = list(song.codec[:region.start]) + [token - CODEC_OFFSET for token in tokens[:chosen]] \
                + list(song.codec[region.stop:])
            runs = ops.edited_noise(song.noise, region, noise_seeds[index], chosen)
            known, held = _held(song, region, chosen)
            began = time.perf_counter()
            latents = repaint(models, prefix, codec, noise_of(runs), known, held,
                              int(settings["ode_steps"]), progress,
                              _part(Stages.ACOUSTIC, index, len(seeds)), cancelled)
            timing["acoustic"] = time.perf_counter() - began
            drawn.append((codec, runs, latents))

    takes = []
    for index, ((seed, tokens, chosen, scores, ended, timing), (codec, runs, latents)) in \
            enumerate(zip(sung, drawn)):
        began = time.perf_counter()
        sound, _spent = splice(models, old, latents, region, chosen, progress,
                               _part(Stages.DECODE, index, len(seeds)), cancelled)
        timing["decode"] = time.perf_counter() - began
        edited = _made(song, sound, prefix, unconditional, codec, runs, latents, lyrics, score)
        log.info("[yue2_comfy.inpaint] take %d of %d, seed %d: frames %d-%d sung again as %d "
                 "for %d, join %s, in %.1f s", index + 1, len(sung), seed, region.start,
                 region.stop, chosen, region.length,
                 "the model's own end" if ended else "none" if not scores
                 else "{:.3f}".format(scores[chosen]), sum(timing.values()))
        takes.append(Take(seed=seed, waveform=sound, song=edited, count=chosen,
                          join=scores.get(chosen), joins=scores, ended=ended, timing=timing,
                          natural=own))
    return takes


def cut(models, song, waveform, region, lyrics, score, settings, progress=None,
        cancelled=None) -> Take:
    """The stretch ``region`` of ``song`` taken out: the two sides joined, under the cut's lyrics and score.

    ``lyrics`` and ``score`` are the song's own with the cut applied (see
    ``ops.cut_words`` and ``notation.without``); they make the prompt the seam
    is drawn under and the one the edited song is remembered with.
    """
    settings = _settings(settings)
    old = _editable(song, waveform, region)
    prefix, negative = _prompts(models, song, lyrics, score, settings)
    codec = list(song.codec[:region.start]) + list(song.codec[region.stop:])
    runs = ops.edited_noise(song.noise, region, 0, 0)
    known, held = _held(song, region, 0)
    timing = {}
    with _singing(models, settings):
        began = time.perf_counter()
        latents = repaint(models, prefix, codec, noise_of(runs), known, held,
                          int(settings["ode_steps"]), progress, Stages.ACOUSTIC, cancelled)
        timing["acoustic"] = time.perf_counter() - began
    began = time.perf_counter()
    sound, _spent = splice(models, old, latents, region, 0, progress, Stages.DECODE, cancelled)
    timing["decode"] = time.perf_counter() - began
    edited = _made(song, sound, prefix, negative, codec, runs, latents, lyrics, score)
    log.info("[yue2_comfy.inpaint] frames %d-%d cut, %.1f s shorter, in %.1f s", region.start,
             region.stop, region.removed * FRAME_SECONDS, sum(timing.values()))
    return Take(seed=0, waveform=sound, song=edited, count=0, join=None, joins={}, ended=False,
                timing=timing)
