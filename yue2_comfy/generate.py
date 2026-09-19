"""The four stages, in the order the protocol fixes them.

    style + lyrics  ->  ABC score  ->  semantic codec tokens
                                   ->  acoustic latents  ->  waveform

This is what YuE2Pipeline does, minus the parts that only make sense in a
command line tool: no process-wide torch flags (runtime.py scopes those), no
numpy round trip on the way out, and no VRAM-dependent tile size.

Where the weights sit does depend on the VRAM, stage by stage -- see
placement.py -- and that never changes a byte of what comes out.
"""

from __future__ import annotations

import dataclasses
import logging
import time

from . import phrasing, placement, runtime, transpose
from .constants import (
    AUTO_MIN_SECONDS, CONTEXT, FRAME_SECONDS, MAX_SECONDS, SAMPLE_RATE, length_ceiling,
    normalize_seed, seconds_to_tokens, sung_lines,
)

log = logging.getLogger(__name__)

FALLBACK_CORE_FRAMES = 512

LOW_VRAM_CORE_FRAMES = 256
LOW_VRAM_FALLBACK_CORE_FRAMES = 64
"""The tiles ``low_vram`` decodes in, and what it retries with.

Measured on 2026-09-17 over 5828 frames, a four-minute song's worth: 1.04 GiB of
work at 256 against 2.43 GiB at the released 1024, and no slower. The latents
were random, which is fair for both questions -- neither what a tile costs nor
how it rounds depends on their values. The tiling keeps every core's full
receptive field, so the waveform is the same one the whole decoder would write;
what differs is how the floating point rounds. The decode runs outside
deterministic_math, with TF32 convolutions, and there tile sizes from 128 frames
to the whole song land 64-68 dB apart -- measured on 2026-09-19, on a real
song's latents and on random ones alike. With TF32 off they are 116-120 dB
apart.

A retry has to ask for less than the try that failed, which is why this mode
brings its own: 64 frames took 0.28 GiB and 2.18 seconds against 1.27, where
the ordinary fallback of 512 would ask for more than 256 did."""


@dataclasses.dataclass(frozen=True)
class Performance:
    """What a song was sung from and what it became: enough to sing a part of it again.

    'prefix' is the prompt the performance was conditioned on, the score
    included, token for token, and 'negative' the prompt of the unconditional
    branch when guidance ran one, None when it did not. 'codec' is the
    performance itself, one codec token per frame of audio, and 'latents' the
    acoustic stage's answer to it, solved from noise drawn with 'seed'. None of
    this survives in a sound file, which is why songs.py keeps it beside the
    sound.
    """

    prefix: tuple
    negative: tuple | None
    codec: tuple
    seed: int
    latents: object


class Stages:
    """Where each stage sits on a 0..100 bar, and what it is called.

    The shares are not a measurement and cannot be one, because they move with
    the length of the song. Measured on a 5090 on 2026-09-13, two lines of
    lyrics and a 40-second ceiling: score 2.5 s, semantic 4.3 s, acoustic 1.9 s,
    decode 0.3 s -- the score stage a quarter of the run. The semantic stage is
    the one that grows with length, so at the 360-second ceiling it dominates
    and the score stage falls to a few percent. These numbers sit between the
    two cases; a bar that is honest at one length is wrong at the other.
    """

    ABC = (0.0, 10.0, "Writing the score")
    SEMANTIC = (10.0, 55.0, "Composing")
    ACOUSTIC = (55.0, 90.0, "Synthesizing audio")
    DECODE = (90.0, 100.0, "Decoding audio")


def _band(progress, stage, done, total, suffix="") -> None:
    """Report a fraction of one stage as a fraction of the whole bar.

    The table is in percent and the bar takes a fraction, so the value goes
    through ``ratio``: sent as it stands, the first percent past one filled a bar
    whose total is one, and it stood full for the whole of the singing.
    """
    if progress is None:
        return
    low, high, title = stage
    share = 0.0 if total <= 0 else max(0.0, min(1.0, float(done) / float(total)))
    progress.ratio((low + (high - low) * share) / 100.0, title + suffix)


def alone(*stages):
    """The same stages rescaled to fill a progress bar by themselves.

    A staged node runs one or two of the four and still owns a whole bar.
    Deriving its bands from the table above, instead of writing a second table
    beside it, keeps one set of proportions: editing Stages moves every bar.
    """
    low, high = stages[0][0], stages[-1][1]
    span = (high - low) or 1.0
    return tuple((100.0 * (a - low) / span, 100.0 * (b - low) / span, title)
                 for a, b, title in stages)


def _request(style, lyrics, seed, settings, abc=None):
    """The upstream request for these inputs, checked by its own rules.

    Empty ABC text is passed as None rather than as an empty string: upstream
    refuses the empty string, and the two mean the same thing here.
    """
    from .vendor.yue2.protocol import SongRequest

    cfg_scale = float(settings["cfg_scale"]) or None
    return SongRequest(style=style, lyrics=lyrics, cot=settings["cot"],
                       seed=normalize_seed(seed), cfg_scale=cfg_scale,
                       abc=(abc or "").strip() or None)


def _override(sampling, **fields):
    """A frozen Sampling with a few fields replaced and the rest left alone."""
    return dataclasses.replace(
        sampling, **{key: value for key, value in fields.items() if value is not None})


def _budget_message(prefix_tokens: int, max_tokens: int) -> str:
    """What to say when the lyrics and the requested length will not both fit."""
    room = max(0, CONTEXT - prefix_tokens)
    return (
        "The lyrics and the requested length do not both fit in the model's "
        "context of {:d} tokens.\n\n"
        "The prompt already takes {:d} tokens, which leaves room for about "
        "{:.0f} seconds of song, and {:.0f} seconds were asked for.\n\n"
        "Either lower 'max_seconds' to {:.0f} or below, or shorten the lyrics."
    ).format(CONTEXT, prefix_tokens, room * FRAME_SECONDS,
             max_tokens * FRAME_SECONDS, room * FRAME_SECONDS)


def write_score(models, style, lyrics, seed, settings, progress=None,
                cancelled=None, stages=None):
    """Stage one: the readable score, before a note of audio exists.

    Returns ``(text, ids, timing)``, and both forms of the score come back on
    purpose. The ids are what the next stage is really conditioned on, so a
    score nobody edited travels forward exactly as the model wrote it instead
    of being re-encoded from its own printed form.

    An empty score is the honest answer for cot='off', which goes straight from
    the lyrics to audio and never writes one.
    """
    from . import quantized, vocabulary
    from .vendor.yue2.protocol import GenerationConfig, token_prefixes
    from .vendor.yue2.sampling import generate_tokens

    band = (stages or (Stages.ABC,))[0]
    request = _request(style, lyrics, seed, settings)
    if request.cot == "off":
        return "", [], {}

    sampling = _override(GenerationConfig().abc,
                         temperature=float(settings["abc_temperature"]),
                         top_p=float(settings["abc_top_p"]),
                         top_k=int(settings["abc_top_k"]))
    with runtime.deterministic_math(), \
            runtime.pinned_attention(settings["attention_backend"]), \
            vocabulary.narrowed(models, placement.narrows(settings["offload"])), \
            quantized.captured(models):
        prefix = token_prefixes(request, models.tokenizer)

        def score(mode):
            placement.arrange(models, placement.AR, placement.ar_stage_bytes(
                len(prefix), sampling.max_tokens), "the score", mode)
            vocabulary.tune(models, vocabulary.ABC, prefix)
            return generate_tokens(
                models.lm, prefix, sampling, request.seed, "abc", cancelled=cancelled,
                on_token=_counter(progress, band, sampling.max_tokens))

        ids, spent, truncated = placement.guarded(models, "the score", score)
    if truncated:
        log.warning("[yue2_comfy.generate] the score hit its token budget")
    return models.tokenizer.decode(ids), list(ids), {"abc": spent}


def song_ceiling(max_seconds, lyrics: str, tune_seconds=None) -> float:
    """The most seconds a run sings: 'max_seconds', or at 0 the tune's ceiling, or the one the lyrics give.

    A score the lyrics were laid along (see ``phrasing``) knows how long the song
    is, so at 0 the ceiling is that length and a little over rather than twelve
    seconds a line: over a tune the words do not fit, the model runs on past the
    last bar in words of its own.
    """
    if float(max_seconds or 0) <= 0 and tune_seconds:
        return min(MAX_SECONDS, phrasing.ceiling(tune_seconds))
    return length_ceiling(max_seconds, lyrics)


def sing(models, style, lyrics, seed, settings, abc_ids=None, abc="",
         progress=None, cancelled=None, stages=None, tune_seconds=None):
    """Stages two and three: a score becomes acoustic latents.

    Returns ``(latents, timing, performance)``. The performance holds the same
    latents together with everything they were made from, which the latents
    alone cannot say; see ``Performance``.

    ``abc_ids`` wins when it is given, and ``abc`` is encoded when it is not,
    which is how an edited score re-enters the pipeline. Choosing between them
    is the caller's job, because only the caller knows whether the text in its
    hands is still the text those ids were decoded from.

    The two torch-backed modules are imported below the check rather than at
    the top, so a graph with nothing connected to its plan input is told what
    is missing instead of being told about torch.
    """
    from .vendor.yue2.protocol import (
        CODEC_OFFSET, GenerationConfig, negative_prefix, token_prefixes,
    )

    bands = stages or (Stages.SEMANTIC, Stages.ACOUSTIC)
    defaults = GenerationConfig()
    request = _request(style, lyrics, seed, settings)

    ids = None if abc_ids is None else [int(token) for token in abc_ids]
    if request.cot != "off" and ids is None:
        if not (abc or "").strip():
            raise ValueError(
                "There is no score to sing. Connect a 'YuE2 Plan' node to the "
                "'plan' input, or set 'cot' to 'off' in the options to go "
                "straight from the lyrics to audio."
            )
        ids = list(models.tokenizer.encode(abc))

    from . import quantized, vocabulary
    from .vendor.yue2 import nar
    from .vendor.yue2.sampling import generate_tokens

    automatic = float(settings["max_seconds"]) <= 0
    seconds = song_ceiling(settings["max_seconds"], lyrics, tune_seconds)
    if automatic and tune_seconds:
        log.info("[yue2_comfy.generate] length ceiling %.0f s, from the %.0f s tune the lyrics were laid along",
                 seconds, tune_seconds)
    elif automatic:
        log.info("[yue2_comfy.generate] length ceiling %.0f s, from %d sung lines",
                 seconds, sung_lines(lyrics))

    budget = seconds_to_tokens(seconds)
    sampling = _override(defaults.semantic,
                         temperature=float(settings["temperature"]),
                         top_p=float(settings["top_p"]),
                         top_k=int(settings["top_k"]),
                         repetition_penalty=float(settings["repetition_penalty"]),
                         min_tokens=min(defaults.semantic.min_tokens, budget),
                         max_tokens=budget)

    timing = {}
    with runtime.deterministic_math(), \
            runtime.pinned_attention(settings["attention_backend"]), \
            vocabulary.narrowed(models, placement.narrows(settings["offload"])), \
            quantized.captured(models):
        prefix = token_prefixes(request, models.tokenizer, ids)

        if len(prefix) + sampling.max_tokens > CONTEXT:
            room = max(0, CONTEXT - len(prefix))
            if not automatic or room < seconds_to_tokens(AUTO_MIN_SECONDS):
                raise ValueError(_budget_message(len(prefix), sampling.max_tokens))
            log.info("[yue2_comfy.generate] length ceiling cut to %.0f s by the prompt",
                     room * FRAME_SECONDS)
            sampling = dataclasses.replace(
                sampling, max_tokens=room,
                min_tokens=min(sampling.min_tokens, room))

        negative = None
        if request.guidance != 1:
            negative = negative_prefix(
                request, models.tokenizer, None if request.cot == "off" else ids)

        width = max(len(prefix), len(negative or ()))
        branches = 1 if request.guidance == 1 else 2

        def perform(mode):
            placement.arrange(models, placement.AR, placement.ar_stage_bytes(
                width, sampling.max_tokens, branches), "the performance", mode)
            vocabulary.tune(models, vocabulary.MUSIC, prefix, negative)
            return generate_tokens(
                models.lm, prefix, sampling, request.seed, "semantic",
                negative=negative, cfg_scale=request.guidance,
                legacy_off=request.cot == "off", cancelled=cancelled,
                on_token=_counter(progress, bands[0], sampling.max_tokens, seconds=True))

        semantic_ids, timing["semantic"], truncated = placement.guarded(
            models, "the performance", perform)
        if truncated:
            log.info("[yue2_comfy.generate] the song ran to the full length budget")

        codec = [int(token) - CODEC_OFFSET for token in semantic_ids]
        if not codec:
            raise ValueError(
                "The model ended the song before writing a single frame of audio. "
                "A different seed, or a style with more to go on, usually fixes it."
            )

        def synthesize(mode):
            with placement.acoustic(models, mode), runtime.fused_attention():
                return nar.synthesize(
                    models.lm, prefix, codec, request.seed, steps=int(settings["ode_steps"]),
                    context=CONTEXT, attention="sdpa", cancelled=cancelled,
                    on_progress=lambda done, total: _band(progress, bands[1], done, total))

        start = time.perf_counter()
        latents = placement.guarded(models, "the audio", synthesize)
        timing["acoustic"] = {"seconds": time.perf_counter() - start,
                              "frames": int(latents.shape[0])}
    performance = Performance(
        prefix=tuple(int(token) for token in prefix),
        negative=None if negative is None else tuple(int(token) for token in negative),
        codec=tuple(codec), seed=request.seed, latents=latents)
    return latents, timing, performance


def decode(models, latents, progress=None, cancelled=None, stages=None):
    """Stage four: latents to a waveform, in the layout ComfyUI's AUDIO wants."""
    band = (stages or (Stages.DECODE,))[0]
    timing = {}
    waveform = _decode(models, latents, progress, cancelled, timing, band)
    timing["seconds_of_audio"] = waveform.shape[-1] / SAMPLE_RATE
    return waveform, timing


def moved(score, semitones, cot):
    """The score moved by *semitones* before it is sung, or a ValueError saying why not.

    Both nodes that sing call this -- 'YuE2 Generate Song' between writing the
    score and singing it, 'YuE2 Render Plan' before it sings a plan -- so a move
    is the same move wherever it is asked for, and it is logged the same way.
    """
    if cot == "off":
        raise ValueError(transpose.COT_OFF)
    result = transpose.move(score, semitones)
    log.info("[yue2_comfy.generate] the score moved %s, from %s to %s",
             transpose.describe(semitones), result.before, result.after)
    return result.text


def run(models, style, lyrics, seed, settings, progress=None, cancelled=None,
        edited=None, tune_seconds=None):
    """One song. Returns (waveform, sung, written, timing, performance).

    The waveform is exactly what ComfyUI's AUDIO type wants: float32 [1, 2, S]
    on the CPU. decode_tiled already allocates that shape, so nothing here
    reshapes or copies the song again. The performance is what ``sing`` hands
    back beside its latents, passed on so the node can remember the song.

    The three calls below are the same three the staged nodes make one at a
    time. Keeping this node on the same path is the point: whatever the staged
    ones can do, this one has already done, and there is no second pipeline to
    keep in step.

    'sung' is the score the song was sung from, which is what the score_abc
    output hands on, and 'written' is the score the model wrote this run. With
    'transpose' set, the score is moved between the first call and the second
    and sung as text, so the two differ by the move. At 0 nothing changes, down
    to the ids stage one produced.

    With 'edited', a score a person changed, the first call is skipped: the edit
    is sung as text, exactly as 'YuE2 Render Plan' sings one, on a progress bar
    the three remaining stages fill by themselves. 'written' is then empty,
    because the model wrote nothing. 'tune_seconds' is how long an edit laid
    along its lyrics runs, which then sets the length ceiling at 'max_seconds' 0.
    """
    started = time.perf_counter()
    if edited:
        abc_text, abc_ids, timing, written = edited, None, {}, ""
        bands = alone(Stages.SEMANTIC, Stages.ACOUSTIC, Stages.DECODE)
    else:
        abc_text, abc_ids, timing = write_score(
            models, style, lyrics, seed, settings, progress, cancelled)
        written, bands = abc_text, None
    semitones = int(settings.get("transpose") or 0)
    if semitones:
        abc_text, abc_ids = moved(abc_text, semitones, settings["cot"]), None
    latents, spent, performance = sing(
        models, style, lyrics, seed, settings, abc_ids, abc=abc_text, progress=progress,
        cancelled=cancelled, stages=None if bands is None else bands[:2],
        tune_seconds=tune_seconds)
    timing.update(spent)
    waveform, spent = decode(models, latents, progress, cancelled,
                             stages=None if bands is None else bands[2:])
    timing.update(spent)
    timing["total_seconds"] = time.perf_counter() - started
    return waveform, abc_text, written, timing, performance


def _counter(progress, stage, total, seconds: bool = False):
    """An on_token callback that moves one band of the bar."""
    if progress is None:
        return None
    seen = 0

    def on_token(phase, token):
        nonlocal seen
        seen += 1
        suffix = " ({:.0f} s so far)".format(seen * FRAME_SECONDS) if seconds else ""
        _band(progress, stage, seen, total, suffix)

    return on_token


def _decode(models, latents, progress, cancelled, timing, band=None):
    """Latents to a waveform, with the VAE on the card only for as long as it takes.

    The backbone plays no part in this stage. Whatever of it would crowd the
    decoder is moved off first -- placement.py decides how much -- and left off:
    the next stage that needs a half brings it back, and a run that unloads the
    model straight afterwards never pays for the trip back at all.
    """
    import torch

    vae, device = models.vae, models.device
    low_vram = bool(getattr(models, "low_vram", False))
    z = latents.T.unsqueeze(0).contiguous()
    frames = int(z.shape[-1])
    moved = placement.arrange(models, None, placement.decode_bytes(low_vram),
                              "the decode")
    start = time.perf_counter()
    sizes = [LOW_VRAM_CORE_FRAMES if low_vram else int(vae.config.decode_core_frames)]
    smaller = LOW_VRAM_FALLBACK_CORE_FRAMES if low_vram else FALLBACK_CORE_FRAMES
    if smaller not in sizes:
        sizes.append(smaller)

    vae.to(device)
    try:
        for position, core in enumerate(sizes):
            try:
                audio = _decode_tiled(vae, z, core, progress, cancelled, band)
                break
            except torch.cuda.OutOfMemoryError:
                if position == len(sizes) - 1:
                    raise
                log.warning(
                    "[yue2_comfy.generate] out of memory decoding in %d-frame tiles, "
                    "retrying with %d. This run will not sound bit-identical to one "
                    "that decoded in a single tile size.", core, sizes[position + 1],
                )
                torch.cuda.empty_cache()
    finally:
        vae.to("cpu")
        _forget_computed_weights(vae)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not torch.isfinite(audio).all():
        raise ValueError("The decoder produced non-finite audio")
    timing["decode"] = {"seconds": time.perf_counter() - start, "frames": frames,
                        "core_frames": core, "moved": moved}
    return audio.clone().float().clamp_(-1, 1)


def _forget_computed_weights(vae) -> None:
    """Drop the weights weight_norm computed on the card, which ``vae.to`` does not move.

    The decoder's 44 convolutions use the old ``torch.nn.utils.weight_norm``,
    whose hook computes each weight from its two parameters before every call
    and keeps it as a plain attribute. ``Module.to`` moves parameters and
    buffers only, so after the decode those 0.25 GiB stayed on the card for as
    long as the model was kept loaded -- measured on 2026-09-18, together with
    another 0.2-0.5 GiB of the allocator's pool they pinned. The hook computes
    them again on the next call, so dropping them changes no sample.
    """
    from torch.nn.utils.weight_norm import WeightNorm

    for module in vae.modules():
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm) and hook.name in module.__dict__:
                delattr(module, hook.name)


def _decode_tiled(vae, z, core_frames, progress, cancelled, band=None):
    """One decode pass, with cancellation checked between tiles.

    decode_tiled takes no cancelled parameter, so the progress callback carries
    the check. Raising InterruptedError from it lands in the same place as a
    cancellation from any other stage.

    The same callback gives the allocator's cache back after every tile. The
    native allocator otherwise holds the blocks of every tile it has seen:
    8.07 GiB of the card for a decode that allocates 3.16, measured on
    2026-09-18 over 4500 frames on an RTX 5090, and 5.07 with the cache emptied
    per tile. On Windows that difference is not an error but a card spilling
    into system memory. The pool ComfyUI uses by default on CUDA 13 held 4.0-4.5
    either way; the tiles and the samples are the same, and it costs about a
    tenth of a second.
    """
    import torch

    stage = band or Stages.DECODE
    trim = next(vae.parameters()).device.type == "cuda"

    def report(done, total):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during decode")
        if trim:
            torch.cuda.empty_cache()
        _band(progress, stage, done, total)

    return vae.decode_tiled(z, core_frames=core_frames,
                            halo_frames=int(vae.config.decode_halo_frames),
                            output_device="cpu", on_progress=report)
