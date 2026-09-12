"""The four stages, in the order the protocol fixes them.

    style + lyrics  ->  ABC score  ->  semantic codec tokens
                                   ->  acoustic latents  ->  waveform

This is what YuE2Pipeline does, minus the parts that only make sense in a
command line tool: no process-wide torch flags (runtime.py scopes those), no
numpy round trip on the way out, and no VRAM-dependent tile size.
"""

from __future__ import annotations

import dataclasses
import logging
import time

from . import runtime
from .constants import (
    AUTO_MIN_SECONDS, CONTEXT, FRAME_SECONDS, SAMPLE_RATE, auto_seconds,
    normalize_seed, seconds_to_tokens, sung_lines,
)

log = logging.getLogger(__name__)

FALLBACK_CORE_FRAMES = 512

DECODE_RESERVE_BYTES = 512 * 1024 ** 2 + 2 * 1024 ** 3


class Stages:
    """Where each stage sits on a 0..100 bar, and what it is called."""

    ABC = (0.0, 10.0, "Writing the score")
    SEMANTIC = (10.0, 55.0, "Composing")
    ACOUSTIC = (55.0, 90.0, "Synthesizing audio")
    DECODE = (90.0, 100.0, "Decoding audio")


def _band(progress, stage, done, total, suffix="") -> None:
    """Report a fraction of one stage as a fraction of the whole bar."""
    if progress is None:
        return
    low, high, title = stage
    share = 0.0 if total <= 0 else max(0.0, min(1.0, float(done) / float(total)))
    progress.update(low + (high - low) * share, title + suffix)


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


def run(models, style, lyrics, seed, settings, progress=None, cancelled=None):
    """One song. Returns (waveform, abc_text, timing).

    The waveform is exactly what ComfyUI's AUDIO type wants: float32 [1, 2, S]
    on the CPU. decode_tiled already allocates that shape, so nothing here
    transposes or copies the song again.
    """
    from .vendor.yue2 import nar
    from .vendor.yue2.protocol import (
        CODEC_OFFSET, GenerationConfig, SongRequest, negative_prefix, token_prefixes,
    )
    from .vendor.yue2.sampling import generate_tokens

    defaults = GenerationConfig()
    lm, vae, tokenizer, device = models.lm, models.vae, models.tokenizer, models.device

    cfg_scale = float(settings["cfg_scale"]) or None
    request = SongRequest(style=style, lyrics=lyrics, cot=settings["cot"],
                          seed=normalize_seed(seed), cfg_scale=cfg_scale)

    abc_sampling = _override(defaults.abc,
                             temperature=float(settings["abc_temperature"]),
                             top_p=float(settings["abc_top_p"]),
                             top_k=int(settings["abc_top_k"]))
    requested = float(settings["max_seconds"])
    automatic = requested <= 0
    seconds = auto_seconds(lyrics) if automatic else requested
    if automatic:
        log.info("[yue2_comfy.generate] length ceiling %.0f s, from %d sung lines",
                 seconds, sung_lines(lyrics))

    budget = seconds_to_tokens(seconds)
    semantic_sampling = _override(defaults.semantic,
                                  temperature=float(settings["temperature"]),
                                  top_p=float(settings["top_p"]),
                                  top_k=int(settings["top_k"]),
                                  repetition_penalty=float(settings["repetition_penalty"]),
                                  min_tokens=min(defaults.semantic.min_tokens, budget),
                                  max_tokens=budget)

    timing = {}
    started = time.perf_counter()

    with runtime.deterministic_math(), runtime.pinned_attention(settings["attention_backend"]):
        prefix = token_prefixes(request, tokenizer)
        abc_ids, abc_text = [], ""
        if request.cot != "off":
            abc_ids, timing["abc"], truncated = generate_tokens(
                lm, prefix, abc_sampling, request.seed, "abc", cancelled=cancelled,
                on_token=_counter(progress, Stages.ABC, abc_sampling.max_tokens))
            abc_text = tokenizer.decode(abc_ids)
            if truncated:
                log.warning("[yue2_comfy.generate] the score hit its token budget")
            prefix = token_prefixes(request, tokenizer, abc_ids)

        if len(prefix) + semantic_sampling.max_tokens > CONTEXT:
            room = max(0, CONTEXT - len(prefix))
            if not automatic or room < seconds_to_tokens(AUTO_MIN_SECONDS):
                raise ValueError(_budget_message(len(prefix), semantic_sampling.max_tokens))
            log.info("[yue2_comfy.generate] length ceiling cut to %.0f s by the prompt",
                     room * FRAME_SECONDS)
            semantic_sampling = dataclasses.replace(
                semantic_sampling, max_tokens=room,
                min_tokens=min(semantic_sampling.min_tokens, room))

        negative = None
        if request.guidance != 1:
            negative = negative_prefix(
                request, tokenizer, None if request.cot == "off" else abc_ids)

        semantic_ids, timing["semantic"], truncated = generate_tokens(
            lm, prefix, semantic_sampling, request.seed, "semantic",
            negative=negative, cfg_scale=request.guidance,
            legacy_off=request.cot == "off", cancelled=cancelled,
            on_token=_counter(progress, Stages.SEMANTIC, semantic_sampling.max_tokens,
                              seconds=True))
        if truncated:
            log.info("[yue2_comfy.generate] the song ran to the full length budget")

        codec = [int(token) - CODEC_OFFSET for token in semantic_ids]
        if not codec:
            raise ValueError(
                "The model ended the song before writing a single frame of audio. "
                "A different seed, or a style with more to go on, usually fixes it."
            )

        start = time.perf_counter()
        latents = nar.synthesize(
            lm, prefix, codec, request.seed, steps=int(settings["ode_steps"]),
            context=CONTEXT, attention="sdpa", cancelled=cancelled,
            on_progress=lambda done, total: _band(progress, Stages.ACOUSTIC, done, total))
        timing["acoustic"] = {"seconds": time.perf_counter() - start,
                              "frames": int(latents.shape[0])}

    waveform = _decode(vae, lm, latents, device, progress, cancelled, timing)
    timing["total_seconds"] = time.perf_counter() - started
    timing["seconds_of_audio"] = waveform.shape[-1] / SAMPLE_RATE
    return waveform, abc_text, timing


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


def _decode(vae, lm, latents, device, progress, cancelled, timing):
    """Latents to a waveform, with the VAE on the card only for as long as it takes."""
    import torch

    z = latents.T.unsqueeze(0).contiguous()
    frames = int(z.shape[-1])
    moved_lm = _make_room(lm, device)
    start = time.perf_counter()
    sizes = [int(vae.config.decode_core_frames)]
    if FALLBACK_CORE_FRAMES not in sizes:
        sizes.append(FALLBACK_CORE_FRAMES)

    vae.to(device)
    try:
        for position, core in enumerate(sizes):
            try:
                audio = _decode_tiled(vae, z, core, progress, cancelled)
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
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if moved_lm:
            lm.to(device)

    if not torch.isfinite(audio).all():
        raise ValueError("The decoder produced non-finite audio")
    timing["decode"] = {"seconds": time.perf_counter() - start, "frames": frames,
                        "core_frames": core, "offloaded_lm": moved_lm}
    return audio.clone().float().clamp_(-1, 1)


def _decode_tiled(vae, z, core_frames, progress, cancelled):
    """One decode pass, with cancellation checked between tiles.

    decode_tiled takes no cancelled parameter, so the progress callback carries
    the check. Raising InterruptedError from it lands in the same place as a
    cancellation from any other stage.
    """
    def report(done, total):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during decode")
        _band(progress, Stages.DECODE, done, total)

    return vae.decode_tiled(z, core_frames=core_frames,
                            halo_frames=int(vae.config.decode_halo_frames),
                            output_device="cpu", on_progress=report)


def _make_room(lm, device) -> bool:
    """Push the backbone to the CPU only if the decoder would not otherwise fit.

    Upstream does this unconditionally. On a card with room to spare that is
    6.8 GB copied out and back for nothing, and the copy costs more than the
    decode it was meant to make room for.
    """
    import torch

    if getattr(device, "type", None) != "cuda":
        return False
    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:
        log.debug("[yue2_comfy.generate] cannot read free VRAM", exc_info=True)
        return False
    if free >= DECODE_RESERVE_BYTES:
        return False
    log.info("[yue2_comfy.generate] %.1f GiB free, moving the backbone to the CPU "
             "for the decode", free / 1024 ** 3)
    lm.to("cpu")
    torch.cuda.empty_cache()
    return True
