"""Qwen3-ASR kept between runs, the words it has already heard, and the layouts already made of them.

Recognition is deterministic: the same recording cut at the same sections
gives the same words, so a result is kept by the recording's mark, the section
plan and the weights, and a run that changes only the seed, the mode or an edit
hears nothing again. The language model's layout is kept too, by the words,
the model file and the seed: a new seed asks for a new layout and nothing more.

The words of a song are recognised in one pass over the whole recording, which
hears them best. Each section with a voice in it is then heard once more on its
own, and those rougher texts only guide where the whole song's words are cut
(see ``align``).

The model is kept only when 'keep_model_loaded' asks for it, and the Unload
Models button lets it go (see ``memory``).
"""

from __future__ import annotations

import collections
import gc
import logging
import os
import threading

log = logging.getLogger(__name__)

KEEP_RESULTS = 8
ROOM_BYTES = 6 * 1024 ** 3
"""Free VRAM wanted before loading: 3.8 GB of weights, a song's cache, the audio encoder's activations and a margin."""
SHORTEST_PIECE = 1600
"""A section shorter than a tenth of a second at 16 kHz is not worth hearing on its own."""

_LOCK = threading.RLock()
_STATE = {"key": None, "net": None, "tokenizer": None}
_RESULTS = collections.OrderedDict()
_LAYOUTS = collections.OrderedDict()


def stamp(folder: str) -> tuple:
    """The weights file's identity for caching: a rewritten file is a different file."""
    path = os.path.join(folder, "model.safetensors")
    info = os.stat(path)
    return (os.path.normcase(os.path.abspath(path)), info.st_size, info.st_mtime_ns)


def is_loaded() -> bool:
    return _STATE["net"] is not None


def unload() -> None:
    """Let go of the network and give its memory back."""
    with _LOCK:
        net = _STATE["net"]
        _STATE.update(key=None, net=None, tokenizer=None)
    if net is None:
        return
    del net
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            log.debug("[yue2_comfy.asr] no cache to empty", exc_info=True)


def _free_bytes(device) -> int:
    """Free memory on the card, or a very large number when it cannot be read: nothing is evicted on a guess."""
    try:
        import torch

        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except Exception:
        log.debug("[yue2_comfy.asr] could not check free memory", exc_info=True)
        return 1 << 62


def _make_room(device) -> None:
    """Room on the card for the model, taken only when it is short: first ComfyUI's models, then this pack's other kept ones.

    A card with room to spare keeps everything where it is, so a run that
    follows a song does not pay to reload the song model afterwards.
    """
    if getattr(device, "type", "cpu") != "cuda":
        return
    if _free_bytes(device) >= ROOM_BYTES:
        return
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache(force=True)
    except Exception:
        log.debug("[yue2_comfy.asr] ComfyUI's models left in place", exc_info=True)
    try:
        from .. import loader
        from ..sheetsage import runtime as sheetsage_runtime

        for name, keeper in (("SheetSage2", sheetsage_runtime), ("YuE2", loader)):
            if _free_bytes(device) >= ROOM_BYTES:
                break
            if keeper.is_loaded():
                log.info("[yue2_comfy.asr] unloading the kept %s model to make room", name)
                keeper.unload()
    except Exception:
        log.debug("[yue2_comfy.asr] the pack's other models left in place", exc_info=True)


def acquire(folder: str, device, progress=None) -> tuple:
    """The network and tokenizer for this folder on this device, loading them when they are not already."""
    key = (stamp(folder), str(device))
    with _LOCK:
        if _STATE["key"] == key and _STATE["net"] is not None:
            return _STATE["net"], _STATE["tokenizer"]
        unload()
        _make_room(device)
        if progress is not None:
            progress.text("Loading Qwen3-ASR", force=True)
        from . import model
        from .tokenizer import Tokenizer

        tokenizer = Tokenizer(os.path.join(folder, "tokenizer.json"))
        net = model.load(folder, device)
        _STATE.update(key=key, net=net, tokenizer=tokenizer)
        log.info("[yue2_comfy.asr] loaded %s on %s", os.path.basename(folder), device)
        return net, tokenizer


def _get(store, key):
    with _LOCK:
        found = store.get(key)
        if found is not None:
            store.move_to_end(key)
        return found


def _put(store, key, value) -> None:
    with _LOCK:
        store[key] = value
        store.move_to_end(key)
        while len(store) > KEEP_RESULTS:
            store.popitem(last=False)


def plan(timed: list) -> tuple:
    """The part of a section list recognition depends on: where each section is and whether it is sung."""
    return tuple((round(float(s["start"]), 2), round(float(s["end"]), 2), s["notes"] > 0) for s in timed)


def guide_language(language) -> str:
    """The language the section passes are told, or "" when the model named one it was not trained to name.

    The whole pass lets the model name the language, and it may write one
    outside its own list; naming that back to it would be refused, so the
    section passes let it tell again.
    """
    from . import prompt

    try:
        return prompt.language_name(language)
    except ValueError:
        log.info("[yue2_comfy.asr] the model named a language it does not list, %r; the sections are heard unguided",
                 language)
        return ""


def recognise(folder: str, device, waveform, rate: int, timed: list, key, progress=None, cancelled=None) -> dict:
    """``{"language", "text", "parts"}``: the whole song's words, and those words cut into ``timed``'s sections.

    The captured decoding step and its cache are let go however the
    recognition ends, so a cancelled run leaves nothing on the card.
    """
    full_key = tuple(key) + (plan(timed),)
    found = _get(_RESULTS, full_key)
    if found is not None:
        if progress is not None:
            progress.text("Words reused: the recording has not changed", force=True)
        return found
    net, tokenizer = acquire(folder, device, progress)
    try:
        result = _recognise(net, tokenizer, waveform, rate, timed, progress, cancelled)
    finally:
        forget = getattr(net, "forget_steps", None)
        if callable(forget):
            forget()
    _put(_RESULTS, full_key, result)
    return result


def _recognise(net, tokenizer, waveform, rate: int, timed: list, progress, cancelled) -> dict:
    from . import align, model, prompt

    audio = model.mono_16k(waveform, rate)
    sung = [index for index, section in enumerate(timed) if section["notes"] > 0]
    guided = len(sung) > 1
    steps = 1 + (len(sung) if guided else 0)

    def report(step, text):
        if progress is None:
            return None
        progress.ratio(step / steps, text)
        limit = model.token_limit(audio.numel() / prompt.SAMPLE_RATE) if step == 0 else 0
        return (lambda count: progress.ratio(min(1.0, count / max(limit / 4, 1)) / steps,
                                             "Listening for the words: {} tokens".format(count))) if limit else None

    import time

    began = time.monotonic()
    whole = model.recognise(net, tokenizer, audio, cancelled=cancelled,
                            progress=report(0, "Listening for the words"))
    log.info("[yue2_comfy.asr] heard %.0f s of audio in %.1f s, %d tokens",
             audio.numel() / prompt.SAMPLE_RATE, time.monotonic() - began, len(whole.get("tokens", ())))
    began = time.monotonic()
    language = whole["language"] or ""
    guide = guide_language(language)
    parts = [""] * len(timed)
    if guided:
        guides = [""] * len(timed)
        for step, index in enumerate(sung, start=1):
            if cancelled is not None and cancelled():
                raise InterruptedError("recognition cancelled")
            report(step, "Finding where each section's words begin, {} of {}".format(step, len(sung)))
            section = timed[index]
            piece = audio[int(section["start"] * prompt.SAMPLE_RATE):int(section["end"] * prompt.SAMPLE_RATE)]
            if piece.numel() >= SHORTEST_PIECE:
                guides[index] = model.recognise(net, tokenizer, piece, language=guide, cancelled=cancelled)["text"]
        parts = align.split(whole["text"], guides)
        log.info("[yue2_comfy.asr] guided %d sections in %.1f s", len(sung), time.monotonic() - began)
    elif timed:
        parts[sung[0] if sung else 0] = whole["text"]
    return {"language": language, "text": whole["text"], "parts": parts}


def laid_out(key, produce):
    """The layout kept for ``key``, or ``produce()``'s answer, kept for next time."""
    found = _get(_LAYOUTS, key)
    if found is None:
        found = produce()
        _put(_LAYOUTS, key, found)
    return found
