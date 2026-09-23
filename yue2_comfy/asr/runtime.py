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

The forced aligner is kept here as well, in a second slot: a different model
on the same network, asked by 'YuE2 Edit Track' when each word of a song is
sung. Its answer is kept by the song and the words it was measured on, so
rewriting a second line of a song costs nothing.

The same node asks the speech model to hear a few seconds at a time -- each
take of a change of words, to keep the one that sings them (``hear``) -- and
those answers are kept too, by the clip's name.

The model is kept only when 'keep_model_loaded' asks for it, and the Unload
Models button lets it go (see ``memory``).

With 'low_vram' both are loaded for a small card -- the speech model's layers
packed, the audio's convolutions in parts -- and they take turns: loading one
lets the other go. What is kept is kept apart by it too, because the packed
model can hear a few words differently.
"""

from __future__ import annotations

import collections
import gc
import logging
import os
import threading

log = logging.getLogger(__name__)

KEEP_RESULTS = 8
ROOM_BYTES = int(6.5 * 1024 ** 3)
"""Free VRAM wanted before loading: 3.8 GiB of weights, the audio encoder's convolutions and a margin.

Measured on 2026-09-23: over six minutes of audio the convolutions took 2.36
GiB above the weights, more than decoding the answer did (1.58 with its
cache); that is their ceiling, a longer recording goes through them in parts.
"""
SMALL_ROOM_BYTES = int(4.5 * 1024 ** 3)
"""The same with 'low_vram': 2.49 GiB of packed weights and 1.58 GiB for decoding six minutes, measured, and a margin."""
SHORTEST_PIECE = 1600
"""A section shorter than a tenth of a second at 16 kHz is not worth hearing on its own."""

ALIGNER_ROOM_BYTES = int(4.5 * 1024 ** 3)
"""Free VRAM wanted before the aligner is loaded: 1.75 GiB of weights, 2.34 GiB for a pass over six minutes, a margin."""
SMALL_ALIGNER_ROOM_BYTES = int(2.75 * 1024 ** 3)
"""The same with 'low_vram', whose pass over six minutes took 0.65 GiB above the weights."""

KEEP_TIMES = 48
"""How many songs' word times are kept: a list of words each, and a window asks for one a take."""

KEEP_HEARD = 64
"""How many short clips' words are kept: a line of text each."""

_LOCK = threading.RLock()
_STATE = {"key": None, "net": None, "tokenizer": None}
_ALIGNER = {"key": None, "held": None}
_RESULTS = collections.OrderedDict()
_LAYOUTS = collections.OrderedDict()
_TIMES = collections.OrderedDict()
_HEARD = collections.OrderedDict()


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


def _make_room(device, room: int = ROOM_BYTES, other=None) -> None:
    """Room on the card for the model, taken only when it is short: first ComfyUI's models, then this pack's other kept ones.

    A card with room to spare keeps everything where it is, so a run that
    follows a song does not pay to reload the song model afterwards. ``other``
    is the other listener kept here, ``(name, is_loaded, unload)``: it goes
    first, being the quickest to load again.
    """
    if getattr(device, "type", "cpu") != "cuda":
        return
    if _free_bytes(device) >= room:
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
        from ..vocals import runtime as vocals_runtime

        keepers = [other] if other is not None else []
        keepers += [(name, keeper.is_loaded, keeper.unload) for name, keeper in (
            ("SheetSage2", sheetsage_runtime), ("Mel-Band RoFormer", vocals_runtime), ("YuE2", loader))]
        for name, loaded, let_go in keepers:
            if _free_bytes(device) >= room:
                break
            if loaded():
                log.info("[yue2_comfy.asr] unloading the kept %s model to make room", name)
                let_go()
    except Exception:
        log.debug("[yue2_comfy.asr] the pack's other models left in place", exc_info=True)


def acquire(folder: str, device, progress=None, low_vram: bool = False) -> tuple:
    """The network and tokenizer for this folder on this device, loading them when they are not already.

    With ``low_vram`` the model is the one packed for a small card
    (``model.load``), and the word aligner leaves the card before it comes:
    on a small card the two listen one at a time.
    """
    key = (stamp(folder), str(device), bool(low_vram))
    with _LOCK:
        if _STATE["key"] == key and _STATE["net"] is not None:
            return _STATE["net"], _STATE["tokenizer"]
        unload()
        if low_vram:
            unload_aligner()
        _make_room(device, SMALL_ROOM_BYTES if low_vram else ROOM_BYTES,
                   ("Qwen3-ForcedAligner", aligner_loaded, unload_aligner))
        if progress is not None:
            progress.text("Loading Qwen3-ASR", force=True)
        from . import model
        from .tokenizer import Tokenizer

        tokenizer = Tokenizer(os.path.join(folder, "tokenizer.json"))
        net = model.load(folder, device, low_vram=bool(low_vram))
        _STATE.update(key=key, net=net, tokenizer=tokenizer)
        log.info("[yue2_comfy.asr] loaded %s on %s%s", os.path.basename(folder), device,
                 ", packed for a small card" if low_vram else "")
        return net, tokenizer


def aligner_loaded() -> bool:
    """Whether the forced aligner is on a device, for the report Unload Models writes."""
    return _ALIGNER["held"] is not None


def unload_aligner() -> None:
    """Let go of the forced aligner and give its memory back; the times it measured stay."""
    with _LOCK:
        held = _ALIGNER["held"]
        _ALIGNER.update(key=None, held=None)
    if held is None:
        return
    del held
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


def acquire_aligner(folder: str, tokenizer_path: str, device, progress=None, low_vram: bool = False):
    """The forced aligner for this folder on this device, loaded when it is not already held.

    With ``low_vram`` it is loaded for a small card (``aligner.load``) and the
    speech model leaves the card before it comes.
    """
    key = (stamp(folder), os.path.normcase(os.path.abspath(tokenizer_path)), str(device), bool(low_vram))
    with _LOCK:
        if _ALIGNER["key"] == key and _ALIGNER["held"] is not None:
            return _ALIGNER["held"]
        unload_aligner()
        if low_vram:
            unload()
        _make_room(device, SMALL_ALIGNER_ROOM_BYTES if low_vram else ALIGNER_ROOM_BYTES,
                   ("Qwen3-ASR", is_loaded, unload))
        if progress is not None:
            progress.text("Loading the word aligner", force=True)
        from . import aligner as aligner_module

        held = aligner_module.load(folder, tokenizer_path, device, low_vram=bool(low_vram))
        _ALIGNER.update(key=key, held=held)
        log.info("[yue2_comfy.asr] loaded %s on %s", os.path.basename(folder), device)
        return held


def word_times(folder: str, tokenizer_path: str, device, waveform, rate: int, text: str, key,
               progress=None, cancelled=None, low_vram: bool = False) -> list:
    """``[(word, start, stop)]`` for ``text`` sung in ``waveform``, measured once and kept.

    ``key`` names the sound the times belong to -- the song's key, or whatever
    stands for the sound an edit has made -- and the words go into the same
    name, because other words are other times. ``low_vram`` times them with
    the aligner as a small card holds it.
    """
    import hashlib

    said = hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]
    full_key = (stamp(folder), str(key), said, bool(low_vram))
    found = _get(_TIMES, full_key)
    if found is not None:
        return found
    from . import aligner as aligner_module, model

    held = acquire_aligner(folder, tokenizer_path, device, progress, low_vram=low_vram)
    audio = model.mono_16k(waveform, rate)
    times = aligner_module.align(held, audio, text, cancelled=cancelled, progress=progress)
    _put(_TIMES, full_key, times, KEEP_TIMES)
    return times


def hear(folder: str, device, clips, language: str = "", cancelled=None, progress=None,
         low_vram: bool = False) -> list:
    """``[{"language", "text"}]``, one for each ``(name, samples)`` in ``clips``, heard one by one.

    A clip is ``[samples]`` of mono 16 kHz audio, a few seconds of a song, and
    ``name`` says which few seconds: the same name heard again is answered
    from what was kept. With ``language`` empty the first clip's language, as
    the model names it, is the one the rest are heard in, so that a run of
    takes is heard alike; the caller puts first the clip whose language is
    surest. The captured decoding step is let go however this ends.
    ``low_vram`` hears them with the model packed for a small card.
    """
    weights = stamp(folder)
    answers = []
    net = tokenizer = None
    try:
        for index, (name, samples) in enumerate(clips):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled while the takes were being heard")
            key = (weights, str(name), language, bool(low_vram))
            found = _get(_HEARD, key)
            if found is None:
                from . import model

                if net is None:
                    net, tokenizer = acquire(folder, device, progress, low_vram=low_vram)
                answer = model.recognise(net, tokenizer, samples, language=language,
                                         cancelled=cancelled)
                found = {"language": answer["language"] or language, "text": answer["text"]}
                _put(_HEARD, key, found, KEEP_HEARD)
            answers.append(found)
            if index == 0 and not language:
                language = guide_language(found["language"])
            if progress is not None:
                progress.ratio((index + 1) / len(clips))
    finally:
        forget = getattr(net, "forget_steps", None)
        if callable(forget):
            forget()
    return answers


def _get(store, key):
    with _LOCK:
        found = store.get(key)
        if found is not None:
            store.move_to_end(key)
        return found


def _put(store, key, value, keep: int = KEEP_RESULTS) -> None:
    with _LOCK:
        store[key] = value
        store.move_to_end(key)
        while len(store) > keep:
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


def recognise(folder: str, device, waveform, rate: int, timed: list, key, progress=None, cancelled=None,
              low_vram: bool = False) -> dict:
    """``{"language", "text", "parts"}``: the whole song's words, and those words cut into ``timed``'s sections.

    The captured decoding step and its cache are let go however the
    recognition ends, so a cancelled run leaves nothing on the card.
    ``low_vram`` hears them with the model packed for a small card.
    """
    full_key = tuple(key) + (plan(timed), bool(low_vram))
    found = _get(_RESULTS, full_key)
    if found is not None:
        if progress is not None:
            progress.text("Words reused: the recording has not changed", force=True)
        return found
    net, tokenizer = acquire(folder, device, progress, low_vram=low_vram)
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
