"""SheetSage2 kept between runs, and the transcriptions it has already made.

Transcription is deterministic: the same recording gives the same tokens
whatever the seed and whichever mode is asked for, because the mode only
changes how the score is written from them. So a result is kept by the mark of
the recording and the weights file, and a run that changes nothing but the
seed, the mode or a kept edit costs no model time at all. Eight results are
kept, each a list of events and a few thousand tokens.

The model itself is kept only when 'keep_model_loaded' asks for it, and the
Unload Models button lets it go (see ``memory``).
"""

from __future__ import annotations

import collections
import gc
import logging
import os
import threading

log = logging.getLogger(__name__)

KEEP_RESULTS = 8
ROOM_BYTES = 4 * 1024 ** 3
"""Free VRAM wanted before loading: 1.3 GB of weights, a window's activations (1.9 GiB measured) and a margin."""

_LOCK = threading.RLock()
_STATE = {"key": None, "net": None}
_RESULTS = collections.OrderedDict()


def stamp(path: str) -> tuple:
    """A weights file's identity for caching: a rewritten file is a different file."""
    info = os.stat(path)
    return (os.path.normcase(os.path.abspath(path)), info.st_size, info.st_mtime_ns)


def is_loaded() -> bool:
    return _STATE["net"] is not None


def unload() -> None:
    """Let go of the network and give its memory back."""
    with _LOCK:
        net = _STATE["net"]
        _STATE.update(key=None, net=None)
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
            log.debug("[yue2_comfy.sheetsage] no cache to empty", exc_info=True)


def _free_bytes(device) -> int:
    """Free memory on the card, or a very large number when it cannot be read: nothing is evicted on a guess."""
    try:
        import torch

        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except Exception:
        log.debug("[yue2_comfy.sheetsage] could not check free memory", exc_info=True)
        return 1 << 62


def _make_room(device) -> None:
    """Room on the card for the model, taken only when it is short: first ComfyUI's models, then this pack's YuE2.

    A card with room to spare keeps everything where it is, so a cover made
    right after a song does not pay to reload the song model.
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
        log.debug("[yue2_comfy.sheetsage] ComfyUI's models left in place", exc_info=True)
    try:
        from .. import loader
        from ..vocals import runtime as vocals_runtime

        for name, keeper in (("Mel-Band RoFormer", vocals_runtime), ("YuE2", loader)):
            if _free_bytes(device) >= ROOM_BYTES:
                break
            if keeper.is_loaded():
                log.info("[yue2_comfy.sheetsage] unloading the kept %s model to make room", name)
                keeper.unload()
    except Exception:
        log.debug("[yue2_comfy.sheetsage] the pack's other models left in place", exc_info=True)


def acquire(path: str, device, progress=None):
    """The network for this weights file on this device, loading it when it is not already."""
    key = (stamp(path), str(device))
    with _LOCK:
        if _STATE["key"] == key and _STATE["net"] is not None:
            return _STATE["net"]
        unload()
        _make_room(device)
        if progress is not None:
            progress.text("Loading SheetSage2", force=True)
        from . import model

        net = model.load(path, device)
        _STATE.update(key=key, net=net)
        log.info("[yue2_comfy.sheetsage] loaded %s on %s", os.path.basename(path), device)
        return net


def remembered(key):
    """A transcription already made for this key, or None."""
    with _LOCK:
        result = _RESULTS.get(key)
        if result is not None:
            _RESULTS.move_to_end(key)
        return result


def remember(key, result) -> None:
    with _LOCK:
        _RESULTS[key] = result
        _RESULTS.move_to_end(key)
        while len(_RESULTS) > KEEP_RESULTS:
            _RESULTS.popitem(last=False)


def transcribe(path: str, device, waveform, rate: int, key, progress=None, cancelled=None) -> dict:
    """A recording's events and tokens, from the cache when this key has been heard before."""
    result = remembered(key)
    if result is not None:
        if progress is not None:
            progress.text("Transcription reused: the recording has not changed", force=True)
        return result
    net = acquire(path, device, progress)
    from . import model

    def report(stage, window, windows, tokens):
        if progress is None:
            return
        share = 1.0 / max(windows, 1)
        if stage == "encode":
            progress.ratio(window * share + 0.05 * share,
                           "Listening to the recording" + ("" if windows == 1 else
                                                           ", part {} of {}".format(window + 1, windows)))
        else:
            done = min(1.0, tokens / 5120.0)
            progress.ratio(window * share + share * (0.1 + 0.9 * done),
                           "Writing down what it hears: {} tokens".format(tokens))

    result = model.transcribe(net, waveform, rate, cancelled=cancelled, progress=report)
    for warning in result.get("warnings", ()):
        log.info("[yue2_comfy.sheetsage] %s", warning)
    for part in result.get("cut_short", ()):
        log.warning("[yue2_comfy.sheetsage] part %d of %d filled the decoder's tokens before its end",
                    part, len(result.get("windows", ())))
    for handover in result.get("handed_over", ()):
        log.warning("[yue2_comfy.sheetsage] part %d of %d takes over from %.1f s, where the part before it stopped "
                    "(%.1f s short of its end)", handover["window"] + 1, len(result.get("windows", ())),
                    handover["from"], handover["to"] - handover["from"])
    remember(key, result)
    return result
