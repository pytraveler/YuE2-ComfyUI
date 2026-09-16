"""Mel-Band RoFormer kept between runs, for 'vocals_only' and YuE2 Vocals Only.

Separation is not cached: its input is a fresh song almost every time, and a
song kept for a second separation would cost a copy of the audio for a
saving of a few seconds. Only the model is kept, and only when
'keep_model_loaded' asks for it; the Unload Models button lets it go (see
``memory``).
"""

from __future__ import annotations

import gc
import logging
import os
import threading

log = logging.getLogger(__name__)

ROOM_BYTES = int(3.5 * 1024 ** 3)
"""Free VRAM wanted before loading: 2.5 GiB peak measured for a 198-second song (0.85 GB of weights, a window's
activations, the song's sums), and a margin for a six-minute one."""

_LOCK = threading.RLock()
_STATE = {"key": None, "net": None}


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
            log.debug("[yue2_comfy.vocals] no cache to empty", exc_info=True)


def _free_bytes(device) -> int:
    """Free memory on the card, or a very large number when it cannot be read: nothing is evicted on a guess."""
    try:
        import torch

        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except Exception:
        log.debug("[yue2_comfy.vocals] could not check free memory", exc_info=True)
        return 1 << 62


def _make_room(device) -> None:
    """Room on the card for the model, taken only when it is short: ComfyUI's models, then this pack's kept ones.

    The song model goes last. A voice is usually separated right after a song,
    and the next song would pay to load it again.
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
        log.debug("[yue2_comfy.vocals] ComfyUI's models left in place", exc_info=True)
    try:
        from .. import loader
        from ..asr import runtime as asr_runtime
        from ..sheetsage import runtime as sheetsage_runtime

        for name, keeper in (("SheetSage2", sheetsage_runtime), ("Qwen3-ASR", asr_runtime), ("YuE2", loader)):
            if _free_bytes(device) >= ROOM_BYTES:
                break
            if keeper.is_loaded():
                log.info("[yue2_comfy.vocals] unloading the kept %s model to make room", name)
                keeper.unload()
    except Exception:
        log.debug("[yue2_comfy.vocals] the pack's other models left in place", exc_info=True)


def precision(device):
    """The dtype the network runs in on ``device``: float32 everywhere.

    Measured on 81 songs on a 5090: float16 separates twice as fast (4 s
    rather than 8 s for 198 seconds of song) and lands 44 to 72 dB from float32,
    bfloat16 only 47 dB on music. float32 is kept because it is the reference
    network's own arithmetic to the last bit and has no range to run out of on
    an unusually loud recording.
    """
    import torch

    return torch.float32


def acquire(path: str, device, progress=None):
    """The network for this weights file on this device, loading it when it is not already."""
    key = (stamp(path), str(device))
    with _LOCK:
        if _STATE["key"] == key and _STATE["net"] is not None:
            return _STATE["net"]
        unload()
        _make_room(device)
        if progress is not None:
            progress.text("Loading Mel-Band RoFormer", force=True)
        from . import model

        net = model.load(path, device, precision(device))
        _STATE.update(key=key, net=net)
        log.info("[yue2_comfy.vocals] loaded %s on %s", os.path.basename(path), device)
        return net


def separate(path: str, device, waveform, rate: int, progress=None, cancelled=None):
    """The vocals of ComfyUI audio ``[batch, channels, samples]``: the same shape and rate, float32 on the CPU.

    ``progress`` gets the bar's share done; the model is loaded first if it
    is not already.
    """
    from . import model

    net = acquire(path, device, progress)
    report = None
    if progress is not None:
        def report(share):
            progress.ratio(share, "Separating the voice")
    return model.separate(net, waveform, rate, cancelled=cancelled, progress=report)
