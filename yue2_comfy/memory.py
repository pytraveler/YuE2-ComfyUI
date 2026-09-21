"""ComfyUI's Unload Models reaching the models this pack keeps.

'keep_model_loaded' holds the YuE2 model, and the writer's GGUF, in this pack's
own caches, where ComfyUI's model management never sees them. The Unload Models
button, and the POST /free it sends, only raises a flag on the prompt queue:
ComfyUI's main loop reads the flags between prompts and unloads its own models.
So the pack listens at that exact point. The queue's get_flags is wrapped, and
when the flags it hands out ask for models to be unloaded or memory freed, every
model this pack keeps is let go as well.

Wrapping comfy.model_management.unload_all_models instead would be wrong. The
pack calls it itself to make room before it loads, and ComfyUI calls it when a
card runs short; neither should throw away a model the user asked to keep.

Between prompts is also the one safe moment: nothing of this pack is running
then. The wrap goes on once, is guarded, and does nothing outside ComfyUI.
"""

from __future__ import annotations

import importlib
import logging
import threading

log = logging.getLogger(__name__)

MARK = "__yue2_releases_models__"
"""Set on the wrapper, so installing twice (a reloaded pack) never wraps the wrapper."""

_LOCK = threading.Lock()
_KEEPERS: list = []


def keeper(name: str, module: str) -> None:
    """Register a module of this pack whose ``unload()`` lets go of a kept model.

    The module is named rather than imported: importing it here would pull its
    imports into ComfyUI's start-up for a button that may never be pressed.
    """
    with _LOCK:
        if (name, module) not in _KEEPERS:
            _KEEPERS.append((name, module))


def keepers() -> list:
    """The registered ``(name, module)`` pairs, in the order they were added."""
    with _LOCK:
        return list(_KEEPERS)


def asks_to_release(flags) -> bool:
    """Whether queue flags ask for models to go, read the way ComfyUI's main loop reads them.

    The loop unloads on 'unload_models', which defaults to 'free_memory', and
    'free_memory' frees everything besides; either one is enough here.
    """
    if not isinstance(flags, dict):
        return False
    free_memory = bool(flags.get("free_memory", False))
    return bool(flags.get("unload_models", free_memory)) or free_memory


def release_all() -> list:
    """Unload every kept model; the names of those that were loaded.

    One keeper failing does not keep the others loaded, and nothing raised here
    may reach ComfyUI's main loop.
    """
    released = []
    for name, module in keepers():
        try:
            target = importlib.import_module(module, __package__)
            is_loaded = getattr(target, "is_loaded", None)
            loaded = bool(is_loaded()) if callable(is_loaded) else True
            target.unload()
            if loaded:
                released.append(name)
        except Exception:
            log.warning("[yue2_comfy] could not unload the %s model", name, exc_info=True)
    if released:
        log.info("[yue2_comfy] Unload Models: released %s", ", ".join(released))
    return released


def install(queue_class=None) -> bool:
    """Wrap the prompt queue's get_flags; False when it is wrapped already.

    Flags read without resetting them are a look, not a hand-over, so only a
    reset read releases anything: the flags are consumed exactly once.
    """
    if queue_class is None:
        import execution

        queue_class = execution.PromptQueue
    original = queue_class.get_flags
    if getattr(original, MARK, False):
        return False

    def get_flags(self, *args, **kwargs):
        flags = original(self, *args, **kwargs)
        reset = kwargs.get("reset", args[0] if args else True)
        try:
            if reset and asks_to_release(flags):
                release_all()
        except Exception:
            log.warning("[yue2_comfy] releasing kept models failed", exc_info=True)
        return flags

    setattr(get_flags, MARK, True)
    get_flags.__wrapped__ = original
    queue_class.get_flags = get_flags
    return True


keeper("YuE2", ".loader")
keeper("writer", ".llm")
keeper("SheetSage2", ".sheetsage.runtime")
keeper("Qwen3-ASR", ".asr.runtime")
keeper("Mel-Band RoFormer", ".vocals.runtime")
keeper("edited takes", ".edit_track")

try:
    install()
    log.info("[yue2_comfy] Unload Models also releases the models this pack keeps")
except Exception as error:  # noqa: BLE001
    log.debug("[yue2_comfy] Unload Models hook not installed: %s", error)
