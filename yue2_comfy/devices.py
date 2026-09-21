"""Choosing which GPU the song is generated on.

One spelling, and it has to survive a saved workflow moving between machines:

===========  =================================================================
``auto``     whatever ComfyUI itself is using
``cuda:N``   a specific CUDA ordinal
``cpu``      no GPU at all; correct, and about an hour per song
===========  =================================================================

The values are deliberately plain. A label carrying the card's model name would
read better and would break every saved workflow the day the card is replaced.

Ported from the MiniMax-H3 Prompt Rewriter pack, minus the llama.cpp helpers.
torch is imported inside the functions so this module is safe to import while
ComfyUI is still building its node list.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

AUTO = "auto"
CPU = "cpu"
PREFIX = "cuda:"


def _torch():
    import torch

    return torch


def count() -> int:
    """How many CUDA devices this process can see.

    Note *this process*: ComfyUI started with --cuda-device 1 sets
    CUDA_VISIBLE_DEVICES, so there is exactly one visible device and it is
    numbered 0.
    """
    try:
        torch = _torch()
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        log.debug("[yue2_comfy.devices.count] no CUDA visible", exc_info=True)
        return 0


def describe(index: int) -> str:
    """"NVIDIA GeForce RTX 5090, 32.0 GB", or "" when it cannot be read."""
    try:
        torch = _torch()
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / 1024 ** 3
        return "{}, {:.1f} GB".format(name, total)
    except Exception:
        log.debug("[yue2_comfy.devices.describe] %s unreadable", index, exc_info=True)
        return ""

_CHOICES = None
_TOOLTIP = None


def choices() -> list:
    """The values offered by the options node, in a stable spelling."""
    global _CHOICES
    if _CHOICES is None:
        _CHOICES = [AUTO] + [PREFIX + str(index) for index in range(count())] + [CPU]
    return list(_CHOICES)


def tooltip() -> str:
    global _TOOLTIP
    if _TOOLTIP is not None:
        return _TOOLTIP
    lines = [
        "Which device generates the song. 'auto' follows ComfyUI.",
        "",
        "Pick a second card and two things change: the model no longer competes "
        "with ComfyUI's own for VRAM, and 'keep_model_loaded' becomes worth "
        "turning on, because nothing has to be evicted to make room.",
        "",
        "Two YuE2 nodes with different devices in one graph will reload 6.8 GB on "
        "every run, because the loaded model is cached per device.",
        "",
        "'cpu' works and takes roughly an hour per song.",
    ]
    found = ["  " + PREFIX + str(index) + " -- " + (describe(index) or "unreadable")
             for index in range(count())]
    if found:
        lines += ["", "Visible here:"] + found
    else:
        lines += ["", "No CUDA device is visible to ComfyUI, so only 'cpu' will do anything."]
    _TOOLTIP = "\n".join(lines)
    return _TOOLTIP


def index(spec: str):
    """The CUDA ordinal of a device spec, or None for 'auto' and 'cpu'."""
    spec = (spec or AUTO).strip().lower()
    if not spec.startswith(PREFIX):
        return None
    try:
        return int(spec[len(PREFIX):])
    except ValueError:
        return None


def is_cpu(spec: str) -> bool:
    return (spec or AUTO).strip().lower() == CPU


CPU_NOTICE = ("This run is on the CPU, which takes about an hour for a song. "
              "Point 'device' at a CUDA card if this machine has one.")


def cpu_notice(spec: str):
    """What to say when a run is about to happen on the CPU, or None when it is not.

    'auto' is the spelling that surprises people: on a machine whose CUDA is
    missing or broken it resolves to the CPU, and the run then takes an hour
    without ever saying why. Refusing instead would be worse -- upstream
    supports the CPU, and a mode somebody deliberately chose should run -- so
    the run goes ahead and says what it is doing. The tooltip carries the same
    sentence, but a tooltip is read before the choice, not after it.
    """
    try:
        device = resolve(spec)
    except Exception:
        log.debug("[yue2_comfy.devices.cpu_notice] device unreadable", exc_info=True)
        return None
    return CPU_NOTICE if getattr(device, "type", None) == CPU else None


def validate(spec: str) -> str:
    """Return the spec, refusing one this machine cannot honour.

    Refused rather than quietly demoted to 'auto': a workflow moved from a
    two-card machine asks for a card that is not there, and silently running on
    the wrong one is how somebody's video model gets evicted mid-batch.
    """
    spec = (spec or AUTO).strip() or AUTO
    ordinal = index(spec)
    if ordinal is None:
        if spec.lower() in (AUTO, CPU):
            return spec.lower()
        raise ValueError(
            "'{}' is not a device. Use '{}', '{}', or '{}N' for a CUDA card.".format(
                spec, AUTO, CPU, PREFIX))
    visible = count()
    if ordinal >= visible:
        names = ", ".join(PREFIX + str(i) for i in range(visible)) or "none"
        raise RuntimeError(
            "This workflow asks for '{}', but ComfyUI can see {} CUDA device{} ({}). "
            "Pick one that exists, or '{}'.".format(
                spec, visible if visible else "no", "" if visible == 1 else "s",
                names, AUTO))
    return spec.lower()


def resolve(spec: str):
    """The torch.device a run should use, with 'auto' asking ComfyUI."""
    torch = _torch()
    spec = (spec or AUTO).strip().lower()
    if spec == CPU:
        return torch.device("cpu")
    ordinal = index(spec)
    if ordinal is not None:
        return torch.device("cuda", ordinal)
    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:
        log.debug("[yue2_comfy.devices.resolve] no comfy device", exc_info=True)
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def llama_arguments(spec: str) -> list:
    """The device part of an llama.cpp command line.

    'auto' adds nothing at all, which lets llama.cpp make its own choice -- it
    is the only one of the three that knows whether this build has a Vulkan
    device, a CUDA one or neither.
    """
    if is_cpu(spec):
        return ["--device", "none"]
    ordinal = index(spec)
    return ["--device", "CUDA{}".format(ordinal)] if ordinal is not None else []


def layers_for(spec: str, gpu_layers: int) -> int:
    """The offload count once the device has had its say."""
    return 0 if is_cpu(spec) else int(gpu_layers)


def shares_comfy_device(spec: str) -> bool:
    """Would this run compete with ComfyUI's own models for VRAM?

    True for 'auto' -- the safe assumption. False only when the answer is
    definitely no, so an unreadable ComfyUI device leaves the eviction in place
    rather than skipping it on a guess.
    """
    ordinal = index(spec)
    if ordinal is None:
        return not is_cpu(spec)
    try:
        import comfy.model_management as mm

        current = mm.get_torch_device()
    except Exception:
        log.debug("[yue2_comfy.devices.shares_comfy_device] no comfy device", exc_info=True)
        return True
    if getattr(current, "type", None) != "cuda":
        return False
    return (current.index or 0) == ordinal
