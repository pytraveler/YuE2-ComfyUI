"""Building the two checkpoints from bare safetensors, and keeping them resident.

Nothing here goes through from_pretrained. Each model is constructed from a
config and then filled by load_state_dict, which is what makes the file layout
the user's business rather than ours: a single model.safetensors anywhere the
pack can find it is enough, with or without the rest of the Hub repository
beside it, under any file name.

The config is read from config.json next to the weights when one is there. When
there is not, the vendored constructors already carry the released architecture
as their defaults -- checked field by field against both published config.json
files -- so there is nothing to ship and nothing to keep in sync. Either way the
real check is load_state_dict(strict=True): a wrong config gives a loud tensor
mismatch, never a quiet wrong answer.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import struct
import threading
from typing import NamedTuple

from . import devices
from .constants import install_command
from .discovery import Files, locate  # noqa: F401

log = logging.getLogger(__name__)

HEADER_LIMIT = 64 * 1024 * 1024

_LOCK = threading.RLock()
_STATE = {"key": None, "lm": None, "vae": None, "tokenizer": None}


class Models(NamedTuple):
    """What a run needs. The VAE stays on the CPU until the fourth stage."""

    lm: object
    vae: object
    tokenizer: object
    device: object


def read_header(path: str) -> dict:
    """The tensor table of a safetensors file, without mapping the tensors.

    Eight bytes of little-endian length, then that many bytes of JSON. Around a
    hundred kilobytes of reading tells us the tensor names, dtypes and shapes,
    which is enough to recognise a checkpoint without touching seven gigabytes.
    """
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("Not a safetensors file: shorter than its own header")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= HEADER_LIMIT:
            raise ValueError("Not a safetensors file: implausible header length")
        body = handle.read(length)
    if len(body) != length:
        raise ValueError("Truncated safetensors header")
    header = json.loads(body.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("Malformed safetensors header")
    return header


def tensor_names(path: str) -> list:
    """Tensor keys in file order, with the metadata entry dropped."""
    return [key for key in read_header(path) if key != "__metadata__"]


def _read_state(path: str, prefix: str = "") -> dict:
    """Tensors from one file, on the CPU, optionally only those under a prefix."""
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - ComfyUI always ships it
        raise RuntimeError(
            "safetensors is missing from this Python. ComfyUI requires it, so "
            "something has gone wrong with the install.\n\n"
            + install_command("safetensors")
        ) from error
    state = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                state[key] = handle.get_tensor(key)
    return state


def _config_dict(weights_path: str) -> dict:
    """config.json beside the weights, or an empty dict for the defaults.

    A missing config is the normal case for a hand-placed single file and not
    an error. A malformed one is worth a warning, because the user put it there
    on purpose and silently ignoring it would hide the mistake.
    """
    candidate = os.path.join(os.path.dirname(weights_path) or ".", "config.json")
    if not os.path.isfile(candidate):
        return {}
    try:
        with open(candidate, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except Exception:
        log.warning(
            "[yue2_comfy.loader] ignoring unreadable %s, using the built-in "
            "architecture defaults", candidate, exc_info=True,
        )
        return {}
    if not isinstance(loaded, dict):
        log.warning("[yue2_comfy.loader] %s is not a JSON object, ignoring", candidate)
        return {}
    return loaded


def _build_lm(state: dict, settings: dict, device):
    """Fill the backbone from a state dict that is already in released layout.

    Shared by the ordinary path and the repack path, so that a checkpoint
    rebuilt from Comfy-Org's single file goes through exactly the same
    constructor and the same strict load as one read straight off disk.
    """
    import torch

    from .vendor.yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM

    config = YuE2Config(**settings)
    with torch.device("meta"):
        model = YuE2ForCausalLM(config)
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    model.to(device)
    return model


def _build_vae(state: dict, settings: dict, variant: str):
    """Fill the decoder from a state dict already in released layout."""
    import torch

    from .vendor.yue2.modeling_vae import YuE2VAE, YuE2VAEConfig

    settings = dict(settings)
    settings.setdefault("release_variant", variant)
    config = YuE2VAEConfig(**settings)
    model = YuE2VAE(config, decoder_only=True)
    decoder = {key: value for key, value in state.items() if key.startswith("decoder.")}
    off = sorted(key for key, value in decoder.items() if value.dtype != torch.float32)
    if off:
        raise ValueError(
            "The VAE decoder has to stay FP32; these tensors are not: "
            + ", ".join(off[:4]) + (" ..." if len(off) > 4 else "")
        )
    model.load_state_dict(decoder, strict=True)
    model.eval().requires_grad_(False)
    return model, config.release_variant, len(decoder)


def load_repack(path: str, device, variant: str = "standard", progress=None):
    """The backbone, decoder and vocabulary from Comfy-Org's single file.

    One read of the file yields all three. The conversion in repack.py was
    checked tensor by tensor against the released checkpoints, and what
    ultimately guards it here is the same strict load the ordinary path uses:
    a mistake in the rebuild is a loud mismatch, not a quiet wrong model.
    """
    from . import paths, repack

    if progress is not None:
        progress.text("Reading the repacked checkpoint (7.8 GB)")
    state = _read_state(path)

    if progress is not None:
        progress.text("Rebuilding the 3B backbone")
    lm = _build_lm(repack.lm_state(state), _config_dict(path), device)
    log.info("[yue2_comfy.loader] LM: rebuilt from the repack at %s", path)

    if progress is not None:
        progress.text("Rebuilding the VAE decoder")
    vae, release, count = _build_vae(repack.vae_state(state), {}, variant)
    log.info("[yue2_comfy.loader] VAE (%s): %d tensors from the repack", release, count)

    if progress is not None:
        progress.text("Reading the embedded vocabulary")
    cache = os.path.join(paths.models_root(), ".vocabulary")
    tokenizer = load_tokenizer(repack.merges_beside(path, cache))
    return lm, vae, tokenizer


def load_lm(weights_path: str, device):
    """The 3B mixture-of-transformers backbone, in the dtype the file carries.

    Built on the meta device so that no CPU copy of seven gigabytes is ever
    allocated, then filled with assign=True, which hands the mapped tensors
    straight to the parameters. The weights are already BF16 in the file, so
    nothing is cast and nothing is chosen for the user.
    """
    import torch

    from .vendor.yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM

    config = YuE2Config(**_config_dict(weights_path))
    with torch.device("meta"):
        model = YuE2ForCausalLM(config)
    state = _read_state(weights_path)
    model.load_state_dict(state, strict=True, assign=True)
    log.info("[yue2_comfy.loader] LM: %d tensors from %s", len(state), weights_path)
    model.eval().requires_grad_(False)
    model.to(device)
    return model


def load_vae(weights_path: str, variant: str = "standard"):
    """The Oobleck decoder, FP32, on the CPU.

    Built on the CPU rather than the meta device: OobleckDecoder wraps every
    convolution in torch.nn.utils.weight_norm, which computes the norm at
    registration time and has nothing to compute on meta. 217 tensors and half
    a gigabyte take about a third of a second, so there is nothing to win.

    It stays on the CPU here and moves to the accelerator only for the fourth
    stage, which is half a gigabyte and about a tenth of a second each way --
    cheaper than holding it on the card while the LM is generating.
    """
    import torch

    from .vendor.yue2.modeling_vae import YuE2VAE, YuE2VAEConfig

    settings = _config_dict(weights_path)
    settings.setdefault("release_variant", variant)
    config = YuE2VAEConfig(**settings)
    model = YuE2VAE(config, decoder_only=True)
    state = _read_state(weights_path, prefix="decoder.")
    off = sorted(key for key, value in state.items() if value.dtype != torch.float32)
    if off:
        raise ValueError(
            "The VAE decoder has to stay FP32; these tensors are not: "
            + ", ".join(off[:4]) + (" ..." if len(off) > 4 else "")
        )
    model.load_state_dict(state, strict=True)
    log.info(
        "[yue2_comfy.loader] VAE (%s): %d tensors from %s",
        config.release_variant, len(state), weights_path,
    )
    model.eval().requires_grad_(False)
    return model


def load_tokenizer(merges_path: str):
    """The frozen text and score BPE.

    tiktoken is the pack's one dependency beyond what ComfyUI already installs,
    and this constructor is the only place that needs it. The message names this
    interpreter, because in a portable build the embedded Python is not on PATH
    and a bare 'pip install' puts the package somewhere the node will never see.
    """
    from .vendor.yue2.tokenization_yue2 import YuE2TextTokenizer

    try:
        return YuE2TextTokenizer(merges_path)
    except ImportError as error:
        if "tiktoken" not in str(error):
            raise
        raise RuntimeError(
            "YuE2 needs tiktoken to read the lyrics, and it is not installed in "
            "this Python.\n\nInstall it with:\n\n" + install_command("tiktoken>=0.7")
        ) from error


def _stamp(path: str):
    info = os.stat(path)
    return (os.path.normcase(os.path.abspath(path)), info.st_size, info.st_mtime_ns)


def _cache_key(files: Files, device, variant: str):
    """What has to change before the seven gigabytes are worth reloading.

    attention_backend is deliberately not in here. GraphAR is constructed fresh
    for every call and writes nothing back into the model, so the backend is a
    property of a run, not of the loaded weights: putting it in the key would
    make toggling it in the options node evict and reload for nothing.
    """
    if files.repack:
        return (_stamp(files.repack), str(device), variant)
    return (_stamp(files.lm), _stamp(files.vae), _stamp(files.merges),
            str(device), variant)


def _empty_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        log.debug("[yue2_comfy.loader._empty_cache] skipped", exc_info=True)


def _free_comfy_vram(spec: str) -> None:
    """Evict ComfyUI's models, unless this run is going somewhere else entirely.

    Making room is right when both want the same card and actively harmful when
    they do not: on a second GPU, unloading the diffusion model costs a full
    reload afterwards and buys nothing.
    """
    if not devices.shares_comfy_device(spec):
        log.info(
            "[yue2_comfy.loader] running on %s, leaving ComfyUI's models where "
            "they are", spec,
        )
        return
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache(force=True)
    except Exception:
        log.debug("[yue2_comfy.loader._free_comfy_vram] skipped", exc_info=True)


def unload() -> None:
    """Drop the resident models and give the memory back."""
    with _LOCK:
        if _STATE["lm"] is None and _STATE["vae"] is None and _STATE["tokenizer"] is None:
            _STATE["key"] = None
            return
        _STATE.update(key=None, lm=None, vae=None, tokenizer=None)
        _empty_cache()
        log.info("[yue2_comfy.loader] unloaded")


def is_loaded() -> bool:
    return _STATE["lm"] is not None


def acquire(files: Files, device_spec: str = "auto", variant: str = "standard",
            progress=None) -> Models:
    """The three objects a run needs, from the cache when nothing has changed."""
    device = devices.resolve(device_spec)
    key = _cache_key(files, device, variant)
    with _LOCK:
        if _STATE["key"] == key and _STATE["lm"] is not None:
            return Models(_STATE["lm"], _STATE["vae"], _STATE["tokenizer"], device)

        unload()
        _free_comfy_vram(device_spec)
        if files.repack:
            lm, vae, tokenizer = load_repack(files.repack, device, variant, progress)
        else:
            if progress is not None:
                progress.text("Loading the tokenizer")
            tokenizer = load_tokenizer(files.merges)
            if progress is not None:
                progress.text("Loading the 3B backbone (7.3 GB)")
            lm = load_lm(files.lm, device)
            if progress is not None:
                progress.text("Loading the VAE decoder")
            vae = load_vae(files.vae, variant)
        _STATE.update(key=key, lm=lm, vae=vae, tokenizer=tokenizer)
        log.info("[yue2_comfy.loader] resident on %s%s", device, _vram_suffix(device))
        return Models(lm, vae, tokenizer, device)


def _vram_suffix(device) -> str:
    """', 7.13 GiB allocated' when that number exists, otherwise nothing."""
    try:
        import torch

        if getattr(device, "type", None) != "cuda":
            return ""
        gib = torch.cuda.memory_allocated(device) / (1024 ** 3)
        return ", {:.2f} GiB allocated".format(gib)
    except Exception:
        return ""
