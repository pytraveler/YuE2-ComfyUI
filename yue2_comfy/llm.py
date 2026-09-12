"""Running a small language model from a GGUF, for the writer node.

This is the second model in a pack whose whole point is not making people hunt
for the first one, so it is deliberately the least demanding kind: one file, no
config directory, no tokenizer next to it, and llama.cpp will run it on a card
that YuE2 itself would not fit on.

What a GGUF is gets decided by its header, not its name. ``general.type`` says
``model``, ``adapter`` or ``mmproj`` outright -- measured across seventeen files
in a real model folder -- and only a ``model`` with a chat template inside it can
answer a chat turn on its own. A LoRA offered in the list would load, produce
nothing, and look like the writer was simply bad at its job.

llama-cpp-python is an optional dependency. Without it the node says so and
names the command to install it into this interpreter, rather than failing on an
ImportError deep inside a load.
"""

from __future__ import annotations

import logging
import os
import threading

from . import chat_template, devices, gguf_meta, paths
from .constants import (WRITER_AUTO, WRITER_MAX_NEW_TOKENS, WRITER_NAME,
                        install_command)

log = logging.getLogger(__name__)

PACKAGE = "llama-cpp-python"
SUFFIX = ".gguf"
RUNNABLE = "model"
HEADER_KEYS = ("general.architecture", "general.type", "tokenizer.chat_template")

DEFAULT_GPU_LAYERS = -1
LOOP_EVERY = 32
PREVIEW_TAIL = 240

_STATE: dict = {"key": None, "llama": None}
_LOCK = threading.RLock()
_HEADERS: dict = {}
_CATALOGUE: dict = {"roots": None, "entries": []}


def install_hint() -> str:
    """How to get the runtime, for the interpreter that is actually running."""
    return ("The writer needs " + PACKAGE + ", which is not installed here.\n\n"
            + install_command(PACKAGE)
            + "\n\nPrebuilt wheels for CUDA and Vulkan are published at "
              "https://github.com/abetlen/llama-cpp-python/releases -- on Windows the "
              "plain command above usually builds from source, which needs a compiler.")


def available() -> bool:
    import importlib.util

    return importlib.util.find_spec("llama_cpp") is not None


def _kind(path: str) -> dict:
    """``{"type", "arch", "chat"}`` for one GGUF, cached per file identity."""
    try:
        stat = os.stat(path)
    except OSError:
        return {}
    key = (os.path.normcase(path), stat.st_size, int(stat.st_mtime))
    cached = _HEADERS.get(key)
    if cached is not None:
        return cached
    found = {}
    try:
        header = gguf_meta.keys(path, HEADER_KEYS, verify=True)
        found = {
            "type": str(header.get("general.type") or RUNNABLE),
            "arch": str(header.get("general.architecture") or ""),
            "chat": bool(header.get("tokenizer.chat_template")),
        }
    except Exception as error:
        log.debug("[yue2_comfy.llm] %s is not a usable GGUF (%s)", path, error)
    _HEADERS[key] = found
    return found


def runnable(path: str) -> bool:
    """Whether this file is a model that can answer on its own."""
    found = _kind(path)
    return bool(found) and found.get("type") == RUNNABLE and found.get("chat")


def _sweep(root: str, depth: int = 0) -> list:
    """Every .gguf at most ``paths.GGUF_DEPTH`` levels under one root.

    Bounded rather than a full walk: ``checkpoints`` is in the search now, and
    on a working install that is a large tree whose every subdirectory would
    otherwise be listed each time ComfyUI rebuilds its node list. Two levels
    covers the two layouts that exist -- a flat folder of quants, and a folder
    per model -- and nothing is gained by going deeper.
    """
    found: list = []
    try:
        entries = sorted(os.scandir(root), key=lambda entry: entry.name)
    except OSError:
        return found
    with_dirs = []
    for entry in entries:
        try:
            if entry.is_file() and entry.name.lower().endswith(SUFFIX):
                found.append(entry.path)
            elif entry.is_dir(follow_symlinks=False):
                with_dirs.append(entry.path)
        except OSError:
            continue
    if depth + 1 < paths.GGUF_DEPTH:
        for path in with_dirs:
            found.extend(_sweep(path, depth + 1))
    return found


def _where(path: str) -> str:
    """A short, human name for the place a file came from."""
    return paths.hf_repo_for(path) or os.path.basename(os.path.dirname(path))


def catalogue(refresh: bool = False) -> list:
    """Every GGUF on this machine that could write a song, as (label, path).

    Labelled by file name, because that is what the person sees in their own
    folder. A name that appears twice keeps where it came from as well, so the
    copy in a model folder and the copy in the Hugging Face cache stay tellable
    apart -- by repository, not by the commit hash the cache names things with.
    """
    roots = paths.gguf_roots()
    if not refresh and _CATALOGUE["roots"] == roots:
        return list(_CATALOGUE["entries"])

    files: list = []
    for root in roots:
        for path in _sweep(root):
            if path not in files:
                files.append(path)

    counts: dict = {}
    for path in files:
        name = os.path.basename(path)
        counts[name] = counts.get(name, 0) + 1

    entries: list = []
    for path in files:
        if not runnable(path):
            continue
        name = os.path.basename(path)
        if counts.get(name, 0) > 1:
            name = _where(path) + "/" + name
        entries.append((name, path))
    entries.sort(key=lambda entry: entry[0].lower())

    _CATALOGUE["roots"] = roots
    _CATALOGUE["entries"] = entries
    return list(entries)


def choices() -> list:
    """The model widget's list: 'auto' first, then what is already on disk."""
    return [WRITER_AUTO] + [name for name, _path in catalogue()]


def _preferred(entries: list) -> str:
    """The model 'auto' picks from what is here, or "" to fetch the default."""
    for name, path in entries:
        if os.path.basename(path) == WRITER_NAME:
            return path
    return entries[0][1] if entries else ""


def resolve(choice: str, settings: dict, progress=None) -> str:
    """The GGUF this run should use, downloading the default if it must.

    'auto' means "do not make me think about it": an existing file wins, the
    pack's own default wins among several, and only an empty machine pays for a
    download. Anything else is a name from the list, which is resolved against
    the list rather than treated as a path -- the widget is not a text box, and
    a stale workflow naming a file that has since been deleted should say so.
    """
    from . import download

    entries = catalogue()
    wanted = (choice or WRITER_AUTO).strip()
    if wanted and wanted != WRITER_AUTO:
        for name, path in entries:
            if name == wanted:
                return path
        known = ", ".join(name for name, _path in entries) or "nothing"
        raise FileNotFoundError(
            "'" + wanted + "' is not in the model folders any more. What is there now: "
            + known + ".\n\nPick another entry, or set the model widget back to '"
            + WRITER_AUTO + "'.")

    here = _preferred(entries)
    if here:
        return here
    return download.fetch_writer(settings, progress)


def _free_comfy_vram(spec: str) -> None:
    """Evict ComfyUI's own models when the writer wants the same card."""
    if not devices.shares_comfy_device(spec):
        return
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        log.debug("[yue2_comfy.llm] could not free VRAM", exc_info=True)


def _interrupted() -> None:
    try:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        return
    except AttributeError:
        return


def load(path: str, n_ctx: int, device: str, progress=None):
    """A loaded llama, reusing the one in memory when it is the same model."""
    if not available():
        raise RuntimeError(install_hint())

    key = (os.path.normcase(path), int(n_ctx), (device or devices.AUTO).lower())
    with _LOCK:
        if _STATE["key"] == key and _STATE["llama"] is not None:
            return _STATE["llama"]
        unload()

        from llama_cpp import Llama

        _free_comfy_vram(device)
        gpu_layers = 0 if devices.is_cpu(device) else DEFAULT_GPU_LAYERS
        main_gpu = devices.index(device) or 0
        if progress is not None:
            progress.text("Loading " + os.path.basename(path), force=True)
        log.info("[yue2_comfy.llm] loading %s with n_ctx=%d, gpu_layers=%d, main_gpu=%d",
                 path, n_ctx, gpu_layers, main_gpu)
        llama = Llama(model_path=path, n_ctx=int(n_ctx), n_gpu_layers=gpu_layers,
                      main_gpu=main_gpu, verbose=False)
        _STATE["key"] = key
        _STATE["llama"] = llama
        return llama


def unload() -> None:
    """Drop the loaded model and give the VRAM back."""
    import gc

    with _LOCK:
        llama = _STATE.get("llama")
        _STATE["key"] = None
        _STATE["llama"] = None
    if llama is None:
        return
    try:
        llama.close()
    except Exception:
        log.debug("[yue2_comfy.llm] close failed", exc_info=True)
    del llama
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:
        log.debug("[yue2_comfy.llm] no cache to empty", exc_info=True)


def is_loaded() -> bool:
    return _STATE.get("llama") is not None


def render(llama, messages: list) -> str:
    """The prompt string, from the GGUF's own chat template."""
    metadata = dict(getattr(llama, "metadata", {}) or {})
    return chat_template.from_metadata(metadata, messages, enable_thinking=False)


def generate(llama, messages: list, seed: int, greedy: bool = True,
             max_new_tokens: int = WRITER_MAX_NEW_TOKENS, temperature: float = 0.9,
             top_p: float = 0.95, top_k: int = 40, repetition_penalty: float = 1.05,
             progress=None) -> str:
    """One completion, streamed so the caption moves and Cancel is answered."""
    rendered = render(llama, messages)
    call = {
        "max_tokens": int(max_new_tokens),
        "repeat_penalty": float(repetition_penalty),
        "stream": True,
        "seed": int(seed) & 0xFFFFFFFF,
    }
    if greedy:
        call["temperature"] = 0.0
    else:
        call.update(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k))

    if progress is not None:
        progress.set_total(max(int(max_new_tokens), 1))
        progress.update(0, "Writing\n0 tokens")

    try:
        stream = llama(rendered, **call)
    except TypeError:
        call.pop("seed", None)
        stream = llama(rendered, **call)

    pieces: list = []
    for chunk in stream:
        pieces.append(chunk["choices"][0]["text"])
        if len(pieces) % LOOP_EVERY:
            continue
        _interrupted()
        if progress is not None:
            tail = "".join(pieces)[-PREVIEW_TAIL:]
            progress.update(len(pieces), "Writing\n{} tokens\n{}".format(len(pieces), tail))
    return "".join(pieces)


def run(path: str, messages: list, seed: int, n_ctx: int, device: str,
        keep_loaded: bool = False, progress=None, **sampling) -> str:
    """Load, answer once, and let go again unless asked to hold on."""
    llama = load(path, n_ctx, device, progress)
    try:
        return generate(llama, messages, seed, progress=progress, **sampling)
    finally:
        if not keep_loaded:
            unload()
