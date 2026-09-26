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

llama-cpp-python is an optional dependency, and nothing waits for it. When the
wheel is importable the model is loaded in this process, which is the fast path
and the only one that can hold a model between runs; when it is not, the same
GGUF is run by the official llama.cpp binaries in a subprocess, fetched on first
use. That second road exists because the wheel is not installable everywhere --
see ``llamacpp.py`` -- and because "install this and try again" is a poor answer
from a node whose point is not having to set anything up.

Both roads render the prompt from the same chat template, with the same sampler
settings and the same seed, so which one a machine takes is a question of speed
rather than of what comes out.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from . import chat_template, devices, gguf_meta, paths
from .constants import (WRITER_AUTO, WRITER_MAX_NEW_TOKENS, WRITER_NAME,
                        install_command)

log = logging.getLogger(__name__)

PACKAGE = "llama-cpp-python"
WHEEL = PACKAGE
BINARY = "llama.cpp binary"
SUFFIX = ".gguf"
RUNNABLE = "model"
OLLAMA_PREFIX = "ollama: "
PROBLEM_PREFIX = "!! "
HEADER_KEYS = ("general.architecture", "general.type", "tokenizer.chat_template")
SCAN_SECONDS = 2.0

DEFAULT_GPU_LAYERS = -1
LOOP_EVERY = 32
PREVIEW_TAIL = 240

_STATE: dict = {"key": None, "llama": None}
_LOCK = threading.RLock()
_HEADERS: dict = {}
_CATALOGUE: dict = {"roots": None, "entries": [], "at": 0.0}


def install_hint() -> str:
    """The other road, for the interpreter that is actually running.

    Only reached when the binaries could not be had either, so it is an
    alternative rather than an instruction: the node does not ask anybody to
    install anything before it will write a song.
    """
    return ("The other way to run a writer model is " + PACKAGE + ", installed into "
            "the Python that runs ComfyUI:\n\n" + install_command(PACKAGE)
            + "\n\nPrebuilt wheels for CUDA and Vulkan are published at "
              "https://github.com/abetlen/llama-cpp-python/releases -- on Windows the "
              "plain command above usually builds from source, which needs a compiler.")


def available() -> bool:
    """Whether the in-process backend, llama-cpp-python, can be imported here."""
    import importlib.util

    return importlib.util.find_spec("llama_cpp") is not None


def backend() -> str:
    """Which of the two ways to run a GGUF this machine takes, as a label."""
    return WHEEL if available() else BINARY


def _kind(path: str) -> dict:
    """``{"type", "arch", "chat"}`` for one GGUF, cached per file identity.

    ``chat`` is the template itself, not a flag. Both backends need the text --
    the subprocess one has no model object to ask afterwards -- and it was read
    out of the header anyway to decide whether the file can write at all.
    """
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
            "chat": str(header.get("tokenizer.chat_template") or ""),
        }
    except Exception as error:
        log.debug("[yue2_comfy.llm] %s is not a usable GGUF (%s)", path, error)
    _HEADERS[key] = found
    return found


def runnable(path: str) -> bool:
    """Whether this file is a model that can answer on its own."""
    found = _kind(path)
    return bool(found) and found.get("type") == RUNNABLE and bool(found.get("chat"))


def template(path: str) -> str:
    """The chat template inside a GGUF, or "" if it carries none."""
    return _kind(path).get("chat", "")


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


def _sized(label: str, path: str) -> str:
    """The label with the file's size on it, since that is the deciding fact.

    A dropdown of file names asks somebody with an 8 GB card to guess which of
    them fits. The size is the one number that answers it, and it costs a stat.
    """
    from . import download

    try:
        size = os.path.getsize(path)
    except OSError:
        return label
    return (label + " (" + download.human_size(size) + ")") if size else label


def catalogue(refresh: bool = False) -> list:
    """Every GGUF on this machine that could write a song, as (label, path).

    Labelled by file name, because that is what the person sees in their own
    folder. A name that appears twice keeps where it came from as well, so the
    copy in a model folder and the copy in the Hugging Face cache stay tellable
    apart -- by repository, not by the commit hash the cache names things with.

    Ollama's store is read too, and its models are marked. Everything in this
    list is on disk, so "on disk" would mark nothing; what is worth saying is
    which store a file lives in, and only one of them is not a model folder.

    The folders are looked at again whenever the list is more than
    SCAN_SECONDS old. It used to be kept for the life of the process, so a GGUF
    copied into models/LLM stayed out of the list until ComfyUI restarted --
    which is what copying a model in is supposed to avoid. The look costs about
    4 ms over 26 folders (measured 2026-09-26, headers already cached); the few
    seconds only spare the second node's INPUT_TYPES in the same answer.
    """
    from . import ollama

    roots = paths.gguf_roots()
    key = (tuple(roots), tuple(ollama.roots()))
    now = time.monotonic()
    if not refresh and _CATALOGUE["roots"] == key and now - _CATALOGUE.get("at", 0.0) < SCAN_SECONDS:
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
        entries.append((_sized(name, path), path))

    seen = {os.path.normcase(path) for _label, path in entries}
    for name, path in ollama.entries():
        if os.path.normcase(path) in seen or not runnable(path):
            continue
        entries.append((_sized(OLLAMA_PREFIX + name, path), path))

    entries.sort(key=lambda entry: entry[0].lower())

    _CATALOGUE["roots"] = key
    _CATALOGUE["entries"] = entries
    _CATALOGUE["at"] = time.monotonic()
    return list(entries)


def _listed_names(listed: list) -> tuple:
    """The file names of listed Hub entries, and the paths of listed local ones, for telling twins apart."""
    names = {os.path.basename(entry.file.replace("\\", "/")).lower() for entry in listed if not entry.local}
    local = {os.path.normcase(os.path.abspath(entry.file)) for entry in listed if entry.local}
    return names, local


def offered() -> list:
    """``(label, entry or path)`` after 'auto', in the order the widget shows them.

    The model list's entries come first, in the list's own order, then the
    GGUFs found on disk and in Ollama. A file on disk that a listed entry names
    is not shown a second time under its file name: the entry stands for it,
    and picking the entry uses the file without downloading anything. A
    workflow saved with that file's own label still runs (``resolve`` and
    ``known`` read the whole catalogue). When the live list does not parse,
    a line starting with PROBLEM_PREFIX says so at the top.
    """
    from . import catalog

    rows = []
    why = catalog.problem()
    if why:
        rows.append((PROBLEM_PREFIX + why + " -- the pack's own list is shown instead", None))
    listed = catalog.entries()
    rows.extend((entry.label, entry) for entry in listed)
    names, local = _listed_names(listed)
    for label, path in catalogue():
        if label.startswith(OLLAMA_PREFIX):
            rows.append((label, path))
            continue
        if os.path.basename(path).lower() in names or os.path.normcase(os.path.abspath(path)) in local:
            continue
        rows.append((label, path))
    return rows


def choices() -> list:
    """The model widget's list: 'auto', the model list's entries, then what is on disk and in Ollama."""
    return [WRITER_AUTO] + [label for label, _target in offered()]


def listed_file(entry, found=None) -> str:
    """The file on this machine a listed entry stands for, or "" when it has to be downloaded.

    A local entry is its own path. A Hub entry is its file name in the folder
    it downloads to, or the same name anywhere the search finds it -- a model
    somebody already has in another folder is not fetched twice.
    """
    from . import download

    if entry.local:
        return entry.file if os.path.isfile(entry.file) else ""
    name = os.path.basename(entry.file.replace("\\", "/"))
    here = os.path.join(download.writer_root(), name)
    if os.path.isfile(here) and runnable(here):
        return here
    for _label, path in (catalogue() if found is None else found):
        if os.path.basename(path).lower() == name.lower():
            return path
    return ""


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

    The size in a label is cosmetic, so a saved workflow is matched without it
    too. Replacing a quant with a different one of the same name changes the
    label and must not break a graph that was working yesterday.

    An entry of the model list is used from disk when its file is here and
    downloaded into models/LLM when it is not; it is found by its label and,
    failing that, by its name, so a size or VRAM note edited since the
    workflow was saved does not lose it.
    """
    from . import catalog, download

    entries = catalogue()
    wanted = (choice or WRITER_AUTO).strip()
    if wanted and wanted != WRITER_AUTO:
        if wanted.startswith(PROBLEM_PREFIX):
            raise FileNotFoundError(
                wanted[len(PROBLEM_PREFIX):] + ".\n\nThis line only reports the problem. Mend "
                + catalog.live_file() + " or delete it, then pick a model.")
        entry = catalog.find(wanted)
        if entry is not None:
            here = listed_file(entry, entries)
            if here:
                return here
            if entry.local:
                raise FileNotFoundError(
                    "'" + entry.name + "' in the model list points at " + entry.file
                    + ", which is not there.\n\nCorrect the entry, or pick another model.")
            return download.fetch_listed(entry.name, entry.repo, entry.file, settings, progress)
        found = _on_disk(wanted, entries)
        if found:
            return found
        known = ", ".join(name for name, _path in entries) or "nothing"
        raise FileNotFoundError(
            "'" + wanted + "' is neither in the model list nor in the model folders any more. "
            "What is on disk now: " + known + ".\n\nPick another entry, or set the model widget "
            "back to '" + WRITER_AUTO + "'.")

    here = _preferred(entries)
    if here:
        return here
    return download.fetch_writer(settings, progress)


def _on_disk(wanted: str, entries: list) -> str:
    """The path of a file on disk whose label is ``wanted``, with or without its size."""
    for name, path in entries:
        if name == wanted:
            return path
    stem = wanted.split(" (")[0]
    for name, path in entries:
        if name.split(" (")[0] == stem:
            return path
    return ""


def known(choice) -> str:
    """Why a saved model choice cannot run, or "" when it can -- without downloading anything.

    For the nodes' VALIDATE_INPUTS, which takes the widget out of ComfyUI's
    own "Value not in list" check: a choice the widget no longer lists may
    still name a model (a file shown now under the list's entry, an entry
    whose note or size was edited), and one that names nothing gets a reason
    a person can act on.
    """
    from . import catalog

    if not isinstance(choice, str):
        return "the model is chosen from the list, not connected from another node"
    wanted = choice.strip()
    if not wanted or wanted == WRITER_AUTO:
        return ""
    if wanted.startswith(PROBLEM_PREFIX):
        return wanted[len(PROBLEM_PREFIX):] + " -- pick a model from the list"
    if catalog.find(wanted) is not None or _on_disk(wanted, catalogue()):
        return ""
    return ("'" + wanted + "' is neither in the model list nor in the model folders any more. "
            "Pick another model, or '" + WRITER_AUTO + "'.")


def free_comfy_vram(spec: str) -> None:
    """Evict ComfyUI's own models when the writer wants the same card.

    Making room is right when both want the same card and actively harmful when
    they do not: on a second GPU, unloading the song model costs a full reload
    and buys nothing.
    """
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

        free_comfy_vram(device)
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
        keep_loaded: bool = False, progress=None, settings=None, **sampling) -> str:
    """One answer, by whichever backend this machine has.

    The wheel wins when it is importable: it is faster per run and it is the
    only one that can hold a model between runs, which is what
    ``keep_model_loaded`` is for. Otherwise the binaries run the same file in a
    subprocess. Nothing here asks the user which; a widget for that would be a
    question about llama.cpp builds, and this node is for people who would
    rather think about the song.
    """
    if not available():
        from . import cli

        return cli.run(path, messages, seed, n_ctx, device, keep_loaded, progress,
                       settings, **sampling)

    llama = load(path, n_ctx, device, progress)
    try:
        return generate(llama, messages, seed, progress=progress, **sampling)
    finally:
        if not keep_loaded:
            unload()
