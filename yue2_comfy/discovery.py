"""Finding the three files, wherever this machine already keeps them.

Two passes, in this order, because they cost very different amounts.

The layout pass believes the published folder names -- YuE2-3B beside
YuE2-Vae, exactly as the Hub serves them and as the downloader will write them
-- and costs three isfile calls. It is what nearly every install hits.

The identification pass runs only for what the first pass did not find. It
sweeps the same roots for safetensors files and asks each one what it is, by
size first and then by reading its header: eight bytes of length and about a
hundred kilobytes of JSON, never the seven gigabytes behind it. This is the
pass that finds a file somebody renamed or dropped into models/diffusion_models,
which is the single most common way a working install looks broken.

Neither pass is the real check. load_state_dict(strict=True) in loader.py is,
and a file that fools both passes fails there with a loud tensor mismatch
rather than a quiet wrong answer.
"""

from __future__ import annotations

import json
import logging
import os
from typing import NamedTuple

from . import paths
from .constants import (
    LM_BYTES, LM_DIRNAME, MERGES_BYTES, MERGES_NAME, VAE_BYTES, VAE_DIRNAME,
    VAE_LEGACY_DIRNAME, WEIGHTS_NAME,
)

log = logging.getLogger(__name__)

REPACK_NAMES = ("yue2_3b_bf16.safetensors",)

FOLDER_KINDS = {
    LM_DIRNAME.lower(): "lm",
    VAE_DIRNAME.lower(): "standard",
    VAE_LEGACY_DIRNAME.lower(): "legacy",
}

LM_MARKERS = ("lm_head.weight", "vae2llm.weight")
VAE_PREFIX = "decoder."

_identified: dict = {}


class Files(NamedTuple):
    """Where the weights are, in one of the two shapes they come in.

    Either three released files, or the one repacked file Comfy-Org publishes
    for the native ComfyUI nodes. When repack is set the other three are empty:
    that file carries the backbone, the decoder and the vocabulary together.
    """

    lm: str = ""
    vae: str = ""
    merges: str = ""
    repack: str = ""


def _stamp(path: str):
    """Identity of a file for caching: a rewritten file is a different file."""
    info = os.stat(path)
    return (os.path.normcase(os.path.abspath(path)), info.st_size, info.st_mtime_ns)


def folder_kind(directory: str) -> str:
    """Which published folder this is, by name, or "" when it is none of them.

    A Hugging Face snapshot directory is named after the revision, so the name
    that means anything sits two levels up, in models--m-a-p--YuE2-3B.
    """
    normalized = os.path.normpath(directory)
    name = os.path.basename(normalized)
    if os.path.basename(os.path.dirname(normalized)) == "snapshots":
        repo = os.path.basename(os.path.dirname(os.path.dirname(normalized)))
        name = repo.replace("models--", "", 1).split("--")[-1]
    return FOLDER_KINDS.get(name.lower(), "")


def identify(path: str) -> str:
    """"lm", "vae", "repack" or "" for one file, as cheaply as possible."""
    try:
        key = _stamp(path)
    except OSError:
        return ""
    if key in _identified:
        return _identified[key]

    size = key[1]
    if size == LM_BYTES:
        verdict = "lm"
    elif size == VAE_BYTES:
        verdict = "vae"
    else:
        verdict = _identify_by_header(path)
    _identified[key] = verdict
    return verdict


def _identify_by_header(path: str) -> str:
    from .loader import read_header
    from .repack import is_repack

    try:
        header = read_header(path)
        if is_repack(header):
            return "repack"
        names = {key for key in header if key != "__metadata__"}
    except Exception:
        log.debug("[yue2_comfy.discovery] cannot read %s", path, exc_info=True)
        return ""
    if all(marker in names for marker in LM_MARKERS):
        return "lm"
    if LM_MARKERS[0] not in names and any(name.startswith(VAE_PREFIX) for name in names):
        return "vae"
    return ""


def vae_variant(weights_path: str) -> str:
    """"standard" or "legacy" for a VAE file, by config first and name second.

    The two releases are the same size, so nothing cheap distinguishes them
    except what the publisher wrote down. release_variant is that; the folder
    name is the fallback for a hand-placed file with no config beside it.
    """
    directory = os.path.dirname(weights_path) or "."
    config = os.path.join(directory, "config.json")
    if os.path.isfile(config):
        try:
            with open(config, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            declared = str(loaded.get("release_variant", "")).strip().lower()
            if declared in ("standard", "legacy"):
                return declared
        except Exception:
            log.debug("[yue2_comfy.discovery] unreadable %s", config, exc_info=True)
    kind = folder_kind(directory)
    return kind if kind in ("standard", "legacy") else "standard"


def _homes(root: str) -> list:
    """A root and its immediate subdirectories, which is as deep as this goes.

    Depth two covers every layout anyone actually has: the published folders
    under models/YuE2, a Hugging Face snapshot whose files sit at its own top
    level, and a loose file dropped straight into models/vae. Deeper than that
    and a sweep of a large model tree starts costing real time for nothing.
    """
    found = [root]
    try:
        for entry in sorted(os.listdir(root)):
            path = os.path.join(root, entry)
            if os.path.isdir(path):
                found.append(path)
    except OSError:
        pass
    return found


def _first_file(home: str, name: str) -> str:
    candidate = os.path.join(home, name)
    return candidate if os.path.isfile(candidate) else ""


def _layout_pass(roots: list, variant: str) -> dict:
    """What the published folder names give us, for three isfile calls apiece."""
    want_vae = "legacy" if variant == "legacy" else "standard"
    found = {"lm": "", "vae": "", "merges": ""}
    for root in roots:
        for home in _homes(root):
            kind = folder_kind(home)
            if not kind:
                continue
            if kind == "lm":
                if not found["lm"]:
                    found["lm"] = _first_file(home, WEIGHTS_NAME)
                if not found["merges"]:
                    found["merges"] = _first_file(home, MERGES_NAME)
            elif kind == want_vae and not found["vae"]:
                found["vae"] = _first_file(home, WEIGHTS_NAME)
        if all(found.values()):
            break
    return found


def _sweep(roots: list, want: dict, variant: str) -> None:
    """Fill whatever the layout pass missed, by looking inside the files."""
    want_vae = "legacy" if variant == "legacy" else "standard"
    for root in roots:
        if all(want.values()):
            return
        for home in _homes(root):
            try:
                entries = sorted(os.listdir(home))
            except OSError:
                continue
            for entry in entries:
                path = os.path.join(home, entry)
                if not os.path.isfile(path):
                    continue
                if not want["merges"] and _looks_like_merges(path, entry):
                    want["merges"] = path
                if not entry.lower().endswith(".safetensors"):
                    continue
                if want["lm"] and want["vae"]:
                    continue
                kind = identify(path)
                if kind == "lm" and not want["lm"]:
                    want["lm"] = path
                elif kind == "vae" and not want["vae"] and vae_variant(path) == want_vae:
                    want["vae"] = path


def _looks_like_merges(path: str, entry: str) -> bool:
    """The tiktoken vocabulary, by size when possible and by name otherwise."""
    if entry.lower().endswith(".tiktoken"):
        return True
    try:
        return os.path.getsize(path) == MERGES_BYTES
    except OSError:
        return False


def find_repack(roots: list) -> str:
    """Comfy-Org's single file, by name first and by what is inside it second.

    The published name is checked before any file is opened, because in a
    models/checkpoints folder full of multi-gigabyte checkpoints the difference
    between one isfile call and a header read for every one of them is the
    difference between instant and noticeable.
    """
    from .repack import is_repack

    for root in roots:
        for home in _homes(root):
            for name in REPACK_NAMES:
                candidate = os.path.join(home, name)
                if os.path.isfile(candidate):
                    return candidate
    for root in roots:
        for home in _homes(root):
            try:
                entries = sorted(os.listdir(home))
            except OSError:
                continue
            for entry in entries:
                if not entry.lower().endswith(".safetensors"):
                    continue
                candidate = os.path.join(home, entry)
                if identify(candidate) == "repack":
                    return candidate
    return ""


def locate(variant: str = "standard") -> Files:
    """The weights, in whichever shape this machine has them.

    The three released files win when they are all present, because they load
    without any conversion. Comfy-Org's repack is the fallback and, increasingly,
    the common case: it is what the ComfyUI model manager installs. Both were
    checked to produce the same model tensor for tensor.

    The variant chooses between the two VAE releases and nothing else. The
    repack carries only the standard decoder, so asking it for the legacy one
    falls through to the released files rather than quietly handing back the
    wrong decoder.
    """
    roots = paths.search_roots()
    found = _layout_pass(roots, variant)
    if not all(found.values()):
        _sweep(roots, found, variant)
    if all(found.values()):
        log.info("[yue2_comfy.discovery] weights found: %s", os.path.dirname(found["lm"]))
        return Files(lm=found["lm"], vae=found["vae"], merges=found["merges"])

    if variant != "legacy":
        repacked = find_repack(roots)
        if repacked:
            log.info("[yue2_comfy.discovery] using the repacked checkpoint at %s", repacked)
            return Files(repack=repacked)

    raise FileNotFoundError(_missing_message(roots, found, variant))


def _expected_root() -> str:
    """Where the files are supposed to go, for a message that can be acted on."""
    override = os.environ.get(paths.ENV_ROOT)
    if override:
        return override
    try:
        return paths.models_root()
    except Exception:
        return os.path.join("ComfyUI", "models", "YuE2")


def _missing_message(roots: list, found: dict, variant: str) -> str:
    vae_dirname = VAE_LEGACY_DIRNAME if variant == "legacy" else VAE_DIRNAME
    root = _expected_root()
    wanted = {
        "lm": os.path.join(root, LM_DIRNAME, WEIGHTS_NAME),
        "vae": os.path.join(root, vae_dirname, WEIGHTS_NAME),
        "merges": os.path.join(root, LM_DIRNAME, MERGES_NAME),
    }
    missing = [wanted[key] for key in ("lm", "vae", "merges") if not found[key]]
    lines = ["YuE2 weights are not on this machine yet. Missing:", ""]
    lines += ["  " + path for path in missing]
    if roots:
        lines += ["", "Looked in " + str(len(roots)) + " places, including:", ""]
        lines += ["  " + path for path in roots[:6]]
    else:
        lines += ["", "There was nowhere to look: no ComfyUI model folders were found."]
    lines += [
        "",
        "Put the files at the paths above, or set " + paths.ENV_ROOT + " to a "
        "directory holding " + LM_DIRNAME + " and " + vae_dirname + ".",
    ]
    return "\n".join(lines)
