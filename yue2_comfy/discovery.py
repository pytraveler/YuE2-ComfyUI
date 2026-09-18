"""Finding the weights, wherever this machine already keeps them.

They come in two shapes: the three files m-a-p released, and the single file
Comfy-Org repacked for the native ComfyUI nodes, which itself comes in a BF16
and an INT8 build. All of them are looked for, and all of them were measured to
give the same model.

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

What it asks of a file is its own tensor names, not the shape of them. A
decoder is recognised by the names in VAE_MARKERS and nothing looser: the
sweep reaches models/vae, where a machine that generates video keeps decoders
of its own, and "has a tensor called decoder.something" is true of every one
of them.

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
    ASR_BYTES, ASR_DIRNAME, ASR_FILES, ASR_MARKER, ASR_MARKER_SHAPE, ASR_REPO, ASR_TOKENIZER_NAME,
    LM_BYTES, LM_DIRNAME, MERGES_BYTES, MERGES_NAME, REPACK_BF16_BYTES,
    REPACK_BF16_NAME, REPACK_INT8_BYTES, REPACK_INT8_NAME, REPACK_REPO,
    SHEETSAGE_BYTES, SHEETSAGE_MARKERS, SHEETSAGE_NAME, SHEETSAGE_PATH,
    VAE_BYTES, VAE_DIRNAME, VAE_LEGACY_DIRNAME, VAE_MARKERS, VOCALS_BYTES, VOCALS_KNOWN_BYTES,
    VOCALS_MARKERS, VOCALS_NAME, VOCALS_REPO, VOCALS_REVISION, WEIGHTS_NAME,
)

log = logging.getLogger(__name__)

REPACK_KINDS = {REPACK_BF16_NAME: "repack", REPACK_INT8_NAME: "repack_int8"}
REPACK_FOR = {"bf16": REPACK_BF16_NAME, "int8": REPACK_INT8_NAME}

FOLDER_KINDS = {
    LM_DIRNAME.lower(): "lm",
    VAE_DIRNAME.lower(): "standard",
    VAE_LEGACY_DIRNAME.lower(): "legacy",
}

LM_MARKERS = ("lm_head.weight", "vae2llm.weight")

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
    """"lm", "vae", "repack", "repack_int8", "sheetsage", "vocals" or "", as cheaply as possible."""
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
    elif size == REPACK_BF16_BYTES:
        verdict = "repack"
    elif size == REPACK_INT8_BYTES:
        verdict = "repack_int8"
    elif size == SHEETSAGE_BYTES:
        verdict = "sheetsage"
    elif size in VOCALS_KNOWN_BYTES:
        verdict = "vocals"
    else:
        verdict = _identify_by_header(path)
    _identified[key] = verdict
    return verdict


def _identify_by_header(path: str) -> str:
    from .loader import read_header
    from .repack import is_quantized, is_repack

    try:
        header = read_header(path)
        if is_repack(header):
            return "repack_int8" if is_quantized(header) else "repack"
        names = {key for key in header if key != "__metadata__"}
    except Exception:
        log.debug("[yue2_comfy.discovery] cannot read %s", path, exc_info=True)
        return ""
    if all(marker in names for marker in SHEETSAGE_MARKERS):
        return "sheetsage"
    if all(marker in names for marker in VOCALS_MARKERS):
        return "vocals"
    if all(marker in names for marker in LM_MARKERS):
        return "lm"
    if all(marker in names for marker in VAE_MARKERS):
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


def find_repack(roots: list, quantization: str = "bf16") -> str:
    """Comfy-Org's single file, by name first and by what is inside it second.

    The published names are checked before any file is opened, because in a
    models/checkpoints folder full of multi-gigabyte checkpoints the difference
    between two isfile calls and a header read for every one of them is the
    difference between instant and noticeable.

    The requested build is preferred and the other one is still accepted. A
    machine that has the BF16 file should not be told to download the INT8 one
    to satisfy a switch, and the node says in the log which build it used.
    """
    preferred = REPACK_FOR.get(quantization, REPACK_BF16_NAME)
    order = [preferred] + [name for name in REPACK_KINDS if name != preferred]
    for name in order:
        for root in roots:
            for home in _homes(root):
                candidate = os.path.join(home, name)
                if os.path.isfile(candidate):
                    return candidate
    wanted = REPACK_KINDS[preferred]
    fallback = ""
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
                kind = identify(candidate)
                if kind == wanted:
                    return candidate
                if kind in ("repack", "repack_int8") and not fallback:
                    fallback = candidate
    return fallback


def locate(variant: str = "standard", quantization: str = "bf16") -> Files:
    """The weights, in whichever shape this machine has them.

    The order is what costs the user least. An INT8 build that is already here
    is used when INT8 was asked for, because the alternative is downloading
    seven gigabytes to say the same thing less compactly. Otherwise the three
    released files win when they are all present, since they load without any
    conversion, and Comfy-Org's repack is the fallback -- increasingly the
    common case, as it is what the ComfyUI model manager installs. All three
    layouts were checked to produce the same model, the INT8 one to within an
    ordinary quantization round trip.

    The variant chooses between the two VAE releases and nothing else. Neither
    repack carries the legacy decoder, so asking for it falls through to the
    released files rather than quietly handing back the wrong decoder.
    """
    roots = paths.search_roots()
    if quantization == "int8" and variant != "legacy":
        candidate = find_repack(roots, "int8")
        if candidate and identify(candidate) == "repack_int8":
            log.info("[yue2_comfy.discovery] using the INT8 checkpoint at %s", candidate)
            return Files(repack=candidate)

    found = _layout_pass(roots, variant)
    if not all(found.values()):
        _sweep(roots, found, variant)
    if all(found.values()):
        log.info("[yue2_comfy.discovery] weights found: %s", os.path.dirname(found["lm"]))
        return Files(lm=found["lm"], vae=found["vae"], merges=found["merges"])

    if variant != "legacy":
        repacked = find_repack(roots, quantization)
        if repacked:
            log.info("[yue2_comfy.discovery] using the repacked checkpoint at %s", repacked)
            return Files(repack=repacked)

    raise FileNotFoundError(_missing_message(roots, found, variant, quantization))


def _expected_root() -> str:
    """Where the files are supposed to go, for a message that can be acted on."""
    override = os.environ.get(paths.ENV_ROOT)
    if override:
        return override
    try:
        return paths.models_root()
    except Exception:
        return os.path.join("ComfyUI", "models", "YuE2")


def _checkpoints_root() -> str:
    override = os.environ.get(paths.ENV_ROOT)
    if override:
        return os.path.join(override, "checkpoints")
    try:
        return paths.checkpoints_root()
    except Exception:
        return os.path.join("ComfyUI", "models", "checkpoints")


def _link(repo: str, repo_path: str, revision: str = "main") -> str:
    return "https://huggingface.co/" + repo + "/resolve/" + revision + "/" + repo_path


def _missing_message(roots: list, found: dict, variant: str, quantization: str) -> str:
    """What is missing, where it goes, and the exact links to fetch it.

    Written for somebody who has turned downloading off, or is behind a proxy
    that will not let the node reach the Hub. Every path in it is the real path
    on this machine, and every link is one a browser or a download manager can
    take as it stands.
    """
    from .constants import (
        LM_REPO, REPACK_BF16_PATH, REPACK_INT8_PATH, VAE_LEGACY_REPO, VAE_REPO,
    )

    lines = ["YuE2 weights are not on this machine yet.", ""]
    if variant != "legacy":
        name = REPACK_INT8_NAME if quantization == "int8" else REPACK_BF16_NAME
        repo_path = REPACK_INT8_PATH if quantization == "int8" else REPACK_BF16_PATH
        size = REPACK_INT8_BYTES if quantization == "int8" else REPACK_BF16_BYTES
        lines += [
            "One file is enough ({:.2f} GB), and it is the same file ComfyUI's own "
            "YuE2 nodes use:".format(size / 1024 ** 3),
            "",
            "  " + _link(REPACK_REPO, repo_path),
            "  -> " + os.path.join(_checkpoints_root(), name),
            "",
            "Or the three files as m-a-p released them:",
            "",
        ]
    else:
        lines += ["The legacy decoder comes only as the released files:", ""]

    root = _expected_root()
    vae_dirname = VAE_LEGACY_DIRNAME if variant == "legacy" else VAE_DIRNAME
    vae_repo = VAE_LEGACY_REPO if variant == "legacy" else VAE_REPO
    wanted = [
        ("lm", LM_REPO, WEIGHTS_NAME, os.path.join(root, LM_DIRNAME, WEIGHTS_NAME)),
        ("merges", LM_REPO, MERGES_NAME, os.path.join(root, LM_DIRNAME, MERGES_NAME)),
        ("vae", vae_repo, WEIGHTS_NAME, os.path.join(root, vae_dirname, WEIGHTS_NAME)),
    ]
    for key, repo, repo_path, destination in wanted:
        if found.get(key):
            continue
        lines += ["  " + _link(repo, repo_path), "  -> " + destination, ""]

    if roots:
        where = "1 place" if len(roots) == 1 else str(len(roots)) + " places"
        lines += ["Looked in " + where + ", including:", ""]
        lines += ["  " + path for path in roots[:6]]
    else:
        lines += ["There was nowhere to look: no ComfyUI model folders were found."]
    lines += [
        "",
        "Set " + paths.ENV_ROOT + " to point at a folder you keep them in, if it is "
        "none of the above.",
    ]
    return "\n".join(lines)


def find_sheetsage(roots=None) -> str:
    """SheetSage2's file, by its published name first and by its tensors second, or ""."""
    roots = paths.sheetsage_roots() if roots is None else roots
    for root in roots:
        for home in _homes(root):
            candidate = os.path.join(home, SHEETSAGE_NAME)
            if os.path.isfile(candidate):
                return candidate
    for root in roots:
        for home in _homes(root):
            try:
                entries = sorted(os.listdir(home))
            except OSError:
                continue
            for entry in entries:
                if entry.lower().endswith(".safetensors"):
                    candidate = os.path.join(home, entry)
                    if identify(candidate) == "sheetsage":
                        return candidate
    return ""


def _audio_encoders_root() -> str:
    override = os.environ.get(paths.ENV_ROOT)
    if override:
        return os.path.join(override, "audio_encoders")
    try:
        return paths.audio_encoders_root()
    except Exception:
        return os.path.join("ComfyUI", "models", "audio_encoders")


def sheetsage_missing_message(roots: list) -> str:
    """Where SheetSage2 goes and the link to it, for somebody who turned downloading off."""
    lines = [
        "SheetSage2, the model that reads a recording into a score, is not on this machine yet.",
        "",
        "It is one file ({:.2f} GB), the same one ComfyUI's own audio encoder loader reads:"
        .format(SHEETSAGE_BYTES / 1024 ** 3),
        "",
        "  " + _link(REPACK_REPO, SHEETSAGE_PATH),
        "  -> " + os.path.join(_audio_encoders_root(), SHEETSAGE_NAME),
        "",
    ]
    if roots:
        where = "1 place" if len(roots) == 1 else str(len(roots)) + " places"
        lines += ["Looked in " + where + ", including:", ""]
        lines += ["  " + path for path in roots[:6]]
    else:
        lines += ["There was nowhere to look: no ComfyUI model folders were found."]
    lines += ["", "Its weights are licensed CC BY-NC 4.0, like YuE2's."]
    return "\n".join(lines)


def is_asr_folder(folder: str) -> bool:
    """A folder holding Qwen3-ASR-1.7B's weights and tokenizer: known by size, or by the projector's shape."""
    weights = os.path.join(folder, WEIGHTS_NAME)
    if not (os.path.isfile(weights) and os.path.isfile(os.path.join(folder, ASR_TOKENIZER_NAME))):
        return False
    try:
        if os.path.getsize(weights) == ASR_BYTES:
            return True
        from .loader import read_header

        entry = read_header(weights).get(ASR_MARKER)
    except Exception:
        log.debug("[yue2_comfy.discovery] cannot read %s", weights, exc_info=True)
        return False
    return isinstance(entry, dict) and entry.get("shape") == ASR_MARKER_SHAPE


def find_asr(roots=None) -> str:
    """The folder of Qwen3-ASR-1.7B's weights, or ""."""
    roots = paths.asr_roots() if roots is None else roots
    for root in roots:
        for home in _homes(root):
            if is_asr_folder(home):
                return home
    return ""


def asr_missing_message(roots: list) -> str:
    """Where the speech model goes and the links to it, for somebody who turned downloading off."""
    lines = [
        "Qwen3-ASR-1.7B, the speech model that recognises the sung words, is not on this machine yet.",
        "",
        "It is three files from Qwen's own release, the weights {:.2f} GB of them, kept together in one folder:"
        .format(ASR_BYTES / 1024 ** 3),
        "",
    ]
    lines += ["  " + _link(ASR_REPO, name) for name in ASR_FILES]
    lines += ["  -> " + os.path.join(_expected_root(), ASR_DIRNAME), ""]
    if roots:
        where = "1 place" if len(roots) == 1 else str(len(roots)) + " places"
        lines += ["Looked in " + where + ", including:", ""]
        lines += ["  " + path for path in roots[:6]]
    else:
        lines += ["There was nowhere to look: no ComfyUI model folders were found."]
    lines += ["", "Its weights are licensed Apache-2.0."]
    return "\n".join(lines)


def find_vocals(roots=None) -> str:
    """The voice separator's weights: the released file by name and size, then a conversion by its tensors, or "".

    The released file is preferred when both are on the machine: it is the one
    the pack was checked against, in full precision.
    """
    roots = paths.search_roots() if roots is None else roots
    for root in roots:
        for home in _homes(root):
            candidate = os.path.join(home, VOCALS_NAME)
            try:
                if os.path.getsize(candidate) == VOCALS_BYTES:
                    return candidate
            except OSError:
                continue
    for root in roots:
        for home in _homes(root):
            try:
                entries = sorted(os.listdir(home))
            except OSError:
                continue
            for entry in entries:
                candidate = os.path.join(home, entry)
                lowered = entry.lower()
                if lowered.endswith(".ckpt") and _size(candidate) == VOCALS_BYTES:
                    return candidate
                if lowered.endswith(".safetensors") and identify(candidate) == "vocals":
                    return candidate
    return ""


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def vocals_missing_message(roots: list) -> str:
    """Where the voice separator goes and the link to it, for somebody who turned downloading off."""
    lines = [
        "Mel-Band RoFormer, the model that separates the voice from a song, is not on this machine yet.",
        "",
        "It is one file ({:.2f} GB):".format(VOCALS_BYTES / 1024 ** 3),
        "",
        "  " + _link(VOCALS_REPO, VOCALS_NAME, VOCALS_REVISION),
        "  -> " + os.path.join(_expected_root(), VOCALS_NAME),
        "",
    ]
    if roots:
        where = "1 place" if len(roots) == 1 else str(len(roots)) + " places"
        lines += ["Looked in " + where + ", including:", ""]
        lines += ["  " + path for path in roots[:6]]
    else:
        lines += ["There was nowhere to look: no ComfyUI model folders were found."]
    lines += ["", "Its weights are licensed MIT."]
    return "\n".join(lines)
