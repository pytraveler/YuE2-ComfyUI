"""Explicit local YuE2 checkpoints and the source a saved performance belongs to.

The picker uses relative names; absolute paths stay in the local plan and song
store. A remembered source is checked before reuse, so a missing or replaced
file cannot quietly turn a plan or a retake into another model's performance.
"""

from __future__ import annotations

import hashlib
import os

from . import discovery, paths

SOURCE = "checkpoint_source"
KINDS = ("lm", "repack", "repack_int8")


def roots() -> list:
    return paths.search_roots()


def catalogue() -> dict:
    """Relative picker labels mapped to compatible files, including nested checkpoints."""
    grouped = {}
    seen = set()
    for root in roots():
        for home in discovery._homes(root):
            try:
                entries = sorted(os.listdir(home))
            except OSError:
                continue
            for entry in entries:
                if not entry.lower().endswith(".safetensors"):
                    continue
                path = os.path.join(home, entry)
                key = os.path.normcase(os.path.realpath(path))
                if key in seen or discovery.identify(path) not in KINDS:
                    continue
                seen.add(key)
                name = os.path.basename(os.path.normpath(root)) + "/" + os.path.relpath(path, root).replace(os.sep, "/")
                grouped.setdefault(name, []).append(path)
    found = {}
    for name, candidates in grouped.items():
        for path in candidates:
            label = name
            if len(candidates) > 1:
                identity = os.path.normcase(os.path.realpath(path))
                suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
                label += " [" + suffix + "]"
            found[label] = path
    return found


def choices() -> list:
    return ["auto"] + sorted(catalogue(), key=str.lower)


def _stamp(path: str):
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return [stat.st_size, stat.st_mtime_ns]


def _inside(path: str, allowed: list) -> bool:
    """A saved source may only name files under configured roots, including HF snapshot symlinks."""
    if not isinstance(path, str) or paths.is_network_path(path):
        return False
    resolved = os.path.normcase(os.path.abspath(path))
    for root in allowed:
        parent = os.path.normcase(os.path.abspath(root))
        try:
            if os.path.commonpath([resolved, parent]) == parent:
                return True
        except ValueError:
            continue
    return False


def remember(settings: dict, files) -> None:
    """Attach the resolved files, their identities and a public name to this run."""
    source = files._asdict()
    main = files.repack or files.lm
    settings[SOURCE] = {
        "choice": settings.get("checkpoint", "auto"),
        "files": source,
        "stamps": {key: _stamp(path) for key, path in source.items() if path},
        "name": os.path.basename(main),
        "vae_variant": settings.get("vae", "standard"),
    }


def pinned(settings: dict):
    """The exact files remembered by a plan, or None for a fresh choice."""
    source = settings.get(SOURCE) or {}
    if source.get("choice") != settings.get("checkpoint", "auto"):
        return None
    files = source.get("files")
    if not files:
        return None
    allowed = roots()
    for key, path in files.items():
        if path and not _inside(path, allowed):
            raise ValueError("The checkpoint saved with this song or plan is outside the configured model folders. "
                             "Select it again in YuE2 Options and make a new plan or song.")
        stamp = _stamp(path) if path else None
        if path and (stamp is None or stamp != source.get("stamps", {}).get(key)):
            raise ValueError(
                "The checkpoint files saved with this song or plan are missing or changed: "
                + source.get("name", "YuE2") + ". Select the checkpoint again in YuE2 Options "
                "and make a new plan or song."
            )
    return discovery.Files(**files)


def selected(settings: dict, progress=None):
    """Resolve an explicit checkpoint, fetching only missing supporting files when allowed."""
    from . import download
    from .constants import LM_REPO, MERGES_NAME

    kept = pinned(settings)
    if kept is not None:
        if settings.get("vae", "standard") == settings[SOURCE].get("vae_variant", "standard"):
            return kept
        path = kept.repack or kept.lm
    else:
        choice = settings.get("checkpoint", "auto")
        if choice == "auto":
            return None
        path = catalogue().get(choice)
        if not path:
            raise ValueError("The selected YuE2 checkpoint is not available: " + str(choice)
                             + ". Refresh the node definitions and select a local checkpoint.")
    kind = discovery._identify_by_header(path)
    if kind not in KINDS:
        raise ValueError("The selected file is not a supported YuE2 checkpoint: " + os.path.basename(path))
    variant = settings.get("vae", "standard")
    if kind != "lm" and variant != "legacy":
        return discovery.Files(repack=path)
    roots = paths.search_roots()
    found = discovery._layout_pass(roots, variant)
    if kind == "lm":
        found["lm"] = path
        adjacent = os.path.join(os.path.dirname(path), MERGES_NAME)
        if os.path.isfile(adjacent):
            found["merges"] = adjacent
    discovery._sweep(roots, found, variant)
    if not found["vae"]:
        if settings.get("download") == "off":
            raise ValueError("The selected checkpoint needs the " + variant + " YuE2 VAE. "
                             "Install it or enable downloads in YuE2 Options.")
        download.fetch_decoder(variant, progress)
    if kind == "lm" and not found["merges"]:
        if settings.get("download") == "off":
            raise ValueError("The selected checkpoint needs qwen.tiktoken. Put it beside the weights "
                             "or enable downloads in YuE2 Options.")
        target = os.path.join(paths.local_dir_for_repo(LM_REPO), MERGES_NAME)
        download.fetch(LM_REPO, {MERGES_NAME: target}, "Downloading the YuE2 vocabulary", progress)
    if not found["vae"] or (kind == "lm" and not found["merges"]):
        discovery._sweep(paths.search_roots(), found, variant)
    if not found["vae"] or (kind == "lm" and not found["merges"]):
        raise ValueError("Supporting files for the selected YuE2 checkpoint could not be found.")
    if kind != "lm":
        return discovery.Files(repack=path, vae=found["vae"])
    return discovery.Files(lm=path, vae=found["vae"], merges=found["merges"])


def overlay(settings: dict, options) -> None:
    """An options node at auto preserves a plan's explicit checkpoint choice."""
    if not options:
        return
    choice = settings.get("checkpoint", "auto")
    settings.update(options)
    if options.get("checkpoint", "auto") == "auto" and settings.get(SOURCE):
        settings["checkpoint"] = choice
