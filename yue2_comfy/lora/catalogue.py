"""The LoRA files on this machine that are for YuE2, by the names the node shows.

The folders are ComfyUI's own LoRA folders (``paths.lora_roots``), so a file
put where ComfyUI's LoraLoader finds it is found here too, under the same
relative name, always written with forward slashes: a workflow saved on Windows
names the same file on Linux.

Only files whose header lands on YuE2 are listed. Measured on 2026-09-19: the 45
image LoRAs of a working install (43 GB) were read and turned away in 0.46 s
the first time and 0.16 s after, and none was taken for a YuE2 one. What was
learnt about a file is kept by its size and modification time, in memory and in
``ComfyUI/user/yue2_comfy/lora_headers.json``, so a folder of a thousand
LoRAs costs its reading once, not every time the list opens. That file keys
each verdict by a hash of the path, size and time, never by the path itself: a
folder someone keeps private leaves no names behind in it.

A file that is for YuE2 but cannot be folded -- a part that lands nowhere, an
adapter that brings modules of its own -- is listed with the reason, because
the person looking for it deserves to know why it will not run.

Nothing here imports torch.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import threading

from .. import paths
from . import formats

log = logging.getLogger(__name__)

SUFFIX = ".safetensors"

CACHE_NAME = "lora_headers.json"
CACHE_FORMAT = 1
CACHE_LIMIT = 5000
"""Verdicts the file keeps. A file replaced or deleted leaves a verdict nobody
asks for again, so past this the oldest go first."""

HASH_CHUNK = 1 << 22

_LOCK = threading.RLock()
_MEMORY: dict = {}
_DISK: dict = {"loaded": False, "entries": {}, "dirty": False}
_HASHES: dict = {}


@dataclasses.dataclass(frozen=True)
class Entry:
    """One LoRA file for YuE2: the name the node shows, where it is, and what it holds."""

    name: str
    path: str
    summary: dict

    @property
    def usable(self) -> bool:
        return not self.summary.get("problem")


def stamp(path: str):
    """A file's identity on disk -- path, size and modification time -- or None when it is gone."""
    try:
        info = os.stat(path)
    except OSError:
        return None
    return "{}|{}|{}".format(os.path.normcase(os.path.abspath(path)), info.st_size,
                             info.st_mtime_ns)


def _cache_path() -> str:
    return os.path.join(paths.user_dir(), CACHE_NAME)


def _disk_key(mark: str) -> str:
    return hashlib.sha256(mark.encode("utf-8")).hexdigest()[:32]


def _load_disk() -> None:
    """Read the kept verdicts once per process. A missing or damaged file is an empty one."""
    if _DISK["loaded"]:
        return
    _DISK["loaded"] = True
    try:
        with open(_cache_path(), "r", encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return
    if isinstance(saved, dict) and saved.get("format") == CACHE_FORMAT:
        entries = saved.get("entries")
        if isinstance(entries, dict):
            _DISK["entries"] = entries


def _save_disk() -> None:
    """Write the verdicts back when something was learnt, never failing the caller."""
    if not _DISK["dirty"]:
        return
    kept = _DISK["entries"]
    while len(kept) > CACHE_LIMIT:
        kept.pop(next(iter(kept)))
    target = _cache_path()
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        temporary = target + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({"format": CACHE_FORMAT, "entries": _DISK["entries"]}, handle)
        os.replace(temporary, target)
        _DISK["dirty"] = False
    except OSError:
        log.debug("[yue2_comfy.lora] could not keep what the LoRA headers said", exc_info=True)


def verdict(path: str):
    """What a file is, as a summary dict, or None when it is not a LoRA for YuE2.

    Kept per file identity, so a file replaced under the same name is read
    again, and the ones already read cost nothing.
    """
    mark = stamp(path)
    if mark is None:
        return None
    key = _disk_key(mark)
    with _LOCK:
        if mark in _MEMORY:
            return _MEMORY[mark]
        _load_disk()
        if key in _DISK["entries"]:
            found = _DISK["entries"][key]
            _MEMORY[mark] = found
            return found
    try:
        found = formats.summary(formats.read(path))
    except formats.NotYuE2:
        found = None
    except (OSError, ValueError) as error:
        log.debug("[yue2_comfy.lora] %s could not be read: %s", path, error)
        return None
    with _LOCK:
        _MEMORY[mark] = found
        _DISK["entries"][key] = found
        _DISK["dirty"] = True
    return found


def roots() -> list:
    """The folders to look in; see ``paths.lora_roots``."""
    return paths.lora_roots()


def _walk(root: str) -> list:
    """Every .safetensors under one root, as (name, path), the name relative and with slashes."""
    found = []
    for folder, directories, files in os.walk(root, followlinks=True):
        directories.sort()
        for file_name in sorted(files):
            if file_name.lower().endswith(SUFFIX):
                path = os.path.join(folder, file_name)
                name = os.path.relpath(path, root).replace(os.sep, "/")
                found.append((name, path))
    return found


def entries() -> list:
    """Every LoRA file for YuE2 in the LoRA folders, sorted by name.

    A name found under two roots is the first root's, as in ComfyUI's own list.
    """
    seen, listed = set(), []
    for root in roots():
        for name, path in _walk(root):
            key = name.lower()
            if key in seen:
                continue
            found = verdict(path)
            if found is None:
                continue
            seen.add(key)
            listed.append(Entry(name, path, found))
    with _LOCK:
        _save_disk()
    listed.sort(key=lambda entry: entry.name.lower())
    return listed


def listing() -> list:
    """The list the node's picker and its rows read, as plain JSON."""
    return [dict(entry.summary, name=entry.name) for entry in entries()]


def _clean(name: str) -> str:
    return str(name or "").strip().replace("\\", "/")


def find(name: str) -> Entry:
    """The entry a row names, or FileNotFoundError saying what is there instead."""
    wanted = _clean(name)
    listed = entries()
    for entry in listed:
        if entry.name == wanted:
            return entry
    for entry in listed:
        if entry.name.lower() == wanted.lower():
            return entry
    known = ", ".join(entry.name for entry in listed[:12]) or "none"
    raise FileNotFoundError(
        "'{}' is not in the LoRA folders any more, or is not a LoRA for YuE2. What is there "
        "now: {}{}.\n\nPut the file back into ComfyUI's models/loras, or pick another one on "
        "the 'YuE2 LoRA' node.".format(wanted, known, " ..." if len(listed) > 12 else ""))


def identity(path: str) -> str:
    """The SHA-256 of a file, kept per file identity: what the song memory writes down."""
    mark = stamp(path)
    with _LOCK:
        if mark is not None and mark in _HASHES:
            return _HASHES[mark]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(HASH_CHUNK)
            if not block:
                break
            digest.update(block)
    value = digest.hexdigest()
    with _LOCK:
        if mark is not None:
            _HASHES[mark] = value
    return value


def forget() -> None:
    """Drop what was learnt in memory, for tests and for a folder that changed under us."""
    with _LOCK:
        _MEMORY.clear()
        _HASHES.clear()
        _DISK.update(loaded=False, entries={}, dirty=False)
