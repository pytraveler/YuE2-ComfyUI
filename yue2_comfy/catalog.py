"""The writer models the pack offers for download, in a list the person can edit.

YuE2 Write Song and YuE2 Transcribe run one GGUF language model. Every such file
in ComfyUI's model folders is offered already (``llm.catalogue``); this adds the
models that are *not* on the machine yet, each named with its download size and
the card it needs, and fetched the first time it is picked. Without it the list
held only what happened to be on disk -- on most machines the one 4B the pack
downloads for itself -- and a video review of the pack took that one 4B for all
it could use.

**The packaged ``writers.json`` is a seed, not the live file.** The live list is
``ComfyUI/user/yue2_comfy/writers.json``; until something is written there, the
seed is read as it stands, so merely listing the models creates no file. The
first edit writes the whole list there, and from then on an update of the pack
never overwrites it.

**New entries of the pack are merged in, though.** The live file records, under
``seed_offered``, every name the packaged list has put in front of this
installation; an update adds exactly the seed's names that are neither in the
file nor already offered. A model somebody deleted stays deleted, a renamed one
does not come back under its old name, and a model added to the pack later
arrives. The idea and its reasons come from the MiniMax rewriter's catalogue,
where "we will not overwrite your list" had turned into "you will never see a
model added after you installed".

**A live file that does not parse is left alone.** The seed is offered instead,
``problem()`` says why, and every write is refused until the file parses again:
saving over it would replace the person's own entries with the seed's.

**The widget says where each entry's file is**: ``name (on disk, VRAM note)``
or ``name (download N GB, VRAM note)``. Without it the list's models and the
files on the machine looked alike, and nothing said which pick would start a
20 GB download (the person's own check, 2026-09-26). That label is also the
value a workflow saves, so it changes when the file arrives. ``find`` knows
an entry by every label it can have and by its name alone, so a saved label
still reaches its model through ``llm.resolve``, and the browser moves an
open graph onto the new label (web/js/yue2_writers.js). The note is not part
of the label and can change freely.

Standard library only.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass

from . import paths

log = logging.getLogger(__name__)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = "writers.json"
SEED_FILE = os.path.join(PACKAGE_DIR, FILE_NAME)
SECTION = "writers"
OFFERED_KEY = "seed_offered"
BACKUP_SUFFIX = ".bak"
SUFFIX = ".gguf"

NAME_LIMIT = 80
VRAM_LIMIT = 40
NOTE_LIMIT = 200
LARGEST_GB = 1000.0
RESERVED = ("auto",)
RESERVED_PREFIXES = ("!!", "ollama:")
REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
ON_DISK = "on disk"
TO_DOWNLOAD = "download"
NOT_FOUND = "not found"
TAIL = re.compile(r" \([^()]*\)$")

_CACHE: dict = {"key": None, "data": None, "problem": ""}
_SEED: dict = {"key": None, "data": None}


class CatalogWriteError(RuntimeError):
    """An edit the list refuses, with the reason in words a person can act on."""


@dataclass(frozen=True)
class Entry:
    name: str
    file: str
    repo: str = ""
    download_gb: float = 0.0
    vram: str = ""
    note: str = ""

    @property
    def label(self) -> str:
        """Name, then size and card in brackets: the label without where the file is.

        It tells entries apart, and the widget showed it before it said where
        each file was, so a workflow saved then carries it.
        """
        parts = []
        if self.download_gb:
            parts.append("{:g} GB".format(self.download_gb))
        if self.vram:
            parts.append(self.vram)
        return self.name + (" (" + ", ".join(parts) + ")" if parts else "")

    def shown(self, here: bool) -> str:
        """What the model widget shows and a workflow saves: the name, where the file is, and the card.

        ``here`` is whether the file is on this machine. A Hub entry that is
        not says how much its first run downloads; a local one that is not
        says so, because nothing will fetch it.
        """
        if here:
            where = ON_DISK
        elif self.local:
            where = NOT_FOUND
        else:
            where = TO_DOWNLOAD + (" {:g} GB".format(self.download_gb) if self.download_gb else "")
        return self.name + " (" + ", ".join([where] + ([self.vram] if self.vram else [])) + ")"

    @property
    def local(self) -> bool:
        """An entry naming a file on this machine rather than one on the Hub."""
        return not self.repo


def live_file() -> str:
    """Where the live list is, whether or not it exists yet."""
    return os.path.join(paths.user_dir(), FILE_NAME)


def forget() -> None:
    """Drop what was read, so the next call reads both files again."""
    _CACHE.update(key=None, data=None, problem="")
    _SEED.update(key=None, data=None)


def problem() -> str:
    """Why the live list is not being read, or "" when it is (or does not exist)."""
    _data()
    return _CACHE["problem"]


def _identity(path: str):
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (os.path.normcase(os.path.abspath(path)), stat.st_size, stat.st_mtime_ns)


def _read(path: str) -> tuple:
    """``(data, "")``, or ``({}, why)`` when the file cannot be read as a list."""
    try:
        with open(path, "rb") as handle:
            data = json.loads(handle.read().decode("utf-8"))
    except OSError as error:
        return {}, FILE_NAME + " could not be read: " + str(error)
    except (UnicodeDecodeError, ValueError) as error:
        return {}, FILE_NAME + " is not valid JSON: " + str(error)
    if not isinstance(data, dict):
        return {}, FILE_NAME + " should hold an object with a '" + SECTION + "' list"
    if SECTION in data and not isinstance(data[SECTION], list):
        return {}, "'" + SECTION + "' in " + FILE_NAME + " should be a list"
    return data, ""


def _seed() -> dict:
    """The packaged list, read once per version of the file."""
    key = _identity(SEED_FILE)
    if _SEED["key"] != key or _SEED["data"] is None:
        data, why = _read(SEED_FILE)
        if why:
            log.warning("[yue2_comfy.catalog] the packaged list is unusable: %s", why)
        _SEED["key"] = key
        _SEED["data"] = data
    return _SEED["data"]


def _name(raw) -> str:
    return raw.get("name").strip() if isinstance(raw, dict) and isinstance(raw.get("name"), str) else ""


def seed_names() -> list:
    return [name for name in (_name(raw) for raw in _seed().get(SECTION) or []) if name]


def merge(live: dict, seed: dict) -> tuple:
    """``(merged, added names)``: the seed's entries this installation has never been offered, added.

    Pure. A list removed from the file on purpose (``seed_offered`` present,
    ``writers`` gone) stays removed; the offered names are recorded either way.
    """
    seed_list = [raw for raw in seed.get(SECTION) or [] if _name(raw)]
    offered = live.get(OFFERED_KEY)
    offered = [name for name in offered if isinstance(name, str)] if isinstance(offered, list) else None
    had_list = isinstance(live.get(SECTION), list)
    current = list(live.get(SECTION)) if had_list else []
    present = {_name(raw).casefold() for raw in current}
    already = set(offered or [])
    if not had_list and offered is not None:
        fresh = []
    else:
        fresh = [raw for raw in seed_list if _name(raw).casefold() not in present and _name(raw) not in already]
    recorded = list(offered or []) + [_name(raw) for raw in seed_list if _name(raw) not in already]
    if not fresh and offered is not None and recorded == offered:
        return live, []
    merged = dict(live)
    if had_list or fresh:
        merged[SECTION] = current + [dict(raw) for raw in fresh]
    merged[OFFERED_KEY] = recorded
    return merged, [_name(raw) for raw in fresh]


def _write(path: str, data: dict, backup: bool) -> None:
    """The list written whole or not at all; the old file kept as ``.bak`` only when ``backup``.

    The backup is for a merge -- the pack changing the file behind somebody's
    back -- and not for their own edits in the window, which would otherwise
    spend it on every click.
    """
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=FILE_NAME, suffix=".part", dir=folder)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        if backup and os.path.isfile(path):
            try:
                with open(path, "rb") as old, open(path + BACKUP_SUFFIX, "wb") as kept:
                    kept.write(old.read())
            except OSError:
                log.debug("[yue2_comfy.catalog] no backup of %s", path, exc_info=True)
        os.replace(temporary, path)
    except OSError:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise
    finally:
        _CACHE["key"] = None


def _data() -> dict:
    """The list as it is to be used: the live file merged with the seed, or the seed itself."""
    path = live_file()
    key = _identity(path)
    if key is None:
        _CACHE.update(key=None, data=None, problem="")
        return _seed()
    if _CACHE["key"] == key and _CACHE["data"] is not None:
        return _CACHE["data"]
    data, why = _read(path)
    if why:
        log.warning("[yue2_comfy.catalog] %s; the packaged list is offered instead", why)
        _CACHE.update(key=key, data=_seed(), problem=why)
        return _seed()
    merged, added = merge(data, _seed())
    if merged is not data:
        try:
            _write(path, merged, backup=True)
            for name in added:
                log.info("[yue2_comfy.catalog] added '%s' from the pack's list", name)
            key = _identity(path)
        except OSError:
            log.warning("[yue2_comfy.catalog] could not write the merged list to %s", path, exc_info=True)
    _CACHE.update(key=key, data=merged, problem="")
    return merged


def _entry(raw) -> Entry:
    """An entry as the widget uses it, or None for one it cannot use."""
    if not isinstance(raw, dict):
        return None
    name = _name(raw)
    file = raw.get("file").strip() if isinstance(raw.get("file"), str) else ""
    repo = raw.get("repo").strip() if isinstance(raw.get("repo"), str) else ""
    if not name or not file:
        return None
    try:
        size = float(raw.get("download_gb") or 0)
    except (TypeError, ValueError):
        size = 0.0
    return Entry(name=name, file=file, repo=repo, download_gb=size if size > 0 else 0.0,
                 vram=str(raw.get("vram") or "").strip(), note=str(raw.get("note") or "").strip())


def entry(raw):
    """One entry as written in the file, as the widget would use it, or None when it cannot be."""
    return _entry(raw)


def entries() -> list:
    """The listed models the widget offers, in the file's order; malformed ones are skipped."""
    found = []
    for raw in _data().get(SECTION) or []:
        entry = _entry(raw)
        if entry is None:
            log.debug("[yue2_comfy.catalog] skipped an unusable entry: %r", raw)
            continue
        found.append(entry)
    return found


def stem(label: str) -> str:
    """A label without its last bracketed part, where the size, the place and the card go."""
    return TAIL.sub("", label).strip()


def find(choice: str):
    """The entry a widget value names: by any label it can have, then by its name.

    The name is tried whole and then without the last bracketed part, so
    'Qwen (abliterated)' and 'Qwen (abliterated) (on disk)' find the same
    entry, and a label whose size, place or card has changed since the
    workflow was saved still finds its model.
    """
    wanted = (choice or "").strip()
    if not wanted:
        return None
    listed = entries()
    for entry in listed:
        if wanted in (entry.shown(True), entry.shown(False), entry.label):
            return entry
    for key in (wanted.casefold(), stem(wanted).casefold()):
        for entry in listed:
            if entry.name.casefold() == key:
                return entry
    return None


def raw_entries() -> list:
    """The entries as written, the ones ``entries`` skips included, for the window."""
    return [copy.deepcopy(raw) for raw in _data().get(SECTION) or [] if isinstance(raw, dict)]


def _is_network(path: str) -> bool:
    text = path.replace("/", "\\") if os.name == "nt" else path
    if text.startswith("\\\\?\\") or text.startswith("\\\\.\\"):
        return False
    return text.startswith("\\\\") or (os.name == "nt" and path.startswith("//"))


def _line(value, limit: int, what: str) -> str:
    text = "" if value is None else str(value).strip()
    if "\n" in text or "\r" in text:
        raise CatalogWriteError(what + " has to fit on one line.")
    if len(text) > limit:
        raise CatalogWriteError(what + " is longer than {} characters.".format(limit))
    return text


def clean_entry(raw: dict) -> dict:
    """An entry fit to be written, or CatalogWriteError saying what to change.

    A Hub entry names a repository and the .gguf inside it; a local one names
    no repository and a full path. A network path is refused: the window
    reaches this over ComfyUI's API, and merely looking at a UNC path is an
    authentication attempt against the host it names. A path typed into the
    file by hand is not checked here.
    """
    if not isinstance(raw, dict):
        raise CatalogWriteError("An entry is a set of fields.")
    name = _line(raw.get("name"), NAME_LIMIT, "The name")
    if not name:
        raise CatalogWriteError("Give the model a name: it is what the list shows.")
    lowered = name.casefold()
    if lowered in RESERVED or lowered.startswith(RESERVED_PREFIXES) or lowered.endswith(SUFFIX):
        raise CatalogWriteError(
            "'" + name + "' would be mistaken for another kind of entry in the list. "
            "Name the model rather than its file, for example 'Qwen3.5-9B Q4_K_M'.")
    repo = _line(raw.get("repo"), 200, "The repository")
    file = _line(raw.get("file"), 400, "The file")
    if _is_network(repo) or _is_network(file):
        raise CatalogWriteError(
            "Network paths are not accepted here. Copy the file into ComfyUI/models/LLM, "
            "or write the path into writers.json by hand.")
    if not file:
        raise CatalogWriteError("Name the .gguf file.")
    if not file.lower().endswith(SUFFIX):
        raise CatalogWriteError("The file has to be a .gguf: '" + file + "' is not.")
    if repo:
        if not REPO_ID.match(repo):
            raise CatalogWriteError(
                "'" + repo + "' is not a Hugging Face repository id; it looks like 'owner/name'.")
        parts = file.replace("\\", "/").split("/")
        if file.startswith(("/", "\\")) or ":" in file or any(part in ("", ".", "..") for part in parts):
            raise CatalogWriteError(
                "With a repository, the file is its path inside the repository, such as "
                "'Qwen3.5-9B-Q4_K_M.gguf'.")
    elif not os.path.isabs(file):
        raise CatalogWriteError(
            "Without a repository the file is a full path on this machine. For a file in "
            "ComfyUI/models/LLM nothing needs listing: it is offered already.")
    size = raw.get("download_gb")
    if size in (None, ""):
        size = 0.0
    try:
        size = float(size)
    except (TypeError, ValueError):
        raise CatalogWriteError("The download size is a number of gigabytes.") from None
    if not 0 <= size < LARGEST_GB:
        raise CatalogWriteError("The download size is a number of gigabytes, 0 or more.")
    out = {"name": name}
    if repo:
        out["repo"] = repo
    out["file"] = file
    if size:
        out["download_gb"] = round(size, 2)
    vram = _line(raw.get("vram"), VRAM_LIMIT, "The VRAM note")
    if "(" in vram or ")" in vram:
        raise CatalogWriteError("The VRAM note cannot hold brackets: the list shows it inside its own.")
    note = _line(raw.get("note"), NOTE_LIMIT, "The note")
    if vram:
        out["vram"] = vram
    if note:
        out["note"] = note
    return out


def label_of(raw: dict) -> str:
    entry = _entry(raw)
    return entry.label if entry is not None else _name(raw)


def writable() -> str:
    """The live file's path, or CatalogWriteError when writing it would lose somebody's list."""
    why = problem()
    if why:
        raise CatalogWriteError(
            why + ". It has to parse before it can be edited here, or saving would replace "
            "your own entries with the pack's list. Mend it by hand, or delete it to start "
            "again from the pack's list.")
    return live_file()


def _mutable() -> tuple:
    """``(path, a deep copy of the list)`` to edit; the seed with its names recorded when there is no file yet."""
    path = writable()
    data = copy.deepcopy(_data())
    if not isinstance(data.get(SECTION), list):
        data[SECTION] = []
    if not isinstance(data.get(OFFERED_KEY), list):
        data[OFFERED_KEY] = seed_names()
    return path, data


def _position(listed: list, name: str) -> int:
    for index, raw in enumerate(listed):
        if _name(raw).casefold() == name.strip().casefold():
            return index
    return -1


def _refuse_clash(listed: list, entry: dict, skip: int = -1) -> None:
    """Two entries with one name cannot be told apart, and two with one label shadow each other."""
    label = label_of(entry)
    for index, raw in enumerate(listed):
        if index == skip:
            continue
        if _name(raw).casefold() == entry["name"].casefold():
            raise CatalogWriteError("The list already has a model called '" + entry["name"] + "'.")
        if label_of(raw) == label:
            raise CatalogWriteError("'" + label + "' is already in the list under another entry.")


def add(raw: dict) -> dict:
    """The entry added at the end of the list, as written."""
    entry = clean_entry(raw)
    path, data = _mutable()
    _refuse_clash(data[SECTION], entry)
    data[SECTION].append(entry)
    _write(path, data, backup=False)
    return entry


def update(name: str, raw: dict) -> dict:
    """The entry called ``name`` replaced, in place; renaming it is allowed."""
    entry = clean_entry(raw)
    path, data = _mutable()
    index = _position(data[SECTION], name)
    if index < 0:
        raise CatalogWriteError(
            "'" + name + "' is no longer in the list; something else changed the file. "
            "Open the list again.")
    _refuse_clash(data[SECTION], entry, skip=index)
    data[SECTION][index] = entry
    _write(path, data, backup=False)
    return entry


def remove(name: str) -> bool:
    """Whether the entry called ``name`` was there and is gone now."""
    path, data = _mutable()
    index = _position(data[SECTION], name)
    if index < 0:
        return False
    del data[SECTION][index]
    _write(path, data, backup=False)
    return True


def materialize() -> str:
    """The live file's path, written from the pack's list first when there is none yet.

    For 'Open writers.json': until something is edited the live file does not
    exist, and opening the pack's own copy instead would send hand edits where
    the next update overwrites them. A file that is there is left as it is,
    parsing or not -- opening it to mend it is the point.
    """
    path = live_file()
    if os.path.isfile(path):
        return path
    path, data = _mutable()
    _write(path, data, backup=False)
    return path


def restorable() -> list:
    """The pack's own entries that are missing from the list."""
    present = {_name(raw).casefold() for raw in _data().get(SECTION) or []}
    return [name for name in seed_names() if name.casefold() not in present]


def restore_packaged() -> list:
    """The pack's missing entries put back at the end, in the pack's order; their names."""
    path, data = _mutable()
    present = {_name(raw).casefold() for raw in data[SECTION]}
    back = [dict(raw) for raw in _seed().get(SECTION) or [] if _name(raw) and _name(raw).casefold() not in present]
    if not back:
        return []
    data[SECTION].extend(back)
    _write(path, data, backup=False)
    return [_name(raw) for raw in back]
