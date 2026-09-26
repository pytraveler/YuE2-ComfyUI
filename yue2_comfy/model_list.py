"""The server half of the Model list window on YuE2 Write Song and YuE2 Transcribe.

The list itself is ``catalog`` and what is on this machine is ``llm``; this puts
the two together the way the window draws them and answers its buttons. Every
function returns a plain dict for ``routes``. A refusal is a
``catalog.CatalogWriteError`` whose words the window prints as they are.

The window is the MiniMax rewriter's Model list (0.27.0) carried over as an
idea: cards marked "from the pack" and "on disk", a form with a live label and
"Check it", and the same rule for a deleted entry. A node that held it moves to
the entry's file when that file is on disk, which is the same model under its
own name; otherwise it keeps the choice, and the next queue says the model is
gone instead of quietly writing with another one -- or starting a download
nobody asked for.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

from . import catalog, chat_template, download, gguf_meta, llm

log = logging.getLogger(__name__)

FIELDS = ("name", "repo", "file", "download_gb", "vram", "note")
GB = 1024 ** 3
SHOWN_REPO_FILES = 40
"""How many of a repository's .gguf files 'Check it' offers to pick from; quant repositories hold 20-30."""
PROBE = ({"role": "system", "content": "You write songs."},
         {"role": "user", "content": "A short song about rain."})
"""What the chat template is rendered with, to learn whether it renders at all, as the writer's prompt is."""
CHECKED_NAME = "the model being checked"
"""Stands in for an empty name: 'Check it' judges the file, and a name is only needed to save."""


def _gigabytes(size) -> float:
    return round(float(size) / GB, 2)


def _disk_rows() -> list:
    """What the widget offers from disk and Ollama; the files that listed entries stand for are left out."""
    return [{"label": label, "path": target, "ollama": label.startswith(llm.OLLAMA_PREFIX)}
            for label, target in llm.offered() if isinstance(target, str)]


def listing() -> dict:
    """Everything the window draws: the entries as written, what each is now, and the files found.

    An entry the widget cannot use (no name, no file) is still shown, marked,
    so that it can be mended or deleted here rather than only in the file.
    """
    found = llm.catalogue()
    seeded = {name.casefold() for name in catalog.seed_names()}
    rows = []
    for raw in catalog.raw_entries():
        row = {key: raw[key] for key in FIELDS if raw.get(key) not in (None, "")}
        entry = catalog.entry(raw)
        if entry is None:
            row.update(label=catalog.label_of(raw) or "(no name)", usable=False, on_disk=False,
                       where="", from_pack=False, local=False)
        else:
            here = llm.listed_file(entry, found)
            row.update(name=entry.name, label=entry.shown(bool(here)), usable=True,
                       on_disk=bool(here), where=here,
                       from_pack=entry.name.casefold() in seeded, local=entry.local)
        rows.append(row)
    problem = catalog.problem()
    return {
        "ok": True,
        "path": catalog.live_file(),
        "writable": not problem,
        "problem": problem,
        "entries": rows,
        "found": _disk_rows(),
        "restorable": catalog.restorable(),
        "folder": download.writer_root(),
        "choices": llm.choices(),
    }


def save(was: str, raw: dict) -> dict:
    """Add an entry, or replace the one called ``was``, and say where open graphs should move.

    ``renamed`` is the old name when the name changed: a node holding any label
    of it moves to ``label``. A change of size or card needs no such help --
    the name still finds the entry, and the browser moves a stale label to the
    new one of the same name by itself.
    """
    was = (was or "").strip()
    written = catalog.update(was, raw) if was else catalog.add(raw)
    entry = catalog.entry(written)
    renamed = bool(was) and was.casefold() != entry.name.casefold()
    return {
        "ok": True,
        "entry": written,
        "name": entry.name,
        "label": entry.shown(bool(llm.listed_file(entry))),
        "renamed": was if renamed else "",
        "choices": llm.choices(),
    }


def delete(name: str) -> dict:
    """Delete one entry, and say which row a node that held it moves to, or "" to keep it.

    The row is the entry's own file, offered under its file name again now
    that no entry stands for it.
    """
    name = (name or "").strip()
    held = next((one for one in catalog.entries() if one.name.casefold() == name.casefold()), None)
    here = llm.listed_file(held) if held is not None else ""
    gone = catalog.remove(name)
    moved_to = ""
    if gone and here:
        same = os.path.normcase(os.path.abspath(here))
        moved_to = next((row["label"] for row in _disk_rows()
                         if os.path.normcase(os.path.abspath(row["path"])) == same), "")
    return {"ok": True, "gone": gone, "name": name, "moved_to": moved_to, "choices": llm.choices()}


def restore() -> dict:
    """The pack's own entries that were deleted, put back at the end of the list."""
    return {"ok": True, "restored": catalog.restore_packaged(), "choices": llm.choices()}


def _say(lines: list, level: str, text: str) -> None:
    lines.append({"level": level, "text": text})


def _worst(lines: list) -> str:
    levels = {line["level"] for line in lines}
    return "bad" if "bad" in levels else "warn" if "warn" in levels else "good"


def _ggufs(sizes: dict) -> list:
    """The models in a repository listing, to pick from: the first part standing for a split set.

    Projectors are left out -- they cannot write -- and so are the later parts
    of a split model, whose first part is offered with the size of them all.
    """
    offered = []
    for path in sorted(sizes, key=str.lower):
        base = path.replace("\\", "/").split("/")[-1]
        if not base.lower().endswith(catalog.SUFFIX) or "mmproj" in base.lower():
            continue
        parts = gguf_meta.split_names(path)
        named = gguf_meta.SPLIT_NAME.match(base)
        if named and int(named.group("no")) != 1:
            continue
        offered.append({"file": path, "gb": _gigabytes(sum(sizes.get(part, 0) for part in parts))})
    return offered[:SHOWN_REPO_FILES]


def _judge_file(path: str, lines: list):
    """What a file on this machine is, read from its header; its size in GB, or None when it cannot write."""
    name = os.path.basename(path)
    why = llm.unfit(path)
    if why:
        _say(lines, "bad", "'" + name + "' is " + why + ".")
        return None
    size = llm.size_of(path)
    _say(lines, "good", "'{}' is here, {}: a '{}' model with its own chat template.".format(
        name, download.human_size(size), llm.architecture(path)))
    try:
        chat_template.render(llm.template(path), [dict(one) for one in PROBE], enable_thinking=False)
    except Exception as error:  # noqa: BLE001 - whatever the template raises is the answer
        _say(lines, "bad", "Its chat template does not render here ({}), so the first run would "
                           "stop on it.".format(error))
        return _gigabytes(size)
    _say(lines, "good", "Its chat template renders, so it can write. Picking it downloads nothing.")
    return _gigabytes(size)


def _listed(repo: str, lines: list):
    """A repository's files and sizes, or None with the reason said."""
    try:
        return download.list_repo_files(repo, token=download.access_token())
    except download.DownloadError as error:
        _say(lines, "bad", str(error))
        return None


def _judge_hub(entry, lines: list) -> tuple:
    """What the Hub says about an entry that is not on this machine: ``(size in GB or None, files to pick)``."""
    sizes = _listed(entry.repo, lines)
    if sizes is None:
        return None, []
    parts = gguf_meta.split_names(entry.file)
    missing = [part for part in parts if part not in sizes]
    if missing:
        offered = _ggufs(sizes)
        _say(lines, "bad", "'{}' has no file '{}'.".format(entry.repo, missing[0]) + (
            " Pick one of its models below." if offered else " It holds no GGUF models at all."))
        return None, offered
    size = sum(sizes[part] for part in parts)
    _say(lines, "good", "'{}' has '{}'{}, {} to download into {}.".format(
        entry.repo, entry.file, " and its other parts" if len(parts) > 1 else "",
        download.human_size(size), download.writer_root()))
    _say(lines, "note", "Its header can be read only once it is here, so whether it can write is "
                        "checked on the first run -- or by Check it again after the download.")
    return _gigabytes(size), []


def _repo_models(repo: str) -> dict:
    """'Check it' with a repository and no file yet: the models in it, to pick one."""
    if not catalog.REPO_ID.match(repo):
        raise catalog.CatalogWriteError(
            "'" + repo + "' is not a Hugging Face repository id; it looks like 'owner/name'.")
    lines = []
    sizes = _listed(repo, lines)
    offered = _ggufs(sizes) if sizes is not None else []
    if sizes is not None:
        if offered:
            _say(lines, "good", "'{}' holds {} GGUF model{}. Pick one below; its file and size go into "
                                "the form.".format(repo, len(offered), "" if len(offered) == 1 else "s"))
        else:
            _say(lines, "bad", "'" + repo + "' holds no GGUF models, and this list runs GGUF files only.")
    return {"ok": True, "verdict": _worst(lines), "lines": lines, "download_gb": None,
            "files": offered, "here": False}


def check(raw) -> dict:
    """Judge an entry as far as that can be done without downloading it.

    A file on this machine is read outright: whether it is a model that can
    write, and whether its chat template renders. A file only on the Hub is
    asked the one question the Hub can answer without weights -- is it there,
    and how big -- which is the question that catches a misspelt file before a
    run fails minutes into a download. With a repository and no file yet, the
    repository's models are listed to pick from.
    """
    if not isinstance(raw, dict):
        raise catalog.CatalogWriteError("An entry is a set of fields.")
    repo = str(raw.get("repo") or "").strip()
    if repo and not str(raw.get("file") or "").strip():
        return _repo_models(repo)
    cleaned = catalog.clean_entry(dict(raw, name=str(raw.get("name") or "").strip() or CHECKED_NAME))
    entry = catalog.entry(cleaned)
    lines: list = []
    here = llm.listed_file(entry)
    offered: list = []
    if here:
        size = _judge_file(here, lines)
    elif entry.local:
        size = None
        _say(lines, "bad", "'" + entry.file + "' is not on this machine. A file here is named by its "
                           "full path; one on Hugging Face needs its repository as well.")
    else:
        size, offered = _judge_hub(entry, lines)
    return {"ok": True, "verdict": _worst(lines), "lines": lines, "download_gb": size,
            "files": offered, "here": bool(here)}


def _open(path: str) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - the person's own list or model folder
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def reveal(what: str) -> dict:
    """Open the live list, or the folder downloads go to, on the machine ComfyUI runs on.

    Only these two, never a path from the request. The path comes back either
    way, so a browser on another machine can say where to look.
    """
    if what == "list":
        path = catalog.materialize()
    elif what == "folder":
        path = download.writer_root()
        os.makedirs(path, exist_ok=True)
    else:
        raise catalog.CatalogWriteError("Only the list or the models folder can be opened from here.")
    try:
        _open(path)
    except OSError as error:
        return {"ok": False, "error": "It could not be opened here: {}".format(error), "path": path}
    return {"ok": True, "path": path}
