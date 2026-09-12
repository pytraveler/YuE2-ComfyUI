"""Checking weight files against the checksums published beside them.

m-a-p ships a ``weights_manifest.json`` next to each release: usually a schema
number, and per file its size and its SHA-256. Size is checked on every run
because it costs a directory listing. This is the other half, and it costs time --
5.4 s over 7.8 GB, measured -- so it runs once, right after a download, and not
again on every graph execution.

What it catches is the failure that size cannot see. A transfer cut in half
leaves a short file and the downloader refetches it; a transfer that arrives
complete but wrong -- a proxy that rewrote it, a disk that dropped a sector --
leaves a file of exactly the right length, and every later run skips it as
already present. That file then loads, and the model sounds subtly broken with
nothing anywhere saying why.

Comfy-Org's single-file repack publishes no manifest. There is nothing to check
it against, and this module says so rather than inventing a verdict.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os

from .constants import MANIFEST_NAME

log = logging.getLogger(__name__)

SCHEMA = 1
READ_CHUNK = 1 << 20
MAX_MANIFEST_BYTES = 1 << 20


class Corrupt(RuntimeError):
    """A file whose contents disagree with the checksum published for it."""


def read(directory: str) -> dict:
    """``{filename: (size, sha256)}`` for the manifest in *directory*.

    Empty for a folder that has no manifest, and empty for one this code does
    not understand: guessing at an unknown schema is how a checker starts
    reporting failures that are its own fault rather than the file's.

    A missing ``schema`` key counts as schema 1 rather than as unknown. YuE2's
    own exporter writes the file that way: ``copy_model_files`` in
    ``yue2/storage.py`` ends on ``{"files": ...}`` with no label, over entries
    of exactly the shape below, so a folder re-saved through the model's own
    code would otherwise arrive unverifiable. A document that names a different
    schema is still declined -- unlabelled and labelled as something else are
    different claims, and only the second one says this reader is out of date.
    """
    path = os.path.join(directory, MANIFEST_NAME)
    try:
        if os.path.getsize(path) > MAX_MANIFEST_BYTES:
            log.warning("[yue2_comfy.manifest] %s is implausibly large, ignoring", path)
            return {}
        with open(path, "rb") as handle:
            document = json.loads(handle.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}

    if not isinstance(document, dict):
        return {}

    declared = document.get("schema", SCHEMA)
    if declared != SCHEMA:
        log.warning("[yue2_comfy.manifest] %s is schema %r, not %d; not checking",
                    path, declared, SCHEMA)
        return {}

    files = document.get("files")
    if not isinstance(files, dict):
        log.warning("[yue2_comfy.manifest] %s lists no files this code can read; "
                    "not checking", path)
        return {}

    listed = {}
    for name, entry in files.items():
        if not isinstance(entry, dict) or os.path.basename(name) != name:
            continue
        checksum = str(entry.get("sha256") or "").lower()
        if len(checksum) != 64 or not all(c in "0123456789abcdef" for c in checksum):
            continue
        try:
            size = int(entry.get("bytes", 0))
        except (TypeError, ValueError):
            size = 0
        listed[name] = (size, checksum)
    return listed


def digest(path: str, on_read=None, cancelled=None) -> str:
    """The SHA-256 of one file, read a megabyte at a time.

    ``on_read`` is handed the running byte count so a caller can move a bar;
    ``cancelled`` is asked between chunks so a seven-gigabyte check can be
    stopped the way everything else in this pack can.
    """
    total = 0
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("cancelled while checking " + os.path.basename(path))
            block = handle.read(READ_CHUNK)
            if not block:
                break
            hasher.update(block)
            total += len(block)
            if on_read is not None:
                on_read(total)
    return hasher.hexdigest()


def verify(directory: str, progress=None, cancelled=None) -> list:
    """Check every file the manifest in *directory* lists that is present.

    Returns the names that failed. A file the manifest names but the folder has
    not got is not a failure here: whether it should be there is the loader's
    question, and answering it twice in two voices helps nobody.
    """
    listed = read(directory)
    if not listed:
        return []

    present = [(name, os.path.join(directory, name)) for name in sorted(listed)]
    present = [(name, path) for name, path in present if os.path.isfile(path)]
    if not present:
        return []

    total = sum(os.path.getsize(path) for _name, path in present) or 1
    done = 0
    bad = []
    for name, path in present:
        size, checksum = listed[name]
        if progress is not None:
            progress.text("Checking " + name)

        base = done

        def on_read(read_so_far, base=base):
            if progress is not None:
                progress.ratio(min(1.0, (base + read_so_far) / float(total)))

        actual = digest(path, on_read, cancelled)
        done += os.path.getsize(path)
        if size and os.path.getsize(path) != size:
            bad.append(name)
            log.error("[yue2_comfy.manifest] %s is %d bytes, the manifest says %d",
                      path, os.path.getsize(path), size)
        elif actual != checksum:
            bad.append(name)
            log.error("[yue2_comfy.manifest] %s hashes to %s, the manifest says %s",
                      path, actual, checksum)
        else:
            log.info("[yue2_comfy.manifest] %s matches its published checksum", name)
    return bad


def advice(directory: str, bad: list) -> str:
    """What to tell someone whose freshly downloaded weights do not match."""
    where = os.path.join(directory, bad[0]) if len(bad) == 1 else directory
    return (
        "{} of the downloaded file(s) do not match the checksums published with "
        "them:\n\n{}\n\nThe file is the right length, so nothing will notice it "
        "again and every later run will use it as it is. Delete it and run the "
        "node once more:\n\n    {}\n\nIf a second download fails the same way, "
        "something between this machine and Hugging Face is rewriting the "
        "transfer -- a proxy or an antivirus that inspects downloads is the "
        "usual cause."
    ).format(len(bad), "\n".join("  " + name for name in bad), where)


def check(directory: str, progress=None, cancelled=None) -> None:
    """Verify *directory*, and refuse if anything in it is wrong."""
    bad = verify(directory, progress, cancelled)
    if bad:
        raise Corrupt(advice(directory, bad))
