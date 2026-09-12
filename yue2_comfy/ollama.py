"""Models already pulled for Ollama, offered without a second copy on disk.

Somebody who runs Ollama has the writer this pack wants sitting on their disk
already, and asking them to download the same quant again into ``models/LLM`` is
[issue #12](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/12)
against the pack this module comes from, word for word.

Nothing here is a new backend. Ollama stores what it downloads as a plain GGUF:
a blob whose first four bytes are ``GGUF``, which both of this pack's backends
load by path like any other file. What this adds is the index -- turning a store
into ``(label, path)`` pairs that ``llm.catalogue`` already knows how to hand
around, and which it then puts through the same header check as everything else.

Three things about the layout are worth knowing, because they are what make the
index cheap:

- **A manifest is the pairing.** ``manifests/<registry>/<namespace>/<name>/<tag>``
  is a small JSON listing the model's layers by media type.
- **The blobs are content-addressed and extensionless.** ``blobs/sha256-<hex>``,
  the digest's colon turned into a dash. No name and no ``.gguf``, which is fine:
  this pack decides what a file is by reading its header rather than its name.
- **Ollama's own template, params and system layers are ignored.** The GGUF
  carries a chat template of its own and the writer brings its own prompt;
  taking Ollama's would mean honouring its Modelfile, which is another program's
  configuration and not ours to interpret.

**Where it looks, and why not everywhere.** The automatic roots are the places a
local store can be, and they are the same strings on every platform because
Ollama's layout does not vary -- only ``~`` does. A store on the far side of a
virtual machine or a container is deliberately not found automatically: reaching
``\\\\wsl$\\<distro>\\...`` starts a stopped WSL distribution, and this index is
rebuilt whenever ComfyUI repopulates a dropdown. Opening a browser tab must not
boot somebody's virtual machine. Those stores are named by hand instead, in
``YUE2_OLLAMA_MODELS``, which covers WSL, a Docker volume and a store moved to
another drive with one mechanism and no code that knows what WSL is.
"""

from __future__ import annotations

import json
import logging
import os
import re

from . import paths

log = logging.getLogger(__name__)

STORE_ENV = "YUE2_OLLAMA_MODELS"
OLLAMA_ENV = "OLLAMA_MODELS"

DEFAULT_ROOTS = (
    os.path.join("~", ".ollama", "models"),
    os.path.join(os.sep, "usr", "share", "ollama", ".ollama", "models"),
)

MODEL_LAYER = "application/vnd.ollama.image.model"
LIBRARY_PREFIX = ("registry.ollama.ai", "library")

MANIFEST_DEPTH = 5
MANIFEST_MAX_BYTES = 1 << 20

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def roots() -> list:
    """Every store to index, hand-named ones first, each one only once."""
    found: list = []
    seen: set = set()

    def add(value: str, named: bool) -> None:
        """A store that exists, is not a guess at somebody's network share."""
        text = os.path.expanduser((value or "").strip().strip('"'))
        if not text:
            return
        if not named and paths.is_network_path(text):
            log.debug("[yue2_comfy.ollama] %s is remote, not scanned", text)
            return
        try:
            if not os.path.isdir(os.path.join(text, "manifests")):
                return
        except OSError:
            return
        key = os.path.normcase(os.path.abspath(text))
        if key in seen:
            return
        seen.add(key)
        found.append(text)

    for value in (os.environ.get(STORE_ENV) or "").split(os.pathsep):
        add(value, True)
    add(os.environ.get(OLLAMA_ENV) or "", False)
    for value in DEFAULT_ROOTS:
        add(value, False)
    return found


def _manifest_files(root: str) -> list:
    """Every manifest under one store, depth-limited against a loop of links."""
    found: list = []
    stack = [(os.path.join(root, "manifests"), 0)]
    while stack:
        directory, level = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda one: one.name)
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    if level < MANIFEST_DEPTH:
                        stack.append((entry.path, level + 1))
                elif entry.is_file():
                    found.append(entry.path)
            except OSError:
                continue
    return sorted(found)


def model_name(root: str, manifest: str) -> str:
    """``registry/namespace/name/tag`` on disk as ``name:tag`` on screen."""
    try:
        relative = os.path.relpath(manifest, os.path.join(root, "manifests"))
    except ValueError:
        return ""
    parts = [part for part in relative.replace(os.sep, "/").split("/")
             if part and part != "."]
    if len(parts) < 2 or parts[0] == "..":
        return ""
    name, tag = parts[:-1], parts[-1]
    if len(name) > len(LIBRARY_PREFIX) and tuple(name[:len(LIBRARY_PREFIX)]) == LIBRARY_PREFIX:
        name = name[len(LIBRARY_PREFIX):]
    return "/".join(name) + ":" + tag


def _layers(manifest: str) -> dict:
    """``{media type: digest}`` for one manifest, or empty if it is not one."""
    try:
        if os.path.getsize(manifest) > MANIFEST_MAX_BYTES:
            return {}
        with open(manifest, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}

    found: dict = {}
    for layer in data.get("layers") or []:
        if not isinstance(layer, dict):
            continue
        media, digest = layer.get("mediaType"), layer.get("digest")
        if isinstance(media, str) and isinstance(digest, str) and media not in found:
            found[media] = digest
    return found


def _blob(root: str, digest: str) -> str:
    """The file one digest names, or "" when it is absent or is not a digest.

    The shape is checked rather than trusted. A manifest is a file like any
    other, and a ``digest`` of ``sha256:../../..`` would otherwise be a path
    leaving the store -- cheap to refuse, and the refusal costs nothing real
    because every digest Ollama writes is 64 hex characters.
    """
    text = (digest or "").strip()
    prefix = "sha256:"
    if not text.startswith(prefix) or not _DIGEST.match(text[len(prefix):]):
        return ""
    path = os.path.join(root, "blobs", "sha256-" + text[len(prefix):])
    try:
        return path if os.path.isfile(path) else ""
    except OSError:
        return ""


def entries() -> list:
    """``(name, path)`` for every model in every store, one per set of files.

    Tags sharing a blob -- ``qwen3:8b`` and ``qwen3:latest`` usually do -- are
    one model with two names, so the first name in path order stands for both
    rather than the same file being offered twice.
    """
    found: list = []
    seen: set = set()

    for root in roots():
        for manifest in _manifest_files(root):
            name = model_name(root, manifest)
            if not name:
                continue
            model = _blob(root, _layers(manifest).get(MODEL_LAYER, ""))
            if not model:
                continue
            key = os.path.normcase(os.path.abspath(model))
            if key in seen:
                continue
            seen.add(key)
            found.append((name, model))
    return found
