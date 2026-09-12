"""Where this machine keeps models, asked of ComfyUI rather than assumed.

Two rules shape this module. The first is that ComfyUI already knows: every
root here comes from folder_paths, which has already applied
extra_model_paths.yaml by the time any custom node is imported, so a user who
keeps ten terabytes of weights on another drive needs to tell us nothing. The
second is that a registered root need not exist -- extra_model_paths.yaml maps
folders in from anywhere and some entries are simply wrong -- so every path
that leaves this module has been checked, not merely computed.

Nothing here reads a weight file. Deciding what a file *is* belongs to
discovery.py; this module only says where to look.
"""

from __future__ import annotations

import logging
import os
import re

from .constants import CHECKPOINTS_SUBDIR, MODELS_SUBDIR

log = logging.getLogger(__name__)

SCAN_FOLDERS = (MODELS_SUBDIR, "diffusion_models", "vae", "LLM", "checkpoints")

ENV_ROOT = "YUE2_MODELS_ROOT"

_EXTENDED_LOCAL = re.compile(r"^\\\\[?.]\\[A-Za-z]:")


def is_network_path(value: str) -> bool:
    """Whether this string names a UNC share rather than something local."""
    text = (value or "").strip()
    if os.name == "nt":
        text = text.replace("/", "\\")
    return text.startswith("\\\\") and not _EXTENDED_LOCAL.match(text)


def refuse_network_path(value: str, field: str, advice: str) -> None:
    """Refuse a network location that arrived from outside this machine.

    Merely looking at a UNC path is an authentication attempt against whatever
    host it names, and it happens on the isfile call, long before anything
    judges the file. A workflow downloaded from a stranger, or a request to the
    ComfyUI API -- which has no CSRF token and is routinely served on --listen
    -- must not be able to make this machine reach out and introduce itself.

    Nothing legitimate is lost: a share holding your models is reachable
    through a drive letter or a mount point like any other folder.
    """
    if not is_network_path(value):
        return
    raise RuntimeError(
        "'" + value + "' is a network path, and " + field + " will not follow one."
        "\n\n" + advice
    )


def _folder_paths():
    try:
        import folder_paths
    except ImportError:
        return None
    return folder_paths


def models_root() -> str:
    """ComfyUI/models/YuE2, registered and created on first use.

    This is the one directory the pack writes to, so it is also the only one it
    creates. Registering it with folder_paths is what lets a user redirect it
    later from extra_model_paths.yaml and have us follow without being told.
    """
    folder_paths = _folder_paths()
    if folder_paths is None:
        root = os.environ.get(ENV_ROOT) or os.path.join(os.path.expanduser("~"), MODELS_SUBDIR)
        os.makedirs(root, exist_ok=True)
        return root

    try:
        registered = list(folder_paths.get_folder_paths(MODELS_SUBDIR))
    except KeyError:
        registered = []
    if registered:
        os.makedirs(registered[0], exist_ok=True)
        return registered[0]

    root = os.path.join(folder_paths.models_dir, MODELS_SUBDIR)
    try:
        folder_paths.add_model_folder_path(MODELS_SUBDIR, root)
    except Exception:
        log.debug("[yue2_comfy.paths] could not register %s", MODELS_SUBDIR, exc_info=True)
    os.makedirs(root, exist_ok=True)
    return root


def checkpoints_root() -> str:
    """ComfyUI/models/checkpoints, where Comfy-Org's single file belongs.

    Their repository says to put it there, ComfyUI's own model manager puts it
    there, and the native YuE2 nodes read it from there. Writing it anywhere
    else would give a machine two copies of eight gigabytes, so this pack uses
    the same folder rather than its own.

    An existing registered path wins over a computed one, because a user who
    redirected checkpoints in extra_model_paths.yaml did it to keep large files
    off this drive. Without ComfyUI -- in the tests, or a bare interpreter --
    there is no checkpoints folder to speak of, so the pack's own root is used.
    """
    folder_paths = _folder_paths()
    if folder_paths is None:
        return models_root()
    try:
        registered = [path for path in folder_paths.get_folder_paths(CHECKPOINTS_SUBDIR)]
    except KeyError:
        registered = []
    for path in registered:
        try:
            if os.path.isdir(path):
                return path
        except OSError:
            continue
    if registered:
        os.makedirs(registered[0], exist_ok=True)
        return registered[0]
    root = os.path.join(folder_paths.models_dir, CHECKPOINTS_SUBDIR)
    os.makedirs(root, exist_ok=True)
    return root


def local_dir_for_repo(repo_id: str) -> str:
    """Where a Hub repository lands when this pack downloads it."""
    return os.path.join(models_root(), repo_id.rstrip("/").split("/")[-1])


def _add(roots: list, path: str) -> None:
    if not path:
        return
    if is_network_path(path):
        log.debug("[yue2_comfy.paths] skipping network root %s", path)
        return
    try:
        if not os.path.isdir(path):
            return
    except OSError:
        return
    key = os.path.normcase(os.path.abspath(path))
    if key not in {os.path.normcase(os.path.abspath(known)) for known in roots}:
        roots.append(path)


def comfy_roots() -> list:
    """Every ComfyUI model folder worth sweeping, in the order to sweep it."""
    roots: list = []
    folder_paths = _folder_paths()
    if folder_paths is None:
        return roots
    for name in SCAN_FOLDERS:
        try:
            registered = list(folder_paths.get_folder_paths(name))
        except KeyError:
            registered = []
        for path in registered:
            _add(roots, path)
        try:
            _add(roots, os.path.join(folder_paths.models_dir, name))
        except Exception:
            log.debug("[yue2_comfy.paths] no models_dir", exc_info=True)
    return roots


def hf_cache_roots() -> list:
    """Hugging Face cache roots, so a copy pulled by any other tool is found."""
    candidates = []
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        candidates.append(hub)
    home = os.environ.get("HF_HOME")
    if home:
        candidates.append(os.path.join(home, "hub"))
    else:
        candidates.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    roots: list = []
    for path in candidates:
        _add(roots, path)
    return roots


def snapshot_dirs(cache_root: str) -> list:
    """The snapshot directories inside one Hugging Face cache root."""
    found: list = []
    try:
        entries = sorted(os.listdir(cache_root))
    except OSError:
        return found
    for entry in entries:
        if not entry.startswith("models--"):
            continue
        snapshots = os.path.join(cache_root, entry, "snapshots")
        try:
            revisions = sorted(os.listdir(snapshots))
        except OSError:
            continue
        for revision in revisions:
            path = os.path.join(snapshots, revision)
            if os.path.isdir(path):
                found.append(path)
    return found


def repo_label(snapshot: str) -> str:
    """'HF cache: m-a-p/YuE2-3B@1a2b3c4d' for a snapshot directory."""
    repo = os.path.basename(os.path.dirname(os.path.dirname(snapshot)))
    pretty = repo.replace("models--", "", 1).replace("--", "/")
    return "HF cache: " + pretty + "@" + os.path.basename(snapshot)[:8]


def checkout_sibling_root() -> str:
    """A models directory beside this checkout, which is a development layout.

    Harmless in a released install -- it resolves to custom_nodes/models, which
    nobody has -- and exactly right when the pack is junctioned into ComfyUI
    from a working tree that keeps its weights next door. Last in the order, so
    it can only ever answer a question the real roots did not.
    """
    here = os.path.realpath(__file__)
    pack = os.path.dirname(os.path.dirname(here))
    return os.path.join(os.path.dirname(pack), "models")


def search_roots() -> list:
    """Everywhere to look: the override alone, or ComfyUI and the caches.

    YUE2_MODELS_ROOT is not the first of several roots, it is the only one. An
    override that still lets the search wander is not an override: it cannot be
    used to stop the pack guessing, and it makes "where did it get that file
    from" unanswerable. Set it and the answer is one directory; leave it unset
    and the pack looks everywhere ComfyUI knows about.

    A value that is not a directory is a typo, and a typo should not brick the
    pack silently, so that case warns and searches normally instead.
    """
    roots: list = []
    override = os.environ.get(ENV_ROOT)
    if override:
        refuse_network_path(
            override, ENV_ROOT,
            "Map the share to a drive letter or a mount point and set "
            + ENV_ROOT + " to that instead.",
        )
        _add(roots, override)
        if roots:
            return roots
        log.warning("[yue2_comfy.paths] %s is set to %s, which is not a directory; "
                    "searching the usual places instead", ENV_ROOT, override)
    for path in comfy_roots():
        _add(roots, path)
    for cache_root in hf_cache_roots():
        for snapshot in snapshot_dirs(cache_root):
            _add(roots, snapshot)
    _add(roots, checkout_sibling_root())
    return roots
