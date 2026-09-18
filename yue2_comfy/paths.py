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

from .constants import AUDIO_ENCODERS_SUBDIR, CHECKPOINTS_SUBDIR, MODELS_SUBDIR, WRITER_SUBDIR

log = logging.getLogger(__name__)

SCAN_FOLDERS = (MODELS_SUBDIR, "diffusion_models", "vae", "LLM", "checkpoints")

GGUF_SUBDIR = WRITER_SUBDIR
GGUF_FOLDERS = (WRITER_SUBDIR, "llm", "text_encoders", "clip", "transformers",
                "diffusion_models", "unet", "unet_gguf", "checkpoints")
GGUF_DEPTH = 2

ENV_ROOT = "YUE2_MODELS_ROOT"
USER_SUBDIR = __name__.split(".")[0]

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


def _override_root() -> str:
    """YUE2_MODELS_ROOT when ``search_roots`` would search it, otherwise "".

    The folders the pack writes to have to be the folder it then looks in, and
    until 2026-09-18 they were not: inside ComfyUI the override narrowed the
    search to itself while the downloads still went to ComfyUI/models, so a
    fetch succeeded and the next look found nothing, on every run. It surfaced
    when a test harness with the override set wrote stand-in weights into a
    production ComfyUI's models folder.
    """
    override = os.environ.get(ENV_ROOT)
    if not override or is_network_path(override) or not os.path.isdir(override):
        return ""
    return override


def models_root() -> str:
    """ComfyUI/models/YuE2, registered and created on first use.

    This is the one directory the pack writes to, so it is also the only one it
    creates. Registering it with folder_paths is what lets a user redirect it
    later from extra_model_paths.yaml and have us follow without being told.
    YUE2_MODELS_ROOT, when it names a directory, wins over both: it is where
    the search looks, so it is where the downloads go.
    """
    folder_paths = _folder_paths()
    override = _override_root()
    if override:
        return override
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


def user_dir() -> str:
    """ComfyUI/user/yue2_comfy: what the pack keeps that is not a model.

    Not the pack's own folder, and that is the whole point of it being separate.
    ComfyUI Manager replaces a custom node directory wholesale when it updates
    one, which would throw away a runtime that took a download to get; the user
    directory survives that, and it is also what gets backed up when somebody
    moves an install.
    """
    folder_paths = _folder_paths()
    base = ""
    if folder_paths is not None:
        try:
            base = folder_paths.get_user_directory()
        except Exception:
            log.debug("[yue2_comfy.paths] no user directory from ComfyUI", exc_info=True)
    if not base:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_user")
    return os.path.join(base, USER_SUBDIR)


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
    With YUE2_MODELS_ROOT set it is the checkpoints folder inside that root,
    which is where the search and the refusal message look for the repack.
    """
    folder_paths = _folder_paths()
    override = _override_root()
    if override:
        root = os.path.join(override, CHECKPOINTS_SUBDIR)
        os.makedirs(root, exist_ok=True)
        return root
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


def audio_encoders_root() -> str:
    """ComfyUI/models/audio_encoders, where Comfy-Org's SheetSage2 file belongs.

    The same reasoning as for the checkpoint: it is where their repository puts
    it and where ComfyUI's own audio encoder loader looks, so one copy serves
    both. An existing registered path wins, and without ComfyUI the pack's own
    root stands in. With YUE2_MODELS_ROOT set it is the audio_encoders folder
    inside that root, where the search and the refusal message look.
    """
    folder_paths = _folder_paths()
    override = _override_root()
    if override:
        root = os.path.join(override, AUDIO_ENCODERS_SUBDIR)
        os.makedirs(root, exist_ok=True)
        return root
    if folder_paths is None:
        return os.path.join(models_root(), AUDIO_ENCODERS_SUBDIR)
    try:
        registered = [path for path in folder_paths.get_folder_paths(AUDIO_ENCODERS_SUBDIR)]
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
    root = os.path.join(folder_paths.models_dir, AUDIO_ENCODERS_SUBDIR)
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


def gguf_roots() -> list:
    """Where a GGUF plausibly lives, which is not everywhere models live.

    Most of these names ComfyUI does not register: a stock install knows
    twenty-seven folders and ``LLM`` is not one of them, nor ``llm``, ``clip``
    or ``unet_gguf``. That is why every name is also tried under ``models_dir``
    directly -- on a real install that fallback, not the registry, is what finds
    the language models. The registered spellings are still asked for first, so
    a folder redirected in extra_model_paths.yaml is followed.

    The list is wider than it looks reasonable because people put GGUFs
    wherever their last node pack told them to. Breadth is cheap here: the
    search is bounded to GGUF_DEPTH levels, only ``.gguf`` names go any
    further, and a header is read at most once per file.

    The Hugging Face cache is swept too, which is where ``huggingface-cli
    download`` leaves a GGUF. It was left out at first on the assumption that
    reaching it meant walking every cached repository; measured, it does not --
    ``snapshot_dirs`` already narrows to ``models--*/snapshots/*``, and on a
    machine with nineteen of them the whole sweep cost four milliseconds.
    """
    roots: list = []
    folder_paths = _folder_paths()
    if folder_paths is None:
        _add(roots, os.path.join(models_root(), GGUF_SUBDIR))
    else:
        for name in GGUF_FOLDERS:
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
    for cache_root in hf_cache_roots():
        for snapshot in snapshot_dirs(cache_root):
            _add(roots, snapshot)
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


def hf_repo_for(path: str) -> str:
    """'Qwen/Qwen3-VL-8B-Instruct-GGUF' for a file inside the HF cache, else "".

    A cache snapshot is named by its commit hash, so using the parent directory
    to tell two files of the same name apart would offer somebody a choice
    between their model and 'f982a07559d4a2f6c8744d840bf6fccab30eea96'.
    """
    walk = os.path.dirname(os.path.abspath(path))
    for _ in range(GGUF_DEPTH + 2):
        parent = os.path.dirname(walk)
        if os.path.basename(parent) == "snapshots":
            repo = os.path.basename(os.path.dirname(parent))
            if repo.startswith("models--"):
                return repo.replace("models--", "", 1).replace("--", "/")
            return ""
        if parent == walk:
            return ""
        walk = parent
    return ""


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


def sheetsage_roots() -> list:
    """Everywhere SheetSage2's file may be, audio encoder folders first.

    The override is still the only root when it is set. Otherwise ComfyUI's
    audio_encoders folders come before the places YuE2 itself is looked for,
    and a downloaded copy of the Comfy-Org repository beside the checkout keeps
    the file one level down, in its own audio_encoders folder.
    """
    override = os.environ.get(ENV_ROOT)
    if override:
        return search_roots()
    roots: list = []
    folder_paths = _folder_paths()
    if folder_paths is not None:
        try:
            registered = list(folder_paths.get_folder_paths(AUDIO_ENCODERS_SUBDIR))
        except KeyError:
            registered = []
        for path in registered:
            _add(roots, path)
        try:
            _add(roots, os.path.join(folder_paths.models_dir, AUDIO_ENCODERS_SUBDIR))
        except Exception:
            log.debug("[yue2_comfy.paths] no models_dir", exc_info=True)
    for path in search_roots():
        _add(roots, path)
    sibling = checkout_sibling_root()
    try:
        children = sorted(os.listdir(sibling))
    except OSError:
        children = []
    for child in children:
        _add(roots, os.path.join(sibling, child, AUDIO_ENCODERS_SUBDIR))
    return roots


def asr_roots() -> list:
    """Everywhere Qwen3-ASR's folder may be: the usual roots, then collections beside the checkout.

    The override is still the only root when it is set. Speech models
    downloaded as a set next to the checkout keep each size one level further
    down, so every folder there is a root of its own.
    """
    roots = search_roots()
    if os.environ.get(ENV_ROOT):
        return roots
    sibling = checkout_sibling_root()
    try:
        children = sorted(os.listdir(sibling))
    except OSError:
        children = []
    for child in children:
        _add(roots, os.path.join(sibling, child))
    return roots
