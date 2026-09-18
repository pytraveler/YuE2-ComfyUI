"""Fetching the weights, resumably, with the bytes reported to the node.

Ported from the MiniMax-H3 Prompt Rewriter pack, where the same four details
were load-bearing and are here too:

- The token never reaches the CDN. A Hub URL answers 302 with a pre-signed
  link, and an Authorization header on that hop is a 400 rather than an ignored
  header. requests strips the header itself when a redirect crosses hosts,
  which is exactly the behaviour wanted.
- A signed link expires, so every retry re-requests the original Hub URL
  instead of a resolved one; resuming an hour later would otherwise write an
  error page into the middle of a checkpoint.
- Accept-Encoding: identity. A transparently gzipped response makes the bytes
  written disagree with both the Range offsets and the expected size.
- Space is checked before the first byte, because failing seven gigabytes into
  an eight gigabyte download costs the whole transfer.

Completion is tested by size. Bytes land in <name>.part and are renamed into
place only once whole, so a half-written checkpoint is never mistaken for a
model -- by this pack or by ComfyUI's own nodes, which read the same folder.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from typing import NamedTuple

from . import paths
from .constants import (
    LM_ALLOW, LM_DIRNAME, LM_REPO, PACK, REPACK_BF16_BYTES, REPACK_BF16_NAME,
    REPACK_BF16_PATH, REPACK_INT8_BYTES, REPACK_INT8_NAME, REPACK_INT8_PATH,
    REPACK_REPO, VAE_ALLOW, VAE_DIRNAME, VAE_LEGACY_DIRNAME, VAE_LEGACY_REPO,
    VAE_REPO, WRITER_AUTO, WRITER_BYTES, WRITER_NAME, WRITER_PATH, WRITER_REPO,
    WRITER_SUBDIR, install_command,
)

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://huggingface.co"
CHUNK_SIZE = 1 << 22
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 60
MAX_ATTEMPTS = 5
BACKOFF_CAP = 30.0
RETRY_STATUS = (408, 425, 429, 500, 502, 503, 504)
SPACE_MARGIN = 1.02


class DownloadError(RuntimeError):
    """The transfer cannot proceed; retrying the same call will not help."""


class _Retryable(Exception):
    """A transient failure that another attempt may clear."""


class Item(NamedTuple):
    """One file to fetch: where it lives on the Hub and where it goes here."""

    repo_path: str
    dest: str
    size: int = 0


def _requests():
    """The HTTP library, named in the error when this Python has not got it.

    ComfyUI installs requests as a matter of course, so this is a message for
    the unusual environment rather than the usual one, and it names this
    interpreter because in a portable build a bare pip install goes somewhere
    the node will never look.
    """
    try:
        import requests
    except ImportError as error:
        raise DownloadError(
            "Downloading needs the requests package, and it is not installed in "
            "this Python.\n\nInstall it with:\n\n" + install_command("requests")
        ) from error
    return requests


_SESSIONS = {}
_OS_TRUST = {"on": False}


def _os_trust_adapter():
    """A requests adapter that trusts what Windows and the distribution trust.

    requests verifies against the certifi bundle, which knows nothing about a
    root installed locally. Plenty of machines have one: an antivirus or a
    company proxy that opens TLS to inspect it installs its own certificate
    authority in the system store, and from certifi's point of view every
    huggingface.co connection is then signed by a stranger.

    ssl.create_default_context loads that store, which is why a browser on the
    same machine is perfectly happy. Nothing is being skipped here -- the
    certificate is still verified, against the same authorities the rest of the
    operating system uses.
    """
    import ssl

    from requests.adapters import HTTPAdapter

    class _SystemTrust(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            kwargs["ssl_context"] = ssl.create_default_context()
            return super().init_poolmanager(*args, **kwargs)

    return _SystemTrust()


def _session(os_trust: bool):
    requests = _requests()
    key = "os" if os_trust else "certifi"
    if key not in _SESSIONS:
        session = requests.Session()
        if os_trust:
            session.mount("https://", _os_trust_adapter())
        _SESSIONS[key] = session
    return _SESSIONS[key]


def _request(method: str, url: str, headers: dict, stream: bool = False):
    """One request, retried against the system trust store if certifi refuses.

    The switch is remembered, so a machine behind an inspecting proxy pays the
    failed handshake once rather than once per file.
    """
    requests = _requests()
    try:
        return _session(_OS_TRUST["on"]).request(
            method, url, headers=headers, stream=stream, allow_redirects=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except requests.exceptions.SSLError:
        if _OS_TRUST["on"]:
            raise
        _OS_TRUST["on"] = True
        log.warning("[yue2_comfy.download] the certificate chain was not one certifi "
                    "knows, retrying against this machine's own trust store")
        return _session(True).request(
            method, url, headers=headers, stream=stream, allow_redirects=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))


def _get(url: str, headers: dict, stream: bool = False):
    return _request("GET", url, headers, stream)


def content_length(url: str) -> int:
    """How big one file is, asked of the host that serves it, or 0 if it will not say.

    A HEAD rather than a releases API call. GitHub's API allows sixty anonymous
    requests an hour per address and answers 403 with no sizes at all once that
    is spent, which would leave a progress bar without a denominator exactly
    when a half-gigabyte download makes one worth having. This follows the
    redirect to wherever the bytes come from and carries no such limit.
    """
    try:
        response = _request("HEAD", url, _headers(None))
        if response.status_code >= 400:
            return 0
        return int(response.headers.get("Content-Length") or 0)
    except Exception:
        log.debug("[yue2_comfy.download] HEAD %s failed", url, exc_info=True)
        return 0


def _ssl_advice(error) -> str:
    return (
        "The connection to " + endpoint() + " could not be verified: " + str(error)
        + "\n\nThis is usually an antivirus or a company proxy opening TLS to "
        "inspect it. Point requests at the same certificates the rest of this "
        "machine uses by setting REQUESTS_CA_BUNDLE, or download the files by "
        "hand -- 'download' set to 'off' prints the links and the folders."
    )


def endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")


def access_token():
    """A Hub token from the environment, or the one huggingface_hub has saved."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def human_size(size: float) -> str:
    for unit, scale in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if size >= scale:
            return "{:.2f} {}".format(size / scale, unit)
    return "{} B".format(int(size))


def _headers(token) -> dict:
    headers = {"User-Agent": PACK, "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def _interrupted() -> None:
    """Let a cancelled queue stop the transfer between chunks."""
    try:
        import comfy.model_management

        comfy.model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _safe_name(repo_path: str) -> str:
    """The leaf of a repository path, refusing anything that escapes a folder."""
    parts = [part for part in re.split(r"[/\\]", repo_path) if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts) or ":" in repo_path:
        raise DownloadError("refusing unsafe repository path " + repr(repo_path))
    return parts[-1]


def free_space(directory: str) -> int:
    probe = os.path.abspath(directory)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free


def check_space(directory: str, needed: int) -> None:
    """Refuse up front rather than filling the volume and failing at 99 percent."""
    if needed <= 0:
        return
    available = free_space(directory)
    if available < needed * SPACE_MARGIN:
        raise DownloadError(
            human_size(needed) + " is needed in " + directory + ", only "
            + human_size(available) + " is free."
        )


def _explain(status: int, what: str) -> str:
    if status in (401, 403):
        return (
            "HTTP {} for {} -- the repository is private or gated. Accept its "
            "licence on huggingface.co and set HF_TOKEN in the environment "
            "ComfyUI runs in.".format(status, what)
        )
    if status == 404:
        return "HTTP 404 for {} -- no such repository, revision, or file.".format(what)
    return "HTTP {} for {}".format(status, what)


def _pause(attempt: int) -> float:
    return min(2.0 ** attempt, BACKOFF_CAP)


def list_repo_files(repo_id: str, revision: str = "main", token=None) -> dict:
    """Every file in a repository and its size, as one request."""
    requests = _requests()
    url = "{}/api/models/{}/tree/{}?recursive=1".format(endpoint(), repo_id, revision)
    try:
        response = _get(url, _headers(token))
    except requests.exceptions.SSLError as error:
        raise DownloadError(_ssl_advice(error)) from error
    except requests.RequestException as error:
        raise DownloadError("Could not reach " + endpoint() + ": " + str(error)) from error
    if response.status_code >= 400:
        raise DownloadError(_explain(response.status_code,
                                     "'" + repo_id + "' (" + revision + ")"))
    sizes = {}
    for entry in response.json():
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs") or {}
        sizes[entry["path"]] = int(lfs.get("size") or entry.get("size") or 0)
    return sizes


def plan(repo_id: str, wanted: dict, revision: str = "main", token=None) -> list:
    """Turn 'this repository path goes to that local file' into sized items.

    The sizes come from the Hub rather than from a table in this pack, so the
    space check and the progress bar are right on the day rather than right on
    the day this was written. A path that has disappeared from the repository
    is said out loud here instead of becoming a 404 mid-transfer.
    """
    sizes = list_repo_files(repo_id, revision, token)
    items = []
    for repo_path, dest in wanted.items():
        if repo_path not in sizes:
            raise DownloadError(
                "'" + repo_id + "' no longer has " + repo_path
                + ". The repository has been rearranged; update this pack.")
        _safe_name(repo_path)
        items.append(Item(repo_path, dest, sizes[repo_path]))
    return items


def _finalize(part: str, dest: str) -> int:
    final = os.path.getsize(part)
    if os.path.isfile(dest):
        os.remove(dest)
    os.replace(part, dest)
    return final


def _stream_to_part(item: Item, url: str, token, base: int, on_progress) -> int:
    """Fetch one file, resuming its .part remainder when the server allows it.

    'base' is the absolute byte count of every other item, so the reported
    position stays right whether Range is honoured or the transfer restarts
    from zero.
    """
    requests = _requests()
    part = item.dest + ".part"
    name = os.path.basename(item.dest)
    os.makedirs(os.path.dirname(item.dest) or ".", exist_ok=True)

    offset = os.path.getsize(part) if os.path.isfile(part) else 0
    if item.size and offset >= item.size:
        if offset == item.size:
            return _finalize(part, item.dest)
        os.remove(part)
        offset = 0

    headers = _headers(token)
    if offset:
        headers["Range"] = "bytes={}-".format(offset)

    try:
        response = _get(url, headers, stream=True)
    except requests.exceptions.SSLError as error:
        raise DownloadError(_ssl_advice(error)) from error
    except requests.RequestException as error:
        raise _Retryable(str(error)) from error

    with response:
        if response.status_code in RETRY_STATUS:
            raise _Retryable("HTTP {}".format(response.status_code))
        if response.status_code >= 400:
            raise DownloadError(_explain(response.status_code, item.repo_path))

        mode = "ab"
        if offset and response.status_code != 206:
            log.warning("[yue2_comfy.download] %s: range ignored, restarting",
                        item.repo_path)
            offset = 0
            mode = "wb"

        written = offset
        on_progress(base + written, name)
        try:
            with open(part, mode) as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    _interrupted()
                    handle.write(chunk)
                    written += len(chunk)
                    on_progress(base + written, name)
        except requests.RequestException as error:
            raise _Retryable(str(error)) from error

    if item.size and os.path.getsize(part) != item.size:
        raise _Retryable("{}: {} bytes written, expected {}".format(
            item.repo_path, os.path.getsize(part), item.size))
    return _finalize(part, item.dest)


def fetch_item(repo_id: str, item: Item, revision: str, token, base: int,
               on_progress) -> int:
    """One file, retried on the failures that another attempt can clear."""
    url = "{}/{}/resolve/{}/{}".format(endpoint(), repo_id, revision, item.repo_path)
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _stream_to_part(item, url, token, base, on_progress)
        except _Retryable as error:
            last_error = error
            log.warning("[yue2_comfy.download] %s failed on attempt %d/%d: %s",
                        item.repo_path, attempt, MAX_ATTEMPTS, error)
            if attempt < MAX_ATTEMPTS:
                time.sleep(_pause(attempt))
        except OSError as error:
            raise DownloadError(
                "Could not write '" + item.dest + "': " + str(error)) from error
    raise DownloadError(
        "Gave up on '" + item.repo_path + "' after {} attempts: {}".format(
            MAX_ATTEMPTS, last_error))


def fetch_url(url: str, dest: str, size: int = 0, base: int = 0,
              on_progress=None, label: str = "") -> int:
    """One file from a plain URL, with the same resume and retries as a Hub file.

    The llama.cpp runtime does not come from the Hub, and that is the only
    difference: it wants the resumed .part file, the backoff, the system trust
    store and the cancellable chunk loop just as much as a model does. No token
    is sent -- a Hub credential has no business travelling to another host.
    """
    item = Item(label or os.path.basename(dest), dest, int(size or 0))
    report = on_progress if on_progress is not None else _silent
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _stream_to_part(item, url, None, base, report)
        except _Retryable as error:
            last_error = error
            log.warning("[yue2_comfy.download] %s failed on attempt %d/%d: %s",
                        item.repo_path, attempt, MAX_ATTEMPTS, error)
            if attempt < MAX_ATTEMPTS:
                time.sleep(_pause(attempt))
        except OSError as error:
            raise DownloadError(
                "Could not write '" + dest + "': " + str(error)) from error
    raise DownloadError(
        "Gave up on '" + url + "' after {} attempts: {}".format(MAX_ATTEMPTS, last_error))


def fetch(repo_id: str, wanted: dict, title: str, progress=None,
          revision: str = "main") -> int:
    """Mirror the named files of one repository, skipping what is already whole.

    A file that is present but the wrong size is not a file, it is the remains
    of something that went wrong; it is removed rather than resumed, because
    there is no telling which of its bytes are good.
    """
    from .progress import TransferReporter

    token = access_token()
    if progress is not None:
        progress.text("Listing " + repo_id)
    items = plan(repo_id, wanted, revision, token)

    total = sum(item.size for item in items)
    pending = []
    transferred = 0
    for item in items:
        if os.path.isfile(item.dest):
            local = os.path.getsize(item.dest)
            if item.size and local == item.size:
                transferred += local
                continue
            os.remove(item.dest)
        part = item.dest + ".part"
        if os.path.isfile(part):
            already = os.path.getsize(part)
            if item.size and already < item.size:
                transferred += already
        pending.append(item)

    if not pending:
        return total

    directory = os.path.dirname(items[0].dest) or "."
    os.makedirs(directory, exist_ok=True)
    check_space(directory, total - transferred)
    log.info("[yue2_comfy.download] %s: %d file(s), %s to fetch",
             repo_id, len(pending), human_size(total - transferred))

    report = TransferReporter(progress, total, title) if progress is not None \
        else _silent
    report(transferred, "")
    for item in pending:
        _interrupted()
        base = transferred
        transferred = base + fetch_item(repo_id, item, revision, token, base, report)
    report(total, "")
    return total


def _silent(transferred, name) -> None:
    """The progress sink for a call that has no node to report to."""


def repack_wanted(quantization: str) -> dict:
    """The one file Comfy-Org publishes, and where ComfyUI expects to find it."""
    root = paths.checkpoints_root()
    if quantization == "int8":
        return {REPACK_INT8_PATH: os.path.join(root, REPACK_INT8_NAME)}
    return {REPACK_BF16_PATH: os.path.join(root, REPACK_BF16_NAME)}


def repack_bytes(quantization: str) -> int:
    return REPACK_INT8_BYTES if quantization == "int8" else REPACK_BF16_BYTES


def fetch_repack(quantization: str, progress=None) -> str:
    """Comfy-Org's checkpoint, into ComfyUI/models/checkpoints where it belongs.

    Not into a folder of this pack's own. That file is what ComfyUI's native
    YuE2 nodes read, and putting it anywhere else would leave the machine with
    two copies of eight gigabytes for no reason.
    """
    wanted = repack_wanted(quantization)
    title = "Downloading YuE2 ({}, {})".format(
        quantization, human_size(repack_bytes(quantization)))
    fetch(REPACK_REPO, wanted, title, progress)
    return list(wanted.values())[0]


def sheetsage_wanted() -> dict:
    """SheetSage2's one file, and where ComfyUI expects audio encoders."""
    from .constants import SHEETSAGE_NAME, SHEETSAGE_PATH

    return {SHEETSAGE_PATH: os.path.join(paths.audio_encoders_root(), SHEETSAGE_NAME)}


def fetch_sheetsage(progress=None) -> str:
    """Comfy-Org's SheetSage2 file, into ComfyUI/models/audio_encoders.

    Always this file, whatever 'download' names: m-a-p publish SheetSage2 only
    as adapters over a separate encoder, which this pack does not read.
    """
    from .constants import SHEETSAGE_BYTES

    wanted = sheetsage_wanted()
    fetch(REPACK_REPO, wanted, "Downloading SheetSage2 ({})".format(human_size(SHEETSAGE_BYTES)), progress)
    return list(wanted.values())[0]


def ensure_sheetsage(settings: dict, progress=None) -> str:
    """SheetSage2's weights, fetched first when the machine has not got them and downloading is on."""
    from . import discovery

    found = discovery.find_sheetsage()
    if found:
        return found
    if settings.get("download", "auto") == "off":
        raise FileNotFoundError(
            discovery.sheetsage_missing_message(paths.sheetsage_roots())
            + "\n\nOr set 'download' in YuE2 Options to 'auto', and the node will fetch it itself.")
    path = fetch_sheetsage(progress)
    if progress is not None:
        progress.text("Download finished")
    return path


def asr_wanted() -> dict:
    """Qwen3-ASR-1.7B's three files, into their own folder under models/YuE2."""
    from .constants import ASR_DIRNAME, ASR_FILES

    folder = os.path.join(paths.models_root(), ASR_DIRNAME)
    return {name: os.path.join(folder, name) for name in ASR_FILES}


def fetch_asr(progress=None) -> str:
    """Qwen's release of the speech model, whatever 'download' names; the folder it landed in."""
    from .constants import ASR_BYTES, ASR_REPO

    wanted = asr_wanted()
    fetch(ASR_REPO, wanted, "Downloading Qwen3-ASR-1.7B ({})".format(human_size(ASR_BYTES)), progress)
    return os.path.dirname(list(wanted.values())[0])


def ensure_asr(settings: dict, progress=None) -> str:
    """The speech model's folder, fetched first when the machine has not got it and downloading is on."""
    from . import discovery

    found = discovery.find_asr()
    if found:
        return found
    if settings.get("download", "auto") == "off":
        raise FileNotFoundError(
            discovery.asr_missing_message(paths.asr_roots())
            + "\n\nOr set 'download' in YuE2 Options to 'auto', and the node will fetch it itself.")
    folder = fetch_asr(progress)
    if progress is not None:
        progress.text("Download finished")
    return folder


def vocals_wanted() -> dict:
    """The voice separator's one file, into models/YuE2."""
    from .constants import VOCALS_NAME

    return {VOCALS_NAME: os.path.join(paths.models_root(), VOCALS_NAME)}


def fetch_vocals(progress=None) -> str:
    """Kimberley Jensen's released weights at the pinned revision, whatever 'download' names; the file's path.

    Pinned because the repository is one person's and a later upload under the
    same name would be a different model from the one this pack was checked
    against.
    """
    from .constants import VOCALS_BYTES, VOCALS_REPO, VOCALS_REVISION

    wanted = vocals_wanted()
    fetch(VOCALS_REPO, wanted, "Downloading Mel-Band RoFormer ({})".format(human_size(VOCALS_BYTES)), progress,
          revision=VOCALS_REVISION)
    return list(wanted.values())[0]


def ensure_vocals(settings: dict, progress=None) -> str:
    """The voice separator's weights, fetched first when the machine has not got them and downloading is on."""
    from . import discovery

    found = discovery.find_vocals()
    if found:
        return found
    if settings.get("download", "auto") == "off":
        raise FileNotFoundError(
            discovery.vocals_missing_message(paths.search_roots())
            + "\n\nOr set 'download' in YuE2 Options to 'auto', and the node will fetch it itself.")
    path = fetch_vocals(progress)
    if progress is not None:
        progress.text("Download finished")
    return path


def verify_folder(directory: str, progress=None) -> None:
    """Check a folder against the manifest that came down with it.

    Only the released layout has one, so this is silently nothing for the
    repack. The failure it reports is raised as a DownloadError because that is
    what the node already catches, and to a person a file that arrived wrong is
    a download that went wrong however the machine chooses to file it.
    """
    from . import manifest

    try:
        manifest.check(directory, progress, lambda: bool(_interrupted()))
    except manifest.Corrupt as error:
        raise DownloadError(str(error)) from error


def fetch_original(variant: str, progress=None) -> None:
    """The released files, into ComfyUI/models/YuE2 under their published names.

    The repository holds a whole paper beside the weights -- audio samples,
    figures, two wheels -- and none of it is any use to a node, so only the
    files the loader actually reads are asked for.
    """
    fetch_backbone(progress)
    fetch_decoder(variant, progress)


def fetch_backbone(progress=None) -> None:
    """m-a-p's backbone and vocabulary, the larger half of the released files."""
    lm_dir = os.path.join(paths.models_root(), LM_DIRNAME)
    fetch(LM_REPO, {name: os.path.join(lm_dir, name) for name in LM_ALLOW},
          "Downloading YuE2-3B (6.76 GB)", progress)
    verify_folder(lm_dir, progress)


def fetch_decoder(variant: str, progress=None) -> None:
    """One of m-a-p's two decoders, on its own.

    On its own is what a legacy run needs on a machine that has the backbone
    already, in either shape: see ``ensure``.
    """
    legacy = variant == "legacy"
    vae_dir = os.path.join(paths.models_root(), VAE_LEGACY_DIRNAME if legacy else VAE_DIRNAME)
    fetch(VAE_LEGACY_REPO if legacy else VAE_REPO,
          {name: os.path.join(vae_dir, name) for name in VAE_ALLOW},
          "Downloading the {} decoder (0.49 GB)".format("legacy" if legacy else "VAE"),
          progress)
    verify_folder(vae_dir, progress)


def writer_wanted() -> dict:
    """The default writer model, and where a GGUF belongs on this machine."""
    root = os.path.join(paths.models_root(), WRITER_SUBDIR)
    folder_paths = paths._folder_paths()
    if folder_paths is not None:
        try:
            registered = [path for path in folder_paths.get_folder_paths(WRITER_SUBDIR)
                          if os.path.isdir(path)]
        except KeyError:
            registered = []
        if registered:
            root = registered[0]
        else:
            root = os.path.join(folder_paths.models_dir, WRITER_SUBDIR)
    return {WRITER_PATH: os.path.join(root, WRITER_NAME)}


def fetch_writer(settings: dict, progress=None) -> str:
    """The writer's language model, when the machine has no GGUF of its own.

    Refused outright when downloading is off, and the refusal names the file
    rather than the repository: someone who keeps their models by hand wants a
    link and a folder, not an apology.
    """
    if settings.get("download", "auto") == "off":
        wanted = writer_wanted()
        destination = list(wanted.values())[0]
        raise FileNotFoundError(
            "The writer found no language model on this machine, and 'download' in YuE2 "
            "Options is 'off'.\n\n"
            "Put any instruction-following GGUF into your ComfyUI models folder, or fetch "
            "this one:\n\n"
            + endpoint() + "/" + WRITER_REPO + "/resolve/main/" + WRITER_PATH
            + "\n  -> " + destination
            + "\n\nThen pick it in the model widget, or leave that widget on '"
            + WRITER_AUTO + "'.")

    wanted = writer_wanted()
    fetch(WRITER_REPO, wanted,
          "Downloading the writer model ({})".format(human_size(WRITER_BYTES)), progress)
    return list(wanted.values())[0]


def source_for(settings: dict) -> str:
    """Which layout to fetch, once 'auto' has been resolved against the rest.

    Auto means the repack: one file instead of three, it is what the ComfyUI
    model manager installs, and it is what the native YuE2 nodes read, so the
    one download serves both. The exception is a setting the repack cannot
    satisfy -- the legacy decoder, which Comfy-Org does not publish. On a
    machine with nothing on it, m-a-p's backbone and that decoder are the
    smaller download (6.76 + 0.49 GB against 7.26 + 0.49); on a machine that
    has a backbone already, ``ensure`` fetches the decoder alone whatever this
    says.
    """
    choice = settings.get("download", "auto")
    if choice != "auto":
        return choice
    if settings.get("vae") == "legacy":
        return "original"
    return "comfy-org"


def ensure(settings: dict, progress=None):
    """The weights, fetched first if this machine has not got them yet.

    Nothing is downloaded when the files are already here, which is the usual
    case after the first run: that costs a few directory listings, not a
    request. When they are not here, what the user sees is a progress bar
    rather than a wall of instructions -- and when downloading is off, the wall
    of instructions is exactly what they asked for, with the links in it.

    A legacy run on a machine that has the backbone, as m-a-p's files or as
    Comfy-Org's BF16 repack, fetches the legacy decoder and nothing else.
    Before 2026-09-18 it fetched m-a-p's 6.76 GB backbone beside a repack that
    already held the same one, and with 'download' at 'comfy-org' it fetched
    nothing and refused, since the repack cannot carry that decoder. A standard
    run with m-a-p's backbone here and only its decoder missing fetches that
    decoder too, unless 'download' asks for Comfy-Org's file by name. A decoder
    the search already found somewhere else is not fetched a second time.
    """
    from . import discovery

    quantization = settings.get("quantization", "bf16")
    variant = settings.get("vae", "standard")
    try:
        return discovery.locate(variant, quantization)
    except FileNotFoundError as absent:
        source = source_for(settings)
        backbone = bool(getattr(absent, "backbone", False))
        decoder = bool((getattr(absent, "found", None) or {}).get("vae"))
        if source == "off":
            raise FileNotFoundError(
                str(absent) + "\n\nOr set 'download' in YuE2 Options to 'auto', "
                "and the node will fetch {} itself.".format(
                    "it" if backbone else "them")) from absent

    if backbone and (variant == "legacy" or settings.get("download", "auto") != "comfy-org"):
        fetch_decoder(variant, progress)
    elif source == "original":
        fetch_original(variant, progress)
    else:
        fetch_repack(quantization, progress)
        if variant == "legacy" and not decoder:
            fetch_decoder(variant, progress)
    if progress is not None:
        progress.text("Download finished")
    return discovery.locate(variant, quantization)
