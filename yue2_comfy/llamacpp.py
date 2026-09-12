"""The official llama.cpp binaries, fetched on demand.

``llama-cpp-python`` is the fast path when it is already installed, and a
dependency nobody should have to fight when it is not. Its prebuilt CUDA wheels
fail on ordinary consumer hardware in two unrelated ways -- the ones built
against CUDA 13.0 are compiled with AVX-512 and die with ``0xC000001D`` on any
consumer Intel 12th to 14th generation chip, and the 13.2 ones drop AVX-512 but
ship PTX that a driver older than CUDA 13.2 refuses to compile -- and building a
wheel from source needs a compiler, a CUDA toolkit and an hour. Both failures
were measured in the pack this module comes from, not read about.

The upstream release archives have neither problem. They carry fourteen CPU
backend variants and pick one at run time, which is exactly why the same model
runs under ``llama-completion`` on a machine where the wheel crashes. The CUDA
archive carries native SASS and no PTX at all, so the driver never has to
compile anything.

So: if the wheel is there, use it; otherwise fetch about thirty megabytes of
official binaries and write the song in a subprocess. Nobody is asked to choose,
because a person who wanted to think about llama.cpp builds would not be using a
node whose point is not having to.

``auto`` takes CUDA where the card can actually run it and Vulkan everywhere
else. Vulkan is 32 MB against 511 MB, works on AMD and Intel too, and is the
only build upstream ships for Linux with any GPU acceleration at all, at roughly
half the tokens per second.

None of which applies when the machine already has an llama.cpp:
``YUE2_LLAMA_BIN`` names one outright, ``llama_bin.txt`` in the user directory
says the same to a server whose environment you cannot set, PATH is read when
neither is there, and whatever is found is run as it is. That is the only road
to a CUDA llama.cpp on Linux, since no CUDA archive is published for it.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tarfile
import zipfile

from . import paths

log = logging.getLogger(__name__)

RELEASE = "b10310"
"""Pinned, so a workflow behaves the same next month. Raise it deliberately."""

REPO = "ggml-org/llama.cpp"
DOWNLOAD_URL = "https://github.com/{}/releases/download/{}".format(REPO, RELEASE)

BACKENDS = ("auto", "vulkan", "cuda", "cpu")
DEFAULT_BACKEND = "vulkan"

BIN_ENV = "YUE2_LLAMA_BIN"
"""An llama.cpp this machine already has, named outright.

Either the executable or the directory the binaries live in. Set it when the
wanted build is not the first one on PATH, or when there is no PATH entry at
all -- a container that runs ComfyUI as a service usually has none.
"""

BIN_FILE = "llama_bin.txt"
"""The same thing written down, for a server whose environment is not yours.

An export in a shell reaches a server started from that shell and nothing else:
a systemd unit, a container entrypoint or a launcher script hands the process an
environment of its own, and the variable above is simply absent there. This file
is read from ComfyUI's own user directory instead -- one path in it, blank lines
and '#' comments ignored.
"""

ASSETS = {
    ("win32", "vulkan"): ("llama-{}-bin-win-vulkan-x64.zip".format(RELEASE),),
    ("win32", "cuda"): (
        "llama-{}-bin-win-cuda-13.3-x64.zip".format(RELEASE),
        "cudart-llama-bin-win-cuda-13.3-x64.zip",
    ),
    ("win32", "cpu"): ("llama-{}-bin-win-cpu-x64.zip".format(RELEASE),),
    ("linux", "vulkan"): ("llama-{}-bin-ubuntu-vulkan-x64.tar.gz".format(RELEASE),),
    ("linux", "cpu"): ("llama-{}-bin-ubuntu-x64.tar.gz".format(RELEASE),),
    ("darwin", "cpu"): ("llama-{}-bin-macos-arm64.tar.gz".format(RELEASE),),
}
"""(platform, backend) -> the archive names within the release.

The CUDA entry is two archives: the runtime libraries live in a separate
``cudart`` download, and without them the CUDA backend silently fails to load --
``--list-devices`` prints "(none)" and everything falls back to the CPU.
"""

UNAVAILABLE = {
    ("linux", "cuda"): (
        "upstream publishes no CUDA build of llama.cpp for Linux; the Vulkan build "
        "runs on NVIDIA cards too, and one you compiled yourself is run as it is. "
        "Build llama.cpp itself -- 'cmake -B build -DGGML_CUDA=ON' then 'cmake "
        "--build build --target llama-completion' -- and give its bin folder to the "
        "node: write it into " + BIN_FILE + " in ComfyUI's user directory, name it "
        "in " + BIN_ENV + ", or put it on PATH"
    ),
    ("darwin", "cuda"): "macOS has no CUDA; the macOS build uses Metal",
    ("darwin", "vulkan"): "upstream publishes no Vulkan build for macOS",
}
"""Where asking for a backend is a mistake worth naming, not a silent fallback."""

EXE = ".exe" if sys.platform == "win32" else ""

BINARIES = ("llama-completion" + EXE, "llama-cli" + EXE)
"""``llama-completion``, emphatically not ``llama-cli``.

As of b10310 ``llama-cli`` is an interactive terminal UI: it draws a spinner and
an ASCII banner to stdout and then waits, so a one-shot run never ends and the
caller reads hundreds of kilobytes of animation frames instead of a song.
``llama-completion`` takes the same arguments and writes nothing but the
completion. Older builds shipped only ``llama-cli``, which behaved like today's
``llama-completion``, so that name stays as a fallback.
"""

SUBDIR = "runtime"

CUDA_CAPABILITIES = {(8, 6), (8, 9), (12, 0), (12, 1)}
"""Compute capabilities the CUDA archive actually carries code for.

It ships native SASS and no PTX, which is what makes it immune to the
driver-versus-toolkit problem that kills the CUDA 13.2 wheel. The price is that
a card outside this set has nothing to run and nothing to compile from, so
'auto' must send it to Vulkan rather than to a 511 MB download it cannot use.
"""

_ANNOUNCED = set()


def nvidia_capability():
    """The compute capability of the card ComfyUI is using, if it is NVIDIA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return tuple(torch.cuda.get_device_capability(0))
    except Exception:
        log.debug("[yue2_comfy.llamacpp] no CUDA device visible", exc_info=True)
        return None


def resolve_backend(backend: str) -> str:
    """Turn 'auto' into the fastest backend this machine can actually run.

    Not the smallest download: on Windows with a supported NVIDIA card that
    means CUDA, which is roughly twice as fast as Vulkan and roughly fifteen
    times the download. Anyone who would rather keep the 32 MB version can pick
    'vulkan' by name.
    """
    backend = (backend or "auto").strip().lower()
    if backend and backend != "auto":
        return backend
    if nvidia_capability() in CUDA_CAPABILITIES and (sys.platform, "cuda") in ASSETS:
        return "cuda"
    if (sys.platform, DEFAULT_BACKEND) in ASSETS:
        return DEFAULT_BACKEND
    return "cpu"


def assets(backend: str) -> tuple:
    key = (sys.platform, backend)
    if key in UNAVAILABLE:
        raise RuntimeError("llama.cpp " + backend + ": " + UNAVAILABLE[key] + ".")
    names = ASSETS.get(key)
    if not names:
        here = ", ".join(sorted(b for p, b in ASSETS if p == sys.platform))
        raise RuntimeError(
            "No llama.cpp " + backend + " build is published for " + sys.platform
            + ". Available here: " + here + ".")
    return names


def root() -> str:
    """Where fetched runtimes live: in ComfyUI's user folder, not in the pack."""
    return os.path.join(paths.user_dir(), SUBDIR)


def install_dir(backend: str) -> str:
    return os.path.join(root(), RELEASE + "-" + backend)


def bin_file() -> str:
    """``<ComfyUI user>/yue2_comfy/llama_bin.txt``, read if it is there."""
    return os.path.join(paths.user_dir(), BIN_FILE)


def find_binary(directory: str) -> str:
    """Locate the completion executable inside an unpacked release, at any depth."""
    for name in BINARIES:
        direct = os.path.join(directory, name)
        if os.path.isfile(direct):
            return direct
    for name in BINARIES:
        for current, _dirs, files in os.walk(directory):
            if name in files:
                return os.path.join(current, name)
    return ""


def installed(backend: str) -> str:
    """Path to the completion binary for this backend, or "" if not unpacked."""
    directory = install_dir(backend)
    return find_binary(directory) if os.path.isdir(directory) else ""


def _binary_at(value: str, source: str) -> str:
    """Resolve a path somebody gave us to a binary, or say why it is not one.

    A path that is given but points nowhere useful raises rather than falls
    through: naming a build is an instruction, and quietly downloading a
    different one instead would hide a typo behind half a gigabyte.
    """
    value = (value or "").strip().strip('"')
    if not value:
        return ""
    if EXE and not os.path.exists(value) and os.path.isfile(value + EXE):
        value += EXE

    wanted = ", ".join(BINARIES)
    if os.path.isfile(value):
        if os.path.basename(value) in BINARIES:
            return value
        beside = find_binary(os.path.dirname(os.path.abspath(value)))
        if beside:
            return beside
        raise RuntimeError(
            source + " is '" + value + "', and neither " + wanted
            + " is that file or sits beside it.")
    if os.path.isdir(value):
        found = find_binary(value)
        if found:
            return found
        raise RuntimeError(source + " is '" + value + "', which holds neither " + wanted + ".")
    raise RuntimeError(source + " is '" + value + "', which does not exist.")


def _named_binary() -> str:
    """The binary ``BIN_ENV`` points at, or "" when the variable is unset."""
    return _binary_at(os.environ.get(BIN_ENV) or "", BIN_ENV)


def _file_binary() -> str:
    """The binary ``BIN_FILE`` names, or "" when there is no such file."""
    path = bin_file()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle]
    except OSError:
        return ""
    wanted = ""
    for line in lines:
        if line and not line.startswith("#"):
            wanted = line
            break
    return _binary_at(wanted, path)


def _path_binary() -> str:
    """The first of ``BINARIES`` on PATH: an llama.cpp built on this machine."""
    for name in BINARIES:
        found = shutil.which(name)
        if found:
            return os.path.abspath(found)
    return ""


def _announce(binary: str, source: str) -> str:
    if binary not in _ANNOUNCED:
        _ANNOUNCED.add(binary)
        log.info("[yue2_comfy.llamacpp] llama.cpp %s: %s", source, binary)
    return binary


def external() -> str:
    """An llama.cpp that is already here: given by name, or found on PATH."""
    return _named_binary() or _file_binary() or _path_binary()


def available() -> bool:
    """True when a runtime is reachable: unpacked here, or already on the machine."""
    for platform, backend in ASSETS:
        if platform == sys.platform and installed(backend):
            return True
    try:
        return bool(external())
    except RuntimeError:
        return False


def asset_sizes(names: tuple) -> dict:
    from . import download

    return {name: download.content_length(DOWNLOAD_URL + "/" + name) for name in names}


def download_size(backend: str) -> int:
    return sum(asset_sizes(assets(resolve_backend(backend))).values())


def _safe_extract(archive: str, destination: str) -> None:
    """Unpack, refusing any member that would land outside the destination.

    The tar path asks for the 'data' filter as well, which applies the same rule
    inside the standard library and becomes the default in Python 3.14. It is
    belt and braces -- every dangerous member has already been refused above --
    but a pack that lives inside somebody else's interpreter should not be the
    reason their next upgrade prints a warning. Older interpreters have no such
    keyword and keep the checks they always had.
    """
    destination = os.path.abspath(destination)

    def inside(path: str) -> bool:
        target = os.path.abspath(os.path.join(destination, path))
        return target == destination or target.startswith(destination + os.sep)

    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                if not inside(member):
                    raise RuntimeError(
                        "refusing archive member outside the target: " + repr(member))
            handle.extractall(destination)
        return

    def link_target_inside(member) -> bool:
        """Whether a link points somewhere the extraction is allowed to reach.

        SONAME symlinks -- libllama.so.0 pointing at libllama.so.0.0.10310 --
        ship in every Linux and macOS release, and the binaries link against the
        link rather than the file behind it, so refusing links outright refuses
        the archive. Only a target that leaves the destination is the risk.

        The two kinds resolve from different places, and it matters. tarfile
        writes a symlink's linkname verbatim, so it is read relative to the
        link's own directory; a hardlink's is joined onto the extraction root
        instead. Using one base for both counts the link's own depth twice and
        waves through 'build/bin/x -> ../../secret', two levels above the
        destination, as though it were 'secret' inside it.
        """
        if os.path.isabs(member.linkname):
            return False
        base = os.path.dirname(member.name) if member.issym() else ""
        return inside(os.path.normpath(os.path.join(base, member.linkname)))

    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if not inside(member.name):
                raise RuntimeError(
                    "refusing archive member outside the target: " + repr(member.name))
            if (member.issym() or member.islnk()) and not link_target_inside(member):
                raise RuntimeError(
                    "refusing archive link pointing outside the target: "
                    + repr(member.name) + " -> " + repr(member.linkname))
        try:
            handle.extractall(destination, filter="data")
        except TypeError:
            handle.extractall(destination)


def _make_executable(directory: str) -> None:
    """Restore the bit the tar releases carry and a zip cannot."""
    if sys.platform == "win32":
        return
    for current, _dirs, files in os.walk(directory):
        for name in files:
            path = os.path.join(current, name)
            if name.startswith("llama-") or name.endswith(".so") or ".so." in name:
                try:
                    os.chmod(path, os.stat(path).st_mode | 0o111)
                except OSError:
                    log.debug("[yue2_comfy.llamacpp] chmod failed: %s", path)


def _fetch(backend: str, names: tuple, staging: str, progress=None) -> None:
    """Put every archive of one backend into the staging folder."""
    from . import download

    if progress is not None:
        progress.text("Fetching llama.cpp " + backend + "\nasking for the download size",
                      force=True)
    sizes = asset_sizes(names)
    total = sum(sizes.values()) if all(sizes.values()) else 0
    os.makedirs(staging, exist_ok=True)
    download.check_space(staging, total * 3 if total else 0)

    reporter = None
    if progress is not None and total:
        from .progress import TransferReporter

        reporter = TransferReporter(progress, total, "Fetching llama.cpp " + backend)

    transferred = 0
    for name in names:
        destination = os.path.join(staging, name)
        expected = sizes.get(name, 0)
        if expected and os.path.isfile(destination) and os.path.getsize(destination) == expected:
            transferred += expected
            if reporter is not None:
                reporter(transferred, name)
            continue

        def report(position, label, _name=name):
            """Drive the bar, or say why there is none rather than look stuck."""
            if reporter is not None:
                reporter(position, _name)
            elif progress is not None:
                progress.text(
                    "Fetching llama.cpp " + backend + " - " + _name + "\n"
                    + download.human_size(position) + " (total size unknown, no bar)")

        transferred += download.fetch_url(
            DOWNLOAD_URL + "/" + name, destination, expected, transferred, report, name)


def _unpack(backend: str, names: tuple, staging: str, progress=None) -> None:
    for index, name in enumerate(names, start=1):
        if progress is not None:
            progress.set_total(len(names))
            progress.update(index - 1, "Unpacking llama.cpp " + backend + "\n" + name
                            + " ({}/{})".format(index, len(names)))
        _safe_extract(os.path.join(staging, name), staging)
        os.remove(os.path.join(staging, name))
    if progress is not None:
        progress.update(len(names), "Unpacking llama.cpp " + backend + "\ndone")
    _make_executable(staging)


def _packaged(backend: str, auto_download: bool, progress=None) -> str:
    """The pack's own runtime: the unpacked one, or a fresh download of it."""
    from . import download

    backend = resolve_backend(backend)
    existing = installed(backend)
    if existing:
        return existing

    try:
        names = assets(backend)
    except RuntimeError as error:
        raise RuntimeError(str(error) + "\n\n" + where_looked(backend)) from error
    directory = install_dir(backend)

    if not auto_download:
        raise RuntimeError(
            "The llama.cpp " + backend + " runtime is not in '" + directory
            + "' and downloading is off. Set 'download' back to 'auto', or unpack "
            + ", ".join(names) + " from https://github.com/" + REPO + "/releases/tag/"
            + RELEASE + " into that folder.\n\n" + where_looked(backend))

    staging = directory + ".part"
    try:
        _fetch(backend, names, staging, progress)
    except download.DownloadError as error:
        raise RuntimeError(str(error) + "\n\n" + by_hand(backend)) from error
    _unpack(backend, names, staging, progress)

    if not find_binary(staging):
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            "The llama.cpp " + backend + " archive unpacked without either of "
            + ", ".join(BINARIES) + " in it. Release " + RELEASE
            + " may have changed its layout.")

    if os.path.isdir(directory):
        shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(os.path.dirname(directory), exist_ok=True)
    os.replace(staging, directory)

    binary = find_binary(directory)
    log.info("[yue2_comfy.llamacpp] %s runtime ready at %s", backend, binary)
    return binary


def where_looked(backend: str) -> str:
    """The places that were searched and what each held, as a block of text.

    "Put it on PATH" is useless advice to somebody who did exactly that in a
    shell the server never saw, so a refusal reports the environment this
    process actually has rather than repeating the instruction.
    """
    named = os.environ.get(BIN_ENV)
    entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    shown = os.pathsep.join(entries)
    if len(shown) > 400:
        shown = shown[:400] + " ..."
    directory = install_dir(resolve_backend(backend))
    written = bin_file()

    lines = [
        "Where this ComfyUI process (pid {}) looked:".format(os.getpid()),
        "  " + BIN_ENV + ": " + ("'" + named + "'" if named else "not set in this process"),
        "  " + written + ": " + ("read" if os.path.isfile(written) else "no such file"),
        "  unpacked runtime: " + directory
        + ("" if os.path.isdir(directory) else " (not there)"),
        "  PATH: {} entries, none holding {}".format(len(entries), " or ".join(BINARIES)),
        "    " + shown,
    ]
    if not named:
        lines.append(
            "An export in your shell reaches a server started from that shell and "
            "nothing else -- a service is handed an environment of its own.")
    lines.append(
        "The way that needs no environment at all: write the path to your build "
        "into " + written + ", or put the build itself in " + directory + ".")
    return "\n".join(lines)


def by_hand(backend: str) -> str:
    """Installing the runtime without a connection, in full.

    A machine that is offline stays offline, and a stack of identical connection
    errors is not advice. The archive is public, small and needed only once, so
    the way through is the way the weights came: fetch it somewhere else and
    unpack it where this looked.
    """
    backend = resolve_backend(backend)
    names = ASSETS.get((sys.platform, backend), ())
    lines = [
        "This runtime can also be installed by hand, which is the way through on a "
        "machine with no connection. Fetch these on one that has:",
    ]
    lines += ["  " + DOWNLOAD_URL + "/" + name for name in names]
    lines += [
        "and unpack them into:",
        "  " + install_dir(backend),
        "That folder name is where the node looks, so nothing else is needed, and "
        "the layout inside the archive does not matter -- the binaries are found at "
        "any depth.",
    ]
    if sys.platform != "win32":
        lines.append("Unpack with 'tar -xf', not a file manager: the archive ships its "
                     "shared libraries as symlinks and they have to stay symlinks.")
    return "\n".join(lines)


def ensure(backend: str = "auto", auto_download: bool = True, progress=None) -> str:
    """The path to the completion binary, fetching the release if it must.

    Five places, in this order: what ``BIN_ENV`` names and what ``BIN_FILE``
    names, because giving a path is an instruction; then what this pack has
    already unpacked, which is the pinned, known-good one; then PATH, so an
    llama.cpp compiled on this machine is used rather than a second copy
    downloaded beside it; then the download.

    Those given paths are the only road to a CUDA llama.cpp on Linux, where
    upstream publishes no CUDA archive at all. They are also why the backend
    stops mattering once a build is found: it picks which archive to fetch, not
    what an existing binary was compiled against.
    """
    named = _named_binary()
    if named:
        return _announce(named, "named in " + BIN_ENV)

    written = _file_binary()
    if written:
        return _announce(written, "named in " + bin_file())

    backend = resolve_backend(backend)
    existing = installed(backend)
    if existing:
        return existing

    on_path = _path_binary()
    if on_path:
        return _announce(on_path, "found on PATH")

    return _packaged(backend, auto_download, progress)
