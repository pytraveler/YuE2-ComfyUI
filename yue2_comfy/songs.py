"""Songs this pack has sung, remembered by the sound they made.

A sound file is not enough to sing one part of a song again. That takes what
the song was sung from -- the prompt with its score, the codec tokens of the
performance -- and what the acoustic stage made of them: the latents, and the
noise they were solved from. None of it is in the audio, so the nodes that sing
hand it here, and an edit finds it again from the audio alone.

The key is the SHA-256 of the samples as 16-bit PCM, rounded the way Save Audio
rounds them into a FLAC. A song handed straight on from the node that sang it
and the same song saved as FLAC and loaded back have one key. An MP3 or Opus
file has another: a lossy codec changes nearly every sample. Measured on
2026-09-19, 1.2 percent of the samples of an MP3 at 320k came back as they
went in.

Songs are held in memory for the session and written to disk, into
ComfyUI/user/yue2_comfy/songs, each within a budget that lets the song used
longest ago go first. The disk copy is what lets a FLAC saved yesterday be
edited today, after a restart. ComfyUI's temp folder would not do: ComfyUI
empties it every time it starts.

Only ``key`` needs torch, to quantize a waveform. The store itself is plain
Python, so it is tested where CI runs, without torch or numpy.
"""

from __future__ import annotations

import array
import collections
import dataclasses
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import zipfile

from . import paths
from .constants import normalize_seed

log = logging.getLogger(__name__)

FORMAT = 1
"""The layout of a song file. A file written in any other is left alone and not read."""

FOLDER = "songs"

SUFFIX = ".song"

LATENT_WIDTH = 64

CODEC_SIZE = 32768

MEMORY_BYTES = 128 << 20
"""How much of the session's memory remembered songs may hold: 128 MiB.

A song costs 256 bytes of latents a frame, 6.4 KB a second: 1.2 MB for three
minutes, 2.3 MB for six, so this holds a hundred songs or more, the edits of a
session included."""

DISK_BYTES = 1 << 30
"""How much disk remembered songs may take: 1 GiB.

The acoustic stage runs in bf16, so half of every float32 latent is zeros, and
the file deflates to about half. Measured on three songs on 2026-09-19: 52
percent at level 1, in 8 to 17 ms. A three-minute song takes about 0.6 MB, so
this holds well over a thousand."""

COMPRESS_LEVEL = 1

STALE_SECONDS = 600
"""How old a half-written file must be before it counts as left over from a crash."""

PCM_SCALE = 32768.0

CHUNK_SAMPLES = 1 << 20
"""How many samples ``key`` quantizes at a time, so a six-minute song never
needs a second copy of itself in memory."""

_KEY = re.compile(r"[0-9a-f]{64}")


@dataclasses.dataclass
class Song:
    """Everything needed to sing a part of a song again, apart from its sound.

    'origin' names the node that made it. 'style', 'lyrics', 'seed' and
    'settings' are what it was sung under, and an edit samples the way these
    settings say. 'transpose' among them has already been applied: 'score' and
    'prefix' hold the moved score. 'score' is the text of the score that was
    sung, empty for a song sung with cot 'off'.

    'prefix', 'negative' and 'codec' are token ids, as generate.Performance
    holds them, kept as compact arrays. 'noise' says where the noise of each
    frame comes from, as [seed, offset, count] runs in order: frames offset to
    offset + count of the full-song draw the acoustic stage makes from that
    seed. A song straight from a singing node is one run. An edited one splices
    runs of the song before it with runs of the edit's own. 'latents' is the
    acoustic stage's result, frames x 64 float32 as little-endian bytes.
    'sample_rate', 'channels' and 'samples' describe the audio it became.
    """

    origin: str
    style: str
    lyrics: str
    seed: int
    settings: dict
    score: str
    prefix: array.array
    negative: array.array | None
    codec: array.array
    noise: list
    latents: bytes
    sample_rate: int
    channels: int
    samples: int

    @property
    def frames(self) -> int:
        return len(self.codec)

    def size(self) -> int:
        """About how many bytes this song holds in memory, for the memory budget."""
        tokens = (self.prefix.itemsize * len(self.prefix) + self.codec.itemsize * len(self.codec)
                  + (0 if self.negative is None else self.negative.itemsize * len(self.negative)))
        text = len(self.style) + len(self.lyrics) + len(self.score)
        return len(self.latents) + tokens + text + 1024

    def check(self) -> None:
        """Refuse a song whose parts do not fit together, with a ValueError saying which.

        A song that fails here would fail later in the middle of an edit, far
        from whatever wrote it wrong, so both the store's writes and its reads
        come through here.
        """
        frames = self.frames
        if frames < 1:
            raise ValueError("a song needs at least one frame of codec tokens")
        if min(self.codec) < 0:
            raise ValueError("a codec token is outside 0..{}".format(CODEC_SIZE - 1))
        if not self.prefix:
            raise ValueError("a song needs the prompt it was sung from")
        if len(self.latents) != frames * LATENT_WIDTH * 4:
            raise ValueError("{} bytes of latents do not make {} frames of {} floats".format(
                len(self.latents), frames, LATENT_WIDTH))
        covered = 0
        for run in self.noise:
            if (len(run) != 3 or any(isinstance(value, bool) or not isinstance(value, int)
                                     for value in run)
                    or run[0] < 0 or run[1] < 0 or run[2] < 1):
                raise ValueError("a noise run is [seed, offset, count], not {!r}".format(run))
            covered += run[2]
        if covered != frames:
            raise ValueError("the noise covers {} frames of {}".format(covered, frames))
        if self.sample_rate < 1 or self.channels < 1 or self.samples < 1:
            raise ValueError("a song needs a sample rate, channels and samples")


def key(waveform, sample_rate) -> str:
    """The name a song is remembered by: SHA-256 of its samples as 16-bit PCM.

    Save Audio hands FFmpeg float samples and asks for a 16-bit FLAC, and FFmpeg
    rounds each to the nearest step of 1/32768, halves to even, clipped to the
    int16 range. Load Audio divides what it reads by 32768. Rounding the same
    way here makes both ends of that trip one key. Measured on 2026-09-19 with
    the calls of ComfyUI 0.36 and PyAV 17.0.1: every sample of three seconds of
    noise, halves and clipped values came back as this rounding says. Rounding
    halves up would have missed 296 of 288000, because float32 samples near
    full scale land on halves that often.

    Takes [channels, samples], or a batch of one, [1, channels, samples]. The
    waveform itself is never written to.
    """
    import torch

    samples = waveform.detach()
    if samples.dim() == 3:
        if samples.shape[0] != 1:
            raise ValueError("A batch of {} waveforms is {} songs; a song is remembered one at a "
                             "time.".format(samples.shape[0], samples.shape[0]))
        samples = samples[0]
    if samples.dim() != 2:
        raise ValueError("A waveform is [channels, samples], not {}".format(tuple(samples.shape)))
    samples = samples.to(device="cpu", dtype=torch.float32).contiguous()
    channels, length = samples.shape
    digest = hashlib.sha256("yue2 song pcm16 {} {} {}\n".format(
        int(sample_rate), int(channels), int(length)).encode("ascii"))
    flat = samples.reshape(-1)
    for start in range(0, flat.numel(), CHUNK_SAMPLES):
        pcm = (flat[start:start + CHUNK_SAMPLES] * PCM_SCALE).round_().clamp_(-32768, 32767)
        digest.update(pcm.to(torch.int16).numpy().astype("<i2", copy=False).tobytes())
    return digest.hexdigest()


def _named(name) -> str:
    """*name*, when it is a key; a ValueError otherwise, since it becomes a file name."""
    if not isinstance(name, str) or not _KEY.fullmatch(name):
        raise ValueError("{!r} is not a song key".format(name))
    return name


def _numbers(typecode: str, values) -> array.array:
    """Integers as an array of *typecode*, from a sequence or from little-endian bytes."""
    result = array.array(typecode)
    if isinstance(values, (bytes, bytearray)):
        result.frombytes(values)
        if sys.byteorder == "big":
            result.byteswap()
    else:
        result.extend(int(value) for value in values)
    return result


def _little(numbers: array.array) -> bytes:
    """An array as little-endian bytes, whatever this machine's own order is."""
    if sys.byteorder == "big":
        numbers = array.array(numbers.typecode, numbers)
        numbers.byteswap()
    return numbers.tobytes()


def _floats(latents) -> bytes:
    """The latents as float32 little-endian bytes, from a tensor, a numpy array or bytes."""
    if isinstance(latents, (bytes, bytearray)):
        return bytes(latents)
    if hasattr(latents, "detach"):
        latents = latents.detach().to("cpu").float().contiguous().numpy()
    return latents.astype("<f4").tobytes()


def _scalar(value) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _plain(settings) -> dict:
    """The settings that can be written down: text, numbers, switches, and lists of plain rows.

    The rows are the adapters a song was sung with (``lora.node``), each a
    name, a file, its SHA-256 and two strengths, so that an edit of the song
    later folds the same ones.
    """
    kept = {}
    for name, value in dict(settings or {}).items():
        if _scalar(value):
            kept[str(name)] = value
        elif isinstance(value, (list, tuple)) and all(
                isinstance(row, dict) and all(_scalar(item) for item in row.values())
                for row in value):
            kept[str(name)] = [{str(key): item for key, item in row.items()} for row in value]
    return kept


def _entries(song: Song) -> list:
    """The files inside a song file, as (name, bytes).

    song.json is UTF-8 with every letter as it is, so lyrics in any alphabet
    read as words when the file is opened with an archiver.
    """
    meta = {"format": FORMAT, "origin": song.origin, "style": song.style,
            "lyrics": song.lyrics, "seed": song.seed, "settings": song.settings,
            "score": song.score, "noise": song.noise, "frames": song.frames,
            "negative": song.negative is not None, "sample_rate": song.sample_rate,
            "channels": song.channels, "samples": song.samples}
    entries = [("song.json", json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8")),
               ("prefix.i32", _little(song.prefix)),
               ("codec.i16", _little(song.codec)),
               ("latents.f32", song.latents)]
    if song.negative is not None:
        entries.append(("negative.i32", _little(song.negative)))
    return entries


def _unpacked(archive: zipfile.ZipFile):
    """The song in an open song file, or None when it was written in another format."""
    meta = json.loads(archive.read("song.json").decode("utf-8"))
    if meta.get("format") != FORMAT:
        return None
    song = Song(
        origin=str(meta["origin"]), style=str(meta["style"]), lyrics=str(meta["lyrics"]),
        seed=int(meta["seed"]), settings=dict(meta["settings"]), score=str(meta["score"]),
        prefix=_numbers("i", archive.read("prefix.i32")),
        negative=_numbers("i", archive.read("negative.i32")) if meta["negative"] else None,
        codec=_numbers("h", archive.read("codec.i16")),
        noise=[[int(value) for value in run] for run in meta["noise"]],
        latents=archive.read("latents.f32"), sample_rate=int(meta["sample_rate"]),
        channels=int(meta["channels"]), samples=int(meta["samples"]))
    song.check()
    return song


class Store:
    """Songs by key: the ones used last in memory, every one on disk until the disk budget is spent.

    With 'folder' None songs are kept in memory only. The lock is there because
    ComfyUI answers HTTP requests on another thread than the one that runs the
    nodes, and a route that reads songs is one of the things an editor wants.
    """

    def __init__(self, folder, memory_bytes: int = MEMORY_BYTES, disk_bytes: int = DISK_BYTES):
        self.folder = folder
        self.memory_bytes = int(memory_bytes)
        self.disk_bytes = int(disk_bytes)
        self._recent = collections.OrderedDict()
        self._held = 0
        self._lock = threading.Lock()

    def put(self, name: str, song: Song) -> None:
        """Keep *song* under *name*, in memory and, when it can be written, on disk.

        A disk that refuses is logged and not raised: the song is still in
        memory, so it can be edited until ComfyUI restarts.
        """
        _named(name)
        song.check()
        with self._lock:
            self._hold(name, song)
            if self.folder is None:
                return
            try:
                self._write(name, song)
            except OSError as error:
                log.warning("[yue2_comfy.songs] could not write %s (%s). The song can be "
                            "edited until ComfyUI restarts, not after.", self._path(name), error)

    def get(self, name: str):
        """The song kept under *name*, or None when there is none."""
        _named(name)
        with self._lock:
            song = self._recent.get(name)
            if song is not None:
                self._recent.move_to_end(name)
                return song
            song = None if self.folder is None else self._read(name)
            if song is not None:
                self._hold(name, song)
            return song

    def _path(self, name: str) -> str:
        return os.path.join(self.folder, name + SUFFIX)

    def _hold(self, name: str, song: Song) -> None:
        """Put *song* in memory as the newest, and let the oldest go past the budget.

        The newest stays even when it alone is over the budget: it is the song
        that is about to be used.
        """
        previous = self._recent.pop(name, None)
        if previous is not None:
            self._held -= previous.size()
        self._recent[name] = song
        self._held += song.size()
        while self._held > self.memory_bytes and len(self._recent) > 1:
            _name, dropped = self._recent.popitem(last=False)
            self._held -= dropped.size()

    def _write(self, name: str, song: Song) -> None:
        """One file per song, written aside and moved into place, so no reader sees half of one."""
        os.makedirs(self.folder, exist_ok=True)
        path = self._path(name)
        partial = "{}.{}.partial".format(path, os.getpid())
        try:
            with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=COMPRESS_LEVEL) as archive:
                for entry, blob in _entries(song):
                    archive.writestr(entry, blob)
            os.replace(partial, path)
        finally:
            if os.path.exists(partial):
                os.remove(partial)
        self._trim(path)

    def _trim(self, kept: str) -> None:
        """Delete the songs used longest ago until the folder is within its budget.

        Reading a song touches its file, so the order is the order of use, not
        of making. 'kept', the file just written, is never deleted, even when it
        alone is over the budget. Half-written files older than STALE_SECONDS
        are what a crash left behind and go too.
        """
        now = time.time()
        found = []
        with os.scandir(self.folder) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                stat = entry.stat()
                if entry.name.endswith(".partial"):
                    if now - stat.st_mtime > STALE_SECONDS:
                        _remove(entry.path)
                    continue
                if entry.name.endswith(SUFFIX):
                    found.append((stat.st_mtime, entry.path, stat.st_size))
        total = sum(size for _mtime, _path, size in found)
        for _mtime, path, size in sorted(found):
            if total <= self.disk_bytes:
                break
            if os.path.normcase(path) == os.path.normcase(kept):
                continue
            if _remove(path):
                total -= size

    def _read(self, name: str):
        """The song on disk under *name*, or None when there is none or it cannot be read."""
        path = self._path(name)
        if not os.path.isfile(path):
            return None
        try:
            with zipfile.ZipFile(path) as archive:
                song = _unpacked(archive)
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
            log.warning("[yue2_comfy.songs] %s cannot be read and is ignored: %s", path, error)
            return None
        if song is None:
            log.info("[yue2_comfy.songs] %s was written in another format and is ignored", path)
            return None
        try:
            os.utime(path, None)
        except OSError:
            log.debug("[yue2_comfy.songs] could not touch %s", path, exc_info=True)
        return song


def _remove(path: str) -> bool:
    """Delete one file, and say whether it went."""
    try:
        os.remove(path)
    except OSError:
        log.debug("[yue2_comfy.songs] could not delete %s", path, exc_info=True)
        return False
    return True


_store = None


def store() -> Store:
    """The store every node shares, in ComfyUI/user/yue2_comfy/songs."""
    global _store
    if _store is None:
        _store = Store(os.path.join(paths.user_dir(), FOLDER))
    return _store


def remember(waveform, sample_rate, song: Song) -> str:
    """Keep *song* under the key of *waveform*, and return the key."""
    began = time.perf_counter()
    name = key(waveform, sample_rate)
    store().put(name, song)
    log.info("[yue2_comfy.songs] remembered for editing as %s: %d frames, %.1f MB, in %.2f s",
             name[:12], song.frames, song.size() / 1e6, time.perf_counter() - began)
    return name


def recall(waveform, sample_rate):
    """The song *waveform* is the sound of, or None when this pack does not know it."""
    return store().get(key(waveform, sample_rate))


def keep(audio, origin, style, lyrics, seed, settings, score, performance):
    """Remember a song a node has just sung. Returns its key, or None when it was not kept.

    Never raises. The song is finished by the time it gets here and is worth
    more than the memory of it, so a failure is logged with its traceback and
    the node hands its audio on as it would have without this module. A node
    that has no performance to give, None, keeps nothing.
    """
    if performance is None:
        return None
    try:
        waveform = audio["waveform"]
        codec = _numbers("h", performance.codec)
        song = Song(
            origin=origin, style=style or "", lyrics=lyrics or "", seed=normalize_seed(seed),
            settings=_plain(settings), score=score or "",
            prefix=_numbers("i", performance.prefix),
            negative=None if performance.negative is None else _numbers("i", performance.negative),
            codec=codec, noise=[[int(performance.seed), 0, len(codec)]],
            latents=_floats(performance.latents), sample_rate=int(audio["sample_rate"]),
            channels=int(waveform.shape[-2]), samples=int(waveform.shape[-1]))
        return remember(waveform, song.sample_rate, song)
    except Exception:
        log.warning("[yue2_comfy.songs] this song could not be remembered, so it cannot be "
                    "edited", exc_info=True)
        return None
