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

What else a song has sits beside its file under the same key: ``.alias``, the
other name its sound has; ``.note``, the word the person put on it;
``.flac``, its own sound, when the store is asked to keep it; and
``.grid.json``, where 'YuE2 Edit Track' found the score on it. All of them go
when the song does, whether it is deleted by hand or dropped by the budget.

Only ``key`` needs torch, to quantize a waveform. The store itself is plain
Python, so it is tested where CI runs, without torch or numpy.
"""

from __future__ import annotations

import array
import base64
import collections
import dataclasses
import hashlib
import io
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

ALIAS_SUFFIX = ".alias"
"""What a song's other key is written in: the key it points at, and nothing else."""

ALIASES_HELD = 256
"""How many other keys a session keeps in memory, for a store with no folder to write them in."""

NOTE_SUFFIX = ".note"
"""Where a song's own note waits: beside the song rather than inside it.

A note is the one thing about a song that the song itself does not hold --
what the person who made it wanted it for. Putting it in the file would mean
writing the three quarters of a megabyte the latents take again every time a
word of it changes, and a note is worth keeping even for a song written in a
format this version cannot read."""

NOTE_LETTERS = 80
"""How long a note may be: a badge on a row, not a second set of lyrics."""

SOUND_SUFFIX = ".flac"
"""Where a song's own sound waits, when the store is asked to keep it; see ``sound_keep``."""

SETTINGS_NAME = "settings.json"
"""What the store remembers about itself, beside the songs: one small JSON object."""

LATENT_WIDTH = 64

CODEC_SIZE = 32768

MEMORY_BYTES = 128 << 20
"""How much of the session's memory remembered songs may hold: 128 MiB.

A song costs 256 bytes of latents a frame, 6.4 KB a second: 1.2 MB for three
minutes, 2.3 MB for six, so this holds a hundred songs or more, the edits of a
session included."""

DISK_BYTES = 4 << 30
"""How much disk remembered songs may take: 4 GiB.

The acoustic stage runs in bf16, so half of every float32 latent is zeros, and
the file deflates to about half. Measured on three songs on 2026-09-19: 52
percent at level 1, in 8 to 17 ms. A three-minute song takes about 0.6 MB, so
this holds some thousands of them.

It was a gigabyte while this was a cache nobody looked into. 'YuE2 Edit Track'
shows the songs in a list and opens one by name, which makes it a shelf the
user keeps things on, and dropping the oldest to save four gigabytes of a disk
that holds the models is the wrong trade."""

COMPRESS_LEVEL = 1

SOUND_BYTES = 4 << 30
"""How much disk the sounds kept beside the songs may take: 4 GiB.

The sound is the one part of a song that can always be made again -- the
latents are a decode away from it -- so it has a budget of its own rather
than eating into the one that holds the songs themselves, and the sound used
longest ago goes first. Measured on the user's own songs on 2026-09-22: a
FLAC of a 3:42 stereo song at 48 kHz is 22 MB, so this holds about 180 of
them."""

STRIP = 512
"""How many slices of a song the picture kept with it holds.

A picker draws a song as a thin bar a few hundred pixels across. Two bytes a
slice, a peak and a body, is a kilobyte beside the 0.76 MB a three-minute song
already costs, and it is the only part of the sound the store keeps: the
latents are the performance, not a picture of it."""

SHOWN_LETTERS = 600
"""How much of the lyrics a listing carries: enough to know the song, not the whole text.

A listing is read to choose between songs, and a thousand songs of full lyrics
would be megabytes of JSON for a table that shows three lines of each."""

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

    The rest is what a picker shows. 'parent' is the song this one was edited
    from and 'root' the first of that line, both empty for a song straight
    from a singing node; 'root' is written down rather than walked to, so a
    line still holds together when the song in the middle of it is dropped.
    'created' is when it was made, which the file's own time is not: reading a
    song touches it. 'edit' is what was done to the parent to get here, one
    entry an edit, see ``_marks``. 'peaks' and 'body' are the picture, see
    ``strip``.
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
    parent: str = ""
    root: str = ""
    created: float = 0.0
    edit: list = dataclasses.field(default_factory=list)
    peaks: bytes = b""
    body: bytes = b""

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
        if len(self.peaks) != len(self.body):
            raise ValueError("a song's picture is a peak and a body for every slice")
        for name in (self.parent, self.root):
            if name and not _KEY.fullmatch(name):
                raise ValueError("{!r} is not a song key".format(name))


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


def _marks(edits) -> list:
    """What an edit did, as plain JSON, one entry an edit in the order they were made.

    ``op`` is 'retake' or 'cut'; ``at`` is where it sits in this song, in
    seconds, start and stop, the same second twice for a cut; ``bars`` the
    bars asked for, when they were bars; ``seed`` the take that was kept;
    ``took`` how many seconds the edit took in hand. Anything else in the file
    is left out, so a picker never reads more than it was promised.
    """
    kept = []
    for mark in edits or []:
        if not isinstance(mark, dict):
            continue
        at = mark.get("at")
        at = at if isinstance(at, (list, tuple)) and len(at) == 2 else (0.0, 0.0)
        bars = mark.get("bars")
        bars = bars if isinstance(bars, (list, tuple)) and len(bars) == 2 else None
        seed = mark.get("seed")
        kept.append({"op": str(mark.get("op") or ""),
                     "at": [round(float(at[0]), 3), round(float(at[1]), 3)],
                     "bars": None if bars is None else [int(bars[0]), int(bars[1])],
                     "seed": None if seed is None else normalize_seed(seed),
                     "took": round(float(mark.get("took") or 0.0), 3)})
    return kept


def strip(waveform, count: int = STRIP):
    """A song drawn small enough to keep beside it: (peaks, body), a byte a slice each.

    The two lines the track window draws, at ``count`` slices: the loudest
    sample of each slice and the root mean square of it, over every channel,
    scaled from 0 to 255. A song too short for a slice draws as nothing rather
    than as a picture of one sample. Takes [channels, samples] or a batch of
    one, and never writes to what it is given.
    """
    import torch

    samples = waveform.detach()
    if samples.dim() == 3:
        samples = samples[0]
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)
    samples = samples.to(device="cpu", dtype=torch.float32)
    samples = samples.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    total = int(samples.shape[-1])
    if total < 1 or samples.shape[0] < 1:
        return b"", b""
    step = max(1, -(-total // max(1, int(count))))
    usable = (total // step) * step
    if usable < 1:
        return b"", b""
    loud = samples.abs().amax(dim=0)[:usable].reshape(-1, step).amax(dim=1)
    body = samples.pow(2).mean(dim=0)[:usable].reshape(-1, step).mean(dim=1).sqrt()
    drawn = []
    for line in (loud, body):
        drawn.append(bytes((line.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).tolist()))
    return drawn[0], drawn[1]


def pcm16(waveform) -> bytes:
    """A waveform as interleaved little-endian 16-bit samples, the way ``key`` rounds them.

    The same rounding as the key, so a song written out this way and read back
    is the same song under the same key. Takes [channels, samples] or a batch
    of one, and never writes to what it is given.
    """
    import torch

    samples = waveform.detach()
    if samples.dim() == 3:
        samples = samples[0]
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)
    samples = samples.to(device="cpu", dtype=torch.float32).transpose(0, 1).contiguous()
    pcm = (samples.reshape(-1) * PCM_SCALE).round_().clamp_(-32768, 32767)
    return pcm.to(torch.int16).numpy().astype("<i2", copy=False).tobytes()


def flac_of(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    """Interleaved 16-bit samples as a FLAC file.

    FLAC because the numbers said so. Measured on the user's own three songs
    on 2026-09-22, each 3:42 of stereo at 48 kHz, against the same sound as a
    WAV: gzip at level 1 left 93 percent of it and took 0.97 s; differencing
    the samples first and then gzip, 81 percent; FLAC, 53 percent in 0.27 s,
    read back in 0.19 s, every sample of it the sample that went in. It is
    also a file that plays when it is double-clicked, which a gzipped WAV is
    not, and PyAV is what ComfyUI already loads and saves audio with.
    """
    import av
    import numpy

    layout = {1: "mono", 2: "stereo"}.get(int(channels))
    if layout is None:
        raise ValueError("a song of {} channels is not one this keeps".format(channels))
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="flac") as container:
        stream = container.add_stream("flac", rate=int(sample_rate))
        stream.layout = layout
        stream.format = "s16"
        frame = av.AudioFrame.from_ndarray(
            numpy.frombuffer(pcm, dtype="<i2").reshape(1, -1), format="s16", layout=layout)
        frame.sample_rate = int(sample_rate)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def pcm_of(blob: bytes):
    """A FLAC file as ``(pcm, sample_rate, channels)``, the samples interleaved and 16-bit."""
    import av
    import numpy

    pieces, rate, channels = [], 0, 0
    with av.open(io.BytesIO(blob)) as container:
        for frame in container.decode(audio=0):
            rate = rate or int(frame.sample_rate)
            channels = channels or int(frame.layout.nb_channels)
            pieces.append(frame.to_ndarray())
    if not pieces or rate < 1 or channels < 1:
        raise ValueError("this file holds no sound")
    return (numpy.concatenate(pieces, axis=1).reshape(-1).astype("<i2", copy=False).tobytes(),
            rate, channels)


def _short(text) -> str:
    """A note as it is kept: one line, no longer than ``NOTE_LETTERS``."""
    return " ".join(str(text or "").split())[:NOTE_LETTERS]


def _named(name) -> str:
    """*name*, when it is a key; a ValueError otherwise, since it becomes a file name."""
    if not isinstance(name, str) or not _KEY.fullmatch(name):
        raise ValueError("{!r} is not a song key".format(name))
    return name


def is_key(name) -> bool:
    """Whether *name* can be used as a song key, and so as a file name beside the songs."""
    try:
        _named(name)
    except ValueError:
        return False
    return True


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
            "channels": song.channels, "samples": song.samples,
            "parent": song.parent, "root": song.root, "created": song.created,
            "edit": song.edit}
    entries = [("song.json", json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))]
    if song.peaks:
        entries += [("peaks.u8", song.peaks), ("body.u8", song.body)]
    entries += [("prefix.i32", _little(song.prefix)),
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
    held = set(archive.namelist())
    song = Song(
        origin=str(meta["origin"]), style=str(meta["style"]), lyrics=str(meta["lyrics"]),
        seed=int(meta["seed"]), settings=dict(meta["settings"]), score=str(meta["score"]),
        prefix=_numbers("i", archive.read("prefix.i32")),
        negative=_numbers("i", archive.read("negative.i32")) if meta["negative"] else None,
        codec=_numbers("h", archive.read("codec.i16")),
        noise=[[int(value) for value in run] for run in meta["noise"]],
        latents=archive.read("latents.f32"), sample_rate=int(meta["sample_rate"]),
        channels=int(meta["channels"]), samples=int(meta["samples"]),
        parent=str(meta.get("parent") or ""), root=str(meta.get("root") or ""),
        created=float(meta.get("created") or 0.0), edit=_marks(meta.get("edit")),
        peaks=archive.read("peaks.u8") if "peaks.u8" in held else b"",
        body=archive.read("body.u8") if "body.u8" in held else b"")
    song.check()
    return song


class Store:
    """Songs by key: the ones used last in memory, every one on disk until the disk budget is spent.

    With 'folder' None songs are kept in memory only. The lock is there because
    ComfyUI answers HTTP requests on another thread than the one that runs the
    nodes, and a route that reads songs is one of the things an editor wants.
    """

    def __init__(self, folder, memory_bytes: int = MEMORY_BYTES, disk_bytes: int = DISK_BYTES,
                 sound_bytes: int = SOUND_BYTES):
        self.folder = folder
        self.memory_bytes = int(memory_bytes)
        self.disk_bytes = int(disk_bytes)
        self.sound_bytes = int(sound_bytes)
        self._recent = collections.OrderedDict()
        self._aliases = collections.OrderedDict()
        self._notes = {}
        self._own = {}
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
        """The song kept under *name*, or None when there is none.

        A key with no song of its own may be another key of one, see ``alias``.
        """
        _named(name)
        with self._lock:
            song = self._got(name)
            if song is not None:
                return song
            other = self._target(name)
            return None if other is None else self._got(other)

    def _got(self, name: str):
        """The song under *name*, from memory or from disk, held as the newest; None when none."""
        song = self._recent.get(name)
        if song is not None:
            self._recent.move_to_end(name)
            return song
        song = None if self.folder is None else self._read(name)
        if song is not None:
            self._hold(name, song)
        return song

    def alias(self, name: str, target: str) -> None:
        """Let *name* answer with the song kept under *target*, in a file of 64 bytes.

        An edited song has two sounds. One is what the edit made, the old sound
        with a decoded window laid into it, which is what the node handed on
        and what the song was remembered under. The other is what its latents
        decode to when it is opened by key later, and the two differ in the low
        bits, because the parts outside the window come from a decode of
        another length. Remembering the second sound would write every latent
        of the song a second time and put a twin of it in the listing. An alias
        is the same song under both names: one file, one row.

        A key that already names a song of its own is left as it is. Never
        raises: without the alias the song is still there under its own key.
        """
        _named(name)
        _named(target)
        if name == target:
            return
        with self._lock:
            if self._here(name):
                return
            self._aliases.pop(name, None)
            self._aliases[name] = target
            while len(self._aliases) > ALIASES_HELD:
                self._aliases.popitem(last=False)
            if self.folder is None:
                return
            path = os.path.join(self.folder, name + ALIAS_SUFFIX)
            partial = "{}.{}.partial".format(path, os.getpid())
            try:
                os.makedirs(self.folder, exist_ok=True)
                with open(partial, "w", encoding="ascii") as handle:
                    handle.write(target)
                os.replace(partial, path)
            except OSError as error:
                log.warning("[yue2_comfy.songs] could not write %s (%s). The song answers to "
                            "its own key only.", path, error)
            finally:
                if os.path.exists(partial):
                    _remove(partial)

    def canonical(self, name: str) -> str:
        """The key the song under *name* is kept as: *name*, unless it is another name for one.

        A node that was handed the sound a song decodes to, or a field holding
        that sound's key, is working on the song itself, and everything kept
        beside a song -- its measured grid, the takes sung of it -- is kept
        under the one key.
        """
        _named(name)
        with self._lock:
            if self._here(name):
                return name
            other = self._target(name)
            return other if other is not None and self._here(other) else name

    def _here(self, name: str) -> bool:
        """Whether a song of that name is held or written down, without reading it."""
        return name in self._recent or (self.folder is not None
                                        and os.path.isfile(self._path(name)))

    def _target(self, name: str):
        """The key *name* is another name for, or None when it is not one."""
        found = self._aliases.get(name)
        if found is None and self.folder is not None:
            try:
                with open(os.path.join(self.folder, name + ALIAS_SUFFIX), "r",
                          encoding="ascii") as handle:
                    found = handle.read(len(name) + 8).strip()
            except (OSError, ValueError):
                return None
        if not isinstance(found, str) or found == name or not is_key(found):
            return None
        return found

    def _path(self, name: str) -> str:
        return os.path.join(self.folder, name + SUFFIX)

    def _beside(self, name: str, suffix: str) -> str:
        return os.path.join(self.folder, name + suffix)

    def note(self, name: str) -> str:
        """The note written on the song under *name*, or an empty string when there is none."""
        _named(name)
        if self.folder is None:
            return self._notes.get(name, "")
        try:
            with open(self._beside(name, NOTE_SUFFIX), "r", encoding="utf-8") as handle:
                return _short(handle.read(NOTE_LETTERS * 8))
        except OSError:
            return ""

    def remark(self, name: str, text) -> str:
        """Write a note on the song under *name*, and return what is now written.

        An empty note takes the file away rather than leaving an empty one
        behind. Never raises: a note that could not be written is a note, and
        the song is untouched either way.
        """
        _named(name)
        kept = _short(text)
        with self._lock:
            if self.folder is None:
                if kept:
                    self._notes[name] = kept
                else:
                    self._notes.pop(name, None)
                return kept
            path = self._beside(name, NOTE_SUFFIX)
            if not kept:
                _remove(path)
                return ""
            partial = "{}.{}.partial".format(path, os.getpid())
            try:
                os.makedirs(self.folder, exist_ok=True)
                with open(partial, "w", encoding="utf-8") as handle:
                    handle.write(kept)
                os.replace(partial, path)
            except OSError as error:
                log.warning("[yue2_comfy.songs] could not write %s (%s). The note is not kept.",
                            path, error)
                return self.note(name)
            finally:
                if os.path.exists(partial):
                    _remove(partial)
        return kept

    def line(self, name: str) -> list:
        """Every key of the line the song under *name* belongs to, in order.

        A song and the edits made of it are one line, named by the song they
        all start from -- which is the song itself, for one straight from a
        singing node. A song whose own start has been dropped answers with
        the line it was part of, so deleting the row a picker shows deletes
        the rows folded under it.
        """
        _named(name)
        roots = {row["key"]: (row.get("root") or row["key"]) for row in self.listing()}
        line = roots.get(name, name)
        kept = sorted(one for one, root in roots.items() if one == line or root == line)
        return kept or [name]

    def drop(self, name: str, family: bool = False) -> dict:
        """Delete the song under *name* and everything kept beside it.

        With *family* the line it belongs to goes with it: the song it was
        edited from and every edit made of that one, which is what a picker
        offers on the row a song's edits are folded under. The answer says
        which keys went and how many bytes that gave back. Never raises: a
        file that will not go is logged, and the rest still goes.
        """
        _named(name)
        with self._lock:
            wanted = self.line(name) if family else [name]
            gone, freed = [], 0
            for one in wanted:
                size = self._erase(one)
                if size is not None:
                    gone.append(one)
                    freed += size
        return {"keys": gone, "bytes": freed}

    def _erase(self, name: str):
        """Delete one song, its files and every other name for it; None when there was none.

        Everything under the song's own key goes, whatever the suffix, so a
        file written beside a song by another part of the pack -- the grid
        'YuE2 Edit Track' measures, the note, the sound -- never outlives it.
        """
        held = self._recent.pop(name, None)
        if held is not None:
            self._held -= held.size()
        self._aliases.pop(name, None)
        for other in [one for one, target in self._aliases.items() if target == name]:
            self._aliases.pop(other, None)
        self._notes.pop(name, None)
        if self.folder is None:
            return 0 if held is not None else None
        freed, found = 0, held is not None
        start = name + "."
        try:
            with os.scandir(self.folder) as entries:
                rows = [(entry.name, entry.path, entry.is_file(),
                         entry.stat().st_size if entry.is_file() else 0) for entry in entries]
        except OSError as error:
            log.warning("[yue2_comfy.songs] %s cannot be read: %s", self.folder, error)
            return None
        for entry, path, is_file, size in rows:
            if not is_file:
                continue
            if entry.startswith(start):
                found = True
                if _remove(path):
                    freed += size
            elif entry.endswith(ALIAS_SUFFIX) and self._target(entry[:-len(ALIAS_SUFFIX)]) == name:
                _remove(path)
        return freed if found else None

    def wants_sound(self) -> bool:
        """Whether a song's own sound is kept beside it, so opening one is a read, not a decode."""
        if self.folder is None:
            return bool(self._own.get("sounds"))
        return bool(self._kept_settings().get("sounds"))

    def keep_sound(self, on) -> bool:
        """Say whether sounds are kept from now on, and answer with what is set."""
        wanted = bool(on)
        with self._lock:
            if self.folder is None:
                self._own["sounds"] = wanted
                return wanted
            values = dict(self._kept_settings())
            values["sounds"] = wanted
            path = os.path.join(self.folder, SETTINGS_NAME)
            partial = "{}.{}.partial".format(path, os.getpid())
            try:
                os.makedirs(self.folder, exist_ok=True)
                with open(partial, "w", encoding="utf-8") as handle:
                    json.dump(values, handle)
                os.replace(partial, path)
            except OSError as error:
                log.warning("[yue2_comfy.songs] could not write %s (%s). Sounds are kept as "
                            "they were.", path, error)
                return self.wants_sound()
            finally:
                if os.path.exists(partial):
                    _remove(partial)
        return wanted

    def _kept_settings(self) -> dict:
        """What the store was told about itself, or nothing when it was told nothing."""
        try:
            with open(os.path.join(self.folder, SETTINGS_NAME), "r", encoding="utf-8") as handle:
                kept = json.load(handle)
        except (OSError, ValueError):
            return {}
        return kept if isinstance(kept, dict) else {}

    def sounds(self) -> dict:
        """The switch, and what the sounds kept take: ``on``, ``count``, ``bytes``, ``budget``."""
        count, held = 0, 0
        if self.folder is not None:
            try:
                with os.scandir(self.folder) as entries:
                    for entry in entries:
                        if entry.name.endswith(SOUND_SUFFIX) and entry.is_file():
                            count += 1
                            held += entry.stat().st_size
            except OSError:
                log.debug("[yue2_comfy.songs] the sounds in %s cannot be counted", self.folder,
                          exc_info=True)
        return {"on": self.wants_sound(), "count": count, "bytes": held,
                "budget": self.sound_bytes}

    def sound_write(self, name: str, blob: bytes) -> bool:
        """Keep *blob* as the sound of the song under *name*, within the sounds' own budget."""
        _named(name)
        if self.folder is None or not blob:
            return False
        path = self._beside(name, SOUND_SUFFIX)
        partial = "{}.{}.partial".format(path, os.getpid())
        try:
            os.makedirs(self.folder, exist_ok=True)
            with open(partial, "wb") as handle:
                handle.write(blob)
            os.replace(partial, path)
        except OSError as error:
            log.warning("[yue2_comfy.songs] could not write %s (%s). The song is opened by "
                        "decoding it, as it was before.", path, error)
            return False
        finally:
            if os.path.exists(partial):
                _remove(partial)
        with self._lock:
            self._trim_sounds(path)
        return True

    def sound_read(self, name: str):
        """The sound kept for the song under *name* as bytes, or None when there is none.

        Reading one touches it, so the sound used longest ago is the one the
        budget drops, and a song listened to often keeps its own.
        """
        _named(name)
        if self.folder is None:
            return None
        path = self._beside(name, SOUND_SUFFIX)
        try:
            with open(path, "rb") as handle:
                blob = handle.read()
        except OSError:
            return None
        try:
            os.utime(path, None)
        except OSError:
            log.debug("[yue2_comfy.songs] could not touch %s", path, exc_info=True)
        return blob or None

    def sweep_sounds(self) -> dict:
        """Delete every sound kept, and say how many went and how much that gave back.

        A sound is a decode away from being back, so this is the one thing in
        the folder that can be thrown away without losing anything at all.
        """
        count, freed = 0, 0
        if self.folder is None:
            return {"count": 0, "bytes": 0}
        with self._lock:
            try:
                with os.scandir(self.folder) as entries:
                    rows = [(entry.path, entry.stat().st_size) for entry in entries
                            if entry.name.endswith(SOUND_SUFFIX) and entry.is_file()]
            except OSError as error:
                log.warning("[yue2_comfy.songs] %s cannot be read: %s", self.folder, error)
                return {"count": 0, "bytes": 0}
            for path, size in rows:
                if _remove(path):
                    count += 1
                    freed += size
        return {"count": count, "bytes": freed}

    def _trim_sounds(self, kept: str) -> None:
        """Delete the sounds used longest ago until they are within their own budget."""
        found = []
        try:
            with os.scandir(self.folder) as entries:
                for entry in entries:
                    if entry.name.endswith(SOUND_SUFFIX) and entry.is_file():
                        stat = entry.stat()
                        found.append((stat.st_mtime, entry.path, stat.st_size))
        except OSError:
            log.debug("[yue2_comfy.songs] the sounds in %s cannot be counted", self.folder,
                      exc_info=True)
            return
        total = sum(size for _mtime, _path, size in found)
        for _mtime, path, size in sorted(found):
            if total <= self.sound_bytes:
                break
            if os.path.normcase(path) == os.path.normcase(kept):
                continue
            if _remove(path):
                total -= size

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
        are what a crash left behind and go too, and so does an
        alias whose song has gone. A song that goes takes what is kept beside
        it with it -- its note, its sound, its measured grid -- since none of
        them is anything without the song.
        """
        now = time.time()
        found, named, beside = [], [], {}
        with os.scandir(self.folder) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                stat = entry.stat()
                if entry.name.endswith(".partial"):
                    if now - stat.st_mtime > STALE_SECONDS:
                        _remove(entry.path)
                    continue
                if entry.name.endswith(ALIAS_SUFFIX):
                    named.append((entry.name[:-len(ALIAS_SUFFIX)], entry.path))
                    continue
                if entry.name.endswith(SUFFIX):
                    found.append((stat.st_mtime, entry.path, stat.st_size,
                                  entry.name[:-len(SUFFIX)]))
                    continue
                under = entry.name.partition(".")[0]
                if is_key(under):
                    beside.setdefault(under, []).append(entry.path)
        total = sum(size for _mtime, _path, size, _name in found)
        for _mtime, path, size, under in sorted(found):
            if total <= self.disk_bytes:
                break
            if os.path.normcase(path) == os.path.normcase(kept):
                continue
            if _remove(path):
                total -= size
                for other in beside.pop(under, ()):
                    _remove(other)
        for other, path in named:
            song = self._target(other)
            if song is None or not os.path.isfile(self._path(song)):
                _remove(path)

    def listing(self, limit: int = 0) -> list:
        """What the store holds, the song used last first, without the sound of any of it.

        A picker shows a song by what it is -- its style, the words it sings,
        how long it is -- because the key says none of that. Only song.json is
        read out of each file: the latents are the weight of a song and none
        of them is needed to choose. 'limit' caps how many rows come back, 0
        for all of them. Files this version does not read, and names that are
        not keys, are left out rather than shown as empty rows.
        """
        if self.folder is None:
            return []
        found, notes, sounds = [], {}, {}
        try:
            with os.scandir(self.folder) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    if entry.name.endswith(SUFFIX):
                        stat = entry.stat()
                        found.append((stat.st_mtime, entry.name[:-len(SUFFIX)], entry.path,
                                      stat.st_size))
                    elif entry.name.endswith(NOTE_SUFFIX):
                        notes[entry.name[:-len(NOTE_SUFFIX)]] = entry.path
                    elif entry.name.endswith(SOUND_SUFFIX):
                        sounds[entry.name[:-len(SOUND_SUFFIX)]] = entry.stat().st_size
        except OSError as error:
            log.warning("[yue2_comfy.songs] %s cannot be listed: %s", self.folder, error)
            return []
        rows = []
        for used, name, path, size in sorted(found, reverse=True):
            row = _row(name, path, used, size, _written(notes.get(name)), sounds.get(name, 0))
            if row is not None:
                rows.append(row)
            if limit and len(rows) >= limit:
                break
        return rows

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


def _written(path):
    """The note in a file beside a song, or an empty string; never raises."""
    if path is None:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return _short(handle.read(NOTE_LETTERS * 8))
    except OSError:
        return ""


def _row(name: str, path: str, used: float, size: int, note: str = "", sound: int = 0):
    """One song as a listing shows it, or None when the file is not one to offer.

    Everything here but the picture, the note and the sound comes out of
    song.json, the first and smallest member of the archive; the picture is a
    kilobyte of the two after it, handed over as base64. The note and the
    size of the sound kept come from beside the file, and the latents, which
    are the weight of a song, are not read at all.
    """
    try:
        _named(name)
    except ValueError:
        return None
    drawn = {}
    try:
        with zipfile.ZipFile(path) as archive:
            meta = json.loads(archive.read("song.json").decode("utf-8"))
            held = set(archive.namelist())
            for entry in ("peaks.u8", "body.u8"):
                if entry in held:
                    drawn[entry] = base64.b64encode(archive.read(entry)).decode("ascii")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        log.debug("[yue2_comfy.songs] %s cannot be listed: %s", path, error)
        return None
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        return None
    settings = meta.get("settings")
    settings = settings if isinstance(settings, dict) else {}
    lyrics = str(meta.get("lyrics") or "")
    rate = int(meta.get("sample_rate") or 0)
    samples = int(meta.get("samples") or 0)
    return {"key": name,
            "origin": str(meta.get("origin") or ""),
            "style": str(meta.get("style") or ""),
            "lyrics": lyrics[:SHOWN_LETTERS],
            "whole": len(lyrics) <= SHOWN_LETTERS,
            "seed": int(meta.get("seed") or 0),
            "bars": bool(meta.get("score")),
            "voice": bool(settings.get("vocals_only")),
            "parent": str(meta.get("parent") or ""),
            "root": str(meta.get("root") or ""),
            "created": round(float(meta.get("created") or 0.0), 3),
            "edit": _marks(meta.get("edit")),
            "peaks": drawn.get("peaks.u8", ""),
            "body": drawn.get("body.u8", ""),
            "seconds": round(samples / float(rate), 2) if rate else 0.0,
            "sample_rate": rate,
            "channels": int(meta.get("channels") or 0),
            "note": _short(note),
            "sound": int(sound),
            "used": round(float(used), 3),
            "bytes": int(size)}


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


def listing(limit: int = 0) -> list:
    """The songs this install remembers, the one used last first; see ``Store.listing``."""
    return store().listing(limit)


def remember(waveform, sample_rate, song: Song) -> str:
    """Keep *song* under the key of *waveform*, and return the key.

    The sound goes beside it when the store is asked to keep sounds, and it
    is this sound, the one the key was made of, so opening the song again
    gives back the file the node handed on rather than a decode of it.
    """
    began = time.perf_counter()
    name = key(waveform, sample_rate)
    store().put(name, song)
    log.info("[yue2_comfy.songs] remembered for editing as %s: %d frames, %.1f MB, in %.2f s",
             name[:12], song.frames, song.size() / 1e6, time.perf_counter() - began)
    if store().wants_sound():
        sound_keep(name, waveform, song.sample_rate)
    return name


def note(name: str) -> str:
    """The note written on the song under *name*; see ``Store.note``."""
    return store().note(name)


def remark(name: str, text) -> str:
    """Write a note on the song under *name*; see ``Store.remark``."""
    return store().remark(name, text)


def drop(name: str, family: bool = False) -> dict:
    """Delete the song under *name*, and its line with it when asked; see ``Store.drop``."""
    return store().drop(name, family)


def sounds() -> dict:
    """What the picker's switch shows: whether sounds are kept, what they take, whether they can be.

    ``can`` is false where PyAV is not installed, which is also where ComfyUI
    cannot read or write audio at all. The window says so rather than
    offering a switch that would quietly do nothing.
    """
    import importlib.util

    said = store().sounds()
    try:
        can = importlib.util.find_spec("av") is not None
    except (ImportError, ValueError):
        can = False
    said["can"] = can
    said["why"] = "" if can else ("A song's sound is kept as a FLAC, and PyAV, which writes one, "
                                  "is not installed here.")
    return said


def keep_sounds(on) -> dict:
    """Turn the kept sounds on or off, and answer as ``sounds`` does."""
    store().keep_sound(on)
    return sounds()


def sweep_sounds() -> dict:
    """Delete every sound kept, and answer as ``sounds`` does, with what went."""
    gone = store().sweep_sounds()
    said = sounds()
    said["gone"] = gone
    return said


def sound_keep(name: str, waveform, sample_rate: int) -> bool:
    """Keep the sound of *waveform* beside the song under *name*. Never raises.

    What it buys is the decode it saves: 1.1 to 1.7 seconds of card for a
    four-minute song, against a read and 0.19 s of unpacking. What it costs is
    22 MB of disk and 0.27 s of writing, measured on the user's own songs on
    2026-09-22.
    """
    began = time.perf_counter()
    try:
        channels = int(waveform.shape[-2]) if waveform.dim() > 1 else 1
        blob = flac_of(pcm16(waveform), int(sample_rate), channels)
        if not store().sound_write(name, blob):
            return False
    except Exception:
        log.warning("[yue2_comfy.songs] the sound of this song could not be kept beside it, so "
                    "opening it decodes it as before", exc_info=True)
        return False
    log.info("[yue2_comfy.songs] the sound of %s is kept beside it: %.1f MB in %.2f s",
             name[:12], len(blob) / 1e6, time.perf_counter() - began)
    return True


def sound_of(name: str):
    """The sound kept beside the song under *name*, or None when there is none.

    ``{"waveform": [1, channels, samples], "sample_rate": rate}``, as a node
    hands audio on. The samples are the ones the key was made of, so this is
    the song under the key it was remembered by: no decode, and no second
    name for a sound that differs in its last bits. Never raises: a file that
    cannot be read is a song that is decoded, as it was before.
    """
    import torch

    blob = store().sound_read(store().canonical(name))
    if not blob:
        return None
    began = time.perf_counter()
    try:
        pcm, rate, channels = pcm_of(blob)
        samples = torch.frombuffer(bytearray(pcm), dtype=torch.int16).to(torch.float32)
        waveform = (samples / PCM_SCALE).reshape(-1, channels).transpose(0, 1)
        waveform = waveform.unsqueeze(0).contiguous()
    except Exception:
        log.warning("[yue2_comfy.songs] the sound kept beside %s cannot be read, so the song is "
                    "decoded instead", name[:12], exc_info=True)
        return None
    log.info("[yue2_comfy.songs] the sound of %s came off the disk: %.1f s of it in %.2f s",
             name[:12], int(waveform.shape[-1]) / float(rate), time.perf_counter() - began)
    return {"waveform": waveform, "sample_rate": int(rate)}


def canonical(name: str) -> str:
    """The key the song under *name* is kept as; see ``Store.canonical``."""
    return store().canonical(name)


def alias(waveform, sample_rate, target: str) -> str:
    """Let the sound of *waveform* answer with the song already kept under *target*.

    Returns the key that sound has, which is *target* itself when the two are
    the same sound. See ``Store.alias``.
    """
    name = key(waveform, sample_rate)
    if name != target:
        store().alias(name, target)
        log.info("[yue2_comfy.songs] %s is the sound %s decodes to: one song, two keys",
                 name[:12], target[:12])
    return name


def recall(waveform, sample_rate):
    """The song *waveform* is the sound of, or None when this pack does not know it."""
    return store().get(key(waveform, sample_rate))


def made(origin, style, lyrics, seed, settings, score, prefix, negative, codec, noise, latents,
         sample_rate, channels, samples, parent="", root="", created=None, edit=None,
         peaks=b"", body=b"") -> Song:
    """A song from plain sequences, its latents and the size of its audio, checked.

    What a singing node remembers and what an edit makes both come through
    here, so both are written down the same way. ``latents`` may be a tensor, a
    numpy array or the bytes themselves. ``created`` defaults to now, which is
    what making a song means.
    """
    song = Song(
        origin=origin, style=style or "", lyrics=lyrics or "", seed=normalize_seed(seed),
        settings=_plain(settings), score=score or "", prefix=_numbers("i", prefix),
        negative=None if negative is None else _numbers("i", negative),
        codec=_numbers("h", codec), noise=[[int(value) for value in run] for run in noise],
        latents=_floats(latents), sample_rate=int(sample_rate), channels=int(channels),
        samples=int(samples), parent=str(parent or ""), root=str(root or ""),
        created=time.time() if created is None else float(created), edit=_marks(edit),
        peaks=bytes(peaks), body=bytes(body))
    song.check()
    return song


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
        peaks, body = strip(waveform)
        song = made(origin, style, lyrics, seed, settings, score, performance.prefix,
                    performance.negative, performance.codec,
                    [[int(performance.seed), 0, len(performance.codec)]], performance.latents,
                    audio["sample_rate"], waveform.shape[-2], waveform.shape[-1],
                    peaks=peaks, body=body)
        return remember(waveform, song.sample_rate, song)
    except Exception:
        log.warning("[yue2_comfy.songs] this song could not be remembered, so it cannot be "
                    "edited", exc_info=True)
        return None
