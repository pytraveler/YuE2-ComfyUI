"""The song memory: what a node sang, found again from its sound alone.

Two things have to hold for an edit to work. The store has to give back every
part of a song exactly, from memory or from a file written before a restart,
and drop old songs within its budgets. And the key has to be the same for the
audio a node hands on and for that audio saved as FLAC and loaded back, while a
lossy file gets a key of its own. The store is plain Python and is tested
everywhere; the key needs torch, and the FLAC trip needs PyAV as well.
"""

from __future__ import annotations

import array
import io
import logging
import os
import types
import zipfile

import pytest

from yue2_comfy import songs

FRAMES = 6


def a_song(frames=FRAMES, style="a style", negative=True) -> songs.Song:
    """A small song whose every part is told apart from every other song's."""
    latents = array.array("f", (float(index % 97) / 7.0 for index in range(frames * 64)))
    return songs.Song(
        origin="YuE2 Generate Song", style=style, lyrics="[verse]\nwords", seed=831001,
        settings={"cot": "full", "cfg_scale": 1.5, "vae": "standard", "low_vram": False},
        score="X:1\nK:C\nCDEF|", prefix=array.array("i", [151643, 100, 200, 151851]),
        negative=array.array("i", [151643, 7, 151851]) if negative else None,
        codec=array.array("h", [index * 1000 % 32768 for index in range(frames)]),
        noise=[[831001, 0, frames]], latents=songs._little(latents),
        sample_rate=48000, channels=2, samples=frames * 1920)


def name(number: int) -> str:
    return "{:064x}".format(number)


def test_a_song_comes_back_from_its_file_as_it_went_in(tmp_path):
    """A second store on the same folder is what ComfyUI is after a restart."""
    song = a_song()
    songs.Store(str(tmp_path)).put(name(1), song)
    again = songs.Store(str(tmp_path)).get(name(1))
    assert again == song
    assert again is not song
    assert again.frames == FRAMES


def test_lyrics_in_another_alphabet_are_written_as_words(tmp_path):
    """Opened with an archiver, song.json shows the lyrics, not escape codes."""
    words = "[verse]\n\u0441\u043b\u043e\u0432\u0430 \u043f\u0435\u0441\u043d\u0438"
    song = a_song()
    song.lyrics = words
    songs.Store(str(tmp_path)).put(name(1), song)
    with zipfile.ZipFile(str(tmp_path / (name(1) + songs.SUFFIX))) as archive:
        text = archive.read("song.json").decode("utf-8")
    assert "\u0441\u043b\u043e\u0432\u0430" in text
    assert songs.Store(str(tmp_path)).get(name(1)).lyrics == words


def test_a_song_without_guidance_has_no_negative_prompt_on_disk_either(tmp_path):
    songs.Store(str(tmp_path)).put(name(1), a_song(negative=False))
    assert songs.Store(str(tmp_path)).get(name(1)).negative is None


def test_a_song_nobody_kept_is_none(tmp_path):
    assert songs.Store(str(tmp_path)).get(name(9)) is None
    assert songs.Store(None).get(name(9)) is None


def test_memory_lets_the_song_used_longest_ago_go_first():
    size = a_song().size()
    store = songs.Store(None, memory_bytes=2 * size)
    store.put(name(1), a_song())
    store.put(name(2), a_song())
    assert store.get(name(1)) is not None, "reading it makes it the newest"
    store.put(name(3), a_song())
    assert store.get(name(2)) is None
    assert store.get(name(1)) is not None
    assert store.get(name(3)) is not None


def test_the_newest_song_stays_in_memory_even_alone_over_budget():
    store = songs.Store(None, memory_bytes=1)
    store.put(name(1), a_song())
    store.put(name(2), a_song())
    assert store.get(name(1)) is None
    assert store.get(name(2)) is not None


def test_the_disk_lets_the_song_used_longest_ago_go_first(tmp_path):
    """Reading a file touches it, so the order on disk is the order of use."""
    store = songs.Store(str(tmp_path), memory_bytes=1)
    store.put(name(1), a_song())
    store.put(name(2), a_song())
    one, two = (os.path.join(str(tmp_path), name(n) + songs.SUFFIX) for n in (1, 2))
    os.utime(one, (1000.0, 1000.0))
    os.utime(two, (2000.0, 2000.0))
    assert store.get(name(1)) is not None
    store.disk_bytes = os.path.getsize(one) + os.path.getsize(two)
    store.put(name(3), a_song())
    assert os.path.exists(one)
    assert not os.path.exists(two)
    assert store.get(name(3)) is not None


def test_the_file_just_written_stays_even_alone_over_budget(tmp_path):
    store = songs.Store(str(tmp_path), disk_bytes=1)
    store.put(name(1), a_song())
    store.put(name(2), a_song())
    left = sorted(os.listdir(str(tmp_path)))
    assert left == [name(2) + songs.SUFFIX]


def test_a_file_that_cannot_be_read_is_ignored(tmp_path, caplog):
    path = tmp_path / (name(1) + songs.SUFFIX)
    path.write_bytes(b"not a song")
    with caplog.at_level(logging.WARNING):
        assert songs.Store(str(tmp_path)).get(name(1)) is None
    assert "cannot be read" in caplog.text


def test_a_file_whose_parts_do_not_fit_is_ignored(tmp_path):
    store = songs.Store(str(tmp_path))
    store.put(name(1), a_song())
    path = tmp_path / (name(1) + songs.SUFFIX)
    with zipfile.ZipFile(str(path)) as archive:
        entries = {entry: archive.read(entry) for entry in archive.namelist()}
    entries["latents.f32"] = entries["latents.f32"][:-4]
    with zipfile.ZipFile(str(path), "w") as archive:
        for entry, blob in entries.items():
            archive.writestr(entry, blob)
    assert songs.Store(str(tmp_path)).get(name(1)) is None


def test_a_file_written_in_another_format_is_left_alone(tmp_path):
    path = tmp_path / (name(1) + songs.SUFFIX)
    with zipfile.ZipFile(str(path), "w") as archive:
        archive.writestr("song.json", '{"format": 99}')
    assert songs.Store(str(tmp_path)).get(name(1)) is None
    assert path.exists()


def test_what_a_crash_left_half_written_is_cleared_and_a_fresh_write_is_not(tmp_path):
    old = tmp_path / (name(7) + songs.SUFFIX + ".123.partial")
    fresh = tmp_path / (name(8) + songs.SUFFIX + ".456.partial")
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    os.utime(str(old), (1000.0, 1000.0))
    songs.Store(str(tmp_path)).put(name(1), a_song())
    assert not old.exists()
    assert fresh.exists()


def test_a_disk_that_refuses_leaves_the_song_in_memory(tmp_path, caplog):
    """A folder that is a file cannot be written into, which stands for a full or read-only disk."""
    blocked = tmp_path / "blocked"
    blocked.write_bytes(b"")
    store = songs.Store(str(blocked))
    with caplog.at_level(logging.WARNING):
        store.put(name(1), a_song())
    assert "until ComfyUI restarts" in caplog.text
    assert store.get(name(1)) == a_song()


def test_only_a_key_can_become_a_file_name(tmp_path):
    store = songs.Store(str(tmp_path))
    for bad in ("../" + name(1)[3:], name(0xabc).upper(), name(1)[:-1], 5):
        with pytest.raises(ValueError):
            store.put(bad, a_song())
        with pytest.raises(ValueError):
            store.get(bad)


@pytest.mark.parametrize("broken, says", [
    (dict(latents=b"\0" * 12), "bytes of latents"),
    (dict(noise=[[1, 0, FRAMES - 1]]), "noise covers"),
    (dict(noise=[[1, 0, 2], [2, 0, 0], [1, 2, FRAMES - 2]]), "noise run"),
    (dict(noise=[[True, 0, FRAMES]]), "noise run"),
    (dict(codec=array.array("h", [-1] * FRAMES)), "codec token"),
    (dict(prefix=array.array("i")), "prompt"),
    (dict(samples=0), "samples"),
])
def test_a_song_whose_parts_do_not_fit_is_refused(broken, says):
    song = a_song()
    for field, value in broken.items():
        setattr(song, field, value)
    with pytest.raises(ValueError) as error:
        songs.Store(None).put(name(1), song)
    assert says in str(error.value)


def test_an_edited_song_is_several_noise_runs():
    song = a_song()
    song.noise = [[831001, 0, 2], [1000014, 0, 3], [831001, 5, 1]]
    song.check()


def test_the_widths_the_file_format_assumes():
    """The file names say int32 and int16, and the arrays have to agree on every platform."""
    assert array.array("i").itemsize == 4
    assert array.array("h").itemsize == 2
    assert array.array("f").itemsize == 4


def test_the_settings_kept_are_the_ones_that_can_be_written_down():
    kept = songs._plain({"cot": "full", "top_k": 100, "cfg": 1.5, "low_vram": False,
                         "none": None, "device": object(), "list": [1]})
    assert kept == {"cot": "full", "top_k": 100, "cfg": 1.5, "low_vram": False, "none": None}


def test_keep_without_a_performance_keeps_nothing(monkeypatch):
    def refuse(*args):
        raise AssertionError("nothing should be remembered")

    monkeypatch.setattr(songs, "remember", refuse)
    assert songs.keep({"waveform": None, "sample_rate": 48000}, "YuE2 Render Plan", "s", "l",
                      1, {}, "", None) is None


def test_keep_never_fails_the_song(caplog):
    """The audio is finished by then; a memory that cannot be written is only logged."""
    performance = types.SimpleNamespace(prefix=[1], negative=None, codec=[1, 2], seed=3,
                                        latents=b"\0" * (2 * 64 * 4))
    with caplog.at_level(logging.WARNING):
        assert songs.keep({"waveform": "not a tensor", "sample_rate": 48000},
                          "YuE2 Generate Song", "s", "l", 1, {}, "", performance) is None
    assert "could not be remembered" in caplog.text


def noise_song(torch, seconds=2.0, seed=0):
    """Stereo noise at 48 kHz within full scale, the way a song comes out of the decoder."""
    generator = torch.Generator().manual_seed(seed)
    samples = int(48000 * seconds)
    return (torch.rand(1, 2, samples, generator=generator) * 2 - 1).clamp_(-1, 1)


def flac_trip(torch, waveform):
    """The samples Load Audio gives back for a 16-bit FLAC of *waveform*, by arithmetic."""
    pcm = torch.round(waveform * 32768.0).clamp(-32768, 32767).to(torch.int16)
    return pcm.float() / (2 ** 15)


def test_the_key_is_the_same_for_a_song_and_its_flac():
    torch = pytest.importorskip("torch")
    waveform = noise_song(torch)
    assert songs.key(waveform, 48000) == songs.key(flac_trip(torch, waveform), 48000)


def test_one_sample_one_step_off_is_another_key():
    torch = pytest.importorskip("torch")
    waveform = flac_trip(torch, noise_song(torch))
    other = waveform.clone()
    other[0, 1, 1000] += 1.0 / 32768
    assert songs.key(waveform, 48000) != songs.key(other, 48000)


def test_the_rate_and_the_shape_are_part_of_the_key():
    torch = pytest.importorskip("torch")
    waveform = noise_song(torch)
    assert songs.key(waveform, 48000) != songs.key(waveform, 44100)
    assert songs.key(waveform, 48000) != songs.key(waveform.reshape(1, 1, -1), 48000)


def test_a_batch_of_one_is_the_song_and_a_batch_of_two_is_refused():
    torch = pytest.importorskip("torch")
    waveform = noise_song(torch)
    assert songs.key(waveform, 48000) == songs.key(waveform[0], 48000)
    with pytest.raises(ValueError):
        songs.key(torch.cat([waveform, waveform]), 48000)


def test_the_key_does_not_depend_on_how_it_is_cut_into_chunks(monkeypatch):
    torch = pytest.importorskip("torch")
    waveform = noise_song(torch, seconds=1.0)
    whole = songs.key(waveform, 48000)
    monkeypatch.setattr(songs, "CHUNK_SAMPLES", 4097)
    assert songs.key(waveform, 48000) == whole


def test_the_key_leaves_the_waveform_alone():
    torch = pytest.importorskip("torch")
    waveform = noise_song(torch)
    before = waveform.clone()
    songs.key(waveform, 48000)
    assert torch.equal(waveform, before)


def test_a_kept_song_is_recalled_from_its_audio_and_from_its_flac():
    torch = pytest.importorskip("torch")
    waveform = noise_song(torch, seconds=FRAMES * 1920 / 48000)
    latents = torch.arange(FRAMES * 64, dtype=torch.float32).reshape(FRAMES, 64) / 64
    performance = types.SimpleNamespace(prefix=(151643, 5, 151851), negative=None,
                                        codec=tuple(range(FRAMES)), seed=42, latents=latents)
    name_ = songs.keep({"waveform": waveform, "sample_rate": 48000}, "YuE2 Generate Song",
                       "a style", "words", 42 + (1 << 63), {"cot": "full"}, "X:1", performance)
    assert name_ == songs.key(waveform, 48000)
    song = songs.recall(flac_trip(torch, waveform), 48000)
    assert song is songs.recall(waveform, 48000)
    assert song.seed == 42, "the seed is kept the way the song was sung with it"
    assert song.noise == [[42, 0, FRAMES]]
    assert list(song.codec) == list(range(FRAMES))
    assert song.latents == latents.numpy().astype("<f4").tobytes()
    assert (song.channels, song.samples, song.sample_rate) == (2, FRAMES * 1920, 48000)


def comfy_save_and_load(av, torch, waveform, fmt):
    """What Save Audio writes and Load Audio reads, in the calls ComfyUI 0.36 makes.

    Copied from comfy_api/latest/_ui.py (AudioSaveHelper.save_audio) and
    comfy_extras/nodes_audio.py (load), so a change in FFmpeg's rounding shows
    here and not in somebody's edit.
    """
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format=fmt)
    layout = "stereo"
    codec = {"flac": "flac", "mp3": "libmp3lame"}[fmt]
    stream = container.add_stream(codec, rate=48000, layout=layout)
    frame = av.AudioFrame.from_ndarray(waveform[0].movedim(0, 1).reshape(1, -1).float().numpy(),
                                       format="flt", layout=layout)
    frame.sample_rate = 48000
    frame.pts = 0
    container.mux(stream.encode(frame))
    container.mux(stream.encode(None))
    container.close()
    buffer.seek(0)
    with av.open(buffer) as source:
        audio = source.streams.audio[0]
        channels = audio.channels
        frames = []
        for decoded in source.decode(streams=audio.index):
            chunk = torch.from_numpy(decoded.to_ndarray())
            if chunk.shape[0] != channels:
                chunk = chunk.view(-1, channels).t()
            frames.append(chunk)
    wav = torch.cat(frames, dim=1)
    if wav.dtype == torch.int16:
        wav = wav.float() / (2 ** 15)
    return wav.unsqueeze(0)


def test_save_audio_as_flac_then_load_audio_keeps_the_key_and_mp3_does_not():
    torch = pytest.importorskip("torch")
    av = pytest.importorskip("av")
    waveform = noise_song(torch, seconds=1.0, seed=3)
    halves = (torch.arange(-40, 40, dtype=torch.float32) + 0.5) / 32768
    waveform[0, 0, :80] = halves
    waveform[0, 1, :80] = halves.flip(0)
    waveform[0, :, 80:84] = torch.tensor([1.0, -1.0, 32767.5 / 32768, -32768.5 / 32768])
    flac = comfy_save_and_load(av, torch, waveform, "flac")
    assert songs.key(flac, 48000) == songs.key(waveform, 48000)
    mp3 = comfy_save_and_load(av, torch, waveform, "mp3")
    assert songs.key(mp3, 48000) != songs.key(waveform, 48000)
