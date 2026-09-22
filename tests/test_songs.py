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
import base64
import dataclasses
import io
import logging
import os
import time
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


def test_a_song_answers_to_the_other_key_its_sound_has(tmp_path):
    """An edit's sound and the sound its latents decode to are two keys of one song."""
    store = songs.Store(str(tmp_path))
    store.put(name(1), a_song())
    store.alias(name(2), name(1))
    assert store.get(name(2)) is not None
    assert store.canonical(name(2)) == name(1)
    assert sorted(os.listdir(str(tmp_path))) == [name(1) + songs.SUFFIX,
                                                 name(2) + songs.ALIAS_SUFFIX]
    assert [row["key"] for row in store.listing()] == [name(1)], "one song, one row"
    assert songs.Store(str(tmp_path)).get(name(2)) is not None, "a restart reads it too"


def test_another_name_never_hides_a_song_of_its_own(tmp_path):
    """Whatever else answers to a key, a song written under it is that key's own song."""
    store = songs.Store(str(tmp_path))
    store.put(name(1), a_song(style="the first"))
    store.put(name(2), a_song(style="the second"))
    store.alias(name(2), name(1))
    assert store.get(name(2)).style == "the second"
    assert store.canonical(name(2)) == name(2)
    fresh = songs.Store(str(tmp_path))
    fresh.alias(name(2), name(1))
    assert fresh.get(name(2)).style == "the second"
    assert not os.path.exists(os.path.join(str(tmp_path), name(2) + songs.ALIAS_SUFFIX))


def test_a_name_that_points_at_nothing_is_no_name_at_all(tmp_path):
    """An alias written by something else, or left by a song that has gone, answers nothing."""
    (tmp_path / (name(2) + songs.ALIAS_SUFFIX)).write_text("the first one", encoding="ascii")
    (tmp_path / (name(3) + songs.ALIAS_SUFFIX)).write_text(name(9), encoding="ascii")
    (tmp_path / (name(4) + songs.ALIAS_SUFFIX)).write_text(name(4), encoding="ascii")
    store = songs.Store(str(tmp_path))
    assert [store.get(name(number)) for number in (2, 3, 4)] == [None, None, None]
    assert [store.canonical(name(number)) for number in (2, 3, 4)] == [
        name(2), name(3), name(4)]


def test_the_name_of_a_song_that_has_gone_goes_with_it(tmp_path):
    store = songs.Store(str(tmp_path))
    store.put(name(1), a_song())
    store.alias(name(2), name(1))
    store.disk_bytes = 1
    store.put(name(3), a_song())
    assert sorted(os.listdir(str(tmp_path))) == [name(3) + songs.SUFFIX]
    assert songs.Store(str(tmp_path)).get(name(2)) is None


def test_a_store_with_no_folder_answers_to_both_keys_until_it_restarts():
    store = songs.Store(None)
    store.put(name(1), a_song())
    store.alias(name(2), name(1))
    assert store.get(name(2)) is not None


def test_only_a_key_can_be_another_name_and_a_key_is_not_another_name_for_itself(tmp_path):
    store = songs.Store(str(tmp_path))
    store.put(name(1), a_song())
    store.alias(name(1), name(1))
    with pytest.raises(ValueError, match="not a song key"):
        store.alias("the first one", name(1))
    with pytest.raises(ValueError, match="not a song key"):
        store.alias(name(2), "the first one")
    assert os.listdir(str(tmp_path)) == [name(1) + songs.SUFFIX]


def test_the_picture_a_song_keeps_is_a_byte_a_slice():
    """A picker draws a song without its sound, so the sound leaves a kilobyte behind."""
    torch = pytest.importorskip("torch")
    wave = torch.zeros((2, 48000))
    wave[:, :24000] = 0.5
    wave[0, :2400] = -1.0
    peaks, body = songs.strip(wave, count=8)
    assert len(peaks) == len(body) == 8
    assert list(peaks) == [255, 128, 128, 128, 0, 0, 0, 0], "the loudest sample of each slice"
    assert list(body)[1:] == [128, 128, 128, 0, 0, 0, 0], "and how loud the slice is"
    assert body[0] < peaks[0], "one loud sample does not make a loud slice"
    assert songs.strip(torch.zeros((2, 0))) == (b"", b"")
    assert songs.strip(torch.zeros((1, 2, 480)), count=512) == songs.strip(
        torch.zeros((2, 480)), count=512), "a batch of one is the song"


def test_the_picture_and_the_line_travel_with_the_song(tmp_path):
    made = dataclasses.replace(a_song(), peaks=bytes([1, 2, 3]), body=bytes([4, 5, 6]),
                               parent=name(7), root=name(5), created=1000.5,
                               edit=[{"op": "retake", "at": [1.0, 2.0], "bars": [3, 4],
                                      "seed": 40, "took": 1.0}])
    songs.Store(str(tmp_path)).put(name(1), made)
    back = songs.Store(str(tmp_path)).get(name(1))
    assert back == made
    row = songs.Store(str(tmp_path)).listing()[0]
    assert base64.b64decode(row["peaks"]) == bytes([1, 2, 3])
    assert base64.b64decode(row["body"]) == bytes([4, 5, 6])
    assert (row["parent"], row["root"], row["created"]) == (name(7), name(5), 1000.5)
    assert row["edit"] == made.edit


def test_a_song_from_a_singing_node_starts_a_line_of_its_own(tmp_path):
    store = songs.Store(str(tmp_path))
    store.put(name(1), a_song())
    row = store.listing()[0]
    assert (row["parent"], row["root"], row["edit"]) == ("", "", [])
    assert row["peaks"] == "" and row["body"] == "", "a song saved before this was drawn has none"


def test_what_an_edit_did_is_written_down_plainly_or_not_at_all(tmp_path):
    """The file is edited by hand as easily as read; a picker must not choke on that."""
    store = songs.Store(str(tmp_path))
    store.put(name(1), dataclasses.replace(a_song(), edit=[
        {"op": "cut", "at": [1.0, 1.0], "bars": None, "seed": None, "took": 2.0},
        {"op": "retake", "at": (3, 4), "bars": (5, 6), "seed": 7.0, "took": 1},
        "not an edit at all",
        {"at": "nowhere"}]))
    assert store.listing()[0]["edit"] == [
        {"op": "cut", "at": [1.0, 1.0], "bars": None, "seed": None, "took": 2.0},
        {"op": "retake", "at": [3.0, 4.0], "bars": [5, 6], "seed": 7, "took": 1.0},
        {"op": "", "at": [0.0, 0.0], "bars": None, "seed": None, "took": 0.0}]


def test_a_picture_with_a_body_and_no_peaks_is_not_a_picture():
    with pytest.raises(ValueError, match="peak and a body"):
        dataclasses.replace(a_song(), body=bytes([1, 2, 3])).check()
    with pytest.raises(ValueError, match="not a song key"):
        dataclasses.replace(a_song(), parent="the one before").check()


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


def test_the_sound_a_song_decodes_to_is_written_down_beside_it(tmp_path, monkeypatch):
    """What a node calls: it has the sound in hand, not the key of it."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    songs.store().put(name(1), a_song())
    wave = torch.full((2, 480), 0.25)
    other = songs.alias(wave, 48000, name(1))
    assert other == songs.key(wave, 48000) != name(1)
    assert songs.store().get(other).style == "a style"
    assert songs.alias(wave, 48000, other) == other, "a sound needs no other name for itself"
    assert sorted(os.listdir(str(tmp_path))) == sorted(
        [name(1) + songs.SUFFIX, other + songs.ALIAS_SUFFIX])


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


def test_a_song_made_from_plain_sequences_is_the_song_written_out_field_by_field():
    """What a singing node keeps and what an edit makes come through the same door."""
    song = a_song()
    made = songs.made(song.origin, song.style, song.lyrics, song.seed, song.settings, song.score,
                      list(song.prefix), list(song.negative), list(song.codec), song.noise,
                      song.latents, song.sample_rate, song.channels, song.samples)
    assert made.created == pytest.approx(time.time(), abs=60), "making one is when it was made"
    assert dataclasses.replace(made, created=song.created) == song
    with pytest.raises(ValueError, match="noise covers"):
        songs.made(song.origin, song.style, song.lyrics, song.seed, song.settings, song.score,
                   list(song.prefix), None, list(song.codec), [[1, 0, FRAMES - 1]], song.latents,
                   song.sample_rate, song.channels, song.samples)


def _kept(folder, songs_by_name):
    """Songs on disk with the times they were last used set apart, newest last in the argument."""
    store = songs.Store(str(folder))
    for index, (key, song) in enumerate(songs_by_name):
        store.put(key, song)
        os.utime(os.path.join(str(folder), key + songs.SUFFIX), (1000 + index, 1000 + index))
    return store


def test_the_listing_says_what_each_song_is_without_reading_its_latents(tmp_path):
    """The picker shows songs by what they are: the key alone tells a user nothing."""
    store = _kept(tmp_path, [(name(1), a_song(style="older")), (name(2), a_song(style="newer"))])
    rows = store.listing()
    assert [row["key"] for row in rows] == [name(2), name(1)], "the one used last comes first"
    assert [row["style"] for row in rows] == ["newer", "older"]
    row = rows[0]
    assert row["origin"] == "YuE2 Generate Song" and row["seed"] == 831001
    assert row["lyrics"] == "[verse]\nwords" and row["whole"] is True
    assert row["bars"] is True, "this song has a score, so it can be cut by bars"
    assert row["voice"] is False
    assert row["seconds"] == round(FRAMES * 1920 / 48000.0, 2)
    assert row["sample_rate"] == 48000 and row["channels"] == 2
    assert row["used"] == 1001.0 and row["bytes"] > 0


def test_the_listing_marks_the_songs_that_cannot_be_edited_by_bars_or_at_all(tmp_path):
    """A song sung with cot 'off' has no bars, and a voice-only song is refused by the node."""
    plain = a_song()
    plain.score = ""
    voice = a_song()
    voice.settings = dict(voice.settings, vocals_only=True)
    rows = {row["key"]: row for row in
            _kept(tmp_path, [(name(1), plain), (name(2), voice)]).listing()}
    assert rows[name(1)]["bars"] is False
    assert rows[name(2)]["voice"] is True


def test_the_listing_carries_enough_of_the_lyrics_to_know_the_song(tmp_path):
    """Whole lyrics of a thousand songs would be megabytes of JSON for a table of three lines."""
    long = a_song()
    long.lyrics = "la la la\n" * 400
    row = _kept(tmp_path, [(name(1), long)]).listing()[0]
    assert len(row["lyrics"]) == songs.SHOWN_LETTERS
    assert row["whole"] is False, "the window says the words go on"


def test_the_listing_leaves_out_what_it_cannot_offer(tmp_path, caplog):
    """A file from another version, a name that is not a key, and rubbish are not rows."""
    store = _kept(tmp_path, [(name(1), a_song())])
    (tmp_path / "notakey.song").write_bytes((tmp_path / (name(1) + songs.SUFFIX)).read_bytes())
    (tmp_path / (name(5) + songs.SUFFIX)).write_bytes(b"not a zip at all")
    (tmp_path / (name(6) + songs.SUFFIX + ".partial")).write_bytes(b"half of one")
    with caplog.at_level(logging.DEBUG, logger="yue2_comfy.songs"):
        rows = store.listing()
    assert [row["key"] for row in rows] == [name(1)]


def test_the_listing_can_be_capped_and_a_store_with_no_folder_has_none(tmp_path):
    store = _kept(tmp_path, [(name(1), a_song()), (name(2), a_song()), (name(3), a_song())])
    assert [row["key"] for row in store.listing(2)] == [name(3), name(2)]
    assert songs.Store(None).listing() == []


def test_a_note_is_kept_beside_the_song_and_read_back_with_the_listing(tmp_path):
    """A word on a song is the one thing the song cannot say: what it was wanted for."""
    store = songs.Store(str(tmp_path))
    store.put(name(1), a_song())
    assert store.note(name(1)) == ""
    assert store.remark(name(1), "  the  loud\n one  ") == "the loud one", "one line, trimmed"
    assert store.note(name(1)) == "the loud one"
    again = songs.Store(str(tmp_path))
    assert again.note(name(1)) == "the loud one", "a restart reads it too"
    assert again.listing()[0]["note"] == "the loud one"
    assert store.remark(name(1), "") == "", "an empty note takes the file away"
    assert os.listdir(str(tmp_path)) == [name(1) + songs.SUFFIX]
    assert store.listing()[0]["note"] == ""


def test_a_note_is_a_badge_and_not_a_second_set_of_lyrics(tmp_path):
    store = songs.Store(str(tmp_path))
    store.put(name(1), a_song())
    assert len(store.remark(name(1), "la " * 200)) == songs.NOTE_LETTERS


def test_a_note_can_be_written_without_the_song_being_rewritten(tmp_path):
    """Changing a word would otherwise write three quarters of a megabyte of latents again."""
    store = songs.Store(str(tmp_path))
    store.put(name(1), a_song())
    path = tmp_path / (name(1) + songs.SUFFIX)
    os.utime(str(path), (1000.0, 1000.0))
    store.remark(name(1), "a word")
    assert os.path.getmtime(str(path)) == 1000.0


def test_deleting_a_song_takes_everything_kept_beside_it(tmp_path):
    """Its note, its sound, the grid a node measured on it, and the other name its sound has."""
    store = songs.Store(str(tmp_path))
    store.put(name(1), a_song())
    store.put(name(2), a_song(style="the other one"))
    store.alias(name(3), name(1))
    store.remark(name(1), "gone soon")
    store.sound_write(name(1), b"this stands in for a flac")
    (tmp_path / (name(1) + ".grid.json")).write_bytes(b"{}")
    gone = store.drop(name(1))
    assert gone["keys"] == [name(1)] and gone["bytes"] > 0
    assert os.listdir(str(tmp_path)) == [name(2) + songs.SUFFIX], "and nothing of another song"
    assert store.get(name(1)) is None and store.get(name(3)) is None
    assert store.get(name(2)) is not None
    assert store.drop(name(1))["keys"] == [], "and deleting it twice deletes nothing"


def test_deleting_a_song_can_take_the_edits_made_of_it(tmp_path):
    """The row a picker folds the edits under offers both: this edit, or the song and its line."""
    store = songs.Store(str(tmp_path))
    store.put(name(1), a_song())
    store.put(name(2), dataclasses.replace(a_song(), parent=name(1), root=name(1)))
    store.put(name(3), dataclasses.replace(a_song(), parent=name(2), root=name(1)))
    store.put(name(4), a_song(style="a song of its own"))
    assert store.drop(name(2))["keys"] == [name(2)], "one edit and no more"
    assert store.drop(name(1), family=True)["keys"] == sorted([name(1), name(3)])
    assert [row["key"] for row in store.listing()] == [name(4)]


def test_a_line_whose_first_song_is_gone_is_still_deleted_as_one(tmp_path):
    """The picker heads that line with the oldest edit left, so that row must delete the line."""
    store = songs.Store(str(tmp_path))
    store.put(name(2), dataclasses.replace(a_song(), parent=name(1), root=name(1)))
    store.put(name(3), dataclasses.replace(a_song(), parent=name(2), root=name(1)))
    assert store.line(name(2)) == sorted([name(2), name(3)])
    assert store.drop(name(2), family=True)["keys"] == sorted([name(2), name(3)])
    assert store.listing() == []


def test_the_switch_that_keeps_the_sounds_outlives_the_session(tmp_path):
    store = songs.Store(str(tmp_path))
    assert store.wants_sound() is False
    assert store.keep_sound(True) is True
    again = songs.Store(str(tmp_path))
    assert again.wants_sound() is True
    said = again.sounds()
    assert said["on"] is True and said["count"] == 0 and said["bytes"] == 0
    assert said["budget"] == songs.SOUND_BYTES
    assert again.keep_sound(False) is False and songs.Store(str(tmp_path)).wants_sound() is False


def test_the_sounds_have_a_budget_of_their_own_and_the_songs_keep_theirs(tmp_path):
    """A sound is a decode away from being back, so it is what goes when there is no room."""
    store = songs.Store(str(tmp_path), sound_bytes=400)
    for index in (1, 2, 3):
        store.put(name(index), a_song())
        store.sound_write(name(index), bytes(300))
    said = store.sounds()
    assert said["count"] == 1 and said["bytes"] == 300, "the sound written last is the one kept"
    assert store.sound_read(name(3)) == bytes(300)
    assert store.sound_read(name(1)) is None
    assert len(store.listing()) == 3, "and every song is still a song"


def test_the_listing_says_which_songs_have_their_sound_beside_them(tmp_path):
    store = songs.Store(str(tmp_path))
    store.put(name(1), a_song())
    store.put(name(2), a_song())
    store.sound_write(name(2), bytes(128))
    rows = {row["key"]: row["sound"] for row in store.listing()}
    assert rows[name(1)] == 0 and rows[name(2)] == 128


def test_every_sound_can_be_swept_without_touching_a_song(tmp_path):
    store = songs.Store(str(tmp_path))
    for index in (1, 2):
        store.put(name(index), a_song())
        store.sound_write(name(index), bytes(64))
    assert store.sweep_sounds() == {"count": 2, "bytes": 128}
    assert store.sounds()["count"] == 0
    assert len(store.listing()) == 2 and store.get(name(1)) == a_song()


def test_a_song_the_budget_drops_takes_its_sound_and_its_note_with_it(tmp_path):
    """Otherwise the folder keeps 22 MB of sound for a song nobody can open any more."""
    store = songs.Store(str(tmp_path), disk_bytes=1)
    store.put(name(1), a_song())
    store.remark(name(1), "about to go")
    store.sound_write(name(1), bytes(4096))
    store.put(name(2), a_song())
    assert sorted(os.listdir(str(tmp_path))) == [name(2) + songs.SUFFIX]


def test_a_song_comes_back_from_the_sound_kept_beside_it_under_the_same_key(tmp_path,
                                                                            monkeypatch):
    """The point of keeping it: the same song, the same key, and no decode to get there."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("av")
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    assert songs.keep_sounds(True)["on"] is True
    waveform = noise_song(torch, seconds=0.4)
    key = songs.remember(waveform, 48000, a_song())
    assert songs.sounds()["count"] == 1
    back = songs.sound_of(key)
    assert back["sample_rate"] == 48000
    assert tuple(back["waveform"].shape) == tuple(waveform.shape)
    assert torch.equal(back["waveform"], flac_trip(torch, waveform)), "16-bit, as a FLAC is"
    assert songs.key(back["waveform"], 48000) == key, "so it is the song, not another name"
    assert songs.sound_of(name(9)) is None


def test_a_sound_that_cannot_be_read_only_means_decoding_the_song(tmp_path, monkeypatch, caplog):
    pytest.importorskip("torch")
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    songs.store().put(name(1), a_song())
    songs.store().sound_write(name(1), b"not a flac at all")
    with caplog.at_level(logging.WARNING):
        assert songs.sound_of(name(1)) is None
    assert "decoded instead" in caplog.text


def test_the_route_hands_the_window_the_songs_and_says_when_it_cannot(tmp_path, monkeypatch):
    """The picker asks for this list; a store that throws must not take the window with it."""
    from yue2_comfy import routes

    monkeypatch.setattr(songs, "_store", _kept(tmp_path, [(name(1), a_song(style="a jig"))]))
    payload, status = routes.answer_songs()
    assert status == 200 and payload["ok"] is True
    assert [row["style"] for row in payload["songs"]] == ["a jig"]

    def broken(limit=0):
        raise RuntimeError("the disk is on fire")

    monkeypatch.setattr(songs, "listing", broken)
    payload, status = routes.answer_songs()
    assert status == 200 and payload["ok"] is False and payload["songs"] == []
    assert "the disk is on fire" in payload["error"]
    assert payload["sounds"]["on"] is False, "and the window is still told about the sounds"


def test_the_route_writes_a_note_and_refuses_what_is_not_a_song(tmp_path, monkeypatch):
    from yue2_comfy import routes

    monkeypatch.setattr(songs, "_store", _kept(tmp_path, [(name(1), a_song())]))
    payload, status = routes.answer_song_note({"key": name(1), "note": "the loud one"})
    assert status == 200 and payload["note"] == "the loud one"
    assert songs.store().note(name(1)) == "the loud one"
    assert routes.answer_song_note({"key": "not a key", "note": ""})[1] == 400
    assert routes.answer_song_note({"note": "no key at all"})[1] == 400
    assert routes.answer_song_note({"key": name(1), "note": 7})[1] == 400


def test_the_route_deletes_a_song_and_says_what_went(tmp_path, monkeypatch):
    from yue2_comfy import routes

    monkeypatch.setattr(songs, "_store", _kept(tmp_path, [
        (name(1), a_song()),
        (name(2), dataclasses.replace(a_song(), parent=name(1), root=name(1)))]))
    payload, status = routes.answer_song_drop({"key": name(2)})
    assert status == 200 and payload["keys"] == [name(2)] and payload["bytes"] > 0
    assert [row["key"] for row in songs.listing()] == [name(1)]
    payload, status = routes.answer_song_drop({"key": name(1), "family": True})
    assert payload["keys"] == [name(1)] and songs.listing() == []
    assert routes.answer_song_drop({"key": "nope"})[1] == 400


def test_the_route_turns_the_kept_sounds_on_and_sweeps_them(tmp_path, monkeypatch):
    from yue2_comfy import routes

    monkeypatch.setattr(songs, "_store", _kept(tmp_path, [(name(1), a_song())]))
    songs.store().sound_write(name(1), bytes(256))
    payload, status = routes.answer_song_sounds({"on": True})
    assert status == 200 and payload["ok"] is True and payload["on"] is True
    assert payload["count"] == 1 and payload["bytes"] == 256
    assert songs.store().wants_sound() is True
    payload, _status = routes.answer_song_sounds({"sweep": True})
    assert payload["gone"] == {"count": 1, "bytes": 256} and payload["count"] == 0
    assert payload["on"] is True, "sweeping is not switching off"
    assert routes.answer_song_sounds({})[1] == 400
    assert routes.answer_song_sounds(None)[1] == 400
