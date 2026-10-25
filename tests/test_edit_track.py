"""The Edit Track node: what it refuses, what it sings, what it does not sing twice, and what it hands the window.

The model is not here. What is here is everything around it: finding the song
behind the audio, working the edits out one on top of another, keeping the
takes so that a second run of the same list sings nothing, choosing between
them, and the payload the track window draws. The singing itself is stood in
for, which is the only way to check that an edit list of three sings three
times and a fourth run sings none.
"""

from __future__ import annotations

import json
import re
import time
import types

import pytest

from yue2_comfy import edit_track, notation, songs
from yue2_comfy.constants import FRAME_SECONDS
from yue2_comfy.edits import EDIT_TRACK_UI
from yue2_comfy.inpaint import grid, track

HEADER = ('X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n'
          'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
          'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n')


def score(*sections):
    lines = [HEADER.rstrip("\n")]
    for name, bars in sections:
        lines.append("% " + name)
        for start in range(0, len(bars), 4):
            group = bars[start:start + 4]
            lines += ["V: Vocal", "|".join(group) + "|", "V: Ins", "|".join(["Z"] * len(group)) + "|"]
    return "\n".join(lines) + "\n"


RAP = score(("intro", ["z16", "z8z2DDA2AA"]),
            ("verse", ["A2DDAAA2AGGGG2GG", "AGGGGFFFFDD2B4"]),
            ("chorus", ["A2z8D2B4", "A2z4FG3F2B2B2", "A2z8DDBB3", "A2z8z2G2G2"]),
            ("outro", ["z16", "z16"]))

LYRICS = "[Intro]\n\n[Verse]\none two three\nfour five six\n\n[Chorus]\nseven eight\n\n[Outro]"

FRAMES = 525
"""Twenty-one seconds: the ten bars from 0.5 s and a little after them."""

RATE = 48000

SAMPLES_A_FRAME = 1920

SPEECH_FOLDER = edit_track._speech_folder
"""The real lookup, kept before the fixture below stands it in for every test."""

VOICE_PARTS = edit_track._voice_parts
"""The real separating of a song's sections, put back by the test about it."""

VOICE_LEVEL = edit_track._voice_level
"""The real one too: a move hears the voice through it, and a test about that puts it back."""

WORDS = ('[{"op": "words", "bars": [4, 8], "lines": [3, 4], "text": "one two four", '
         '"seed": 40}]')
"""A change of words on the stretch selected: line 4 of the words, sung over bars 5 to 8."""

GOES_ON = "% bridge\nV: Vocal\nA2z8D2B4|A2z4FG3F2B2B2|\nV: Ins\nZ|Z|\n"
"""What the stand-in for the model writes after the score it goes on from: two bars of a bridge."""

BRIDGE = "[Bridge]\nnine ten eleven"


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def a_song(frames=FRAMES, score_text=RAP, lyrics=LYRICS, settings=None, style="a style"):
    """A song the store will take: the smallest one that is still a song."""
    return songs.made("YuE2 Generate Song", style, lyrics, 7,
                      dict(settings or {"cot": "full"}), score_text, [11, 12, 13], None,
                      [1] * frames, [[3, 0, frames]], b"\0" * (frames * 64 * 4), RATE, 2,
                      frames * SAMPLES_A_FRAME)


def a_wave(torch, frames=FRAMES, fill=0.0):
    return torch.full((1, 2, frames * SAMPLES_A_FRAME), float(fill))


def a_clock():
    sheet = notation.read(RAP)
    return grid.Grid(offset=0.5, rate=1.0, tick=grid.tick_seconds(sheet),
                     starts=grid.starts_of(sheet), by_voice=True)


@pytest.fixture(autouse=True)
def a_temp_folder_of_its_own(tmp_path, monkeypatch):
    """Excerpts go where the test can see them, never into a real ComfyUI.

    The embedded Python this pack is tested with has ComfyUI on its path, so
    folder_paths imports and the node would write its takes into a running
    ComfyUI's temp folder.
    """
    monkeypatch.setattr(edit_track, "_temp_folder", lambda: str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def no_ears_of_the_machine(monkeypatch):
    """The node never reaches the speech models of the machine the tests run on.

    The embedded Python these tests run with finds the aligner and Qwen3-ASR
    where this machine keeps them, so a run would load both onto the card and
    hear silence -- and a machine without them, downloading on, would fetch
    4 GB. A test about hearing stands them in itself. So is the separator a
    move hears the voice with, and the one a song's sections are separated
    with before their words are timed: without it they are heard as mixed.
    What one test heard of a section is not there for the next.
    """
    monkeypatch.setattr(edit_track, "_aligner_at_hand", lambda settings: None)
    monkeypatch.setattr(edit_track, "_speech_folder", lambda settings, progress, notices: None)
    monkeypatch.setattr(edit_track, "_ears_at_hand", lambda settings: None)
    monkeypatch.setattr(edit_track, "_voice_level", lambda *args, **kwargs: None)
    monkeypatch.setattr(edit_track, "_voices_in", lambda *args, **kwargs: [])
    monkeypatch.setattr(edit_track, "_voice_parts", no_separator)
    monkeypatch.setattr(edit_track, "_PARTS", type(edit_track._PARTS)())
    monkeypatch.setattr(edit_track, "_WHOLE_VOICES", type(edit_track._WHOLE_VOICES)())


def no_separator(*args, **kwargs):
    raise FileNotFoundError("the tests have no voice separator")


@pytest.fixture
def stand(torch, monkeypatch):
    """A song in the store, a known grid, and a stand-in for the singing that counts its calls."""
    from yue2_comfy.inpaint import core

    song = a_song()
    wave = a_wave(torch)
    name = songs.remember(wave, RATE, song)
    monkeypatch.setattr(edit_track, "_GRIDS", type(edit_track._GRIDS)())
    monkeypatch.setattr(edit_track, "RESULTS", edit_track.Results())
    monkeypatch.setattr(edit_track, "_grid_of", lambda *args, **kwargs: a_clock())
    monkeypatch.setattr("yue2_comfy.staged.session", _no_models)
    calls = []
    words = []
    scores = []

    def retakes(models, old, waveform, region, seeds, settings, progress=None, cancelled=None,
                noise_seeds=None, natural=False, lyrics=None, score=None):
        calls.append(("retake", region.start, region.stop, tuple(seeds)))
        words.append(lyrics)
        scores.append(score)
        made = []
        for index, seed in enumerate(seeds):
            frames = old.frames - region.removed + region.length
            made.append(core.Take(seed=seed, waveform=a_wave(torch, frames, 0.1 * (index + 1)),
                                  song=a_song(frames, score_text=score or RAP,
                                              lyrics=lyrics or LYRICS), count=region.length,
                                  join=-4.0 + index, joins={}, ended=False, timing={},
                                  natural=-3.0 if natural else None))
        return made

    def cut(models, old, waveform, region, lyrics, score_text, settings, progress=None,
            cancelled=None):
        calls.append(("cut", region.start, region.stop))
        frames = old.frames - region.removed
        return core.Take(seed=0, waveform=a_wave(torch, frames, 0.5),
                         song=a_song(frames, score_text, lyrics), count=0, join=None, joins={},
                         ended=False, timing={})

    def extended(models, old, waveform, start, head, lyrics, seeds, settings, frames_for,
                 progress=None, cancelled=None):
        calls.append(("extend", start, tuple(seeds)))
        words.append(lyrics)
        scores.append(head)
        made = []
        for index, seed in enumerate(seeds):
            written = head + GOES_ON
            count = frames_for(notation.read(written))
            frames = start + count
            made.append(core.Take(seed=seed, waveform=a_wave(torch, frames, 0.1 * (index + 1)),
                                  song=a_song(frames, score_text=written, lyrics=lyrics),
                                  count=count, join=None, joins={}, ended=seed % 2 == 0,
                                  timing={}))
        return made

    def moved(models, old, waveform, pieces, count, lyrics, score_text, settings, progress=None,
              cancelled=None, sung=(), pulse=None):
        calls.append(("move", tuple(tuple(piece) for piece in pieces), count, tuple(sung), pulse))
        words.append(lyrics)
        scores.append(score_text)
        made = sung_by_stand(sung)
        frames = old.frames + sum(new - (stop - start) for start, stop, new in made)
        return core.Take(seed=0, waveform=a_wave(torch, frames, 0.7),
                         song=a_song(frames, score_text, lyrics), count=count, join=None,
                         joins={}, ended=False, timing={}, sung=made)

    def broke(models, old, waveform, start, head, lyrics, bars, seeds, settings, build, frames_for,
              progress=None, cancelled=None):
        calls.append(("break", start, bars, tuple(seeds)))
        words.append(lyrics)
        scores.append(head)
        made = []
        for index, seed in enumerate(seeds):
            written = build(head + played(bars))
            count = frames_for(notation.read(written)) + index
            frames = old.frames + count
            made.append(core.Take(seed=seed, waveform=a_wave(torch, frames, 0.1 * (index + 1)),
                                  song=a_song(frames, score_text=written, lyrics=lyrics),
                                  count=count, join=-4.0 + index, joins={}, ended=False,
                                  timing={}))
        return made

    monkeypatch.setattr(core, "retakes", retakes)
    monkeypatch.setattr(core, "cut", cut)
    monkeypatch.setattr(core, "extended", extended)
    monkeypatch.setattr(core, "moved", moved)
    monkeypatch.setattr(core, "broke", broke)
    return types.SimpleNamespace(song=song, wave=wave, name=name, calls=calls, words=words,
                                 scores=scores, audio={"waveform": wave, "sample_rate": RATE})


def played(bars):
    """What the stand-in for the model writes after the score a break is written from: bars of playing."""
    return ("V: Vocal\n" + "|".join(["z16"] * bars) + "|\nV: Ins\n" + "|".join(["C16"] * bars)
            + "|\n% chorus\nV: Vocal\nc16|\nV: Ins\nZ|\n")


def sung_by_stand(sung):
    """What the stand-in for the model makes of the stretches a move sings again: each two frames longer."""
    made, shift = [], 0
    for start, seam in sung:
        made.append((start + shift, seam + shift, seam - start + SUNG_LONGER))
        shift += SUNG_LONGER
    return tuple(made)


SUNG_LONGER = 2
"""How many frames longer the stand-in sings each stretch before a seam of a move."""


class _no_models:
    """The session, without weights: nothing here reaches the model."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return types.SimpleNamespace(device="cpu")

    def __exit__(self, *exc):
        return False


def run(stand, edits="", takes=2):
    return edit_track.YuE2EditTrack().edit(stand.audio, takes=takes, edits=edits)


def payload(answer):
    return answer["ui"][EDIT_TRACK_UI][0]


def test_the_widgets_sit_in_the_order_a_saved_workflow_expects():
    """ComfyUI stores widget values by position, so this order is part of every saved workflow."""
    shape = edit_track.YuE2EditTrack.INPUT_TYPES()
    assert list(shape["required"]) == ["takes"]
    assert list(shape["optional"]) == ["audio", "edits", "options", "song_key", "use_audio"]
    fields = dict(list(shape["required"].items()) + list(shape["optional"].items()))
    assert [name for name, spec in fields.items() if spec[0] in ("INT", "STRING", "BOOLEAN")] == [
        "takes", "edits", "song_key", "use_audio"], (
        "widget values are stored by position, so takes and edits keep theirs and each new "
        "field goes last")
    assert shape["optional"]["use_audio"][1]["default"] is True, (
        "a workflow saved before the switch has no value for it, and its audio is used")
    assert shape["required"]["takes"][1]["max"] == track.MAX_TAKES
    assert edit_track.YuE2EditTrack.RETURN_NAMES == ("audio",)


def test_a_song_the_pack_never_sang_is_refused_with_what_can_be_edited(torch):
    node = edit_track.YuE2EditTrack()
    with pytest.raises(ValueError, match="does not remember singing this audio"):
        node.edit({"waveform": a_wave(torch), "sample_rate": RATE})
    with pytest.raises(ValueError, match="Nothing is chosen to edit"):
        node.edit(None)


def test_a_song_whose_sound_is_only_its_voice_is_refused(torch):
    song = a_song(settings={"cot": "full", "vocals_only": True})
    wave = a_wave(torch, fill=0.25)
    songs.remember(wave, RATE, song)
    with pytest.raises(ValueError, match="vocals_only"):
        edit_track.YuE2EditTrack().edit({"waveform": wave, "sample_rate": RATE})


def test_a_list_that_is_not_an_edit_list_is_refused_before_anything_loads(stand):
    with pytest.raises(ValueError, match="does not hold a JSON list"):
        run(stand, "{oops}")
    assert stand.calls == []


def test_a_node_with_no_edits_hands_the_song_on_and_draws_the_track(stand):
    answer = run(stand)
    assert answer["result"][0]["waveform"] is stand.wave
    assert stand.calls == []
    drawn = payload(answer)
    assert drawn["seconds"] == pytest.approx(FRAMES * FRAME_SECONDS)
    assert drawn["takes"] == [] and drawn["chosen"] is None
    assert len(drawn["peaks"]) > 100
    assert drawn["grid"]["bars"][:2] == [0.5, 2.5]
    assert [s["name"] for s in drawn["grid"]["sections"]] == ["intro", "verse", "chorus", "outro"]


def a_decoder(monkeypatch, torch, seen=None, fill=0.25):
    """The acoustic stage standing in for itself: the latents it was given, and a sound back."""
    from yue2_comfy import generate

    def decode(models, latents, progress=None, cancelled=None, stages=None):
        if seen is not None:
            seen.append(tuple(latents.shape))
        return a_wave(torch, fill=fill), {"seconds_of_audio": FRAMES * FRAME_SECONDS}

    monkeypatch.setattr(generate, "decode", decode)


def test_a_song_can_be_opened_by_its_key_with_nothing_joined_to_audio(stand, monkeypatch, torch):
    """A workflow opened after a restart has no graph above this node, only the song it was on.

    Singing that song again to edit a bar of it costs minutes; its latents are
    already remembered, and the last stage of singing brings the sound back.
    """
    seen = []
    a_decoder(monkeypatch, torch, seen)
    answer = edit_track.YuE2EditTrack().edit(None, song_key=stand.name)
    assert seen == [(FRAMES, 64)], "the song came back from its own latents"
    assert stand.calls == [], "and nothing was sung"
    drawn = payload(answer)
    assert drawn["was"] == stand.name
    assert answer["result"][0]["waveform"][0, 0, 0] == pytest.approx(0.25)


def test_the_sound_a_decode_makes_is_the_same_song_under_another_name(stand, monkeypatch,
                                                                      torch, tmp_path):
    """It is not the same file as the one that was saved, so it needs a name of its own.

    Otherwise a FLAC of what this node just handed on would come back as a
    song the pack never sang. Remembering it as a song of its own instead
    wrote every latent a second time and put a twin in the picker, on the
    first open of every song an edit had made.
    """
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    songs.store().put(stand.name, stand.song)
    a_decoder(monkeypatch, torch)
    answer = edit_track.YuE2EditTrack().edit(None, song_key=stand.name)
    assert payload(answer)["song"] == stand.name, "the song is the one that was opened"
    made = songs.key(answer["result"][0]["waveform"], RATE)
    assert made != stand.name, "the sound it hands on is not the sound the song was saved from"
    assert songs.store().get(made) is not None, "and that sound finds the song"
    assert [row["key"] for row in songs.listing()] == [stand.name], "one song, one row"


def test_a_song_opened_by_one_of_its_other_names_is_worked_on_under_its_own(stand, monkeypatch,
                                                                            torch, tmp_path):
    """Its grid, its takes and the mark in the picker all hang off the one key."""
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    songs.store().put(stand.name, stand.song)
    other = "1" * 64
    songs.store().alias(other, stand.name)
    a_decoder(monkeypatch, torch)
    drawn = payload(edit_track.YuE2EditTrack().edit(None, song_key=other))
    assert drawn["was"] == stand.name


def test_a_song_whose_sound_is_kept_beside_it_is_read_and_not_decoded(stand, monkeypatch,
                                                                       torch, tmp_path):
    """What the switch in 'Saved songs...' buys: opening a song takes no card at all."""
    pytest.importorskip("av")
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    songs.store().keep_sound(True)
    songs.store().put(stand.name, stand.song)
    assert songs.sound_keep(stand.name, stand.wave, RATE) is True
    seen = []
    a_decoder(monkeypatch, torch, seen)
    answer = edit_track.YuE2EditTrack().edit(None, song_key=stand.name)
    assert seen == [], "nothing was decoded"
    assert payload(answer)["was"] == stand.name
    assert torch.equal(answer["result"][0]["waveform"], stand.wave), "and it is the song's own"


def test_a_song_opened_with_the_switch_on_keeps_the_sound_it_decoded(stand, monkeypatch,
                                                                     torch, tmp_path):
    """A song sung before the switch was turned on pays one decode, the first time it is opened."""
    pytest.importorskip("av")
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    songs.store().keep_sound(True)
    songs.store().put(stand.name, stand.song)
    seen = []
    a_decoder(monkeypatch, torch, seen)
    edit_track.YuE2EditTrack().edit(None, song_key=stand.name)
    assert seen == [(FRAMES, 64)] and songs.store().sounds()["count"] == 1
    seen.clear()
    answer = edit_track.YuE2EditTrack().edit(None, song_key=stand.name)
    assert seen == [], "and never again"
    assert answer["result"][0]["waveform"][0, 0, 0] == pytest.approx(0.25)


def test_a_sound_that_is_not_this_songs_is_left_where_it_is(stand, monkeypatch, torch, tmp_path):
    """A file of another length is not this song, whatever the name on it says."""
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    songs.store().put(stand.name, stand.song)
    monkeypatch.setattr(songs, "sound_of",
                        lambda key: {"waveform": a_wave(torch, 3), "sample_rate": RATE})
    seen = []
    a_decoder(monkeypatch, torch, seen)
    edit_track.YuE2EditTrack().edit(None, song_key=stand.name)
    assert seen == [(FRAMES, 64)], "so the song was decoded as it would have been"


def test_an_edit_keeps_its_own_sound_beside_it_too(stand, monkeypatch, torch, tmp_path):
    """The song an edit makes is a song like any other, and opening it again is a read."""
    pytest.importorskip("av")
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    songs.store().keep_sound(True)
    songs.store().put(stand.name, stand.song)
    drawn = payload(run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]'))
    assert songs.store().sound_read(drawn["song"]) is not None
    assert songs.store().sounds()["count"] == 1, "the song it was made from was not sung again"


def test_a_song_joined_to_audio_wins_over_the_key(stand, monkeypatch, torch):
    """The graph above the node is what the run is about; the field is for when there is none."""
    a_decoder(monkeypatch, torch)
    drawn = payload(edit_track.YuE2EditTrack().edit(stand.audio, song_key="0" * 64))
    assert drawn["was"] == stand.name


def test_an_edit_on_a_song_opened_by_key_is_sung_on_the_sound_it_decoded(stand, monkeypatch,
                                                                         torch):
    a_decoder(monkeypatch, torch)
    answer = edit_track.YuE2EditTrack().edit(
        None, song_key=stand.name, edits='[{"op": "retake", "bars": [4, 8], "seed": 40}]')
    assert stand.calls == [("retake", 192, 414, (40, 41))]
    assert [take["seed"] for take in payload(answer)["takes"]] == [40, 41]


def test_a_key_that_names_nothing_and_a_key_that_is_not_one_both_say_so(stand, torch):
    node = edit_track.YuE2EditTrack()
    with pytest.raises(ValueError, match="no longer remembered"):
        node.edit(None, song_key="0" * 64)
    with pytest.raises(ValueError, match="does not hold a song key"):
        node.edit(None, song_key="the third one down")


def test_a_voice_only_song_is_refused_whichever_way_it_is_opened(stand, torch):
    voice = songs.made("YuE2 Generate Song", "a style", LYRICS, 7, {"vocals_only": True}, RAP,
                       [11, 12, 13], None, [1] * FRAMES, [[3, 0, FRAMES]],
                       b"\0" * (FRAMES * 64 * 4), RATE, 2, FRAMES * SAMPLES_A_FRAME)
    key = songs.remember(a_wave(torch, fill=0.75), RATE, voice)
    with pytest.raises(ValueError, match="vocals_only"):
        edit_track.YuE2EditTrack().edit(None, song_key=key)


def test_the_grid_measured_for_a_song_waits_beside_it_for_the_next_session(tmp_path, monkeypatch):
    """Measuring one separates the song's voice -- eight seconds of card -- and never changes.

    A song is remembered under the key of its own samples, so the place its
    score sits is an answer about that key and nothing else. It used to live
    in the session alone, and every song reopened after a restart paid the
    separator again.
    """
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    key = "{:064x}".format(3)
    edit_track._grid_keep(key, a_clock())
    assert edit_track._grid_read(key, notation.read(RAP)) == a_clock()


def test_a_grid_that_cannot_be_read_only_means_measuring_it_again(tmp_path, monkeypatch):
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    key = "{:064x}".format(3)
    (tmp_path / (key + edit_track.GRID_SUFFIX)).write_text("half a file", encoding="utf-8")
    assert edit_track._grid_read(key, notation.read(RAP)) is None
    assert edit_track._grid_read("not a key", notation.read(RAP)) is None
    assert edit_track._grid_file("not a key") is None


def test_a_song_whose_grid_is_on_disk_does_not_separate_its_voice_again(tmp_path, monkeypatch,
                                                                        torch):
    """Which is the whole point: the second session opens the song without the separator."""
    from yue2_comfy import vocals_only

    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    monkeypatch.setattr(edit_track, "_GRIDS", type(edit_track._GRIDS)())
    key = "{:064x}".format(4)
    edit_track._grid_keep(key, a_clock())
    monkeypatch.setattr(vocals_only, "voice_of",
                        lambda *args, **kwargs: pytest.fail("the voice was separated again"))
    monkeypatch.setattr(vocals_only, "separator_weights",
                        lambda *args, **kwargs: pytest.fail("the separator was loaded again"))
    assert edit_track._grid_of(key, a_song(), a_wave(torch), RATE, {}, None, None) == a_clock()


def test_a_retake_sings_its_takes_and_keeps_the_one_that_joins_best(stand):
    answer = run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]')
    assert stand.calls == [("retake", 192, 414, (40, 41))]
    drawn = payload(answer)
    assert drawn["at"] == [pytest.approx(7.68), pytest.approx(16.56)]
    assert drawn["chosen"] == 1
    assert [take["seed"] for take in drawn["takes"]] == [40, 41]
    assert [take["kept"] for take in drawn["takes"]] == [False, True]
    assert drawn["takes"][1]["join"] == -3.0
    assert answer["result"][0]["waveform"][0, 0, 0] == pytest.approx(0.2)


def test_the_same_list_run_again_sings_nothing(stand):
    text = '[{"op": "retake", "bars": [4, 8], "seed": 40}]'
    first = run(stand, text)
    stand.calls.clear()
    again = run(stand, text)
    assert stand.calls == []
    assert payload(again)["chosen"] == payload(first)["chosen"]


def test_keeping_another_take_costs_nothing(stand):
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]')
    stand.calls.clear()
    answer = run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "take": 0}]')
    assert stand.calls == []
    assert payload(answer)["chosen"] == 0
    assert answer["result"][0]["waveform"][0, 0, 0] == pytest.approx(0.1)


def test_asking_for_a_third_take_sings_only_the_third(stand):
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]')
    stand.calls.clear()
    answer = run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "takes": 3}]')
    assert stand.calls == [("retake", 192, 414, (42,))]
    assert [take["seed"] for take in payload(answer)["takes"]] == [40, 41, 42]


def test_a_list_that_already_keeps_a_take_sings_that_one_alone(stand):
    """A workflow saved with a take chosen, opened where nothing of it was sung.

    The takes live in the session that sang them. Reopening a saved list in a
    fresh ComfyUI found none of them, and the node sang all four of an edit to
    throw three of them away -- what the user saw as a node that "generates
    four takes at the first load". The seeds are counted on from the edit's
    own, so the take the list kept is the only one the song needs.
    """
    answer = run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "takes": 4, "take": 2}]')
    assert stand.calls == [("retake", 192, 414, (42,))]
    drawn = payload(answer)
    assert [take["seed"] for take in drawn["takes"]] == [40, 41, 42, 43]
    assert [take["sung"] for take in drawn["takes"]] == [False, False, True, False]
    assert drawn["chosen"] == 2
    assert answer["result"][0]["waveform"][0, 0, 0] == pytest.approx(0.1)


def test_a_take_this_session_never_sang_is_offered_as_its_seed(stand):
    """The window draws a row for it, so it has to say which take it is and that it has no sound."""
    answer = run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "takes": 4, "take": 2}]')
    gap = payload(answer)["takes"][0]
    assert gap["sung"] is False and gap["seed"] == 40 and gap["index"] == 0
    assert gap["audio"] is None and gap["peaks"] == [] and gap["grid"] is None
    assert gap["join"] is None and gap["kept"] is False and gap["flagged"] is False
    assert gap["seconds"] is None and gap["total"] is None


def test_asking_for_one_of_the_takes_left_sings_it_and_keeps_the_other(stand):
    """'Sing it' moves the take the list keeps; the one already sung is not sung again."""
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "takes": 4, "take": 2}]')
    stand.calls.clear()
    answer = run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "takes": 4, "take": 0}]')
    assert stand.calls == [("retake", 192, 414, (40,))]
    assert [take["sung"] for take in payload(answer)["takes"]] == [True, False, True, False]
    assert payload(answer)["chosen"] == 0


def test_a_retake_with_no_take_kept_yet_still_sings_them_all(stand):
    """The pick is made among the takes, so a list that has not chosen needs every one of them."""
    answer = run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "takes": 3}]')
    assert stand.calls == [("retake", 192, 414, (40, 41, 42))]
    assert [take["sung"] for take in payload(answer)["takes"]] == [True, True, True]


def test_an_edit_added_at_the_end_sings_only_itself(stand):
    text = '[{"op": "cut", "bars": [2, 4]}]'
    run(stand, text)
    assert stand.calls == [("cut", 92, 192)]
    stand.calls.clear()
    answer = run(stand, '[{"op": "cut", "bars": [2, 4]}, {"op": "retake", "bars": [2, 6], "seed": 9}]')
    assert stand.calls == [("retake", 81, 314, (9, 10))]
    drawn = payload(answer)
    assert drawn["seconds"] == pytest.approx((FRAMES - 100) * FRAME_SECONDS)
    assert [s["name"] for s in drawn["grid"]["sections"]] == ["intro", "chorus", "outro"]


def test_undoing_the_last_edit_costs_nothing(stand):
    run(stand, '[{"op": "cut", "bars": [2, 4]}, {"op": "retake", "bars": [2, 6], "seed": 9}]')
    stand.calls.clear()
    answer = run(stand, '[{"op": "cut", "bars": [2, 4]}]')
    assert stand.calls == []
    assert payload(answer)["kind"] == "cut"


def test_a_take_sung_later_stands_against_the_same_song(stand):
    """The song's own join is scored once, before the first take; a take asked for afterwards has
    to be shown against that same number, not against nothing."""
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]')
    answer = run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "takes": 3}]')
    assert [take["natural"] for take in payload(answer)["takes"]] == [-3.0, -3.0, -3.0]


def test_the_list_handed_back_names_the_take_kept_for_every_edit(stand):
    """The window saves what comes back, and an earlier edit whose take is unnamed would be a
    different song the day the takes are no longer in memory."""
    answer = run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}, {"op": "cut", "bars": [2, 4]}]')
    made = json.loads(payload(answer)["edits"])
    assert made[0]["take"] == 1
    assert "take" not in made[1]


def test_every_take_is_written_where_it_can_be_heard(stand, a_temp_folder_of_its_own):
    """The window plays the takes from ComfyUI's temp folder, so a take that wrote no file
    cannot be chosen by ear."""
    answer = run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]')
    written = sorted(path.name for path in (a_temp_folder_of_its_own / "yue2_edit").iterdir())
    assert len(written) == 4, "two takes, the song, and the song as it was before the edit"
    assert [name for name in written if name.endswith("_40.wav")]
    assert [name for name in written if name.startswith("song_")]
    assert [name for name in written if name.endswith("_was.wav")]
    drawn = payload(answer)
    for take in drawn["takes"] + [drawn["before"]]:
        assert take["audio"]["subfolder"] == "yue2_edit"
        assert take["audio"]["type"] == "temp"


def test_the_song_as_it_was_is_offered_beside_the_takes_of_a_retake(stand):
    """Comparing a retake with what it replaced is why anyone retakes twice."""
    drawn = payload(run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]'))
    was = drawn["before"]
    assert was["index"] == -1 and was["seed"] is None and was["kept"] is False
    assert was["sung"] is True, "it is not a take to sing, it is the song that was there"
    assert was["seconds"] == pytest.approx((drawn["at"][1] - drawn["at"][0])), (
        "the stretch as it stood, against a take's own length")
    assert was["total"] == pytest.approx(FRAMES * FRAME_SECONDS)
    assert len(was["peaks"]) > 100 and was["grid"]["bars"][:2] == [0.5, 2.5]


def test_a_cut_has_nothing_to_compare_and_writes_no_file_for_it(stand,
                                                                a_temp_folder_of_its_own):
    drawn = payload(run(stand, '[{"op": "cut", "bars": [2, 4]}]'))
    assert "before" not in drawn
    written = [path.name for path in (a_temp_folder_of_its_own / "yue2_edit").iterdir()]
    assert not [name for name in written if name.endswith("_was.wav")]


def test_an_edit_writes_down_what_it_was_made_from_and_what_it_did(stand, tmp_path,
                                                                    monkeypatch):
    """The picker groups the edits of a song under it and paints the stretch each one changed."""
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    songs.store().put(stand.name, stand.song)
    drawn = payload(run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]'))
    child = songs.store().get(drawn["song"])
    assert child.parent == stand.name, "the song it was made from"
    assert child.root == stand.name, "and the first of the line, which is that same song"
    assert child.created == pytest.approx(time.time(), abs=60)
    assert 0 < len(child.peaks) <= songs.STRIP and len(child.body) == len(child.peaks)
    assert child.edit == [{"op": "retake", "at": [drawn["at"][0], drawn["at"][1]],
                           "bars": [4, 8], "seed": 41, "was": "", "now": "",
                           "took": pytest.approx(drawn["at"][1] - drawn["at"][0])}]


def test_the_line_of_a_song_holds_however_many_edits_it_takes(stand, tmp_path, monkeypatch,
                                                              torch):
    """A cut on a retake is the first song's grandchild, and says so without walking anywhere."""
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    songs.store().put(stand.name, stand.song)
    first = payload(run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]'))["song"]
    a_decoder(monkeypatch, torch)
    node = edit_track.YuE2EditTrack()
    second = payload(node.edit(None, song_key=first, edits='[{"op": "cut", "bars": [2, 4]}]'))
    grand = songs.store().get(second["song"])
    assert grand.parent == first and grand.root == stand.name
    assert [mark["op"] for mark in grand.edit] == ["cut"]
    assert grand.edit[0]["at"][0] == grand.edit[0]["at"][1], "a cut takes up no room after it"


def test_two_edits_in_one_run_are_written_where_they_ended_up(stand, tmp_path, monkeypatch):
    """A cut before a retake moves it; the marks are where they are in the song that came out."""
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    songs.store().put(stand.name, stand.song)
    alone = payload(run(stand, '[{"op": "retake", "bars": [8, 10], "seed": 40}]'))["at"][0]
    drawn = payload(run(stand, '[{"op": "retake", "bars": [8, 10], "seed": 40},'
                               ' {"op": "cut", "bars": [2, 4]}]'))
    marks = songs.store().get(drawn["song"]).edit
    assert [mark["op"] for mark in marks] == ["retake", "cut"]
    assert marks[1]["at"][0] < marks[0]["at"][0], "the cut is earlier in the song than the retake"
    assert marks[0]["at"][0] == pytest.approx(alone - marks[1]["took"]), (
        "and moved the retake up by what it took out")


def test_a_retake_can_be_sung_at_a_temperature_and_a_guide_of_its_own(stand, monkeypatch):
    """What the window's knobs are for: this one stretch, not the whole run."""
    from yue2_comfy.inpaint import core

    seen = []
    sings = core.retakes

    def watched(models, old, waveform, region, seeds, settings, *args, **kwargs):
        seen.append((settings["temperature"], settings["cfg_scale"]))
        return sings(models, old, waveform, region, seeds, settings, *args, **kwargs)

    monkeypatch.setattr(core, "retakes", watched)
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "vary": 1.6, "guide": 2.5}]')
    assert seen == [(1.6, 2.5)]
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 41}]')
    assert seen[-1] == (1.0, 0.0), "and an edit that asks for nothing is sung as the song was"


def test_a_retake_sung_at_another_temperature_is_sung_again(stand):
    """Takes sung under other settings are other takes, so the memory of them is keyed by it."""
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "vary": 1.2}]')
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "vary": 1.2}]')
    assert len(stand.calls) == 1, "the same edit twice sings once"
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "vary": 1.9}]')
    assert len(stand.calls) == 2, "another temperature is another take"


def test_a_cut_that_takes_the_first_bars_fades_the_song_in(stand):
    """A cut inside the song is crossfaded at both ends; one at the edge has nothing to fade to."""
    sound = run(stand, '[{"op": "cut", "bars": [0, 4], "fade": 0.5}]')["result"][0]["waveform"]
    assert float(sound[0, 0, 0]) < 0.01, "so it comes in rather than starting mid-signal"
    assert float(sound[0, 0, int(0.5 * RATE)]) == pytest.approx(0.5, abs=0.01)
    assert float(sound[0, 0, -1]) == pytest.approx(0.5), "and the far end is untouched"


def test_a_cut_that_takes_the_last_bars_fades_the_song_out(stand):
    sound = run(stand, '[{"op": "cut", "bars": [6, 10], "fade": 1.0}]')["result"][0]["waveform"]
    assert float(sound[0, 0, -1]) < 0.01
    assert float(sound[0, 0, 0]) == pytest.approx(0.5)


def test_a_cut_in_the_middle_is_left_as_the_join_made_it(stand):
    sound = run(stand, '[{"op": "cut", "bars": [4, 8], "fade": 0.5}]')["result"][0]["waveform"]
    assert float(sound[0, 0, 0]) == pytest.approx(0.5)
    assert float(sound[0, 0, -1]) == pytest.approx(0.5)


def test_the_fade_is_laid_on_a_copy_so_the_take_kept_is_not_touched(stand, torch):
    """The takes live in this session's memory; a fade must not ramp the one in the cache."""
    answer = run(stand, '[{"op": "cut", "bars": [0, 4], "fade": 0.5}]')
    again = run(stand, '[{"op": "cut", "bars": [0, 4], "fade": 0.5}]')
    assert torch.equal(answer["result"][0]["waveform"], again["result"][0]["waveform"])
    assert len(stand.calls) == 1, "and it was not sung twice to find that out"


def test_the_track_says_what_the_song_itself_was_sung_with(stand):
    """The window's knobs start where the song is, so the payload carries it."""
    assert payload(run(stand))["sung"] == {"vary": 1.0, "guide": 1.0}


def test_a_cut_takes_the_words_of_the_section_it_empties(stand):
    answer = run(stand, '[{"op": "cut", "bars": [2, 4]}]')
    drawn = payload(answer)
    assert drawn["dropped"] == ["Verse"]
    assert drawn["takes"][0]["seconds"] == 0.0


def test_the_track_carries_the_words_the_song_sings_now(stand):
    """The window draws the words beside the track, so a cut has to change them there too."""
    assert payload(run(stand))["lyrics"] == LYRICS
    after = payload(run(stand, '[{"op": "cut", "bars": [2, 4]}]'))
    assert after["dropped"] == ["Verse"]
    assert "one two three" not in after["lyrics"] and "[Chorus]" in after["lyrics"]


def test_the_edited_song_is_remembered_so_it_can_be_edited_again(stand):
    answer = run(stand, '[{"op": "cut", "bars": [2, 4]}]')
    made = payload(answer)["song"]
    assert made != stand.name
    again = songs.store().get(made)
    assert again is not None and again.frames == FRAMES - 100


def test_a_take_whose_join_is_far_below_the_songs_own_is_flagged(stand):
    """The stand's one bad retake scored a point and a half below the song; the window says so."""
    from yue2_comfy.inpaint import core

    assert edit_track._flagged(core.Take(0, None, None, 1, -4.9, {}, False, {}, natural=-3.4))
    assert not edit_track._flagged(core.Take(0, None, None, 1, -3.5, {}, False, {}, natural=-3.4))
    assert not edit_track._flagged(core.Take(0, None, None, 1, -9.0, {}, False, {}, natural=None))


def test_a_song_sung_through_an_adapter_is_edited_through_the_same_one(torch, monkeypatch):
    """0.8.0 put a song's adapters in its memory. An edit that did not fold them would sing the
    retake in the base model's voice, into a song in another, which is the one way an edit can be
    wrong that no join score would catch."""
    from yue2_comfy.inpaint import core

    rows = [{"name": "a voice", "file": "voice.safetensors", "sha256": "0" * 64,
             "ar": 1.0, "nar": 0.8}]
    song = a_song(settings={"cot": "full", "loras": rows})
    wave = a_wave(torch, fill=0.75)
    songs.remember(wave, RATE, song)
    seen = {}

    def session(settings, unique_id, progress, loras=None):
        seen["settings"] = dict(settings)
        return _no_models()

    def retakes(models, old, waveform, region, seeds, settings, *args, **kwargs):
        seen["sung"] = dict(settings)
        return [core.Take(seed=seeds[0], waveform=a_wave(torch, old.frames), song=a_song(),
                          count=region.length, join=-3.0, joins={}, ended=False, timing={})]

    monkeypatch.setattr(edit_track, "_GRIDS", type(edit_track._GRIDS)())
    monkeypatch.setattr(edit_track, "RESULTS", edit_track.Results())
    monkeypatch.setattr(edit_track, "_grid_of", lambda *args, **kwargs: a_clock())
    monkeypatch.setattr("yue2_comfy.staged.session", session)
    monkeypatch.setattr("yue2_comfy.staged.adapters", lambda settings, node: seen.setdefault(
        "checked", dict(settings)))
    monkeypatch.setattr(core, "retakes", retakes)
    edit_track.YuE2EditTrack().edit({"waveform": wave, "sample_rate": RATE}, takes=1,
                                    edits='[{"op": "retake", "bars": [4, 8], "seed": 1}]')
    assert seen["settings"]["loras"] == rows
    assert seen["checked"]["loras"] == rows
    assert seen["sung"]["loras"] == rows


def test_an_entry_grown_in_place_is_measured_again_when_it_is_put_back():
    """A third take is added to the entry the cache already holds, and put back under the same name.
    The cache used to subtract the grown size and add it again, so growth never counted and the
    counter could go below zero once an entry was dropped."""
    held = edit_track.Results(budget=100)
    grown = _entry(10)
    held.put("a", grown)
    grown["takes"][1] = grown["takes"][0]
    grown["takes"][2] = grown["takes"][0]
    held.put("a", grown)
    assert held.size() == 120
    held.put("b", _entry(10))
    assert held.get("a") is None and held.size() == 40
    held.clear()
    assert held.size() == 0 and not edit_track.is_loaded()


def test_a_take_is_written_whole_and_written_again_every_time(stand, a_temp_folder_of_its_own):
    """Its name says nothing about the decoder or the card the take came out of, so a file left
    from an earlier run must not stand in for a take sung again."""
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]')
    place = a_temp_folder_of_its_own / "yue2_edit"
    before = {path.name: path.read_bytes() for path in place.iterdir()}
    for path in place.iterdir():
        path.write_bytes(b"stale")
    edit_track.RESULTS.clear()
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]')
    assert {path.name: path.read_bytes() for path in place.iterdir()} == before


def test_every_take_is_a_whole_song_the_window_can_draw_and_play(stand, a_temp_folder_of_its_own,
                                                                  monkeypatch, torch):
    """Switching takes has to change the track under the cursor without the node running again,
    so each take carries the song it makes: its length, its wave, its own grid and a file of the
    whole thing. Two takes of one edit come out different lengths, and the one not kept moves
    every bar after the edit by the difference, so its grid is not the song's."""
    import wave

    from yue2_comfy.inpaint import core

    shorter = 5

    def retakes(models, old, waveform, region, seeds, settings, progress=None, cancelled=None,
                noise_seeds=None, natural=False, lyrics=None, score=None):
        made = []
        for index, seed in enumerate(seeds):
            count = region.length - shorter * index
            frames = old.frames - region.removed + count
            made.append(core.Take(seed=seed, waveform=a_wave(torch, frames, 0.1 * (index + 1)),
                                  song=a_song(frames), count=count, join=-4.0 - index, joins={},
                                  ended=False, timing={}, natural=None))
        return made

    monkeypatch.setattr(core, "retakes", retakes)
    drawn = payload(run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40, "takes": 2}]'))
    kept, other = drawn["takes"][drawn["chosen"]], drawn["takes"][1 - drawn["chosen"]]
    assert drawn["chosen"] == 0 and kept["kept"] and not other["kept"]
    assert kept["total"] == pytest.approx(drawn["seconds"])
    assert other["total"] == pytest.approx(drawn["seconds"] - shorter * SAMPLES_A_FRAME / RATE)
    assert kept["grid"]["bars"] == drawn["grid"]["bars"]
    moved = shorter * SAMPLES_A_FRAME / RATE
    assert other["grid"]["bars"][0] == drawn["grid"]["bars"][0]
    assert other["grid"]["bars"][-1] == pytest.approx(drawn["grid"]["bars"][-1] - moved, abs=0.002)
    assert other["grid"]["seconds"] == pytest.approx(other["total"], abs=0.002)
    for take in drawn["takes"]:
        assert take["seconds"] < take["total"]
        assert len(take["peaks"]) == len(take["rms"]) <= edit_track.PEAKS
        sound = a_temp_folder_of_its_own / "yue2_edit" / take["audio"]["filename"]
        with wave.open(str(sound), "rb") as handle:
            assert handle.getnchannels() == 2 and handle.getframerate() == RATE
            assert handle.getnframes() / RATE == pytest.approx(take["total"], abs=0.001), (
                "a take is written whole, not as an excerpt")


def test_the_wave_reads_any_sound_the_node_can_be_handed(torch):
    """Stereo is measured over both channels, not over the louder one; a sample that is not a
    number draws as silence rather than breaking the JSON; and a song shorter than the slices
    asked for never gives back more slices than that."""
    one_sided = torch.zeros((1, 2, 6400))
    one_sided[0, 0] = 1.0
    wave = edit_track._wave(one_sided, count=4)
    assert wave["peaks"] == [1.0] * 4
    assert wave["rms"] == pytest.approx([0.7071] * 4, abs=0.0002)
    broken = torch.full((1, 2, 4800), float("nan"))
    assert edit_track._wave(broken, count=2) == {"peaks": [0.0, 0.0], "rms": [0.0, 0.0]}
    short = torch.rand((1, 1, 2399))
    drawn = edit_track._wave(short, count=1200)
    assert len(drawn["peaks"]) == len(drawn["rms"]) <= 1200
    assert edit_track._wave(torch.zeros((1, 2, 0))) == {"peaks": [], "rms": []}


def test_the_wave_carries_both_the_peaks_and_the_body_of_the_sound(torch):
    """A mastered song is a solid block drawn from its peaks alone; the root mean square of the
    same slice is what tells a verse from a chorus."""
    loud = torch.zeros((1, 2, FRAMES * SAMPLES_A_FRAME))
    loud[..., ::64] = 1.0
    wave = edit_track._wave(loud, count=10)
    assert wave["peaks"] == [1.0] * 10
    assert all(0.1 < value < 0.2 for value in wave["rms"]), wave["rms"]
    assert edit_track._wave(torch.zeros((1, 2, 0))) == {"peaks": [], "rms": []}


def test_bars_the_score_does_not_have_are_refused_before_the_voice_is_separated(stand, monkeypatch):
    """Laying the score over the song is a download and a minute of card; a list that asks for bar
    40 of a score of 10 must not pay for it first."""
    def never(*args, **kwargs):
        raise AssertionError("the grid was measured")

    monkeypatch.setattr(edit_track, "_grid_of", never)
    with pytest.raises(ValueError, match="Edit 1: Bars 41 to 48 are not bars of this score"):
        run(stand, '[{"op": "cut", "bars": [40, 48]}]')
    with pytest.raises(ValueError, match="not a stretch of a score"):
        run(stand, '[{"op": "cut", "bars": [4, 2]}]')


def test_a_refusal_from_inside_the_session_is_said_once(stand, monkeypatch):
    """staged.session announces every ValueError that reaches it; a refusal raised while it is open
    used to be said twice."""
    import contextlib

    from yue2_comfy import progress

    said = []
    monkeypatch.setattr(progress, "announce", lambda node, findings, kind="notice": said.extend(
        (kind, text) for _level, text in findings))

    @contextlib.contextmanager
    def session(settings, unique_id, bar, loras=None):
        try:
            yield object()
        except ValueError as error:
            progress.refuse(unique_id, str(error))

    monkeypatch.setattr("yue2_comfy.staged.session", session)
    with pytest.raises(ValueError, match="Edit 2: "):
        edit_track.YuE2EditTrack().edit(stand.audio, takes=2, unique_id="7",
                                        edits='[{"op": "cut", "bars": [2, 4]}, {"op": "cut", "bars": [0, 8]}]')
    assert len([text for kind, text in said if kind == "refusal"]) == 1


def test_the_notice_about_ignored_options_names_only_what_an_edit_sings_by(monkeypatch):
    """An options node always carries every key; the score's sampling and max_seconds have no part
    in an edit, so a notice about them would only puzzle."""
    said = []
    monkeypatch.setattr(edit_track, "announce", lambda node, findings, kind="notice": said.extend(
        text for _level, text in findings))
    song = a_song(settings={"cot": "full", "temperature": 1.0, "max_seconds": 300.0})
    edit_track._settings(song, {"max_seconds": 60.0, "abc_top_k": 40, "device": "cpu"}, "7")
    assert said == []
    edit_track._settings(song, {"temperature": 0.5, "cot": "melody"}, "7")
    assert len(said) == 1 and "cot, temperature" in said[0]


def test_a_result_that_could_not_be_remembered_has_no_key_and_no_stale_preview(stand, monkeypatch,
                                                                              a_temp_folder_of_its_own):
    """When the store refuses the edited song, the payload must not claim the original's key, and
    the preview of the original must not be handed over as the edit."""
    run(stand)
    monkeypatch.setattr(edit_track, "_remember", lambda waveform, song: "")
    answer = run(stand, '[{"op": "cut", "bars": [2, 4]}]')
    drawn = payload(answer)
    assert drawn["song"] == "" and drawn["was"] == stand.name
    assert "audio" not in answer["ui"]


def test_the_model_is_loaded_at_most_once_and_not_at_all_when_everything_is_sung(stand, monkeypatch):
    opened = []

    class counting(_no_models):
        def __init__(self, *args, **kwargs):
            opened.append(1)

    monkeypatch.setattr("yue2_comfy.staged.session", counting)
    run(stand, '[{"op": "cut", "bars": [2, 4]}, {"op": "retake", "bars": [2, 6], "seed": 9}]')
    assert len(opened) == 1
    run(stand, '[{"op": "cut", "bars": [2, 4]}, {"op": "retake", "bars": [2, 6], "seed": 9}]')
    assert len(opened) == 1


def test_the_grids_are_kept_a_few_at_a_time_and_let_go_with_the_takes(monkeypatch):
    grids = type(edit_track._GRIDS)()
    monkeypatch.setattr(edit_track, "_GRIDS", grids)
    monkeypatch.setattr(edit_track, "RESULTS", edit_track.Results())
    assert not edit_track.is_loaded()
    grids["a"] = a_clock()
    assert edit_track.is_loaded()
    edit_track.unload()
    assert not grids and not edit_track.is_loaded()


def test_a_take_without_a_join_shows_no_join_of_the_song_either(stand, monkeypatch):
    """A take the model ended itself has no join score, and a song's own join next to nothing
    would read as a comparison."""
    from yue2_comfy.inpaint import core

    from yue2_comfy.inpaint import track

    take = core.Take(0, a_wave(pytest.importorskip("torch"), 10), None, 5, None, {}, True, {},
                     natural=-3.4)
    prior = track.opened(a_song(score_text=""), None)
    step = track.plan(prior, track.read('[{"op": "retake", "seconds": [0.5, 1.0]}]')[0])
    facts = edit_track._take_facts(take, 0, 0, "x" * 64, RATE, prior, step)
    assert facts["join"] is None and facts["natural"] is None and facts["ended"]
    assert facts["grid"] is None, "no score, so no grid of its own"


def test_the_run_decides_the_card_and_the_song_decides_the_singing():
    song = a_song(settings={"cot": "melody", "temperature": 0.5, "device": "cpu"})
    settings = edit_track._settings(song, {"device": "cuda:0", "cot": "full",
                                           "temperature": 1.5}, None)
    assert settings["device"] == "cuda:0"
    assert settings["cot"] == "melody"
    assert settings["temperature"] == 0.5


def _entry(size):
    """One edit's takes, standing in for the sound they hold."""
    return {"takes": {0: types.SimpleNamespace(waveform=types.SimpleNamespace(
        numel=lambda: size))}, "natural": None}


def test_the_takes_kept_are_dropped_oldest_first_when_the_memory_fills():
    """Ten samples are forty bytes, so two edits fit in a hundred and a third does not. Looking at
    an edit is using it, so the one that goes is the one nobody has asked for."""
    held = edit_track.Results(budget=100)
    held.put("a", _entry(10))
    held.put("b", _entry(10))
    assert held.size() == 80
    assert held.get("a") is not None
    held.put("c", _entry(10))
    assert held.size() == 80
    assert held.get("b") is None
    assert held.get("a") is not None and held.get("c") is not None


TAKE_BYTES = FRAMES * SAMPLES_A_FRAME * 2 * 4
"""What one take of the stand's song holds: every take is the whole song, stereo, in float32."""


def retakes_on(count, take=None):
    """``count`` retakes of bars 5 to 8, one on another, each with a seed of its own; ``take`` keeps a take of the last."""
    edits = [{"op": "retake", "bars": [4, 8], "seed": 40 + 10 * index} for index in range(count)]
    if take is not None:
        edits[-1]["take"] = take
    return json.dumps(edits)


def test_a_list_past_the_memory_for_takes_sings_each_edit_once(stand, monkeypatch):
    """Issue #13. A run walks its list from the first edit, and under plain least-recently-used a
    list whose takes outgrew the memory dropped its first edit to make room for its last; the next
    run sang the first again and dropped the second for it, and so on down the list, so every run
    sang every edit. Here there is room for three takes, and the list grows to five edits of two."""
    monkeypatch.setattr(edit_track, "RESULTS", edit_track.Results(budget=3 * TAKE_BYTES))
    for count in range(1, 6):
        stand.calls.clear()
        run(stand, retakes_on(count))
        assert len(stand.calls) == 1, "only the edit added is sung"
        stand.calls.clear()
        run(stand, retakes_on(count))
        assert stand.calls == [], "a run of the same list sings nothing"


def test_past_the_memory_the_edits_before_the_last_keep_only_the_take_kept(stand, monkeypatch):
    """The other takes of an earlier edit are there to compare, and comparing is over once another
    edit stands on it; the last edit keeps every take for the window. Undone back to the first
    edit, its other take is offered as a seed to sing again, as a take is after a restart."""
    monkeypatch.setattr(edit_track, "RESULTS", edit_track.Results(budget=3 * TAKE_BYTES))
    first = payload(run(stand, retakes_on(1)))
    drawn = payload(run(stand, retakes_on(2)))
    assert [take["sung"] for take in drawn["takes"]] == [True, True]
    assert edit_track.RESULTS.size() == 3 * TAKE_BYTES
    stand.calls.clear()
    back = payload(run(stand, retakes_on(1)))
    assert stand.calls == []
    assert back["chosen"] == first["chosen"] == 1
    assert [take["sung"] for take in back["takes"]] == [False, True]
    run(stand, retakes_on(1, take=0))
    assert stand.calls == [("retake", 192, 414, (40,))]


def test_a_run_on_another_song_does_not_set_a_list_singing_itself_again():
    """Another song's takes may push a list's first edit out. The run that sings it again must
    drop the other song's takes to make room, not the edits it is about to walk to."""
    held = edit_track.Results(budget=160)
    for name in ("a1", "a2", "a3"):
        held.put(name, _entry(10), keep=held.walked("a"))
    held.settle("a", ["a1", "a2", "a3"])
    for name in ("b1", "b2"):
        held.put(name, _entry(10), keep=held.walked("b"))
    held.settle("b", ["b1", "b2"])
    assert held.get("a1") is None and held.get("a2") is not None
    held.put("a1", _entry(10), keep=held.walked("a"))
    assert held.get("a2") is not None and held.get("a3") is not None
    assert held.get("b1") is None and held.size() == 160


def test_a_words_edit_sings_the_stretch_selected_with_the_new_words(stand):
    """The cheap path: the bars say where, so nothing is heard and no other model is loaded."""
    answer = run(stand, '[{"op": "words", "bars": [4, 8], "lines": [3, 4], '
                        '"text": "one two four", "seed": 40}]')
    assert stand.calls == [("retake", 192, 414, (40, 41))]
    assert stand.words and "one two four" in stand.words[0], (
        "the new words have to reach the model, or the old ones are sung again")
    assert "one two three" not in stand.words[0]
    assert "one two four" in payload(answer)["lyrics"]


def test_a_words_edit_with_no_bars_is_placed_by_hearing_the_song(stand, monkeypatch):
    """The precise path: the aligner is given the words and says where that line is sung.

    It is given a section at a time, each on the stretch that section is
    sung in. The line chosen is the song's first, so it opens where it does,
    which is where the verse is heard from, and closes a step before the
    next line's first word, 1.44 s on: 80 ms before each, frames 82 and 118.
    """
    from yue2_comfy import download
    from yue2_comfy.asr import runtime as asr_runtime

    asked = []

    def word_times(folder, reader, device, waveform, rate, text, key, progress=None,
                   cancelled=None, low_vram=False):
        asked.append((folder, reader, text, round(waveform.shape[-1] / rate, 2)))
        found, at = [], 0.0
        for word in text.split():
            found.append((word, at, at + 0.4))
            at += 0.48
        return found

    monkeypatch.setattr(download, "ensure_aligner", lambda settings, progress=None: "aligner")
    monkeypatch.setattr(download, "aligner_tokenizer",
                        lambda folder, settings, progress=None: "tokenizer")
    monkeypatch.setattr(asr_runtime, "word_times", word_times)
    run(stand, '[{"op": "words", "lines": [3, 4], "text": "one two four", "seed": 40}]')
    assert asked[:2] == [
        ("aligner", "tokenizer", "one two three\nfour five six", 4.32),
        ("aligner", "tokenizer", "seven eight", 13.64)], (
        "the aligner is given the words as they are sung, tags and blank lines left out, a "
        "section at a time on the seconds it is sung in")
    assert stand.calls == [("retake", 82, 118, (40, 41))]
    assert "one two four" in stand.words[0]


def test_a_words_edit_says_what_it_needs_when_the_aligner_is_not_there(stand, monkeypatch):
    """Downloading off and no weights on the machine: the refusal names the file and the way round it."""
    from yue2_comfy import download

    def missing(settings, progress=None):
        raise FileNotFoundError("Qwen3-ForcedAligner-0.6B is not on this machine yet.")

    monkeypatch.setattr(download, "ensure_aligner", missing)
    with pytest.raises(ValueError, match="ForcedAligner"):
        run(stand, '[{"op": "words", "lines": [3, 4], "text": "one two four"}]')


def test_word_times_wait_beside_their_song_until_the_words_change(tmp_path, monkeypatch):
    """Hearing a song is a download and a pass over all of it; the second line rewritten pays neither."""
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    key = "{:064x}".format(5)
    edit_track._times_keep(key, "one two", [("one", 0.0, 0.4), ("two", 0.5, 0.9)])
    assert edit_track._times_read(key, "one two") == [("one", 0.0, 0.4), ("two", 0.5, 0.9)]
    assert edit_track._times_read(key, "one three") is None, (
        "times measured on other words would place every line wrong"
    )
    assert edit_track._times_read("not a key", "one two") is None
    assert edit_track._words_file("not a key") is None
    (tmp_path / (key + edit_track.WORDS_SUFFIX)).write_text("half a file", encoding="utf-8")
    assert edit_track._times_read(key, "one two") is None


def hearing(monkeypatch, said, seen=None):
    """Qwen3-ASR stood in: each clip is heard saying what ``said`` holds for its name.

    The names are the node's own, ``(takes' name, "was")`` for the stretch as
    the song sang it and ``(takes' name, seed)`` for a take, so ``said`` is
    keyed by "was" and by seed. A clip is the fill of the sound it was cut from
    and its two ends, which is enough to see what was cut from what.
    """
    from yue2_comfy.asr import runtime as asr_runtime

    monkeypatch.setattr(edit_track, "_speech_folder", lambda settings, progress, notices: "speech")
    monkeypatch.setattr(edit_track, "_ears_at_hand", lambda settings: "speech")
    monkeypatch.setattr(edit_track, "_clip", lambda waveform, rate, start, stop: (
        round(float(waveform.flatten()[0]), 2), round(start, 2), round(stop, 2)))

    def hear(folder, device, clips, language="", cancelled=None, progress=None, low_vram=False):
        answers = []
        for name, clip in clips:
            if seen is not None:
                seen.append((name[1], clip))
            answers.append({"language": "English", "text": said[name[1]]})
        return answers

    monkeypatch.setattr(asr_runtime, "hear", hear)


HEARD_FROM = (("one two three four five six", 3.36), ("seven eight", 7.36))
"""The words of each sung section of ``LYRICS`` and the second it is heard from.

``lines.SECTION_EARLY`` before the section's first sung note, on the frame
after: the verse's pickup is at 3.75 s and the chorus's at 7.75, so frames 84
and 184."""

LINES = [[3, 3.36, 4.76], [4, 4.86, 6.26], [7, 7.36, 8.26]]
"""When ``lined`` says each line of ``LYRICS`` is sung, as the window is handed it."""


def heard_at(first=0.0, gap=0.5):
    """``[(word, start)]`` for every word of ``LYRICS`` as ``lined(first=first, gap=gap)`` times them."""
    return [(word, opens + first + gap * index) for said, opens in HEARD_FROM
            for index, word in enumerate(said.split())]


def lined(monkeypatch, calls=None, first=0.0, gap=0.5):
    """The aligner stood in and at hand: the first word ``first`` seconds into what it is given, each next ``gap`` after the last.

    What it is given is a section of the song, so every section's words
    start over from where that section is heard.
    """
    from yue2_comfy.asr import runtime as asr_runtime

    monkeypatch.setattr(edit_track, "_aligner_at_hand",
                        lambda settings: ("aligner", "tokenizer", "cuda", bool(settings.get("low_vram"))))

    def word_times(folder, reader, device, waveform, rate, text, key, progress=None,
                   cancelled=None, low_vram=False):
        if calls is not None:
            calls.append((key, text, round(float(waveform.flatten()[0]), 2)))
        found, at = [], first
        for word in text.split():
            found.append((word, round(at, 3), round(at + 0.4, 3)))
            at += gap
        return found

    monkeypatch.setattr(asr_runtime, "word_times", word_times)


def said_by(monkeypatch):
    said = []
    monkeypatch.setattr(edit_track, "announce",
                        lambda node, findings, kind="notice": said.extend(findings))
    return said


def test_new_words_keep_the_take_heard_singing_them_over_one_that_joins_better(stand,
                                                                             monkeypatch):
    """The join is the model's opinion of how the old song goes on; the words are the edit.

    Take 2 joins better and sings the old line; take 1 sings the new one.
    """
    hearing(monkeypatch, {"was": "one two three", 40: "one two four", 41: "one two three"})
    drawn = payload(run(stand, WORDS))
    assert drawn["chosen"] == 0
    assert [take["heard"] for take in drawn["takes"]] == [[3, 3], [2, 3]]
    assert [take["said"] for take in drawn["takes"]] == ["one two four", "one two three"]
    assert [take["mumbled"] for take in drawn["takes"]] == [False, True]
    assert json.loads(drawn["edits"])[0]["take"] == 0, "the list handed back keeps it too"
    assert drawn["before"]["said"] == "one two three", (
        "the stretch as the song sang it is heard first, for the language")


def test_what_is_heard_is_the_stretch_each_take_sang_and_a_little_around_it(stand, monkeypatch):
    seen = []
    hearing(monkeypatch, {"was": "x", 40: "one two four", 41: "one two four"}, seen)
    run(stand, WORDS)
    start = 192 * FRAME_SECONDS - edit_track.HEARD_AROUND
    stop = 414 * FRAME_SECONDS + edit_track.HEARD_AROUND
    assert seen == [("was", (0.0, round(start, 2), round(stop, 2))),
                    (40, (0.1, round(start, 2), round(stop, 2))),
                    (41, (0.2, round(start, 2), round(stop, 2)))]


def test_takes_heard_alike_are_told_apart_by_the_join(stand, monkeypatch):
    hearing(monkeypatch, {"was": "x", 40: "one two four", 41: "one two four"})
    assert payload(run(stand, WORDS))["chosen"] == 1


def test_the_singing_model_is_let_go_before_the_takes_are_heard(stand, monkeypatch):
    """A card that holds the song model or the speech model need not hold both at once, and an
    edit after the one heard loads the song model again."""
    from yue2_comfy.asr import runtime as asr_runtime

    events = []

    class watched(_no_models):
        def __enter__(self):
            events.append("sing")
            return super().__enter__()

        def __exit__(self, *exc):
            events.append("let go")
            return False

    monkeypatch.setattr("yue2_comfy.staged.session", watched)
    hearing(monkeypatch, {"was": "x", 40: "one two four", 41: "one two"})
    heard = asr_runtime.hear
    monkeypatch.setattr(asr_runtime, "hear",
                        lambda *args, **kwargs: events.append("hear") or heard(*args, **kwargs))
    run(stand, WORDS[:-1] + ', {"op": "retake", "bars": [2, 4], "seed": 9}]')
    assert events == ["sing", "let go", "hear", "sing", "let go"]


def test_a_take_sung_later_is_the_only_one_heard(stand, monkeypatch):
    """Asking for one more take hears that one; the rest were heard when they were sung."""
    seen = []
    hearing(monkeypatch, {"was": "x", 40: "one two four", 41: "one two", 42: "one two four"},
            seen)
    run(stand, WORDS)
    seen.clear()
    later = payload(run(stand, WORDS.replace('"seed": 40', '"seed": 40, "takes": 3, "take": 2')))
    assert [name for name, _clip in seen] == ["was", 42]
    assert [take["heard"] for take in later["takes"]] == [[3, 3], [2, 3], [3, 3]]


def test_without_the_speech_model_the_join_picks_and_the_node_says_why(stand, monkeypatch):
    from yue2_comfy import download

    def missing(settings, progress=None):
        raise FileNotFoundError("Qwen3-ASR-1.7B is not on this machine yet.")

    monkeypatch.setattr(edit_track, "_speech_folder", SPEECH_FOLDER)
    monkeypatch.setattr(download, "ensure_asr", missing)
    said = said_by(monkeypatch)
    drawn = payload(run(stand, WORDS))
    assert drawn["chosen"] == 1
    assert all(take["heard"] is None for take in drawn["takes"])
    assert said.count(("notice", edit_track.NO_EARS)) == 1


def test_a_take_that_cannot_be_heard_is_left_to_the_join(stand, monkeypatch):
    from yue2_comfy.asr import runtime as asr_runtime

    hearing(monkeypatch, {})

    def deaf(*args, **kwargs):
        raise RuntimeError("the card said no")

    monkeypatch.setattr(asr_runtime, "hear", deaf)
    said = said_by(monkeypatch)
    drawn = payload(run(stand, WORDS))
    assert drawn["chosen"] == 1
    assert ("notice", edit_track.DEAF.format("the card said no")) in said


def test_a_take_kept_that_is_heard_singing_few_of_its_words_is_said_to(stand, monkeypatch):
    hearing(monkeypatch, {"was": "x", 40: "la la", 41: "one"})
    said = said_by(monkeypatch)
    drawn = payload(run(stand, WORDS))
    assert drawn["chosen"] == 1
    assert ("warn", edit_track.MUMBLED_SAID.format(1, 3)) in said


RETAKE = '[{"op": "retake", "bars": [2, 4], "seed": 9}]'
"""A retake of the verse, bars 3 and 4, from the pickup at the end of bar 2."""

SUNG = "one two three four five six seven eight"
"""Every word of ``LYRICS``, in order, as the aligner is given them."""


def sung_there(at, first=0.6, gap=0.5):
    """The words of ``SUNG`` whose middle lies in ``at``, timed as ``lined(first=0.6)`` times them."""
    return [word for word, start in heard_at(first, gap) if at[0] <= start + 0.2 < at[1]]


def test_a_retake_keeps_the_take_heard_singing_the_words_there(stand, monkeypatch):
    """Take 2 joins better; take 1 is heard singing the verse. The words are what is listened to."""
    lined(monkeypatch, first=0.6)
    hearing(monkeypatch, {"was": SUNG, 9: SUNG, 10: "one"})
    drawn = payload(run(stand, RETAKE))
    there = sung_there(drawn["at"])
    assert 2 <= len(there) < 8, "the words of the stretch, not of the song"
    assert drawn["asked"] == " ".join(there)
    assert drawn["chosen"] == 0
    count = len(there)
    assert [take["heard"] for take in drawn["takes"]] == [[count, count],
                                                          [int("one" in there), count]]
    assert [take["mumbled"] for take in drawn["takes"]] == [False, True]
    assert drawn["before"]["heard"] == [count, count], "the song as it was, heard the same way"
    assert drawn["before"]["said"] == SUNG
    assert drawn["hears"] is True


def test_a_retake_is_heard_after_the_singing_model_is_let_go(stand, monkeypatch):
    from yue2_comfy.asr import runtime as asr_runtime

    events = []

    class watched(_no_models):
        def __enter__(self):
            events.append("sing")
            return super().__enter__()

        def __exit__(self, *exc):
            events.append("let go")
            return False

    monkeypatch.setattr("yue2_comfy.staged.session", watched)
    lined(monkeypatch, first=0.6)
    hearing(monkeypatch, {"was": SUNG, 9: SUNG, 10: SUNG})
    heard = asr_runtime.hear
    monkeypatch.setattr(asr_runtime, "hear",
                        lambda *args, **kwargs: events.append("hear") or heard(*args, **kwargs))
    run(stand, RETAKE)
    assert events == ["sing", "let go", "hear"]


def test_a_retake_is_left_to_the_join_when_its_words_cannot_be_placed(stand, monkeypatch):
    """Without the aligner, and nothing timed beside the song, nobody knows which words are there."""
    seen = []
    hearing(monkeypatch, {}, seen)
    drawn = payload(run(stand, RETAKE))
    assert seen == [] and drawn["chosen"] == 1
    assert all(take["heard"] is None for take in drawn["takes"])
    assert drawn["asked"] is None and drawn["hears"] is False


def test_a_retake_is_heard_by_the_word_times_kept_beside_the_song(stand, monkeypatch, tmp_path):
    """The aligner timed this song in an earlier session; its times are enough without it."""
    from yue2_comfy.inpaint import lines

    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    name = songs.remember(stand.wave, RATE, stand.song)
    text = lines.heard_text(LYRICS)
    edit_track._times_keep(name, text, [(word, start, start + 0.4) for word, start in heard_at(0.6)],
                           sections=True)
    seen = []
    hearing(monkeypatch, {"was": SUNG, 9: SUNG, 10: "one"}, seen)
    drawn = payload(run(stand, RETAKE))
    assert [clip for clip, _cut in seen] == ["was", 9, 10]
    assert drawn["chosen"] == 0 and drawn["asked"] == " ".join(sung_there(drawn["at"]))


def test_a_retake_without_the_speech_model_is_left_to_the_join_without_a_word(stand,
                                                                             monkeypatch):
    """Hearing a retake is a better pick, not the edit: nothing is fetched and nothing is said."""
    from yue2_comfy import download

    lined(monkeypatch, first=0.6)
    fetched = []
    monkeypatch.setattr(download, "ensure_asr",
                        lambda settings, progress=None: fetched.append(1) or "speech")
    monkeypatch.setattr(edit_track, "_speech_folder", SPEECH_FOLDER)
    said = said_by(monkeypatch)
    drawn = payload(run(stand, RETAKE))
    assert drawn["chosen"] == 1 and fetched == [] and said == []
    assert drawn["hears"] is False


def test_a_retake_where_nothing_is_sung_is_not_heard(stand, monkeypatch):
    seen = []
    lined(monkeypatch, first=0.6)
    hearing(monkeypatch, {}, seen)
    drawn = payload(run(stand, '[{"op": "retake", "bars": [8, 10], "seed": 9}]'))
    assert seen == [] and drawn["chosen"] == 1 and drawn["asked"] is None


def test_a_retake_heard_as_well_as_the_song_it_replaces_is_not_mumbling(stand, monkeypatch):
    """The recogniser misses the same words in the song and in its takes."""
    lined(monkeypatch, first=0.6)
    hearing(monkeypatch, {"was": "one", 9: "one", 10: "one"})
    said = said_by(monkeypatch)
    drawn = payload(run(stand, RETAKE))
    assert "one" in sung_there(drawn["at"]) and drawn["before"]["heard"][0] == 1
    assert [take["mumbled"] for take in drawn["takes"]] == [False, False]
    assert [text for kind, text in said if kind == "warn"] == []


def test_a_retake_kept_that_is_heard_singing_less_than_the_song_did_is_said_to(stand,
                                                                               monkeypatch):
    lined(monkeypatch, first=0.6)
    hearing(monkeypatch, {"was": SUNG, 9: "one", 10: "one"})
    said = said_by(monkeypatch)
    drawn = payload(run(stand, RETAKE))
    count = len(sung_there(drawn["at"]))
    assert ("warn", edit_track.MUMBLED_AGAIN.format(1, count, count)) in said


def test_the_models_that_listened_are_let_go_unless_the_run_keeps_them(stand, monkeypatch):
    from yue2_comfy.asr import runtime as asr_runtime

    gone = []
    monkeypatch.setattr(asr_runtime, "unload", lambda: gone.append("speech"))
    monkeypatch.setattr(asr_runtime, "unload_aligner", lambda: gone.append("aligner"))
    run(stand, '[{"op": "retake", "bars": [2, 6], "seed": 9}]')
    assert gone == [], "a run that listened to nothing lets go of nothing another node keeps"
    hearing(monkeypatch, {"was": "x", 40: "one two four", 41: "one two", 50: "one two four",
                          51: "one"})
    lined(monkeypatch)
    run(stand, WORDS)
    assert sorted(gone) == ["aligner", "speech"]
    gone.clear()
    edit_track.YuE2EditTrack().edit(stand.audio, takes=2,
                                    edits=WORDS.replace('"seed": 40', '"seed": 50'),
                                    options={"keep_model_loaded": True})
    assert gone == []


def test_a_small_card_listens_small_and_clears_the_card_before_the_song_model_sings(stand, monkeypatch):
    """'low_vram' reaches both listeners, and both leave the card before the song model is loaded.

    The song model makes its room from what is free and sends nothing else
    away, while the aligner is loaded first in a run, to light the words: on a
    small card it would sit beside the song model as it sings. On a card with
    room nothing is sent away, and a kept listener waits for the next run.
    """
    from yue2_comfy.asr import runtime as asr_runtime

    events = []
    monkeypatch.setattr(asr_runtime, "unload", lambda: events.append("speech away"))
    monkeypatch.setattr(asr_runtime, "unload_aligner", lambda: events.append("aligner away"))
    hearing(monkeypatch, {"was": "x", 40: "one two four", 41: "one two", 50: "one two four", 51: "one"})
    lined(monkeypatch)
    heard, timed = asr_runtime.hear, asr_runtime.word_times
    monkeypatch.setattr(asr_runtime, "hear", lambda *args, **kwargs: events.append(
        ("hear", kwargs["low_vram"])) or heard(*args, **kwargs))
    monkeypatch.setattr(asr_runtime, "word_times", lambda *args, **kwargs: events.append(
        ("times", kwargs["low_vram"])) or timed(*args, **kwargs))

    def session(settings, unique_id, progress, loras=None):
        events.append("sing")
        return _no_models()

    monkeypatch.setattr("yue2_comfy.staged.session", session)
    edit_track.YuE2EditTrack().edit(stand.audio, takes=2, edits=WORDS,
                                    options={"low_vram": True, "keep_model_loaded": True})
    sing = events.index("sing")
    assert events[sing - 2:sing] == ["speech away", "aligner away"]
    assert ("times", True) in events and ("hear", True) in events
    assert ("times", False) not in events and ("hear", False) not in events
    events.clear()
    edit_track.YuE2EditTrack().edit(stand.audio, takes=2, edits=WORDS.replace('"seed": 40', '"seed": 50'),
                                    options={"keep_model_loaded": True})
    assert "speech away" not in events and "aligner away" not in events
    assert ("hear", False) in events


def test_the_window_is_told_when_each_line_is_sung(stand, monkeypatch):
    """Lines 4, 5 and 8 of the words are the sung ones; tags and blank lines have no time."""
    lined(monkeypatch)
    drawn = payload(run(stand))
    assert drawn["lines"] == LINES


def test_without_the_aligner_at_hand_the_window_places_lines_by_the_bars(stand):
    assert payload(run(stand))["lines"] is None


def test_every_take_carries_the_lines_of_the_song_it_makes(stand, monkeypatch):
    """Switching takes swaps the track in the window, and its lines with it."""
    calls = []
    lined(monkeypatch, calls)
    drawn = payload(run(stand, '[{"op": "retake", "bars": [2, 6], "seed": 9}]'))
    kept = drawn["chosen"]
    assert drawn["takes"][kept]["lines"] == drawn["lines"]
    assert all(take["lines"] for take in drawn["takes"])
    assert drawn["before"]["lines"] == LINES
    assert sorted(fill for _key, _text, fill in calls) == [0.0, 0.0, 0.1, 0.1, 0.2, 0.2], (
        "the song, the other take and the song as it was, each heard once, a section at a time")
    assert len({key for key, _text, _fill in calls}) == 6


def test_the_song_as_it_was_is_numbered_as_the_words_the_window_shows(stand, monkeypatch):
    """One line became two, so every line after them moved down by one."""
    lined(monkeypatch)
    drawn = payload(run(stand, WORDS.replace("one two four", "one two\\nfour")))
    assert drawn["lines"] == [[3, 3.36, 4.26], [4, 4.36, 4.76], [5, 4.86, 6.26], [8, 7.36, 8.26]]
    assert drawn["before"]["lines"] == [[3, 3.36, 4.76], [4, 3.36, 4.76], [5, 4.86, 6.26],
                                        [8, 7.36, 8.26]]


def test_the_lines_of_a_song_wait_beside_it_for_the_next_run(stand, monkeypatch, tmp_path):
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    name = songs.remember(stand.wave, RATE, stand.song)
    calls = []
    lined(monkeypatch, calls)
    run(stand)
    assert len(calls) == 2 and (tmp_path / (name + edit_track.WORDS_SUFFIX)).is_file()
    run(stand)
    assert len(calls) == 2, "read beside the song, not heard again"


def test_a_song_whose_lines_cannot_be_found_is_still_drawn(stand, monkeypatch):
    from yue2_comfy.asr import runtime as asr_runtime

    lined(monkeypatch)

    def broken(*args, **kwargs):
        raise RuntimeError("the card said no")

    monkeypatch.setattr(asr_runtime, "word_times", broken)
    drawn = payload(run(stand))
    assert drawn["lines"] is None and drawn["peaks"]


def test_a_switched_off_audio_is_not_even_worked_out():
    """ComfyUI hands a joined input that is not worked out yet as None and leaves an unjoined one
    out; only a joined one, switched on, is asked for, or the node above sings for nothing."""
    node = edit_track.YuE2EditTrack()
    assert edit_track.YuE2EditTrack.INPUT_TYPES()["optional"]["audio"][1]["lazy"] is True
    assert node.check_lazy_status(audio=None, use_audio=True, takes=2) == ["audio"]
    assert node.check_lazy_status(audio=None, use_audio=False, takes=2) == []
    assert node.check_lazy_status(use_audio=True, takes=2) == [], (
        "asking for an input with no link is an error in ComfyUI")
    assert node.check_lazy_status(audio={"waveform": 1}, use_audio=True) == []


def test_with_its_audio_switched_off_the_node_edits_the_song_chosen_by_key(stand, monkeypatch,
                                                                           torch):
    seen = []
    a_decoder(monkeypatch, torch, seen)
    answer = edit_track.YuE2EditTrack().edit(None, takes=2, song_key=stand.name, use_audio=False)
    assert payload(answer)["was"] == stand.name and seen, "the song came back from its latents"
    other = a_song(lyrics="[Verse]\nsomething else entirely")
    songs.remember(a_wave(torch, fill=0.3), RATE, other)
    kept = edit_track.YuE2EditTrack().edit(
        {"waveform": a_wave(torch, fill=0.3), "sample_rate": RATE}, takes=2,
        song_key=stand.name, use_audio=False)
    assert payload(kept)["was"] == stand.name, "a joined song, switched off, is left alone"
    with pytest.raises(ValueError, match="square beside it ticked"):
        edit_track.YuE2EditTrack().edit(None, takes=2, song_key="", use_audio=False)


def test_a_list_made_on_another_song_is_refused_and_says_which_songs(stand, monkeypatch, torch):
    """The same report: a list of bars laid on another song cuts and sings whatever is there.

    The window names the song it made the list on; the node refuses the list on
    any other song, naming both by length and style, since a key means nothing
    to a person. A bare list, as every list written before, is read as before.
    """
    a_decoder(monkeypatch, torch)
    other = a_song(lyrics="[Verse]\nthe one picked days before",
                   style="folk rock with funk influences, " * 4)
    elsewhere = songs.remember(a_wave(torch, fill=0.3), RATE, other)
    cut = '[{"op": "cut", "bars": [2, 4]}]'
    named = '{"song": "%s", "edits": %s}'
    with pytest.raises(ValueError) as refused:
        run(stand, named % (elsewhere, cut))
    said = str(refused.value)
    assert "made on another song" in said and "'Reset track'" in said
    made = r", made \d{4}-\d\d-\d\d \d\d:\d\d, "
    assert re.search(r"another song \(0:21" + made + r"'folk rock with funk influences, folk rock "
                     r"with funk influenc\.\.\.'\) than the one it has now \(0:21" + made
                     + r"'a style'\)", said), said
    assert stand.calls == []
    assert payload(run(stand, named % (stand.name, cut)))["edits"]
    assert payload(run(stand, cut))["edits"], "a list that names no song is read as before"
    assert payload(run(stand, named % (elsewhere, "[]")))["edits"] == "[]", "no edits, no refusal"
    with pytest.raises(ValueError, match="does not hold a JSON list"):
        run(stand, named % ("not a key", cut))
    gone = "cd" * 32
    with pytest.raises(ValueError, match="another song \\(one no longer remembered\\)"):
        run(stand, named % (gone, cut))


def test_a_run_on_the_joined_song_says_so_and_names_the_song_for_song_key(stand, monkeypatch,
                                                                          torch):
    """The window writes 'was' into 'song_key' after such a run, so switching 'audio' off stays on it.

    A user's report of 2026-10-18: a song from 'Load Audio' edited, the square
    beside 'audio' unticked, and the next run opened a song picked days before
    in 'Saved songs...', with the new edits laid on it -- the field had been
    written by the picker alone.
    """
    a_decoder(monkeypatch, torch)
    picked = a_song(lyrics="[Verse]\nthe one picked days before")
    songs.remember(a_wave(torch, fill=0.3), RATE, picked)
    stale = songs.key(a_wave(torch, fill=0.3), RATE)
    joined = payload(edit_track.YuE2EditTrack().edit(stand.audio, takes=2, song_key=stale))
    assert (joined["joined"], joined["was"]) == (True, stand.name)
    off = payload(edit_track.YuE2EditTrack().edit(stand.audio, takes=2, song_key=joined["was"],
                                                  use_audio=False))
    assert (off["joined"], off["was"]) == (False, stand.name), "switched off, the same song"
    picked_off = payload(edit_track.YuE2EditTrack().edit(stand.audio, takes=2, song_key=stale,
                                                         use_audio=False))
    assert (picked_off["joined"], picked_off["was"]) == (False, stale), (
        "the field the old window left behind opens the other song, which is what the user saw")


def test_every_row_of_the_takes_list_has_the_same_keys(stand):
    """The window reads a take, a take not sung yet and the song as it was the same way."""
    drawn = payload(run(stand, '[{"op": "retake", "bars": [2, 6], "seed": 9, "takes": 2, '
                               '"take": 1}]'))
    rows = drawn["takes"] + [drawn["before"]]
    assert [take["sung"] for take in drawn["takes"]] == [False, True]
    assert all(set(row) == set(rows[0]) for row in rows)
    assert {"lines", "heard", "said", "mumbled"} <= set(rows[0])


def notes_of(where, bars=None, seed=9):
    """A change of notes of ``RAP`` as the window writes it: the sung note at each tick of ``where`` moved to the pitch it maps to."""
    sheet = notation.read(RAP)
    for note in sheet["notes"]["Vocal"]:
        note["pitch"] = where.get(note["start"], note["pitch"])
    item = {"op": "notes", "score": notation.write(RAP, sheet)["abc"], "seed": seed}
    if bars is not None:
        item["bars"] = list(bars)
    return item


def test_a_change_of_notes_sings_only_the_bars_it_changed_under_the_new_score(stand):
    """One note of bar 6 moved: bar 6 is sung again, as a retake of it would be, with the new score in the prompt."""
    item = notes_of({86: 64})
    drawn = payload(run(stand, json.dumps([item])))
    run(stand, '[{"op": "retake", "bars": [5, 6], "seed": 9}]')
    assert stand.calls[0][1:3] == stand.calls[1][1:3]
    assert stand.scores == [item["score"] + "\n", None]
    assert drawn["kind"] == "notes" and drawn["score"] == item["score"] + "\n"
    assert json.loads(drawn["edits"]) == [dict(item, takes=2, take=1)]


def test_the_window_is_handed_the_score_it_opens_the_notes_on(stand):
    assert payload(run(stand))["score"] == RAP


def test_a_change_of_notes_is_heard_like_a_retake_of_its_bars(stand, monkeypatch):
    """The words do not change, so the takes are heard against the words the song sings there."""
    lined(monkeypatch, first=0.6)
    hearing(monkeypatch, {"was": SUNG, 9: SUNG, 10: "one"})
    drawn = payload(run(stand, json.dumps([notes_of({36: 67})])))
    there = sung_there(drawn["at"])
    assert there and drawn["asked"] == " ".join(there)
    assert drawn["chosen"] == 0
    assert drawn["before"]["heard"] == [len(there), len(there)]


def test_a_change_of_notes_that_moves_the_tempo_is_refused_before_anything_is_sung(stand):
    item = notes_of({86: 64})
    item["score"] = item["score"].replace("Q:1/4=120", "Q:1/4=100")
    with pytest.raises(Exception, match="another tempo"):
        run(stand, json.dumps([item]))
    assert stand.calls == []


def test_takes_are_heard_in_the_language_their_words_are_written_in(stand, monkeypatch):
    """Left to name it, Qwen3-ASR named Russian singing English and wrote it as a translation."""
    from yue2_comfy.asr import runtime as asr_runtime

    snow = "\u0421\u043d\u0435\u0433 \u043b\u043e\u0436\u0438\u0442\u0441\u044f"
    languages = []
    hearing(monkeypatch, {"was": snow, 40: snow, 41: snow})
    heard = asr_runtime.hear

    def told(folder, device, clips, language="", cancelled=None, progress=None, low_vram=False):
        languages.append(language)
        return heard(folder, device, clips, language, cancelled, progress, low_vram)

    monkeypatch.setattr(asr_runtime, "hear", told)
    item = json.loads(WORDS)[0]
    run(stand, json.dumps([dict(item, text=snow)]))
    run(stand, WORDS)
    assert languages == ["Russian", ""]


def going_on(text=BRIDGE, **more):
    """A song going on as the window writes it."""
    return json.dumps([dict({"op": "extend", "text": text, "seed": 9}, **more)])


def goes_on_from():
    """The frame the stand's song goes on from: the opening of bar 9, after the chorus's last note."""
    sheet = notation.read(RAP)
    return grid.retake_frames(sheet, a_clock(), grid.seams(sheet), 8, 10, FRAMES)[0]


def test_a_song_goes_on_from_where_its_singing_ends_under_the_score_its_take_wrote(stand):
    drawn = payload(run(stand, going_on()))
    lyrics = LYRICS.replace("\n\n[Outro]", "") + "\n\n" + BRIDGE + "\n\n[Outro]"
    assert stand.calls == [("extend", goes_on_from(), (9, 10))]
    assert stand.scores == [notation.head(RAP, 8)] and stand.words == [lyrics]
    assert drawn["kind"] == "extend" and drawn["chosen"] == 1, "the take that ended the song itself"
    assert [take["ended"] for take in drawn["takes"]] == [False, True]
    assert drawn["score"] == notation.head(RAP, 8) + GOES_ON and drawn["lyrics"] == lyrics
    assert len(drawn["grid"]["bars"]) == 11, "eight bars kept, two written, and the end"
    assert drawn["before"]["index"] == -1
    assert json.loads(drawn["edits"]) == [{"op": "extend", "text": BRIDGE, "seed": 9, "takes": 2,
                                           "take": 1}]


def test_the_window_is_told_where_the_song_would_go_on_from(stand):
    assert payload(run(stand))["goes_on"] == {"bar": 8,
                                              "second": round(goes_on_from() * FRAME_SECONDS, 3)}


def test_a_new_ending_alone_is_sung_under_the_songs_own_words_and_not_heard(stand, monkeypatch):
    seen = []
    hearing(monkeypatch, {}, seen)
    drawn = payload(run(stand, going_on(text="")))
    assert stand.words == [LYRICS] and seen == []
    assert drawn["asked"] is None


def test_the_takes_of_a_song_going_on_are_heard_against_its_new_words(stand, monkeypatch):
    """A take stopped at its limit ends mid-note, which is heard however many words it sang."""
    hearing(monkeypatch, {"was": "", 9: "nine ten eleven", 10: "nine"})
    drawn = payload(run(stand, going_on()))
    assert drawn["asked"] == "nine ten eleven"
    assert [take["heard"] for take in drawn["takes"]] == [[3, 3], [1, 3]]
    assert drawn["chosen"] == 1


def test_a_kept_take_stopped_at_its_limit_is_said_to_stop_mid_note(stand, monkeypatch):
    said = said_by(monkeypatch)
    run(stand, going_on(take=0))
    assert any(level == "warn" and "stops mid-note" in text for level, text in said)
    said.clear()
    run(stand, going_on(take=1))
    assert not any("mid-note" in text for _level, text in said)


def test_the_bars_a_song_went_on_with_can_be_edited_like_any_other(stand):
    """The grid after it lays the take's own score on the song, so a retake of the new bars finds them."""
    first = json.loads(going_on(take=1))[0]
    run(stand, json.dumps([first, {"op": "retake", "bars": [8, 10], "seed": 3}]))
    state = track.opened(stand.song, a_clock())
    step = track.plan(state, track.read(going_on())[0])
    written = step.score + GOES_ON
    left = track.after(state, step, track.extension_frames(state, step, notation.read(written)),
                       written)
    wanted = grid.retake_frames(left.sheet, left.clock, grid.seams(left.sheet), 8, 10, left.frames)
    assert stand.calls[0][0] == "extend" and stand.calls[1][:3] == ("retake",) + wanted


TWO_LINES = "[Bridge]\nnine ten eleven\ntwelve thirteen"


def went_from():
    """The second the stand's take of ``TWO_LINES`` is heard from: where its new part opens, here."""
    state = track.opened(a_song(), a_clock())
    step = track.plan(state, track.read(going_on(TWO_LINES))[0])
    return track.sings_from(state, step, notation.read(step.score + GOES_ON))


def test_a_song_going_on_has_its_new_lines_timed_on_its_new_part_alone(stand, monkeypatch):
    """Given the whole song, the aligner lost a user's new outro among the old lines, six seconds early.

    The old lines keep the times the song before had, and only the new ones
    are timed, on the take's sound from where its new part is heard.
    """
    calls = []
    lined(monkeypatch, calls)
    drawn = payload(run(stand, going_on(TWO_LINES, take=1)))
    since = went_from()
    assert since == pytest.approx(goes_on_from() * FRAME_SECONDS)
    assert {(text, fill) for _key, text, fill in calls} == {
        ("one two three\nfour five six", 0.0), ("seven eight", 0.0),
        ("nine ten eleven\ntwelve thirteen", 0.1)}, "never the whole song's words at once"
    assert drawn["lines"] == LINES + [[10, round(since, 3), round(since + 1.4, 3)],
                                      [11, round(since + 1.5, 3), round(since + 2.4, 3)]]


def test_a_line_a_song_went_on_with_is_rewritten_where_its_new_part_sings_it(stand, monkeypatch):
    """The stretch of a new line comes from the new part's times: timed whole, it would open 12.5 s early here."""
    from yue2_comfy import download
    from yue2_comfy.inpaint import lines

    monkeypatch.setattr(download, "ensure_aligner", lambda settings, progress=None: "aligner")
    monkeypatch.setattr(download, "aligner_tokenizer",
                        lambda folder, settings, progress=None: "tokenizer")
    lined(monkeypatch)
    first = json.loads(going_on(TWO_LINES, take=1))[0]
    run(stand, json.dumps([first, {"op": "words", "lines": [11, 12], "text": "twelve fourteen",
                                   "seed": 3}]))
    opening = went_from() + 1.0 - lines.EARLY
    assert stand.calls[1][:2] == ("retake", int(round(opening / FRAME_SECONDS))), (
        "it opens on the last word of the line before, 'eleven', the new part's third word")


def test_times_of_a_song_that_went_on_are_read_only_when_they_say_so(tmp_path, monkeypatch):
    """A song timed whole before its new lines were known to be lost there is timed again."""
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    key = "{:064x}".format(6)
    times = [("one", 0.0, 0.4), ("two", 20.5, 20.9)]
    edit_track._times_keep(key, "one two", times)
    assert edit_track._times_read(key, "one two", 20.0) is None
    edit_track._times_keep(key, "one two", times, 20.0)
    assert edit_track._times_read(key, "one two", 20.0) == times
    assert edit_track._times_read(key, "one two", 21.0) is None
    assert edit_track._times_read(key, "one two") == times, "opened as a song of its own"


def test_a_song_with_no_score_is_offered_no_going_on_and_refuses_one(torch):
    wave = a_wave(torch, fill=0.3)
    songs.remember(wave, RATE, a_song(score_text="", settings={"cot": "off"}))
    audio = {"waveform": wave, "sample_rate": RATE}
    assert payload(edit_track.YuE2EditTrack().edit(audio, takes=2, edits=""))["goes_on"] is None
    with pytest.raises(Exception, match="no score for the model to go on writing"):
        edit_track.YuE2EditTrack().edit(audio, takes=2, edits=going_on())


MOVE = '[{"op": "move", "bars": [4, 8], "to": 2}]'
"""The chorus of the stand's song put before its verse."""


def moved_step():
    state = track.opened(a_song(), a_clock())
    return state, track.plan(state, track.read(MOVE)[0])


def test_a_section_moves_with_its_words_and_its_bars_and_the_bar_before_each_seam_is_sung(stand):
    drawn = payload(run(stand, MOVE))
    state, step = moved_step()
    assert step.sung and step.pulse
    assert stand.calls == [("move", step.pieces, step.stop - step.start, step.sung, step.pulse)]
    assert stand.words == [step.lyrics] and stand.scores == [step.score]
    assert drawn["kind"] == "move" and drawn["chosen"] == 0
    assert len(drawn["takes"]) == 1 and "before" not in drawn, "it comes out one way, like a cut"
    assert drawn["at"] == [round(step.start * FRAME_SECONDS, 3), round(step.stop * FRAME_SECONDS, 3)]
    made = sung_by_stand(step.sung)
    assert drawn["placed"] == [round(frame * FRAME_SECONDS, 3) for frame in track.placed(step, made)]
    assert drawn["seconds"] == pytest.approx(
        (FRAMES + SUNG_LONGER * len(made)) * FRAME_SECONDS), "the bars sung again came out longer"
    left = track.after(state, step, step.stop - step.start, None, made)
    assert drawn["grid"] == grid.layout(left.sheet, left.clock, left.frames)
    assert [section["name"] for section in drawn["grid"]["sections"]] == [
        "intro", "chorus", "verse", "outro"]
    assert (drawn["lyrics"], drawn["score"]) == (step.lyrics, step.score)
    assert json.loads(drawn["edits"]) == json.loads(MOVE)
    mark = songs.store().get(drawn["song"]).edit[0]
    assert (mark["op"], mark["at"], mark["bars"], mark["seed"]) == ("move", drawn["placed"], [4, 8], None)
    run(stand, MOVE)
    assert len(stand.calls) == 1, "the same move is not made twice"


def test_an_edit_after_a_move_is_made_on_the_song_the_move_left(stand):
    run(stand, json.dumps(json.loads(MOVE) + [{"op": "retake", "bars": [2, 6], "seed": 3}]))
    state, step = moved_step()
    left = track.after(state, step, step.stop - step.start, None, sung_by_stand(step.sung))
    wanted = grid.retake_frames(left.sheet, left.clock, grid.seams(left.sheet), 2, 6, left.frames)
    assert stand.calls[1][:3] == ("retake",) + wanted


def test_the_mark_of_an_earlier_edit_goes_where_its_sound_went(stand):
    retake = {"op": "retake", "bars": [2, 4], "seed": 3, "takes": 1}
    drawn = payload(run(stand, json.dumps([retake] + json.loads(MOVE))))
    state, step = moved_step()
    first = stand.calls[0][1]
    at = 0
    for start, stop in step.pieces:
        if start <= first < stop:
            break
        at += stop - start
    marks = songs.store().get(drawn["song"]).edit
    assert [mark["op"] for mark in marks] == ["retake", "move"]
    assert marks[0]["at"][0] == round(
        track.shifted(at + first - start, sung_by_stand(step.sung)) * FRAME_SECONDS, 3)


def test_a_moved_song_keeps_the_times_of_its_words_and_is_not_heard_again(stand, monkeypatch):
    """The words sung are where their sound went, past the bars sung again before the seams."""
    from yue2_comfy.inpaint import lines

    _state, step = moved_step()
    calls = []
    lined(monkeypatch, calls, first=0.6)
    drawn = payload(run(stand, MOVE))
    said = lines.heard_text(LYRICS)
    assert [text for _key, text, _fill in calls] == ["one two three\nfour five six", "seven eight"], (
        "the song as it was, once, a section at a time")
    old = [(word, round(start, 3), round(start + 0.4, 3)) for word, start in heard_at(0.6)]
    pieces = [(start * FRAME_SECONDS, stop * FRAME_SECONDS) for start, stop in step.pieces]
    times = lines.carried_along(old, said, pieces, lines.heard_text(step.lyrics))
    assert times is not None and times[0][0] == "seven"
    times = lines.stretched(times, [(start * FRAME_SECONDS, stop * FRAME_SECONDS, count * FRAME_SECONDS)
                                    for start, stop, count in sung_by_stand(step.sung)])
    assert drawn["lines"] == [[number, round(start, 3), round(stop, 3)]
                              for number, start, stop in lines.placed(step.lyrics, times)]


def test_a_move_of_bars_that_are_not_whole_sections_is_refused(stand):
    with pytest.raises(Exception, match="bar line 6 is inside the chorus"):
        run(stand, '[{"op": "move", "bars": [5, 8], "to": 2}]')
    assert stand.calls == []


def test_a_move_cuts_where_the_separated_voice_rests_and_hears_it_once(stand, monkeypatch):
    """The score says where a phrase starts; a singer who comes in ahead of it is where it really does."""
    monkeypatch.setattr(edit_track, "_voice_level", VOICE_LEVEL)
    monkeypatch.setattr(edit_track, "_VOICES", type(edit_track._VOICES)())
    state, step = moved_step()
    windows = grid.move_windows(state.sheet, state.clock, grid.seams(state.sheet), step.stretches)
    touched, default, top, _beat = grid._move_lines(state.sheet, state.clock,
                                                    grid.seams(state.sheet), step.stretches)
    unit = state.clock.rate * state.clock.tick
    quiet = int(default) + 2
    assert quiet <= top
    asked = []

    def voice_near(sound, rate, stretches, settings, progress):
        asked.append(stretches)
        found = []
        for start, stop in stretches:
            loud = [0.0] * int((stop - start) / edit_track.VOICE_HOP + 2)
            for bar in touched:
                cut = state.clock.at(bar) - quiet * unit
                for index in range(len(loud)):
                    if abs(start + index * edit_track.VOICE_HOP - cut) <= 0.05:
                        loud[index] = -100.0
            found.append((start, loud))
        return found

    monkeypatch.setattr(edit_track, "_voice_near", voice_near)
    run(stand, MOVE)
    level = edit_track._voice_level(None, RATE, windows, {},
                                    track.sound_name(songs.canonical(stand.name), []), None)
    wanted = grid.move_frames(state.sheet, state.clock, grid.seams(state.sheet), step.stretches,
                              FRAMES, level)
    assert asked == [windows]
    assert stand.calls[0][:3] == ("move", tuple(wanted), wanted[1][1] - wanted[1][0])
    assert wanted[1][0] == state.clock.moment(4, quiet) != step.pieces[1][0]
    run(stand, MOVE)
    assert asked == [windows], "the voice of a sound is not separated twice"


BREAK = '[{"op": "break", "to": 4, "length": 2, "seed": 9, "takes": 2}]'
"""Two bars of playing before the chorus, two takes."""


def break_step():
    state = track.opened(a_song(), a_clock())
    return state, track.plan(state, track.read(BREAK)[0])


def break_written(step):
    """The score the stand-in's take lays in, and how long its bars last at the song's tempo."""
    state = track.opened(a_song(), a_clock())
    written = notation.interluded(RAP, step.score + played(2), 4, 2, step.seam)
    return written, track.break_frames(state, step, notation.read(written))


def loud_in_break(inside):
    """The separator stood in: the singing around every break at -10 dB, in each break as ``inside`` says."""
    def voices(pieces, rate, settings, progress):
        found = []
        for (_sound, start, stop), level in zip(pieces, inside):
            first = max(0.0, start - edit_track.VOICE_CONTEXT)
            count = int((stop + edit_track.VOICE_CONTEXT - first) / edit_track.VOICE_HOP)
            opening, closing = start + edit_track.BREAK_AROUND, stop - edit_track.BREAK_AROUND
            found.append((first, [level if opening <= first + at * edit_track.VOICE_HOP < closing
                                  else -10.0 for at in range(count)]))
        return found
    return voices


def test_a_break_is_played_before_its_section_and_what_comes_after_moves_by_it(stand, monkeypatch):
    said = said_by(monkeypatch)
    drawn = payload(run(stand, BREAK))
    state, step = break_step()
    written, length = break_written(step)
    assert stand.calls == [("break", step.start, 2, (9, 10))]
    assert stand.words == [step.lyrics] and "six\n\n[Interlude]\n\n[Chorus]" in step.lyrics
    assert stand.scores == [step.score] and step.score.endswith("% interlude\n")
    assert drawn["kind"] == "break" and drawn["chosen"] == 1, "unheard, the join keeps the take"
    assert drawn["at"] == [round(step.start * FRAME_SECONDS, 3)] * 2
    assert drawn["placed"] == [round(step.start * FRAME_SECONDS, 3),
                               round((step.start + length + 1) * FRAME_SECONDS, 3)]
    assert drawn["seconds"] == pytest.approx((FRAMES + length + 1) * FRAME_SECONDS)
    left = track.after(state, step, length + 1, written)
    assert drawn["grid"] == grid.layout(left.sheet, left.clock, left.frames)
    assert [section["name"] for section in drawn["grid"]["sections"]] == [
        "intro", "verse", "interlude", "chorus", "outro"]
    assert (drawn["score"], drawn["lyrics"]) == (written, step.lyrics)
    assert [(take["voice"], take["voiced"]) for take in drawn["takes"]] == [(None, False)] * 2
    assert drawn["before"]["voice"] is None, "every row of the list has the same keys"
    assert ("notice", edit_track.BREAK_UNHEARD) in said
    mark = songs.store().get(drawn["song"]).edit[0]
    assert (mark["op"], mark["at"], mark["seed"]) == ("break", drawn["placed"], 10)
    run(stand, BREAK)
    assert len(stand.calls) == 1, "the same break is not played twice"


def test_a_break_keeps_the_take_with_no_voice_left_in_it_over_a_better_join(stand, monkeypatch):
    said = said_by(monkeypatch)
    monkeypatch.setattr(edit_track, "_voices_in", loud_in_break([-60.0, -12.0]))
    drawn = payload(run(stand, BREAK))
    assert drawn["chosen"] == 0
    assert [take["voiced"] for take in drawn["takes"]] == [False, True]
    assert [take["voice"] for take in drawn["takes"]] == [-50.0, -2.0], (
        "the voice in the break against the singing around it")
    assert not [finding for finding in said if finding[0] == "warn"]
    kept = run(stand, BREAK.replace('"takes": 2', '"takes": 2, "take": 1'))
    assert len(stand.calls) == 1, "keeping the other take sings nothing"
    assert payload(kept)["chosen"] == 1
    assert ("warn", edit_track.BREAK_KEPT_VOICED.format(-2.0, 1)) in said


def test_a_break_that_sings_in_every_take_says_so(stand, monkeypatch):
    said = said_by(monkeypatch)
    monkeypatch.setattr(edit_track, "_voices_in", loud_in_break([-8.0, -14.0]))
    drawn = payload(run(stand, BREAK))
    assert drawn["chosen"] == 1, "the quieter of two that sing"
    assert ("warn", edit_track.BREAK_VOICED.format(-4.0)) in said


def test_an_edit_after_a_break_is_made_on_the_song_the_break_left(stand):
    run(stand, json.dumps(json.loads(BREAK) + [{"op": "retake", "bars": [6, 8], "seed": 3}]))
    state, step = break_step()
    written, length = break_written(step)
    left = track.after(state, step, length + 1, written)
    wanted = grid.retake_frames(left.sheet, left.clock, grid.seams(left.sheet), 6, 8, left.frames)
    assert stand.calls[1][:3] == ("retake",) + wanted


def test_the_words_after_a_break_are_lit_later_by_it_and_not_heard_again(stand, monkeypatch):
    from yue2_comfy.inpaint import lines

    calls = []
    lined(monkeypatch, calls, first=0.6)
    drawn = payload(run(stand, BREAK))
    state, step = break_step()
    _written, length = break_written(step)
    said = lines.heard_text(LYRICS)
    assert [(text, fill) for _key, text, fill in calls] == [
        ("one two three\nfour five six", 0.0), ("seven eight", 0.0)], (
        "only the song as it was is heard, a section at a time, and no take is")
    old = [(word, round(start, 3), round(start + 0.4, 3)) for word, start in heard_at(0.6)]
    at = step.start * FRAME_SECONDS
    times = lines.carried_along(old, said, [(0.0, at), (at, FRAMES * FRAME_SECONDS)],
                                lines.heard_text(step.lyrics))
    times = lines.stretched(times, [(at, at, (length + 1) * FRAME_SECONDS)])
    assert drawn["lines"] == [[number, round(start, 3), round(stop, 3)]
                              for number, start, stop in lines.placed(step.lyrics, times)]
    assert drawn["lines"][-1][1] > at + (length + 1) * FRAME_SECONDS - 0.5, (
        "the chorus is lit after the break")


VERSE, CHORUS = "one two three\nfour five six", "seven eight"
"""The lines of the two sung sections of ``LYRICS``, as the aligner is given them."""

PLACE = ("aligner", "tokenizer", "cuda", False)

HOP = RATE * FRAME_SECONDS


def a_ramp(torch, frames=FRAMES):
    """A sound whose every sample is another number, so a stretch of it is known from any other."""
    return (torch.arange(frames * SAMPLES_A_FRAME, dtype=torch.float32) / 1000.0).repeat(2, 1)[None]


def the_windows():
    """Where the sections of the stand's song are heard, in frames."""
    from yue2_comfy.inpaint import lines

    state = track.opened(a_song(), a_clock())
    return edit_track._windows_of(state, lines.heard_text(LYRICS))


def test_the_sections_of_a_song_open_on_the_frame_after_and_never_before():
    """The earlier a stretch opens, the more often its first word is put on the playing before it."""
    from yue2_comfy.inpaint import lines

    assert the_windows() == [(84, 192, VERSE), (184, FRAMES, CHORUS)]
    assert 84 * FRAME_SECONDS >= 3.75 - lines.SECTION_EARLY > 83 * FRAME_SECONDS
    state = track.opened(a_song(), a_clock())
    assert edit_track._windows_of(state, "other words") is None, "times for other words are not these"
    assert edit_track._windows_of(None, lines.heard_text(LYRICS)) is None
    bare = track.opened(a_song(score_text="", settings={"cot": "off"}), None)
    assert edit_track._windows_of(bare, lines.heard_text(LYRICS)) is None, "no score, no sections"
    sheet = notation.read(RAP)
    by_the_mix = grid.Grid(offset=0.5, rate=1.0, tick=grid.tick_seconds(sheet),
                           starts=grid.starts_of(sheet), by_voice=False)
    assert edit_track._windows_of(track.opened(a_song(), by_the_mix),
                                  lines.heard_text(LYRICS)) is None, (
        "bars the voice did not place may be seconds off, and sections would be heard there")


def test_a_section_an_edit_did_not_reach_is_not_heard_again(torch, monkeypatch):
    """A take is the song with a few seconds laid into it, and the rest of it is the same samples.

    The second sound has other samples in the chorus alone; the third has
    two frames more in the verse, so its chorus is the first one's, two
    frames later.
    """
    calls = []
    lined(monkeypatch, calls)
    windows = the_windows()
    song = a_ramp(torch)
    first = edit_track._heard_by_sections(PLACE, "", "song", song, RATE, windows, None, None, set())
    assert [text for _key, text, _fill in calls] == [VERSE, CHORUS]
    assert [word for word, _start, _stop in first] == SUNG.split()
    assert first[0][1] == pytest.approx(84 * FRAME_SECONDS) and first[6][1] == pytest.approx(
        184 * FRAME_SECONDS), "each section's words are counted from where its stretch opens"
    take = song.clone()
    take[..., 300 * SAMPLES_A_FRAME:310 * SAMPLES_A_FRAME] += 1.0
    again = edit_track._heard_by_sections(PLACE, "", "take", take, RATE, windows, None, None, set())
    assert [text for _key, text, _fill in calls] == [VERSE, CHORUS, CHORUS], "the verse was not"
    assert again == first
    cut = 100 * SAMPLES_A_FRAME
    longer = torch.cat([song[..., :cut], torch.full((1, 2, 2 * SAMPLES_A_FRAME), 9.0),
                        song[..., cut:]], dim=-1)
    moved = [(84, 194, VERSE), (186, FRAMES + 2, CHORUS)]
    later = edit_track._heard_by_sections(PLACE, "", "longer", longer, RATE, moved, None, None, set())
    assert [text for _key, text, _fill in calls] == [VERSE, CHORUS, CHORUS, VERSE], "the chorus was not"
    assert later[6][1] == pytest.approx(186 * FRAME_SECONDS), "and its words went with it"


def test_a_section_never_heard_cannot_be_timed_without_the_aligner(torch, monkeypatch):
    calls = []
    lined(monkeypatch, calls)
    windows = the_windows()
    song = a_ramp(torch)
    assert edit_track._heard_by_sections(None, "", "song", song, RATE, windows, None, None, set()) is None
    found = edit_track._heard_by_sections(PLACE, "", "song", song, RATE, windows, None, None, set())
    assert edit_track._heard_by_sections(None, "", "song", song, RATE, windows, None, None,
                                         set()) == found, "heard once, it needs nobody"
    assert len(calls) == 2


def test_new_sections_are_separated_in_one_pass_and_timed_on_the_voice(torch, monkeypatch):
    """The separator costs seconds where the aligner costs tenths, so it runs once for all it is asked."""
    calls, asked = [], []
    lined(monkeypatch, calls)

    def voice_parts(samples, rate, spans, settings, progress):
        asked.append(list(spans))
        return [torch.full((high - low,), 7.0) for low, high in spans]

    monkeypatch.setattr(edit_track, "_voice_parts", voice_parts)
    edit_track._heard_by_sections(PLACE, "", "song", a_ramp(torch), RATE, the_windows(),
                                  {"device": "cuda"}, None, set())
    assert asked == [[(84 * SAMPLES_A_FRAME, 192 * SAMPLES_A_FRAME),
                      (184 * SAMPLES_A_FRAME, FRAMES * SAMPLES_A_FRAME)]]
    assert [fill for _key, _text, fill in calls] == [7.0, 7.0], "the voice, not the song as mixed"


def test_without_the_separator_the_sections_are_timed_on_the_song_as_mixed(torch, monkeypatch):
    """'download' off and no separator on the machine: ten lines in a hundred worse, not no lines."""
    calls = []
    lined(monkeypatch, calls)
    found = edit_track._heard_by_sections(PLACE, "", "song", a_wave(torch, fill=0.3), RATE,
                                          the_windows(), {"device": "cuda"}, None, set())
    assert [fill for _key, _text, fill in calls] == [0.3, 0.3] and len(found) == 8


def test_the_voice_a_grid_was_measured_by_is_the_one_its_words_are_timed_on(torch, monkeypatch):
    """Measuring a grid separates the whole voice; the words are timed straight after, and not on a second pass."""
    pytest.importorskip("torchaudio")
    from yue2_comfy.asr import runtime as asr_runtime

    heard = []
    monkeypatch.setattr(edit_track, "_aligner_at_hand", lambda settings: PLACE)
    monkeypatch.setattr(edit_track, "_voice_parts",
                        lambda *args, **kwargs: pytest.fail("the voice was separated again"))

    def word_times(folder, reader, device, waveform, rate, text, key, progress=None,
                   cancelled=None, low_vram=False):
        heard.append((text, rate, round(waveform.shape[-1] / rate, 2), round(float(waveform[0]), 2)))
        return [(word, 0.5 * index, 0.5 * index + 0.4) for index, word in enumerate(text.split())]

    monkeypatch.setattr(asr_runtime, "word_times", word_times)
    edit_track._voice_keep("a song", a_wave(torch, fill=0.5), RATE)
    edit_track._heard_by_sections(PLACE, "a song", "song", a_ramp(torch), RATE, the_windows(), None,
                                  None, set())
    assert heard == [(VERSE, 16000, 4.32, 0.5), (CHORUS, 16000, 13.64, 0.5)]
    edit_track.unload()
    assert not edit_track._WHOLE_VOICES, "the voice goes when the memory is asked back"


def test_the_separated_stretches_come_back_without_the_seconds_around_them(torch, monkeypatch):
    """Each stretch goes to the separator with a second of the song on each side and a second of silence after."""
    from yue2_comfy import devices, download
    from yue2_comfy.vocals import runtime as vocal_runtime

    given, gone = [], []
    monkeypatch.setattr(edit_track, "_voice_parts", VOICE_PARTS)
    monkeypatch.setattr(download, "ensure_vocals", lambda settings, progress=None: "weights")
    monkeypatch.setattr(devices, "resolve", lambda spec: "card")
    monkeypatch.setattr(vocal_runtime, "unload", lambda: gone.append(1))

    def separate(path, device, waveform, rate, progress=None, cancelled=None):
        given.append((path, device, tuple(waveform.shape)))
        return waveform * 2.0

    monkeypatch.setattr(vocal_runtime, "separate", separate)
    samples = a_ramp(torch, 200)[0]
    spans = [(0, 2 * RATE), (4 * RATE, 5 * RATE)]
    found = edit_track._voice_parts(samples, RATE, spans, {"device": "auto"}, None)
    assert given == [("weights", "card", (1, 2, (3 + 1 + 3 + 1) * RATE))]
    assert [tuple(clip.shape) for clip in found] == [(2 * RATE,), (RATE,)]
    assert torch.equal(found[0], samples[0, :2 * RATE] * 2.0)
    assert torch.equal(found[1], samples[0, 4 * RATE:5 * RATE] * 2.0)
    assert gone == [1]
    edit_track._voice_parts(samples, RATE, spans, {"device": "auto", "keep_model_loaded": True}, None)
    assert gone == [1], "a run that keeps what it loads keeps the separator"


def a_reader(monkeypatch, lost, calls):
    """The aligner stood in: ``lost`` says, by the lines it is given, how many stamps of each word came back out of order."""
    from yue2_comfy.asr import aligner, runtime as asr_runtime

    def word_times(folder, reader, device, waveform, rate, text, key, progress=None,
                   cancelled=None, low_vram=False):
        calls.append(text)
        found = aligner.Times((word, 0.5 * index, 0.5 * index + 0.4)
                              for index, word in enumerate(text.split()))
        found.lost = tuple(lost.get(text, [0] * len(found)))
        return found

    monkeypatch.setattr(asr_runtime, "word_times", word_times)


def test_a_section_that_comes_back_out_of_order_is_heard_again_with_the_one_before(torch, monkeypatch):
    """The model sang a verse's first line on the phrase left over at the end of the chorus before it.

    The verse's own stretch then holds three of its four lines, and the
    aligner, given four, answers with its stamps out of order. Heard with
    the section before, the line is where it is sung: ahead of its section.
    """
    calls = []
    a_reader(monkeypatch, {CHORUS: [2, 0]}, calls)
    windows = the_windows()
    song = a_ramp(torch)
    found = edit_track._heard_by_sections(PLACE, "", "song", song, RATE, windows, None, None, set())
    assert calls == [VERSE, CHORUS, VERSE + "\n" + CHORUS]
    assert [word for word, _start, _stop in found] == SUNG.split()
    opens = 84 * FRAME_SECONDS
    assert found[6][1] == pytest.approx(opens + 3.0), (
        "the chorus has the times of the stretch it shared, which opens where the verse does")
    assert found[5][1] == pytest.approx(opens + 2.5), "and the verse keeps its own"
    assert edit_track._heard_by_sections(None, "", "song", song, RATE, windows, None, None,
                                         set()) == found, "known by both stretches, it is not heard again"
    take = song.clone()
    take[..., 100 * SAMPLES_A_FRAME:110 * SAMPLES_A_FRAME] += 1.0
    edit_track._heard_by_sections(PLACE, "", "take", take, RATE, windows, None, None, set())
    assert calls[3:] == [VERSE, CHORUS, VERSE + "\n" + CHORUS], (
        "the stretch before it changed, so what was heard across both is heard again")


def test_a_section_heard_no_better_with_the_one_before_keeps_its_own_times(torch, monkeypatch):
    """Words that ran so far ahead that no hearing finds them come back out of order either way."""
    calls = []
    a_reader(monkeypatch, {CHORUS: [2, 0], VERSE + "\n" + CHORUS: [0, 0, 0, 0, 0, 0, 1, 1]}, calls)
    found = edit_track._heard_by_sections(PLACE, "", "song", a_ramp(torch), RATE, the_windows(), None,
                                          None, set())
    assert calls == [VERSE, CHORUS, VERSE + "\n" + CHORUS]
    assert found[6][1] == pytest.approx(184 * FRAME_SECONDS)


def test_a_section_read_well_enough_is_not_heard_twice(torch, monkeypatch):
    calls = []
    a_reader(monkeypatch, {VERSE: [2, 2, 2, 0, 0, 0], CHORUS: [0, 0]}, calls)
    edit_track._heard_by_sections(PLACE, "", "song", a_ramp(torch), RATE, the_windows(), None, None, set())
    assert calls == [VERSE, CHORUS], "the first section has none before it, and the second read every stamp"
    assert edit_track._unread([("a", 0.0, 0.4)]) == 0.0, "times that do not say are taken as read"


def test_word_times_kept_before_sections_were_heard_are_heard_again(stand, monkeypatch, tmp_path):
    """Until 0.13.0 a song was heard in one pass, which loses its place in a long one."""
    from yue2_comfy.inpaint import lines

    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    name = songs.remember(stand.wave, RATE, stand.song)
    text = lines.heard_text(LYRICS)
    old = [(word, 30.0 + index, 30.4 + index) for index, word in enumerate(text.split())]
    edit_track._times_keep(name, text, old)
    assert edit_track._times_read(name, text) == old, "a song with no sections still reads them"
    assert edit_track._times_read(name, text, sections=True) is None
    calls = []
    lined(monkeypatch, calls)
    assert payload(run(stand))["lines"] == LINES
    assert [said for _key, said, _fill in calls] == [VERSE, CHORUS]
    kept = json.loads((tmp_path / (name + edit_track.WORDS_SUFFIX)).read_text(encoding="utf-8"))
    assert kept["heard"] == edit_track.BY_SECTIONS
    assert edit_track._times_read(name, text, sections=True) is not None


def test_times_read_beside_a_song_say_which_of_its_sections_are_known(torch, monkeypatch):
    """So a take sung on a song opened from the disk hears only what its edit reached."""
    calls = []
    lined(monkeypatch, calls)
    windows = the_windows()
    song = a_ramp(torch)
    times = [(word, start, start + 0.4) for word, start in heard_at()]
    edit_track._parts_learn(song, RATE, windows, times)
    found = edit_track._heard_by_sections(None, "", "song", song, RATE, windows, None, None, set())
    assert [(word, round(start, 3)) for word, start, _stop in found] == [
        (word, round(start, 3)) for word, start in heard_at()]
    monkeypatch.setattr(edit_track, "_PARTS", type(edit_track._PARTS)())
    ahead = times[:6] + [("seven", 6.0, 6.4), ("eight", 7.5, 7.9)]
    edit_track._parts_learn(song, RATE, windows, ahead)
    found = edit_track._heard_by_sections(None, "", "song", song, RATE, windows, None, None, set())
    assert found[6][1] == pytest.approx(6.0), (
        "a section that begins in the stretch before its own is known by both of them")
    monkeypatch.setattr(edit_track, "_PARTS", type(edit_track._PARTS)())
    edit_track._parts_learn(song, RATE, windows, times[:6] + [("seven", 1.0, 1.4), ("eight", 7.5, 7.9)])
    assert edit_track._heard_by_sections(None, "", "song", song, RATE, windows, None, None,
                                         set()) is None, "words that lie in neither are heard when asked for"


def test_a_song_whose_sections_cannot_be_paired_is_heard_in_one_pass(torch, monkeypatch):
    """Four sections of words on a score that sings in three: nothing says where each is sung."""
    words = "[Verse]\none two\n\n[Chorus]\nthree four\n\n[Verse]\nfive six\n\n[Chorus]\nseven eight"
    wave = a_wave(torch, fill=0.4)
    songs.remember(wave, RATE, a_song(lyrics=words))
    monkeypatch.setattr(edit_track, "_GRIDS", type(edit_track._GRIDS)())
    monkeypatch.setattr(edit_track, "_grid_of", lambda *args, **kwargs: a_clock())
    calls = []
    lined(monkeypatch, calls)
    drawn = payload(edit_track.YuE2EditTrack().edit({"waveform": wave, "sample_rate": RATE}, takes=2))
    assert [text for _key, text, _fill in calls] == ["one two\nthree four\nfive six\nseven eight"]
    assert drawn["lines"] == [[1, 0.0, 0.9], [4, 1.0, 1.9], [7, 2.0, 2.9], [10, 3.0, 3.9]]


def test_a_song_with_no_score_is_heard_in_one_pass(torch, monkeypatch):
    wave = a_wave(torch, fill=0.6)
    songs.remember(wave, RATE, a_song(score_text="", settings={"cot": "off"}))
    calls = []
    lined(monkeypatch, calls)
    drawn = payload(edit_track.YuE2EditTrack().edit({"waveform": wave, "sample_rate": RATE}, takes=2))
    assert [text for _key, text, _fill in calls] == [VERSE + "\n" + CHORUS]
    assert drawn["lines"] == [[3, 0.0, 1.4], [4, 1.5, 2.9], [7, 3.0, 3.9]]


RESTYLE = ('[{"op": "restyle", "bars": [4, 8], "style": "hip hop, male rapper", "melody": "same",'
           ' "octave": -1, "seed": 9, "takes": 2}]')
"""The chorus in another style, an octave down, two takes."""


def restyled_by_stand(torch, calls):
    """A stand-in for the core's restyle: the song laid with itself, sung a little slower each take."""
    from yue2_comfy.inpaint import core

    def restyled(models, old, waveform, state, step, style, octave, seeds, settings, clock_of,
                 progress=None, cancelled=None):
        calls.append((style, octave, tuple(seeds), step.stretches, step.start, step.stop))
        made = []
        for index, seed in enumerate(seeds):
            clock = grid.Grid(offset=0.6, rate=1.0 + 0.02 * index, tick=state.clock.tick,
                              starts=state.clock.starts)
            spans = track.restyle_spans(state, step, step.score)
            laid = track.restyle_layout(state, step, state.sheet, clock, FRAMES + 10, spans)
            frames = sum(stop - start for _which, start, stop in laid.pieces)
            made.append(core.Take(seed=seed, waveform=a_wave(torch, frames, 0.1 * (index + 1)),
                                  song=a_song(frames, score_text=RAP),
                                  count=frames - (state.frames - step.stop) - step.start,
                                  join=None, joins={}, ended=False, timing={}, laid=laid))
        return made

    return restyled


def test_a_restyle_is_sung_by_the_core_and_drawn_as_takes_of_the_song_it_makes(stand, torch,
                                                                              monkeypatch):
    from yue2_comfy.inpaint import core

    calls = []
    monkeypatch.setattr(core, "restyled", restyled_by_stand(torch, calls))
    monkeypatch.setattr("yue2_comfy.vocals_only.separator_weights",
                        lambda settings, unique_id, progress: "separator.ckpt")
    drawn = payload(run(stand, RESTYLE))
    state = track.opened(a_song(), a_clock())
    step = track.plan(state, track.read(RESTYLE)[0])
    assert calls == [("hip hop, male rapper", -1, (9, 10), ((4, 8),), step.start, step.stop)]
    assert drawn["kind"] == "restyle" and drawn["style"] == "a style"
    assert drawn["at"] == [round(step.start * FRAME_SECONDS, 3),
                           round(step.stop * FRAME_SECONDS, 3)]
    assert [take["seed"] for take in drawn["takes"]] == [9, 10] and drawn["chosen"] == 0
    first, second = drawn["takes"]
    assert second["total"] > first["total"], "the slower take comes out longer"
    assert drawn["before"]["index"] == -1
    mark = songs.store().get(drawn["song"]).edit[0]
    assert (mark["op"], mark["seed"]) == ("restyle", 9)
    run(stand, RESTYLE)
    assert len(calls) == 1, "the same restyle is not sung twice"


def test_a_restyle_measures_the_song_it_sang_by_its_voice_on_the_bars_it_reached(monkeypatch):
    """The grid of the song sung in the new style comes from its separated voice, on its score read as far as it was sung."""
    found = []

    def voice_of(audio, settings, unique_id, progress, path=None):
        found.append((path, audio["sample_rate"]))
        return {"waveform": "voice"}

    monkeypatch.setattr("yue2_comfy.vocals_only.voice_of", voice_of)

    def measured(sheet, waveform, rate, voice=None):
        found.append((len(sheet["bars"]), voice))
        return grid.Grid(offset=0.7, rate=1.0, tick=grid.tick_seconds(sheet),
                         starts=grid.starts_of(sheet), by_voice=True)

    monkeypatch.setattr(grid, "measured", measured)
    clock = edit_track._measuring({}, "7", "separator.ckpt")
    sung = a_song()
    made = clock(sung, "sound", notation.read(RAP), 6, None)
    assert found == [("separator.ckpt", RATE), (6, "voice")]
    assert made.offset == 0.7 and made.starts == grid.starts_of(notation.read(RAP)), (
        "measured on six bars, it carries the whole score's")
