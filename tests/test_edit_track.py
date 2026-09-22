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


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def a_song(frames=FRAMES, score_text=RAP, lyrics=LYRICS, settings=None):
    """A song the store will take: the smallest one that is still a song."""
    return songs.made("YuE2 Generate Song", "a style", lyrics, 7,
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

    def retakes(models, old, waveform, region, seeds, settings, progress=None, cancelled=None,
                noise_seeds=None, natural=False):
        calls.append(("retake", region.start, region.stop, tuple(seeds)))
        made = []
        for index, seed in enumerate(seeds):
            frames = old.frames - region.removed + region.length
            made.append(core.Take(seed=seed, waveform=a_wave(torch, frames, 0.1 * (index + 1)),
                                  song=a_song(frames), count=region.length,
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

    monkeypatch.setattr(core, "retakes", retakes)
    monkeypatch.setattr(core, "cut", cut)
    return types.SimpleNamespace(song=song, wave=wave, name=name, calls=calls,
                                 audio={"waveform": wave, "sample_rate": RATE})


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
    assert list(shape["optional"]) == ["audio", "edits", "options", "song_key"]
    fields = dict(list(shape["required"].items()) + list(shape["optional"].items()))
    assert [name for name, spec in fields.items() if spec[0] in ("INT", "STRING")] == [
        "takes", "edits", "song_key"], (
        "widget values are stored by position, so takes and edits keep theirs and the new "
        "field goes last")
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
                           "bars": [4, 8], "seed": 41,
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

    def session(settings, unique_id, progress):
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
                noise_seeds=None, natural=False):
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
    def session(settings, unique_id, bar):
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
