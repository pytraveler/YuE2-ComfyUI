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

SPEECH_FOLDER = edit_track._speech_folder
"""The real lookup, kept before the fixture below stands it in for every test."""

WORDS = ('[{"op": "words", "bars": [4, 8], "lines": [3, 4], "text": "one two four", '
         '"seed": 40}]')
"""A change of words on the stretch selected: line 4 of the words, sung over bars 5 to 8."""

GOES_ON = "% bridge\nV: Vocal\nA2z8D2B4|A2z4FG3F2B2B2|\nV: Ins\nZ|Z|\n"
"""What the stand-in for the model writes after the score it goes on from: two bars of a bridge."""

BRIDGE = "[Bridge]\nnine ten eleven"


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


@pytest.fixture(autouse=True)
def no_ears_of_the_machine(monkeypatch):
    """The node never reaches the speech models of the machine the tests run on.

    The embedded Python these tests run with finds the aligner and Qwen3-ASR
    where this machine keeps them, so a run would load both onto the card and
    hear silence -- and a machine without them, downloading on, would fetch
    4 GB. A test about hearing stands them in itself.
    """
    monkeypatch.setattr(edit_track, "_aligner_at_hand", lambda settings: None)
    monkeypatch.setattr(edit_track, "_speech_folder", lambda settings, progress, notices: None)
    monkeypatch.setattr(edit_track, "_ears_at_hand", lambda settings: None)


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

    monkeypatch.setattr(core, "retakes", retakes)
    monkeypatch.setattr(core, "cut", cut)
    monkeypatch.setattr(core, "extended", extended)
    return types.SimpleNamespace(song=song, wave=wave, name=name, calls=calls, words=words,
                                 scores=scores, audio={"waveform": wave, "sample_rate": RATE})


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

    The line chosen is the song's first, so it opens where it does and closes
    a step before the next line's first word: 1.5 s less 80 ms, which is frame
    36.
    """
    from yue2_comfy import download
    from yue2_comfy.asr import runtime as asr_runtime

    asked = {}

    def word_times(folder, reader, device, waveform, rate, text, key, progress=None,
                   cancelled=None):
        asked.update(folder=folder, reader=reader, text=text, key=key)
        found, at = [], 0.0
        for word in text.split():
            found.append((word, at, at + 0.4))
            at += 0.5
        return found

    monkeypatch.setattr(download, "ensure_aligner", lambda settings, progress=None: "aligner")
    monkeypatch.setattr(download, "aligner_tokenizer",
                        lambda folder, settings, progress=None: "tokenizer")
    monkeypatch.setattr(asr_runtime, "word_times", word_times)
    run(stand, '[{"op": "words", "lines": [3, 4], "text": "one two four", "seed": 40}]')
    assert asked["text"].splitlines() == ["one two three", "four five six", "seven eight"], (
        "the aligner is given the words as they are sung, tags and blank lines left out")
    assert stand.calls == [("retake", 0, 36, (40, 41))]
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

    def hear(folder, device, clips, language="", cancelled=None, progress=None):
        answers = []
        for name, clip in clips:
            if seen is not None:
                seen.append((name[1], clip))
            answers.append({"language": "English", "text": said[name[1]]})
        return answers

    monkeypatch.setattr(asr_runtime, "hear", hear)


def lined(monkeypatch, calls=None, first=0.0, gap=0.5):
    """The aligner stood in and at hand: the first word at ``first`` seconds, each next ``gap`` after the last."""
    from yue2_comfy.asr import runtime as asr_runtime

    monkeypatch.setattr(edit_track, "_aligner_at_hand",
                        lambda settings: ("aligner", "tokenizer", "cuda"))

    def word_times(folder, reader, device, waveform, rate, text, key, progress=None,
                   cancelled=None):
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


def sung_there(at, first=4.6, gap=1.0):
    """The words of ``SUNG`` whose middle lies in ``at``, timed as ``lined(first=4.6, gap=1.0)`` times them."""
    return [word for index, word in enumerate(SUNG.split())
            if at[0] <= first + index * gap + 0.2 < at[1]]


def test_a_retake_keeps_the_take_heard_singing_the_words_there(stand, monkeypatch):
    """Take 2 joins better; take 1 is heard singing the verse. The words are what is listened to."""
    lined(monkeypatch, first=4.6, gap=1.0)
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
    lined(monkeypatch, first=4.6, gap=1.0)
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
    edit_track._times_keep(name, text, [(word, 4.6 + index, 5.0 + index)
                                        for index, word in enumerate(text.split())])
    seen = []
    hearing(monkeypatch, {"was": SUNG, 9: SUNG, 10: "one"}, seen)
    drawn = payload(run(stand, RETAKE))
    assert [clip for clip, _cut in seen] == ["was", 9, 10]
    assert drawn["chosen"] == 0 and drawn["asked"] == " ".join(sung_there(drawn["at"]))


def test_a_retake_without_the_speech_model_is_left_to_the_join_without_a_word(stand,
                                                                             monkeypatch):
    """Hearing a retake is a better pick, not the edit: nothing is fetched and nothing is said."""
    from yue2_comfy import download

    lined(monkeypatch, first=4.6, gap=1.0)
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
    lined(monkeypatch, first=4.6, gap=1.0)
    hearing(monkeypatch, {}, seen)
    drawn = payload(run(stand, '[{"op": "retake", "bars": [8, 10], "seed": 9}]'))
    assert seen == [] and drawn["chosen"] == 1 and drawn["asked"] is None


def test_a_retake_heard_as_well_as_the_song_it_replaces_is_not_mumbling(stand, monkeypatch):
    """The recogniser misses the same words in the song and in its takes."""
    lined(monkeypatch, first=4.6, gap=1.0)
    hearing(monkeypatch, {"was": "one", 9: "one", 10: "one"})
    said = said_by(monkeypatch)
    drawn = payload(run(stand, RETAKE))
    assert "one" in sung_there(drawn["at"]) and drawn["before"]["heard"][0] == 1
    assert [take["mumbled"] for take in drawn["takes"]] == [False, False]
    assert [text for kind, text in said if kind == "warn"] == []


def test_a_retake_kept_that_is_heard_singing_less_than_the_song_did_is_said_to(stand,
                                                                               monkeypatch):
    lined(monkeypatch, first=4.6, gap=1.0)
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


def test_the_window_is_told_when_each_line_is_sung(stand, monkeypatch):
    """Lines 4, 5 and 8 of the words are the sung ones; tags and blank lines have no time."""
    lined(monkeypatch)
    drawn = payload(run(stand))
    assert drawn["lines"] == [[3, 0.0, 1.4], [4, 1.5, 2.9], [7, 3.0, 3.9]]


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
    assert drawn["before"]["lines"] == [[3, 0.0, 1.4], [4, 1.5, 2.9], [7, 3.0, 3.9]]
    assert sorted(fill for _key, _text, fill in calls) == [0.0, 0.1, 0.2], (
        "the song, the other take and the song as it was, each heard once")
    assert len({key for key, _text, _fill in calls}) == 3


def test_the_song_as_it_was_is_numbered_as_the_words_the_window_shows(stand, monkeypatch):
    """One line became two, so every line after them moved down by one."""
    lined(monkeypatch)
    drawn = payload(run(stand, WORDS.replace("one two four", "one two\\nfour")))
    assert drawn["lines"] == [[3, 0.0, 0.9], [4, 1.0, 1.4], [5, 1.5, 2.9], [8, 3.0, 3.9]]
    assert drawn["before"]["lines"] == [[3, 0.0, 1.4], [4, 0.0, 1.4], [5, 1.5, 2.9],
                                        [8, 3.0, 3.9]]


def test_the_lines_of_a_song_wait_beside_it_for_the_next_run(stand, monkeypatch, tmp_path):
    monkeypatch.setattr(songs, "_store", songs.Store(str(tmp_path)))
    name = songs.remember(stand.wave, RATE, stand.song)
    calls = []
    lined(monkeypatch, calls)
    run(stand)
    assert len(calls) == 1 and (tmp_path / (name + edit_track.WORDS_SUFFIX)).is_file()
    run(stand)
    assert len(calls) == 1, "read beside the song, not heard again"


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
    lined(monkeypatch, first=4.6, gap=1.0)
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

    def told(folder, device, clips, language="", cancelled=None, progress=None):
        languages.append(language)
        return heard(folder, device, clips, language, cancelled, progress)

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
        ("one two three\nfour five six\nseven eight", 0.0),
        ("nine ten eleven\ntwelve thirteen", 0.1)}, "never the whole song's words at once"
    assert drawn["lines"] == [[3, 0.0, 1.4], [4, 1.5, 2.9], [7, 3.0, 3.9],
                              [10, round(since, 3), round(since + 1.4, 3)],
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
