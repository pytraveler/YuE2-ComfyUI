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
        return object()

    def __exit__(self, *exc):
        return False


def run(stand, edits="", takes=2):
    return edit_track.YuE2EditTrack().edit(stand.audio, takes=takes, edits=edits)


def payload(answer):
    return answer["ui"][EDIT_TRACK_UI][0]


def test_the_widgets_sit_in_the_order_a_saved_workflow_expects():
    """ComfyUI stores widget values by position, so this order is part of every saved workflow."""
    shape = edit_track.YuE2EditTrack.INPUT_TYPES()
    assert list(shape["required"]) == ["audio", "takes"]
    assert list(shape["optional"]) == ["edits", "options"]
    assert shape["required"]["takes"][1]["max"] == track.MAX_TAKES
    assert edit_track.YuE2EditTrack.RETURN_NAMES == ("audio",)


def test_a_song_the_pack_never_sang_is_refused_with_what_can_be_edited(torch):
    node = edit_track.YuE2EditTrack()
    with pytest.raises(ValueError, match="does not remember singing this audio"):
        node.edit({"waveform": a_wave(torch), "sample_rate": RATE})
    with pytest.raises(ValueError, match="No audio is connected"):
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
    assert len(written) == 3
    assert [name for name in written if name.endswith("_40.wav")]
    assert [name for name in written if name.startswith("song_")]
    for take in payload(answer)["takes"]:
        assert take["audio"]["subfolder"] == "yue2_edit"
        assert take["audio"]["type"] == "temp"


def test_a_cut_takes_the_words_of_the_section_it_empties(stand):
    answer = run(stand, '[{"op": "cut", "bars": [2, 4]}]')
    drawn = payload(answer)
    assert drawn["dropped"] == ["Verse"]
    assert drawn["takes"][0]["seconds"] == 0.0


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


def test_an_excerpt_is_written_again_every_time(stand, a_temp_folder_of_its_own):
    """The name of an excerpt says nothing about the decoder or the card the take came out of, so
    a file left from an earlier run must not stand in for a take sung again."""
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]')
    place = a_temp_folder_of_its_own / "yue2_edit"
    before = {path.name: path.stat().st_mtime_ns for path in place.iterdir()}
    for path in place.iterdir():
        path.write_bytes(b"stale")
    edit_track.RESULTS.clear()
    run(stand, '[{"op": "retake", "bars": [4, 8], "seed": 40}]')
    for path in place.iterdir():
        assert path.stat().st_size > 5, path.name
    assert set(path.name for path in place.iterdir()) == set(before)


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

    take = core.Take(0, a_wave(pytest.importorskip("torch"), 10), None, 5, None, {}, True, {},
                     natural=-3.4)
    facts = edit_track._take_facts(take, 0, 0, "x" * 64, RATE, 0)
    assert facts["join"] is None and facts["natural"] is None and facts["ended"]


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
