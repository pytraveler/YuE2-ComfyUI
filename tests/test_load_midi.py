"""YuE2 Load MIDI: files from the input folder, a score and lyrics out, edits kept per file, and its two routes."""

from __future__ import annotations

import base64
import os
import re

import pytest
from midi_files import file_of, kar_texts, line

from yue2_comfy import edits, load_midi, routes
from yue2_comfy.midi import score, smf

TUNE = line([67, 69, 71, 72, 71, 69, 67, 67])
BASS = line([43, 43, 48, 48], step=960)


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """An input folder holding 'song.mid', with the node's toasts caught; returns the folder and the toasts."""
    said = []
    monkeypatch.setattr(load_midi, "input_folder", lambda: str(tmp_path))
    monkeypatch.setattr(load_midi, "announce", lambda node, findings, kind="notice": said.append(findings))
    (tmp_path / "song.mid").write_bytes(file_of(("Vocal", 0, 0, TUNE, []), ("Bass", 1, 33, BASS, [])))
    return tmp_path, said


def run(midi="song.mid", mode="melody", vocal="auto", instrument="auto", score_abc="", lyrics="", **more):
    return load_midi.YuE2LoadMidi().load(midi, mode, vocal, instrument, score_abc=score_abc, lyrics=lyrics,
                                         unique_id="7", **more)


def test_the_widgets_sit_in_their_final_order(folder):
    """'without_sections' came last, after the two edit boxes: a saved workflow's values stay where they were."""
    spec = load_midi.YuE2LoadMidi.INPUT_TYPES()
    assert list(spec["required"]) == ["midi", "mode", "vocal_track", "instrument_track"]
    assert list(spec["optional"]) == ["score_abc", "lyrics", "without_sections"]
    assert spec["optional"]["without_sections"] == ("BOOLEAN", {"default": True,
                                                                "tooltip": load_midi.SECTIONS_TOOLTIP})
    assert spec["required"]["midi"][0] == ["song.mid"]
    assert spec["required"]["vocal_track"][0][:3] == ["auto", "1", "2"]
    assert spec["required"]["instrument_track"][0][:3] == ["auto", "none", "1"]


def test_only_midi_files_at_the_top_of_the_input_folder_are_offered(folder):
    root, _said = folder
    for name in ("b.KAR", "c.txt", "a.midi"):
        (root / name).write_bytes(b"x")
    (root / "sub").mkdir()
    (root / "sub" / "d.mid").write_bytes(b"x")
    assert load_midi.midi_files() == ["a.midi", "b.KAR", "song.mid"]


def test_the_outputs_are_the_files_score_and_lyrics_and_the_ui_carries_marks_and_tracks(folder):
    root, _said = folder
    data = (root / "song.mid").read_bytes()
    expected = score.convert(smf.read(data), "melody")
    out = run()
    assert out["result"] == (expected["abc"], "[Verse]")
    mark = edits.file_mark(data)
    ui = out["ui"]
    assert ui[edits.TRACK_UI] == [mark] and ui[edits.SCORE_UI] == [expected["abc"]]
    assert ui[edits.WORDS_UI] == [edits.midi_mark(mark, "melody", "auto", "auto")]
    assert ui[edits.MARKS_UI] == [{mode: edits.midi_mark(mark, mode, "auto", "auto") for mode in ("melody", "full")}]
    listed = ui[edits.MIDI_UI][0]
    assert listed["name"] == "song.mid" and [part["role"] for part in listed["parts"]] == ["voice", ""]
    assert listed["facts"]["bars"] == 2


def test_an_edit_for_this_file_mode_and_tracks_is_output_and_one_for_another_choice_is_not(folder):
    root, said = folder
    mark = edits.file_mark((root / "song.mid").read_bytes())
    own = edits.attach("X:1\nedited", edits.midi_mark(mark, "melody", "auto", "auto"))
    assert run(score_abc=own)["result"][0] == "X:1\nedited"
    assert run(score_abc=own, vocal="2")["result"][0] != "X:1\nedited"
    assert ("warn", load_midi.OTHER_FILE_SCORE) in said[-1]
    assert run(score_abc="X:1\npasted")["result"][0] == "X:1\npasted"


def test_the_score_goes_out_as_a_bare_tune_unless_its_sections_are_asked_for(folder):
    """The editor still gets the file's sections; only what goes out along the wire loses them."""
    root, _said = folder
    (root / "marked.mid").write_bytes(file_of(("Vocal", 0, 0, TUNE, []), markers=((0, "Chorus"),)))
    kept = run("marked.mid", without_sections=False)
    bare = run("marked.mid")
    assert "% chorus" in kept["result"][0].splitlines()
    assert bare["result"][0] == "\n".join(line for line in kept["result"][0].split("\n") if not line.startswith("% "))
    assert "% chorus" in bare["ui"][edits.SCORE_UI][0].splitlines()
    plain = run(without_sections=False)["result"][0]
    assert [line for line in plain.splitlines() if line.startswith("% ")] == ["% verse"]
    assert run()["result"][0] == score.convert(smf.read((root / "song.mid").read_bytes()), "melody")["abc"]


def test_an_edit_goes_out_bare_or_named_the_same_way(folder):
    root, _said = folder
    mark = edits.file_mark((root / "song.mid").read_bytes())
    written = score.convert(smf.read((root / "song.mid").read_bytes()), "melody")["abc"]
    edited = written.replace("\nV: Vocal\n", "\n% bridge\nV: Vocal\n", 1)
    own = edits.attach(edited, edits.midi_mark(mark, "melody", "auto", "auto"))
    assert "% bridge" not in run(score_abc=own)["result"][0]
    assert "% bridge" in run(score_abc=own, without_sections=False)["result"][0]


def test_lyrics_belong_to_the_file(folder):
    root, said = folder
    mark = edits.file_mark((root / "song.mid").read_bytes())
    assert run(lyrics=edits.attach("[Verse]\nmine", mark))["result"][1] == "[Verse]\nmine"
    assert run(lyrics=edits.attach("[Verse]\nold", "0" * 16))["result"][1] == "[Verse]"
    assert ("warn", load_midi.OTHER_FILE_LYRICS) in said[-1]


def test_karaoke_words_are_the_lyrics(folder):
    root, _said = folder
    syllables = [b"\\Sing", b" a", b" song", b" of", b"/six", b"pence", b" a", b" pocket"]
    (root / "words.kar").write_bytes(file_of(("Vocal", 0, 0, TUNE, kar_texts(syllables, step=480))))
    assert run("words.kar")["result"][1] == "[Verse]\nSing a song of\nsixpence a pocket"


@pytest.mark.parametrize("name, data, vocal, reason", [
    ("broken.mid", b"MThd", "auto", "'broken.mid' could not be read as a MIDI file: the MIDI header is cut short"),
    ("song.mid", None, "5", "No score could be written from 'song.mid': 'vocal_track' is track 5"),
    ("missing.mid", None, "auto", "'missing.mid' is not in ComfyUI's input folder"),
])
def test_what_cannot_be_read_or_sung_is_refused_with_the_reason(folder, name, data, vocal, reason):
    root, _said = folder
    if data is not None:
        (root / name).write_bytes(data)
    with pytest.raises(ValueError, match=re.escape(reason)):
        run(name, vocal=vocal)


def test_a_changed_file_runs_again_and_a_wrong_name_is_caught_before_the_run(folder):
    root, _said = folder
    node = load_midi.YuE2LoadMidi
    before = node.IS_CHANGED(midi="song.mid", mode="melody")
    (root / "song.mid").write_bytes(file_of(("Vocal", 0, 0, line([60, 62]), [])))
    assert node.IS_CHANGED(midi="song.mid", mode="melody") != before
    assert node.VALIDATE_INPUTS(midi="song.mid") is True
    assert "not in ComfyUI's input folder" in node.VALIDATE_INPUTS(midi="gone.mid")
    assert "outside ComfyUI's input folder" in node.VALIDATE_INPUTS(midi="../song.mid")
    assert "not a MIDI file" in node.VALIDATE_INPUTS(midi="song.wav")
    assert load_midi.path_of("song.mid [input]") == os.path.realpath(str(root / "song.mid"))


def test_the_tracks_route_lists_a_file_before_any_run(folder):
    payload, status = routes.answer_midi_tracks({"name": "song.mid"})
    assert status == 200 and payload["ok"]
    assert [(part["number"], part["name"], part["role"]) for part in payload["parts"]] == [
        (1, "Vocal", "voice"), (2, "Bass", "")]
    wrong, status = routes.answer_midi_tracks({"name": "song.mid", "vocal_track": "9"})
    assert status == 200 and not wrong["ok"] and "track 9" in wrong["error"] and len(wrong["parts"]) == 2
    missing, _status = routes.answer_midi_tracks({"name": "gone.mid"})
    assert not missing["ok"] and missing["parts"] == []
    assert routes.answer_midi_tracks({"name": 3})[1] == 400


def test_the_save_route_hands_back_the_score_as_a_midi_file():
    text = score.convert(smf.read(file_of(("Vocal", 0, 0, TUNE, []))))["abc"]
    payload, status = routes.answer_score_midi({"abc": edits.attach(text, "0123456789abcdef")})
    assert status == 200 and payload["ok"]
    song = smf.read(base64.b64decode(payload["data"]))
    assert [note.pitch for note in song.tracks[1].notes] == [67, 69, 71, 72, 71, 69, 67, 67]
    broken, status = routes.answer_score_midi({"abc": "not a score"})
    assert status == 200 and not broken["ok"]
    assert routes.answer_score_midi({})[1] == 400
