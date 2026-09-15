"""YuE2 Transcribe without torch or weights: the transcription is replaced by master's tokens."""

from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest

from yue2_comfy import download, devices, edits, transcribe
from yue2_comfy.sheetsage import abc_rebuild, events, runtime, sections, vocab

DATA = json.loads((pathlib.Path(__file__).parent / "data" / "sheetsage_pack.json").read_text())
SONG = DATA["songs"][0]


def master_result(song=SONG):
    duration = song["seconds"]
    stitched = []
    for index, (window, tokens) in enumerate(zip(events.window_plan(duration), song["windows"])):
        decoded, _warning = vocab.decode_window(tokens + [vocab.EOS])
        stitched.extend(events.stitch(decoded, events.time_map(decoded), window, duration, index))
    return {"events": events.sort_song(stitched), "seconds": duration, "windows": song["windows"], "warnings": [],
            "cut_short": []}


@pytest.fixture
def node(monkeypatch):
    """The node with the recording, the device, the weights and the model replaced; returns its calls."""
    calls = {"runs": 0, "said": []}

    def fake_track(audio):
        return {"samples": "samples", "rate": 44100, "count": audio.get("count", 1),
                "mark": edits.audio_mark(audio["data"], 44100), "seconds": SONG["seconds"]}

    def fake_transcribe(path, device, waveform, rate, key, progress=None, cancelled=None):
        calls["runs"] += 1
        calls["key"] = key
        return master_result()

    monkeypatch.setattr(transcribe, "track_of", fake_track)
    monkeypatch.setattr(devices, "validate", lambda spec: spec)
    monkeypatch.setattr(devices, "resolve", lambda spec: "device")
    monkeypatch.setattr(download, "ensure_sheetsage", lambda settings, progress=None: "weights.safetensors")
    monkeypatch.setattr(runtime, "stamp", lambda path: ("weights", 1, 2))
    monkeypatch.setattr(runtime, "transcribe", fake_transcribe)
    monkeypatch.setattr(runtime, "unload", lambda: None)
    monkeypatch.setattr(transcribe, "announce", lambda node, findings, kind="notice": calls["said"].append(findings))
    return calls


def run(mode="melody", data=b"recording", score_abc="", lyrics="", recognition=False, count=1, seed=1):
    return transcribe.YuE2Transcribe().transcribe(
        {"data": data, "count": count}, mode, recognition, "auto", seed, score_abc=score_abc, lyrics=lyrics,
        unique_id="7")


def faithful_answer(messages):
    """What a well-behaved model answers: the same blocks, a line after each full stop."""
    body = messages[1]["content"].split("\n\n", 1)[1]
    return body.replace(". ", ".\n")


@pytest.fixture
def heard(node, monkeypatch):
    """Recognition and the language model replaced as well; returns what they were asked."""
    from yue2_comfy import llm
    from yue2_comfy.asr import runtime as asr_runtime

    calls = node
    calls.update(asr=0, llm=[], answer=faithful_answer)

    def fake_recognise(folder, device, waveform, rate, timed, key, progress=None, cancelled=None):
        calls["asr"] += 1
        calls["timed"] = timed
        calls["asr_key"] = key
        parts = ["" if section["notes"] == 0 else "Words of the {}. Sung once more.".format(section["tag"].lower())
                 for section in timed]
        return {"language": "English", "text": " ".join(part for part in parts if part), "parts": parts}

    def fake_run(path, messages, seed, n_ctx, device, keep_loaded=False, progress=None, settings=None, **sampling):
        calls["llm"].append((path, seed, sampling.get("greedy")))
        return calls["answer"](messages)

    monkeypatch.setattr(download, "ensure_asr", lambda settings, progress=None: "asr-folder")
    monkeypatch.setattr(asr_runtime, "stamp", lambda folder: ("asr", 1, 2))
    monkeypatch.setattr(asr_runtime, "recognise", fake_recognise)
    monkeypatch.setattr(asr_runtime, "unload", lambda: None)
    monkeypatch.setattr(asr_runtime, "_LAYOUTS", asr_runtime.collections.OrderedDict())
    monkeypatch.setattr(llm, "resolve", lambda choice, settings, progress=None: "writer.gguf")
    monkeypatch.setattr(llm, "run", fake_run)
    monkeypatch.setattr(llm, "unload", lambda: None)
    return calls


def test_the_widgets_sit_in_their_final_order():
    """ComfyUI restores saved widget values by position, so this order is a promise."""
    spec = transcribe.YuE2Transcribe.INPUT_TYPES()
    assert list(spec["required"]) == ["audio", "mode", "lyrics_auto_recognition", "model", "seed"]
    assert list(spec["optional"]) == ["options", "score_abc", "lyrics"]
    assert transcribe.YuE2Transcribe.RETURN_NAMES == ("score_abc", "lyrics")
    assert spec["required"]["lyrics_auto_recognition"][1]["default"] is False
    assert spec["required"]["seed"][1]["control_after_generate"] == "fixed"


@pytest.mark.parametrize("mode", ["melody", "full"])
def test_the_score_is_masters_and_the_ui_carries_it_with_its_mark(node, mode):
    out = run(mode)
    assert out["result"][0] == SONG["abc"][mode]
    recording = edits.audio_mark(b"recording", 44100)
    assert out["ui"][edits.SCORE_UI] == [SONG["abc"][mode]]
    assert out["ui"][edits.WORDS_UI] == [edits.track_mark(recording, mode)]
    assert out["ui"][edits.TRACK_UI] == [recording]
    assert out["ui"][edits.MARKS_UI] == [{chosen: edits.track_mark(recording, chosen) for chosen in ("melody", "full")}]
    assert node["key"] == (recording, ("weights", 1, 2))


def test_the_lyrics_are_the_section_tags_of_the_score(node):
    out = run()
    found = sections.sections(SONG["abc"]["melody"])
    assert out["result"][1] == sections.skeleton(found)
    assert out["result"][1].startswith("[")
    assert all(line.startswith("[") or line == "" for line in out["result"][1].splitlines())


def test_an_edit_for_this_recording_and_mode_is_output_without_listening(node):
    edited = SONG["abc"]["melody"].replace("Q:1/4=", "Q:1/4=1", 1)
    mark = edits.track_mark(edits.audio_mark(b"recording", 44100), "melody")
    out = run(score_abc=edits.attach(edited, mark))
    assert node["runs"] == 0
    assert out["result"][0] == edited.strip()
    assert edits.SCORE_UI not in out["ui"]


def test_an_edit_for_the_other_mode_is_left_out_and_the_node_says_so(node):
    mark = edits.track_mark(edits.audio_mark(b"recording", 44100), "full")
    out = run(mode="melody", score_abc=edits.attach(SONG["abc"]["full"], mark))
    assert node["runs"] == 1
    assert out["result"][0] == SONG["abc"]["melody"]
    assert node["said"] == [[("warn", transcribe.OTHER_TRACK_SCORE)]]


def test_lyrics_written_for_this_recording_are_kept(node):
    recording = edits.audio_mark(b"recording", 44100)
    out = run(lyrics=edits.attach("[Verse]\nla la", recording))
    assert out["result"][1] == "[Verse]\nla la"
    out = run(data=b"another", lyrics=edits.attach("[Verse]\nla la", recording))
    assert out["result"][1].startswith("[") and "la la" not in out["result"][1]
    assert ("warn", transcribe.OTHER_TRACK_LYRICS) in node["said"][-1]


def test_an_edit_without_a_mark_is_output_as_it_is(node):
    """Pasted or written before the first run, it has no recording to belong to, as on the song node."""
    out = run(score_abc="X:1\nK:C\nV:Vocal\nC D E F|\n", lyrics="[Verse]\nmy own words")
    assert node["runs"] == 0
    assert out["result"] == ("X:1\nK:C\nV:Vocal\nC D E F|", "[Verse]\nmy own words")
    assert node["said"] == [[]]


def test_recognised_words_are_laid_out_under_the_section_tags(heard):
    out = run(recognition=True)
    lyrics = out["result"][1]
    sung = [section for section in heard["timed"] if section["notes"] > 0]
    assert lyrics == "\n\n".join("[{0}]\nWords of the {1}\nSung once more".format(section["tag"], section["tag"].lower())
                                 for section in sung)
    assert out["ui"][edits.LYRICS_UI] == [lyrics]
    assert heard["asr_key"] == (edits.audio_mark(b"recording", 44100), ("asr", 1, 2))
    assert heard["llm"] == [("writer.gguf", 1, False)]
    assert heard["said"] == [[]]


def test_the_sections_heard_are_the_transcriptions_in_seconds(heard):
    run(recognition=True)
    rows = events.score_rows(master_result()["events"], SONG["seconds"])
    assert heard["timed"] == sections.timed(rows, SONG["seconds"])
    assert heard["timed"][0]["start"] == 0.0 and heard["timed"][-1]["end"] == SONG["seconds"]


def test_a_new_seed_lays_the_words_out_again_and_the_same_seed_does_not(heard):
    run(recognition=True, seed=1)
    run(recognition=True, seed=1)
    run(recognition=True, seed=2)
    assert [seed for _path, seed, _greedy in heard["llm"]] == [1, 2]


def test_a_section_whose_words_the_model_changed_keeps_the_words_as_heard(heard):
    heard["answer"] = lambda messages: faithful_answer(messages).replace("Sung once more.", "Sung twice more.", 1)
    lyrics = run(recognition=True)["result"][1]
    assert "Sung twice more" not in lyrics and lyrics.count("Sung once more") == len(
        [section for section in heard["timed"] if section["notes"] > 0])
    total = len([section for section in heard["timed"] if section["notes"] > 0])
    assert heard["said"] == [[("notice", transcribe.LAYOUT_CHANGED.format(changed=1, total=total))]]


def test_without_a_language_model_the_words_break_at_their_punctuation(heard, monkeypatch):
    from yue2_comfy import llm

    def missing(choice, settings, progress=None):
        raise FileNotFoundError("no model here")

    monkeypatch.setattr(llm, "resolve", missing)
    lyrics = run(recognition=True)["result"][1]
    assert "Words of the" in lyrics and "\nSung once more" in lyrics
    assert heard["said"] == [[("notice", transcribe.LAYOUT_FAILED.format(reason="no model here"))]]


def test_recording_without_words_gives_the_tags_and_says_so(heard, monkeypatch):
    from yue2_comfy.asr import runtime as asr_runtime

    monkeypatch.setattr(asr_runtime, "recognise", lambda *args, **kwargs: {
        "language": "", "text": "", "parts": [""] * len(kwargs.get("timed", args[4]))})
    out = run(recognition=True)
    assert out["result"][1] == sections.skeleton(sections.sections(SONG["abc"]["melody"]))
    assert heard["llm"] == []
    assert heard["said"] == [[("notice", transcribe.NO_WORDS)]]


def test_kept_lyrics_still_win_and_the_editor_gets_the_recognised_words(heard):
    recording = edits.audio_mark(b"recording", 44100)
    out = run(recognition=True, lyrics=edits.attach("[Verse]\nmy words", recording))
    assert out["result"][1] == "[Verse]\nmy words"
    assert "Words of the" in out["ui"][edits.LYRICS_UI][0]


def test_a_kept_score_is_still_transcribed_for_the_sections_of_the_words(heard):
    mark = edits.track_mark(edits.audio_mark(b"recording", 44100), "melody")
    out = run(recognition=True, score_abc=edits.attach(SONG["abc"]["melody"], mark))
    assert heard["runs"] == 1 and heard["asr"] == 1
    assert edits.SCORE_UI not in out["ui"]


def test_recognition_with_downloading_off_is_refused_with_the_links(heard, monkeypatch):
    def refused(settings, progress=None):
        raise FileNotFoundError("Qwen3-ASR-1.7B ... is not on this machine yet")

    monkeypatch.setattr(download, "ensure_asr", refused)
    with pytest.raises(ValueError, match="not on this machine"):
        run(recognition=True)


def test_the_asr_runtime_reuses_words_for_the_same_recording_and_sections(monkeypatch):
    from yue2_comfy.asr import runtime as asr_runtime

    heard = []
    fake_model = types.SimpleNamespace(
        mono_16k=lambda waveform, rate: types.SimpleNamespace(numel=lambda: 16000 * 30),
        token_limit=lambda seconds: 400,
        recognise=lambda net, tokenizer, audio, language="", cancelled=None, progress=None:
            heard.append(language) or {"language": "English", "text": "la la"})
    monkeypatch.setitem(sys.modules, "yue2_comfy.asr.model", fake_model)
    monkeypatch.setattr(asr_runtime, "acquire", lambda folder, device, progress=None: ("net", "tokenizer"))
    monkeypatch.setattr(asr_runtime, "_RESULTS", asr_runtime.collections.OrderedDict())
    timed = [{"start": 0.0, "end": 30.0, "notes": 12}]
    first = asr_runtime.recognise("f", "d", "samples", 44100, timed, key=("a", 1))
    second = asr_runtime.recognise("f", "d", "samples", 44100, timed, key=("a", 1))
    asr_runtime.recognise("f", "d", "samples", 44100, [{"start": 0.0, "end": 29.0, "notes": 12}], key=("a", 1))
    assert first is second and first["parts"] == ["la la"] and heard == ["", ""]


def test_the_asr_runtime_keeps_a_layout_per_key(monkeypatch):
    from yue2_comfy.asr import runtime as asr_runtime

    monkeypatch.setattr(asr_runtime, "_LAYOUTS", asr_runtime.collections.OrderedDict())
    made = []
    assert asr_runtime.laid_out(("w", 1), lambda: made.append(1) or "A") == "A"
    assert asr_runtime.laid_out(("w", 1), lambda: made.append(2) or "B") == "A"
    for key in range(10):
        asr_runtime.laid_out(key, lambda: "x")
    assert len(asr_runtime._LAYOUTS) == asr_runtime.KEEP_RESULTS and made == [1]


def test_unload_models_also_releases_the_speech_model():
    from yue2_comfy import memory

    assert ("Qwen3-ASR", ".asr.runtime") in memory.keepers()


def test_a_batch_is_transcribed_from_its_first_recording_with_a_notice(node):
    run(count=3)
    assert node["said"] == [[("notice", transcribe.BATCH_NOTE.format(count=3))]]


def test_a_recording_with_no_beat_is_refused_with_the_reason(node, monkeypatch):
    monkeypatch.setattr(runtime, "transcribe", lambda *args, **kwargs: {"events": [], "seconds": 5.0,
                                                                         "windows": [], "warnings": []})
    with pytest.raises(ValueError, match="could not write a score"):
        run()


def test_a_recording_the_model_cannot_take_is_refused_with_the_reason(node, monkeypatch):
    def too_short(*args, **kwargs):
        raise ValueError("the recording needs at least 1025 finite samples at 24 kHz")

    monkeypatch.setattr(runtime, "transcribe", too_short)
    with pytest.raises(ValueError, match="could not transcribe this recording: the recording needs"):
        run()


def test_a_part_the_decoder_could_not_finish_is_said_on_the_node(node, monkeypatch):
    result = master_result()
    result["cut_short"] = [1]
    monkeypatch.setattr(runtime, "transcribe", lambda *args, **kwargs: result)
    run()
    assert node["said"] == [[("notice", transcribe.CUT_SHORT.format(parts=1, total=len(result["windows"]),
                                                                    tokens=5120))]]


def test_a_machine_without_a_language_model_is_told_about_the_download(heard, monkeypatch):
    from yue2_comfy import llm

    monkeypatch.setattr(llm, "catalogue", lambda: [])
    run(recognition=True)
    assert heard["said"][0][0][0] == "notice" and "downloaded" in heard["said"][0][0][1]
    monkeypatch.setattr(llm, "catalogue", lambda: [("writer.gguf (2.7 GB)", "writer.gguf")])
    heard["said"].clear()
    run(recognition=True, seed=3)
    assert heard["said"] == [[]]


def test_the_runtime_reuses_a_transcription_for_the_same_recording(monkeypatch):
    heard = []
    fake_model = types.SimpleNamespace(transcribe=lambda net, waveform, rate, cancelled=None, progress=None:
                                       heard.append(rate) or {"events": [], "seconds": 1.0})
    monkeypatch.setitem(sys.modules, "yue2_comfy.sheetsage.model", fake_model)
    monkeypatch.setattr(runtime, "acquire", lambda path, device, progress=None: "net")
    monkeypatch.setattr(runtime, "_RESULTS", runtime.collections.OrderedDict())
    first = runtime.transcribe("w", "d", "samples", 44100, key=("a", 1))
    second = runtime.transcribe("w", "d", "samples", 44100, key=("a", 1))
    runtime.transcribe("w", "d", "samples", 48000, key=("b", 1))
    assert heard == [44100, 48000] and first is second


def test_the_runtime_keeps_only_the_last_eight_transcriptions(monkeypatch):
    monkeypatch.setattr(runtime, "_RESULTS", runtime.collections.OrderedDict())
    for key in range(10):
        runtime.remember(key, {"n": key})
    assert list(runtime._RESULTS) == list(range(2, 10))
    assert runtime.remembered(0) is None and runtime.remembered(9) == {"n": 9}


def test_sections_count_every_vocal_note_once():
    from yue2_comfy.vendor.yue2_music import abc_tools

    for song in DATA["songs"]:
        text = song["abc"]["melody"]
        found = sections.sections(text)
        assert found and found[0]["start"] == 0.0
        assert sum(s["notes"] for s in found) == len(abc_tools.parse(text).voices["Vocal"].notes)
        assert all(s["tag"] in {"Intro", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Outro", "Interlude",
                                "Instrumental"} for s in found)


def test_the_marks_of_a_recording_and_its_mode_differ():
    recording = edits.audio_mark(b"samples", 44100)
    assert recording != edits.audio_mark(b"samples", 48000)
    assert recording != edits.audio_mark(b"sample5", 44100)
    assert edits.track_mark(recording, "melody") != edits.track_mark(recording, "full")
    assert len(recording) == 16 and edits.MARK_LINE.fullmatch("%yue2-words " + recording)
