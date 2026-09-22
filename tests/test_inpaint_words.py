"""Changing the words of a song: the rewrite, the line the aligner places, and the edit that sings it.

Nothing here needs a model. The forced aligner's answer is a list of times,
so the arithmetic on it is checked with times written by hand, and the parts
of the aligner that are not the model -- the config it is built from and the
rule that sorts its answer -- are checked on their own.
"""

import json

import pytest

from yue2_comfy.inpaint import lines, ops, track

LF = chr(10)

SONG = LF.join([
    "[verse]",
    "the road is long",
    "the night is cold",
    "",
    "[chorus]",
    "we rise up tonight",
    "we rise up again",
])


def test_a_line_is_rewritten_and_everything_else_is_the_text_it_was():
    """Only the lines chosen change, to the character: a song is mostly what it already says."""
    made = ops.change_words(SONG, 2, 3, "the night is gold")
    assert made.text == SONG.replace("the night is cold", "the night is gold")
    assert made.before == ("the night is cold",)
    assert made.after == ("the night is gold",)
    assert made.tag == "verse"
    assert made.syllables == (4, 4)


def test_two_lines_can_become_one_and_one_can_become_two():
    """The number of lines is the person's business; the tune under them does not move."""
    together = ops.change_words(SONG, 5, 7, "we rise up tonight and again")
    assert together.text.splitlines()[5:] == ["we rise up tonight and again"]
    apart = ops.change_words(SONG, 1, 2, "the road is long" + LF + "and so is the night")
    assert apart.text.splitlines()[1:3] == ["the road is long", "and so is the night"]
    assert apart.after == ("the road is long", "and so is the night")


def test_the_lines_that_stay_keep_the_line_endings_they_had():
    """A song written by a browser has CRLF in it, and a rewrite is not a reason to convert a file."""
    windows = SONG.replace(LF, chr(13) + LF)
    made = ops.change_words(windows, 2, 3, "the night is gold")
    assert made.text.count(chr(13) + LF) == windows.count(chr(13) + LF)
    assert "the night is gold" + chr(13) + LF in made.text


def test_a_rewrite_that_reaches_the_last_line_adds_no_line_ending_of_its_own():
    made = ops.change_words(SONG, 6, 7, "we rise up once more")
    assert made.text.endswith("we rise up once more")


@pytest.mark.parametrize("first,stop,said,why", [
    (0, 1, "anything", "section tag"),
    (3, 4, "anything", "Nothing is sung"),
    (2, 3, "", "say nothing"),
    (2, 3, "   ", "say nothing"),
    (2, 3, "[chorus]", "name a section"),
    (2, 99, "anything", "not lines of these words"),
    (2, 2, "anything", "not lines of these words"),
])
def test_what_a_rewrite_will_not_do(first, stop, said, why):
    """Each refusal says which it is, because a person reads it in a window and has to act on it."""
    with pytest.raises(ValueError) as raised:
        ops.change_words(SONG, first, stop, said)
    assert why in str(raised.value)


def test_the_words_of_a_song_are_counted_the_way_the_aligner_counts_them():
    """The times come back one pair a word, so both sides have to agree on what a word is."""
    assert lines.sung_lines(SONG) == [
        (1, ["the", "road", "is", "long"]),
        (2, ["the", "night", "is", "cold"]),
        (5, ["we", "rise", "up", "tonight"]),
        (6, ["we", "rise", "up", "again"]),
    ]
    assert lines.heard_text(SONG).splitlines() == [
        "the road is long", "the night is cold", "we rise up tonight", "we rise up again"]
    assert lines.sung_lines("[verse]" + LF + "Don't, said he -- twice!") == [
        (1, ["Don't", "said", "he", "twice"])]


def made_times(text, step=0.5, length=0.4):
    """A time for every word of ``text``, half a second apart, as the aligner would answer."""
    at = 0.0
    found = []
    for _number, words in lines.sung_lines(text):
        for word in words:
            found.append((word, at, at + length))
            at += step
    return found


def test_a_line_is_sung_from_the_last_word_of_the_line_before_it():
    """The stand measured it: opening on the line's own first word left the model singing late."""
    times = made_times(SONG)
    start, stop = lines.region(SONG, times, 2, 3)
    assert start == pytest.approx(1.5 - lines.EARLY)
    assert stop == pytest.approx(4.0 - lines.EARLY)


def test_the_first_line_has_nothing_to_run_in_from_and_the_last_runs_to_the_end():
    times = made_times(SONG)
    assert lines.region(SONG, times, 1, 2)[0] == pytest.approx(0.0)
    assert lines.region(SONG, times, 6, 7)[1] is None


def test_times_measured_on_other_words_are_refused_rather_than_read_wrong():
    """One pair a word, in order: a list of another length is another song's list."""
    with pytest.raises(ValueError) as raised:
        lines.placed(SONG, made_times(SONG)[:-1])
    assert "not these" in str(raised.value)


def test_a_words_edit_is_read_written_and_named_by_what_it_sings():
    """The list holds the lines and the words; the name a take is kept under holds them too."""
    written = json.dumps([{"op": "words", "lines": [2, 3], "text": "the night is gold"}])
    edit = track.read(written, 2)[0]
    assert (edit.op, edit.lines, edit.text) == ("words", (2, 3), "the night is gold")
    assert edit.takes == 2 and edit.bars is None and edit.seconds is None
    assert track.needs_times(edit) is True
    assert json.loads(track.written((edit,)))[0]["text"] == "the night is gold"
    other = track.read(json.dumps([{"op": "words", "lines": [2, 3], "text": "the night is old"}]),
                       2)[0]
    assert track.name("song", [], edit) != track.name("song", [], other), (
        "two rewrites of one line would share their takes")


def test_a_words_edit_that_names_bars_needs_no_hearing():
    """Selecting the stretch is the cheap path: the bars say where, and nothing is measured."""
    edit = track.read(json.dumps([{"op": "words", "bars": [4, 8], "lines": [2, 3],
                                   "text": "the night is gold"}]), 1)[0]
    assert edit.bars == (4, 8)
    assert track.needs_times(edit) is False


def test_the_song_after_a_words_edit_says_the_new_words():
    state = track.State(lyrics=SONG, score="", sheet=None, clock=None, frames=500)
    edit = track.read(json.dumps([{"op": "words", "lines": [2, 3],
                                   "text": "the night is gold"}]), 1)[0]
    step = track.plan(state, edit, span=(120, 190))
    assert step.kind == "words"
    assert (step.start, step.stop) == (120, 190)
    assert "the night is gold" in step.lyrics and "the night is cold" not in step.lyrics
    after = track.after(state, step, 74)
    assert after.lyrics == step.lyrics
    assert after.frames == 500 - 70 + 74


def test_a_line_that_will_not_fit_its_tune_is_said_so_rather_than_refused():
    state = track.State(lyrics=SONG, score="", sheet=None, clock=None, frames=500)
    edit = track.read(json.dumps([{"op": "words", "lines": [2, 3],
                                   "text": "the night is cold and long and dark and very old"}]),
                      1)[0]
    notices = track.plan(state, edit, span=(120, 190)).notices
    assert notices and notices[0][0] == "notice" and "syllables" in notices[0][1]
    same = track.read(json.dumps([{"op": "words", "lines": [2, 3], "text": "the night is gold"}]),
                      1)[0]
    assert track.plan(state, same, span=(120, 190)).notices == ()


def test_an_edit_with_no_bars_and_no_span_says_what_is_missing():
    state = track.State(lyrics=SONG, score="", sheet=None, clock=None, frames=500)
    edit = track.read(json.dumps([{"op": "words", "lines": [2, 3], "text": "gold"}]), 1)[0]
    with pytest.raises(ValueError) as raised:
        track.plan(state, edit)
    assert track.NO_TIMES == str(raised.value)


def test_the_sound_the_edits_have_made_has_a_name_of_its_own():
    """Word times are measured on a sound, so they are kept under what the edits before made."""
    edit = track.read(json.dumps([{"op": "retake", "bars": [4, 8], "seed": 3}]), 1)[0]
    assert track.sound_name("song", []) == track.sound_name("song", [])
    assert track.sound_name("song", []) != track.sound_name("other", [])
    assert track.sound_name("song", []) != track.sound_name("song", [(edit, 3)])
    assert track.sound_name("song", [(edit, 3)]) != track.sound_name("song", [(edit, 4)])


def test_the_aligner_sorts_an_answer_that_came_back_out_of_order():
    """Qwen's own rule: keep the longest rising run, fill a short gap from the nearer side."""
    pytest.importorskip("torch")
    from yue2_comfy.asr import aligner

    assert aligner.repair([]) == []
    assert aligner.repair([80, 160, 240]) == [80, 160, 240]
    assert aligner.repair([80, 4000, 160, 240]) == [80, 80, 160, 240]
    assert aligner.repair([80, 160, 0, 400]) == [80, 160, 160, 400]
    assert aligner.repair([80, 160, 240, 320, 0, 0, 0, 800, 880, 960, 1040]) == [
        80, 160, 240, 320, 440, 560, 680, 800, 880, 960, 1040]
    assert aligner.repair([500, 80, 160]) == [80, 80, 160]


def test_the_aligner_is_built_from_the_sizes_its_own_config_gives(tmp_path):
    """The network is the speech model's; a release of another shape is refused rather than loaded."""
    pytest.importorskip("torch")
    from yue2_comfy.asr import aligner, network

    config = {"timestamp_token_id": 151705, "timestamp_segment_time": 80,
              "thinker_config": {"model_type": "qwen3_forced_aligner", "classify_num": 5000,
                                 "text_config": {"hidden_size": 1024, "intermediate_size": 3072,
                                                 "vocab_size": 152064, "num_hidden_layers": 28,
                                                 "num_attention_heads": network.HEADS,
                                                 "num_key_value_heads": network.KV_HEADS,
                                                 "head_dim": network.HEAD}}}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    sizes = aligner.sizes_of(str(tmp_path))
    assert sizes["WIDTH"] == 1024 and sizes["LAYERS"] == 28 and sizes["classify"] == 5000
    assert sizes["stamp"] == 151705 and sizes["step_ms"] == 80

    was = (network.WIDTH, network.FFN, network.VOCAB, network.LAYERS)
    with aligner._shaped(sizes):
        assert (network.WIDTH, network.FFN) == (1024, 3072)
    assert (network.WIDTH, network.FFN, network.VOCAB, network.LAYERS) == was

    config["thinker_config"]["text_config"]["num_attention_heads"] = 3
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        aligner.sizes_of(str(tmp_path))
    assert "attention shape" in str(raised.value)


def test_the_widths_go_back_even_when_the_network_will_not_build():
    """Nothing else may be built while they are swapped, so they may not be left swapped."""
    pytest.importorskip("torch")
    from yue2_comfy.asr import aligner, network

    was = network.WIDTH
    with pytest.raises(RuntimeError):
        with aligner._shaped({"WIDTH": 1, "FFN": 2, "VOCAB": 3, "LAYERS": 4}):
            raise RuntimeError("the weights did not fit")
    assert network.WIDTH == was


def test_a_whole_audio_value_is_heard_as_its_first_recording():
    """The node hands the aligner what a workflow gave it, which is a batch of recordings.

    A song off the disk is ``[channels, samples]`` and a song off a wire is
    ``[recordings, channels, samples]``; the ears take either, because a node
    that times the words of a sung line should not care which it was handed.
    """
    pytest.importorskip("torch")
    import torch

    from yue2_comfy.asr import model

    loud = torch.zeros(1, 2, 16000)
    loud[0, 0, :] = 1.0
    heard = model.mono_16k(loud, 16000)
    assert heard.shape == (16000,)
    assert float(heard[0]) == 0.5
    assert model.mono_16k(loud[0], 16000).shape == (16000,)
    assert model.mono_16k(loud[0, 0], 16000).shape == (16000,)


def test_a_planned_change_of_words_carries_the_lines_it_swapped():
    """The picker shows an edit by the stretch it took, and one stretch looks like another.

    What tells a change of words from a retake of the same seconds is the
    words themselves, so the step hands them on and the song remembers them.
    """
    state = track.State(lyrics=SONG, score="", sheet=None, clock=None, frames=500)
    edit = track.read(json.dumps([{"op": "words", "lines": [2, 3],
                                   "text": "the night is gold"}]), 1)[0]
    step = track.plan(state, edit, span=(120, 190))
    assert step.was == ("the night is cold",)
    assert step.now == ("the night is gold",)
    plain = track.plan(state, track.read('[{"op": "retake", "seconds": [1, 3]}]', 1)[0],
                       span=(120, 190))
    assert (plain.was, plain.now) == ((), ()), "a retake swapped nothing"


@pytest.mark.parametrize("said,found", [
    ("I can smell the ocean drifting slow", 7),
    ("Open door. I can smell the ocean drifting slow.", 7),
    ("I can hear the city waking slow", 4),
    ("slow drifting ocean the smell can I", 1),
    ("", 0),
])
def test_a_take_is_scored_by_the_words_asked_for_that_are_heard_in_order(said, found):
    """The stand's own measure: the line before, sung again as the run-up, costs nothing; words
    out of their order are not words sung; and the old line shares words with the new one."""
    assert lines.heard("I can smell the ocean drifting slow", said) == (found, 7)


def test_what_is_heard_is_compared_without_case_punctuation_or_yo():
    assert lines.heard("\u0401\u043b\u043a\u0430, \u043d\u0430\u0448\u0430!",
                       "\u0435\u043b\u043a\u0430 \u041d\u0430\u0448\u0430") == (2, 2)
    assert lines.heard("don't stop", "Don't stop.") == (2, 2)
    assert lines.heard("", "anything at all") == (0, 0)


def test_a_retake_is_heard_against_the_words_whose_middle_it_holds():
    """A word cut across an end of the stretch belongs to the side that sings most of it."""
    times = [("we", 0.0, 0.4), ("rise", 0.5, 1.5), ("up", 1.6, 2.0), ("again", 2.2, 3.4)]
    assert lines.within(times, 0.9, 2.9) == "rise up again"
    assert lines.within(times, 1.1, 2.7) == "up", "rise is sung mostly before, again after"
    assert lines.within(times, 5.0, 9.0) == "", "nothing is sung there"
    assert lines.within([], 0.0, 9.0) == ""


def test_the_lines_before_a_change_of_words_keep_their_places_under_the_new_numbers():
    """The song as it was is drawn beside the takes, while the words panel shows the words after.

    A line the change did not touch is the same line under a new number; one
    it rewrote is given the stretch of the line it replaced.
    """
    spans = [(1, 0.0, 1.0), (2, 1.2, 2.0), (5, 3.0, 4.0), (6, 4.5, 5.5)]
    same = ops.change_words(SONG, 2, 3, "the night is gold").text
    assert lines.carried(spans, 2, 3, same, 1) == spans
    longer = ops.change_words(SONG, 2, 3, "the night" + LF + "is gold").text
    assert lines.carried(spans, 2, 3, longer, 2) == [
        (1, 0.0, 1.0), (2, 1.2, 2.0), (3, 1.2, 2.0), (6, 3.0, 4.0), (7, 4.5, 5.5)]
    shorter = ops.change_words(SONG, 5, 7, "we rise").text
    assert lines.carried(spans, 5, 7, shorter, 1) == [(1, 0.0, 1.0), (2, 1.2, 2.0), (5, 3.0, 5.5)]
    two = ops.change_words(SONG, 1, 3, "the road" + LF + "the night").text
    assert lines.carried(spans, 1, 3, two, 2) == spans, "as many lines on each side: one for one"


def test_a_take_heard_singing_its_words_is_kept_over_one_that_joins_better():
    """The join is the model's opinion of how the old song goes on; the words are the edit."""
    core = pytest.importorskip("yue2_comfy.inpaint.core")

    def take(join, heard=None):
        return core.Take(seed=0, waveform=None, song=None, count=1, join=join, joins={},
                         ended=False, timing={}, heard=heard)

    assert core.best([take(-3.0, (8, 10)), take(-4.0, (10, 10))]) == 1
    assert core.best([take(-4.0, (9, 10)), take(-3.0, (9, 10))]) == 1, "a tie goes to the join"
    assert core.best([take(-4.0), take(-3.0)]) == 1, "nothing heard: the join, as before"
    assert core.best([take(-3.0), take(-4.0, (1, 10))]) == 1, "a take heard beats one not heard"
    assert core.best([None, take(None, (5, 10)), take(None, (6, 10))]) == 2
    assert core.best([take(None), take(None)]) == 0


def test_short_clips_are_heard_in_the_language_of_the_first(monkeypatch):
    """The song as it was is heard first, and its language is the one every take is heard in."""
    import sys
    import types

    from yue2_comfy.asr import runtime as asr_runtime

    asked, gone = [], []

    def recognise(net, tokenizer, audio, language="", cancelled=None, progress=None):
        asked.append((audio, language))
        return {"language": language or "Russian", "text": "heard " + audio}

    fake_model = types.SimpleNamespace(recognise=recognise)
    monkeypatch.setitem(sys.modules, "yue2_comfy.asr.model", fake_model)
    monkeypatch.setattr(sys.modules["yue2_comfy.asr"], "model", fake_model, raising=False)
    net = types.SimpleNamespace(forget_steps=lambda: gone.append(1))
    monkeypatch.setattr(asr_runtime, "acquire", lambda folder, device, progress=None: (net, "tok"))
    monkeypatch.setattr(asr_runtime, "stamp", lambda folder: ("weights",))
    monkeypatch.setattr(asr_runtime, "_HEARD", asr_runtime.collections.OrderedDict())
    answers = asr_runtime.hear("f", "d", [("was", "a"), ("take 1", "b"), ("take 2", "c")])
    assert asked == [("a", ""), ("b", "Russian"), ("c", "Russian")]
    assert [answer["text"] for answer in answers] == ["heard a", "heard b", "heard c"]
    assert gone == [1], "the captured step is let go once the clips are heard"
    asked.clear()
    again = asr_runtime.hear("f", "d", [("was", "a"), ("take 3", "d")])
    assert asked == [("d", "Russian")], "a clip heard before is not heard again"
    assert again[0]["text"] == "heard a"


def test_hearing_stops_when_the_run_is_cancelled(monkeypatch):
    from yue2_comfy.asr import runtime as asr_runtime

    monkeypatch.setattr(asr_runtime, "stamp", lambda folder: ("weights",))
    with pytest.raises(InterruptedError):
        asr_runtime.hear("f", "d", [("was", "a")], cancelled=lambda: True)
