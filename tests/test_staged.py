"""The staged nodes, proved on a machine that has never downloaded the model.

What these have to establish is narrow and important: that a score which was
not touched is sung exactly as the model wrote it, and that a score which was
touched is sung as edited. Everything else about these nodes is convenience;
that one distinction is the reason they exist.
"""

from __future__ import annotations

import contextlib
import json
import re

import pytest

from yue2_comfy import constants, edits, generate, phrasing, songs, staged, transpose

SCORE = "X:1\nK:C\nCDEF|\n"
IDS = [88, 58, 49]


class FakeTokenizer:
    """Encodes by code point, so a test can tell two scores apart by eye."""

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


class FakeLatents:
    """Stands in for the tensor, and remembers where it was asked to go."""

    def __init__(self):
        self.moved = []

    def cpu(self):
        self.moved.append("cpu")
        return self

    def to(self, device):
        self.moved.append(device)
        return self


class FakeModels:
    device = "cpu"
    lm = vae = None
    tokenizer = FakeTokenizer()


@contextlib.contextmanager
def fake_session(settings, unique_id, progress):
    """staged.session without the six gigabytes or the ComfyUI it talks to."""
    yield FakeModels()


def stub_score(monkeypatch, score=SCORE, ids=IDS):
    """Replace stage one and hand back every call it was asked to make."""
    calls = []

    def write_score(models, style, lyrics, seed, settings, progress=None,
                    cancelled=None, stages=None):
        calls.append({"style": style, "lyrics": lyrics, "seed": seed,
                      "settings": dict(settings), "stages": stages})
        return score, list(ids), {"abc": {"seconds": 0.5}}

    monkeypatch.setattr(staged, "session", fake_session)
    monkeypatch.setattr(generate, "write_score", write_score)
    return calls


def stub_singing(monkeypatch, seconds=42.0, performance=None):
    """Replace stages two to four and hand back every call they were asked to make."""
    calls = []

    def sing(models, style, lyrics, seed, settings, abc_ids=None, abc="",
             progress=None, cancelled=None, stages=None, tune_seconds=None):
        calls.append({"ids": None if abc_ids is None else list(abc_ids), "abc": abc,
                      "seed": seed, "settings": dict(settings), "stages": stages,
                      "tune_seconds": tune_seconds})
        return FakeLatents(), {"semantic": {}, "acoustic": {}}, performance

    def decode(models, latents, progress=None, cancelled=None, stages=None):
        calls.append({"decode": latents, "stages": stages})
        return "waveform", {"seconds_of_audio": seconds}

    monkeypatch.setattr(staged, "session", fake_session)
    monkeypatch.setattr(generate, "sing", sing)
    monkeypatch.setattr(generate, "decode", decode)
    return calls


def test_a_plan_carries_everything_the_render_node_needs(monkeypatch):
    stub_score(monkeypatch)
    plan, score = staged.YuE2Plan().plan(style="a style", lyrics="words", seed=7)["result"]
    assert score == SCORE
    assert plan["style"] == "a style"
    assert plan["lyrics"] == "words"
    assert plan["seed"] == 7
    assert plan["ids"] == IDS
    assert plan["score"] == SCORE
    assert plan["settings"]["cot"] == constants.DEFAULT_OPTIONS["cot"]


def test_a_plan_survives_being_saved_in_a_workflow(monkeypatch):
    """ComfyUI caches what travels on a link and writes it into the .json.

    A plan holding a tensor, a model handle or anything else live would work
    perfectly until somebody reopened the workflow the next morning.
    """
    stub_score(monkeypatch)
    plan, _score = staged.YuE2Plan().plan(style="a style", lyrics="words", seed=7)["result"]
    assert json.loads(json.dumps(plan)) == plan


def test_the_plan_nodes_hand_their_score_to_the_score_editor(monkeypatch):
    """They can run on their own, so the editor can ask for a score without a song.

    ComfyUI lets a partial run target only an output node. The render node stays
    an ordinary node and the batch node too, so queueing a plan node alone never
    sings. With the score goes the mark of the words it was written for, which
    the editor puts on an edit so the render node can tell when they move on, and
    the ceiling those words give a song, which the editor draws on the piano roll
    when they came in through a wire.
    """
    stub_score(monkeypatch)
    written = staged.YuE2Plan().plan(style="a style", lyrics="words", seed=7)
    cot = constants.DEFAULT_OPTIONS["cot"]
    assert written["ui"] == {staged.SCORE_UI: [SCORE],
                             staged.WORDS_UI: [edits.mark("a style", "words", cot)],
                             staged.AUTO_SECONDS_UI: [constants.auto_seconds("words")]}
    plans = [{"score": "first"},
             {"score": "second", "style": "s", "lyrics": "l", "settings": {"cot": "melody"}}]
    assert staged.YuE2SelectPlan().select(plans, 1)["ui"] == {
        staged.SCORE_UI: ["second"], staged.WORDS_UI: [edits.mark("s", "l", "melody")],
        staged.AUTO_SECONDS_UI: [constants.auto_seconds("l")]}
    assert staged.YuE2Plan.OUTPUT_NODE is True
    assert staged.YuE2SelectPlan.OUTPUT_NODE is True
    assert not getattr(staged.YuE2RenderPlan, "OUTPUT_NODE", False)
    assert not getattr(staged.YuE2PlanBatch, "OUTPUT_NODE", False)


def test_an_untouched_score_is_sung_as_the_model_wrote_it():
    """Not re-encoded from its own printed form: the ids go through as they are."""
    plan = {"score": SCORE, "ids": IDS}
    assert staged._chosen(plan, "") == (IDS, SCORE)
    assert staged._chosen(plan, SCORE) == (IDS, SCORE)
    assert staged._chosen(plan, "  " + SCORE + "  ") == (IDS, SCORE)
    assert staged._chosen(plan, edits.attach(SCORE, "0123456789abcdef")) == (IDS, SCORE)


def test_an_edited_score_is_sung_instead_of_the_ids():
    """The reason these nodes exist at all.

    The text arrives trimmed at the ends: a paste that brought a blank line
    with it would otherwise put tokens into the prompt that carry no music.
    """
    plan = {"score": SCORE, "ids": IDS}
    ids, text = staged._chosen(plan, "\nX:1\nK:G\nGABc|\n\n")
    assert ids is None, "the model's ids no longer describe this text"
    assert text == "X:1\nK:G\nGABc|"


def test_the_render_node_passes_an_edit_down_to_the_stage(monkeypatch):
    """An edit that stopped at the node would be the worst of both worlds."""
    calls = stub_singing(monkeypatch)
    plan = {"style": "s", "lyrics": "l", "seed": 5, "score": SCORE, "ids": IDS,
            "settings": dict(constants.DEFAULT_OPTIONS)}
    staged.YuE2RenderPlan().render(plan, score_abc="X:1\nK:G\n")
    assert calls[0]["ids"] is None
    assert calls[0]["abc"] == "X:1\nK:G"


def plain_plan():
    return {"style": "s", "lyrics": "l", "seed": 5, "score": SCORE, "ids": IDS,
            "settings": dict(constants.DEFAULT_OPTIONS)}


def test_the_render_node_sings_an_edit_made_for_its_plans_words(monkeypatch):
    calls = stub_singing(monkeypatch)
    plan = plain_plan()
    words = edits.mark("s", "l", plan["settings"]["cot"])
    staged.YuE2RenderPlan().render(plan, score_abc=edits.attach("X:1\nK:G\n", words))
    assert calls[0]["ids"] is None
    assert calls[0]["abc"] == "X:1\nK:G", "the mark must never reach the model"


def test_the_render_node_leaves_an_edit_for_other_words_unsung_and_says_so(monkeypatch):
    """New words under old notes is the outcome nobody asks for; the plan's own score is sung."""
    calls = stub_singing(monkeypatch)
    said = []
    monkeypatch.setattr(staged, "announce",
                        lambda node, findings, kind="notice": said.append(findings))
    plan = plain_plan()
    stale = edits.attach("X:1\nK:G\n", edits.mark("s", "other words", plan["settings"]["cot"]))
    staged.YuE2RenderPlan().render(plan, score_abc=stale, unique_id="9")
    assert calls[0]["ids"] == IDS
    assert [level for level, _message in said[0]] == ["warn"]
    assert staged.RENDER_INSTEAD in said[0][0][1]


def test_the_render_node_warns_when_a_melody_only_score_meets_cot_full(monkeypatch):
    """Sung anyway: the warning says what to change, it does not stop the song."""
    calls = stub_singing(monkeypatch)
    said = []
    monkeypatch.setattr(staged, "announce",
                        lambda node, findings, kind="notice": said.append(findings))
    plan = plain_plan()
    plan["settings"]["cot"] = "full"
    staged.YuE2RenderPlan().render(plan, score_abc="X:1\nK:G\nGABc|\n", unique_id="9")
    assert calls[0]["abc"] == "X:1\nK:G\nGABc|"
    assert said == [[("warn", edits.CHORDLESS)]]
    said.clear()
    staged.YuE2RenderPlan().render(plan, score_abc='X:1\nK:G\n"G"GABc|\n', unique_id="9")
    assert said == []


def test_the_render_node_warns_when_a_chorded_score_meets_cot_melody(monkeypatch):
    """The pairing the other way: 'melody' tells the model there are no chords to read."""
    calls = stub_singing(monkeypatch)
    said = []
    monkeypatch.setattr(staged, "announce",
                        lambda node, findings, kind="notice": said.append(findings))
    plan = plain_plan()
    plan["settings"]["cot"] = "melody"
    staged.YuE2RenderPlan().render(plan, score_abc='X:1\nK:G\n"G"GABc|\n', unique_id="9")
    assert calls[0]["abc"] == 'X:1\nK:G\n"G"GABc|'
    assert said == [[("warn", edits.CHORDED)]]
    said.clear()
    staged.YuE2RenderPlan().render(plan, score_abc="X:1\nK:G\nGABc|\n", unique_id="9")
    assert said == []


BARE_TUNE = ('X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=100\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
             'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\nV: Vocal\nz2C2D2E2F2G2A2B2|\nV: Ins\nZ|\n')


def test_the_render_node_lays_the_plans_lyrics_along_a_bare_tune(monkeypatch):
    """The same rule as the song node's: a tune that names no section gets the plan's words laid along it."""
    calls = stub_singing(monkeypatch)
    said = []
    monkeypatch.setattr(staged, "announce",
                        lambda node, findings, kind="notice": said.append(findings))
    plan = plain_plan()
    plan["lyrics"] = "[Chorus]\nla la la la la la la"
    plan["settings"]["cot"] = "melody"
    staged.YuE2RenderPlan().render(plan, score_abc=BARE_TUNE, unique_id="9")
    laid = phrasing.lay(BARE_TUNE, plan["lyrics"])
    assert calls[0]["ids"] is None
    assert calls[0]["abc"] == laid.score
    assert calls[0]["tune_seconds"] == laid.seconds
    assert [level for level, _message in said[0]] == ["notice"]


def test_a_song_laid_along_a_tune_is_let_sing_as_long_as_the_tune_and_a_little_over():
    """At 'max_seconds' 0 the ceiling comes from the tune, not from twelve seconds a line; a set value still wins."""
    assert generate.song_ceiling(0, "one line", 18.8) == phrasing.ceiling(18.8) == 22.7
    assert generate.song_ceiling(0, "one line", None) == constants.length_ceiling(0, "one line")
    assert generate.song_ceiling(30, "one line", 18.8) == 30.0
    assert generate.song_ceiling(0, "", 1000.0) == constants.MAX_SECONDS


def test_the_render_node_passes_an_untouched_score_as_ids(monkeypatch):
    calls = stub_singing(monkeypatch)
    plan = {"style": "s", "lyrics": "l", "seed": 5, "score": SCORE, "ids": IDS,
            "settings": dict(constants.DEFAULT_OPTIONS)}
    staged.YuE2RenderPlan().render(plan, score_abc="")
    assert calls[0]["ids"] == IDS


def test_the_render_node_returns_audio_and_latents_on_the_cpu(monkeypatch):
    """Latents sit on a link between two nodes, which may be a long time."""
    stub_singing(monkeypatch)
    plan = {"style": "s", "lyrics": "l", "seed": 5, "score": SCORE, "ids": IDS,
            "settings": dict(constants.DEFAULT_OPTIONS)}
    audio, latents = staged.YuE2RenderPlan().render(plan)
    assert set(audio) == {"waveform", "sample_rate"}
    assert audio["sample_rate"] == constants.SAMPLE_RATE
    assert latents["latents"].moved == ["cpu"]
    assert latents["seed"] == 5


def test_the_render_node_remembers_the_song_it_sang(monkeypatch):
    """The song goes to the song memory, and what it was sung from rides on the latents.

    That second part is what lets 'YuE2 Decode Latents' remember the same song
    decoded again, through the other decoder.
    """
    kept = []
    monkeypatch.setattr(songs, "keep", lambda *args: kept.append(args))
    stub_singing(monkeypatch, performance="sung from")
    plan = {"style": "s", "lyrics": "l", "seed": 5, "score": SCORE, "ids": IDS,
            "settings": dict(constants.DEFAULT_OPTIONS)}
    audio, latents = staged.YuE2RenderPlan().render(plan)
    assert len(kept) == 1
    given, origin, style, lyrics, seed, settings, score, performance = kept[0]
    assert given is audio
    assert (origin, style, lyrics, seed, score, performance) == (
        "YuE2 Render Plan", "s", "l", 5, SCORE, "sung from")
    assert settings["cot"] == constants.DEFAULT_OPTIONS["cot"]
    assert (latents["style"], latents["lyrics"], latents["score"], latents["performance"]) == (
        "s", "l", SCORE, "sung from")


def test_the_decode_node_remembers_the_song_under_its_own_decoder(monkeypatch):
    kept = []
    monkeypatch.setattr(songs, "keep", lambda *args: kept.append(args))
    stub_singing(monkeypatch)
    carried = {"latents": FakeLatents(), "settings": dict(constants.DEFAULT_OPTIONS),
               "seed": 5, "style": "s", "lyrics": "l", "score": SCORE,
               "performance": "sung from"}
    audio, = staged.YuE2DecodeLatents().decode(
        carried, options=dict(constants.DEFAULT_OPTIONS, vae="legacy"))
    given, origin, style, lyrics, seed, settings, score, performance = kept[0]
    assert given is audio
    assert (origin, style, lyrics, seed, score, performance) == (
        "YuE2 Decode Latents", "s", "l", 5, SCORE, "sung from")
    assert settings["vae"] == "legacy"


def test_latents_from_before_the_song_memory_are_decoded_and_not_remembered(monkeypatch):
    """A link cached by an older version carries no performance, and keep is told None."""
    kept = []
    monkeypatch.setattr(songs, "keep", lambda *args: kept.append(args))
    stub_singing(monkeypatch)
    audio, = staged.YuE2DecodeLatents().decode(
        {"latents": FakeLatents(), "settings": dict(constants.DEFAULT_OPTIONS)})
    assert audio["sample_rate"] == constants.SAMPLE_RATE
    assert kept[0][-1] is None


def test_the_decode_node_moves_the_latents_to_wherever_the_model_is(monkeypatch):
    """They arrive on the CPU, because that is where a link keeps them."""
    calls = stub_singing(monkeypatch)
    tensor = FakeLatents()
    audio, = staged.YuE2DecodeLatents().decode(
        {"latents": tensor, "settings": dict(constants.DEFAULT_OPTIONS)})
    assert audio["sample_rate"] == constants.SAMPLE_RATE
    assert calls[0]["decode"] is tensor
    assert tensor.moved == [FakeModels.device]


def test_options_on_the_render_node_win_over_the_plan_except_for_cot():
    """cot decides what the model was told before it wrote the score.

    Honouring a change here would sing a melody-only score under the
    instructions written for a chord-annotated one.
    """
    plan = {"settings": dict(constants.DEFAULT_OPTIONS, cot="melody",
                             temperature=1.0)}
    settings = staged._settings(plan, {"cot": "full", "temperature": 0.5}, None)
    assert settings["cot"] == "melody"
    assert settings["temperature"] == 0.5


def test_a_render_node_with_nothing_connected_uses_the_plans_own_settings():
    plan = {"settings": dict(constants.DEFAULT_OPTIONS, cot="melody",
                             max_seconds=120.0)}
    settings = staged._settings(plan, None, None)
    assert settings["cot"] == "melody"
    assert settings["max_seconds"] == 120.0


def test_a_batch_writes_consecutive_seeds(monkeypatch):
    calls = stub_score(monkeypatch)
    plans, scores = staged.YuE2PlanBatch().batch(
        style="s", lyrics="l", seed=100, count=3)
    assert [call["seed"] for call in calls] == [100, 101, 102]
    assert [plan["seed"] for plan in plans] == [100, 101, 102]
    assert "seed 101" in scores


def test_a_batch_gives_every_take_its_own_slice_of_the_bar(monkeypatch):
    """Four scores on one bar that restarts four times would read as a hang."""
    calls = stub_score(monkeypatch)
    staged.YuE2PlanBatch().batch(style="s", lyrics="l", seed=1, count=4)
    bands = [call["stages"][0] for call in calls]
    assert bands[0][0] == 0.0
    assert bands[-1][1] == 100.0
    for earlier, later in zip(bands, bands[1:]):
        assert earlier[1] == later[0]


def test_the_selector_takes_the_score_that_was_asked_for():
    plans = [{"score": "first"}, {"score": "second"}, {"score": "third"}]
    plan, score = staged.YuE2SelectPlan().select(plans, 1)["result"]
    assert score == "second"
    assert plan is plans[1]


def test_the_selector_clamps_rather_than_stopping_the_run():
    """Lowering 'count' and forgetting the index should not cost a whole graph."""
    plans = [{"score": "first"}, {"score": "second"}]
    _plan, score = staged.YuE2SelectPlan().select(plans, 9)["result"]
    assert score == "second"


def test_an_empty_batch_is_refused():
    with pytest.raises(ValueError):
        staged.YuE2SelectPlan().select([], 0)


def test_planning_with_cot_off_says_why_there_is_nothing_to_plan():
    options = dict(constants.DEFAULT_OPTIONS, cot="off")
    for node, call in ((staged.YuE2Plan(), "plan"), (staged.YuE2PlanBatch(), "batch")):
        kwargs = {"style": "s", "lyrics": "l", "seed": 1, "options": options}
        if call == "batch":
            kwargs["count"] = 2
        with pytest.raises(ValueError) as error:
            getattr(node, call)(**kwargs)
        assert "cot" in str(error.value)


def test_a_render_node_with_no_plan_says_what_to_connect():
    with pytest.raises(ValueError) as error:
        staged.YuE2RenderPlan().render(None)
    assert "YuE2 Plan" in str(error.value)


def test_a_decode_node_with_no_latents_says_what_to_connect():
    with pytest.raises(ValueError) as error:
        staged.YuE2DecodeLatents().decode({})
    assert "YuE2 Render Plan" in str(error.value)


def test_singing_with_no_score_at_all_says_what_to_connect():
    """Checked before the torch-backed imports, so it works on a bare machine."""
    with pytest.raises(ValueError) as error:
        generate.sing(FakeModels(), "s", "l", 1, dict(constants.DEFAULT_OPTIONS))
    assert "YuE2 Plan" in str(error.value)


def test_the_single_node_still_walks_the_same_three_stages(monkeypatch):
    """One pipeline, not two.

    The staged nodes are worth having only while they are the same code the
    plain node runs. This pins the seam: stage one's ids reach stage two, and
    stage two's latents reach stage three.
    """
    seen = []

    def write_score(models, style, lyrics, seed, settings, progress=None,
                    cancelled=None, stages=None):
        seen.append("write_score")
        return SCORE, IDS, {"abc": {"seconds": 1.0}}

    def sing(models, style, lyrics, seed, settings, abc_ids=None, abc="",
             progress=None, cancelled=None, stages=None, tune_seconds=None):
        seen.append(("sing", list(abc_ids or [])))
        return "latents", {"semantic": {}, "acoustic": {}}, "performance"

    def decode(models, latents, progress=None, cancelled=None, stages=None):
        seen.append(("decode", latents))
        return "waveform", {"seconds_of_audio": 12.0}

    monkeypatch.setattr(generate, "write_score", write_score)
    monkeypatch.setattr(generate, "sing", sing)
    monkeypatch.setattr(generate, "decode", decode)

    waveform, score, written, timing, performance = generate.run(
        FakeModels(), "s", "l", 3, dict(constants.DEFAULT_OPTIONS))
    assert seen == ["write_score", ("sing", IDS), ("decode", "latents")]
    assert (waveform, score, written) == ("waveform", SCORE, SCORE)
    assert performance == "performance", "what stage two sang from reaches the node"
    assert timing["abc"]["seconds"] == 1.0
    assert timing["seconds_of_audio"] == 12.0
    assert "total_seconds" in timing


def test_the_single_node_sings_an_edit_without_writing_a_score(monkeypatch):
    """Stage one is skipped, and the three stages left fill the bar by themselves."""
    seen = []

    def write_score(models, style, lyrics, seed, settings, progress=None,
                    cancelled=None, stages=None):
        seen.append("write_score")
        return SCORE, IDS, {"abc": {"seconds": 1.0}}

    def sing(models, style, lyrics, seed, settings, abc_ids=None, abc="",
             progress=None, cancelled=None, stages=None, tune_seconds=None):
        seen.append(("sing", abc_ids, abc, stages))
        return "latents", {"semantic": {}, "acoustic": {}}, None

    def decode(models, latents, progress=None, cancelled=None, stages=None):
        seen.append(("decode", stages))
        return "waveform", {"seconds_of_audio": 12.0}

    monkeypatch.setattr(generate, "write_score", write_score)
    monkeypatch.setattr(generate, "sing", sing)
    monkeypatch.setattr(generate, "decode", decode)

    edit = MOVABLE.replace('"C"C8E8G8c8|', '"C"E8G8c8e8|')
    _waveform, sung, written, timing, _performance = generate.run(
        FakeModels(), "s", "l", 3, dict(constants.DEFAULT_OPTIONS), edited=edit)
    bands = generate.alone(generate.Stages.SEMANTIC, generate.Stages.ACOUSTIC,
                           generate.Stages.DECODE)
    assert seen == [("sing", None, edit, bands[:2]), ("decode", bands[2:])]
    assert (sung, written) == (edit, "")
    assert "abc" not in timing


def test_a_staged_node_owns_the_whole_progress_bar():
    bands = generate.alone(generate.Stages.SEMANTIC, generate.Stages.ACOUSTIC,
                           generate.Stages.DECODE)
    assert bands[0][0] == 0.0
    assert bands[-1][1] == 100.0
    for earlier, later in zip(bands, bands[1:]):
        assert earlier[1] == later[0]
    assert [band[2] for band in bands] == [generate.Stages.SEMANTIC[2],
                                           generate.Stages.ACOUSTIC[2],
                                           generate.Stages.DECODE[2]]


def test_the_rescaled_bands_keep_the_proportions_of_a_whole_run():
    """The long stage in a full run has to stay the long one on its own bar."""
    full = generate.Stages
    span = full.DECODE[1] - full.SEMANTIC[0]
    share = (full.SEMANTIC[1] - full.SEMANTIC[0]) / span
    bands = generate.alone(full.SEMANTIC, full.ACOUSTIC, full.DECODE)
    assert bands[0][1] == pytest.approx(100.0 * share)


def test_one_stage_alone_fills_the_bar():
    assert generate.alone(generate.Stages.ABC) == (
        (0.0, 100.0, generate.Stages.ABC[2]),)


MOVABLE = ('X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=90\n'
           'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
           'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
           'K:C\n% verse\nV: Vocal\n"C"C8E8G8c8|\nV: Ins\nZ|\n')
"""A score in the dialect the model writes, which the toy SCORE above is not."""


def moving_plan(score=MOVABLE):
    return {"style": "s", "lyrics": "l", "seed": 5, "score": score, "ids": IDS,
            "settings": dict(constants.DEFAULT_OPTIONS)}


def test_the_render_node_moves_the_plans_score_before_singing_it(monkeypatch):
    """Moved text goes down, not the plan's ids, which describe the score before the move."""
    calls = stub_singing(monkeypatch)
    options = dict(constants.DEFAULT_OPTIONS, transpose=2)
    staged.YuE2RenderPlan().render(moving_plan(), options=options)
    assert calls[0]["ids"] is None
    assert calls[0]["abc"] == transpose.move(MOVABLE, 2).text
    assert "\nK:D\n" in calls[0]["abc"]


def test_the_render_node_moves_a_pasted_score_too(monkeypatch):
    calls = stub_singing(monkeypatch)
    pasted = MOVABLE.replace('"C"C8E8G8c8|', '"C"E8G8c8e8|')
    options = dict(constants.DEFAULT_OPTIONS, transpose=-3)
    staged.YuE2RenderPlan().render(moving_plan(), score_abc=pasted, options=options)
    assert calls[0]["abc"] == transpose.move(pasted, -3).text


def test_a_plan_carries_its_move_to_the_render_node(monkeypatch):
    """Settings travel with the plan, and 'transpose' is one of them."""
    calls = stub_singing(monkeypatch)
    plan = moving_plan()
    plan["settings"]["transpose"] = 5
    staged.YuE2RenderPlan().render(plan)
    assert calls[0]["abc"] == transpose.move(MOVABLE, 5).text


def test_a_score_that_cannot_be_moved_is_refused_before_the_model_loads(monkeypatch):
    entered = []

    @contextlib.contextmanager
    def session(settings, unique_id, progress):
        entered.append(True)
        yield FakeModels()

    monkeypatch.setattr(staged, "session", session)
    options = dict(constants.DEFAULT_OPTIONS, transpose=2)
    with pytest.raises(ValueError) as error:
        staged.YuE2RenderPlan().render(moving_plan(SCORE), options=options)
    assert "'transpose' to 0" in str(error.value)
    assert entered == []


def test_the_single_node_sings_the_moved_score_and_hands_it_back(monkeypatch):
    """score_abc is the score the song was sung from, so here it is the moved one."""
    seen = {}

    def write_score(models, style, lyrics, seed, settings, progress=None,
                    cancelled=None, stages=None):
        return MOVABLE, IDS, {"abc": {"seconds": 1.0}}

    def sing(models, style, lyrics, seed, settings, abc_ids=None, abc="",
             progress=None, cancelled=None, stages=None, tune_seconds=None):
        seen.update(ids=abc_ids, abc=abc)
        return "latents", {"semantic": {}, "acoustic": {}}, None

    def decode(models, latents, progress=None, cancelled=None, stages=None):
        return "waveform", {"seconds_of_audio": 12.0}

    monkeypatch.setattr(generate, "write_score", write_score)
    monkeypatch.setattr(generate, "sing", sing)
    monkeypatch.setattr(generate, "decode", decode)

    settings = dict(constants.DEFAULT_OPTIONS, transpose=-5)
    _waveform, score, written, _timing, _performance = generate.run(
        FakeModels(), "s", "l", 3, settings)
    moved = transpose.move(MOVABLE, -5).text
    assert seen == {"ids": None, "abc": moved}
    assert score == moved
    assert written == MOVABLE, "the editor edits the score as written, and the move goes on top"


def test_moving_with_cot_off_says_there_is_no_score_to_move():
    with pytest.raises(ValueError) as error:
        generate.moved("", 2, "off")
    assert "'cot'" in str(error.value)


REAL = ('X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=90\n'
        'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
        'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
        'K:C\n% verse\nV: Vocal\nC8D8E8G8|E8F8G8c8|\nV: Ins\nZ|Z|\n')
"""Two bars of 4/4 at 90 BPM: eight quarter notes, five and a third seconds."""


@pytest.mark.parametrize("style,expected", [
    ("Russian, soft female voice, disco, 127 BPM", "127 BPM and the score is written at 90 BPM"),
    ("Russian, soft female voice, disco, 90 BPM", ""),
    ("Russian, soft female voice, disco, 92 BPM", ""),
    ("Russian, soft female voice, disco", ""),
    ("fast, 180 bpm", "180 BPM and the score is written at 90 BPM"),
])
def test_a_style_that_argues_with_the_score_about_the_tempo_is_reported(style, expected):
    """Measured on a real cover: the model follows the style and the score drifts away from it."""
    said = generate.tempo_clash(style, REAL)
    assert (expected in said) if expected else (said == "")
    if expected:
        assert ("slower" in said) == (int(re.search(r"(\d+)", style.split(",")[-1]).group(1)) < 90)


def test_a_style_beside_a_score_the_roll_cannot_read_says_nothing():
    assert generate.tempo_clash("pop, 120 BPM", "X:1\nK:C\nCDEF|\n") == ""
    assert generate.tempo_clash("pop, 120 BPM", "") == ""


def test_a_song_much_shorter_than_its_score_says_so():
    assert generate.ended_early(REAL, 5.34, 360) == ""
    said = generate.ended_early(REAL, 2.0, 360)
    assert "0:02" in said and "0:05" in said
    assert "stopped there by itself" in said


def test_a_song_stopped_by_the_length_limit_blames_the_limit_and_not_the_model():
    said = generate.ended_early(REAL, 2.0, 2.0)
    assert "length limit" in said and "max_seconds" in said


def test_a_score_nothing_can_read_ends_no_song_early():
    assert generate.ended_early("X:1\nK:C\nCDEF|\n", 1.0, 360) == ""
