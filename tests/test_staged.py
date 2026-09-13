"""The staged nodes, proved on a machine that has never downloaded the model.

What these have to establish is narrow and important: that a score which was
not touched is sung exactly as the model wrote it, and that a score which was
touched is sung as edited. Everything else about these nodes is convenience;
that one distinction is the reason they exist.
"""

from __future__ import annotations

import contextlib
import json

import pytest

from yue2_comfy import constants, generate, staged, transpose

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


def stub_singing(monkeypatch, seconds=42.0):
    """Replace stages two to four and hand back every call they were asked to make."""
    calls = []

    def sing(models, style, lyrics, seed, settings, abc_ids=None, abc="",
             progress=None, cancelled=None, stages=None):
        calls.append({"ids": None if abc_ids is None else list(abc_ids), "abc": abc,
                      "seed": seed, "settings": dict(settings), "stages": stages})
        return FakeLatents(), {"semantic": {}, "acoustic": {}}

    def decode(models, latents, progress=None, cancelled=None, stages=None):
        calls.append({"decode": latents, "stages": stages})
        return "waveform", {"seconds_of_audio": seconds}

    monkeypatch.setattr(staged, "session", fake_session)
    monkeypatch.setattr(generate, "sing", sing)
    monkeypatch.setattr(generate, "decode", decode)
    return calls


def test_a_plan_carries_everything_the_render_node_needs(monkeypatch):
    stub_score(monkeypatch)
    plan, score = staged.YuE2Plan().plan(style="a style", lyrics="words", seed=7)
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
    plan, _score = staged.YuE2Plan().plan(style="a style", lyrics="words", seed=7)
    assert json.loads(json.dumps(plan)) == plan


def test_an_untouched_score_is_sung_as_the_model_wrote_it():
    """Not re-encoded from its own printed form: the ids go through as they are."""
    plan = {"score": SCORE, "ids": IDS}
    assert staged._chosen(plan, "") == (IDS, SCORE)
    assert staged._chosen(plan, SCORE) == (IDS, SCORE)
    assert staged._chosen(plan, "  " + SCORE + "  ") == (IDS, SCORE)


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
    plan, score = staged.YuE2SelectPlan().select(plans, 1)
    assert score == "second"
    assert plan is plans[1]


def test_the_selector_clamps_rather_than_stopping_the_run():
    """Lowering 'count' and forgetting the index should not cost a whole graph."""
    plans = [{"score": "first"}, {"score": "second"}]
    _plan, score = staged.YuE2SelectPlan().select(plans, 9)
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
             progress=None, cancelled=None, stages=None):
        seen.append(("sing", list(abc_ids or [])))
        return "latents", {"semantic": {}, "acoustic": {}}

    def decode(models, latents, progress=None, cancelled=None, stages=None):
        seen.append(("decode", latents))
        return "waveform", {"seconds_of_audio": 12.0}

    monkeypatch.setattr(generate, "write_score", write_score)
    monkeypatch.setattr(generate, "sing", sing)
    monkeypatch.setattr(generate, "decode", decode)

    waveform, score, timing = generate.run(
        FakeModels(), "s", "l", 3, dict(constants.DEFAULT_OPTIONS))
    assert seen == ["write_score", ("sing", IDS), ("decode", "latents")]
    assert (waveform, score) == ("waveform", SCORE)
    assert timing["abc"]["seconds"] == 1.0
    assert timing["seconds_of_audio"] == 12.0
    assert "total_seconds" in timing


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
             progress=None, cancelled=None, stages=None):
        seen.update(ids=abc_ids, abc=abc)
        return "latents", {"semantic": {}, "acoustic": {}}

    def decode(models, latents, progress=None, cancelled=None, stages=None):
        return "waveform", {"seconds_of_audio": 12.0}

    monkeypatch.setattr(generate, "write_score", write_score)
    monkeypatch.setattr(generate, "sing", sing)
    monkeypatch.setattr(generate, "decode", decode)

    settings = dict(constants.DEFAULT_OPTIONS, transpose=-5)
    _waveform, score, _timing = generate.run(FakeModels(), "s", "l", 3, settings)
    moved = transpose.move(MOVABLE, -5).text
    assert seen == {"ids": None, "abc": moved}
    assert score == moved


def test_moving_with_cot_off_says_there_is_no_score_to_move():
    with pytest.raises(ValueError) as error:
        generate.moved("", 2, "off")
    assert "'cot'" in str(error.value)
