"""'vocals_only' on the song nodes and the YuE2 Vocals Only node, proved without weights or torch."""

from __future__ import annotations

import contextlib

import pytest

from yue2_comfy import constants, devices, download, generate, memory, nodes, staged, vocals_only
from yue2_comfy.vocals import runtime


class Waveform:
    """Stands in for a tensor: only its length is ever read by the node itself."""

    def __init__(self, name, samples=48000):
        self.name = name
        self.shape = (1, 2, samples)


def stub_separation(monkeypatch, found="weights.ckpt"):
    """The separator replaced; hands back every call and every unload."""
    calls = {"separate": [], "unload": 0}

    def separate(path, device, waveform, rate, progress=None, cancelled=None):
        calls["separate"].append({"path": path, "device": device, "waveform": waveform, "rate": rate,
                                  "progress": progress})
        return Waveform("voice of " + getattr(waveform, "name", "?"), waveform.shape[-1])

    def unload():
        calls["unload"] += 1

    monkeypatch.setattr(download, "ensure_vocals", lambda settings, progress=None: found)
    monkeypatch.setattr(devices, "resolve", lambda spec: "cpu")
    monkeypatch.setattr(runtime, "separate", separate)
    monkeypatch.setattr(runtime, "unload", unload)
    return calls


def test_the_switch_is_the_last_options_widget_and_starts_off():
    spec = nodes.YuE2Options.INPUT_TYPES()
    assert list(spec["optional"])[-1] == "vocals_only"
    kind, settings = spec["optional"]["vocals_only"]
    assert kind == "BOOLEAN" and settings["default"] is False
    assert constants.DEFAULT_OPTIONS["vocals_only"] is False


def test_the_node_takes_audio_and_hands_back_vocals():
    spec = vocals_only.YuE2VocalsOnly.INPUT_TYPES()
    assert list(spec["required"]) == ["audio"]
    assert list(spec["optional"]) == ["options"]
    assert vocals_only.YuE2VocalsOnly.RETURN_TYPES == ("AUDIO",)
    assert vocals_only.YuE2VocalsOnly.RETURN_NAMES == ("vocals",)
    assert nodes.NODE_CLASS_MAPPINGS["YuE2VocalsOnly"] is vocals_only.YuE2VocalsOnly
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["YuE2VocalsOnly"] == "YuE2 Vocals Only"


def test_the_node_keeps_the_rate_it_was_given_and_lets_the_model_go(monkeypatch):
    calls = stub_separation(monkeypatch)
    song = Waveform("song", 22050)
    vocals, = vocals_only.YuE2VocalsOnly().separate({"waveform": song, "sample_rate": 22050})
    assert vocals["sample_rate"] == 22050
    assert vocals["waveform"].name == "voice of song"
    assert calls["separate"][0]["rate"] == 22050 and calls["separate"][0]["path"] == "weights.ckpt"
    assert calls["unload"] == 1


def test_keep_model_loaded_keeps_the_separator(monkeypatch):
    calls = stub_separation(monkeypatch)
    vocals_only.YuE2VocalsOnly().separate({"waveform": Waveform("song"), "sample_rate": 48000},
                                          options=dict(constants.DEFAULT_OPTIONS, keep_model_loaded=True))
    assert calls["unload"] == 0


def test_no_audio_says_what_to_connect():
    with pytest.raises(ValueError) as refused:
        vocals_only.YuE2VocalsOnly().separate(None)
    assert "audio" in str(refused.value)


def test_weights_that_cannot_be_had_are_refused_with_the_reason(monkeypatch):
    def missing(settings, progress=None):
        raise FileNotFoundError("Mel-Band RoFormer is not on this machine yet.")

    monkeypatch.setattr(download, "ensure_vocals", missing)
    with pytest.raises(ValueError) as refused:
        vocals_only.YuE2VocalsOnly().separate({"waveform": Waveform("song"), "sample_rate": 48000})
    assert "not on this machine" in str(refused.value)


def test_the_pack_registers_the_separator_for_unload_models():
    assert ("Mel-Band RoFormer", ".vocals.runtime") in memory.keepers()
    assert callable(runtime.unload) and callable(runtime.is_loaded)


class FakeModels:
    device = "cpu"
    lm = vae = None


def fake_timing(seconds):
    return {"seconds_of_audio": seconds, "total_seconds": 1.0,
            "semantic": {"output_tokens": 1, "output_tps": 1.0, "execution": "eager", "attention": "sdpa"}}


def stub_song(monkeypatch):
    @contextlib.contextmanager
    def session(settings, unique_id, progress):
        yield FakeModels()

    monkeypatch.setattr(nodes, "session", session)
    monkeypatch.setattr(generate, "run", lambda models, style, lyrics, seed, settings, **kwargs: (
        Waveform("song"), "X:1", "X:1", fake_timing(1.0)))


@pytest.mark.parametrize("switch", [False, True])
def test_the_song_node_separates_only_when_the_switch_is_on(monkeypatch, switch):
    calls = stub_separation(monkeypatch)
    stub_song(monkeypatch)
    out = nodes.YuE2GenerateSong().generate(style="a style", lyrics="words", seed=1,
                                            options=dict(constants.DEFAULT_OPTIONS, vocals_only=switch))
    audio, score = out["result"]
    assert score == "X:1"
    assert audio["sample_rate"] == constants.SAMPLE_RATE
    assert audio["waveform"].name == ("voice of song" if switch else "song")
    assert len(calls["separate"]) == (1 if switch else 0)
    if switch:
        assert calls["separate"][0]["rate"] == constants.SAMPLE_RATE


def stub_staged(monkeypatch):
    class Latents:
        def cpu(self):
            return self

        def to(self, device):
            return self

    @contextlib.contextmanager
    def session(settings, unique_id, progress):
        yield FakeModels()

    monkeypatch.setattr(staged, "session", session)
    monkeypatch.setattr(generate, "sing", lambda *args, **kwargs: (Latents(), {"semantic": {}}))
    monkeypatch.setattr(generate, "decode", lambda *args, **kwargs: (Waveform("song"), {"seconds_of_audio": 1.0}))
    return Latents


@pytest.mark.parametrize("switch", [False, True])
def test_the_render_node_separates_only_when_the_switch_is_on(monkeypatch, switch):
    calls = stub_separation(monkeypatch)
    stub_staged(monkeypatch)
    plan = {"style": "s", "lyrics": "l", "seed": 5, "score": "X:1\nK:C\nCDEF|\n", "ids": [1, 2],
            "settings": dict(constants.DEFAULT_OPTIONS)}
    audio, latents = staged.YuE2RenderPlan().render(plan, options={"vocals_only": switch})
    assert audio["waveform"].name == ("voice of song" if switch else "song")
    assert latents["settings"]["vocals_only"] is switch
    assert len(calls["separate"]) == (1 if switch else 0)


def test_the_decode_node_follows_the_switch_the_render_node_ran_with(monkeypatch):
    calls = stub_separation(monkeypatch)
    latents_class = stub_staged(monkeypatch)
    latents = {"latents": latents_class(), "settings": dict(constants.DEFAULT_OPTIONS, vocals_only=True)}
    audio, = staged.YuE2DecodeLatents().decode(latents)
    assert audio["waveform"].name == "voice of song"
    audio, = staged.YuE2DecodeLatents().decode(latents, options={"vocals_only": False})
    assert audio["waveform"].name == "song"
    assert len(calls["separate"]) == 1


def test_a_song_node_refuses_missing_weights_before_it_sings(monkeypatch):
    """Weights that cannot be had are said before the song is sung, not after it is lost."""
    def missing(settings, progress=None):
        raise FileNotFoundError("Mel-Band RoFormer is not on this machine yet.")

    sung = []
    stub_song(monkeypatch)
    stub_staged(monkeypatch)
    monkeypatch.setattr(download, "ensure_vocals", missing)
    monkeypatch.setattr(generate, "run", lambda *args, **kwargs: sung.append("run"))
    monkeypatch.setattr(generate, "sing", lambda *args, **kwargs: sung.append("sing"))
    monkeypatch.setattr(generate, "decode", lambda *args, **kwargs: sung.append("decode"))
    options = dict(constants.DEFAULT_OPTIONS, vocals_only=True)
    with pytest.raises(ValueError) as refused:
        nodes.YuE2GenerateSong().generate(style="a style", lyrics="words", seed=1, options=options)
    assert "not on this machine" in str(refused.value)
    plan = {"style": "s", "lyrics": "l", "seed": 5, "score": "X:1\nK:C\nCDEF|\n", "ids": [1, 2],
            "settings": dict(constants.DEFAULT_OPTIONS)}
    with pytest.raises(ValueError):
        staged.YuE2RenderPlan().render(plan, options={"vocals_only": True})
    with pytest.raises(ValueError):
        staged.YuE2DecodeLatents().decode({"latents": object(), "settings": options})
    assert sung == []
