"""Node contract checks that run without ComfyUI."""

import pytest

from yue2_comfy import constants, nodes


def node_classes():
    return sorted(nodes.NODE_CLASS_MAPPINGS.items())


@pytest.mark.parametrize("name,cls", node_classes())
def test_every_node_is_wired(name, cls):
    assert name in nodes.NODE_DISPLAY_NAME_MAPPINGS, "no display name"
    assert cls.CATEGORY in (constants.CATEGORY, constants.ADVANCED_CATEGORY)
    assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
    assert hasattr(cls, cls.FUNCTION)
    assert cls.DESCRIPTION.strip()


def test_the_menu_offers_three_nodes_and_hides_the_rest_one_level_down():
    """The whole product is that someone can write a song without reading.

    Eight nodes in one menu is the pack this one was written not to be. The
    staged four and their selector earn their place by being one click further
    in, so demoting a headline node or promoting a staged one has to be done on
    purpose rather than by editing a class and not noticing.
    """
    plain = {name for name, cls in node_classes()
             if cls.CATEGORY == constants.CATEGORY}
    advanced = {name for name, cls in node_classes()
                if cls.CATEGORY == constants.ADVANCED_CATEGORY}
    assert plain == {"YuE2GenerateSong", "YuE2WriteSong", "YuE2Options"}
    assert advanced == set(nodes.STAGED_CLASSES)


@pytest.mark.parametrize("name,cls", node_classes())
def test_input_types_are_well_formed(name, cls):
    spec = cls.INPUT_TYPES()
    assert set(spec) <= {"required", "optional", "hidden"}
    for section in ("required", "optional"):
        for key, entry in spec.get(section, {}).items():
            assert isinstance(entry, tuple) and entry, key
            if len(entry) > 1:
                assert isinstance(entry[1], dict), key


def options_widgets():
    spec = nodes.YuE2Options.INPUT_TYPES()
    widgets = {}
    for section in ("required", "optional"):
        widgets.update(spec.get(section, {}))
    return widgets


def test_every_default_option_has_a_widget():
    assert set(options_widgets()) == set(constants.DEFAULT_OPTIONS)


def test_widget_defaults_match_the_single_source_of_truth():
    for key, entry in options_widgets().items():
        assert entry[1]["default"] == constants.DEFAULT_OPTIONS[key], key


def test_options_round_trip():
    """An unconnected options socket must never be a special case.

    Building the node with every widget at its default has to produce exactly
    the dict that every consumer starts from.
    """
    widgets = options_widgets()
    built = nodes.YuE2Options().build(
        **{key: constants.DEFAULT_OPTIONS[key] for key in widgets})[0]
    assert built == constants.DEFAULT_OPTIONS


def test_seed_range_matches_the_protocol():
    """SongRequest demands 0 <= seed < 2**63, so the widget must not offer more."""
    entry = nodes.YuE2GenerateSong.INPUT_TYPES()["required"]["seed"]
    assert entry[1]["min"] == 0
    assert entry[1]["max"] == (1 << 63) - 1


class FakeModels:
    """What loader.acquire hands back, minus seven gigabytes."""

    device = "cpu"
    lm = vae = tokenizer = None


def fake_timing(seconds):
    """The timing dict the node reads to write its one summary line."""
    return {
        "seconds_of_audio": seconds,
        "total_seconds": 1.0,
        "semantic": {"output_tokens": 500, "output_tps": 140.0,
                     "execution": "cuda_graph", "attention": "sdpa"},
    }


def test_generate_returns_a_comfyui_audio_dict(monkeypatch):
    """The AUDIO contract, proved without loading any weights.

    The loader and the stages are stubbed on purpose. This test is about what
    the node hands back and about the residency contract around it, and both
    have to hold on a machine that has never downloaded the model.
    """
    torch = pytest.importorskip("torch")
    from yue2_comfy import generate, loader

    unloaded = []
    monkeypatch.setattr(loader, "locate",
                        lambda variant="standard": loader.Files("lm", "vae", "merges"))
    monkeypatch.setattr(loader, "acquire", lambda *args, **kwargs: FakeModels())
    monkeypatch.setattr(loader, "unload", lambda: unloaded.append(True))
    monkeypatch.setattr(generate, "run", lambda *args, **kwargs: (
        torch.zeros(1, 2, constants.SAMPLE_RATE, dtype=torch.float32),
        "X:1",
        fake_timing(1.0),
    ))

    audio, score = nodes.YuE2GenerateSong().generate(
        style="test", lyrics="[Verse]" + chr(10) + "hello", seed=1)
    assert unloaded == [True], "keep_model_loaded is off by default, so it must unload"
    assert set(audio) == {"waveform", "sample_rate"}
    assert audio["sample_rate"] == constants.SAMPLE_RATE
    waveform = audio["waveform"]
    assert waveform.ndim == 3 and waveform.shape[0] == 1 and waveform.shape[1] == 2
    assert waveform.dtype == torch.float32
    assert isinstance(score, str)


def test_empty_prompt_is_refused():
    with pytest.raises(ValueError):
        nodes.YuE2GenerateSong().generate(style="  ", lyrics="", seed=1)


def test_max_seconds_widget_offers_the_automatic_zero():
    entry = options_widgets()["max_seconds"]
    assert entry[1]["min"] == 0.0
    assert entry[1]["max"] == constants.MAX_SECONDS


def test_song_length_is_chosen_by_word_rather_than_by_number():
    """Seconds were a unit nobody could check against the song they got."""
    entry = nodes.YuE2WriteSong.INPUT_TYPES()["required"]["length"]
    assert entry[0] == list(constants.WRITER_LENGTH_CHOICES)
    assert entry[1]["default"] == constants.WRITER_LENGTH_DEFAULT
    assert entry[1]["default"] in entry[0]


def test_the_length_tooltip_names_every_choice_and_its_size():
    tooltip = nodes.WRITER_LENGTH_TOOLTIP
    for name, lines in constants.WRITER_LENGTH_LINES.items():
        assert "'" + name + "'" in tooltip
        assert "{} sung lines".format(lines) in tooltip
