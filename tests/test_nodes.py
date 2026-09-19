"""Node contract checks that run without ComfyUI."""

import contextlib

import pytest

from yue2_comfy import constants, edits, nodes, phrasing, songs


def node_classes():
    return sorted(nodes.NODE_CLASS_MAPPINGS.items())


@pytest.mark.parametrize("name,cls", node_classes())
def test_every_node_is_wired(name, cls):
    assert name in nodes.NODE_DISPLAY_NAME_MAPPINGS, "no display name"
    assert cls.CATEGORY in (constants.CATEGORY, constants.ADVANCED_CATEGORY)
    assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
    assert hasattr(cls, cls.FUNCTION)
    assert cls.DESCRIPTION.strip()


def test_the_menu_offers_four_nodes_and_hides_the_rest_one_level_down():
    """The whole product is that someone can write a song without reading.

    Nine nodes in one menu is the pack this one was written not to be. The
    staged four and their selector earn their place by being one click further
    in, so demoting a headline node or promoting a staged one has to be done on
    purpose rather than by editing a class and not noticing. Transcribe is a
    headline node: covering a song is a thing people come for. So is Load
    MIDI, the other way in for a tune someone already has, and Vocals Only,
    the way to an a cappella of any recording. And LoRA: people asked for it
    by name, from ComfyUI's own nodes, before it existed.
    """
    plain = {name for name, cls in node_classes()
             if cls.CATEGORY == constants.CATEGORY}
    advanced = {name for name, cls in node_classes()
                if cls.CATEGORY == constants.ADVANCED_CATEGORY}
    assert plain == {"YuE2GenerateSong", "YuE2WriteSong", "YuE2Options", "YuE2Transcribe", "YuE2LoadMidi",
                     "YuE2VocalsOnly", "YuE2LoRA"}
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


def test_new_widgets_go_last_so_saved_options_keep_their_places():
    """ComfyUI hands a saved node its widget values by position.

    A widget added anywhere but the end would move every value after it into
    the wrong widget, in every workflow saved before it existed. At the end, an
    older workflow simply has a value or two fewer, and the new widgets take
    their defaults. 'offload' was added first, 'transpose' after it,
    'vocals_only' after that, and 'low_vram' last.
    """
    spec = nodes.YuE2Options.INPUT_TYPES()
    assert list(spec["optional"])[-4:] == ["offload", "transpose", "vocals_only",
                                           "low_vram"]


def test_transpose_reaches_an_octave_either_way_and_starts_at_zero():
    entry = options_widgets()["transpose"]
    assert entry[0] == "INT"
    assert entry[1]["default"] == 0
    assert (entry[1]["min"], entry[1]["max"]) == (-constants.TRANSPOSE_LIMIT,
                                                  constants.TRANSPOSE_LIMIT)


def test_moving_a_song_with_cot_off_is_refused_before_anything_loads(monkeypatch):
    """With cot 'off' there is no score to move, and learning that should not cost a load."""
    entered = []

    class Session:
        def __init__(self, *args):
            entered.append(args)

        def __enter__(self):
            return FakeModels()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(nodes, "session", Session)
    options = dict(constants.DEFAULT_OPTIONS, cot="off", transpose=2)
    with pytest.raises(ValueError) as error:
        nodes.YuE2GenerateSong().generate(style="s", lyrics="l", seed=1, options=options)
    assert "cot" in str(error.value)
    assert entered == []


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
        "X:1",
        fake_timing(1.0),
        None,
    ))

    out = nodes.YuE2GenerateSong().generate(
        style="test", lyrics="[Verse]" + chr(10) + "hello", seed=1)
    audio, score = out["result"]
    assert out["ui"][edits.SCORE_UI] == ["X:1"]
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


def test_the_edited_score_box_is_the_last_widget_on_the_song_node():
    """ComfyUI hands a saved node its widget values by position.

    'options' is a socket and holds no value, so the new box sits right after
    the seed and its control, and a workflow saved before it existed simply has
    one value fewer. 'lora' is a socket too, and came later, so it goes last.
    """
    spec = nodes.YuE2GenerateSong.INPUT_TYPES()
    assert list(spec["required"]) == ["style", "lyrics", "seed"]
    assert list(spec["optional"]) == ["options", "score_abc", "lora"]
    assert spec["optional"]["score_abc"][1]["default"] == ""
    assert spec["optional"]["lora"][0] == constants.LORA_TYPE


WRITTEN = "X:1\nK:C\n\"C\"CDEF|"
EDITED = "X:1\nK:C\n\"Am\"EFGA|"
MELODY_ONLY = "X:1\nV: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\nK:C\nV: Vocal\nEFGA|"


def stub_run(monkeypatch, performance=None):
    """The song node with its pipeline replaced; returns what reached the pipeline and the notices."""
    calls = []
    said = []

    @contextlib.contextmanager
    def session(settings, unique_id, progress):
        yield FakeModels()

    def run(models, style, lyrics, seed, settings, progress=None, cancelled=None, edited=None,
            tune_seconds=None):
        calls.append({"edited": edited, "cot": settings["cot"]})
        if tune_seconds is not None:
            calls[-1]["tune_seconds"] = tune_seconds
        sung = edited or WRITTEN
        return "waveform", sung, "" if edited else WRITTEN, fake_timing(3.0), performance

    from yue2_comfy import generate

    monkeypatch.setattr(nodes, "session", session)
    monkeypatch.setattr(generate, "run", run)
    monkeypatch.setattr(nodes, "announce",
                        lambda node, findings, kind="notice": said.append(findings))
    return calls, said


def sing_with(score_abc, lyrics="words", options=None):
    return nodes.YuE2GenerateSong().generate(style="a style", lyrics=lyrics, seed=4,
                                             options=options, score_abc=score_abc,
                                             unique_id="3")


def test_the_song_node_remembers_the_song_it_sang(monkeypatch):
    """The song memory gets the score that was sung, which after a move is not the one written."""
    kept = []
    monkeypatch.setattr(songs, "keep", lambda *args: kept.append(args))
    stub_run(monkeypatch, performance="sung from")
    out = sing_with("", options=dict(constants.DEFAULT_OPTIONS, cot="full"))
    audio, score = out["result"]
    assert len(kept) == 1
    given, origin, style, lyrics, seed, settings, sung, performance = kept[0]
    assert given is audio
    assert (origin, style, lyrics, seed, sung, performance) == (
        "YuE2 Generate Song", "a style", "words", 4, score, "sung from")
    assert settings["cot"] == "full"


def test_an_empty_box_writes_a_score_as_the_node_always_has(monkeypatch):
    calls, said = stub_run(monkeypatch)
    out = sing_with("")
    cot = constants.DEFAULT_OPTIONS["cot"]
    assert calls == [{"edited": None, "cot": cot}]
    assert said == []
    assert out["ui"] == {edits.SCORE_UI: [WRITTEN],
                         edits.WORDS_UI: [edits.mark("a style", "words", cot)],
                         edits.AUTO_SECONDS_UI: [constants.auto_seconds("words")]}
    assert out["result"][1] == WRITTEN


def test_an_edit_for_these_words_is_sung_instead_of_writing_a_score(monkeypatch):
    """The ui carries no score then: the model wrote none, and the editor keeps the one it had."""
    calls, said = stub_run(monkeypatch)
    cot = constants.DEFAULT_OPTIONS["cot"]
    out = sing_with(edits.attach(EDITED, edits.mark("a style", "words", cot)))
    assert calls == [{"edited": EDITED, "cot": cot}]
    assert said == []
    assert out["ui"] == {edits.WORDS_UI: [edits.mark("a style", "words", cot)],
                         edits.AUTO_SECONDS_UI: [constants.auto_seconds("words")]}
    assert out["result"][1] == EDITED


def test_an_edit_for_other_words_is_left_unsung_and_the_node_says_so(monkeypatch):
    calls, said = stub_run(monkeypatch)
    cot = constants.DEFAULT_OPTIONS["cot"]
    out = sing_with(edits.attach(EDITED, edits.mark("a style", "words", cot)), lyrics="new words")
    assert calls == [{"edited": None, "cot": cot}]
    assert [level for level, _message in said[0]] == ["warn"]
    assert nodes.GENERATE_INSTEAD in said[0][0][1]
    assert out["ui"][edits.SCORE_UI] == [WRITTEN]


def test_a_score_with_no_mark_is_sung_whatever_the_words(monkeypatch):
    calls, said = stub_run(monkeypatch)
    sing_with(EDITED, lyrics="any words at all")
    assert calls[0]["edited"] == EDITED
    assert said == []


def test_with_cot_off_an_edit_waits_on_the_node(monkeypatch):
    calls, said = stub_run(monkeypatch)
    sing_with(EDITED, options=dict(constants.DEFAULT_OPTIONS, cot="off"))
    assert calls == [{"edited": None, "cot": "off"}]
    assert said[0][0][1] == edits.COT_OFF


def test_a_melody_only_score_sung_under_cot_full_is_sung_with_a_warning(monkeypatch):
    """A transcription for a cover has no chords; 'full' promises the model it has them."""
    calls, said = stub_run(monkeypatch)
    sing_with(MELODY_ONLY, options=dict(constants.DEFAULT_OPTIONS, cot="full"))
    assert calls == [{"edited": MELODY_ONLY, "cot": "full"}]
    assert said == [[("warn", edits.CHORDLESS)]]


BARE_TUNE = ('X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=100\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
             'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\nV: Vocal\nz2C2D2E2F2G2A2B2|\nV: Ins\nZ|\n')


def test_a_bare_tune_wired_in_is_sung_with_the_lyrics_laid_along_it(monkeypatch):
    """A MIDI file's tune names no section; the words decide them, and the node says on which bars."""
    calls, said = stub_run(monkeypatch)
    out = sing_with(BARE_TUNE, lyrics="[Chorus]\nla la la la la la la",
                    options=dict(constants.DEFAULT_OPTIONS, cot="melody"))
    sung = calls[0]["edited"]
    laid = phrasing.lay(BARE_TUNE, "[Chorus]\nla la la la la la la")
    assert sung == laid.score
    assert calls[0]["tune_seconds"] == laid.seconds == 4.8
    assert "% chorus" in sung.splitlines()
    assert [level for level, _message in said[0]] == ["notice"]
    assert "[Chorus] on bar 1" in said[0][0][1]
    assert out["result"][1] == sung


def test_a_score_that_names_its_sections_is_sung_as_it_arrives(monkeypatch):
    calls, said = stub_run(monkeypatch)
    named = BARE_TUNE.replace("K:C\n", "K:C\n% verse\n")
    sing_with(named, lyrics="[Chorus]\nla la la la la la la", options=dict(constants.DEFAULT_OPTIONS, cot="melody"))
    assert calls[0]["edited"] == named.strip()
    assert said == []


def test_a_melody_only_score_under_cot_melody_says_nothing(monkeypatch):
    calls, said = stub_run(monkeypatch)
    sing_with(MELODY_ONLY, options=dict(constants.DEFAULT_OPTIONS, cot="melody"))
    assert calls == [{"edited": MELODY_ONLY, "cot": "melody"}]
    assert said == []


def test_the_song_node_tells_the_editor_how_long_its_lyrics_let_a_song_run(monkeypatch):
    """Lyrics wired in from 'YuE2 Write Song' are not on the canvas for the editor to count.

    The editor reads 'max_seconds' off the options node itself, so what the node
    reports is the ceiling at 0, whatever the options say.
    """
    stub_run(monkeypatch)
    lyrics = "[Verse]\none\ntwo\nthree"
    out = sing_with("", lyrics=lyrics, options=dict(constants.DEFAULT_OPTIONS, max_seconds=60.0))
    assert out["ui"][edits.AUTO_SECONDS_UI] == [constants.auto_seconds(lyrics)]


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


def test_the_log_says_where_a_song_spent_its_time():
    """Every stage that ran, and what loading added outside them."""
    from yue2_comfy.nodes import stage_times

    line = stage_times({"abc": {"seconds": 70.2}, "semantic": {"seconds": 71.0},
                        "acoustic": {"seconds": 15.04}, "decode": {"seconds": 1.1},
                        "total_seconds": 160.0}, 166.4)
    assert line == ("stages: score 70.2 s, performance 71.0 s, acoustic 15.0 s, "
                    "decode 1.1 s | loading and the rest 6.4 s")
    assert "score" not in stage_times({"semantic": {"seconds": 1.0}, "total_seconds": 1.0}, 1.0)
