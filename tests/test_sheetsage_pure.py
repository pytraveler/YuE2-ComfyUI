"""SheetSage2's tokens, grammar, windows and notation, without torch or weights.

The strongest check here is the last one: ComfyUI master's tokens for three
songs this pack sang, run through these modules, give master's ABC to the byte
in both modes. Everything above it pins a smaller rule that check relies on.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from yue2_comfy.sheetsage import abc_rebuild, events, grammar, vocab
from yue2_comfy.vendor.yue2_music import abc_tools

DATA = json.loads((pathlib.Path(__file__).parent / "data" / "sheetsage_pack.json").read_text())


def test_the_vocabulary_layout_fills_the_released_embedding_exactly():
    assert vocab.SIZE == 31678
    assert vocab.BLOCKS["shift"] == (260, 517)
    assert vocab.BLOCKS["time"] == (517, 30517)
    assert vocab.BLOCKS["duration"] == (31654, 31678)
    assert len(vocab.FULL_CHORDS) == 361 and vocab.FULL_CHORDS[0] == "N"
    assert vocab.FULL_CHORDS[1:5] == ("C:maj/2", "C:maj/3", "C:maj/5", "C:maj")


def test_the_full_task_prefix_is_the_one_master_decodes_with():
    for song in DATA["songs"]:
        assert song["windows"][0][:8] == vocab.prompt_prefix()
    assert vocab.prompt_prefix() == [1, 4, 5, 6, 7, 9, 11, 3]


def test_prompts_that_ask_for_the_same_field_twice_are_refused():
    with pytest.raises(ValueError):
        vocab.canonical_prompts(["melody_vocal", "melody_full"])
    assert vocab.canonical_prompts(["key", "timestamp", "key"]) == ("timestamp", "key")


def test_long_shifts_are_written_as_several_tokens():
    assert vocab.shift_tokens(0) == [260]
    assert vocab.shift_tokens(256) == [516]
    assert vocab.shift_tokens(300) == [516, 260 + 44]


@pytest.mark.parametrize("song", DATA["songs"], ids=lambda song: song["name"])
def test_decoding_and_encoding_a_window_gives_back_its_tokens(song):
    tokens = song["windows"][0] + [vocab.EOS]
    decoded = vocab.decode(tokens)
    assert decoded["prompts"] == vocab.FULL_PROMPTS
    assert vocab.encode(decoded["prompts"], decoded["events"]) == tokens


def test_an_empty_event_is_refused_strictly_and_skipped_leniently():
    tokens = vocab.prompt_prefix() + [260, vocab.first("time") + 5, 262, 262, vocab.EOS]
    with pytest.raises(ValueError, match="empty event"):
        vocab.decode(tokens)
    decoded, warning = vocab.decode_window(tokens)
    assert len(decoded["events"]) == 1 and "empty event" in warning


@pytest.mark.parametrize("song", DATA["songs"], ids=lambda song: song["name"])
def test_every_token_master_chose_is_one_the_grammar_allows(song):
    tokens = song["windows"][0]
    out = tokens.index(vocab.OUT)
    state = grammar.Grammar().follow(tokens[:out + 1])
    for token in tokens[out + 1:]:
        assert any(a <= token < b for a, b in state.allowed()), vocab.kind(token)
        assert not state.update(token)
    assert any(a <= vocab.EOS < b for a, b in state.allowed())


def test_the_grammar_holds_a_meter_to_its_position_and_a_pitch_to_its_length():
    state = grammar.Grammar().follow(vocab.prompt_prefix())
    assert (vocab.EOS, vocab.EOS + 1) not in state.allowed()
    state.update(260)
    state.update(vocab.first("meter"))
    assert state.allowed() == [(vocab.EOS, vocab.EOS + 1), vocab.BLOCKS["shift"], vocab.BLOCKS["eighth"]]
    state.update(vocab.first("eighth"))
    state.update(vocab.first("pitch") + 60)
    assert state.allowed()[-2:] == [vocab.BLOCKS["duration"], vocab.BLOCKS["pitch"]]
    for _ in range(4):
        state.update(260)
    assert vocab.BLOCKS["shift"] not in state.allowed()


def test_a_long_song_is_read_in_windows_a_hundred_seconds_apart():
    plan = events.window_plan(450.0)
    assert [(w["start"], w["accept_start"], w["accept_end"], w["stop"]) for w in plan] == [
        (0.0, 0.0, 200.0, 200.0), (100.0, 200.0, 300.0, 200.0), (150.0, 300.0, 450.0, None)]
    assert events.stop_seconds(plan[-1], 450.0) == 300.0
    assert [(w["start"], w["stop"]) for w in events.window_plan(35.0)] == [(0.0, None)]
    assert events.stop_seconds(events.window_plan(35.0)[0], 35.0) == 35.0


def window_events(first: float, last: float, step: float = 0.32) -> list:
    """Beat events from ``first`` to ``last``, one every ``step`` seconds."""
    kept, time = [], first
    while time <= last + 1e-9:
        kept.append({"time": time, "values": {"rhythm": {"meter": (4, 4),
                                                         "eighth_position": 2 * (len(kept) % 4)}}})
        time += step
    return kept


def test_a_window_that_ran_out_of_tokens_hands_its_last_seconds_to_the_next_one():
    window = events.window_plan(450.0)[0]
    assert events.resume_point(window, window_events(0.0, 198.4)) == pytest.approx(198.44, abs=1e-3)
    assert events.resume_point(window, window_events(0.0, 199.76)) == 0.0
    assert events.resume_point(window, []) == 0.0
    beatless = [{"time": 120.0, "values": {"melody": []}}]
    assert events.resume_point(window, beatless) == pytest.approx(120.125, abs=1e-3)


def test_a_note_sounding_into_the_next_onset_is_cut_there():
    notes = [[0.0, 1.0, 60, 0], [0.5, 0.8, 62, 0], [0.2, 2.0, 50, 1]]
    assert events.notation_notes(notes) == [[0.0, 0.5, 60, 0], [0.2, 2.0, 50, 1], [0.5, 0.8, 62, 0]]


def tiny_rows(notes, chords=()):
    beats = [[0.5 * i, i % 4 + 1, 4, 4] for i in range(21)]
    return {"beats": beats, "chords": list(chords), "keys": [[0.0, 10.0, "C:major"]],
            "structures": [[0.0, 10.0, "verse"]], "notes": notes}


def test_a_note_or_chord_on_the_grids_last_point_alone_is_dropped_rather_than_refusing_the_score():
    text = abc_rebuild.build(tiny_rows([[0.0, 1.0, 60, 0], [9.97, 10.0, 62, 0]]), melody_only=True)
    assert "C8z8" in text and "D" not in text.split("K:C", 1)[1].replace("Vocal Melody", "")
    with_chord = abc_rebuild.build(tiny_rows([[0.0, 1.0, 60, 0]], chords=[[0.0, 9.95, "C:maj"], [9.95, 10.0, "N"]]))
    assert '"C"' in with_chord


def test_a_note_shorter_than_half_a_subbeat_inside_the_grid_still_refuses_the_score():
    with pytest.raises(ValueError, match="shorter than a subbeat"):
        abc_rebuild.build(tiny_rows([[5.0, 5.03, 62, 0]]), melody_only=True)


def test_a_bar_whose_beat_numbers_repeat_or_skip_is_as_long_as_its_beats():
    rows = tiny_rows([[0.0, 1.0, 60, 0]])
    numbers = [1, 2, 3, 4, 1, 2, 2, 3, 4, 1, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1]
    rows["beats"] = [[0.5 * i, number, 4, 4] for i, number in enumerate(numbers)]
    bars = abc_rebuild.infer_bars([abc_rebuild.Beat(*row) for row in rows["beats"]])
    assert [bar.numerator for bar in bars] == [4, 5, 3, 4, 4]
    text = abc_rebuild.build(rows, melody_only=True)
    assert "M:5/4" in text and "M:3/4" in text
    abc_tools.parse(text)


def test_a_meter_the_model_ends_an_event_on_is_a_meter_change_that_places_no_beat():
    tokens = vocab.prompt_prefix() + [260, vocab.first("time") + 5, vocab.first("meter") + 1,
                                      262, vocab.first("eighth"), vocab.EOS]
    decoded = vocab.decode(tokens)
    assert decoded["events"][0]["values"]["rhythm"] == {"meter": vocab.METERS[1]}
    assert decoded["events"][1]["values"]["rhythm"] == {"eighth_position": 0}
    timed = [{"time": 0.0, "values": decoded["events"][0]["values"]}, {"time": 1.0, "values": decoded["events"][1]["values"]}]
    assert events.beats(timed) == [[1.0, 1, vocab.METERS[1][0], vocab.METERS[1][1]]]


def test_model_chord_and_key_labels_become_abc_symbols():
    assert abc_rebuild.chord_text("A:min7/b3") == "Am7/C"
    assert abc_rebuild.chord_text("D#:sus4(b7)") == "D#7sus4"
    assert abc_rebuild.chord_text("N") is None
    assert abc_rebuild.key_text("A#:major") == "Bb"
    assert abc_rebuild.key_text("C#:minor") == "C#m"


@pytest.mark.parametrize("mode", ["melody", "full"])
@pytest.mark.parametrize("song", DATA["songs"], ids=lambda song: song["name"])
def test_masters_tokens_give_masters_abc_to_the_byte(song, mode):
    duration = song["seconds"]
    stitched = []
    for index, (window, tokens) in enumerate(zip(events.window_plan(duration), song["windows"])):
        decoded, _warning = vocab.decode_window(tokens + [vocab.EOS])
        stitched.extend(events.stitch(decoded, events.time_map(decoded), window, duration, index))
    rows = events.score_rows(events.sort_song(stitched), duration)
    text = abc_rebuild.build(rows, melody_only=(mode == "melody"))
    assert text == song["abc"][mode]
    abc_tools.parse(text)
