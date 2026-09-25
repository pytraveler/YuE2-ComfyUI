"""SheetSage2's tokens, grammar, windows and notation, without torch or weights.

The strongest check here is near the end: ComfyUI master's tokens for three
songs this pack sang, run through these modules, give master's ABC to the byte
in both modes as long as keys and chords keep the model's own sharp names.
Named from the key, as this pack writes them since 0.9.3, the same scores
differ from master's in names only: every note, bar, chord and key sounds the
same. Everything above pins a smaller rule those checks rely on.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from yue2_comfy.sheetsage import abc_rebuild, events, grammar, spelling, vocab
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


def test_a_minute_at_a_time_is_the_same_plan_in_fifths():
    """The short window keeps the shape of the long one: two thirds overlap, a third kept."""
    assert events.plan_for(450.0) == events.window_plan(450.0)
    plan = events.plan_for(450.0, events.MINUTE)
    assert [(w["start"], w["accept_start"], w["accept_end"], w["stop"]) for w in plan[:3]] == [
        (0.0, 0.0, 40.0, 40.0), (20.0, 40.0, 60.0, 40.0), (40.0, 60.0, 80.0, 40.0)]
    assert len(plan) == 21 and plan[-1]["accept_end"] == 450.0
    assert events.stop_seconds(plan[-1], 450.0, length=events.MINUTE) == events.MINUTE
    assert events.plan_for(35.0, events.MINUTE) == [
        {"start": 0.0, "end": 35.0, "accept_start": 0.0, "accept_end": 35.0, "prefix_end": 0.0, "stop": None}]


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


def steady_rows(count: int = 12, step: float = 0.5) -> list:
    """A plain 4/4 grid: every beat placed, every number in order."""
    return [[round(step * i, 6), i % 4 + 1, 4, 4] for i in range(count)]


def test_the_beat_a_window_seam_dropped_is_put_back():
    """The bar is short by one beat and both the clock and the numbering say which."""
    rows = [row for row in steady_rows() if row[0] != 3.0]
    filled = events.filled_beats(rows)
    assert len(filled) == 12
    assert filled[6] == [3.0, 3, 4, 4]
    assert [row[1] for row in filled] == [1, 2, 3, 4] * 3


def test_a_gap_the_numbering_does_not_confirm_is_left_where_it_is():
    """A pause in the beat is not a lost beat, and inventing one would move the bar line."""
    rows = steady_rows(8)
    rows[4][0] = 3.0
    assert events.filled_beats(rows) == rows


def test_a_grid_with_every_beat_on_it_is_handed_back_unchanged():
    rows = steady_rows(16)
    assert events.filled_beats(rows) == rows
    assert events.filled_beats([]) == []


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


def test_a_note_too_short_for_the_grid_is_dropped_rather_than_refusing_the_score():
    """One 30-millisecond note is not a reason to throw away a five-minute transcription."""
    text = abc_rebuild.build(tiny_rows([[0.0, 1.0, 60, 0], [5.0, 5.03, 62, 0]]), melody_only=True)
    assert "C8" in text and "D" not in text.split("K:C", 1)[1].replace("Vocal Melody", "")
    kept = abc_rebuild.build(tiny_rows([[0.0, 1.0, 60, 0]], chords=[[0.0, 5.0, "C:maj"],
                                                                   [5.0, 5.03, "G:maj"],
                                                                   [5.03, 10.0, "C:maj"]]))
    assert '"G"' not in kept and '"C"' in kept


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


def written(song, mode) -> str:
    """The ABC this pack writes from master's tokens for one of the reference songs."""
    duration = song["seconds"]
    stitched = []
    for index, (window, tokens) in enumerate(zip(events.window_plan(duration), song["windows"])):
        decoded, _warning = vocab.decode_window(tokens + [vocab.EOS])
        stitched.extend(events.stitch(decoded, events.time_map(decoded), window, duration, index))
    rows = events.score_rows(events.sort_song(stitched), duration)
    return abc_rebuild.build(rows, melody_only=(mode == "melody"))


@pytest.mark.parametrize("mode", ["melody", "full"])
@pytest.mark.parametrize("song", DATA["songs"], ids=lambda song: song["name"])
def test_masters_tokens_give_masters_abc_to_the_byte_with_the_models_own_names(song, mode, monkeypatch):
    monkeypatch.setattr(spelling, "key_name", lambda label: label)
    monkeypatch.setattr(spelling, "respelled", lambda chords, keys: [list(row) for row in chords])
    text = written(song, mode)
    assert text == song["abc"][mode]
    abc_tools.parse(text)


def pitch(name):
    return None if name is None else (abc_tools.NATURAL[name[0]] + name.count("#") - name.count("b")) % 12


def sounding(symbol: str):
    """A chord symbol as what it sounds: root pitch class, quality, bass pitch class."""
    root, quality, bass = re.fullmatch(r"([A-G](?:bb|##|b|#)?)(.*?)(?:/([A-G](?:bb|##|b|#)?))?", symbol).groups()
    return pitch(root), quality, pitch(bass)


def tonic(key: str):
    """An ABC key as what it sounds: the tonic's pitch class and whether it is minor."""
    return pitch(key.rstrip("m")), key.endswith("m")


@pytest.mark.parametrize("mode", ["melody", "full"])
@pytest.mark.parametrize("song", DATA["songs"], ids=lambda song: song["name"])
def test_named_from_the_key_masters_score_changes_names_and_nothing_that_sounds(song, mode):
    ours, masters = abc_tools.parse(written(song, mode)), abc_tools.parse(song["abc"][mode])
    assert abc_tools.compare(masters, ours)["match"]
    for name in abc_tools.VOICES:
        was, now = masters.voices[name], ours.voices[name]
        assert tonic(now.key) == tonic(was.key)
        assert [(time, tonic(key)) for time, key in now.keys] == [(time, tonic(key)) for time, key in was.keys]
        assert [(time, sounding(chord)) for time, chord in now.chords] == [
            (time, sounding(chord)) for time, chord in was.chords]


def test_a_song_in_b_flat_is_written_with_b_flat_chords():
    song = next(song for song in DATA["songs"] if "K:Bb" in song["abc"]["full"])
    text = written(song, "full")
    chords = set(re.findall(r'"([^"]*)"', text)) - {"Vocal Melody", "Ins Melody", "Vocal", "Inst."}
    assert chords == {"Bb", "Bb6", "Eb", "F", "Gm"}
    assert "A#" not in text and "D#" not in text
