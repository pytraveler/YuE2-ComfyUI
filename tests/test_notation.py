"""Scores as notes, and edits written back, proved against upstream's reader.

What the score editor promises is narrow. An untouched score comes back byte
for byte. An edit reads back, through vendor/yue2_music/abc_tools.py, as
exactly the notes and chords that were asked for. Only the bars that changed,
and the bars a tie joins to them, are written again, and they are spelled the
way the model spells. An edit the score cannot hold is refused with a sentence
the person editing can act on.
"""

from __future__ import annotations

import copy
import json
import pathlib
import random

import pytest

from yue2_comfy import notation, routes
from yue2_comfy.vendor.yue2_music import abc_tools

HEADER = ('X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=90\n'
          'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
          'V: Ins clef=treble name="Ins Melody" snm="Inst."\n')

AWKWARD = HEADER + (
    'K:D\n% verse\nV: Vocal\n'
    '"D"F8A8"G/B"B8^c8-|"A7"c8e8-e8=c8|"E"^G8g8G16|\n'
    'V: Ins\nZ|z16F,16|Z|\n'
    '% chorus\nV: Vocal\nK:Bm\n'
    '"Bm"B,8^A,8B,16|"F#7"^a16z16|\n'
    'V: Ins\nK:Bm\nd8f8b16|Z|\n')
"""A tie across a barline, an accidental carried into another octave, a natural
cancelling the key, slash chords, full-bar rests and a key change between
groups."""

MODEL_SCORES = json.loads(
    (pathlib.Path(__file__).resolve().parent / "data" / "model_scores.json")
    .read_text(encoding="utf-8"))

SCORES = [("awkward", AWKWARD)] + [
    (name, entry["text"] if isinstance(entry, dict) else entry)
    for name, entry in MODEL_SCORES.items()]


def tiny(key, vocal, ins=None, meter="4/4"):
    """A one-group score in *key*; the instrumental part rests unless given."""
    count = len(vocal.strip()[:-1].split("|"))
    head = HEADER.replace("M:4/4", "M:" + meter)
    return head + "K:{}\n% verse\nV: Vocal\n{}\nV: Ins\n{}\n".format(
        key, vocal, ins or "Z|" * count)


def notes(sheet, part="Vocal"):
    return sorted((n["start"], n["length"], n["pitch"]) for n in sheet["notes"][part])


def with_notes(sheet, part, items):
    changed = copy.deepcopy(sheet)
    changed["notes"][part] = [{"start": s, "length": l, "pitch": p} for s, l, p in items]
    return changed


def vocal_line(text):
    lines = text.strip().splitlines()
    return lines[lines.index("V: Vocal") + 1]


def ins_line(text):
    lines = text.strip().splitlines()
    return lines[lines.index("V: Ins") + 1]


@pytest.mark.parametrize("name,text", SCORES, ids=[name for name, _ in SCORES])
def test_every_score_reads_as_the_notes_upstream_resolves(name, text):
    sheet = notation.read(text)
    score = abc_tools.parse(text.strip())
    per_quarter = score.unit.denominator / 4
    for part in abc_tools.VOICES:
        assert notes(sheet, part) == sorted(
            (int(start * per_quarter), int(length * per_quarter), pitch)
            for start, pitch, length in score.voices[part].notes)
    assert len(sheet["bars"]) == len(score.voices["Vocal"].bars)
    assert [c["name"] for c in sheet["chords"]] == [c for _, c in score.voices["Vocal"].chords]
    assert sum(section["bars"] for section in sheet["sections"]) == len(sheet["bars"])
    assert sheet["total"] == sum(bar["length"] for bar in sheet["bars"])


@pytest.mark.parametrize("name,text", SCORES, ids=[name for name, _ in SCORES])
def test_an_untouched_score_comes_back_byte_for_byte(name, text):
    assert notation.write(text, notation.read(text)) == {"abc": text.strip(), "bars": []}


def test_the_sheet_counts_in_steps_of_the_scores_own_note_length():
    sixteenths = tiny("C", "C4D4E8G16|").replace("L:1/32", "L:1/16")
    sheet = notation.read(sixteenths.replace("C4D4E8G16", "C2D2E4G8"))
    assert (sheet["unit"], sheet["per_quarter"], sheet["total"]) == (16, 4, 16)
    assert notes(sheet) == [(0, 2, 60), (2, 2, 62), (4, 4, 64), (8, 8, 67)]


def test_the_sections_follow_the_comments_and_cover_every_bar():
    sheet = notation.read(AWKWARD)
    assert sheet["sections"] == [{"name": "verse", "bar": 0, "bars": 3},
                                 {"name": "chorus", "bar": 3, "bars": 2}]
    assert [bar["key"] for bar in sheet["bars"]] == ["D", "D", "D", "Bm", "Bm"]


def test_one_changed_note_rewrites_one_bar_in_the_models_spelling():
    text = tiny("D", '"D"d8f8a8f8|"G"g8b8d\'8b8|')
    sheet = notation.read(text)
    items = notes(sheet)
    items[1] = (8, 8, 79)
    result = notation.write(text, with_notes(sheet, "Vocal", items))
    assert result["bars"] == [0]
    assert vocal_line(result["abc"]) == '"D"d8g8a8f8|"G"g8b8d\'8b8|'
    assert result["abc"].replace(vocal_line(result["abc"]), "") == \
        text.strip().replace(vocal_line(text), "")


def test_a_note_outside_the_key_is_marked_and_the_bar_remembers_the_mark():
    text = tiny("C", "C8D8E8F8|")
    sheet = notation.read(text)
    result = notation.write(text, with_notes(sheet, "Vocal",
                                             [(0, 8, 60), (8, 8, 61), (16, 8, 61), (24, 8, 60)]))
    assert vocal_line(result["abc"]) == "C8^C8C8=C8|"


def test_a_mark_is_written_again_in_another_octave_so_every_reader_agrees():
    """The dialect would sharpen the second c without its mark; most readers would not."""
    text = tiny("C", "C16E16|")
    sheet = notation.read(text)
    result = notation.write(text, with_notes(sheet, "Vocal", [(0, 16, 61), (16, 16, 73)]))
    assert vocal_line(result["abc"]) == "^C16^c16|"


@pytest.mark.parametrize("key,pitch,written", [
    ("F", 66, "_G32|"), ("Bb", 66, "_G32|"), ("G", 70, "^A32|"), ("C", 70, "^A32|"),
    ("Dm", 73, "^c32|"), ("Gm", 66, "^F32|"), ("Cm", 71, "=B32|"), ("Am", 68, "^G32|"),
    ("F", 70, "B32|"), ("D", 66, "F32|"), ("C#", 65, "E32|"), ("Gb", 59, "C32|")])
def test_out_of_key_notes_lean_the_way_the_key_leans(key, pitch, written):
    """The last two are in the key: E-sharp and C-flat, named as the signature names them."""
    text = tiny(key, "C32|")
    sheet = notation.read(text)
    result = notation.write(text, with_notes(sheet, "Vocal", [(0, 32, pitch)]))
    assert vocal_line(result["abc"]) == written


def test_a_bar_after_a_key_change_is_spelled_in_the_new_key():
    text = HEADER + ("K:C\n% verse\nV: Vocal\nC32|\nV: Ins\nZ|\n"
                     "% chorus\nV: Vocal\nK:Eb\nE32|\nV: Ins\nK:Eb\nZ|\n")
    sheet = notation.read(text)
    assert [bar["key"] for bar in sheet["bars"]] == ["C", "Eb"]
    result = notation.write(text, with_notes(sheet, "Vocal", [(0, 32, 60), (32, 32, 70)]))
    assert result["abc"].splitlines()[16] == "B32|"


def test_an_edit_that_does_not_read_back_is_refused_and_nothing_is_changed(monkeypatch):
    """The last line of defence has to be able to fire."""
    text = tiny("C", "C32|")
    sheet = notation.read(text)
    monkeypatch.setattr(notation, "_bar_text", lambda *args: "D32")
    with pytest.raises(ValueError, match="could not be written"):
        notation.write(text, with_notes(sheet, "Vocal", [(0, 32, 64)]))


def test_a_note_dragged_across_a_barline_is_tied_from_both_ends():
    text = tiny("C", "C16E16|G16c16|")
    sheet = notation.read(text)
    result = notation.write(text, with_notes(sheet, "Vocal",
                                             [(0, 16, 60), (16, 32, 64), (48, 16, 72)]))
    assert result["bars"] == [0, 1]
    assert vocal_line(result["abc"]) == "C16E16-|E16c16|"


def test_an_unchanged_tie_pulls_its_other_bar_along_when_one_end_is_edited():
    text = tiny("C", "C16E16-|E16c16|")
    sheet = notation.read(text)
    result = notation.write(text, with_notes(sheet, "Vocal",
                                             [(0, 16, 62), (16, 32, 64), (48, 16, 72)]))
    assert result["bars"] == [0, 1]
    assert vocal_line(result["abc"]) == "D16E16-|E16c16|"


def test_a_length_the_dialect_cannot_write_in_one_token_is_tied_inside_the_bar():
    text = tiny("C", "C32|")
    sheet = notation.read(text)
    result = notation.write(text, with_notes(sheet, "Vocal", [(0, 10, 60)]))
    assert vocal_line(result["abc"]) == "C8-C2z16z6|"
    assert notes(notation.read(result["abc"])) == [(0, 10, 60)]


def test_a_chord_inside_a_held_note_splits_it_with_a_tie():
    text = tiny("C", "C32|")
    sheet = notation.read(text)
    changed = copy.deepcopy(sheet)
    changed["chords"] = [{"start": 0, "name": "C"}, {"start": 16, "name": "F/C"}]
    result = notation.write(text, changed)
    assert vocal_line(result["abc"]) == '"C"C16-"F/C"C16|'


def test_a_chord_on_a_resting_bar_keeps_the_rest_written_out():
    text = tiny("C", "C32|Z|")
    sheet = notation.read(text)
    changed = copy.deepcopy(sheet)
    changed["chords"] = [{"start": 48, "name": "Am7"}]
    result = notation.write(text, changed)
    assert vocal_line(result["abc"]) == 'C32|z16"Am7"z16|'


def test_an_emptied_bar_is_a_full_rest_and_a_long_rest_opens_for_one_note():
    text = tiny("C", "C32|D32|E32|", ins="Z3|")
    sheet = notation.read(text)
    changed = with_notes(sheet, "Vocal", [(0, 32, 60), (64, 32, 64)])
    changed = with_notes(changed, "Ins", [(32, 32, 55)])
    result = notation.write(text, changed)
    assert vocal_line(result["abc"]) == "C32|Z|E32|"
    assert ins_line(result["abc"]) == "Z|G,32|Z|"
    assert result["bars"] == [1]


def test_a_new_score_is_refused_rather_than_guessed_at():
    with pytest.raises(ValueError, match="no score yet"):
        notation.read("   ")
    with pytest.raises(ValueError, match="cannot be read note by note"):
        notation.read("X:1\nK:C\nCDEF|")


def test_a_bar_that_changes_key_inside_itself_is_left_to_the_abc_text():
    text = tiny("C", "C16[K:G]F16|D32|", ins="z16[K:G]z16|Z|")
    sheet = notation.read(text)
    assert [bar["editable"] for bar in sheet["bars"]] == [False, True]
    items = notes(sheet)
    touched_second = [items[0], items[1], (32, 32, 64)]
    assert notation.write(text, with_notes(sheet, "Vocal", touched_second))["bars"] == [1]
    with pytest.raises(ValueError, match="changes key halfway"):
        notation.write(text, with_notes(sheet, "Vocal", [(0, 16, 62)] + items[1:]))


@pytest.mark.parametrize("change,message", [
    (lambda s: with_notes(s, "Vocal", [(0, 16, 60), (8, 16, 62)]), "overlap in bar 1"),
    (lambda s: with_notes(s, "Vocal", [(40, 32, 60)]), "outside the song"),
    (lambda s: with_notes(s, "Ins", [(0, 8, 128)]), "pitches go from 0 to 127"),
    (lambda s: with_notes(s, "Vocal", [(0, 0, 60)]), "outside the song"),
    (lambda s: dict(s, chords=[{"start": 0, "name": "Cmaj9"}]), "not a chord symbol"),
    (lambda s: dict(s, chords=[{"start": 0, "name": "C"}, {"start": 0, "name": "Dm"}]),
     "same moment"),
    (lambda s: dict(s, notes={"Vocal": [{"start": 1.5, "length": 8, "pitch": 60}]}),
     "whole-number"),
    (lambda s: [], "object with 'notes'"),
])
def test_an_edit_the_score_cannot_hold_is_refused_with_a_reason(change, message):
    text = tiny("C", "C32|C32|")
    with pytest.raises(ValueError, match=message):
        notation.write(text, change(notation.read(text)))


def _random_edit(rng, sheet):
    """One edit a person could make in the roll, applied to a copy of *sheet*."""
    changed = copy.deepcopy(sheet)
    part = rng.choice(abc_tools.VOICES)
    items = sorted(changed["notes"][part], key=lambda n: n["start"])
    action = rng.choice(["pitch", "delete", "shorten", "add", "chord", "unchord"])
    if action in ("pitch", "delete", "shorten") and items:
        note = rng.choice(items)
        if action == "pitch":
            note["pitch"] = max(30, min(100, note["pitch"] + rng.choice([-7, -3, -1, 1, 2, 5])))
        elif action == "delete":
            items.remove(note)
        elif note["length"] > 1:
            note["length"] = rng.randint(1, note["length"] - 1)
    elif action == "add":
        bar = rng.choice(changed["bars"])
        start, end = bar["start"], bar["start"] + bar["length"]
        taken = [(n["start"], n["start"] + n["length"]) for n in items]
        free = start
        for low, high in sorted(taken):
            if high <= free or low >= end:
                continue
            if low > free:
                break
            free = max(free, high)
        limit = min([low for low, _ in taken if low >= free] + [end])
        if free < limit:
            begin = rng.randint(free, limit - 1)
            items.append({"start": begin, "length": rng.randint(1, limit - begin),
                          "pitch": rng.randint(48, 84)})
    elif action == "chord":
        used = {c["start"] for c in changed["chords"]}
        tick = rng.randrange(changed["total"])
        if tick not in used:
            root = rng.choice(["C", "D", "Eb", "F#", "G", "Ab", "B"])
            quality = rng.choice(abc_tools.QUALITIES)
            changed["chords"].append({"start": tick, "name": root + quality})
    elif changed["chords"]:
        changed["chords"].pop(rng.randrange(len(changed["chords"])))
    changed["notes"][part] = items
    return changed


@pytest.mark.parametrize("name,text", SCORES, ids=[name for name, _ in SCORES])
def test_random_edits_read_back_exactly_and_touch_only_their_bars(name, text):
    rng = random.Random(name)
    source = text.strip()
    for _ in range(40):
        sheet = notation.read(source)
        changed = _random_edit(rng, sheet)
        result = notation.write(source, changed)
        back = notation.read(result["abc"])
        for part in abc_tools.VOICES:
            assert notes(back, part) == notes(changed, part)
        assert sorted((c["start"], c["name"]) for c in back["chords"]) == \
            sorted((c["start"], c["name"]) for c in changed["chords"])
        score = abc_tools.parse(source)
        lines, pieces, _sections, _inline = notation._layout(source, score)
        allowed = {piece["line"] for part in abc_tools.VOICES for piece in pieces[part]
                   if set(piece["bars"]) & set(result["bars"])}
        after = result["abc"].splitlines()
        changed_lines = {i for i, line in enumerate(source.splitlines()) if after[i] != line}
        assert changed_lines <= allowed
        source = result["abc"]


def test_the_read_route_draws_a_score_or_says_why_not():
    assert routes.answer_score_read(None)[1] == 400
    assert routes.answer_score_read({"abc": 5})[1] == 400
    payload, status = routes.answer_score_read({"abc": "junk"})
    assert status == 200 and payload["ok"] is False and "cannot be read" in payload["error"]
    payload, status = routes.answer_score_read({"abc": AWKWARD})
    assert status == 200 and payload["ok"] is True
    assert payload["sheet"] == notation.read(AWKWARD)


def test_the_write_route_hands_back_the_text_and_the_score_it_now_reads_as():
    assert routes.answer_score_write({"abc": AWKWARD})[1] == 400
    sheet = notation.read(AWKWARD)
    payload, status = routes.answer_score_write({"abc": AWKWARD, "sheet": sheet})
    assert (status, payload["ok"], payload["abc"], payload["bars"]) == \
        (200, True, AWKWARD.strip(), [])
    payload, status = routes.answer_score_write(
        {"abc": AWKWARD, "sheet": dict(sheet, chords=[{"start": 0, "name": "H7"}])})
    assert status == 200 and payload["ok"] is False and "not a chord symbol" in payload["error"]


def test_a_failure_nobody_foresaw_reaches_the_window_as_a_sentence(monkeypatch):
    def broken(_text):
        raise RuntimeError("boom")
    monkeypatch.setattr(notation, "read", broken)
    payload, status = routes.answer_score_read({"abc": AWKWARD})
    assert status == 200 and payload["ok"] is False and "boom" in payload["error"]
