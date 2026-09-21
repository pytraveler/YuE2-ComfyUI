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


def named_sections(sheet):
    """The sections of a sheet as an edit sends them back: the named ones, in order."""
    return [{"name": group["name"], "bar": group["bar"]}
            for group in sheet["sections"] if group["name"]]


def with_sections(sheet, items):
    changed = copy.deepcopy(sheet)
    changed["sections"] = items
    return changed


def music_of(text):
    """Everything a moved comment or a finer note length must leave alone."""
    score = abc_tools.parse(text)
    return ([[list(note) for note in score.voices[name].notes] for name in abc_tools.VOICES],
            [score.voices[name].bars for name in abc_tools.VOICES],
            [score.voices[name].keys for name in abc_tools.VOICES],
            score.voices["Vocal"].chords, score.bpm)


def test_a_section_boundary_moves_to_a_bar_that_begins_no_group():
    """The comment can only stand between groups, so the group is cut in two."""
    text = tiny("C", "C8D8E8G8|E8F8G8c8|G8A8B8c8|")
    sheet = notation.read(text)
    assert named_sections(sheet) == [{"name": "verse", "bar": 0}]
    result = notation.write(text, with_sections(
        sheet, [{"name": "verse", "bar": 0}, {"name": "chorus", "bar": 2}]))
    assert result["bars"] == []
    assert notation.read(result["abc"])["sections"] == [
        {"name": "verse", "bar": 0, "bars": 2}, {"name": "chorus", "bar": 2, "bars": 1}]
    assert music_of(result["abc"]) == music_of(text)
    assert result["abc"].splitlines()[8:] == [
        "% verse", "V: Vocal", "C8D8E8G8|E8F8G8c8|", "V: Ins", "Z|Z|",
        "% chorus", "V: Vocal", "G8A8B8c8|", "V: Ins", "Z|"]


def test_a_section_is_renamed_where_it_stands():
    text = tiny("C", "C8D8E8G8|")
    sheet = notation.read(text)
    result = notation.write(text, with_sections(sheet, [{"name": "pre-chorus", "bar": 0}]))
    assert result["bars"] == []
    assert result["abc"] == text.strip().replace("% verse", "% pre-chorus")


def test_a_boundary_taken_away_joins_its_bars_to_the_section_before():
    sheet = notation.read(AWKWARD)
    assert [group["name"] for group in sheet["sections"]] == ["verse", "chorus"]
    result = notation.write(AWKWARD, with_sections(sheet, [{"name": "verse", "bar": 0}]))
    assert notation.read(result["abc"])["sections"] == [{"name": "verse", "bar": 0, "bars": 5}]
    assert music_of(result["abc"]) == music_of(AWKWARD)
    assert "% chorus" not in result["abc"]


def test_a_boundary_inside_a_long_rest_splits_the_rest_and_not_the_music():
    """Three silent bars are written Z3; a section beginning at the second is Z then Z2."""
    text = tiny("C", "C8D8E8G8|Z3|", ins="Z|Z3|")
    sheet = notation.read(text)
    assert len(sheet["bars"]) == 4
    result = notation.write(text, with_sections(
        sheet, [{"name": "verse", "bar": 0}, {"name": "outro", "bar": 2}]))
    assert music_of(result["abc"]) == music_of(text)
    assert notation.read(result["abc"])["sections"] == [
        {"name": "verse", "bar": 0, "bars": 2}, {"name": "outro", "bar": 2, "bars": 2}]
    assert result["abc"].splitlines()[8:] == [
        "% verse", "V: Vocal", "C8D8E8G8|Z|", "V: Ins", "Z|Z|",
        "% outro", "V: Vocal", "Z2|", "V: Ins", "Z2|"]


def test_a_section_name_keeps_every_other_line_where_it_was():
    """Only the comment lines move: a diff of a renaming is one line."""
    sheet = notation.read(AWKWARD)
    result = notation.write(AWKWARD, with_sections(
        sheet, [{"name": "intro", "bar": 0}, {"name": "chorus", "bar": 3}]))
    before = AWKWARD.strip().splitlines()
    after = result["abc"].splitlines()
    assert len(before) == len(after)
    assert [i for i, (one, other) in enumerate(zip(before, after)) if one != other] == [8]


@pytest.mark.parametrize("name,text", SCORES, ids=[name for name, _ in SCORES])
def test_a_section_can_begin_at_any_bar_of_any_score(name, text):
    """A boundary lands where it is asked for, and no bar of music is rewritten."""
    source = text.strip()
    sheet = notation.read(source)
    total = len(sheet["bars"])
    here = named_sections(sheet)
    for bar in sorted({0, 1, total - 1} | set(range(0, total, 5))):
        wanted = [dict(item) for item in here if item["bar"] != bar]
        wanted.append({"name": "bridge", "bar": bar})
        wanted.sort(key=lambda item: item["bar"])
        result = notation.write(source, with_sections(sheet, wanted))
        assert result["bars"] == []
        assert music_of(result["abc"]) == music_of(source)
        assert named_sections(notation.read(result["abc"])) == wanted


@pytest.mark.parametrize("sections,message", [
    ([{"name": "verse", "bar": 0}, {"name": "chorus", "bar": 0}], "Two sections start at bar 1"),
    ([{"name": "verse", "bar": 99}], "this song has"),
    ([{"name": "x" * 60, "bar": 0}], "cannot be a section name"),
    ([{"name": "verse"}], "whole-number bar"),
    ("verse", "must be a list"),
])
def test_sections_an_edit_cannot_hold_are_refused_with_a_sentence(sections, message):
    text = tiny("C", "C8D8E8G8|E8F8G8c8|G8A8B8c8|")
    sheet = notation.read(text)
    with pytest.raises(ValueError) as raised:
        notation.write(text, with_sections(sheet, sections))
    assert message in str(raised.value)


def sixteenths(vocal, ins=None):
    """A score on L:1/16, where a thirty-second note cannot be written at all."""
    return tiny("C", vocal, ins).replace("L:1/32", "L:1/16")


def test_a_finer_note_length_writes_the_same_song_on_a_grid_that_holds_it():
    text = sixteenths("C4D4E4G4|")
    sheet = notation.read(text)
    assert (sheet["unit"], sheet["total"]) == (16, 16)
    doubled = {"notes": {part: [{"start": n["start"] * 2, "length": n["length"] * 2,
                                 "pitch": n["pitch"]} for n in rows]
                         for part, rows in sheet["notes"].items()},
               "chords": [{"start": c["start"] * 2, "name": c["name"]} for c in sheet["chords"]],
               "unit": 32}
    result = notation.write(text, doubled)
    after = notation.read(result["abc"])
    assert (after["unit"], after["per_quarter"], after["total"]) == (32, 8, 32)
    assert result["abc"].splitlines()[notation.UNIT_LINE] == "L:1/32"
    assert vocal_line(result["abc"]) == "C8D8E8G8|"
    assert music_of(result["abc"]) == music_of(text)
    assert after["seconds"] == sheet["seconds"]


def test_a_finer_grid_is_what_lets_a_note_be_half_a_sixteenth():
    """The point of the rewrite: the dialect has no fractions, only whole units."""
    text = sixteenths("C4D4E4G4|")
    sheet = notation.read(text)
    finer = notation.write(text, {"notes": {"Vocal": [{"start": 0, "length": 1, "pitch": 60}],
                                            "Ins": []},
                                  "chords": [], "unit": 32})
    assert notation.read(finer["abc"])["notes"]["Vocal"] == [{"start": 0, "length": 1, "pitch": 60}]
    assert notation.read(finer["abc"])["unit"] == 32
    assert sheet["unit"] == 16


@pytest.mark.parametrize("name,text", SCORES, ids=[name for name, _ in SCORES])
def test_a_score_already_that_fine_is_handed_back_untouched(name, text):
    source = text.strip()
    sheet = notation.read(source)
    if sheet["unit"] >= notation.FINEST:
        assert notation.write(source, dict(sheet, unit=notation.FINEST))["abc"] == source
    assert notation.write(source, dict(sheet, unit=4))["abc"] == source


@pytest.mark.parametrize("unit", [48, 64, 33])
def test_a_note_length_the_roll_does_not_offer_is_refused(unit):
    text = sixteenths("C4D4E4G4|")
    with pytest.raises(ValueError) as raised:
        notation.write(text, dict(notation.read(text), unit=unit))
    assert "at the most" in str(raised.value)


def test_a_tempo_moves_the_header_line_and_nothing_else():
    """One number, one line: the notes keep their lengths and the song is sung faster."""
    text = tiny("C", "C8D8E8G8|")
    sheet = notation.read(text)
    result = notation.write(text, dict(sheet, bpm=120))
    assert result["bars"] == []
    assert result["abc"].splitlines()[notation.TEMPO_LINE] == "Q:1/4=120"
    assert result["abc"].replace("Q:1/4=120", "Q:1/4=90") == text.strip()
    again = notation.read(result["abc"])
    assert again["bpm"] == 120
    assert again["notes"] == sheet["notes"]
    assert again["total"] == sheet["total"]
    assert again["seconds"] < sheet["seconds"]


def test_a_tempo_and_a_changed_note_travel_together():
    text = tiny("D", '"D"d8f8a8f8|"G"g8b8d\'8b8|')
    sheet = notation.read(text)
    items = notes(sheet)
    items[1] = (8, 8, 79)
    result = notation.write(text, dict(with_notes(sheet, "Vocal", items), bpm=140))
    assert result["bars"] == [0]
    assert vocal_line(result["abc"]) == '"D"d8g8a8f8|"G"g8b8d\'8b8|'
    assert notation.read(result["abc"])["bpm"] == 140


def test_a_sheet_with_no_tempo_in_it_leaves_the_score_where_it_was():
    """Every caller that does not offer a tempo goes on sending what it always sent."""
    text = tiny("C", "C8D8E8G8|")
    sheet = notation.read(text)
    sheet.pop("bpm")
    assert notation.write(text, dict(sheet, bpm=None)) == {"abc": text.strip(), "bars": []}
    assert notation.write(text, sheet) == {"abc": text.strip(), "bars": []}


def test_a_tempo_the_dialect_cannot_hold_is_refused_with_a_sentence():
    text = tiny("C", "C8D8E8G8|")
    sheet = notation.read(text)
    with pytest.raises(ValueError) as problem:
        notation.write(text, dict(sheet, bpm=400))
    assert "outside 40 to 200" in str(problem.value)
    for junk in ("fast", True, float("nan")):
        with pytest.raises(ValueError):
            notation.write(text, dict(sheet, bpm=junk))
    assert notation.write(text, dict(sheet, bpm=96.4))["abc"].splitlines()[
        notation.TEMPO_LINE] == "Q:1/4=96"


def test_a_score_faster_than_the_range_keeps_its_own_tempo():
    """A transcription of something fast opens and closes without being pulled back to 200."""
    text = tiny("C", "C8D8E8G8|").replace("Q:1/4=90", "Q:1/4=210")
    sheet = notation.read(text)
    assert notation.write(text, sheet) == {"abc": text.strip(), "bars": []}
    assert notation.write(text, dict(sheet, bpm=205))["abc"].splitlines()[
        notation.TEMPO_LINE] == "Q:1/4=205"
    with pytest.raises(ValueError):
        notation.write(text, dict(sheet, bpm=215))


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


def test_the_write_route_carries_a_new_tempo_into_the_sheet_it_reads_back():
    sheet = notation.read(AWKWARD)
    payload, status = routes.answer_score_write({"abc": AWKWARD, "sheet": dict(sheet, bpm=72)})
    assert status == 200 and payload["ok"] is True
    assert payload["bars"] == []
    assert payload["sheet"]["bpm"] == 72
    assert payload["sheet"]["notes"] == sheet["notes"]
    payload, status = routes.answer_score_write({"abc": AWKWARD, "sheet": dict(sheet, bpm=9)})
    assert status == 200 and payload["ok"] is False and "outside" in payload["error"]


def test_a_blank_score_is_the_right_length_and_holds_no_notes():
    for bars in (1, 3, 4, 5, 7, 16, 33):
        sheet = notation.read(notation.blank(bars))
        assert len(sheet["bars"]) == bars
        assert sheet["notes"] == {"Vocal": [], "Ins": []}
        assert sheet["chords"] == []
        assert sheet["bpm"] == notation.BLANK_BPM
        assert sheet["unit"] == notation.BLANK_UNIT
        assert [group["name"] for group in sheet["sections"]] == [notation.BLANK_SECTION]
    head = notation.blank().splitlines()[:notation.HEADER_LINES]
    assert head == ["X:1", "T:", "M:" + notation.BLANK_METER,
                    "L:1/{}".format(notation.BLANK_UNIT),
                    "Q:1/4={}".format(notation.BLANK_BPM)] \
        + AWKWARD.splitlines()[5:7] + ["K:" + notation.BLANK_KEY]
    assert abc_tools.parse(notation.blank())


def test_a_blank_score_takes_a_tempo_and_refuses_a_length_nothing_could_hold():
    assert notation.read(notation.blank(4, 90))["bpm"] == 90
    for bars in (0, -1, notation.MOST_BARS + 1, 2.5, True, "8"):
        with pytest.raises(ValueError, match="at most"):
            notation.blank(bars)
    for bpm in (9, 400, "120"):
        with pytest.raises(ValueError, match="outside|BPM"):
            notation.blank(8, bpm)


@pytest.mark.parametrize("name,text", SCORES)
def test_added_bars_leave_every_note_of_the_song_where_it_was(name, text):
    before = notation.read(text)
    added = 7
    longer = notation.lengthened(text, len(before["bars"]) + added)
    after = notation.read(longer)
    assert len(after["bars"]) == len(before["bars"]) + added
    assert after["notes"] == before["notes"], name
    assert after["chords"] == before["chords"], name
    assert longer.startswith(text.strip())
    assert after["total"] > before["total"]
    for group, was in zip(after["sections"], before["sections"]):
        assert group["bar"] == was["bar"] and group["name"] == was["name"]


def test_the_last_section_runs_on_into_the_bars_that_were_added():
    text = tiny("C", "C16D16|Z|")
    after = notation.read(notation.lengthened(text, 6))
    assert [(g["name"], g["bar"], g["bars"]) for g in after["sections"]] == [("verse", 0, 6)]


def test_added_bars_keep_the_line_endings_the_score_came_with():
    windows = AWKWARD.replace("\n", "\r\n")
    longer = notation.lengthened(windows, len(notation.read(windows)["bars"]) + 3)
    assert "\r\n" in longer and "\n" not in longer.replace("\r\n", "")
    assert len(notation.read(longer)["bars"]) == len(notation.read(AWKWARD)["bars"]) + 3


def test_a_score_is_only_made_longer_never_shorter():
    bars = len(notation.read(AWKWARD)["bars"])
    for asked in (bars, bars - 1, 1):
        with pytest.raises(ValueError, match="only makes a song longer"):
            notation.lengthened(AWKWARD, asked)
    with pytest.raises(ValueError, match="at most"):
        notation.lengthened(AWKWARD, notation.MOST_BARS + 1)
    with pytest.raises(ValueError, match="cannot be read"):
        notation.lengthened("junk", 40)


def test_the_length_route_makes_a_score_from_nothing_and_makes_one_longer():
    assert routes.answer_score_length(None)[1] == 400
    assert routes.answer_score_length({"abc": ""})[1] == 400
    assert routes.answer_score_length({"abc": "", "bars": "8"})[1] == 400
    assert routes.answer_score_length({"abc": "", "bars": True})[1] == 400
    assert routes.answer_score_length({"abc": "", "bars": 8, "bpm": "90"})[1] == 400

    payload, status = routes.answer_score_length({"abc": "   ", "bars": 8})
    assert (status, payload["ok"]) == (200, True)
    assert len(notation.read(payload["abc"])["bars"]) == 8

    payload, status = routes.answer_score_length({"abc": AWKWARD, "bars": 9})
    assert (status, payload["ok"]) == (200, True)
    assert len(notation.read(payload["abc"])["bars"]) == 9
    assert notation.read(payload["abc"])["notes"] == notation.read(AWKWARD)["notes"]

    payload, status = routes.answer_score_length({"abc": AWKWARD, "bars": 2})
    assert status == 200 and payload["ok"] is False and "longer" in payload["error"]


CUTTABLE = HEADER + (
    'K:C\n% intro\nV: Vocal\nZ2|\nV: Ins\nc32|e32|\n'
    '% verse\nV: Vocal\nC8D8E8F8|G8A8B8c8-|c8B8A8G8|\nV: Ins\nZ|e32|Z|\n'
    '% chorus\nV: Vocal\nE16G16|C32|\nV: Ins\nc32-|c32|\n'
    '% outro\nV: Vocal\nZ|\nV: Ins\nC32|\n')
"""Eight bars: intro 0-1, verse 2-4, chorus 5-6, outro 7. A Vocal tie from bar 3
into 4, an Ins tie from bar 5 into 6, and whole-bar rests on both sides of an Ins
bar that can be cut."""


def bar_texts(text):
    """Each part's bars as written, a Z2 to Z4 rest counted as that many Z bars."""
    score = abc_tools.parse(text.strip())
    lines = text.strip().splitlines()
    found = {name: [] for name in abc_tools.VOICES}
    for index, name in sorted(score.music_lines.items()):
        for piece in lines[index][:-1].split("|"):
            rest = notation.FULL_REST.fullmatch(piece)
            found[name].extend(["Z"] * int(rest.group(1) or 1) if rest else [piece])
    return found


def test_a_cut_takes_whole_bars_out_and_leaves_every_other_line_as_it_was():
    cut = notation.without(CUTTABLE, 5, 7)
    assert cut == CUTTABLE.replace('% chorus\nV: Vocal\nE16G16|C32|\nV: Ins\nc32-|c32|\n', '')
    assert len(notation.read(cut)["bars"]) == 6


def test_a_line_that_loses_bars_is_written_again_with_its_rests_folded():
    cut = notation.without(CUTTABLE, 3, 4)
    assert "V: Vocal\nC8D8E8F8|c8B8A8G8|\nV: Ins\nZ2|\n" in cut
    assert cut.count("\n") == CUTTABLE.count("\n")


def test_a_tie_into_the_cut_is_taken_off_in_both_parts():
    assert "C8D8E8F8|G8A8B8c8|\n" in notation.without(CUTTABLE, 4, 5)
    cut = notation.without(CUTTABLE, 6, 7)
    assert "V: Vocal\nE16G16|\nV: Ins\nc32|\n" in cut


def test_a_cut_takes_every_name_written_above_the_section():
    """A group may carry several names; all of them go with its bars, and none of the others."""
    named = CUTTABLE.replace('% chorus\n', '% chorus\n% hook\n').replace('% outro\n', '% outro\n% end\n')
    assert notation.without(named, 5, 7) == notation.without(CUTTABLE, 5, 7).replace(
        '% outro\n', '% outro\n% end\n')
    assert notation.without(named, 7, 8) == CUTTABLE.replace('% chorus\n', '% chorus\n% hook\n').replace(
        '% outro\nV: Vocal\nZ|\nV: Ins\nC32|\n', '')


def test_a_cut_to_the_end_leaves_no_tie_hanging_and_no_empty_section():
    cut = notation.without(CUTTABLE, 6, 8)
    assert "% outro" not in cut and cut.endswith("V: Ins\nc32|\n")
    assert len(notation.read(cut)["bars"]) == 6


def test_a_cut_keeps_what_surrounds_the_score():
    wrapped = "\n" + CUTTABLE + "\n"
    cut = notation.without(wrapped, 5, 7)
    assert cut.startswith("\nX:1") and cut.endswith("C32|\n\n")


@pytest.mark.parametrize("start,stop,message", [
    (0, 8, "every bar"), (3, 9, "not bars"), (-1, 2, "not bars"), (4, 4, "not bars")])
def test_a_cut_the_score_cannot_take_is_refused(start, stop, message):
    with pytest.raises(ValueError, match=message):
        notation.without(CUTTABLE, start, stop)


def test_a_bar_that_changes_key_inside_itself_is_not_cut_out():
    text = tiny("C", "C16[K:G]F16|D32|", ins="z16[K:G]z16|Z|")
    with pytest.raises(ValueError, match="changes key halfway"):
        notation.without(text, 0, 1)


def test_a_cut_that_would_take_a_key_change_with_it_is_refused_by_reading_back():
    text = HEADER + ('K:C\n% verse\nV: Vocal\nC32|\nV: Ins\nZ|\n'
                     '% chorus\nV: Vocal\nK:G\nG32|\nV: Ins\nK:G\nZ|\n'
                     '% verse\nV: Vocal\nD32|\nV: Ins\nZ|\n')
    with pytest.raises(ValueError, match="could not be cut.*changed its key"):
        notation.without(text, 1, 2)
    assert "K:G" in notation.without(text, 2, 3)


@pytest.mark.parametrize("name,text", SCORES, ids=[name for name, _ in SCORES])
def test_random_cuts_leave_every_other_bar_as_it_was_written(name, text):
    rng = random.Random(name)
    before = bar_texts(text)
    count = len(before["Vocal"])
    done = 0
    for _ in range(8):
        start = rng.randrange(count - 1)
        stop = rng.randrange(start + 1, min(count, start + 12) + 1)
        if stop - start >= count:
            continue
        try:
            cut = notation.without(text, start, stop)
        except ValueError as error:
            assert "could not be cut" in str(error) or "changes key" in str(error)
            continue
        done += 1
        after = bar_texts(cut)
        assert cut.endswith(text[len(text.rstrip()):])
        for part in abc_tools.VOICES:
            kept = before[part][:start] + before[part][stop:]
            if start and kept[start - 1].rstrip().endswith("-"):
                kept[start - 1] = kept[start - 1].rstrip()[:-1]
            assert after[part] == kept, (part, start, stop)
    assert done >= 4
