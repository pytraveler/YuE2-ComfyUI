"""Cutting a whole song's words into sections by rougher section-wise hearings, and the sections in seconds."""

from __future__ import annotations

from yue2_comfy.asr import align
from yue2_comfy.sheetsage import sections

WHOLE = ("Over the hills, the evening falls. Lanterns are lit along the walls. "
         "Turn with me slowly, turn with me round, until the morning finds the ground.")


def test_words_compare_without_case_punctuation_or_yo():
    assert align.normal("Hills,") == "hills"
    assert align.normal("\u0401\u0436") == "\u0435\u0436"
    assert align.normal("--") == ""


def test_the_whole_text_is_cut_where_the_guides_change():
    guides = ["Over the hills the evening falls lanterns are lit along the walls",
              "turn with me slowly turn with me round until the morning finds the ground"]
    parts = align.split(WHOLE, guides)
    assert parts == ["Over the hills, the evening falls. Lanterns are lit along the walls.",
                     "Turn with me slowly, turn with me round, until the morning finds the ground."]


def test_misheard_guides_still_place_the_cut_and_the_words_come_from_the_whole_song():
    guides = ["Over the hill, the evening fall. Lantern are lit along the wall.",
              "Turn with me slowly, turn with me around until the morning fine the ground"]
    parts = align.split(WHOLE, guides)
    assert parts[0].endswith("along the walls.") and parts[1].startswith("Turn with me slowly")


def test_every_word_is_kept_once_in_order():
    guides = ["Over the", "hills the evening", "", "falls lanterns are lit", "turn with me slowly round the ground"]
    parts = align.split(WHOLE, guides)
    assert " ".join(part for part in parts if part) == WHOLE
    assert parts[2] == ""


def test_a_cut_between_matches_moves_to_the_nearby_comma():
    whole = "one two three four five, six seven eight nine ten"
    guides = ["one two three x y", "z w eight nine ten"]
    assert align.split(whole, guides) == ["one two three four five,", "six seven eight nine ten"]


def test_a_pickup_after_the_last_full_stop_opens_the_next_section():
    assert align.move_pickups(["Falls down. And then", "we run."]) == ["Falls down.", "And then we run."]
    assert align.move_pickups(["Falls down and then sings on", "we run."]) == ["Falls down and then sings on", "we run."]
    assert align.move_pickups(["Falls. One two three four", "we run."]) == ["Falls. One two three four", "we run."]


def test_without_guides_to_match_the_words_stay_together():
    assert align.split("a b c", ["x y", "z w"]) == ["a b c", ""]
    assert align.split("a b c", ["anything"]) == ["a b c"]
    assert align.split("", ["x", "y"]) == ["", ""]
    assert align.split("a b c", ["", "x y"]) == ["", "a b c"]


def test_sections_in_seconds_merge_repeated_labels_and_count_vocal_notes():
    rows = {"structures": [[0.2, 5.0, "intro"], [5.0, 9.0, "intro"], [9.0, 20.0, "verse"], [20.0, 25.0, "verse"],
                           [25.0, 40.0, "chorus"]],
            "notes": [[1.0, 1.5, 60, 1], [9.5, 10.0, 62, 0], [19.9, 20.4, 64, 0], [21.0, 22.0, 65, 0],
                      [30.0, 31.0, 67, 0], [30.0, 31.0, 48, 1]]}
    found = sections.timed(rows, 42.0)
    assert [(s["label"], s["tag"], s["start"], s["end"], s["notes"]) for s in found] == [
        ("intro", "Intro", 0.0, 9.0, 0), ("verse", "Verse", 9.0, 25.0, 3), ("chorus", "Chorus", 25.0, 42.0, 1)]


def test_a_transcription_without_sections_is_one_verse():
    assert sections.timed({"structures": [], "notes": [[1.0, 2.0, 60, 0]]}, 10.0) == [
        {"label": "", "tag": "Verse", "start": 0.0, "end": 10.0, "notes": 1}]
