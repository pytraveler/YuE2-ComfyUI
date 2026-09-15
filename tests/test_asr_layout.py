"""Laying recognised words out as lyrics: the model's lines kept only when faithful, punctuation otherwise."""

from __future__ import annotations

from yue2_comfy.asr import layout

SECTIONS = [
    {"tag": "Intro", "text": ""},
    {"tag": "Verse", "text": "Over the hills, the evening falls. Lanterns are lit along the walls."},
    {"tag": "Chorus", "text": "Turn with me slowly, turn with me round, oh oh oh, until the morning finds the ground."},
]

GOOD = """[Verse]
Over the hills the evening falls
Lanterns are lit along the walls

[Chorus]
Turn with me slowly, turn with me round
Until the morning finds the ground"""


def test_the_request_holds_only_sections_with_words_under_their_tags():
    chat = layout.messages(SECTIONS, "English")
    assert chat[0]["role"] == "system" and "Keep every word" in chat[0]["content"]
    assert chat[1]["content"] == ("The words are in English.\n\n[Verse]\n" + SECTIONS[1]["text"]
                                  + "\n\n[Chorus]\n" + SECTIONS[2]["text"])


def test_a_faithful_answer_is_kept_and_may_drop_ad_libs():
    lyrics, kept = layout.lay_out(SECTIONS, GOOD)
    assert kept == 2
    assert lyrics == GOOD


def test_thinking_before_the_answer_is_ignored():
    lyrics, kept = layout.lay_out(SECTIONS, "<think>lines, lines</think>\n" + GOOD)
    assert kept == 2 and lyrics == GOOD


def test_a_changed_word_sends_only_that_section_to_the_punctuation():
    answer = GOOD.replace("Lanterns are lit", "Lamps are lit")
    lyrics, kept = layout.lay_out(SECTIONS, answer)
    assert kept == 1
    assert lyrics.startswith("[Verse]\nOver the hills, the evening falls\nLanterns are lit along the walls\n\n[Chorus]\nTurn")


def test_an_answer_that_drops_most_words_is_not_kept():
    answer = "[Verse]\nOver the hills\n\n[Chorus]\nTurn with me slowly"
    assert layout.lay_out(SECTIONS, answer)[1] == 0


def test_moved_words_and_other_tags_are_not_kept():
    moved = GOOD.replace("Over the hills the evening falls", "The evening falls over the hills")
    assert layout.lay_out(SECTIONS, moved)[1] == 1
    assert layout.lay_out(SECTIONS, GOOD.replace("[Chorus]", "[Bridge]"))[1] == 0
    assert layout.lay_out(SECTIONS, "")[1] == 0


def test_without_a_model_lines_break_at_full_stops_long_commas_and_ten_words():
    assert layout.punctuated_lines("One two. Three four five six, seven eight, nine") == \
        ["One two", "Three four five six", "seven eight, nine"]
    assert layout.punctuated_lines(" ".join(str(n) for n in range(12))) == \
        ["0 1 2 3 4 5 6 7 8 9", "10 11"]


def test_a_single_word_after_a_comma_stays_on_its_line():
    assert layout.punctuated_lines("You wait for me, wait, wait. Touch it never, only look, look.") == \
        ["You wait for me, wait, wait", "Touch it never, only look, look"]


def test_faithful_ignores_case_and_punctuation():
    assert layout.faithful(["over THE hills", "the evening falls!"], "Over the hills, the evening falls.")
    assert not layout.faithful(["over the hills the evening falls again"], "Over the hills, the evening falls.")
