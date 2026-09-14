"""An edit, and the words it belongs to.

The whole promise of the mark is two sentences: an edit made for these words is
sung, and an edit made for other words is not. Everything below pins one side
of that, or the one thing that would quietly break it -- the mark reaching the
model as part of the score.
"""

from __future__ import annotations

import inspect

from yue2_comfy import edits

SCORE = "X:1\nK:C\n% verse\nV: Vocal\nCDEF|\n"
INSTEAD = "something else happened"


def test_the_mark_names_the_words_and_not_how_they_were_trimmed():
    """The nodes trim the style and lyrics before the model reads them, so the mark does too."""
    words = edits.mark("style", "lyrics", "full")
    assert len(words) == 16 and all(c in "0123456789abcdef" for c in words)
    assert edits.mark(" style\n", "lyrics\n\n", "full") == words


def test_every_part_of_the_words_changes_the_mark():
    words = edits.mark("style", "lyrics", "full")
    assert edits.mark("style.", "lyrics", "full") != words
    assert edits.mark("style", "Lyrics", "full") != words
    assert edits.mark("style", "lyrics", "melody") != words
    assert edits.mark("lyrics", "style", "full") != words


def test_the_seed_is_not_part_of_the_mark():
    """A new seed sings the same edit as a new take: that is how a bar that did not take is tried again."""
    assert list(inspect.signature(edits.mark).parameters) == ["style", "lyrics", "cot"]


def test_a_mark_comes_off_exactly_as_it_went_on():
    words = edits.mark("s", "l", "full")
    marked = edits.attach(SCORE, words)
    assert marked == SCORE.rstrip() + "\n%yue2-words " + words
    assert edits.read(marked) == edits.Edit(SCORE.strip(), words)


def test_a_score_without_a_mark_reads_as_it_always_did():
    """Trimmed at the ends and otherwise untouched, as the render node has always taken it."""
    assert edits.read("\n\n" + SCORE + "\n") == edits.Edit(SCORE.strip(), None)
    assert edits.read("") == edits.Edit("", None)
    assert edits.read(None) == edits.Edit("", None)
    assert edits.attach(SCORE, None) == SCORE.rstrip()


def test_windows_line_ends_and_a_mark_out_of_place_are_read_too():
    words = edits.mark("s", "l", "full")
    crlf = edits.attach(SCORE, words).replace("\n", "\r\n") + "\r\n"
    assert edits.read(crlf).words == words
    assert "%yue2-words" not in edits.read(crlf).score
    middle = "X:1\n%yue2-words " + words + "\nK:C\nCDEF|"
    assert edits.read(middle) == edits.Edit("X:1\nK:C\nCDEF|", words)


def test_a_comment_that_only_looks_like_a_mark_stays_in_the_score():
    """Taking out a line the editor did not write would change the score behind a person's back."""
    text = SCORE + "%yue2-words not-a-mark\n% yue2-words 0123456789abcdef\n%yue2-words 0123456789ABCDEF"
    found = edits.read(text)
    assert found.words is None
    assert found.score == text.strip()


def test_an_edit_for_these_words_is_sung():
    edit = edits.read(edits.attach(SCORE, edits.mark("s", "l", "full")))
    assert edits.mismatch(edit, "s", "l", "full", INSTEAD) == ""


def test_an_edit_for_other_words_is_not_and_the_node_says_what_it_did_instead():
    edit = edits.read(edits.attach(SCORE, edits.mark("s", "l", "full")))
    for style, lyrics, cot in (("s", "other", "full"), ("other", "l", "full"), ("s", "l", "melody")):
        said = edits.mismatch(edit, style, lyrics, cot, INSTEAD)
        assert "other words" in said and INSTEAD in said


def test_an_unmarked_edit_is_sung_whatever_the_words():
    """Pasted, wired in, or saved before marks existed: nobody said which words it was for."""
    assert edits.mismatch(edits.read(SCORE), "any", "words", "full", INSTEAD) == ""


def test_no_edit_is_never_a_mismatch():
    assert edits.mismatch(edits.read(""), "s", "l", "off", INSTEAD) == ""
    assert edits.mismatch(edits.read("%yue2-words 0123456789abcdef"), "s", "l", "full", INSTEAD) == ""


def test_with_cot_off_no_edit_is_sung_because_nothing_is():
    for text in (SCORE, edits.attach(SCORE, edits.mark("s", "l", "off"))):
        assert edits.mismatch(edits.read(text), "s", "l", "off", INSTEAD) == edits.COT_OFF
