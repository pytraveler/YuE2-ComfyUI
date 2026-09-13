"""The token cuts the song editor draws, checked against what the model reads.

Most of this runs on hand-made token lists, so it needs neither tiktoken nor the
weights. The last test uses the real vocabulary when this machine has it, and
is the one that proves the cuts come from the prompt in context rather than
from a line on its own.
"""

from __future__ import annotations

import pytest

from yue2_comfy import routes, tokens
from yue2_comfy.vendor.yue2.protocol import SongRequest


class Split:
    """A stand-in tokenizer that cuts the text wherever it is told to."""

    def __init__(self, marks):
        self.marks = marks
        self._enc = self

    def encode(self, text):
        data = text.encode("utf-8")
        edges = sorted({0, len(data), *(m for m in self.marks(text) if 0 < m < len(data))})
        self.pieces = [data[a:b] for a, b in zip(edges, edges[1:])]
        return list(range(len(self.pieces)))

    def decode_single_token_bytes(self, token):
        return self.pieces[token]


def test_the_lyrics_start_right_after_the_lyrics_header():
    text = SongRequest(style="pop", lyrics="la la").text()
    assert text[tokens.lyric_start("pop"):].startswith("la la")


def test_a_cut_on_a_character_edge_is_reported_where_it_falls():
    text = "ab cd"
    pieces = [b"ab", b" cd"]
    assert tokens.cuts(pieces, text, 0, len(text)) == ([2], [])


def test_a_cut_through_a_character_marks_it_torn_and_cuts_after_it():
    """An emoji's four bytes can be spread over two tokens; measured below."""
    text = "a\U0001f3b5b"
    data = text.encode("utf-8")
    pieces = [data[:3], data[3:]]
    assert tokens.cuts(pieces, text, 0, len(text)) == ([2], [1])


def test_cuts_outside_the_stretch_are_left_out_and_the_rest_are_relative():
    text = "head|body|tail"
    pieces = [b"head|", b"bo", b"dy|", b"tail"]
    assert tokens.cuts(pieces, text, 5, 9) == ([2], [])


def test_tokens_that_do_not_spell_the_text_are_refused():
    with pytest.raises(ValueError):
        tokens.cuts([b"abc"], "abd", 0, 3)


def test_offsets_are_moved_back_onto_the_lyrics_as_typed():
    """The nodes strip the lyrics; the editor draws on the unstripped text."""
    word = "record"
    lyrics = "\n  " + word + "\n"

    def marks(text):
        at = len(text.encode("utf-8")) - len("\n") - len(word)
        return [at, at + 3]

    found = tokens.lyric_cuts("pop", lyrics, source=Split(marks))
    assert found["available"]
    assert found["cuts"] == [3 + 3]
    assert lyrics[found["cuts"][0]:].startswith("ord")
    assert found["tokens"] == 2


def test_no_vocabulary_is_a_note_rather_than_an_error(monkeypatch):
    def missing():
        raise FileNotFoundError("no weights")

    monkeypatch.setattr(tokens, "tokenizer", missing)
    found = tokens.lyric_cuts("pop", "la")
    assert found == {"ok": True, "available": False, "problem": tokens.NO_VOCABULARY}


def test_text_the_tokenizer_would_normalise_draws_no_cuts():
    decomposed = "e\u0301"
    found = tokens.lyric_cuts("pop", decomposed, source=Split(lambda text: []))
    assert not found["available"]
    assert found["problem"] == tokens.NOT_NFC


@pytest.mark.parametrize("body, status", [
    (None, 400),
    ([], 400),
    ({"style": 1, "lyrics": ""}, 400),
    ({"style": "", "lyrics": "x" * (routes.LONGEST + 1)}, 413),
])
def test_the_route_refuses_what_it_cannot_read(body, status):
    payload, code = routes.answer_tokens(body)
    assert code == status
    assert payload["ok"] is False


def test_a_tokenizer_failure_reaches_the_editor_as_a_note(monkeypatch):
    def broken(style, lyrics):
        raise RuntimeError("tiktoken is not installed")

    monkeypatch.setattr(tokens, "lyric_cuts", broken)
    payload, code = routes.answer_tokens({"style": "", "lyrics": "la"})
    assert code == 200
    assert payload == {"ok": True, "available": False,
                       "problem": "tiktoken is not installed"}


def real_tokenizer():
    """The vocabulary on this machine, or a skip."""
    pytest.importorskip("tiktoken")
    try:
        return tokens.tokenizer()
    except Exception as error:
        pytest.skip("no YuE2 vocabulary here: {}".format(error))


def test_the_cuts_are_the_prompt_tokens_that_fall_inside_the_lyrics():
    """Rebuilt independently: token texts from the full prompt, sliced by position."""
    made = real_tokenizer()
    style = "English, pop, 88 BPM"
    lyrics = "[Verse]\nNeon fades along the lane,\nrecORD the NIGHT\n\n[Chorus]\n" \
             "\u0434\u043e\u0440\u043e\u0433\u0430 \u0434\u043e\u0440\u041e\u0433\u0430"
    found = tokens.lyric_cuts(style, lyrics)
    assert found["available"]

    marked = [lyrics[a:b] for a, b in zip([0] + found["cuts"], found["cuts"] + [len(lyrics)])]
    assert "".join(marked) == lyrics
    assert "rec" in marked and "ORD" in marked
    assert ",\n" in marked, "the comma and the newline after it are one token in context"
    assert marked[-3:] == [" \u0434\u043e\u0440", "\u041e", "\u0433\u0430"]
    assert found["torn"] == [], "no Cyrillic letter is shared between two tokens"

    alone = [made._enc.decode_single_token_bytes(t) for t in made.encode("recORD")]
    assert alone == [b"rec", b"ORD"]


def test_an_emoji_is_torn_by_the_real_vocabulary_and_drawn_whole():
    real_tokenizer()
    lyrics = "la \U0001f3b5 la"
    found = tokens.lyric_cuts("pop", lyrics)
    assert found["torn"] == [3]
    assert 3 not in found["cuts"] and 4 in found["cuts"]
