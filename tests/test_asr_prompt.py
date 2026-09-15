"""The request around the audio and the reading of Qwen3-ASR's answer, without torch or weights."""

from __future__ import annotations

import pytest

from yue2_comfy.asr import prompt


def closed_form(frames):
    """The release's own arithmetic for placeholder tokens: whole chunks give 13, the rest is halved three times."""
    rest = frames % 100
    rest = (rest - 1) // 2 + 1
    rest = (rest - 1) // 2 + 1
    return (rest - 1) // 2 + 1 + (frames // 100) * 13


@pytest.mark.parametrize("frames", list(range(0, 450)) + [999, 1000, 1001, 6000, 18037])
def test_placeholder_tokens_follow_the_release_arithmetic(frames):
    assert prompt.audio_tokens(frames) == closed_form(frames)


def test_short_audio_is_heard_as_half_a_second():
    assert prompt.mel_frames(100) == 50
    assert prompt.mel_frames(16000 * 3) == 300
    assert prompt.mel_frames(16000 * 3 + 159) == 300


def test_attention_groups_eight_chunks_then_the_rest():
    assert prompt.attention_windows(1000) == [104, 26]
    assert prompt.attention_windows(250) == [33]
    assert prompt.attention_windows(50) == [7]
    assert prompt.attention_windows(1600) == [104, 104]
    assert prompt.attention_windows(0) == []


def fake_encode(text):
    return [ord(character) for character in text]


def test_the_request_is_the_chat_with_one_placeholder_per_audio_frame():
    ids = prompt.request_ids(fake_encode, 300)
    head = fake_encode("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>")
    tail = fake_encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")
    assert ids == head + [prompt.AUDIO_PAD] * 39 + tail


def test_a_named_language_opens_the_answer():
    ids = prompt.request_ids(fake_encode, 100, language="Russian")
    assert "".join(chr(token) for token in ids[-26:]) == "language Russian<asr_text>"


@pytest.mark.parametrize("given, name", [("ru", "Russian"), ("russian", "Russian"), ("English", "English"),
                                         ("", ""), (None, ""), ("auto", "")])
def test_languages_are_named_as_the_model_knows_them(given, name):
    assert prompt.language_name(given) == name


def test_an_unknown_language_is_refused():
    with pytest.raises(ValueError):
        prompt.language_name("Klingon")


def test_the_answer_gives_the_language_it_names_and_the_words_after_the_marker():
    assert prompt.read_answer("language English<asr_text>Over the hills ") == \
        {"language": "English", "text": "Over the hills"}


def test_an_answer_that_hears_no_language_has_none():
    assert prompt.read_answer("language None<asr_text>") == {"language": None, "text": ""}


def test_an_answer_without_the_marker_is_all_words():
    assert prompt.read_answer("just words") == {"language": None, "text": "just words"}


def test_an_answer_cut_off_after_its_language_line_heard_no_words():
    assert prompt.read_answer("language Russian") == {"language": "Russian", "text": ""}
    assert prompt.read_answer("language Russian\n") == {"language": "Russian", "text": ""}
    assert prompt.read_answer("language is a gift") == {"language": None, "text": "language is a gift"}


def test_with_the_language_in_the_request_the_whole_answer_is_words():
    assert prompt.read_answer("Over the hills", "English") == {"language": "English", "text": "Over the hills"}


def test_a_model_stuck_on_a_sound_is_cut_back_to_once():
    assert prompt.collapse_repeats("a" * 25 + "b") == "ab"
    assert prompt.collapse_repeats("go " + "la " * 25 + "end") == "go la end"


def test_ordinary_repetition_in_a_song_is_kept():
    chorus = "la la la la, hey hey hey"
    assert prompt.collapse_repeats(chorus) == chorus
    assert prompt.collapse_repeats("a" * 20) == "a" * 20
