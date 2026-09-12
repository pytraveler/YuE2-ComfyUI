"""The unit conversions the options node relies on."""

import pytest

from yue2_comfy import constants


def test_one_frame_is_forty_milliseconds():
    """Measured against a real run: 679 frames came back as 27.16 seconds."""
    assert constants.FRAME_SECONDS == pytest.approx(0.04)
    assert constants.tokens_to_seconds(679) == pytest.approx(27.16)


def test_seconds_round_trip_to_tokens():
    for seconds in (4.0, 30.0, 180.0, constants.MAX_SECONDS):
        tokens = constants.seconds_to_tokens(seconds)
        assert constants.tokens_to_seconds(tokens) == pytest.approx(seconds, abs=0.04)


def test_the_ceiling_matches_the_released_token_budget():
    """360 seconds is exactly the 9000-token default the model shipped with."""
    assert constants.seconds_to_tokens(constants.MAX_SECONDS) == 9000


def test_normalize_seed_folds_rather_than_raises():
    assert constants.normalize_seed(831001) == 831001
    assert 0 <= constants.normalize_seed(0xFFFFFFFFFFFFFFFF) < (1 << 63)
    assert 0 <= constants.normalize_seed(-1) < (1 << 63)


def test_install_command_names_this_interpreter():
    line = constants.install_command("tiktoken")
    assert "pip install tiktoken" in line
    assert "python" in line.lower()


def test_section_markers_are_not_sung():
    lyrics = "[Verse]\nNeon fades\nFootsteps keep time\n\n[Chorus]\nLet the day come"
    assert constants.sung_lines(lyrics) == 3
    assert constants.sung_lines("") == 0
    assert constants.sung_lines("   \n\n") == 0


def test_the_automatic_ceiling_clears_a_measured_song():
    """Four sung lines ran to 34 seconds before the model stopped on its own.

    The ceiling has to sit comfortably above that -- it exists to catch a song
    that will not end, and cutting off one that would have ended is worse than
    letting it run a little long.
    """
    ceiling = constants.auto_seconds("[Verse]\na\nb\n\n[Chorus]\nc\nd")
    assert ceiling >= 34.0 * 1.5
    assert ceiling <= 90.0


def test_the_automatic_ceiling_has_no_lyrics_to_count():
    """An instrumental gives nothing to measure, so the released default stands."""
    assert constants.auto_seconds("") == constants.AUTO_INSTRUMENTAL_SECONDS
    assert constants.auto_seconds("[Intro]") == constants.AUTO_INSTRUMENTAL_SECONDS


def test_the_automatic_ceiling_stays_inside_the_model_ceiling():
    assert constants.auto_seconds("line\n" * 200) == constants.MAX_SECONDS
    assert constants.auto_seconds("one line") >= constants.AUTO_MIN_SECONDS


def test_the_default_length_is_automatic():
    assert constants.DEFAULT_OPTIONS["max_seconds"] == 0.0


def test_a_short_ceiling_needs_the_floor_brought_down_with_it():
    """The crash this guards against, seen in the ComfyUI log on 2026-09-12.

    The protocol will not end a song before min_tokens, which is 200 frames or
    8 seconds. Any ceiling under that has to lower the floor too, or upstream
    validation kills the run with a sentence about tokens that means nothing to
    someone who moved a slider.
    """
    from yue2_comfy.vendor.yue2.protocol import Sampling

    cap = constants.seconds_to_tokens(5.0)
    with pytest.raises(ValueError):
        Sampling(max_tokens=cap)
    kept = Sampling(min_tokens=min(Sampling().min_tokens, cap), max_tokens=cap)
    assert kept.max_tokens == cap
