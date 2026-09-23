"""The beat of a stretch of sound, read off its onsets, and the lengths of a new bar that put it back across a seam."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yue2_comfy.inpaint import beat  # noqa: E402

RATE = 48000
PULSE = 0.25
SEAM = 10.0
LENGTH = 17.0


def clicks(moves):
    """A click every ``PULSE`` seconds, each moved by ``moves(second)``: a drummer, and nothing else."""
    generator = torch.Generator().manual_seed(3)
    sound = torch.zeros(int(LENGTH * RATE))
    decay = torch.exp(-torch.arange(400, dtype=torch.float32) / 60.0)
    second = 0.1
    while second < LENGTH - 0.1:
        at = int((second + moves(second)) * RATE)
        burst = torch.randn(400, generator=generator) * decay
        sound[at:at + 400] += burst[:max(0, sound.shape[0] - at)]
        second += PULSE
    return sound[None].repeat(2, 1)


def test_a_beat_that_jumps_at_a_seam_is_read_to_within_a_few_milliseconds():
    sound = clicks(lambda second: 0.06 if second >= SEAM else 0.0)
    moved, wander = beat.read(sound, RATE, SEAM, PULSE)
    assert moved == pytest.approx(0.06, abs=0.005)
    assert wander < 0.01
    assert beat.jump(sound, RATE, SEAM, PULSE) == pytest.approx(0.06, abs=0.005)


def test_a_beat_that_runs_on_reads_no_jump():
    moved, wander = beat.read(clicks(lambda second: 0.0), RATE, SEAM, PULSE)
    assert abs(moved) < 0.005 and wander < 0.01


def test_a_jump_is_read_to_within_an_eighth():
    """A pulse late by 0.3 of itself reads as early by 0.7 of it: the comb cannot tell eighths apart."""
    moved, _wander = beat.read(clicks(lambda second: 0.7 * PULSE if second >= SEAM else 0.0),
                               RATE, SEAM, PULSE)
    assert moved == pytest.approx(-0.3 * PULSE, abs=0.005)


def test_a_side_that_does_not_keep_time_is_not_trusted():
    """The half of the stretch before the seam nearer to it plays a tenth of a second late."""
    sound = clicks(lambda second: 0.1 if SEAM - 3.0 <= second < SEAM else 0.0)
    _moved, wander = beat.read(sound, RATE, SEAM, PULSE)
    assert wander > beat.WANDER
    assert beat.jump(sound, RATE, SEAM, PULSE) is None


def test_a_seam_too_near_either_end_of_the_sound_is_not_read():
    sound = clicks(lambda second: 0.0)
    assert beat.read(sound, RATE, 3.0, PULSE) == (0.0, None)
    assert beat.read(sound, RATE, LENGTH - 3.0, PULSE) == (0.0, None)
    assert beat.jump(sound, RATE, 3.0, PULSE) is None


def test_the_lengths_open_to_the_join_put_the_beat_back_on_an_eighth():
    """The new frames go on from the beat before the seam: each frame more moves what follows by one."""
    found = beat.counts_on_beat(30, 20, 40, 0.057, PULSE, 0.04)
    assert found == [22, 29, 35]
    for count in found:
        left = 0.057 + (count - 30) * 0.04
        assert abs(left - PULSE * round(left / PULSE)) <= 0.02
    assert beat.counts_on_beat(30, 30, 34, 0.057, PULSE, 0.04) == []
