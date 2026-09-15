"""Greedy decoding that checks the replayed step against the plain one, without torch or a card."""

from __future__ import annotations

import pytest

from yue2_comfy.asr import decode

START = 20
FIRST = 11


class Fake:
    """A model whose every pick depends on the whole cache, as attention does.

    Both steps write the fed token at its position. A replayed step broken
    from ``broken_from`` writes garbage there instead, so every later pick
    reads it; one spoiled from ``spoiled_from`` writes the token and soils the
    position beside it, which changes no pick but is what ``check`` finds,
    as the network's check reads the cache.
    """

    def __init__(self, broken_from=None, stop_at=None, spoiled_from=None):
        self.cache = {}
        self.soiled = set()
        self.broken_from = broken_from
        self.spoiled_from = spoiled_from
        self.stop_at = stop_at
        self.plain_positions = []
        self.told = 0

    def disagreed(self):
        self.told += 1

    def check(self, token, position, fast):
        pick = self.plain(token, position)
        return not self.soiled and pick == fast

    def _next(self, position):
        if self.stop_at is not None and position == self.stop_at:
            return 5
        total = sum(self.cache.get(p, 0) for p in range(position + 1))
        return (total * 31 + position) % 997 + 10

    def plain(self, token, position):
        self.plain_positions.append(position)
        self.cache[position] = token
        self.soiled.discard(position)
        return self._next(position)

    def replayed(self, token, position):
        broken = self.broken_from is not None and position >= self.broken_from
        self.cache[position] = -1 if broken else token
        if self.spoiled_from is not None and position >= self.spoiled_from:
            self.soiled.add(position)
        return self._next(position)


def clean(limit, stop_at=None):
    fake = Fake(stop_at=stop_at)
    return decode.greedy(FIRST, START, limit, {5}, fake.plain)


def test_plain_decoding_runs_to_the_limit():
    answer = clean(300)
    assert len(answer) == 300
    assert answer[0] == FIRST


def test_a_sound_replay_gives_the_plain_answer_and_is_checked_sparingly():
    fake = Fake()
    answer = decode.greedy(FIRST, START, 300, {5}, fake.plain, fake.replayed)
    assert answer == clean(300)
    assert fake.plain_positions == [START, START + 63, START + 127, START + 191, START + 255]


@pytest.mark.parametrize("broken_from", [START, START + 1, START + 63, START + 64, START + 100, START + 250])
def test_a_broken_replay_is_dropped_and_its_tokens_decoded_again(broken_from):
    fake = Fake(broken_from=broken_from)
    told = []
    answer = decode.greedy(FIRST, START, 300, {5}, fake.plain, fake.replayed, disagreed=lambda: told.append(1))
    assert answer == clean(300)
    assert told == [1]


def test_plain_decoding_does_not_feed_the_token_that_fills_the_answer():
    fake = Fake()
    decode.greedy(FIRST, START, 3, {5}, fake.plain)
    assert fake.plain_positions == [START, START + 1]


def test_a_stop_token_ends_the_answer_and_is_kept():
    fake = Fake(stop_at=START + 40)
    answer = decode.greedy(FIRST, START, 300, {5}, fake.plain, fake.replayed)
    assert answer == clean(300, stop_at=START + 40)
    assert answer[-1] == 5
    assert len(answer) == 42


def test_the_first_token_can_already_be_a_stop():
    assert decode.greedy(5, START, 300, {5}, Fake().plain) == [5]


def test_progress_is_told_and_cancellation_asked_every_32_tokens():
    told = []
    fake = Fake()
    decode.greedy(FIRST, START, 100, {5}, fake.plain, fake.replayed, progress=told.append)
    assert told == [32, 64, 96]
    with pytest.raises(InterruptedError):
        decode.greedy(FIRST, START, 100, {5}, Fake().plain, cancelled=lambda: True)


@pytest.mark.parametrize("spoiled_from", [START + 1, START + 63, START + 64, START + 200])
def test_a_replay_that_spoils_the_cache_without_changing_a_pick_is_caught_by_the_plain_step(spoiled_from):
    """The network's check reads the cache, so what the replays wrote is checked, not only what they picked."""
    fake = Fake(spoiled_from=spoiled_from)
    answer = decode.greedy(FIRST, START, 300, {5}, fake.plain, fake.replayed, disagreed=fake.disagreed, check=fake.check)
    assert answer == clean(300)
    assert fake.told == 1


def test_a_check_of_its_own_may_let_a_differing_pick_stand():
    fake = Fake(broken_from=START + 100)
    told = []
    answer = decode.greedy(FIRST, START, 300, {5}, fake.plain, fake.replayed, disagreed=lambda: told.append(1),
                           check=lambda token, position, fast: fake.plain(token, position) is not None)
    assert told == [] and len(answer) == 300 and answer != clean(300)
