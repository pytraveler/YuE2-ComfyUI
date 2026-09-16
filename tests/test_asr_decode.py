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


class Looper:
    """A model that ties after its first token: the lower pick repeats two tokens forever, the other sings on.

    Scores follow the tokens the cache holds, so a step taken back and fed
    again answers as a fresh one would. ``ranked`` is the network's: the two
    best picks after a fed token, less the banned ones; None stands for the
    prefill's pick.
    """

    WORDS = [40, 41, 42, 43, 44, 45]

    def __init__(self, loop_everywhere=False, honest=0, words=len(WORDS)):
        self.words = list(range(40, 40 + words))
        self.cache = {}
        self.loop_everywhere = loop_everywhere
        self.honest = honest
        self.plain_positions = []
        self.rewound = []

    def _scores(self, position):
        told = [self.cache[p] for p in range(START, position + 1)]
        last = told[-1]
        if last == FIRST:
            return {30: 1.0, 40: 1.0} if self.honest == 0 else {20: 3.0, 40: 1.0}
        if last == 20:
            return {21: 5.0, 40: 1.0}
        if last == 21:
            sung = sum(1 for token in told if token in (20, 21))
            return {20: 5.0, 40: 1.0} if sung < 2 * self.honest else {40: 5.0, 20: 1.0}
        if last in (30, 31):
            return {31 if last == 30 else 30: 5.0, 5: 0.5}
        if self.loop_everywhere and last >= 40:
            return {30: 1.0, last + 1: 1.0}
        if last in self.words[:-1]:
            return {last + 1: 5.0, 30: 0.5}
        return {5: 5.0, 30: 0.5}

    def _best(self, scores, banned=()):
        kept = sorted(((score, -token) for token, score in scores.items() if token not in banned), reverse=True)
        return [(-token, score) for score, token in kept]

    def plain(self, token, position):
        self.plain_positions.append(position)
        self.cache[position] = token
        return self._best(self._scores(position))[0][0]

    def replayed(self, token, position):
        self.cache[position] = token
        return self._best(self._scores(position))[0][0]

    def ranked(self, token, position, banned):
        if token is None:
            picks = [(FIRST, 9.0), (7, 1.0)]
            return [pick for pick in picks if pick[0] not in banned] + [(8, 0.0)]
        self.cache[position] = token
        picks = self._best(self._scores(position), banned)
        return (picks + [(9, float("-inf"))] * 2)[:2]


def test_a_loop_is_seen_only_at_its_twentieth_saying():
    assert decode.looping([1] + [7, 8] * 19) is None
    assert decode.looping([1] + [7, 8] * 20) == (2, 20)
    assert decode.looping([7] * 25) == (1, 25)
    assert decode.looping([7, 8] * 30 + [9]) is None
    assert decode.looping(list(range(9)) * 20) is None
    assert decode.looping(list(range(8)) * 20) == (8, 20)


def test_without_ranked_a_loop_runs_to_the_limit_as_before():
    answer = decode.greedy(FIRST, START, 300, {5}, Looper().plain)
    assert len(answer) == 300
    assert answer[:5] == [FIRST, 30, 31, 30, 31]


@pytest.mark.parametrize("replay", [False, True])
def test_a_loop_is_taken_back_to_the_tie_it_began_at(replay):
    """The answer the other side of the tie would have given, whichever step decodes it."""
    fake = Looper()
    told = []
    answer = decode.greedy(FIRST, START, 300, {5}, fake.plain, fake.replayed if replay else None,
                           ranked=fake.ranked, rewind=fake.rewound.append,
                           looped=lambda *where: told.append(where))
    assert answer == [FIRST] + Looper.WORDS + [5]
    assert told == [(1, 2, 20)]
    assert fake.rewound == ([START + 1] if replay else [])


def test_an_honest_run_below_the_line_is_left_as_it_was_sung():
    fake = Looper(honest=19)
    answer = decode.greedy(FIRST, START, 300, {5}, fake.plain, ranked=fake.ranked,
                           looped=lambda *where: pytest.fail("an honest run was taken back"))
    assert answer == [FIRST] + [20, 21] * 19 + Looper.WORDS + [5]


def test_a_model_that_loops_whatever_it_picks_is_left_to_the_limit_after_a_few_tries():
    fake = Looper(loop_everywhere=True)
    told = []
    answer = decode.greedy(FIRST, START, 400, {5}, fake.plain, ranked=fake.ranked,
                           looped=lambda *where: told.append(where))
    assert len(told) == decode.LOOP_BREAKS
    assert len(answer) == 400


def test_a_replay_that_fails_its_check_after_a_loop_was_left_does_not_go_back_into_it():
    """The first check passed on the very pick the loop was later left at; a failed check resumes from the other pick."""
    fake = Looper(words=100)
    checks = []
    told = []

    def check(token, position, fast):
        checks.append(position)
        return len(checks) == 1

    answer = decode.greedy(FIRST, START, 300, {5}, fake.plain, fake.replayed, check=check, ranked=fake.ranked,
                           rewind=fake.rewound.append, looped=lambda *where: told.append(where))
    assert answer == [FIRST] + fake.words + [5]
    assert len(checks) == 2
    assert told == [(1, 2, 20)]
