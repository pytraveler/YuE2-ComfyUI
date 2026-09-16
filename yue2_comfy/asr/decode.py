"""Greedy decoding with a fast step that is trusted only as far as it has been checked.

A step replayed from a CUDA graph (see ``network.GraphStep``) is several times
faster than the plain step inside ComfyUI, but a recording can come out wrong
without raising anything: seen once in a live ComfyUI, on a 190 s song. So the
plain step runs beside the replayed one on the first token and on every
``CHECK_EVERY``-th after it. A check that fails drops the tokens since the
last agreeing check, and decoding goes on with the plain step alone from
there, writing its own keys and values over whatever the replayed steps left,
so the answer is the plain step's answer. Left to this module, a check is the
two picks being equal; the network gives a ``check`` of its own that also
reads what the replayed steps wrote into the cache and forgives a pick that
differs by a near tie (see ``network.near_tie``), since between checks the
replayed picks are taken on trust anyway.

A model can also lose its place and say one thing over and over: on a song
that opens with vocalising, the answer went "Ooh, ooh, ooh" to the token limit,
and the words after it were never written. Such a loop begins where two picks
all but tie, since bfloat16 logits tie outright now and then, and which one
wins is a matter of kernels: the replayed and the plain step took different
sides on one song. So an answer that ends in the same few tokens said
``LOOP_REPEATS`` times is taken back to the narrowest choice near where the
repeating began, and the runner-up is taken there instead. Measured on 20
loops, three songs heard at several volumes, one or two such steps back gave
every word the same song gave without the loop, in about a second rather than
five; over 150 recordings nothing else was taken back, and an answer that never
repeats itself that often is decoded exactly as before.

No torch here: the steps are callables, so the bookkeeping is tested without a
card.
"""

from __future__ import annotations

CHECK_EVERY = 64
REPORT_EVERY = 32
LOOP_REPEATS = 20
"""A pattern said this many times in a row is a loop: the line ``prompt.collapse_repeats`` draws in the text.

The longest run measured that was not a loop was 17, a sung "ooh" before the words; taken back, such a run
loses a few of its "ooh"s and none of the words."""
LOOP_LONGEST = 8
"""The longest pattern, in tokens, looked for: a loop measured was two or three."""
LOOP_BREAKS = 8
"""How many times one answer is taken back before it is left to run to the limit."""


def looping(answer, repeats: int = LOOP_REPEATS, longest: int = LOOP_LONGEST):
    """``(size, count)`` when ``answer`` ends in a pattern of at most ``longest`` tokens said ``repeats`` times or more; else None."""
    length = len(answer)
    for size in range(1, longest + 1):
        if length < size * repeats:
            break
        pattern = answer[length - size:]
        count = 1
        while length - (count + 1) * size >= 0 and answer[length - (count + 1) * size:length - count * size] == pattern:
            count += 1
        if count >= repeats:
            return size, count
    return None


def way_out(answer, found, start: int, ranked, banned: dict) -> tuple:
    """``(index, token)``: where a loop is left and the token taken there instead.

    Every pick from a pattern's length before the repeating began to one past
    its first saying is weighed again with ``ranked``, which feeds a token at
    its position and returns the two best picks after it, ``[(token, score),
    (token, score)]``, leaving out the tokens ``banned`` at that index. The
    narrowest margin is where the model chose, the earliest on a tie; the pick
    it made there is banned and the other one taken.
    """
    size, count = found
    begins = len(answer) - size * count
    narrowest = None
    for index in range(max(0, begins - size), min(len(answer), begins + size + 1)):
        fed = answer[index - 1] if index else None
        picks = ranked(fed, start + index - 1, banned.get(index, ()))
        margin = picks[0][1] - picks[1][1]
        if narrowest is None or margin < narrowest[0]:
            narrowest = (margin, index, picks)
    _margin, index, picks = narrowest
    banned.setdefault(index, set()).add(answer[index])
    return index, (picks[1][0] if picks[0][0] == answer[index] else picks[0][0])


def greedy(first: int, start: int, limit: int, stops, plain, replayed=None,
           cancelled=None, progress=None, disagreed=None, check=None, ranked=None, rewind=None,
           looped=None) -> list:
    """The answer's tokens from ``first``, the prefill's pick, up to a stop token or ``limit`` of them.

    ``plain(token, position)`` and ``replayed(token, position)`` feed ``token``
    at ``position`` and return the next pick; ``replayed`` may be None.
    ``check(token, position, fast)`` runs the plain step there and says
    whether the replayed step, which picked ``fast``, is still to be trusted;
    without one, the two picks must be equal. ``disagreed()`` is called once,
    when a check fails. ``cancelled`` is asked and ``progress`` told the count
    every ``REPORT_EVERY`` tokens. The token that fills the answer is not fed:
    nothing after it would be kept.

    With ``ranked`` (see ``way_out``) an answer that starts looping is taken
    back; ``rewind(position)`` then tells the replayed step that the cache
    from ``position`` on is to be written again, and ``looped(index, size,
    count)`` is told where the answer was taken back to. Without it, a loop
    runs to ``limit`` as it always did.
    """
    if check is None:
        check = lambda token, position, fast: plain(token, position) == fast  # noqa: E731
    answer = []
    token, position = first, start
    trusted = (0, first, start)
    banned = {}
    breaks = 0
    while len(answer) < limit:
        answer.append(token)
        if token in stops or len(answer) == limit:
            break
        found = looping(answer) if ranked is not None and breaks < LOOP_BREAKS else None
        if found is not None:
            breaks += 1
            index, token = way_out(answer, found, start, ranked, banned)
            del answer[index:]
            for later in [later for later in banned if later > index]:
                del banned[later]
            position = start + index
            if trusted[0] >= index:
                trusted = (index, token, position)
            if replayed is not None and rewind is not None:
                rewind(position)
            if looped is not None:
                looped(index, *found)
            continue
        count = len(answer)
        if count % REPORT_EVERY == 0:
            if cancelled is not None and cancelled():
                raise InterruptedError("recognition cancelled")
            if progress is not None:
                progress(count)
        if replayed is None:
            token = plain(token, position)
        else:
            fast = replayed(token, position)
            if count == 1 or count % CHECK_EVERY == 0:
                if not check(token, position, fast):
                    replayed = None
                    if disagreed is not None:
                        disagreed()
                    kept, token, position = trusted
                    del answer[kept:]
                    for later in [later for later in banned if later >= kept]:
                        del banned[later]
                    continue
                trusted = (count, fast, position + 1)
            token = fast
        position += 1
    return answer
