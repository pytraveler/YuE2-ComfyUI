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

No torch here: the steps are callables, so the bookkeeping is tested without a
card.
"""

from __future__ import annotations

CHECK_EVERY = 64
REPORT_EVERY = 32


def greedy(first: int, start: int, limit: int, stops, plain, replayed=None,
           cancelled=None, progress=None, disagreed=None, check=None) -> list:
    """The answer's tokens from ``first``, the prefill's pick, up to a stop token or ``limit`` of them.

    ``plain(token, position)`` and ``replayed(token, position)`` feed ``token``
    at ``position`` and return the next pick; ``replayed`` may be None.
    ``check(token, position, fast)`` runs the plain step there and says
    whether the replayed step, which picked ``fast``, is still to be trusted;
    without one, the two picks must be equal. ``disagreed()`` is called once,
    when a check fails. ``cancelled`` is asked and ``progress`` told the count
    every ``REPORT_EVERY`` tokens. The token that fills the answer is not fed:
    nothing after it would be kept.
    """
    if check is None:
        check = lambda token, position, fast: plain(token, position) == fast  # noqa: E731
    answer = []
    token, position = first, start
    trusted = (0, first, start)
    while len(answer) < limit:
        answer.append(token)
        if token in stops or len(answer) == limit:
            break
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
                    continue
                trusted = (count, fast, position + 1)
            token = fast
        position += 1
    return answer
