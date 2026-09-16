"""The bar under a song node fills in step with the stages, and a share of it stays inside its share.

Before this was fixed, the stage table's percentages reached a bar whose total is one,
so the first percent past one filled it, and it stood full for the whole of
the singing. Seen live in ComfyUI's progress events: 0.75 while the score was
written, then 1.0 from the first step of the song to its end.
"""

from __future__ import annotations

import sys
import types

import pytest

from yue2_comfy import generate, progress


class Bar:
    """comfy.utils.ProgressBar as far as a node can see it: every absolute update, against its total."""

    made = []

    def __init__(self, total, node_id=None):
        self.total = total
        self.seen = []
        Bar.made.append(self)

    def update_absolute(self, value, total=None, preview=None):
        self.seen.append((value, total if total is not None else self.total))


@pytest.fixture
def bar(monkeypatch):
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = Bar
    comfy.utils = utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    Bar.made.clear()
    yield Bar


def run_through_every_stage(target):
    for stage in (generate.Stages.ABC, generate.Stages.SEMANTIC, generate.Stages.ACOUSTIC, generate.Stages.DECODE):
        for done in range(0, 11):
            generate._band(target, stage, done, 10)


def test_the_bar_fills_with_the_stages_rather_than_at_once(bar):
    node = progress.NodeProgress("7")
    run_through_every_stage(node)
    fills = [value / total for value, total in bar.made[0].seen]
    assert fills == sorted(fills)
    assert fills[-1] == pytest.approx(1.0)
    assert max(fills[:22]) <= generate.Stages.SEMANTIC[1] / 100.0 + 1e-9


def test_a_band_keeps_the_song_inside_its_share_and_the_voice_after_it(bar):
    node = progress.NodeProgress("7")
    run_through_every_stage(progress.Band(node, 0.0, 0.9))
    song = [value / total for value, total in bar.made[0].seen]
    assert song[-1] == pytest.approx(0.9)
    voice = progress.Band(node, 0.9, 1.0)
    voice.ratio(0.5, "Separating the voice")
    voice.ratio(1.0, "Separating the voice")
    fills = [value / total for value, total in bar.made[0].seen]
    assert fills[-2:] == [pytest.approx(0.95), pytest.approx(1.0)]
