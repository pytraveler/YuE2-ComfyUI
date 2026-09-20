"""The line a node draws on the console, for whoever watches the server rather than the browser.

ComfyUI draws a node's bar in the browser and nowhere else. A song is minutes
long, and a terminal that says nothing for those minutes reads as a hang, which
is what this line is for.

The rules checked here are the ones that bite. A line has to close, or ComfyUI's
own next message lands in the middle of it. One node's line must not swallow the
next node's, including after a node raised instead of finishing. A caption of
several lines has to become one, because the console has no second line to put
the rest on. And an install that asked for silence has to get it.

The text is read back through pytest's own capture rather than a stream of our
own: pytest puts its objects on sys.stderr again when the test body starts, so a
buffer installed by a fixture is not the one the bar ends up writing to.
"""

from __future__ import annotations

import sys
import types

import pytest

from yue2_comfy import progress


class Tqdm:
    """tqdm as far as the console bar uses it."""

    made = []

    def __init__(self, total=100, desc="", bar_format="", leave=True, file=None,
                 dynamic_ncols=True):
        self.total = total
        self.desc = desc
        self.n = 0.0
        self.closed = False
        self.captions = []
        Tqdm.made.append(self)

    def update(self, step=1.0):
        self.n += step

    def set_description_str(self, desc=None, refresh=True):
        self.desc = desc
        self.captions.append(desc)

    def refresh(self):
        pass

    def close(self):
        self.closed = True


def drawn(text) -> list:
    """Every state the line was redrawn in, newest last."""
    return [piece for piece in text.replace("\n", "\r").split("\r") if piece.strip()]


@pytest.fixture
def console(monkeypatch, capsys):
    """A console without tqdm, so PlainBar draws and the text is ours to read."""
    monkeypatch.setitem(sys.modules, "tqdm", None)
    monkeypatch.delenv(progress.CONSOLE_SWITCH, raising=False)
    progress.ConsoleBar.open_bar = None
    yield capsys
    progress.ConsoleBar.open_bar = None


@pytest.fixture
def tqdm_console(monkeypatch):
    """A console with tqdm, recording what the bar asks of it."""
    module = types.ModuleType("tqdm")
    module.tqdm = Tqdm
    monkeypatch.setitem(sys.modules, "tqdm", module)
    monkeypatch.delenv(progress.CONSOLE_SWITCH, raising=False)
    progress.ConsoleBar.open_bar = None
    Tqdm.made.clear()
    yield Tqdm
    progress.ConsoleBar.open_bar = None


def test_the_fraction_and_the_caption_both_reach_the_console(console):
    node = progress.NodeProgress("7", title="YuE2 Generate Song")
    node.update(0.42, "Composing")
    node.finish("42 seconds of audio")
    written = console.readouterr().err
    shown = drawn(written)
    assert shown, "the line was never drawn"
    assert any("YuE2 Generate Song: Composing" in piece and "42%" in piece for piece in shown)
    assert "YuE2 Generate Song: 42 seconds of audio" in shown[-1]
    assert "100%" in shown[-1]
    assert written.endswith("\n")


def test_a_caption_of_several_lines_becomes_one(console):
    node = progress.NodeProgress("7", title="YuE2 Transcribe")
    node.update(0.1, "Downloading SheetSage2\nsheetsage.safetensors\n12.0 MB / 40.0 MB")
    written = console.readouterr().err
    assert written.count("\n") == 0
    assert "Downloading SheetSage2 | sheetsage.safetensors | 12.0 MB / 40.0 MB" in drawn(written)[-1]


def test_a_node_that_says_nothing_draws_nothing(console):
    progress.NodeProgress("7", title="YuE2 Options")
    assert console.readouterr().err == ""


def test_the_next_node_closes_the_line_the_last_one_left_open(console):
    first = progress.NodeProgress("7", title="YuE2 Plan")
    first.update(0.3, "Writing the score")
    progress.NodeProgress("8", title="YuE2 Render Plan").update(0.1, "Composing")
    written = console.readouterr().err
    assert written.count("\n") == 1, "the first line was not closed before the second began"
    assert written.index("\n") < written.index("YuE2 Render Plan")


def test_a_cancelled_node_leaves_its_line_where_it_stood(console, monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy", None)
    node = progress.NodeProgress("7", title="YuE2 Generate Song")
    node.update(0.63, "Composing")
    progress.translate_interrupt()
    written = console.readouterr().err
    assert "63%" in drawn(written)[-1]
    assert "100%" not in written
    assert written.endswith("\n")


def test_a_refusal_closes_the_line(console):
    node = progress.NodeProgress("7", title="YuE2 Generate Song")
    node.update(0.2, "Writing the score")
    with pytest.raises(ValueError):
        progress.refuse("7", "Write a line saying what the song is about.")
    assert console.readouterr().err.endswith("\n")


def test_a_console_told_to_stay_quiet_stays_quiet(console, monkeypatch):
    monkeypatch.setenv(progress.CONSOLE_SWITCH, "0")
    node = progress.NodeProgress("7", title="YuE2 Generate Song")
    node.update(0.5, "Composing")
    node.finish("42 seconds of audio")
    assert console.readouterr().err == ""


def test_tqdm_draws_the_line_where_tqdm_is_installed(tqdm_console):
    node = progress.NodeProgress("7", title="YuE2 Generate Song")
    node.update(0.5, "Composing")
    node.finish("42 seconds of audio")
    assert len(tqdm_console.made) == 1
    bar = tqdm_console.made[0]
    assert bar.closed
    assert bar.n == 100.0
    assert bar.captions[-1] == "YuE2 Generate Song: 42 seconds of audio"


def test_a_stage_that_ends_inside_a_node_does_not_close_the_line(tqdm_console):
    node = progress.NodeProgress("7", title="YuE2 Write Song")
    node.update(0.4, "Loading the model")
    band = progress.Band(node, 0.4, 0.9)
    band.finish("Written - 420 chars")
    assert not tqdm_console.made[0].closed
    node.finish("12 sung lines")
    assert tqdm_console.made[0].closed
