"""On-node progress built on the stock ComfyUI progress channels.

PromptServer.send_progress_text writes the caption under a running node and
comfy.utils.ProgressBar fills the bar beside it. Both are addressed by node id,
so no custom frontend extension is needed.

Ported from the MiniMax-H3 Prompt Rewriter pack.
"""

from __future__ import annotations

import logging
import os
import sys
import time

log = logging.getLogger(__name__)

TEXT_MIN_INTERVAL = 0.25
NOTICES_EVENT = "yue2_comfy.notices"

CONSOLE_SWITCH = "YUE2_CONSOLE_PROGRESS"
"""Set it to 0, no, off or false and nothing is drawn on the console.

Meant for a server whose console is really a log file, where a line redrawn
several times a second is thousands of lines nobody reads.
"""

CONSOLE_INTERVAL = 0.1
CONSOLE_WIDTH = 22
CONSOLE_FORMAT = "{desc} |{bar}| {percentage:3.0f}% [{elapsed}]"
"""No ETA on purpose.

The stages do not take the shares of the bar they are given -- the table in
generate.Stages says as much itself -- so a remaining time worked out from the
percentage would be a number the pack cannot stand behind. Elapsed is measured.
A download is the one stage that knows its own ETA, and TransferReporter writes
that into the caption, where it is true.
"""


def interrupted() -> bool:
    """ComfyUI's cancel flag, read without clearing it.

    processing_interrupted is the non-consuming reader.
    throw_exception_if_processing_interrupted is the consuming one, and calling
    that from inside a generation loop would clear the flag on the first stage
    that noticed, leaving the later stages to run on.
    """
    try:
        import comfy.model_management as mm

        return bool(mm.processing_interrupted())
    except Exception:
        return False


def translate_interrupt() -> None:
    """Hand a cancelled run back to ComfyUI as its own interrupt.

    Upstream raises InterruptedError. ComfyUI wants InterruptProcessingException,
    and throw_exception_if_processing_interrupted is the only function that
    clears the flag on the way, so it is called rather than constructing the
    exception directly. If someone else already consumed the flag it returns
    quietly, and the exception is raised by hand.

    Note that InterruptProcessingException derives from BaseException, not
    Exception, which is why nothing around the generation is wrapped in a bare
    'except Exception'.
    """
    ConsoleBar.shut()
    try:
        import comfy.model_management as mm
    except Exception:
        return
    mm.throw_exception_if_processing_interrupted()
    raise mm.InterruptProcessingException()


def refuse(node_id, message: str):
    """Say it on screen, then raise it.

    A node that stops has to say why twice over. The exception is the truthful
    half -- it stops the run and lands in the console with a traceback -- but a
    traceback is not where anyone looks first, and ComfyUI's error panel is easy
    to close without reading. The toast is the same sentence, before the
    exception goes up.
    """
    ConsoleBar.shut()
    announce(node_id, [("warn", message)], kind="refusal")
    raise ValueError(message)


def announce(node_id, findings, kind: str = "notice") -> None:
    """Hand findings to the frontend, which shows them as a toast.

    Fire and forget: a frontend that is not listening loses nothing but the
    toast. 'findings' is a list of (level, message) pairs.
    """
    if node_id is None or not findings:
        return
    issues = []
    for entry in findings:
        level, message = entry
        issues.append({"level": str(level), "message": str(message)})
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(
            NOTICES_EVENT, {"node": str(node_id), "kind": kind, "issues": issues})
    except Exception:
        log.debug("[yue2_comfy.progress] could not announce the findings", exc_info=True)


def format_size(num_bytes: float) -> str:
    if num_bytes >= 1024 ** 3:
        return "{:.2f} GB".format(num_bytes / 1024 ** 3)
    if num_bytes >= 1024 ** 2:
        return "{:.1f} MB".format(num_bytes / 1024 ** 2)
    if num_bytes >= 1024:
        return "{:.1f} KB".format(num_bytes / 1024)
    return "{} B".format(int(num_bytes))


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return "{}:{:02d}:{:02d}".format(hours, minutes, secs)
    return "{}:{:02d}".format(minutes, secs)


def console_wanted() -> bool:
    """Whether a node draws its progress on the console as well as on itself."""
    return str(os.environ.get(CONSOLE_SWITCH, "1")).strip().lower() not in (
        "0", "no", "off", "false")


class PlainBar:
    """The console line for an install whose Python has no tqdm.

    Only the calls ConsoleBar makes are here, under tqdm's own names, and the
    line drawn is the one CONSOLE_FORMAT asks tqdm for. ComfyUI's requirements
    list tqdm, so this is the spare wheel rather than the road.
    """

    def __init__(self, total, desc, stream):
        self.total = max(float(total), 1.0)
        self.desc = desc
        self.stream = stream
        self.n = 0.0
        self.started = time.monotonic()
        self.drawn = 0
        self.at = 0.0

    def set_description_str(self, desc=None, refresh=True) -> None:
        self.desc = self.desc if desc is None else desc
        if refresh:
            self.refresh()

    def update(self, step=1.0) -> None:
        self.n = min(max(self.n + float(step), 0.0), self.total)
        if time.monotonic() - self.at >= CONSOLE_INTERVAL:
            self.refresh()

    def refresh(self) -> None:
        self.at = time.monotonic()
        share = self.n / self.total
        filled = int(round(CONSOLE_WIDTH * share))
        line = "{} |{}| {:3.0f}% [{}]".format(
            self.desc, "#" * filled + "-" * (CONSOLE_WIDTH - filled), share * 100.0,
            format_duration(time.monotonic() - self.started))
        self._put("\r" + line + " " * max(0, self.drawn - len(line)))
        self.drawn = len(line)

    def close(self) -> None:
        self.refresh()
        self._put("\n")

    def _put(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except Exception:
            log.debug("[yue2_comfy.PlainBar] the console would not take the line", exc_info=True)


class ConsoleBar:
    """The node's own fraction and caption, drawn again on the console.

    ComfyUI draws a node's bar in the browser. Someone who started the server
    from a terminal and watches it work there sees nothing at all for the
    minutes a song takes, and a song is the node where that matters most.

    One line is open at a time, the way ComfyUI runs one node at a time: a bar
    closes the one before it, so a node that raised instead of finishing leaves
    a closed line behind rather than a half-drawn one for the next node to
    write over.
    """

    open_bar = None

    def __init__(self, title: str):
        self.title = title or "YuE2"
        self.caption_text = ""
        self.bar = None
        self.made = False
        ConsoleBar.shut()
        ConsoleBar.open_bar = self

    @classmethod
    def shut(cls) -> None:
        """Close whatever line is open, leaving its number where it stood."""
        if cls.open_bar is not None:
            cls.open_bar.close()

    def _line(self):
        """The tqdm-shaped object, made when there is first something to say on it."""
        if self.made:
            return self.bar
        self.made = True
        stream = getattr(sys, "stderr", None)
        if stream is None:
            return None
        try:
            from tqdm import tqdm

            self.bar = tqdm(total=100, desc=self.title, bar_format=CONSOLE_FORMAT,
                            leave=True, file=stream, dynamic_ncols=True)
        except Exception:
            log.debug("[yue2_comfy.ConsoleBar] no tqdm here, drawing the line by hand",
                      exc_info=True)
            self.bar = PlainBar(100, self.title, stream)
        return self.bar

    def at(self, fraction: float) -> None:
        bar = self._line()
        if bar is None:
            return
        try:
            bar.n = max(0.0, min(1.0, float(fraction))) * 100.0
            bar.update(0)
        except Exception:
            log.debug("[yue2_comfy.ConsoleBar.at] the line could not be drawn", exc_info=True)

    def caption(self, message: str) -> None:
        bar = self._line()
        if bar is None:
            return
        said = " | ".join(piece.strip() for piece in str(message).splitlines() if piece.strip())
        text = self.title + ": " + said if said else self.title
        if text == self.caption_text:
            return
        self.caption_text = text
        try:
            bar.set_description_str(text)
        except Exception:
            log.debug("[yue2_comfy.ConsoleBar.caption] the line could not be drawn", exc_info=True)

    def close(self, done: bool = False) -> None:
        """Finish the line. ``done`` fills it, for a node that really did finish."""
        bar, self.bar = self.bar, None
        self.made = True
        if ConsoleBar.open_bar is self:
            ConsoleBar.open_bar = None
        if bar is None:
            return
        try:
            if done:
                bar.n = 100.0
            bar.close()
        except Exception:
            log.debug("[yue2_comfy.ConsoleBar.close] the line would not close", exc_info=True)


class NodeProgress:
    """The caption and the fill of the bar under one executing node.

    ComfyUI draws the bar itself for as long as a node is running and takes the
    fill width from the percentage the backend reports. Reporting nothing does
    not remove the bar, it leaves an empty trough, which reads as a stall. So
    the number goes to the bar and the words go to the caption, and neither
    repeats the other.
    """

    def __init__(self, node_id, total: float = 1.0, title: str = ""):
        self.node_id = str(node_id) if node_id is not None else None
        self.total = max(float(total), 1.0)
        self._server = None
        self._bar = None
        self._last_text = ""
        self._last_text_at = 0.0
        self._console = ConsoleBar(title) if console_wanted() else None
        if self.node_id is None:
            return
        try:
            from server import PromptServer

            self._server = PromptServer.instance
        except Exception:
            log.debug("[yue2_comfy.NodeProgress] prompt server unavailable", exc_info=True)

    def _ensure_bar(self):
        if self._bar is not None or self.node_id is None:
            return self._bar
        try:
            from comfy.utils import ProgressBar

            self._bar = ProgressBar(self.total, node_id=self.node_id)
        except Exception:
            log.debug("[yue2_comfy.NodeProgress] progress bar unavailable", exc_info=True)
        return self._bar

    def set_total(self, total: float) -> None:
        self.total = max(float(total), 1.0)
        bar = self._ensure_bar()
        if bar is not None:
            bar.total = self.total

    def update(self, value: float, text=None) -> None:
        bar = self._ensure_bar()
        if bar is not None:
            try:
                bar.update_absolute(max(0.0, min(float(value), self.total)), self.total)
            except Exception:
                log.debug("[yue2_comfy.NodeProgress.update] bar update failed", exc_info=True)
        if self._console is not None:
            self._console.at(float(value) / self.total)
        if text is not None:
            self.text(text)

    def ratio(self, fraction: float, text=None) -> None:
        self.update(self.total * max(0.0, min(1.0, fraction)), text)

    def text(self, message: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and message == self._last_text:
            return
        if not force and now - self._last_text_at < TEXT_MIN_INTERVAL:
            return
        self._last_text = message
        self._last_text_at = now
        if self._console is not None:
            self._console.caption(message)
        if self._server is None or self.node_id is None:
            return
        try:
            self._server.send_progress_text(message, self.node_id)
        except Exception:
            log.debug("[yue2_comfy.NodeProgress.text] send failed", exc_info=True)

    def finish(self, message=None) -> None:
        """The node is done: the last word in the caption, and the console line closed.

        Only a node calls this. A stage that is over inside a node says so with
        ``text(..., force=True)`` instead, because closing the console line there
        would leave the rest of the node with nowhere to draw.
        """
        if message is not None:
            self.text(message, force=True)
        if self._console is not None:
            self._console.close(done=True)


class Band:
    """A share of the node's bar: a fraction reported here fills only ``low`` to ``high`` of it.

    A download or the language model reports counts against a total of its
    own; those are scaled into the share too, so the bar never runs ahead of
    the stage it is in.
    """

    def __init__(self, progress, low: float, high: float):
        self._progress = progress
        self._low = low
        self._high = high
        self._total = 1.0

    def ratio(self, fraction: float, text=None) -> None:
        self._progress.ratio(self._low + (self._high - self._low) * max(0.0, min(1.0, fraction)), text)

    def set_total(self, total: float) -> None:
        self._total = max(float(total), 1.0)

    def update(self, value: float, text=None) -> None:
        self.ratio(float(value) / self._total, text)

    def finish(self, message=None) -> None:
        """A share of the bar cannot end the node: it says its piece, the line stays open."""
        if message is not None:
            self._progress.text(message, force=True)

    def __getattr__(self, name):
        return getattr(self._progress, name)


class TransferReporter:
    """Turns byte counts into a human caption.

    Matches the downloader's on_progress(transferred, current_name) callback.
    """

    def __init__(self, progress: NodeProgress, total_bytes: int, title: str):
        self.progress = progress
        self.total_bytes = max(int(total_bytes), 1)
        self.title = title
        self.started_at = time.monotonic()
        self.baseline = None
        self.progress.set_total(self.total_bytes)

    def set_total(self, total_bytes: int) -> None:
        self.total_bytes = max(int(total_bytes), 1)
        self.progress.set_total(self.total_bytes)

    def __call__(self, transferred: int, current_name: str) -> None:
        if self.baseline is None:
            self.baseline = transferred
            self.started_at = time.monotonic()
        elapsed = max(time.monotonic() - self.started_at, 1e-6)
        speed = (transferred - self.baseline) / elapsed
        remaining = (self.total_bytes - transferred) / speed if speed > 0 else float("inf")
        caption = "{}\n{}\n{} / {} | {}/s | ETA {}".format(
            self.title, current_name,
            format_size(transferred), format_size(self.total_bytes),
            format_size(speed), format_duration(remaining))
        self.progress.update(transferred, caption)
