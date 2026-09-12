"""On-node progress built on the stock ComfyUI progress channels.

PromptServer.send_progress_text writes the caption under a running node and
comfy.utils.ProgressBar fills the bar beside it. Both are addressed by node id,
so no custom frontend extension is needed.

Ported from the MiniMax-H3 Prompt Rewriter pack.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

TEXT_MIN_INTERVAL = 0.25
NOTICES_EVENT = "yue2_comfy.notices"


def refuse(node_id, message: str):
    """Say it on screen, then raise it.

    A node that stops has to say why twice over. The exception is the truthful
    half -- it stops the run and lands in the console with a traceback -- but a
    traceback is not where anyone looks first, and ComfyUI's error panel is easy
    to close without reading. The toast is the same sentence, before the
    exception goes up.
    """
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


class NodeProgress:
    """The caption and the fill of the bar under one executing node.

    ComfyUI draws the bar itself for as long as a node is running and takes the
    fill width from the percentage the backend reports. Reporting nothing does
    not remove the bar, it leaves an empty trough, which reads as a stall. So
    the number goes to the bar and the words go to the caption, and neither
    repeats the other.
    """

    def __init__(self, node_id, total: float = 1.0):
        self.node_id = str(node_id) if node_id is not None else None
        self.total = max(float(total), 1.0)
        self._server = None
        self._bar = None
        self._last_text = ""
        self._last_text_at = 0.0
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
        if text is not None:
            self.text(text)

    def ratio(self, fraction: float, text=None) -> None:
        self.update(self.total * max(0.0, min(1.0, fraction)), text)

    def text(self, message: str, force: bool = False) -> None:
        if self._server is None or self.node_id is None:
            return
        now = time.monotonic()
        if not force and message == self._last_text:
            return
        if not force and now - self._last_text_at < TEXT_MIN_INTERVAL:
            return
        self._last_text = message
        self._last_text_at = now
        try:
            self._server.send_progress_text(message, self.node_id)
        except Exception:
            log.debug("[yue2_comfy.NodeProgress.text] send failed", exc_info=True)

    def finish(self, message=None) -> None:
        if message is not None:
            self.text(message, force=True)


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
