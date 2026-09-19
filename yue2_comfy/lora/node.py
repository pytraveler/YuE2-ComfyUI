"""The 'YuE2 LoRA' node: adapters as rows, handed to the nodes that sing.

One row is one file with two strengths, one for each half of YuE2. The rows are
kept as JSON in one text widget, which is what the workflow file saves, and the
browser draws them as rows over it -- ``web/js/yue2_lora.js``. A widget of a
fixed kind in a fixed place is the one arrangement ComfyUI's positional
``widgets_values`` cannot shift.

That widget has no tooltip. With Vue nodes on, a widget's tooltip opens over
the whole widget after a moment's hover, and here the widget is the rows: a
paragraph covered the list of files whenever the pointer rested on the node.
What the tooltip said lives where it is read instead -- AR and NAR, with the
names ComfyUI's LoraLoader gives them, on the column heads and the numbers.

What the node hands on is plain data -- a name, the file, its SHA-256 and the
strengths -- because it travels inside the settings: into a plan that ComfyUI
caches, and into the song memory, which writes it down so that an edit of the
song later folds the same adapters.

Nothing here imports torch.
"""

from __future__ import annotations

import json
import logging

from ..constants import CATEGORY, LORA_TYPE
from ..progress import refuse
from . import catalogue

log = logging.getLogger(__name__)

STRENGTH_LIMIT = 10.0
"""How far a strength goes either way. Concept sliders are made to be driven
past one and below zero; nothing published asks for more than a few."""

CHAIN_TOOLTIP = "More adapters from another 'YuE2 LoRA' node, sung together with this node's own."


def rows(text) -> list:
    """The rows a widget holds, each ``{"name", "ar", "nar", "on"}``, or ValueError saying what is wrong."""
    try:
        parsed = json.loads(text or "[]")
    except ValueError as error:
        raise ValueError("The rows of this node are not JSON: {}".format(error)) from error
    if not isinstance(parsed, list):
        raise ValueError("The rows of this node should be a list, not {}.".format(type(parsed).__name__))
    found = []
    for index, item in enumerate(parsed, 1):
        if not isinstance(item, dict):
            raise ValueError("Row {} is not a row: {!r}".format(index, item))
        name = str(item.get("name") or "").strip()
        strengths = []
        for key in ("ar", "nar"):
            value = item.get(key, 1.0)
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError("Row {} has {} = {!r}, which is not a number.".format(index, key, value)) from None
            if number != number or abs(number) > STRENGTH_LIMIT:
                raise ValueError("Row {} has {} = {}; a strength goes from -{} to {}.".format(
                    index, key, value, STRENGTH_LIMIT, STRENGTH_LIMIT))
            strengths.append(number)
        found.append({"name": name, "ar": strengths[0], "nar": strengths[1],
                      "on": item.get("on", True) is not False})
    return found


def picked(text) -> list:
    """The rows as the settings carry them, with each file found and hashed; ValueError or FileNotFoundError."""
    chosen = []
    for row in rows(text):
        if not row["on"] or not row["name"] or not (row["ar"] or row["nar"]):
            continue
        entry = catalogue.find(row["name"])
        if not entry.usable:
            raise ValueError("{} cannot be used: {}".format(entry.name, entry.summary.get("problem")))
        halves = entry.summary.get("halves") or {}
        chosen.append({"name": entry.name, "path": entry.path, "sha256": catalogue.identity(entry.path),
                       "ar": row["ar"] if "ar" in halves else 0.0,
                       "nar": row["nar"] if "nar" in halves else 0.0})
    return chosen


def missing(specs) -> str:
    """Why a run cannot sing with these adapters -- a file gone since it was chosen -- or ""."""
    gone = [str(spec.get("name") or spec.get("path")) for spec in specs or ()
            if isinstance(spec, dict) and spec.get("path") and catalogue.stamp(spec["path"]) is None]
    if not gone:
        return ""
    return ("The LoRA file{} {} {} no longer where {} chosen. Put {} back, or choose again on the "
            "'YuE2 LoRA' node.".format("s" if len(gone) > 1 else "", ", ".join(gone),
                                       "are" if len(gone) > 1 else "is",
                                       "they were" if len(gone) > 1 else "it was",
                                       "them" if len(gone) > 1 else "it"))


def notices(specs, cot: str) -> list:
    """Warnings about adapters this run does not suit, as (level, message) pairs.

    Only what a file says about itself: Mothersuperior's instrumental adapter
    names the ``cot`` it was trained for, and in its authors' renders the
    score-first path ended songs by itself eight times in nine where the
    score-free one did not.
    """
    found = []
    for spec in specs or ():
        if not isinstance(spec, dict) or not spec.get("path"):
            continue
        verdict = catalogue.verdict(spec["path"]) or {}
        wanted = str(verdict.get("intended_cot") or "").strip()
        if wanted and wanted != cot:
            found.append(("warn", "{} was made to run with 'cot' at '{}', and this run has '{}'.".format(
                spec.get("name") or spec["path"], wanted, cot)))
    return found


class YuE2LoRA:
    """LoRA adapters for YuE2, as rows, for the nodes that sing."""

    DESCRIPTION = (
        "Adds LoRA adapters to the YuE2 nodes that sing: Generate Song, Plan, Plan Batch and "
        "Render Plan. One row is one file, with a strength for each half of the model: AR "
        "writes the score and sings the performance, NAR turns it into sound.\n\n"
        "Files in ComfyUI's own layout -- the ones ComfyUI's LoraLoader reads -- work, and so "
        "do files in m-a-p's layout, which ComfyUI's loader leaves unapplied without saying "
        "so. The list offers only files that are for YuE2, from ComfyUI's loras folders, and "
        "under each row says what the file changes and its trigger word, if it has one.\n\n"
        "Adapters are folded into the weights once per stage, so a song with them runs as "
        "fast as one without; with 'low_vram' they are held beside the packed layers and cost "
        "about a tenth of the token speed.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loras": ("STRING", {"multiline": False, "default": "[]"}),
            },
            "optional": {
                "lora": (LORA_TYPE, {"tooltip": CHAIN_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (LORA_TYPE,)
    RETURN_NAMES = ("lora",)
    FUNCTION = "pick"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, loras="[]", lora=None, unique_id=None):
        """The rows and each named file's identity on disk, so a file replaced under its name is read again."""
        try:
            named = [row["name"] for row in rows(loras) if row["name"]]
        except ValueError:
            return loras
        marks = []
        for name in named:
            try:
                marks.append(catalogue.stamp(catalogue.find(name).path))
            except FileNotFoundError:
                marks.append(None)
        return json.dumps([loras, marks])

    def pick(self, loras="[]", lora=None, unique_id=None):
        try:
            chosen = list(lora or []) + picked(loras)
        except (ValueError, FileNotFoundError) as error:
            refuse(unique_id, str(error))
        log.info("[yue2_comfy.lora] %d adapter%s: %s", len(chosen), "" if len(chosen) == 1 else "s",
                 ", ".join("{} (AR {:g}, NAR {:g})".format(item["name"], item["ar"], item["nar"])
                           for item in chosen) or "none")
        return (chosen,)


LORA_CLASSES = {"YuE2LoRA": YuE2LoRA}
LORA_NAMES = {"YuE2LoRA": "YuE2 LoRA"}
