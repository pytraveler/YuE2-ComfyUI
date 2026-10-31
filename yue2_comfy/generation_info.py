"""Portable JSON about one completed generation, and ordinary ComfyUI outputs for it."""

from __future__ import annotations

import json

from . import checkpoints
from .constants import ADVANCED_CATEGORY, normalize_seed


def encode(style, lyrics, seed, settings, score="") -> str:
    """Capture the values used by this result, without local absolute model paths."""
    public = {key: value for key, value in settings.items()
              if key not in (checkpoints.SOURCE, "loras") and not key.startswith("_")}
    public["loras"] = [{key: row[key] for key in ("name", "sha256", "ar", "nar") if key in row}
                       for row in settings.get("loras", ()) if isinstance(row, dict)]
    source = settings.get(checkpoints.SOURCE) or {}
    data = {"format_version": 1, "lyrics": lyrics or "", "seed": None if seed is None else normalize_seed(seed),
            "style": style or "", "model_name": source.get("name", ""),
            "steps": int(settings["ode_steps"]), "settings": public, "score_abc": score or ""}
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)


def read(text: str) -> dict:
    """Check the stable fields while allowing future versions to carry more data."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as error:
        raise ValueError("generation_info must be the JSON from a YuE2 generation output.") from error
    if not isinstance(data, dict) or data.get("format_version") != 1:
        raise ValueError("This generation_info format is not supported.")
    for key in ("lyrics", "style", "model_name"):
        if not isinstance(data.get(key), str):
            raise ValueError("generation_info has no valid '" + key + "'.")
    for key in ("seed", "steps"):
        if type(data.get(key)) is not int or data[key] < 0:
            raise ValueError("generation_info has no valid '" + key + "'.")
    return data


class YuE2GenerationInfo:
    """One generation's JSON as fields that ordinary text nodes can save."""

    DESCRIPTION = (
        "Reads generation_info from a YuE2 singing node. Outputs the lyrics, seed, style, "
        "checkpoint name and acoustic steps used by that result, plus a readable text to save. "
        "The original input is JSON and can be saved directly too."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"generation_info": ("STRING", {"forceInput": True,
                "tooltip": "The JSON output of YuE2 Generate Song, Render Plan or Decode Latents."})}}

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("lyrics", "seed", "style", "model_name", "steps", "text")
    FUNCTION = "extract"
    CATEGORY = ADVANCED_CATEGORY

    def extract(self, generation_info):
        data = read(generation_info)
        text = "Model: {model_name}\nSeed: {seed}\nSteps: {steps}\n\nStyle:\n{style}\n\nLyrics:\n{lyrics}".format(**data)
        return (data["lyrics"], data["seed"], data["style"], data["model_name"], data["steps"], text)
