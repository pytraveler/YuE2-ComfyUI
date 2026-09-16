"""YuE2 Vocals Only, and the step behind 'vocals_only' in YuE2 Options: a recording in, its voice out.

Both run the same function, so the switch on a song and the node on any
recording refuse the same way, find and fetch the same weights and keep the
model under the same 'keep_model_loaded'.

The model is Kimberley Jensen's Mel-Band RoFormer vocal model, run by this
pack's own implementation (see ``vocals``). YuE2 writes one stream for the whole
mix and has no voice-only output of its own, and 'a cappella' in the style
still leaves a held pad under the voice in most songs (measured on 72), so
the voice is taken from the finished mix instead.

Module scope stays light for the same reason nodes.py does: no torch.
"""

from __future__ import annotations

import logging
import time

from . import devices
from .constants import CATEGORY, OPTIONS_TYPE
from .progress import NodeProgress, interrupted, refuse, translate_interrupt

log = logging.getLogger(__name__)

VOICE_SHARE = 0.1
"""The last tenth of a song node's bar: separating a song takes about a tenth of the time making it did."""

AUDIO_TOOLTIP = (
    "Any audio: a song from this pack, a recording loaded with Load Audio, the output of another "
    "node. Mono and stereo at any sample rate are taken, and the voice comes back in the same shape."
)

OPTIONS_TOOLTIP = (
    "Optional. 'download', 'device' and 'keep_model_loaded' are read from it; the rest belongs to "
    "the song nodes."
)


def separator_weights(settings: dict, unique_id, progress) -> str:
    """The separator's weights file, fetched first when downloading is on; refused on the node when it cannot be had.

    A song node asks before it sings: weights that are missing with 'download'
    off, or a download that fails, are then heard about at once, rather than
    after minutes of singing and at the price of the song.
    """
    from . import download

    try:
        return download.ensure_vocals(settings, progress)
    except (FileNotFoundError, download.DownloadError) as error:
        refuse(unique_id, str(error))


def voice_of(audio: dict, settings: dict, unique_id, progress, path=None) -> dict:
    """ComfyUI audio in, the same shape and rate with only the voice in it; a failure is refused on the node.

    ``path`` is the weights file a song node already found before singing;
    without it they are found, or fetched, here.
    """
    try:
        spec = devices.validate(settings.get("device", "auto"))
    except (ValueError, RuntimeError) as error:
        refuse(unique_id, str(error))
    if path is None:
        path = separator_weights(settings, unique_id, progress)

    from .vocals import runtime

    waveform = audio["waveform"]
    rate = int(audio["sample_rate"])
    began = time.monotonic()
    try:
        vocals = runtime.separate(path, devices.resolve(spec), waveform, rate, progress=progress,
                                  cancelled=interrupted)
    except InterruptedError:
        translate_interrupt()
        raise
    except ValueError as error:
        refuse(unique_id, str(error))
    finally:
        if not settings.get("keep_model_loaded"):
            runtime.unload()
    log.info("[yue2_comfy] separated the voice of %.1f s of audio in %.1f s",
             waveform.shape[-1] / float(rate), time.monotonic() - began)
    return {"waveform": vocals, "sample_rate": rate}


class YuE2VocalsOnly:
    """A recording in, its voice out."""

    DESCRIPTION = (
        "Separates the voice from a recording and outputs only the voice: the same length, sample "
        "rate and channels, with the band taken out. It is the step 'vocals_only' in YuE2 Options "
        "adds to a song, as a node of its own, for audio from anywhere, or for keeping a song and "
        "its voice from one run.\n\n"
        "Uses Mel-Band RoFormer, Kimberley Jensen's vocal model, downloaded on first use into "
        "models/YuE2 (0.85 GB, MIT). A copy already on the machine, kijai's conversions included, "
        "is found and used instead."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": AUDIO_TOOLTIP}),
            },
            "optional": {
                "options": (OPTIONS_TYPE, {"tooltip": OPTIONS_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("vocals",)
    FUNCTION = "separate"
    CATEGORY = CATEGORY

    def separate(self, audio, options=None, unique_id=None):
        from .staged import resolve

        if not audio or audio.get("waveform") is None:
            refuse(unique_id, "No audio is connected. Join a song or a Load Audio node to 'audio'.")
        progress = NodeProgress(unique_id)
        vocals = voice_of(audio, resolve(options), unique_id, progress)
        progress.finish("The voice of {:.0f} seconds of audio".format(
            vocals["waveform"].shape[-1] / float(vocals["sample_rate"])))
        return (vocals,)


VOCALS_CLASSES = {"YuE2VocalsOnly": YuE2VocalsOnly}
VOCALS_NAMES = {"YuE2VocalsOnly": "YuE2 Vocals Only"}
