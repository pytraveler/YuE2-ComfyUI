"""The two nodes.

Nothing heavy is imported at module scope: no torch, and none of the vendored
modeling files. They pull transformers symbols that move between major versions,
and an import error here would un-register the whole pack instead of failing one
node's execution.
"""

from __future__ import annotations

import logging

from . import devices
from .constants import (
    ATTENTION_CHOICES, CATEGORY, COT_CHOICES, DEFAULT_LYRICS, DEFAULT_OPTIONS,
    DEFAULT_STYLE, MAX_SECONDS, OPTIONS_TYPE, SAMPLE_RATE, VAE_CHOICES,
)
from .progress import NodeProgress, refuse

log = logging.getLogger(__name__)

STYLE_TOOLTIP = (
    "What the song should sound like: language, genre, voice, instruments, tempo.\n\n"
    "This is a description, not a list of tags. 'English, warm piano pop, expressive "
    "female voice, 88 BPM' works better than 'pop, piano, female'."
)

LYRICS_TOOLTIP = (
    "The words to sing, with section markers on their own lines: [Verse], [Chorus], "
    "[Bridge], [Outro].\n\n"
    "Leave it empty for an instrumental. Long lyrics eat into the context the song "
    "itself needs, so a very long text lowers the ceiling on 'max_seconds'."
)

SEED_TOOLTIP = (
    "The same seed with the same settings gives the same song, byte for byte.\n\n"
    "That holds only while 'attention_backend' is 'sdpa', which is the default."
)

COT_TOOLTIP = (
    "How much of the composition is planned before any audio is generated.\n\n"
    "'full' writes a melody-and-chord score first and is the default for new songs. "
    "'melody' plans the melody only and lets the accompaniment follow the style. "
    "'off' goes straight from the lyrics to audio, which is faster and gives up the "
    "readable plan."
)

CFG_TOOLTIP = (
    "Classifier-free guidance. 0 means 'use the value the model was released with' "
    "-- 1.0 normally, 1.01 when 'cot' is 'off'.\n\n"
    "Anything other than 1.0 runs a second, unconditional branch: the song takes "
    "about twice as long and needs about twice the KV cache. Raise it only when the "
    "result ignores the style."
)

MAX_SECONDS_TOOLTIP = (
    "Ceiling on the length of the song. 0 works it out from the lyrics: about a "
    "minute for a verse and a chorus, longer as the words do.\n\n"
    "The model usually ends the song by itself, well short of the ceiling. The "
    "ceiling is for when it does not: on very few lines it can sing on long after "
    "the words have run out. Reaching it cuts the song off mid-phrase, and the node "
    "says so when that happens.\n\n"
    "Changing this number changes the song itself, not only its length -- it sizes "
    "the attention cache, and the same seed under a different ceiling is a different "
    "take. Leave it alone while you are hunting for a seed."
)

VAE_TOOLTIP = (
    "Which audio decoder turns the latents into sound.\n\n"
    "'standard' is the one released for listening. 'legacy' is the decoder the "
    "published benchmark numbers were measured with; use it only to reproduce those. "
    "They are different weights of the same size, and the same latents decoded by "
    "each will not sound identical."
)

KEEP_TOOLTIP = (
    "Keep the 6.8 GB model on the card after the run.\n\n"
    "On saves about five seconds per run while you iterate on lyrics or seeds, and "
    "holds the VRAM until ComfyUI restarts or another YuE2 run needs a different "
    "model. Off frees it immediately, which is what you want when video or image "
    "nodes run next in the same graph."
)

ATTENTION_TOOLTIP = (
    "Which attention kernel the decode loop uses.\n\n"
    "'sdpa' is the default and is reproducible: the same seed gives the same song. "
    "'cudnn' is about 17 percent faster and is NOT reproducible -- measured over four "
    "runs of one seed it produced four different songs. Use it only when you are "
    "exploring and do not need to come back to a result."
)

AUTO_DOWNLOAD_TOOLTIP = (
    "Fetch the model files from Hugging Face when they are not on disk.\n\n"
    "That is 7.26 GB on the first run, and the queue is busy until it finishes. Turn "
    "it off to get a message listing the three files, their direct links and the exact "
    "folder, so you can download them with anything you like."
)

ODE_TOOLTIP = (
    "Solver steps for the acoustic stage. 32 is what the model was released with.\n\n"
    "Fewer is faster and thinner; more costs time and changes the result rather than "
    "clearly improving it."
)

SAMPLING_TOOLTIP = "Sampling for the {} stage. The defaults are the released values."


class YuE2Options:
    """Optional settings for YuE2 Generate Song."""

    DESCRIPTION = (
        "Everything about a YuE2 song other than the style, the lyrics and the seed. "
        "Connect it to 'YuE2 Generate Song' when you want to change something; leave "
        "it out and the released defaults are used."
    )

    @classmethod
    def INPUT_TYPES(cls):
        d = DEFAULT_OPTIONS
        return {
            "required": {
                "cot": (list(COT_CHOICES), {"default": d["cot"], "tooltip": COT_TOOLTIP}),
                "max_seconds": ("FLOAT", {"default": d["max_seconds"], "min": 0.0,
                                          "max": MAX_SECONDS, "step": 1.0, "round": 0.1,
                                          "tooltip": MAX_SECONDS_TOOLTIP}),
                "keep_model_loaded": ("BOOLEAN", {"default": d["keep_model_loaded"],
                                                  "tooltip": KEEP_TOOLTIP}),
            },
            "optional": {
                "cfg_scale": ("FLOAT", {"default": d["cfg_scale"], "min": 0.0, "max": 20.0,
                                        "step": 0.05, "round": 0.01, "tooltip": CFG_TOOLTIP}),
                "vae": (list(VAE_CHOICES), {"default": d["vae"], "tooltip": VAE_TOOLTIP}),
                "device": (devices.choices(), {"default": d["device"],
                                               "tooltip": devices.tooltip()}),
                "attention_backend": (list(ATTENTION_CHOICES),
                                      {"default": d["attention_backend"],
                                       "tooltip": ATTENTION_TOOLTIP}),
                "auto_download": ("BOOLEAN", {"default": d["auto_download"],
                                              "tooltip": AUTO_DOWNLOAD_TOOLTIP}),
                "ode_steps": ("INT", {"default": d["ode_steps"], "min": 8, "max": 64,
                                      "tooltip": ODE_TOOLTIP}),
                "abc_temperature": ("FLOAT", {"default": d["abc_temperature"], "min": 0.0,
                                              "max": 5.0, "step": 0.01,
                                              "tooltip": SAMPLING_TOOLTIP.format("score")}),
                "abc_top_p": ("FLOAT", {"default": d["abc_top_p"], "min": 0.01, "max": 1.0,
                                        "step": 0.01,
                                        "tooltip": SAMPLING_TOOLTIP.format("score")}),
                "abc_top_k": ("INT", {"default": d["abc_top_k"], "min": 1, "max": 1000,
                                      "tooltip": SAMPLING_TOOLTIP.format("score")}),
                "temperature": ("FLOAT", {"default": d["temperature"], "min": 0.0, "max": 5.0,
                                          "step": 0.01,
                                          "tooltip": SAMPLING_TOOLTIP.format("song")}),
                "top_p": ("FLOAT", {"default": d["top_p"], "min": 0.01, "max": 1.0,
                                    "step": 0.01,
                                    "tooltip": SAMPLING_TOOLTIP.format("song")}),
                "top_k": ("INT", {"default": d["top_k"], "min": 1, "max": 1000,
                                  "tooltip": SAMPLING_TOOLTIP.format("song")}),
                "repetition_penalty": ("FLOAT", {"default": d["repetition_penalty"],
                                                 "min": 0.1, "max": 5.0, "step": 0.005,
                                                 "tooltip": SAMPLING_TOOLTIP.format("song")}),
            },
        }

    RETURN_TYPES = (OPTIONS_TYPE,)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, **kwargs):
        options = dict(DEFAULT_OPTIONS)
        options.update(kwargs)
        return (options,)



def _interrupted() -> bool:
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


def _translate_interrupt() -> None:
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
    try:
        import comfy.model_management as mm
    except Exception:
        return
    mm.throw_exception_if_processing_interrupted()
    raise mm.InterruptProcessingException()


class YuE2GenerateSong:
    """Style and lyrics in, a finished song out."""

    DESCRIPTION = (
        "Generates a complete song at 48 kHz stereo from a style description and "
        "lyrics, using YuE2-3B. The model plans a readable score first, then sings it. "
        "The weights are downloaded on first use, with progress shown on the node.\n\n"
        "The model weights are licensed CC BY-NC 4.0, which is non-commercial. The code "
        "of this pack is Apache-2.0."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": ("STRING", {"multiline": True, "default": DEFAULT_STYLE,
                                     "tooltip": STYLE_TOOLTIP}),
                "lyrics": ("STRING", {"multiline": True, "default": DEFAULT_LYRICS,
                                      "tooltip": LYRICS_TOOLTIP}),
                "seed": ("INT", {"default": 831001, "min": 0, "max": (1 << 63) - 1,
                                 "control_after_generate": True, "tooltip": SEED_TOOLTIP}),
            },
            "optional": {
                "options": (OPTIONS_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "score_abc")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, style, lyrics, seed, options=None, unique_id=None):
        settings = dict(DEFAULT_OPTIONS)
        if options:
            settings.update(options)
        progress = NodeProgress(unique_id)

        style = (style or "").strip()
        lyrics = (lyrics or "").strip()
        if not style and not lyrics:
            refuse(unique_id, "Give a style, lyrics, or both. With neither there is "
                              "nothing to write a song from.")
        try:
            settings["device"] = devices.validate(settings["device"])
        except (ValueError, RuntimeError) as error:
            refuse(unique_id, str(error))

        from . import generate, loader

        try:
            files = loader.locate(settings["vae"])
        except FileNotFoundError as error:
            refuse(unique_id, str(error))

        try:
            models = loader.acquire(
                files, settings["device"], settings["vae"], progress,
            )
            waveform, score, timing = generate.run(
                models, style, lyrics, seed, settings,
                progress=progress, cancelled=_interrupted,
            )
        except InterruptedError:
            _translate_interrupt()
            raise
        except ValueError as error:
            refuse(unique_id, str(error))
        finally:
            if not settings["keep_model_loaded"]:
                loader.unload()

        log.info(
            "[yue2_comfy] %.1f s of audio in %.1f s | %d semantic tokens at %.1f tok/s "
            "| %s, %s attention | seed %s",
            timing["seconds_of_audio"], timing["total_seconds"],
            timing["semantic"]["output_tokens"], timing["semantic"]["output_tps"],
            timing["semantic"]["execution"], timing["semantic"]["attention"], seed,
        )
        progress.finish("{:.0f} seconds of audio".format(timing["seconds_of_audio"]))
        return ({"waveform": waveform, "sample_rate": SAMPLE_RATE}, score)


NODE_CLASS_MAPPINGS = {
    "YuE2GenerateSong": YuE2GenerateSong,
    "YuE2Options": YuE2Options,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2GenerateSong": "YuE2 Generate Song",
    "YuE2Options": "YuE2 Options",
}
