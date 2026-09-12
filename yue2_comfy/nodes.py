"""The two nodes.

Nothing heavy is imported at module scope: no torch, and none of the vendored
modeling files. They pull transformers symbols that move between major versions,
and an import error here would un-register the whole pack instead of failing one
node's execution.
"""

from __future__ import annotations

import logging
import os

from . import devices
from .constants import (
    ATTENTION_CHOICES, CATEGORY, COT_CHOICES, DEFAULT_IDEA, DEFAULT_LYRICS,
    DEFAULT_OPTIONS, DEFAULT_STYLE, DOWNLOAD_CHOICES, LANGUAGE_CHOICES,
    LYRICS_TOOLTIP, MAX_SECONDS, OPTIONS_TYPE, QUANTIZATION_CHOICES, SAMPLE_RATE,
    SEED_TOOLTIP, STYLE_TOOLTIP, VAE_CHOICES, WRITER_AUTO, WRITER_LENGTH_CHOICES,
    WRITER_LENGTH_DEFAULT, WRITER_LENGTH_LINES, WRITER_MAX_NEW_TOKENS,
    WRITER_REPETITION_PENALTY, WRITER_TEMPERATURE, WRITER_TOP_K, WRITER_TOP_P,
    auto_seconds, length_lines,
)
from .progress import (NodeProgress, announce, interrupted, refuse,
                       translate_interrupt)
from .staged import (STAGED_CLASSES, STAGED_NAMES, resolve, session, words)

log = logging.getLogger(__name__)

IDEA_TOOLTIP = (
    "What the song is about, in one line. 'a sad song about winter, female vocal' is "
    "enough, and so is the same sentence in your own language.\n\n"
    "Everything the style line needs -- genre, voice, instruments, tempo -- is written "
    "for you, so you only have to say what you cannot be bothered to look up."
)

WRITER_MODEL_TOOLTIP = (
    "The language model that does the writing. '" + WRITER_AUTO + "' uses a GGUF you "
    "already have and downloads a 2.7 GB one only if you have none.\n\n"
    "The other entries are the GGUFs found in your ComfyUI model folders. LoRA adapters "
    "and mmproj files are left out: they cannot answer on their own."
)

LANGUAGE_TOOLTIP = (
    "What language the song is sung in. 'auto' lets the writer follow whatever language "
    "your idea is written in, which is usually what you meant."
)

def _length_tooltip() -> str:
    """The length widget's tooltip, written from the table it describes.

    Spelling the numbers out by hand would let the tooltip and the table drift
    apart the first time one of them was edited, and the tooltip is the only
    place a person ever sees them.
    """
    rows = ["'{}' -- {} sung lines, around {:.0f} seconds of singing".format(
        name, lines, auto_seconds("x\n" * lines))
        for name, lines in WRITER_LENGTH_LINES.items()]
    return (
        "How long the song should be.\n\n" + "\n".join(rows) + "\n\n"
        "It is an aim, not a promise: a small model writes somewhat more or fewer "
        "lines than asked, and the node says how many it got. 'very long' is about "
        "as much as YuE2 sings in one pass.\n\n"
        "Leave 'max_seconds' at 0 in the options and the generate node works the "
        "ceiling out from the lyrics it receives, whichever length you pick here.")


WRITER_LENGTH_TOOLTIP = _length_tooltip()

WRITER_SEED_TOOLTIP = (
    "The same seed with the same idea gives the same words. Change it for another take "
    "on the same idea."
)

WRITER_KEEP_TOOLTIP = (
    "Keep the writer in VRAM after it has written. Leave this off when YuE2 generates on "
    "the same card afterwards, or the song model has less room to work in."
)

INSTRUCTIONS_TOOLTIP = (
    "Anything extra for the writer, in your own words: 'no chorus', 'keep it funny', "
    "'first person', 'end on a question'.\n\n"
    "This goes after the writing rules, so it wins where the two disagree."
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

DOWNLOAD_TOOLTIP = (
    "Where to get the weights when they are not on this machine yet.\n\n"
    "'auto' fetches Comfy-Org's single checkpoint into ComfyUI/models/checkpoints. "
    "That is the same file ComfyUI's own YuE2 nodes read, so one download serves "
    "both, and the model manager may well have put it there already.\n\n"
    "'original' fetches the three files m-a-p released, into ComfyUI/models/YuE2. "
    "It is the only source of the legacy decoder, and 'auto' switches to it by "
    "itself when 'vae' is 'legacy'.\n\n"
    "'off' downloads nothing and says instead which files are missing, the direct "
    "link to each, and the exact folder to put it in."
)

QUANTIZATION_TOOLTIP = (
    "Which build of the checkpoint to download.\n\n"
    "'bf16' is the model as released. 'int8' is Comfy-Org's quantized build: 3.69 GB "
    "to fetch instead of 7.26 GB.\n\n"
    "It saves the download and not the VRAM. This pack's layers are ordinary torch "
    "linears, so an INT8 file is restored to BF16 as it loads and the card holds the "
    "same 6.8 GB either way. It is also not quite the same model -- the round trip "
    "costs about a percent of each weight -- so the same seed gives a different song "
    "from the two files.\n\n"
    "Whatever is already on disk is used before anything is downloaded."
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
                "download": (list(DOWNLOAD_CHOICES), {"default": d["download"],
                                                      "tooltip": DOWNLOAD_TOOLTIP}),
                "quantization": (list(QUANTIZATION_CHOICES),
                                 {"default": d["quantization"],
                                  "tooltip": QUANTIZATION_TOOLTIP}),
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
        from . import generate

        progress = NodeProgress(unique_id)
        style, lyrics = words(style, lyrics, unique_id)
        settings = resolve(options)

        with session(settings, unique_id, progress) as models:
            waveform, score, timing = generate.run(
                models, style, lyrics, seed, settings,
                progress=progress, cancelled=interrupted,
            )

        log.info(
            "[yue2_comfy] %.1f s of audio in %.1f s | %d semantic tokens at %.1f tok/s "
            "| %s, %s attention | seed %s",
            timing["seconds_of_audio"], timing["total_seconds"],
            timing["semantic"]["output_tokens"], timing["semantic"]["output_tps"],
            timing["semantic"]["execution"], timing["semantic"]["attention"], seed,
        )
        progress.finish("{:.0f} seconds of audio".format(timing["seconds_of_audio"]))
        return ({"waveform": waveform, "sample_rate": SAMPLE_RATE}, score)


class YuE2WriteSong:
    """One line of intent in, a style line and lyrics out."""

    DESCRIPTION = (
        "Turns a one-line idea into the style description and the tagged lyrics that "
        "YuE2 Generate Song wants, so nobody has to learn the prompt format to get a "
        "song. Any instruction-following GGUF does the writing -- a 4B on an 8 GB card "
        "is enough -- and one is downloaded on first use if the machine has none.\n\n"
        "Runs the model through llama-cpp-python when that is installed and through "
        "the official llama.cpp binaries when it is not, fetching about 32 MB of them "
        "the first time. The writer model is a separate download from YuE2 itself and "
        "carries its own licence."
    )

    @classmethod
    def INPUT_TYPES(cls):
        from . import llm

        return {
            "required": {
                "idea": ("STRING", {"multiline": True, "default": DEFAULT_IDEA,
                                    "tooltip": IDEA_TOOLTIP}),
                "model": (llm.choices(), {"tooltip": WRITER_MODEL_TOOLTIP}),
                "language": (list(LANGUAGE_CHOICES), {"default": LANGUAGE_CHOICES[0],
                                                      "tooltip": LANGUAGE_TOOLTIP}),
                "length": (list(WRITER_LENGTH_CHOICES),
                           {"default": WRITER_LENGTH_DEFAULT,
                            "tooltip": WRITER_LENGTH_TOOLTIP}),
                "seed": ("INT", {"default": 831001, "min": 0, "max": (1 << 63) - 1,
                                 "control_after_generate": True,
                                 "tooltip": WRITER_SEED_TOOLTIP}),
                "keep_model_loaded": ("BOOLEAN", {"default": False,
                                                  "tooltip": WRITER_KEEP_TOOLTIP}),
            },
            "optional": {
                "instructions": ("STRING", {"multiline": True, "default": "",
                                            "tooltip": INSTRUCTIONS_TOOLTIP}),
                "options": (OPTIONS_TYPE,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("style", "lyrics")
    FUNCTION = "write"
    CATEGORY = CATEGORY

    def write(self, idea, model, language, length, seed, keep_model_loaded,
              instructions="", options=None, unique_id=None):
        from . import download, llm, writer

        settings = dict(DEFAULT_OPTIONS)
        if options:
            settings.update(options)
        progress = NodeProgress(unique_id)

        idea = (idea or "").strip()
        if not idea:
            refuse(unique_id, "Write a line saying what the song is about. "
                              "'a sad song about winter, female vocal' is enough.")
        try:
            settings["device"] = devices.validate(settings["device"])
        except (ValueError, RuntimeError) as error:
            refuse(unique_id, str(error))

        lines = length_lines(length)
        try:
            path = llm.resolve(model, settings, progress)
        except (FileNotFoundError, download.DownloadError) as error:
            refuse(unique_id, str(error))

        style, lyrics, raw = "", "", ""
        try:
            for repair in (False, True):
                messages = writer.build_messages(idea, language, lines, instructions, repair)
                raw = llm.run(
                    path, messages, seed + (1 if repair else 0),
                    writer.context_needed(messages, WRITER_MAX_NEW_TOKENS),
                    settings["device"], keep_loaded=True, progress=progress,
                    settings=settings,
                    greedy=False, max_new_tokens=WRITER_MAX_NEW_TOKENS,
                    temperature=WRITER_TEMPERATURE, top_p=WRITER_TOP_P,
                    top_k=WRITER_TOP_K, repetition_penalty=WRITER_REPETITION_PENALTY,
                )
                style, lyrics = writer.split(raw)
                if writer.complete(style, lyrics):
                    break
                log.warning("[yue2_comfy] the writer answered without a usable style or "
                            "lyrics, asking once more")
                progress.text("That answer had no lyrics in it, asking again", force=True)
        except InterruptedError:
            translate_interrupt()
            raise
        finally:
            if not keep_model_loaded:
                llm.unload()

        if not writer.complete(style, lyrics):
            refuse(unique_id,
                   "The writer did not answer with a style and lyrics, twice. This is "
                   "what it said:\n\n" + (raw.strip()[:600] or "<nothing>")
                   + "\n\nA bigger model, or a more instruction-following one, usually "
                     "fixes it. So does saying what you want in 'instructions'.")

        notes = writer.findings(style, lyrics, lines)
        if keep_model_loaded and not llm.available():
            notes.append(
                "'keep_model_loaded' did nothing: without llama-cpp-python the model "
                "runs in a subprocess and leaves with it. Every run reloads it from "
                "the page cache, which costs seconds rather than a download.")
        announce(unique_id, notes)
        log.info("[yue2_comfy] wrote %d sung lines from %s via %s | seed %s",
                 writer.sung(lyrics), os.path.basename(path), llm.backend(), seed)
        progress.finish("{} sung lines".format(writer.sung(lyrics)))
        return (style, lyrics)


NODE_CLASS_MAPPINGS = {
    "YuE2GenerateSong": YuE2GenerateSong,
    "YuE2WriteSong": YuE2WriteSong,
    "YuE2Options": YuE2Options,
}
NODE_CLASS_MAPPINGS.update(STAGED_CLASSES)

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2GenerateSong": "YuE2 Generate Song",
    "YuE2WriteSong": "YuE2 Write Song",
    "YuE2Options": "YuE2 Options",
}
NODE_DISPLAY_NAME_MAPPINGS.update(STAGED_NAMES)
"""One registry, so that whatever reads this module sees every node.

The release workflow and the tests both import these two names to check that
the pack loads. A staged node registered anywhere else would be absent from
both and nobody would find out until it was shipped."""
