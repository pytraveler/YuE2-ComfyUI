"""The two nodes.

Nothing heavy is imported at module scope: no torch, and none of the vendored
modeling files. They pull transformers symbols that move between major versions,
and an import error here would un-register the whole pack instead of failing one
node's execution.
"""

from __future__ import annotations

import logging
import os
import time

from . import devices, edits, phrasing, songs, transpose
from .constants import (
    ATTENTION_CHOICES, CATEGORY, COT_CHOICES, DEFAULT_IDEA, DEFAULT_LYRICS,
    DEFAULT_OPTIONS, DEFAULT_STYLE, DOWNLOAD_CHOICES, LANGUAGE_CHOICES, LEGACY_ATTENTION,
    LORA_INPUT_TOOLTIP, LORA_TYPE, LYRICS_TOOLTIP, MAX_SECONDS, OFFLOAD_CHOICES, OPTIONS_TYPE,
    QUANTIZATION_CHOICES, SAMPLE_RATE, SEED_TOOLTIP, STYLE_TOOLTIP, TRANSPOSE_LIMIT, VAE_CHOICES, WRITER_AUTO,
    WRITER_LENGTH_CHOICES,
    WRITER_LENGTH_DEFAULT, WRITER_LENGTH_LINES, WRITER_MAX_NEW_TOKENS,
    WRITER_REPETITION_PENALTY, WRITER_TEMPERATURE, WRITER_TOP_K, WRITER_TOP_P,
    auto_seconds, length_lines,
)
from .progress import (Band, NodeProgress, announce, interrupted, refuse,
                       translate_interrupt)
from .load_midi import MIDI_CLASSES, MIDI_NAMES
from .edit_track import EDIT_CLASSES, EDIT_NAMES
from .lora.node import LORA_CLASSES, LORA_NAMES
from .staged import (STAGED_CLASSES, STAGED_NAMES, adapters, engine_line, resolve, session,
                     stage_times, words)
from .transcribe import TRANSCRIBE_CLASSES, TRANSCRIBE_NAMES
from .vocals_only import VOCALS_CLASSES, VOCALS_NAMES, VOICE_SHARE, separator_weights, voice_of

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
    "Keep the model loaded after the run.\n\n"
    "On saves about five seconds per run while you iterate on lyrics or seeds, and "
    "holds the memory until Unload Models is pressed, ComfyUI restarts or another "
    "YuE2 run needs a different model: on the card, or partly in system RAM when "
    "'offload' has moved a half off. "
    "Off frees it immediately, which is what you want when video or image nodes run "
    "next in the same graph."
)

ATTENTION_TOOLTIP = (
    "How the model attends while it writes the song, token by token.\n\n"
    "'sdpa' is the default and sings a seed as before. 'fast' is quicker and needs "
    "nothing extra. 'flash' is the quickest and needs the flash-attn package. Both "
    "need an RTX 30 card or newer.\n\n"
    "Each one repeats a seed, but each sings it its own way."
)

DOWNLOAD_TOOLTIP = (
    "Where to get the weights when they are not on this machine yet.\n\n"
    "'auto' fetches Comfy-Org's single checkpoint into ComfyUI/models/checkpoints. "
    "That is the same file ComfyUI's own YuE2 nodes read, so one download serves "
    "both, and the model manager may well have put it there already.\n\n"
    "'original' fetches the three files m-a-p released, into ComfyUI/models/YuE2.\n\n"
    "The legacy decoder is published only by m-a-p. With 'vae' at 'legacy' and the "
    "model already on this machine -- m-a-p's files, or Comfy-Org's BF16 checkpoint, "
    "which holds the same weights bit for bit -- that one 0.49 GB file is all that is "
    "fetched. Comfy-Org's INT8 checkpoint counts only with 'quantization' at 'int8'. "
    "On a machine with nothing yet, 'auto' takes m-a-p's files for 'legacy', the "
    "smaller download.\n\n"
    "'off' downloads nothing and says instead which files are missing, the direct "
    "link to each, and the exact folder to put it in."
)

QUANTIZATION_TOOLTIP = (
    "Which build of the checkpoint to download.\n\n"
    "'bf16' is the model as released. 'int8' is Comfy-Org's quantized build: 3.69 GB "
    "to fetch instead of 7.26 GB.\n\n"
    "It saves the download and not the VRAM: an INT8 file is restored to BF16 as it "
    "loads and the card holds the same 6.8 GB either way. What keeps weights packed on "
    "the card is 'low_vram', which does its own packing from whichever file you have. "
    "This one is also not quite the same model -- the round trip costs about a percent "
    "of each weight -- so the same seed gives a different song from the two files.\n\n"
    "Whatever is already on disk is used before anything is downloaded."
)

ODE_TOOLTIP = (
    "Solver steps for the acoustic stage. 32 is what the model was released with.\n\n"
    "Fewer is faster and thinner; more costs time and changes the result rather than "
    "clearly improving it."
)


LOW_VRAM_TOOLTIP = (
    "For a card of about 4 GB. The models' layers stay on the card as INT8, half their "
    "memory, and the work goes in smaller pieces -- the song and the models that hear its "
    "words alike.\n\n"
    "It is slower, and the same seed sings a different take. Leave it off unless the card "
    "needs it."
)


OFFLOAD_TOOLTIP = (
    "Whether the whole model stays on the card, or only the half the running stage "
    "uses.\n\n"
    "YuE2 has one set of weights that writes the score and the performance, and another "
    "that turns them into audio; no stage needs both. 'on' keeps only the half the stage "
    "needs, and neither during the decode, and it keeps only the rows of the vocabulary "
    "the running phase can use: measured, a 40-second song peaked at 4.4 GiB instead of "
    "9.8, and a four-minute one at 4.5 GiB instead of 10.0, a second or two faster. 'off' keeps everything on the card. 'auto' moves a half off only "
    "when a stage would not fit beside it.\n\n"
    "The score, the notes and the words are the same in every mode: the weights are the "
    "same wherever they are kept. The audio file matches to the last byte too, as long as "
    "the decode has the same room to work in -- on a card so full that cuDNN has to pick a "
    "cheaper convolution, the last stage renders a hair differently, measured at 92 dB "
    "below the song. The weights waiting their turn sit in system RAM, up to 6.7 GiB."
)

TRANSPOSE_TOOLTIP = (
    "Moves the song to another key, in semitones: 2 is a whole tone up, -3 a minor "
    "third down, 0 sings the score as the model wrote it.\n\n"
    "YuE2 takes no key from the style line -- asked for 'A minor' there, it kept its "
    "own key every time -- but it follows its score closely, so this moves the score: "
    "every note, chord and key by the same step, just before it is sung. Measured, the "
    "song lands exactly that far away and the voice moves with it, the octave "
    "included: 12 puts the singer a full octave higher. It is a new take of the same "
    "tune rather than the old recording pitched up, because the moved score is sung "
    "from its first note.\n\n"
    "It needs a score, so not with 'cot' off, and a score it cannot read note by note "
    "is refused rather than guessed at. In 'YuE2 Generate Song' the 'score_abc' output "
    "is the moved score; in 'YuE2 Render Plan' the plan's score, or the one pasted in, "
    "is the one moved."
)

VOCALS_ONLY_TOOLTIP = (
    "Outputs only the voice. The song is made as always, then Mel-Band RoFormer separates the "
    "vocals from the band, and 'audio' carries the voice alone, the same length and rate.\n\n"
    "The song is the one this seed gives with the switch off, so an ordinary song keeps silence "
    "where its intro and instrumental breaks were. For an a cappella song, write 'a cappella' in "
    "the style: the model then keeps the voice going, while 'no instruments' in the style was "
    "measured to change nothing. Without separating, even an a cappella style leaves a soft pad "
    "under the voice in most songs.\n\n"
    "The first time, this downloads the separator (0.85 GB, MIT) into models/YuE2. For a "
    "recording made elsewhere, use 'YuE2 Vocals Only'."
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
                "offload": (list(OFFLOAD_CHOICES), {"default": d["offload"],
                                                    "tooltip": OFFLOAD_TOOLTIP}),
                "transpose": ("INT", {"default": d["transpose"], "min": -TRANSPOSE_LIMIT,
                                      "max": TRANSPOSE_LIMIT, "step": 1,
                                      "tooltip": TRANSPOSE_TOOLTIP}),
                "vocals_only": ("BOOLEAN", {"default": d["vocals_only"],
                                            "tooltip": VOCALS_ONLY_TOOLTIP}),
                "low_vram": ("BOOLEAN", {"default": d["low_vram"],
                                         "tooltip": LOW_VRAM_TOOLTIP}),
            },
        }

    RETURN_TYPES = (OPTIONS_TYPE,)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    @classmethod
    def VALIDATE_INPUTS(cls, attention_backend=DEFAULT_OPTIONS["attention_backend"]):
        """Let a workflow saved with an attention choice that is gone still run.

        ComfyUI checks a list widget against its list unless this names the
        input, and then asks this instead. 'cudnn' was on the list until
        2026-09-24; a workflow saved with it would otherwise stop at the queue
        with "Value not in list". It runs as what replaced it; see
        ``attention.LEGACY``.
        """
        if attention_backend in ATTENTION_CHOICES or attention_backend in LEGACY_ATTENTION:
            return True
        return "attention_backend must be one of {}, not {!r}".format(
            ", ".join(ATTENTION_CHOICES), attention_backend)

    def build(self, **kwargs):
        options = dict(DEFAULT_OPTIONS)
        options.update(kwargs)
        return (options,)



EDITED_SCORE_TOOLTIP = (
    "An edited score kept on this node. While it is empty, as it starts, the node "
    "writes a new score on every run, exactly as it always has.\n\n"
    "'Edit score...' fills it after a run: change notes and chords there, and the "
    "next run sings the edit instead of writing a score, and 'Reset score' empties it "
    "again. The edit belongs to the "
    "style and lyrics it was made for -- with other words the node writes a new "
    "score and says so -- while a new seed sings the same edit as a new take. A "
    "score wired in is sung as it arrives.\n\n"
    "A score that names no section -- a bare tune, as 'YuE2 Load MIDI' hands one on "
    "-- has the lyrics laid along it first: each line on a phrase with about as many "
    "notes as the line has syllables, the tune repeated when the words outlast it, "
    "and the song ending one bar after the last line. The node says which bars "
    "each section is sung on."
)

GENERATE_INSTEAD = "the model wrote a new score for these words"


class YuE2GenerateSong:
    """Style and lyrics in, a finished song out."""

    DESCRIPTION = (
        "Generates a complete song at 48 kHz stereo from a style description and "
        "lyrics, using YuE2-3B. The model plans a readable score first, then sings it; "
        "after a run, 'Edit score...' opens that score to change notes before the "
        "next one. The weights are downloaded on first use, with progress shown on "
        "the node.\n\n"
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
                "score_abc": ("STRING", {"multiline": True, "default": "",
                                         "tooltip": EDITED_SCORE_TOOLTIP}),
                "lora": (LORA_TYPE, {"tooltip": LORA_INPUT_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "score_abc")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, style, lyrics, seed, options=None, score_abc="", lora=None, unique_id=None):
        from . import generate

        progress = NodeProgress(unique_id, title="YuE2 Generate Song")
        style, lyrics = words(style, lyrics, unique_id)
        settings = resolve(options)
        settings["loras"] = list(lora or [])
        if int(settings.get("transpose") or 0) and settings["cot"] == "off":
            refuse(unique_id, transpose.COT_OFF)
        adapters(settings, unique_id)
        edit = edits.read(score_abc)
        problem = edits.mismatch(edit, style, lyrics, settings["cot"], GENERATE_INSTEAD)
        if problem:
            announce(unique_id, [("warn", problem)])
        edited = edit.score if edit.score and not problem else None
        tune_seconds = None
        if edited:
            laid = phrasing.lay(edited, lyrics)
            if laid is not None:
                edited, tune_seconds = laid.score, laid.seconds
                announce(unique_id, laid.notices)
        if edited and settings["cot"] == "full" and edits.chordless(edited):
            announce(unique_id, [("warn", edits.CHORDLESS)])
        if edited and settings["cot"] == "melody" and edits.chorded(edited):
            announce(unique_id, [("warn", edits.CHORDED)])
        if edited:
            clash = generate.tempo_clash(style, edited)
            if clash:
                announce(unique_id, [("warn", clash)])
        if edited:
            log.info("[yue2_comfy] singing the score given to the node, kept on it or "
                     "wired in; no score is written this run")

        voice = bool(settings.get("vocals_only"))
        song_progress = Band(progress, 0.0, 1.0 - VOICE_SHARE) if voice else progress
        separator = separator_weights(settings, unique_id, song_progress) if voice else None
        began = time.perf_counter()
        with session(settings, unique_id, song_progress) as models:
            waveform, score, written, timing, performance = generate.run(
                models, style, lyrics, seed, settings,
                progress=song_progress, cancelled=interrupted, edited=edited,
                tune_seconds=tune_seconds,
            )
        del models
        if voice:
            waveform = voice_of({"waveform": waveform, "sample_rate": SAMPLE_RATE}, settings, unique_id,
                                Band(progress, 1.0 - VOICE_SHARE, 1.0), path=separator)["waveform"]
        audio = {"waveform": waveform, "sample_rate": SAMPLE_RATE}
        songs.keep(audio, "YuE2 Generate Song", style, lyrics, seed, settings, score, performance)

        log.info("[yue2_comfy] %.1f s of audio in %.1f s | %s | seed %s",
                 timing["seconds_of_audio"], timing["total_seconds"], engine_line(timing), seed)
        log.info("[yue2_comfy] %s", stage_times(timing, time.perf_counter() - began))
        short = generate.ended_early(score, timing["seconds_of_audio"],
                                     generate.song_ceiling(settings["max_seconds"], lyrics, tune_seconds))
        if short:
            announce(unique_id, [("warn", short)])
        progress.finish("{:.0f} seconds of {}".format(timing["seconds_of_audio"], "vocals" if voice else "audio"))
        ui = {edits.WORDS_UI: [edits.mark(style, lyrics, settings["cot"])],
              edits.AUTO_SECONDS_UI: [auto_seconds(lyrics)]}
        if edited is None:
            ui[edits.SCORE_UI] = [written]
        return {"ui": ui, "result": (audio, score)}


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
        progress = NodeProgress(unique_id, title="YuE2 Write Song")

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
NODE_CLASS_MAPPINGS.update(TRANSCRIBE_CLASSES)
NODE_CLASS_MAPPINGS.update(MIDI_CLASSES)
NODE_CLASS_MAPPINGS.update(VOCALS_CLASSES)
NODE_CLASS_MAPPINGS.update(LORA_CLASSES)
NODE_CLASS_MAPPINGS.update(EDIT_CLASSES)

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2GenerateSong": "YuE2 Generate Song",
    "YuE2WriteSong": "YuE2 Write Song",
    "YuE2Options": "YuE2 Options",
}
NODE_DISPLAY_NAME_MAPPINGS.update(STAGED_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(TRANSCRIBE_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(MIDI_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(VOCALS_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(LORA_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(EDIT_NAMES)
"""One registry, so that whatever reads this module sees every node.

The release workflow and the tests both import these two names to check that
the pack loads. A staged node registered anywhere else would be absent from
both and nobody would find out until it was shipped."""
