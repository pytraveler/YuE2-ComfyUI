"""The stages of a song as separate nodes, for when one node is not enough.

'YuE2 Generate Song' does all of this in a single step and is what most people
want. These exist for the one thing a single node cannot offer: a look at the
score before it is sung, and a chance to change it. YuE2 writes an ABC
transcription first and then performs it, so the score is the last point at
which a change is cheap -- and it is the only honest lever on where the
stresses of a line fall, which capitalising syllables only approximates.

Nothing here is a second pipeline. Every node calls the same stage functions in
generate.py, in the same order, that the single node calls.

Module scope stays light for the same reason nodes.py does: no torch and none
of the vendored modeling code, so an import error cannot take the pack out of
ComfyUI's node list.
"""

from __future__ import annotations

import contextlib
import logging

from . import devices, edits, phrasing
from .constants import (
    ADVANCED_CATEGORY, DEFAULT_LYRICS, DEFAULT_OPTIONS, DEFAULT_STYLE,
    LATENTS_TYPE, LYRICS_TOOLTIP, OPTIONS_TYPE, PLAN_TYPE, PLANS_TYPE,
    SAMPLE_RATE, SEED_TOOLTIP, STYLE_TOOLTIP, auto_seconds, normalize_seed,
)
from .edits import AUTO_SECONDS_UI, SCORE_UI, WORDS_UI
from .progress import (Band, NodeProgress, announce, interrupted, refuse,
                       translate_interrupt)
from .vocals_only import VOICE_SHARE, separator_weights, voice_of

log = logging.getLogger(__name__)

MAX_TAKES = 8

RENDER_INSTEAD = "the plan's own score was sung"

SEED_MAX = (1 << 63) - 1

PLAN_TOOLTIP = (
    "The score, and the words it was written for, from a 'YuE2 Plan' node.\n\n"
    "It carries the style, the lyrics, the seed and the settings the score was "
    "written under, so the node it feeds needs nothing else in order to sing it."
)

SCORE_TOOLTIP = (
    "An edited score. Leave it empty to sing the one inside the plan.\n\n"
    "The 'Edit score...' button opens the score as a piano roll, as sheet music "
    "and as ABC text, and writes the edit here on Apply; 'Reset score' empties the "
    "box again. An empty box, or text that "
    "still matches the plan, sings the model's own score token for token: the same "
    "song 'YuE2 Generate Song' would have produced from the same seed. An edit made "
    "for other words than the plan's is not sung, and the node says so. A score can "
    "also come in through a wire, and is then sung as it arrives.\n\n"
    "A score that names no section -- a bare tune, as 'YuE2 Load MIDI' hands one on "
    "-- has the plan's lyrics laid along it first, as 'YuE2 Generate Song' does, and "
    "the node says which bars each section is sung on."
)

LATENTS_TOOLTIP = (
    "The acoustic latents from 'YuE2 Render Plan' -- the song after it has been "
    "sung but before it has been turned into sound.\n\n"
    "Decoding them again is seconds of work rather than minutes, which is what "
    "makes it worth hearing the same performance through the other decoder."
)

COUNT_TOOLTIP = (
    "How many scores to write, from consecutive seeds starting at 'seed'.\n\n"
    "Each one is a different take on the same words. Pick between them with "
    "'YuE2 Select Plan' and only the one you pick gets sung."
)

INDEX_TOOLTIP = (
    "Which score to take, counting from 0.\n\n"
    "Changing this number does not rewrite the scores: the batch above is already "
    "done, so moving through them costs nothing. An index past the end is clamped "
    "to the last one rather than stopping the run."
)

PLANS_TOOLTIP = "The batch of scores from 'YuE2 Plan Batch'."

RENDER_OPTIONS_TOOLTIP = (
    "Settings for the singing, laid over the ones the score was written with.\n\n"
    "Leave it unconnected and the plan's own settings are used. 'cot' is the one "
    "exception and always comes from the plan: it decides what the model was told "
    "before it wrote the score, so changing it here would sing one score under the "
    "instructions written for another."
)

DECODE_OPTIONS_TOOLTIP = (
    "Settings for the decode. Only 'vae', 'device' and 'keep_model_loaded' change "
    "anything at this stage -- everything else was settled while the song was sung."
)

COT_OFF_REFUSAL = (
    "'cot' is 'off' in the options, which skips the score entirely and goes "
    "straight from the lyrics to audio. There is nothing to plan or to edit.\n\n"
    "Set 'cot' to 'full' or 'melody', or use 'YuE2 Generate Song' instead."
)


def resolve(options) -> dict:
    """DEFAULT_OPTIONS with a connected options node laid over it."""
    settings = dict(DEFAULT_OPTIONS)
    if options:
        settings.update(options)
    return settings


def words(style, lyrics, unique_id):
    """The two prompts, trimmed, with the one refusal they share."""
    style = (style or "").strip()
    lyrics = (lyrics or "").strip()
    if not style and not lyrics:
        refuse(unique_id, "Give a style, lyrics, or both. With neither there is "
                          "nothing to write a song from.")
    return style, lyrics


@contextlib.contextmanager
def session(settings, unique_id, progress):
    """The weights on disk and the model on the card, unloaded on the way out.

    Every node in this pack that touches YuE2 needs the same steps first, in the
    same order, with the same three ways of failing. Written once here so that a
    fix reaches all of them rather than most of them.
    """
    try:
        settings["device"] = devices.validate(settings["device"])
    except (ValueError, RuntimeError) as error:
        refuse(unique_id, str(error))

    from . import download, loader

    try:
        files = download.ensure(settings, progress)
    except (FileNotFoundError, download.DownloadError) as error:
        refuse(unique_id, str(error))

    try:
        yield loader.acquire(files, settings["device"], settings["vae"], progress,
                             settings.get("offload", "auto"))
    except InterruptedError:
        translate_interrupt()
        raise
    except ValueError as error:
        refuse(unique_id, str(error))
    finally:
        if not settings["keep_model_loaded"]:
            loader.unload()


def _made(style, lyrics, seed, settings, score, ids, timing) -> dict:
    """The plan, which is a plain dictionary on purpose.

    It has to survive being cached by ComfyUI between runs, saved inside a
    workflow, and carried through nodes that know nothing about it. Text,
    numbers and settings do all three. A live object holding a tensor does none.
    """
    return {"style": style, "lyrics": lyrics, "seed": normalize_seed(seed),
            "settings": dict(settings), "score": score, "ids": [int(i) for i in ids],
            "timing": dict(timing)}


def _chosen(plan, edited):
    """Which score to sing and in which form: ``(ids, text)``, ids or None.

    An untouched box means the plan's own score, handed on as the exact ids the
    model produced. Text that differs is handed on as text and encoded by the
    stage, because it is now a person's score rather than the model's and those
    ids no longer describe it.

    Both sides are compared trimmed, so a paste that gained a trailing newline
    is not mistaken for an edit, and the edited text is passed on trimmed for
    the same reason: leading and trailing blank lines become tokens in the
    prompt while carrying no music. The mark of the words an edit was made for
    comes off first, for the same reason again; see edits.py.
    """
    text = edits.read(edited).score
    original = (plan.get("score") or "").strip()
    if not text or text == original:
        return list(plan.get("ids") or []), plan.get("score") or ""
    return None, text


def _settings(plan, options, unique_id) -> dict:
    """The plan's settings with this node's options laid over them.

    'cot' never comes from the override. It is part of the prompt the score was
    written under, not a knob on the performance, and honouring a change here
    would sing a melody-only score under the instructions for a full one.
    """
    settings = dict(DEFAULT_OPTIONS)
    settings.update(plan.get("settings") or {})
    written = settings["cot"]
    if options:
        settings.update(options)
    if settings["cot"] != written:
        announce(unique_id, [("warn",
                              "This score was written with cot='{}' and is sung that "
                              "way. The options node connected here says '{}', which "
                              "is ignored for the score that already exists."
                              .format(written, settings["cot"]))])
        settings["cot"] = written
    return settings


def _labelled(plans) -> str:
    """Every score in a batch as one readable block, for a text preview node."""
    blocks = []
    for index, plan in enumerate(plans):
        blocks.append("=== {} | seed {} ===\n{}".format(
            index, plan.get("seed"), (plan.get("score") or "").strip()))
    return "\n\n".join(blocks)


def _band(index, total):
    """The slice of one progress bar that belongs to take *index* of *total*."""
    low = 100.0 * index / total
    return (low, 100.0 * (index + 1) / total,
            "Writing score {} of {}".format(index + 1, total))


class YuE2Plan:
    """Style and lyrics in, a readable score out, and no audio at all."""

    DESCRIPTION = (
        "Writes the ABC score for a song and stops there, which is a small part of "
        "the work a whole song takes.\n\n"
        "Feed the plan to 'YuE2 Render Plan' to hear it, or edit the score first "
        "with 'Edit score...' there: the render node sings whatever you give it. "
        "Hunting for a seed at this "
        "stage is much cheaper than hunting for one a whole song at a time."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": ("STRING", {"multiline": True, "default": DEFAULT_STYLE,
                                     "tooltip": STYLE_TOOLTIP}),
                "lyrics": ("STRING", {"multiline": True, "default": DEFAULT_LYRICS,
                                      "tooltip": LYRICS_TOOLTIP}),
                "seed": ("INT", {"default": 831001, "min": 0, "max": SEED_MAX,
                                 "control_after_generate": True,
                                 "tooltip": SEED_TOOLTIP}),
            },
            "optional": {"options": (OPTIONS_TYPE,)},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (PLAN_TYPE, "STRING")
    RETURN_NAMES = ("plan", "score_abc")
    FUNCTION = "plan"
    CATEGORY = ADVANCED_CATEGORY
    OUTPUT_NODE = True

    def plan(self, style, lyrics, seed, options=None, unique_id=None):
        from . import generate

        progress = NodeProgress(unique_id)
        style, lyrics = words(style, lyrics, unique_id)
        settings = resolve(options)
        if settings["cot"] == "off":
            refuse(unique_id, COT_OFF_REFUSAL)

        with session(settings, unique_id, progress) as models:
            score, ids, timing = generate.write_score(
                models, style, lyrics, seed, settings, progress, interrupted,
                stages=generate.alone(generate.Stages.ABC))

        log.info("[yue2_comfy] score of %d tokens in %.1f s | seed %s",
                 len(ids), timing.get("abc", {}).get("seconds", 0.0), seed)
        progress.finish("{} tokens of score".format(len(ids)))
        return {"ui": {SCORE_UI: [score],
                       WORDS_UI: [edits.mark(style, lyrics, settings["cot"])],
                       AUTO_SECONDS_UI: [auto_seconds(lyrics)]},
                "result": (_made(style, lyrics, seed, settings, score, ids, timing), score)}


class YuE2PlanBatch:
    """Several scores from consecutive seeds, so one can be chosen by ear."""

    DESCRIPTION = (
        "Writes several scores for the same words, one per consecutive seed, and "
        "hands them on as a batch.\n\n"
        "Pick between them with 'YuE2 Select Plan'. Changing the index there does "
        "not rewrite anything, so trying the next take costs only the singing."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = YuE2Plan.INPUT_TYPES()
        spec["required"]["count"] = ("INT", {"default": 4, "min": 2, "max": MAX_TAKES,
                                             "step": 1, "tooltip": COUNT_TOOLTIP})
        return spec

    RETURN_TYPES = (PLANS_TYPE, "STRING")
    RETURN_NAMES = ("plans", "scores")
    FUNCTION = "batch"
    CATEGORY = ADVANCED_CATEGORY

    def batch(self, style, lyrics, seed, count, options=None, unique_id=None):
        from . import generate

        progress = NodeProgress(unique_id)
        style, lyrics = words(style, lyrics, unique_id)
        settings = resolve(options)
        if settings["cot"] == "off":
            refuse(unique_id, COT_OFF_REFUSAL)

        count = max(2, min(int(count), MAX_TAKES))
        plans = []
        with session(settings, unique_id, progress) as models:
            for index in range(count):
                take = normalize_seed(int(seed) + index)
                score, ids, timing = generate.write_score(
                    models, style, lyrics, take, settings, progress, interrupted,
                    stages=(_band(index, count),))
                plans.append(_made(style, lyrics, take, settings, score, ids, timing))

        log.info("[yue2_comfy] %d scores from seeds %s..%s", len(plans),
                 plans[0]["seed"], plans[-1]["seed"])
        progress.finish("{} scores".format(len(plans)))
        return (plans, _labelled(plans))


class YuE2SelectPlan:
    """One score out of a batch, chosen by number."""

    DESCRIPTION = (
        "Takes one score out of a 'YuE2 Plan Batch' by index, counting from 0.\n\n"
        "This is a separate node so that changing the index does not rewrite the "
        "batch: the scores above it stay as they are and only what follows re-runs."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plans": (PLANS_TYPE, {"tooltip": PLANS_TOOLTIP}),
                "index": ("INT", {"default": 0, "min": 0, "max": MAX_TAKES - 1,
                                  "step": 1, "tooltip": INDEX_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (PLAN_TYPE, "STRING")
    RETURN_NAMES = ("plan", "score_abc")
    FUNCTION = "select"
    CATEGORY = ADVANCED_CATEGORY
    OUTPUT_NODE = True

    def select(self, plans, index, unique_id=None):
        if not plans:
            refuse(unique_id, "The batch is empty, so there is no score to take "
                              "out of it.")
        wanted = int(index)
        taken = max(0, min(wanted, len(plans) - 1))
        if taken != wanted:
            announce(unique_id, [("warn",
                                  "There are {} scores in the batch, so index {} "
                                  "does not exist; taking {} instead."
                                  .format(len(plans), wanted, taken))])
        plan = plans[taken]
        score = plan.get("score") or ""
        cot = (plan.get("settings") or {}).get("cot", DEFAULT_OPTIONS["cot"])
        for_words = edits.mark(plan.get("style"), plan.get("lyrics"), cot)
        return {"ui": {SCORE_UI: [score], WORDS_UI: [for_words],
                       AUTO_SECONDS_UI: [auto_seconds(plan.get("lyrics"))]},
                "result": (plan, score)}


class YuE2RenderPlan:
    """A score, edited or not, sung into a finished song."""

    DESCRIPTION = (
        "Sings a score from 'YuE2 Plan' and decodes it to 48 kHz stereo audio.\n\n"
        "Left unedited, the model's own score is sung, which gives exactly the song "
        "'YuE2 Generate Song' makes from the same seed. 'Edit score...' opens the "
        "score in a piano roll to change notes and chords first; only the bars you "
        "change are written again, and the edit is what gets sung."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {"tooltip": PLAN_TOOLTIP}),
            },
            "optional": {
                "score_abc": ("STRING", {"multiline": True, "default": "",
                                         "tooltip": SCORE_TOOLTIP}),
                "options": (OPTIONS_TYPE, {"tooltip": RENDER_OPTIONS_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO", LATENTS_TYPE)
    RETURN_NAMES = ("audio", "latents")
    FUNCTION = "render"
    CATEGORY = ADVANCED_CATEGORY

    def render(self, plan, score_abc="", options=None, unique_id=None):
        from . import generate

        if not plan:
            refuse(unique_id, "No plan is connected. Add a 'YuE2 Plan' node and "
                              "join its 'plan' output to this input.")
        progress = NodeProgress(unique_id)
        settings = _settings(plan, options, unique_id)
        problem = edits.mismatch(edits.read(score_abc), plan.get("style"),
                                 plan.get("lyrics"), settings["cot"], RENDER_INSTEAD)
        if problem:
            announce(unique_id, [("warn", problem)])
        given = edits.read(score_abc).score
        if given and not problem and settings["cot"] == "full" and edits.chordless(given):
            announce(unique_id, [("warn", edits.CHORDLESS)])
        ids, score = _chosen(plan, "" if problem else score_abc)
        tune_seconds = None
        if ids is None and score:
            laid = phrasing.lay(score, plan["lyrics"])
            if laid is not None:
                score, tune_seconds = laid.score, laid.seconds
                announce(unique_id, laid.notices)
        semitones = int(settings.get("transpose") or 0)
        if semitones:
            try:
                score = generate.moved(score, semitones, settings["cot"])
            except ValueError as error:
                refuse(unique_id, str(error))
            ids = None
        stages = generate.alone(generate.Stages.SEMANTIC, generate.Stages.ACOUSTIC,
                                generate.Stages.DECODE)
        voice = bool(settings.get("vocals_only"))
        song_progress = Band(progress, 0.0, 1.0 - VOICE_SHARE) if voice else progress

        separator = separator_weights(settings, unique_id, song_progress) if voice else None
        with session(settings, unique_id, song_progress) as models:
            latents, timing = generate.sing(
                models, plan["style"], plan["lyrics"], plan["seed"], settings,
                abc_ids=ids, abc=score, progress=song_progress, cancelled=interrupted,
                stages=stages[:2], tune_seconds=tune_seconds)
            waveform, spent = generate.decode(models, latents, song_progress, interrupted,
                                              stages=stages[2:])
        del models
        timing.update(spent)
        if voice:
            waveform = voice_of({"waveform": waveform, "sample_rate": SAMPLE_RATE}, settings, unique_id,
                                Band(progress, 1.0 - VOICE_SHARE, 1.0), path=separator)["waveform"]

        log.info("[yue2_comfy] %.1f s of audio from a %s score | seed %s",
                 timing["seconds_of_audio"],
                 "moved" if semitones else "given" if ids is None else "written",
                 plan["seed"])
        progress.finish("{:.0f} seconds of audio".format(timing["seconds_of_audio"]))
        return ({"waveform": waveform, "sample_rate": SAMPLE_RATE},
                {"latents": latents.cpu(), "settings": dict(settings),
                 "seed": plan["seed"], "timing": dict(timing)})


class YuE2DecodeLatents:
    """The same performance, turned into sound again."""

    DESCRIPTION = (
        "Decodes the latents from 'YuE2 Render Plan' into audio without singing "
        "the song again.\n\n"
        "What it is for: hearing one performance through the other decoder. "
        "'standard' and 'legacy' are different weights, and the same latents "
        "through each do not sound the same."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": (LATENTS_TYPE, {"tooltip": LATENTS_TOOLTIP}),
            },
            "optional": {
                "options": (OPTIONS_TYPE, {"tooltip": DECODE_OPTIONS_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = ADVANCED_CATEGORY

    def decode(self, latents, options=None, unique_id=None):
        from . import generate

        if not latents or latents.get("latents") is None:
            refuse(unique_id, "No latents are connected. Add a 'YuE2 Render Plan' "
                              "node and join its 'latents' output to this input.")
        progress = NodeProgress(unique_id)
        settings = dict(DEFAULT_OPTIONS)
        settings.update(latents.get("settings") or {})
        if options:
            settings.update(options)

        voice = bool(settings.get("vocals_only"))
        song_progress = Band(progress, 0.0, 1.0 - VOICE_SHARE) if voice else progress
        separator = separator_weights(settings, unique_id, song_progress) if voice else None
        with session(settings, unique_id, song_progress) as models:
            tensor = latents["latents"]
            moved = tensor.to(models.device) if hasattr(tensor, "to") else tensor
            waveform, timing = generate.decode(models, moved, song_progress, interrupted,
                                               stages=generate.alone(
                                                   generate.Stages.DECODE))
        del models, moved
        if voice:
            waveform = voice_of({"waveform": waveform, "sample_rate": SAMPLE_RATE}, settings, unique_id,
                                Band(progress, 1.0 - VOICE_SHARE, 1.0), path=separator)["waveform"]

        log.info("[yue2_comfy] decoded %.1f s of audio with the %s decoder",
                 timing["seconds_of_audio"], settings["vae"])
        progress.finish("{:.0f} seconds of audio".format(timing["seconds_of_audio"]))
        return ({"waveform": waveform, "sample_rate": SAMPLE_RATE},)


STAGED_CLASSES = {
    "YuE2Plan": YuE2Plan,
    "YuE2PlanBatch": YuE2PlanBatch,
    "YuE2SelectPlan": YuE2SelectPlan,
    "YuE2RenderPlan": YuE2RenderPlan,
    "YuE2DecodeLatents": YuE2DecodeLatents,
}

STAGED_NAMES = {
    "YuE2Plan": "YuE2 Plan",
    "YuE2PlanBatch": "YuE2 Plan Batch",
    "YuE2SelectPlan": "YuE2 Select Plan",
    "YuE2RenderPlan": "YuE2 Render Plan",
    "YuE2DecodeLatents": "YuE2 Decode Latents",
}
