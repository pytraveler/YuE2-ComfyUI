"""YuE2 Edit Track: a song this pack sang, with part of it sung again or taken out.

The node takes the song's audio, finds the song behind it in the pack's memory
(see ``songs``) and works through a list of edits, each one made on the song
the ones before it left. A retake sings a stretch again, as many takes as
asked for, and keeps the one whose join the model likes best unless the window
says otherwise; a cut takes a stretch out and draws the two sides together.
The sound that comes out is the song with those edits in it, and it is
remembered like any other, so it can be saved, loaded and edited again.

What it costs is paid once. The takes of every edit stay in this session's
memory under a name made of the song, the edits before it and what this one
does, so adding an edit at the end sings that edit alone, switching to another
take of the last one costs nothing, and undoing an edit costs nothing either.
That memory is only this session's. A workflow saved with a list still asks
for the takes it once compared, so reopening it in a fresh ComfyUI sings the
take the list kept and that one alone; the others are offered as seeds, since
a seed sings the same take whenever it is asked.

The window that writes the list is ``web/js/yue2_track.js``; the list's shape
lives in ``inpaint.track``, which the tests read from both sides. Module scope
stays light for the same reason nodes.py does: no torch here.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import time

from . import notation, songs
from .constants import CATEGORY, DEFAULT_OPTIONS, FRAME_SECONDS, OPTIONS_TYPE
from .edits import EDIT_TRACK_UI
from .inpaint import grid, track
from .progress import (Band, NodeProgress, announce, interrupted, refuse,
                       translate_interrupt)

log = logging.getLogger(__name__)

ORIGIN = "YuE2 Edit Track"

PEAKS = 1200
"""How many peaks the wave is drawn from: one for every pixel of a window a thousand wide, and some over."""

TAKE_FILES = """A take is written whole, not as an excerpt around its edit.

The window draws the song a take makes and plays it from anywhere on the
track, which is how a take that runs a minute is listened to at all. That
costs the temp folder a WAV per take -- 21 MB a minute of stereo -- until
ComfyUI clears it at startup, and buys the ear the whole song around the
join."""

GRID_SHARE = 0.12
"""The share of the bar that laying the score over the song takes: the separator and a tenth of a second of arithmetic."""

CACHE_BYTES = 1 << 30
"""How much of the session's memory the takes already sung may hold: 1 GiB.

A take is a whole song as sound, 21 MB a minute in stereo at 48 kHz, so four
takes of an edit of a four-minute song are 340 MB. Past the budget the edit
used longest ago goes first, and is sung again if it is asked for again."""

JOIN_SLACK = 0.75
"""How far below the song's own join a take may score before it is worth saying so.

The stand's one measured failure -- a retake whose section had lost its pickup
to a cut before it -- scored -4.7 to -4.9 where the song itself scored -3.4,
and the takes that sounded right scored around the song's own. Half that gap
is the line: enough to catch that kind of miss, loose enough not to nag."""

RUN_KEYS = ("device", "offload", "low_vram", "download", "keep_model_loaded", "vae",
            "quantization", "attention_backend")
"""What the run decides for itself; everything else the song decides.

'cot', the sampling and 'cfg_scale' are part of how the song was sung, not
knobs on this node: an edit sampled another way would not sound like the song
it goes into. They come out of the song's memory, and so do the adapters it
was sung with, which is what makes a retake of a song sung through a LoRA come
back in the same voice."""

SUNG_KEYS = ("cot", "cfg_scale", "ode_steps", "temperature", "top_p", "top_k",
             "repetition_penalty")
"""The settings an edit sings by, and so the ones worth a word when an options node asks otherwise.

The rest of an options node -- the score's sampling, 'max_seconds', 'transpose'
-- has no part in an edit at all, and a notice about it would only puzzle."""

UNKNOWN = (
    "This pack does not remember singing this audio, so there is nothing to edit.\n\n"
    "Edit Track works on songs this install has sung: connect it to 'YuE2 Generate Song', "
    "'YuE2 Render Plan' or 'YuE2 Decode Latents', or load a song that was saved with 'Save "
    "Audio' as FLAC or WAV. A lossy file -- MP3, Opus, AAC -- changes nearly every sample, so "
    "the song behind it cannot be found. Editing a recording this pack never sang is a "
    "milestone of its own and is not here yet."
)

VOICE_ONLY = (
    "This song was made with 'vocals_only', so what you hear is its voice while the song "
    "remembered behind it is the whole mix. An edit would lay mixed sound into a voice-only "
    "track.\n\n"
    "Edit the song first and take the voice afterwards, with a 'YuE2 Vocals Only' node on this "
    "node's output."
)

NO_AUDIO = "No audio is connected. Join the song you want to edit to 'audio'."

AUDIO_TOOLTIP = (
    "A song this pack has sung: the output of 'YuE2 Generate Song', 'YuE2 Render Plan' or "
    "'YuE2 Decode Latents', or a FLAC or WAV of one loaded with 'Load Audio'.\n\n"
    "An MP3 or another lossy file changes nearly every sample, and the song behind it is not "
    "found."
)

TAKES_TOOLTIP = (
    "How many times a retake sings the same stretch, so there is something to choose between. "
    "The one whose join the model likes best is kept; the track window plays them all and lets "
    "you keep another, which costs nothing -- they are all already sung.\n\n"
    "They are sung once and kept in this session's memory. A workflow opened in a fresh "
    "ComfyUI sings only the take its list kept, however many it once compared; the window "
    "shows the others and sings one on request, a seed always giving back the same take.\n\n"
    "A cut ignores this: a cut is the same cut however often it is made."
)

EDITS_TOOLTIP = (
    "The edits to make, as a JSON list, written by the track window. Empty means no edits: the "
    "node hands the song on as it is and draws the track.\n\n"
    "Each edit is {'op': 'retake' or 'cut'} with either 'bars': [first, stop] counting from 0, "
    "the second bar not included, or 'seconds': [from, to] for a song sung with cot 'off', "
    "which has no score. A retake also takes a 'seed', how many 'takes' to sing, and which "
    "'take' to keep."
)

OPTIONS_TOOLTIP = (
    "Optional. Only what this run decides is read from it -- 'device', 'offload', 'low_vram', "
    "'vae', 'quantization', 'attention_backend', 'download' and 'keep_model_loaded'. How the "
    "song was sung, 'cot' and the sampling among it, comes from the song itself."
)

DESCRIPTION = (
    "Sings part of a song again, or takes part of it out, and hands back the whole song with "
    "the change in it. Everything outside the edit is the sound that was there before, sample "
    "for sample.\n\n"
    "Open 'Edit track...' to see the song's bars and sections, select a stretch, and press "
    "Retake or Cut. It works on songs this pack has sung: wire it after a song node, or load a "
    "FLAC saved from one.\n\n"
    "A retake sings two takes by default and keeps the one that joins the old song best. A cut "
    "takes whole bars, moves both ends back to the same place in the singing so that a pickup "
    "is not left behind, and takes the words of any section it empties out of the lyrics.\n\n"
    "Finding where the score sits in the song listens to its voice, so the first edit of a song "
    "downloads the separator Vocals Only uses (0.85 GB, MIT) if it is not there yet."
)

class _Refused(Exception):
    """A refusal raised inside the model session, said on the node once it is closed.

    ``staged.session`` announces every ValueError that reaches it, so a refusal
    made while it is open would be said twice; this is not a ValueError.
    """


_GRIDS = collections.OrderedDict()
"""The grid measured for each song, by its key: measuring it separates the voice, which is a download and a minute of card."""

_GRID_KEEP = 8


def is_loaded() -> bool:
    """Whether anything is held, for the report Unload Models writes."""
    return bool(RESULTS.size() or _GRIDS)


def unload() -> None:
    """Let go of the takes and the grids, when someone asks for the memory back.

    They are not a model, but they are the largest thing this pack holds that
    a person cannot otherwise let go of: four takes of a four-minute song are
    a third of a gigabyte. Free memory means free memory, and an edit dropped
    here is sung again if it is asked for again.
    """
    RESULTS.clear()
    _GRIDS.clear()


def _temp_folder():
    """ComfyUI's temp folder, where the takes and the preview go; None outside ComfyUI."""
    from . import paths

    folder_paths = paths._folder_paths()
    if folder_paths is None:
        return None
    return folder_paths.get_temp_directory()


def _pcm(waveform):
    """A waveform as interleaved little-endian 16-bit samples, the way ``songs.key`` rounds them."""
    import torch

    samples = waveform.detach()
    if samples.dim() == 3:
        samples = samples[0]
    samples = samples.to(device="cpu", dtype=torch.float32).transpose(0, 1).contiguous()
    pcm = (samples.reshape(-1) * 32768.0).round_().clamp_(-32768, 32767)
    return pcm.to(torch.int16).numpy().astype("<i2", copy=False).tobytes()


def _write_wave(waveform, rate: int, name: str):
    """One stretch of sound in ComfyUI's temp folder, served back to the window by ``routes.send_sound``."""
    import wave

    folder = _temp_folder()
    if folder is None:
        return None
    place = os.path.join(folder, "yue2_edit")
    path = os.path.join(place, name)
    entry = {"filename": name, "subfolder": "yue2_edit", "type": "temp"}
    channels = waveform.shape[-2]
    try:
        os.makedirs(place, exist_ok=True)
        with wave.open(path, "wb") as handle:
            handle.setnchannels(int(channels))
            handle.setsampwidth(2)
            handle.setframerate(int(rate))
            handle.writeframes(_pcm(waveform))
    except Exception:
        log.warning("[yue2_comfy.edit_track] this take could not be written for listening",
                    exc_info=True)
        return None
    return entry


def _wave(waveform, count: int = PEAKS) -> dict:
    """The song drawn as at most ``count`` slices: the loudest sample of each, and how loud it is.

    The peaks alone draw a mastered song as a solid block, because nearly
    every tenth of a second of one touches the ceiling. The root mean square
    of the slice, over every channel, is the body of the sound, and drawn
    inside the peaks it is what makes a verse look different from a chorus.
    A sample that is not a number draws as silence rather than breaking the
    JSON the browser reads.
    """
    samples = waveform.detach().float()
    if samples.dim() == 3:
        samples = samples[0]
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)
    samples = samples.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    total = int(samples.shape[-1])
    if total < 1 or samples.shape[0] < 1:
        return {"peaks": [], "rms": []}
    step = max(1, -(-total // max(1, count)))
    usable = (total // step) * step
    if usable < 1:
        return {"peaks": [], "rms": []}
    loudest = samples.abs().amax(dim=0)[:usable].reshape(-1, step).amax(dim=1)
    body = samples.pow(2).mean(dim=0)[:usable].reshape(-1, step).mean(dim=1).sqrt()
    return {"peaks": [round(float(value), 4) for value in loudest.tolist()],
            "rms": [round(float(value), 4) for value in body.tolist()]}


def _settings(song, options, unique_id) -> dict:
    """The song's own settings, with only what this run decides laid over them."""
    settings = dict(DEFAULT_OPTIONS)
    settings.update(song.settings or {})
    if not options:
        return settings
    ignored = sorted(name for name, value in dict(options).items()
                     if name in SUNG_KEYS and settings.get(name) != value)
    for name in RUN_KEYS:
        if name in options:
            settings[name] = options[name]
    if ignored:
        announce(unique_id, [("notice", "The options node asks for {}, which the song itself "
                                        "decides: an edit is sung the way the song was."
                                        .format(", ".join(ignored)))])
    return settings


def _held_bytes(entry) -> int:
    return sum(int(take.waveform.numel()) * 4 for take in entry["takes"].values())


class Results:
    """The takes of every edit sung in this session, the ones used longest ago dropped first."""

    def __init__(self, budget: int = CACHE_BYTES):
        self._held = collections.OrderedDict()
        self._sizes = {}
        self._budget = int(budget)
        self._bytes = 0

    def get(self, name: str):
        found = self._held.get(name)
        if found is not None:
            self._held.move_to_end(name)
        return found

    def put(self, name: str, entry) -> None:
        """Hold ``entry`` under ``name``, measured now: an entry grown in place is measured again here."""
        if name in self._held:
            self._held.pop(name)
            self._bytes -= self._sizes.pop(name)
        self._held[name] = entry
        self._sizes[name] = _held_bytes(entry)
        self._bytes += self._sizes[name]
        while self._bytes > self._budget and len(self._held) > 1:
            oldest, _gone = self._held.popitem(last=False)
            self._bytes -= self._sizes.pop(oldest)

    def clear(self) -> None:
        self._held.clear()
        self._sizes.clear()
        self._bytes = 0

    def size(self) -> int:
        return self._bytes


RESULTS = Results()


def _grid_of(name: str, song, waveform, rate: int, settings, unique_id, progress):
    """The score laid over the song, measured once per song and kept.

    The song's start is placed by its separated voice, which is what put all
    seven of the stand's songs right where the mix's chroma alone missed one by
    fourteen seconds. That costs a download the first time and a few seconds of
    card after that, so the answer is kept for the session.
    """
    found = _GRIDS.get(name)
    if found is not None:
        _GRIDS.move_to_end(name)
        return found

    from . import notation
    from .inpaint import grid
    from .vocals_only import separator_weights, voice_of

    sheet = notation.read(song.score)
    path = separator_weights(settings, unique_id, progress)
    voice = voice_of({"waveform": waveform, "sample_rate": rate}, settings, unique_id, progress,
                     path=path)["waveform"]
    began = time.perf_counter()
    clock = grid.measured(sheet, waveform, rate, voice=voice)
    log.info("[yue2_comfy.edit_track] the score sits on this song from %.2f s at %.4f of its tempo, "
             "placed by %s, in %.1f s", clock.offset, clock.rate,
             "the voice" if clock.by_voice else "the mix", time.perf_counter() - began)
    _GRIDS[name] = clock
    while len(_GRIDS) > _GRID_KEEP:
        _GRIDS.popitem(last=False)
    return clock


def _flagged(take) -> bool:
    """Whether a take's join is enough worse than the song's own there to be worth saying so."""
    return (take.natural is not None and take.join is not None
            and take.join < take.natural - JOIN_SLACK)


def _take_facts(take, index: int, chosen: int, name: str, rate: int, prior, step) -> dict:
    """One take as the window shows it: how it scored, how long it came out, and the song it makes.

    ``seconds`` is the take itself and ``total`` the song with it in, which is
    what the window draws and plays; ``peaks`` and ``rms`` are that song's
    wave, and ``grid`` its own layout of the score, because a take that came
    out longer or shorter than the one kept moves every bar after the edit --
    so switching takes swaps the track, its bars and its sound under the
    cursor without the node running again. ``prior`` is the song's state
    before this edit and ``step`` the edit's plan. See ``TAKE_FILES`` for
    what the files cost.
    """
    from .inpaint import grid, track

    entry = _write_wave(take.waveform, rate, "{}_{}.wav".format(name[:16], take.seed))
    drawn = _wave(take.waveform)
    own = track.after(prior, step, take.count)
    laid = None
    if own.sheet is not None and own.clock is not None:
        laid = grid.layout(own.sheet, own.clock, own.frames)
    return {"seed": int(take.seed), "index": index, "kept": index == chosen, "sung": True,
            "seconds": round(take.count * FRAME_SECONDS, 3),
            "total": round(int(take.waveform.shape[-1]) / float(rate), 3),
            "peaks": drawn["peaks"], "rms": drawn["rms"], "grid": laid,
            "join": None if take.join is None else round(take.join, 4),
            "natural": None if take.join is None or take.natural is None else round(take.natural, 4),
            "ended": bool(take.ended), "flagged": _flagged(take), "audio": entry}


def _take_gap(seed: int, index: int) -> dict:
    """A take the list asks for that this session has not sung, as a row the window can offer.

    It carries its seed and nothing else, because nothing else exists yet: no
    file, no length, no join. Singing it later gives the take that seed always
    gives, so the row is an offer, not a loss.
    """
    return {"seed": int(seed), "index": index, "kept": False, "sung": False,
            "seconds": None, "total": None, "peaks": [], "rms": [], "grid": None,
            "join": None, "natural": None, "ended": False, "flagged": False, "audio": None}


def _remember(waveform, song) -> str:
    """Remember the edited song, so it can be saved, loaded and edited again. Never fatal."""
    try:
        return songs.remember(waveform, song.sample_rate, song)
    except Exception:
        log.warning("[yue2_comfy.edit_track] this edit could not be remembered, so it cannot be "
                    "edited again", exc_info=True)
        return ""


class YuE2EditTrack:
    """A song in, the same song with part of it sung again or taken out."""

    DESCRIPTION = DESCRIPTION
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": AUDIO_TOOLTIP}),
                "takes": ("INT", {"default": 2, "min": 1, "max": track.MAX_TAKES,
                                  "tooltip": TAKES_TOOLTIP}),
            },
            "optional": {
                "edits": ("STRING", {"multiline": True, "default": "", "tooltip": EDITS_TOOLTIP}),
                "options": (OPTIONS_TYPE, {"tooltip": OPTIONS_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "edit"
    CATEGORY = CATEGORY

    def edit(self, audio, takes=2, edits="", options=None, unique_id=None):
        from .inpaint import core
        from .staged import adapters, session

        if not audio or audio.get("waveform") is None:
            refuse(unique_id, NO_AUDIO)
        progress = NodeProgress(unique_id, title=ORIGIN)
        waveform, rate = audio["waveform"], int(audio["sample_rate"])
        try:
            name = songs.key(waveform, rate)
        except ValueError as error:
            refuse(unique_id, str(error))
        song = songs.store().get(name)
        if song is None:
            refuse(unique_id, UNKNOWN)
        if song.settings.get("vocals_only"):
            refuse(unique_id, VOICE_ONLY)
        try:
            wanted = track.read(edits, int(takes))
        except ValueError as error:
            refuse(unique_id, str(error))
        settings = _settings(song, options, unique_id)
        adapters(settings, unique_id)
        sheet = notation.read(song.score) if song.score else None
        if wanted and wanted[0].bars is not None:
            if sheet is None:
                refuse(unique_id, "Edit 1: " + track.NO_SCORE)
            try:
                grid.bars_of(sheet, wanted[0].bars[0], wanted[0].bars[1])
            except ValueError as error:
                refuse(unique_id, "Edit 1: {}".format(error))

        began = time.perf_counter()
        clock = None
        low = 0.0
        if song.score:
            low = 0.0 if name in _GRIDS else GRID_SHARE
            clock = _grid_of(name, song, waveform, rate, settings,
                             unique_id, Band(progress, 0.0, GRID_SHARE) if wanted else progress)
        state = track.opened(song, clock, sheet)

        stack = contextlib.ExitStack()
        models = [None]

        def loaded():
            if models[0] is None:
                models[0] = stack.enter_context(session(settings, unique_id, progress))
            return models[0]

        history, notices, picks, shown = [], [], [], None
        current, sound = song, waveform
        try:
            with stack:
                for index, edit in enumerate(wanted):
                    if interrupted():
                        raise InterruptedError("Cancelled between edits")
                    band = Band(progress, low + (1.0 - low) * index / len(wanted),
                                low + (1.0 - low) * (index + 1) / len(wanted))
                    try:
                        step = track.plan(state, edit)
                    except ValueError as error:
                        raise _Refused("Edit {}: {}".format(index + 1, error))
                    notices.extend(step.notices)
                    made = track.name(name, history, edit)
                    entry = self._sung(loaded, current, sound, state, step, edit, made,
                                       settings, band)
                    seeds = (0,) if step.kind == "cut" else edit.seeds()
                    ordered = [entry["takes"].get(seed) for seed in seeds]
                    pick = edit.take if edit.take is not None else core.best(ordered)
                    kept = ordered[pick]
                    history.append((edit, kept.seed))
                    picks.append(pick)
                    prior = state
                    state = track.after(state, step, kept.count)
                    current, sound = kept.song, kept.waveform
                    shown = (step, ordered, pick, made, prior, seeds)
        except InterruptedError:
            translate_interrupt()
            raise
        except _Refused as stop:
            refuse(unique_id, str(stop))

        keyed = name if not wanted else _remember(sound, current)
        payload = self._payload(name, keyed, sound, rate, state, wanted, picks, shown)
        if shown is not None and _flagged(shown[1][shown[2]]):
            kept = shown[1][shown[2]]
            notices.append(("warn", "The join of the take kept scores {:.2f} where the song's own "
                                    "join there scores {:.2f}, so it may be heard. Ask for more "
                                    "takes, or open the selection wider.".format(
                                        kept.join, kept.natural)))
        announce(unique_id, notices)
        log.info("[yue2_comfy.edit_track] %d edit%s on %.1f s of song, %.1f s out, in %.1f s",
                 len(wanted), "" if len(wanted) == 1 else "s", song.frames * FRAME_SECONDS,
                 state.frames * FRAME_SECONDS, time.perf_counter() - began)
        progress.finish("{:.0f} seconds of audio, {} edit{}".format(
            state.frames * FRAME_SECONDS, len(wanted), "" if len(wanted) == 1 else "s"))
        out = {"waveform": sound, "sample_rate": rate}
        ui = {EDIT_TRACK_UI: [payload]}
        preview = _write_wave(sound, rate, "song_{}.wav".format(keyed[:16])) if keyed else None
        if preview is not None:
            ui["audio"] = [preview]
        return {"ui": ui, "result": (out,)}

    def _sung(self, loaded, song, waveform, state, step, edit, name, settings, band):
        """The takes of one edit: the ones already sung under this name, and the ones still missing.

        Only the takes the result needs are sung. An edit that already says
        which take was kept needs that one: the others are there to compare,
        and comparing them is over. This is what a workflow saved with a
        chosen take costs when it is opened again -- the list still asks for
        the four takes it once compared, while the takes themselves live in
        this session's memory and a restart empties it. Their seeds are
        counted on from the edit's own, so any of them can be sung later and
        comes out the same take; the window offers them. An edit with nothing
        kept yet sings all of them, because the pick is made among them.
        """
        from .inpaint import core, ops

        entry = RESULTS.get(name) or {"takes": {}, "natural": None}
        seeds = (0,) if step.kind == "cut" else edit.seeds()
        if step.kind != "cut" and edit.take is not None:
            seeds = (seeds[edit.take],)
        missing = [seed for seed in seeds if seed not in entry["takes"]]
        if not missing:
            band.ratio(1.0)
            return entry
        try:
            if step.kind == "cut":
                region = ops.cut(step.start, step.stop, state.frames)
                entry["takes"][0] = core.cut(loaded(), song, waveform, region, step.lyrics,
                                             step.score, settings, band, interrupted)
            else:
                region = ops.retake(step.start, step.stop, state.frames, len(song.prefix))
                fresh = core.retakes(loaded(), song, waveform, region, missing, settings, band,
                                     interrupted, natural=entry["natural"] is None)
                for take in fresh:
                    entry["takes"][take.seed] = take
                if fresh and fresh[0].natural is not None:
                    entry["natural"] = fresh[0].natural
                for take in entry["takes"].values():
                    take.natural = entry["natural"]
        except ValueError as error:
            raise _Refused(str(error))
        RESULTS.put(name, entry)
        return entry

    def _payload(self, name: str, keyed: str, sound, rate: int, state, wanted, picks,
                 shown) -> dict:
        """Everything the track window draws.

        ``song`` is the key the result is remembered under, empty when it could
        not be remembered, and ``was`` the key of the song that came in.
        ``seconds`` is the length of the result; ``peaks`` and ``rms`` its wave,
        at most ``PEAKS`` values from 0 to 1 each; ``grid`` is ``grid.layout`` of the score on
        the result, None for a song without one; ``lyrics`` the words it sings now,
        which the window shows beside the track and a cut takes sections out
        of; ``edits`` the list as it was read, with the take kept written into
        every retake. After at least one edit there are also ``kind``, ``at``
        (the seconds the last edit took in hand, on the song as it was before
        it), ``dropped`` (the section tags a cut took out), ``chosen`` and
        ``takes``: an entry a take, with seed, index, kept, sung, seconds,
        total, peaks, rms, grid (that take's own layout, since its length moves
        the bars after the edit), join, natural, ended, flagged and ``audio``,
        the temp file of the whole song that take makes, None when it could not
        be written. A take the list asks for that this session has not sung is
        ``sung`` false and empty otherwise, see ``_take_gap``.
        """
        import dataclasses

        from .inpaint import grid

        drawn = _wave(sound)
        payload = {"song": keyed, "was": name,
                   "seconds": round(state.frames * FRAME_SECONDS, 3), "sample_rate": rate,
                   "peaks": drawn["peaks"], "rms": drawn["rms"], "grid": None,
                   "lyrics": state.lyrics, "takes": [], "chosen": None,
                   "edits": track.written(
                       [dataclasses.replace(edit, take=pick)
                        for edit, pick in zip(wanted, picks)])}
        if state.sheet is not None and state.clock is not None:
            payload["grid"] = grid.layout(state.sheet, state.clock, state.frames)
        if shown is None:
            return payload
        step, ordered, pick, made, prior, seeds = shown
        payload["chosen"] = pick
        payload["kind"] = step.kind
        payload["at"] = [round(step.start * FRAME_SECONDS, 3), round(step.stop * FRAME_SECONDS, 3)]
        payload["dropped"] = list(step.dropped)
        payload["takes"] = [
            _take_facts(take, index, pick, made, rate, prior, step) if take is not None
            else _take_gap(seeds[index], index)
            for index, take in enumerate(ordered)]
        return payload


EDIT_CLASSES = {"YuE2EditTrack": YuE2EditTrack}
EDIT_NAMES = {"YuE2EditTrack": ORIGIN}
