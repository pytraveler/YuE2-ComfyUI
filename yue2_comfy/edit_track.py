"""YuE2 Edit Track: a song this pack sang, with part of it sung again or taken out.

The node takes the song's audio, finds the song behind it in the pack's memory
(see ``songs``) and works through a list of edits, each one made on the song
the ones before it left. A retake sings a stretch again, as many takes as
asked for, and keeps the one whose join the model likes best unless the window
says otherwise -- or, with the speech models already on the machine, the one
heard singing the most of the words there; a change of words does the same
with other words, and keeps the take heard singing the most of them; a change
of notes sings again, under the score the score editor left, only the bars
whose notes it changed, and keeps a take as a retake does; a cut takes a
stretch out and draws the two sides together; and a song can go on past its
last words, the model writing the score of the new part and the song's new
ending itself, the take heard singing the most of the new words kept. The sound that comes out is the song with those edits
in it, and it is remembered like any other, so it can be saved, loaded and
edited again. When the word aligner is on the machine, the node also finds
when each line of the words is sung, so the window lights the words where the
song sings them rather than where its bars suggest.

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
import io
import json
import logging
import math
import os
import time

from . import notation, songs
from .constants import CATEGORY, DEFAULT_OPTIONS, FRAME_SECONDS, OPTIONS_TYPE
from .edits import EDIT_TRACK_UI
from .inpaint import grid, ops, track
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

DECODE_SHARE = 0.08
"""The share of the bar that turning a remembered song back into sound takes.

Measured 2026-09-19 on a 5090: 1.1 to 1.7 seconds for a four-minute song,
against minutes to sing one, so it is a sliver of any run that also edits."""

GRID_SHARE = 0.12
"""The share of the bar that laying the score over the song takes: the separator and a tenth of a second of arithmetic."""

TIMES_SHARE = 0.10
"""The share of one edit's bar that goes on finding when its words are sung.

Only an edit that rewrites a line takes it: the aligner reads the whole song
in a single pass, and the answer is kept beside the song, so the second line
rewritten pays nothing at all."""

HEARD_SHARE = 0.12
"""The share of one edit's bar that goes on hearing its takes: a change of words, or a retake when the machine can hear it.

Measured 2026-09-22 on a 5090: 1.7 s to load Qwen3-ASR, then 0.2-0.4 s a
take for the seven seconds of a line, so it is loading the model that the
share is for."""

LINES_SHARE = 0.06
"""The share of the bar that goes on finding when each line is sung, for the window to light.

Measured 2026-09-22 on a 5090: 1.7 s to load the aligner, then 0.1-0.2 s a
pass over a whole song of one to three minutes, one pass a take."""

HEARD_AROUND = 0.3
"""Seconds of the song heard on either side of the stretch a take sang.

The stretch is cut on frames of 40 ms and a singer starts a word a hair before
its note; the stand heard its takes with this much around them."""

MUMBLED = 0.75
"""The share of its new words a take kept has to be heard singing before the node stops saying so.

On the stand a take that sang the new line was heard singing 0.8-1.0 of it,
while the old line sung where the new one was asked for still scored 0.5-0.6:
lines that are rewritten keep some of their words, and a word they share is
heard either way.

A retake is held to the song as it was as well. Qwen3-ASR missed a tenth to
a quarter of the words of sung Russian that were sung right (measured
2026-09-15), so a retake heard below this share is only worth a word when the
song as it was is heard singing more of them there."""

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
back in the same voice. The track window can still ask for another temperature
or guide for one edit alone, see ``_edited``; that is a choice made about that
stretch, not a setting of the run."""

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

NO_AUDIO = (
    "Nothing is chosen to edit. Join a song to 'audio' with the square beside it ticked, or press "
    "'Saved songs...' on the node and pick one this install has sung."
)

UNJOINED = object()
"""What ``check_lazy_status`` is handed for 'audio' when nothing is joined to it.

ComfyUI leaves an input that has no link out of the call altogether and hands
one that is joined but not worked out yet as None, which is the only way the
two can be told apart: only the second can be asked for."""

GONE = (
    "The song this node was opened on is no longer remembered. It may have been the oldest when "
    "the store filled up, or ComfyUI/user/yue2_comfy/songs may have been cleared.\n\n"
    "Press 'Saved songs...' and pick another, or join the audio to 'audio'."
)

NOT_A_KEY = (
    "The 'song_key' field does not hold a song key. It is written by 'Saved songs...' on the "
    "node and is not meant to be typed; clear it and pick a song again."
)

SONG_TOOLTIP = (
    "Which remembered song to edit when nothing is joined to 'audio'. Written by the 'Saved "
    "songs...' button on the node, which shows what this install has sung and what each one "
    "is.\n\nThe song comes back from its latents, the last stage of singing it and nothing more: "
    "about a second and a half for four minutes, against minutes to sing it again. The sound it "
    "makes is the same performance but not the same file to the bit, so the song answers to "
    "that sound too. With 'Keep each song's sound beside it' switched on in the 'Saved "
    "songs...' window, the sound is read off the disk instead, which is faster still and is "
    "the song's own file to the bit.\n\nA song joined to 'audio' wins over this field: the "
    "graph is what the run is about, and the field is what to do when there is no graph above "
    "this node yet."
)

AUDIO_TOOLTIP = (
    "A song this pack has sung: the output of 'YuE2 Generate Song', 'YuE2 Render Plan' or "
    "'YuE2 Decode Latents', or a FLAC or WAV of one loaded with 'Load Audio'. Optional: with "
    "nothing joined here the node opens the song chosen with 'Saved songs...'.\n\n"
    "An MP3 or another lossy file changes nearly every sample, and the song behind it is not "
    "found.\n\n"
    "The square beside this input on the node switches it off: the song chosen with 'Saved "
    "songs...' is edited instead, and whatever sings into this input is not run for this node."
)

USE_AUDIO_TOOLTIP = (
    "Whether the song joined to 'audio' is the one edited; the square beside that input on the "
    "node is this switch. Switched off, the node edits the song chosen with 'Saved songs...' and "
    "leaves the link where it is -- and the node above is not run for it at all, so a song sung "
    "before a restart is not sung again to edit one bar of it.\n\n"
    "A run of the whole workflow still runs every node that something else needs, such as a "
    "'Save Audio' joined to the same song."
)

TAKES_TOOLTIP = (
    "How many times a retake, a change of words or a change of notes sings the same stretch, so "
    "there is something to choose between. New words keep the take heard singing the most of "
    "them, the join deciding a tie. A retake or a change of notes does the same with the words "
    "the song sings there when Qwen3-ASR "
    "and the word aligner are already on the machine -- they come with the first change of "
    "words -- and otherwise keeps the take whose join the model likes best. A song that goes on "
    "sings this many takes as well, each writing its own score, and keeps one that ended the "
    "song by itself, heard singing the most of the new words. The track window "
    "plays them all and lets you keep another, which costs nothing -- they are all already "
    "sung.\n\n"
    "They are sung once and kept in this session's memory. A workflow opened in a fresh "
    "ComfyUI sings only the take its list kept, however many it once compared; the window "
    "shows the others and sings one on request, a seed always giving back the same take.\n\n"
    "A cut ignores this: a cut is the same cut however often it is made."
)

EDITS_TOOLTIP = (
    "The edits to make, as a JSON list, written by the track window. Empty means no edits: the "
    "node hands the song on as it is and draws the track.\n\n"
    "Each edit is {'op': 'retake', 'cut' or 'words'} with either 'bars': [first, stop] counting "
    "from 0, the second bar not included, or 'seconds': [from, to] for a song sung with cot "
    "'off', which has no score. A retake also takes a 'seed', how many 'takes' to sing, and "
    "which 'take' to keep. A change of words takes those too, and 'lines': [first, stop] of the "
    "words with the 'text' they become; without bars or seconds it is sung where those lines "
    "are. A change of notes, 'notes', takes a seed and takes as well, and the whole 'score' the "
    "song is to have, as the score editor leaves it: without bars it is sung over the bars whose "
    "notes that score changes, and with bars only the music of those bars is taken from it. A "
    "song that goes on, 'extend', takes a seed and takes, and the 'text' it goes on with, empty "
    "for a new ending alone; it selects no bars, going on from where the singing ends."
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
    "A retake sings two takes by default and keeps the one that joins the old song best, or, "
    "once the speech models below are on the machine, the one heard singing its words. A cut "
    "takes whole bars, moves both ends back to the same place in the singing so that a pickup "
    "is not left behind, and takes the words of any section it empties out of the lyrics.\n\n"
    "Click a line of the words beside the track to give it other words: the tune stays, and the "
    "new words are sung over the stretch that line was sung in. That stretch is found by hearing "
    "the song with Qwen3-ForcedAligner (1.8 GB), and the takes are heard with Qwen3-ASR "
    "(4.1 GB) to keep the one that sings the new words; both are Apache-2.0 and are downloaded "
    "the first time they are needed. With the aligner on the machine, every line of the words "
    "lights up where the song sings it.\n\n"
    "Press 'Notes...' to change the song's notes in the piano roll: only the bars whose notes "
    "changed are sung again, under the new score, and the rest of the song stays as it was "
    "sung. Bars changed far apart are sung as separate edits.\n\n"
    "Press 'Go on...' to make the song longer: write the lines it goes on with -- another "
    "chorus, a bridge -- or none, for a new ending alone. The model writes the tune of the new "
    "part itself, from where the singing ends, sings it and ends the song its own way; the old "
    "ending goes.\n\n"
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

GRID_SUFFIX = ".grid.json"
"""Where a measured grid waits for the next session, beside the song it belongs to.

Measuring one separates the song's voice: eight seconds of card on a
four-minute song, measured on the user's own runs on 2026-09-22. The answer
never changes, a song being remembered under the key of its own samples, but
it lived in the session alone, so every song reopened after a restart paid
those eight seconds again -- and opening saved songs to compare them is what
the picker is for. Only what the measuring found is written; the rest of a
grid comes from the score. A file left behind by a song the store dropped is
a hundred bytes, and still right if that song ever comes back."""

_GRID_KEEP = 8

WORDS_SUFFIX = ".words.json"
"""Where the word times of a song wait for the next session, beside the song they were measured on.

Timing a song reads 1.84 GB of weights and makes one pass over the whole
recording; the answer never changes, a song being remembered under the key of
its own samples, and what it is worth is exactly the second and third line
somebody rewrites. The words they were measured on are written with them, so a
song whose words an edit has changed is timed again rather than read wrong. A
song that went on writes the second its new lines were timed from as well
(``_went_times``), and one timed whole before that is timed again."""

NO_WORDS = ("Nothing is sung in this song's words, so there is no line to rewrite. Select the "
            "bars to sing instead.")

NO_EARS = ("New words are kept by hearing which take sings them, and Qwen3-ASR, which does the "
           "hearing, is not on this machine while 'download' is off. The take whose join the "
           "model likes best is kept instead; listen to the takes in the track window, or set "
           "'download' in YuE2 Options to 'auto' to have it fetched (4.1 GB).")

DEAF = ("The takes of the new words could not be heard ({}), so the one whose join the model "
        "likes best is kept. Listen to them in the track window.")

MUMBLED_SAID = ("The take kept is heard singing {} of the {} new words. Ask for more takes, or "
                "write words that fit the tune more closely.")

MUMBLED_AGAIN = ("The take kept is heard singing {} of the {} words there, where the song as it "
                 "was sang {}. Ask for more takes, or keep the song as it was.")

STOPPED = ("The take kept had not ended the song {:.0f} seconds past where its score ends, so it "
           "was stopped there and the song stops mid-note. Ask for more takes: most end by "
           "themselves.")


def _said_once(notices, kind: str, text: str) -> None:
    """Add a notice unless the run has already said it: a run with two changes of words says it once."""
    if (kind, text) not in notices:
        notices.append((kind, text))


def _aligner_at_hand(settings):
    """The aligner's folder, the file it reads words with and the device, when it can light the words; None otherwise.

    Finding when each line is sung is worth a second of card and nothing more:
    it lights the window's words where the song sings them. So it is never
    worth a download -- the aligner comes with the first change of words --
    nor the CPU, where a pass over a song is half a minute rather than a
    tenth of a second. Without it the window places lines by the bars, as it
    always has. Never fatal.
    """
    from . import devices, discovery, download

    try:
        device = devices.resolve(settings["device"])
        if getattr(device, "type", "cpu") == "cpu":
            return None
        folder = discovery.find_aligner()
        if not folder:
            return None
        return folder, download.aligner_tokenizer(folder, {"download": "off"}), device
    except Exception:
        log.debug("[yue2_comfy.edit_track] the aligner is not at hand to light the words",
                  exc_info=True)
        return None


def _ears_at_hand(settings):
    """Qwen3-ASR's folder when the takes of a retake can be heard without asking for anything; None otherwise.

    A retake sings words the song already has, so hearing its takes makes a
    better pick rather than the edit itself: it is never worth a download of
    4.1 GB, nor the CPU. The model comes with the first change of words, and
    from then on retakes are heard too. Never fatal.
    """
    from . import devices, discovery

    try:
        device = devices.resolve(settings["device"])
        if getattr(device, "type", "cpu") == "cpu":
            return None
        return discovery.find_asr() or None
    except Exception:
        log.debug("[yue2_comfy.edit_track] the speech model is not at hand to hear a retake",
                  exc_info=True)
        return None


def _speech_folder(settings, progress, notices):
    """Qwen3-ASR's folder, fetched first when it is missing and downloading is on; None when it cannot be had.

    Hearing the takes is how new words are picked, not a condition of singing
    them, so a machine without the model and with downloading off still gets
    its words sung: the join picks, and the node says so.
    """
    from . import download

    try:
        return download.ensure_asr(settings, progress)
    except FileNotFoundError:
        _said_once(notices, "notice", NO_EARS)
    except download.DownloadError as error:
        _said_once(notices, "notice", DEAF.format(str(error).rstrip(".")))
    return None


def _clip(waveform, rate: int, start: float, stop: float):
    """Seconds ``start`` to ``stop`` of a sound as the speech model hears it: mono, 16 kHz."""
    from .asr import model as asr_model

    first = max(0, int(round(start * rate)))
    last = max(first, min(int(waveform.shape[-1]), int(round(stop * rate))))
    return asr_model.mono_16k(waveform[..., first:last], rate)


def _ears_off(settings, listened) -> None:
    """Let go of the models that listened in this run, unless it asked to keep what it loads. Never fatal.

    The aligner and the speech model are 1.7 GiB and 3.8 GiB on the card, and
    like the song model they wait for the next run only when
    'keep_model_loaded' says so; loading either again takes under two
    seconds. ``listened`` names the ones this run used, so a model another
    node keeps loaded is not let go by a run that never touched it.
    """
    if settings.get("keep_model_loaded") or not listened:
        return
    try:
        from .asr import runtime as asr_runtime

        if "speech" in listened:
            asr_runtime.unload()
        if "aligner" in listened:
            asr_runtime.unload_aligner()
    except Exception:
        log.debug("[yue2_comfy.edit_track] the models that listened could not be let go",
                  exc_info=True)


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
            handle.writeframes(songs.pcm16(waveform))
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


def _edited(settings, edit) -> dict:
    """The settings one edit sings by: the song's own, with what the window asked for over them.

    An edit is sung the way the song was sung -- that is what keeps a retake
    in the same voice as the song around it -- unless the window says
    otherwise for that one edit. See ``track.VARY`` and ``track.GUIDE``.
    """
    if getattr(edit, "vary", None) is None and getattr(edit, "guide", None) is None:
        return settings
    wanted = dict(settings)
    if edit.vary is not None:
        wanted["temperature"] = float(edit.vary)
    if edit.guide is not None:
        wanted["cfg_scale"] = float(edit.guide)
    return wanted


def _edge(step, frames: int) -> str:
    """Which end of the song a cut left bare: 'head', 'tail', or neither of them."""
    if step.kind != "cut":
        return ""
    if step.start <= 0 and step.stop < frames:
        return "head"
    if step.stop >= frames and step.start > 0:
        return "tail"
    return ""


def _faded(waveform, seconds, head: bool, rate: int):
    """The sound with a ramp laid on the edge a cut left bare.

    A cut inside the song is joined with a crossfade at each end. One that
    takes the first bars, or the last, has no other side to fade into, so the
    song would start or stop wherever the samples happened to be. The ramp is
    the raised cosine the joins use, over as many seconds as the edit asks
    for. Never laid in place: the take it came from is kept for the session.
    """
    import torch

    count = min(int(round(float(seconds) * rate)), int(waveform.shape[-1]))
    if count < 2:
        return waveform
    steps = torch.arange(count, dtype=torch.float32) + 0.5
    ramp = (0.5 - 0.5 * torch.cos(math.pi * steps / count)).to(waveform.dtype)
    faded = waveform.clone()
    if head:
        faded[..., :count] *= ramp
    else:
        faded[..., -count:] *= ramp.flip(0)
    return faded


def _sampled(settings) -> dict:
    """What the song is sung with, for the window's own knobs to start from."""
    from .inpaint import core

    return {"vary": round(float(settings.get("temperature") or 0.0), 3),
            "guide": round(float(core.guidance(settings)), 3)}


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


def _grid_file(name: str):
    """The path the grid for *name* is kept at, or None when there is nowhere to keep it."""
    folder = songs.store().folder
    if not folder or not songs.is_key(name):
        return None
    return os.path.join(folder, name + GRID_SUFFIX)


def _grid_read(name: str, sheet):
    """The grid measured for this song in some other session, or None. Never fatal."""
    path = _grid_file(name)
    if path is None:
        return None
    try:
        with io.open(path, encoding="utf-8") as handle:
            kept = json.load(handle)
        return grid.Grid(offset=float(kept["offset"]), rate=float(kept["rate"]),
                         tick=grid.tick_seconds(sheet), starts=grid.starts_of(sheet),
                         by_voice=bool(kept["by_voice"]))
    except (OSError, ValueError, KeyError, TypeError):
        log.debug("[yue2_comfy.edit_track] %s cannot be read and is measured again", path,
                  exc_info=True)
        return None


def _grid_keep(name: str, clock) -> None:
    """Write the measured grid beside its song. Never fatal: it only saves time."""
    path = _grid_file(name)
    if path is None:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as handle:
            json.dump({"offset": float(clock.offset), "rate": float(clock.rate),
                       "by_voice": bool(clock.by_voice)}, handle)
    except OSError:
        log.debug("[yue2_comfy.edit_track] the grid could not be written to %s", path,
                  exc_info=True)


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
    kept = _grid_read(name, sheet)
    if kept is not None:
        _GRIDS[name] = kept
        log.info("[yue2_comfy.edit_track] the score's place on this song was measured before: "
                 "from %.2f s at %.4f of its tempo, placed by %s", kept.offset, kept.rate,
                 "the voice" if kept.by_voice else "the mix")
        return kept
    path = separator_weights(settings, unique_id, progress)
    voice = voice_of({"waveform": waveform, "sample_rate": rate}, settings, unique_id, progress,
                     path=path)["waveform"]
    began = time.perf_counter()
    clock = grid.measured(sheet, waveform, rate, voice=voice)
    log.info("[yue2_comfy.edit_track] the score sits on this song from %.2f s at %.4f of its tempo, "
             "placed by %s, in %.1f s", clock.offset, clock.rate,
             "the voice" if clock.by_voice else "the mix", time.perf_counter() - began)
    _GRIDS[name] = clock
    _grid_keep(name, clock)
    while len(_GRIDS) > _GRID_KEEP:
        _GRIDS.popitem(last=False)
    return clock


def _words_file(name: str):
    """The path the word times for *name* are kept at, or None when there is nowhere to keep them."""
    folder = songs.store().folder
    if not folder or not name or not songs.is_key(name):
        return None
    return os.path.join(folder, name + WORDS_SUFFIX)


def _said(text: str) -> str:
    """The words the times were measured on, short enough to write beside them."""
    import hashlib

    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def _times_read(name: str, text: str, since=None):
    """The word times measured for this song in some other session, or None. Never fatal.

    ``since`` is the second a song that went on has its new lines timed from,
    and times read for it must say they were measured so: a song timed whole
    before it was known to lose those lines is timed again.
    """
    path = _words_file(name)
    if path is None:
        return None
    try:
        with io.open(path, encoding="utf-8") as handle:
            kept = json.load(handle)
        if kept.get("words") != _said(text):
            return None
        if since is not None and kept.get("from") != round(float(since), 3):
            return None
        return [(str(word), float(start), float(stop)) for word, start, stop in kept["times"]]
    except (OSError, ValueError, KeyError, TypeError):
        log.debug("[yue2_comfy.edit_track] %s cannot be read and the words are timed again", path,
                  exc_info=True)
        return None


def _times_keep(name: str, text: str, times, since=None) -> None:
    """Write the word times beside their song, and ``since`` when the new lines were timed from there. Never fatal: they only save time."""
    path = _words_file(name)
    if path is None:
        return
    kept = {"words": _said(text),
            "times": [[word, round(float(start), 3), round(float(stop), 3)]
                      for word, start, stop in times]}
    if since is not None:
        kept["from"] = round(float(since), 3)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as handle:
            json.dump(kept, handle)
    except OSError:
        log.debug("[yue2_comfy.edit_track] the word times could not be written to %s", path,
                  exc_info=True)


def _flagged(take) -> bool:
    """Whether a take's join is enough worse than the song's own there to be worth saying so."""
    return (take.natural is not None and take.join is not None
            and take.join < take.natural - JOIN_SLACK)


def _mumbled(take, was=None) -> bool:
    """Whether a take was heard singing too few of its words to leave unsaid; see ``MUMBLED``.

    ``was`` is how the song as it was is heard there, for a retake: the same
    words, so a take heard singing as many of them as the song did is not
    mumbling whatever the share, the recogniser missing the same words in both.
    """
    heard = getattr(take, "heard", None)
    if heard is None or heard[1] <= 0 or heard[0] >= MUMBLED * heard[1]:
        return False
    return was is None or heard[0] < was[0]


def _take_facts(take, index: int, chosen: int, name: str, rate: int, prior, step,
                lines=None, was=None) -> dict:
    """One take as the window shows it: how it scored, how long it came out, and the song it makes.

    ``seconds`` is the take itself and ``total`` the song with it in, which is
    what the window draws and plays; ``peaks`` and ``rms`` are that song's
    wave, and ``grid`` its own layout of the score, because a take that came
    out longer or shorter than the one kept moves every bar after the edit --
    so switching takes swaps the track, its bars and its sound under the
    cursor without the node running again. ``lines`` is when each line of the
    words is sung in that song, for the same reason. ``heard`` and ``said``
    are what was heard of the take (see ``core.Take``), and ``was`` how the
    song as it was is heard there, which a retake is measured against.
    ``prior`` is the song's state before this edit and ``step`` the edit's
    plan. See ``TAKE_FILES`` for what the files cost.
    """
    from .inpaint import grid, track

    entry = _write_wave(take.waveform, rate, "{}_{}.wav".format(name[:16], take.seed))
    drawn = _wave(take.waveform)
    own = track.after(prior, step, take.count,
                      take.song.score if step.kind == "extend" else None)
    laid = None
    if own.sheet is not None and own.clock is not None:
        laid = grid.layout(own.sheet, own.clock, own.frames)
    heard = getattr(take, "heard", None)
    return {"seed": int(take.seed), "index": index, "kept": index == chosen, "sung": True,
            "seconds": round(take.count * FRAME_SECONDS, 3),
            "total": round(int(take.waveform.shape[-1]) / float(rate), 3),
            "peaks": drawn["peaks"], "rms": drawn["rms"], "grid": laid, "lines": lines,
            "join": None if take.join is None else round(take.join, 4),
            "natural": None if take.join is None or take.natural is None else round(take.natural, 4),
            "heard": None if heard is None else [int(heard[0]), int(heard[1])],
            "said": getattr(take, "said", None), "mumbled": _mumbled(take, was),
            "ended": bool(take.ended), "flagged": _flagged(take), "audio": entry}


def _was_facts(waveform, rate: int, prior, step, name: str, lines=None, said=None,
               heard=None) -> dict:
    """The song as it stood before the edit, as a row the takes list shows first.

    Shaped like a take, so the window draws, plays and compares it the same
    way, with ``index`` -1 and no seed: it is not a take of the edit, it is
    what the edit replaced. ``seconds`` is the stretch as it was, against a
    take's own length, ``grid`` the bars before the edit moved them and
    ``lines`` the lines where it sang them. ``said`` is what was heard there
    when the takes were heard, and ``heard`` how many of a retake's words
    that is -- the same words the takes are counted on, so the row says what
    the takes are up against.
    """
    from .inpaint import grid

    entry = _write_wave(waveform, rate, "{}_was.wav".format(name[:16]))
    drawn = _wave(waveform)
    laid = None
    if prior.sheet is not None and prior.clock is not None:
        laid = grid.layout(prior.sheet, prior.clock, prior.frames)
    return {"seed": None, "index": -1, "kept": False, "sung": True,
            "seconds": round((step.stop - step.start) * FRAME_SECONDS, 3),
            "total": round(int(waveform.shape[-1]) / float(rate), 3),
            "peaks": drawn["peaks"], "rms": drawn["rms"], "grid": laid, "lines": lines,
            "join": None, "natural": None,
            "heard": None if heard is None else [int(heard[0]), int(heard[1])],
            "said": said, "mumbled": False, "ended": False, "flagged": False, "audio": entry}


def _goes_on(state):
    """Where this song would go on from, as the window marks it: ``{"bar", "second"}``, or None.

    None for a song with no score, which has nothing for the model to go on
    writing, and for one whose bars cannot be found in its sound. Never fatal.
    """
    try:
        found = track.going_on_from(state)
    except ValueError:
        return None
    if found is None:
        return None
    return {"bar": int(found["bar"]), "second": round(found["start"] * FRAME_SECONDS, 3)}


def _moved(marks, step, count: int) -> list:
    """Where the marks of the edits already made sit in the song after this one.

    An edit that put ``count`` frames where ``step.start`` to ``step.stop``
    were moves everything after it by the difference, and swallows whatever
    stood inside it.
    """
    shift = count - (step.stop - step.start)
    moved = []
    for mark in marks:
        if step.start <= mark["start"] < step.stop:
            continue
        if mark["start"] >= step.stop:
            mark = dict(mark, start=mark["start"] + shift)
        moved.append(mark)
    return moved


def _stamped(edited, waveform, parent: str, before, marks):
    """The edited song with what a picker needs of it: its family, its age, its picture.

    ``parent`` is the key the run opened and ``before`` the song kept under
    it, so this one belongs to that song's line, or starts one under that
    song. The picture is a kilobyte and never worth failing the edit over.
    """
    import dataclasses

    peaks, body = b"", b""
    try:
        peaks, body = songs.strip(waveform)
    except Exception:
        log.warning("[yue2_comfy.edit_track] this edit could not be drawn for the picker",
                    exc_info=True)
    return dataclasses.replace(
        edited, parent=parent, root=before.root or parent, created=time.time(),
        peaks=peaks, body=body,
        edit=[{"op": mark["op"],
               "at": [round(mark["start"] * FRAME_SECONDS, 3),
                      round((mark["start"] + mark["count"]) * FRAME_SECONDS, 3)],
               "bars": mark["bars"], "seed": mark["seed"], "took": mark["took"],
               "was": mark["was"], "now": mark["now"]}
              for mark in marks])


def _take_gap(seed: int, index: int) -> dict:
    """A take the list asks for that this session has not sung, as a row the window can offer.

    It carries its seed and nothing else, because nothing else exists yet: no
    file, no length, no join. Singing it later gives the take that seed always
    gives, so the row is an offer, not a loss.
    """
    return {"seed": int(seed), "index": index, "kept": False, "sung": False,
            "seconds": None, "total": None, "peaks": [], "rms": [], "grid": None, "lines": None,
            "join": None, "natural": None, "heard": None, "said": None, "mumbled": False,
            "ended": False, "flagged": False, "audio": None}


def _sound_of(models, song, progress):
    """A remembered song turned back into sound, for a run that was handed no audio.

    The latents are what the acoustic stage made of the performance, so this
    is the last stage of singing the song and nothing else: 1.1 to 1.7 seconds
    for a four-minute song on a 5090, measured 2026-09-19, against minutes to
    sing one. What comes out is the same performance but not the same file to
    the bit -- the decode picks its convolutions from the memory it has, which
    moves the last bits 64 to 68 dB down -- so the caller writes that sound's
    name down beside the song, and what this node hands on can always be
    edited again.
    """
    from . import generate
    from .inpaint import core

    began = time.perf_counter()
    latents = core.latents_of(song).to(models.device)
    waveform, timing = generate.decode(models, latents, progress, interrupted,
                                       stages=generate.alone(generate.Stages.DECODE))
    log.info("[yue2_comfy.edit_track] the remembered song came back as %.1f s of sound in %.1f s",
             timing["seconds_of_audio"], time.perf_counter() - began)
    return waveform


def _kept_sound(name: str, song):
    """The song's own sound, read off the disk instead of decoded, or None when there is none.

    Only when it really is this song's sound: a file whose rate or length is
    not what the song says is not, and the song is decoded as it would have
    been without it. See ``songs.sound_of``.
    """
    try:
        kept = songs.sound_of(name)
    except Exception:
        log.warning("[yue2_comfy.edit_track] the sound kept beside this song could not be read, "
                    "so it is decoded instead", exc_info=True)
        return None
    if kept is None:
        return None
    waveform = kept["waveform"]
    if (int(kept["sample_rate"]) != int(song.sample_rate)
            or int(waveform.shape[-1]) != int(song.samples)):
        log.warning("[yue2_comfy.edit_track] the sound kept beside this song is %d samples at "
                    "%d Hz where the song is %d at %d, so it is decoded instead",
                    int(waveform.shape[-1]), int(kept["sample_rate"]), int(song.samples),
                    int(song.sample_rate))
        return None
    return waveform


def _keep_sound(name: str, waveform, rate: int) -> None:
    """Keep the sound this song has just been decoded into, when the store keeps sounds.

    A song sung before the switch was turned on has no sound beside it, and
    the only way to make one is the decode this run has already paid for. So
    the first time such a song is opened it pays that decode once, and never
    again. Never fatal.
    """
    try:
        if songs.store().wants_sound():
            songs.sound_keep(name, waveform, rate)
    except Exception:
        log.warning("[yue2_comfy.edit_track] the sound of this song could not be kept beside it",
                    exc_info=True)


def _remember(waveform, song) -> str:
    """Remember the edited song, so it can be saved, loaded and edited again. Never fatal."""
    try:
        return songs.remember(waveform, song.sample_rate, song)
    except Exception:
        log.warning("[yue2_comfy.edit_track] this edit could not be remembered, so it cannot be "
                    "edited again", exc_info=True)
        return ""


def _also(waveform, song, name: str) -> None:
    """Write down that a remembered song also answers to the sound it decodes to. Never fatal.

    That sound is not the one the song was remembered by, see
    ``songs.Store.alias``, so without this the audio this node hands on is a
    song the pack does not know, and remembering it again writes a twin of
    every latent of it.
    """
    try:
        songs.alias(waveform, song.sample_rate, name)
    except Exception:
        log.warning("[yue2_comfy.edit_track] the sound this song decodes to could not be "
                    "written down, so editing what this node hands on may not find it",
                    exc_info=True)


class YuE2EditTrack:
    """A song in, the same song with part of it sung again or taken out."""

    DESCRIPTION = DESCRIPTION
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "takes": ("INT", {"default": 2, "min": 1, "max": track.MAX_TAKES,
                                  "tooltip": TAKES_TOOLTIP}),
            },
            "optional": {
                "audio": ("AUDIO", {"lazy": True, "tooltip": AUDIO_TOOLTIP}),
                "edits": ("STRING", {"multiline": True, "default": "", "tooltip": EDITS_TOOLTIP}),
                "options": (OPTIONS_TYPE, {"tooltip": OPTIONS_TOOLTIP}),
                "song_key": ("STRING", {"default": "", "tooltip": SONG_TOOLTIP}),
                "use_audio": ("BOOLEAN", {"default": True, "tooltip": USE_AUDIO_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "edit"
    CATEGORY = CATEGORY

    def check_lazy_status(self, audio=UNJOINED, use_audio=True, **_rest):
        """The inputs ComfyUI has to work out before this node runs: 'audio', unless it is switched off.

        'audio' is lazy so that switching it off means what it says. A joined
        input is worked out before a node runs whether the node looks at it or
        not, so a song joined here would be sung again by the node above --
        minutes, after a restart -- only to be ignored for the song chosen
        with 'Saved songs...'. Asked for here, it is worked out only when it is
        used.
        """
        return ["audio"] if use_audio and audio is None else []

    def edit(self, audio=None, takes=2, edits="", options=None, song_key="", use_audio=True,
             unique_id=None):
        """The song named by 'audio' or by 'song_key', with the list of edits made on it.

        Audio wins when both are there and 'use_audio' is on: the graph above
        this node is what the run is about, and the field is for a node that
        has no graph above it yet -- a workflow opened after a restart, where
        singing the song again to edit a bar of it costs minutes. With
        'use_audio' off the link is left alone and the field decides.
        """
        from .inpaint import core
        from .staged import adapters, session

        progress = NodeProgress(unique_id, title=ORIGIN)
        given = bool(use_audio) and bool(audio) and audio.get("waveform") is not None
        opened = "" if given else str(song_key or "").strip()
        if not given and not opened:
            refuse(unique_id, NO_AUDIO)
        waveform, rate = (audio["waveform"], int(audio["sample_rate"])) if given else (None, 0)
        name = opened
        if given:
            try:
                name = songs.key(waveform, rate)
            except ValueError as error:
                refuse(unique_id, str(error))
        try:
            song = songs.store().get(name)
        except ValueError:
            refuse(unique_id, NOT_A_KEY)
        if song is None:
            refuse(unique_id, UNKNOWN if given else GONE)
        name = songs.canonical(name)
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
        stack = contextlib.ExitStack()
        models = [None]
        singing = [None]
        listened = set()

        def loaded():
            if models[0] is None:
                held = contextlib.ExitStack()
                stack.callback(held.close)
                models[0] = held.enter_context(session(settings, unique_id, progress))
                singing[0] = held
            return models[0]

        def released():
            if singing[0] is not None:
                held, singing[0], models[0] = singing[0], None, None
                held.close()

        history, notices, picks, shown, marks = [], [], [], None, []
        clock, state, current, sound = None, None, song, waveform
        keyed, timed, ears, place = name, {}, None, None
        went = {}
        try:
            with stack:
                stack.callback(_ears_off, settings, listened)
                first = 0.0
                if not given:
                    waveform = _kept_sound(name, song)
                    if waveform is None:
                        first = DECODE_SHARE
                        waveform = _sound_of(loaded(), song,
                                             Band(progress, 0.0, DECODE_SHARE) if wanted
                                             else progress)
                        _keep_sound(name, waveform, int(song.sample_rate))
                    rate, sound = int(song.sample_rate), waveform
                low = first
                if song.score:
                    low = first if name in _GRIDS else first + GRID_SHARE
                    clock = _grid_of(name, song, waveform, rate, settings, unique_id,
                                     Band(progress, first, first + GRID_SHARE) if wanted
                                     else progress)
                state = track.opened(song, clock, sheet)
                place = _aligner_at_hand(settings)
                ears = _ears_at_hand(settings)
                top = 1.0 - LINES_SHARE if place is not None else 1.0
                for index, edit in enumerate(wanted):
                    if interrupted():
                        raise InterruptedError("Cancelled between edits")
                    opens = low + (top - low) * index / len(wanted)
                    closes = low + (top - low) * (index + 1) / len(wanted)
                    span = None
                    if track.needs_times(edit):
                        found_by = opens + (closes - opens) * TIMES_SHARE
                        span = self._span(state, edit, sound, rate, settings, index,
                                          track.sound_name(name, history),
                                          name if not history else "",
                                          Band(progress, opens, found_by), listened, went)
                        opens = found_by
                    try:
                        step = track.plan(state, edit, span)
                    except ValueError as error:
                        raise _Refused("Edit {}: {}".format(index + 1, error))
                    notices.extend(step.notices)
                    made = track.name(name, history, edit)
                    asked = ""
                    if step.kind in ("words", "extend"):
                        asked = chr(10).join(step.now)
                    elif step.kind in ("retake", "notes") and ears is not None:
                        found_by = opens + (closes - opens) * TIMES_SHARE
                        asked = self._words_there(place, state, step, sound, rate, name, history,
                                                  Band(progress, opens, found_by), listened, went)
                        opens = found_by
                    hears = closes - (closes - opens) * HEARD_SHARE if asked else closes
                    entry = self._sung(loaded, current, sound, state, step, edit, made,
                                       settings, Band(progress, opens, hears))
                    seeds = (0,) if step.kind == "cut" else edit.seeds()
                    if asked:
                        self._heard(entry, seeds, step, sound, rate, settings, made, released,
                                    Band(progress, hears, closes), notices, listened, asked,
                                    ears if step.kind in ("retake", "notes") else None)
                    ordered = [entry["takes"].get(seed) for seed in seeds]
                    pick = edit.take if edit.take is not None else core.best(ordered)
                    kept = ordered[pick]
                    history.append((edit, kept.seed))
                    picks.append(pick)
                    prior, earlier = state, sound
                    edge = _edge(step, state.frames)
                    state = track.after(state, step, kept.count,
                                        kept.song.score if step.kind == "extend" else None)
                    current, sound = kept.song, kept.waveform
                    if edge and edit.fade:
                        sound = _faded(sound, edit.fade, edge == "head", rate)
                    marks = _moved(marks, step, kept.count)
                    marks.append({"op": step.kind, "start": step.start, "count": kept.count,
                                  "bars": list(edit.bars) if edit.bars else None,
                                  "seed": None if step.kind == "cut" else int(kept.seed),
                                  "took": round((step.stop - step.start) * FRAME_SECONDS, 3),
                                  "was": "\n".join(step.was), "now": "\n".join(step.now)})
                    shown = (step, ordered, pick, made, prior, seeds, earlier, entry)
                    if step.kind == "extend":
                        before = (track.sound_name(name, history[:-1]),
                                  name if len(history) == 1 else "", earlier, prior.lyrics)
                        for take in ordered:
                            if take is not None and take.song is not None:
                                went[track.sound_name(name, history[:-1] + [(edit, take.seed)])] = (
                                    before + (track.sings_from(prior, step,
                                                               notation.read(take.song.score)),))
                released()
                if wanted:
                    keyed = _remember(sound, _stamped(current, sound, name, song, marks))
                elif not given:
                    _also(sound, song, name)
                if place is None and "aligner" in listened:
                    place = _aligner_at_hand(settings)
                if place is not None:
                    timed = self._timed(place, name, keyed, history, shown, sound, rate, state,
                                        Band(progress, top, 1.0), listened, went)
        except InterruptedError:
            translate_interrupt()
            raise
        except _Refused as stop:
            refuse(unique_id, str(stop))

        payload = self._payload(name, keyed, sound, rate, state, wanted, picks, shown,
                                settings, timed, ears is not None and place is not None)
        if shown is not None and _flagged(shown[1][shown[2]]):
            kept = shown[1][shown[2]]
            notices.append(("warn", "The join of the take kept scores {:.2f} where the song's own "
                                    "join there scores {:.2f}, so it may be heard. Ask for more "
                                    "takes, or open the selection wider.".format(
                                        kept.join, kept.natural)))
        if shown is not None and shown[0].kind == "extend" and not shown[1][shown[2]].ended:
            notices.append(("warn", STOPPED.format(ops.LATER * FRAME_SECONDS)))
        was = None if shown is None else shown[7].get("was_heard")
        if shown is not None and _mumbled(shown[1][shown[2]], was):
            kept = shown[1][shown[2]]
            notices.append(("warn", MUMBLED_SAID.format(kept.heard[0], kept.heard[1])
                            if was is None else
                            MUMBLED_AGAIN.format(kept.heard[0], kept.heard[1], was[0])))
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

    def _span(self, state, edit, waveform, rate: int, settings, index: int, sound: str,
              saved: str, progress, listened=None, went=None):
        """The frames the lines one edit rewrites are sung over, from the song's own word times.

        The score knows a note for every syllable but not which line the
        singer was on, and the grid's sections are whole verses, so a line is
        placed by hearing it: the forced aligner is given the words and says
        when each is sung. It reads the song once, the answer is kept beside
        the song, and the region opens at the last word of the line before --
        the model sings that word again and runs into the new line, which is
        what the stand measured as the difference between every word sung and
        a line that starts late. A song this run made go on is timed as
        ``_went_times`` says.
        """
        from . import devices, download
        from .inpaint import lines

        where = "Edit {}".format(index + 1)
        text = lines.heard_text(state.lyrics)
        if not text.strip():
            raise _Refused("{}: {}".format(where, NO_WORDS))
        listened = set() if listened is None else listened
        times = self._times_of(None, sound, saved, waveform, rate, text, progress, listened, went)
        if times is None:
            try:
                folder = download.ensure_aligner(settings, progress)
                reader = download.aligner_tokenizer(folder, settings, progress)
            except (FileNotFoundError, download.DownloadError) as error:
                raise _Refused(str(error))
            began = time.perf_counter()
            times = self._times_of((folder, reader, devices.resolve(settings["device"])), sound,
                                   saved, waveform, rate, text, progress, listened, went)
            log.info("[yue2_comfy.edit_track] %d words timed in %.1f s", len(times),
                     time.perf_counter() - began)
        try:
            start, stop = lines.region(state.lyrics, times, edit.lines[0], edit.lines[1])
        except ValueError as error:
            raise _Refused("{}: {}".format(where, error))
        return (int(round(start / FRAME_SECONDS)),
                state.frames if stop is None else int(round(stop / FRAME_SECONDS)))

    def _words_there(self, place, state, step, sound, rate: int, name: str, history, progress,
                     listened, went=None) -> str:
        """The words the song sings in the stretch a retake sings again, or "" when they cannot be known.

        They are what the takes are heard against, and only the song's own
        word times can say which words those are: the score has a note for
        every syllable but no idea which line the singer was on. The times
        are read beside the song when it was timed before, and heard with the
        aligner otherwise when it is at hand -- the same pass that lights the
        lines of the song as it was, so it is paid once. A stretch nothing is
        sung in has no words, and its takes are left to the join. Never fatal
        but for a cancel.
        """
        from .inpaint import lines

        text = lines.heard_text(state.lyrics)
        if not text.strip():
            progress.ratio(1.0)
            return ""
        try:
            times = self._times_of(place, track.sound_name(name, history),
                                   name if not history else "", sound, rate, text, progress,
                                   listened, went)
        except InterruptedError:
            raise
        except Exception:
            log.warning("[yue2_comfy.edit_track] the words of the retake could not be found, so "
                        "its takes are left to the join", exc_info=True)
            return ""
        progress.ratio(1.0)
        if times is None:
            return ""
        return lines.within(times, step.start * FRAME_SECONDS, step.stop * FRAME_SECONDS)

    def _heard(self, entry, seeds, step, before, rate: int, settings, made: str, release,
               progress, notices, listened, asked: str, ears=None):
        """Every take of an edit heard, and how many of the words ``asked`` for each sings kept on it.

        ``asked`` is the new words of a change of words or of a song that goes
        on, or the words the song sings where a retake or a change of notes
        sings again. Only a take not heard yet is heard,
        so asking for one more take hears that one alone. The stretch as the
        song sang it before is heard first: its language is the surest, and
        every take is then heard in it, so the takes are heard alike. For a
        retake or a change of notes it is also counted like a take, since it
        sings the same words.
        The singing model is let go before the speech model loads --
        ``release`` does it, and the next edit loads it again -- so a card
        that holds one of them holds the other. The takes stay as they are
        when nothing can hear them, and the join picks.

        ``ears`` is the speech model already on the machine, which is all a
        retake or a change of notes is heard with; new words, changed or
        going on, fetch it when it is missing and downloading is on. Every clip is heard in
        the language the letters of ``asked`` name (``lines.language_of``),
        or, when they name none, in the one the model names for the first.
        """
        from . import devices
        from .asr import runtime as asr_runtime
        from .inpaint import lines

        fresh = [entry["takes"][seed] for seed in seeds
                 if seed in entry["takes"] and entry["takes"][seed].heard is None]
        if not fresh:
            progress.ratio(1.0)
            return
        folder = ears or _speech_folder(settings, progress, notices)
        if folder is None:
            return
        release()
        opens = step.start * FRAME_SECONDS - HEARD_AROUND
        clips = [((made, "was"),
                  _clip(before, rate, opens, step.stop * FRAME_SECONDS + HEARD_AROUND))]
        for take in fresh:
            clips.append(((made, take.seed),
                          _clip(take.waveform, rate, opens,
                                (step.start + take.count) * FRAME_SECONDS + HEARD_AROUND)))
        listened.add("speech")
        began = time.perf_counter()
        try:
            answers = asr_runtime.hear(folder, devices.resolve(settings["device"]), clips,
                                       language=lines.language_of(asked), cancelled=interrupted,
                                       progress=progress)
        except InterruptedError:
            raise
        except Exception as error:
            log.warning("[yue2_comfy.edit_track] the takes could not be heard", exc_info=True)
            _said_once(notices, "notice", DEAF.format(str(error).rstrip(".") or "no reason given"))
            return
        entry["said"] = answers[0]["text"]
        entry["asked"] = asked
        if step.kind in ("retake", "notes"):
            entry["was_heard"] = lines.heard(asked, answers[0]["text"])
            log.info("[yue2_comfy.edit_track] the song as it was is heard singing %d of the %d "
                     "words there: %s", entry["was_heard"][0], entry["was_heard"][1],
                     entry["said"])
        for take, answer in zip(fresh, answers[1:]):
            take.heard = lines.heard(asked, answer["text"])
            take.said = answer["text"]
            log.info("[yue2_comfy.edit_track] take %d is heard singing %d of %d words: %s",
                     take.seed, take.heard[0], take.heard[1], take.said)
        log.info("[yue2_comfy.edit_track] %d take%s heard in %.1f s", len(fresh),
                 "" if len(fresh) == 1 else "s", time.perf_counter() - began)

    def _timed(self, place, name: str, keyed: str, history, shown, sound, rate: int, state,
               progress, listened, went=None) -> dict:
        """When each sung line is sung, in every sound the window can put on the track.

        ``{"song": lines, "takes": {index: lines}, "before": lines}``, where
        lines are ``[[line, start, stop]]`` in seconds, numbered as the lines
        of the words the window shows, and None where they could not be found
        -- the window then places them by the bars, as it did before there was
        anything better. The takes are the last edit's, each a song of its own
        length, the one kept being the song itself. The song as it was comes
        numbered as the lines after the edit, which a change of words may have
        moved (``lines.carried``). A song this run made go on, and each of its
        takes, is timed as ``_went_times`` says.
        """
        from .inpaint import lines

        found = {"song": None, "takes": {}, "before": None}
        jobs = [("song", None, sound, state.lyrics, track.sound_name(name, history), keyed)]
        if shown is not None:
            step, ordered, pick, earlier, prior = shown[0], shown[1], shown[2], shown[6], shown[4]
            edit = history[-1][0]
            for index, take in enumerate(ordered):
                if take is not None and index != pick:
                    jobs.append(("takes", index, take.waveform, state.lyrics,
                                 track.sound_name(name, history[:-1] + [(edit, take.seed)]), ""))
            if step.kind != "cut":
                jobs.append(("before", None, earlier, prior.lyrics,
                             track.sound_name(name, history[:-1]),
                             name if len(history) == 1 else ""))
        for at, (where, index, waveform, lyrics, label, disk) in enumerate(jobs):
            spans = self._lines_of(place, label, disk, waveform, rate, lyrics,
                                   Band(progress, at / len(jobs), (at + 1) / len(jobs)), listened,
                                   went)
            if where == "takes":
                found["takes"][index] = spans
            else:
                found[where] = spans
        if shown is not None:
            step, pick = shown[0], shown[2]
            found["takes"][pick] = found["song"]
            if step.kind == "words" and found["before"] is not None:
                edit = history[-1][0]
                found["before"] = [list(span) for span in lines.carried(
                    found["before"], edit.lines[0], edit.lines[1], state.lyrics, len(step.now))]
        return found

    def _lines_of(self, place, label: str, disk: str, waveform, rate: int, lyrics: str,
                  progress, listened, went=None):
        """``[[line, start, stop]]`` for every sung line of ``lyrics`` in ``waveform``, or None.

        ``label`` names the sound for this session and ``disk`` the song whose
        file keeps its times between sessions -- "" for a take that no song is
        remembered by. Never fatal but for a cancel: the lines are a picture,
        and a song that cannot be timed is still the song.
        """
        from .inpaint import lines

        text = lines.heard_text(lyrics)
        if not text.strip():
            return []
        try:
            times = self._times_of(place, label, disk, waveform, rate, text, progress, listened,
                                   went)
            if times is None:
                return None
            return [[number, round(start, 3), round(stop, 3)]
                    for number, start, stop in lines.placed(lyrics, times)]
        except InterruptedError:
            raise
        except Exception:
            log.warning("[yue2_comfy.edit_track] when the lines are sung could not be found, so "
                        "the window places them by the bars", exc_info=True)
            return None

    def _times_of(self, place, label: str, disk: str, waveform, rate: int, text: str, progress,
                  listened, went=None):
        """The aligner's ``(word, start, stop)`` for ``text`` in ``waveform``; None when they cannot be had.

        Read beside the song ``disk`` names when it was timed on these words
        before, and heard with the aligner in ``place`` otherwise, then kept
        there. ``label`` names the sound for this session's own memory of
        times, so a sound timed once in a run is not timed twice. Without the
        aligner at hand and with nothing on the disk, there is nothing to say.
        A sound ``went`` names is a song going on, timed as ``_went_times``
        says.
        """
        from .asr import runtime as asr_runtime
        from .inpaint import lines

        record = (went or {}).get(label)
        if record is not None:
            old = lines.heard_text(record[3])
            added = lines.added_after(old, text)
            if added is not None:
                return self._went_times(place, label, disk, waveform, rate, text, old, added,
                                        record, progress, listened, went)
        times = _times_read(disk, text) if disk else None
        if times is None and place is not None:
            folder, reader, device = place
            listened.add("aligner")
            times = asr_runtime.word_times(folder, reader, device, waveform, rate, text, label,
                                           progress, interrupted)
            if disk:
                _times_keep(disk, text, times)
        return times

    def _went_times(self, place, label: str, disk: str, waveform, rate: int, text: str, old: str,
                    added: str, record, progress, listened, went):
        """The word times of a song going on: the song before it for its old lines, the new part for the new ones.

        ``record`` is ``(label, disk, sound, lyrics, since)`` of the sound the
        song went on from and the second this take is heard from
        (``track.sings_from``); ``old`` is that sound's words and ``added`` the
        ones it went on with. The old words are where they were -- the sound
        before the new part is the old one -- and the new ones are timed on
        the sound from ``since`` alone. Given the whole song, the aligner lost
        them among the old ones: a user's new outro of four lines, its first
        repeating the old outro's, came back lit six seconds early, and a
        ballad's chorus sung again came back inside the first one. None when
        the aligner is not at hand and the times are not beside the song.
        """
        from .asr import runtime as asr_runtime
        from .inpaint import lines

        before_label, before_disk, before_sound, _lyrics, since = record
        times = _times_read(disk, text, since) if disk else None
        if times is not None:
            return times
        halves = (None, None) if progress is None else (Band(progress, 0.0, 0.5),
                                                        Band(progress, 0.5, 1.0))
        earlier = self._times_of(place, before_label, before_disk, before_sound, rate, old,
                                 halves[0], listened, went)
        if earlier is None:
            return None
        found, first = [], 0
        if added.strip():
            if place is None:
                return None
            folder, reader, device = place
            listened.add("aligner")
            first = max(0, min(int(round(since * rate)), int(waveform.shape[-1]) - rate))
            found = asr_runtime.word_times(folder, reader, device, waveform[..., first:], rate,
                                           added, (label, first), halves[1], interrupted)
        times = lines.went_on(earlier, found, first / rate)
        if disk:
            _times_keep(disk, text, times, since)
        return times

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

        settings = _edited(settings, edit)
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
            elif step.kind == "extend":
                fresh = core.extended(loaded(), song, waveform, step.start, step.score, step.lyrics,
                                      missing, settings,
                                      lambda sheet: track.extension_frames(state, step, sheet),
                                      band, interrupted)
                for take in fresh:
                    entry["takes"][take.seed] = take
            else:
                region = ops.retake(step.start, step.stop, state.frames, len(song.prefix))
                fresh = core.retakes(loaded(), song, waveform, region, missing, settings, band,
                                     interrupted, natural=entry["natural"] is None,
                                     lyrics=step.lyrics if step.kind == "words" else None,
                                     score=step.score if step.kind == "notes" else None)
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
                 shown, settings, timed=None, hears=False) -> dict:
        """Everything the track window draws.

        ``song`` is the key the result is remembered under, empty when it could
        not be remembered; a run with no edits hands back the song it opened,
        so that key is the one it came in under. ``was`` is the key of the
        song that came in, which is the song's own even when the run was
        handed one of its other names.
        ``seconds`` is the length of the result; ``peaks`` and ``rms`` its wave,
        at most ``PEAKS`` values from 0 to 1 each; ``grid`` is ``grid.layout`` of the score on
        the result, None for a song without one, and ``score`` that score's text, which the
        window opens the score editor on for a change of notes; ``lyrics`` the words it sings now,
        which the window shows beside the track and a cut takes sections out
        of; ``lines`` when each sung line of them is sung, ``[[line, start,
        stop]]`` by the line's number in ``lyrics``, or None when the song was
        not heard for it (see ``_timed``); ``edits`` the list as it was read,
        with the take kept written into every retake, and ``sung`` the
        temperature and the guide the song itself was sung with, which is where
        the window's own knobs start. ``goes_on`` is where the song would go on
        from, the bar and the second, None for a song with no score (see
        ``_goes_on``). ``hears`` says whether a retake's takes
        are heard on this machine (``_ears_at_hand``) or left to the join, so
        the window says which before anything is sung. After at least one edit there are also
        ``kind``, ``at`` (the seconds the last edit took in hand, on the song as
        it was before it), ``dropped`` (the section tags a cut took out),
        ``asked`` (the words the takes were heard against: the new words, or
        the words the song sings where a retake sings again; None when they
        were not heard), ``chosen`` and ``takes``: an entry a take, with seed,
        index, kept, sung, seconds, total, peaks, rms, grid and lines (that
        take's own, since its length moves the bars and the lines after the
        edit), join, natural, heard, said and mumbled (what was heard of the
        take, see ``_take_facts``), ended, flagged and ``audio``, the temp file of the
        whole song that take makes, None when it could not be written. A take
        the list asks for that this session has not sung is ``sung`` false and
        empty otherwise, see ``_take_gap``. After a retake or a change of words
        there is also ``before``, the song as it stood before that edit, in the
        same shape, so the list can offer it beside the takes to compare with,
        see ``_was_facts``.
        """
        import dataclasses

        from .inpaint import grid

        timed = timed or {}
        drawn = _wave(sound)
        payload = {"song": keyed, "was": name, "sung": _sampled(settings), "hears": bool(hears),
                   "seconds": round(state.frames * FRAME_SECONDS, 3), "sample_rate": rate,
                   "peaks": drawn["peaks"], "rms": drawn["rms"], "grid": None,
                   "score": state.score or "", "lyrics": state.lyrics, "lines": timed.get("song"), "takes": [],
                   "chosen": None,
                   "edits": track.written(
                       [dataclasses.replace(edit, take=pick)
                        for edit, pick in zip(wanted, picks)])}
        if state.sheet is not None and state.clock is not None:
            payload["grid"] = grid.layout(state.sheet, state.clock, state.frames)
        payload["goes_on"] = _goes_on(state)
        if shown is None:
            return payload
        step, ordered, pick, made, prior, seeds, earlier, entry = shown
        was = entry.get("was_heard")
        payload["chosen"] = pick
        payload["kind"] = step.kind
        payload["at"] = [round(step.start * FRAME_SECONDS, 3), round(step.stop * FRAME_SECONDS, 3)]
        payload["dropped"] = list(step.dropped)
        payload["asked"] = entry.get("asked")
        timings = timed.get("takes") or {}
        payload["takes"] = [
            _take_facts(take, index, pick, made, rate, prior, step, timings.get(index), was)
            if take is not None else _take_gap(seeds[index], index)
            for index, take in enumerate(ordered)]
        if step.kind != "cut":
            payload["before"] = _was_facts(earlier, rate, prior, step, made,
                                           timed.get("before"), entry.get("said"), was)
        return payload


EDIT_CLASSES = {"YuE2EditTrack": YuE2EditTrack}
EDIT_NAMES = {"YuE2EditTrack": ORIGIN}
