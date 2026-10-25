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
from .progress import (Band, NodeProgress, announce, format_duration, interrupted, refuse,
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
used longest ago goes first, and is sung again if it is asked for again --
but never an edit of the list a song was last run with: that list goes over
the budget rather than sing itself again (``Results``)."""

WALKED_KEEP = 16
"""How many songs' last lists ``Results`` remembers the edits of; a list is a few names, not sound."""

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

ANOTHER_SONG = (
    "The edits on this node were made on another song ({}) than the one it has now ({}). Press "
    "'Reset track' on the node to edit this song from the start, or join the song they were "
    "made on to 'audio'."
)

NOT_A_KEY = (
    "The 'song_key' field does not hold a song key. It is written by 'Saved songs...' on the "
    "node and is not meant to be typed; clear it and pick a song again."
)

SONG_TOOLTIP = (
    "Which remembered song to edit when nothing is joined to 'audio'. Written by the 'Saved "
    "songs...' button on the node, which shows what this install has sung and what each one "
    "is, and by every run on a song joined to 'audio', so switching 'audio' off stays on that "
    "song.\n\nThe song comes back from its latents, the last stage of singing it and nothing more: "
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
    "The square beside this input on the node switches it off: the node goes on with the song it "
    "last edited, or one chosen with 'Saved songs...', and whatever sings into this input is not "
    "run for this node."
)

USE_AUDIO_TOOLTIP = (
    "Whether the song joined to 'audio' is the one edited; the square beside that input on the "
    "node is this switch. Switched off, the node goes on with the song it last edited, or one "
    "chosen with 'Saved songs...' since, and leaves the link where it is -- and the node above "
    "is not run for it at all, so a song sung before a restart is not sung again to edit one "
    "bar of it. Opening a song with 'Saved songs...' switches it off.\n\n"
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
    "A cut ignores this, and so does a move: each is the same however often it is made."
)

EDITS_TOOLTIP = (
    "The edits to make, as a JSON list, written by the track window. Empty means no edits: the "
    "node hands the song on as it is and draws the track.\n\n"
    "Each edit is {'op': 'retake', 'cut', 'words', 'notes', 'extend', 'move', 'break' or "
    "'restyle'} with either "
    "'bars': [first, stop] counting "
    "from 0, the second bar not included, or 'seconds': [from, to] for a song sung with cot "
    "'off', which has no score. A retake also takes a 'seed', how many 'takes' to sing, and "
    "which 'take' to keep. A change of words takes those too, and 'lines': [first, stop] of the "
    "words with the 'text' they become; without bars or seconds it is sung where those lines "
    "are. A change of notes, 'notes', takes a seed and takes as well, and the whole 'score' the "
    "song is to have, as the score editor leaves it: without bars it is sung over the bars whose "
    "notes that score changes, and with bars only the music of those bars is taken from it. A "
    "song that goes on, 'extend', takes a seed and takes, and the 'text' it goes on with, empty "
    "for a new ending alone; it selects no bars, going on from where the singing ends. A 'move' "
    "takes the bars of whole sections and 'to', the bar line they go to: where another section "
    "starts, or the number of bars for the end. A 'break' takes 'to', the bar line it goes "
    "before, and its 'length' in bars. A 'restyle' takes a seed and takes, the 'style' its bars "
    "are sung in, 'melody' 'same' or 'new' (whole sections), 'octave' -1, 0 or 1, and 'also', "
    "more bars sung in the same take.\n\n"
    "The window writes the list as {'song': key, 'edits': [...]}, and a list made on another "
    "song is refused."
)

OPTIONS_TOOLTIP = (
    "Optional. Only what this run decides is read from it -- 'device', 'offload', 'low_vram', "
    "'vae', 'quantization', 'attention_backend', 'download' and 'keep_model_loaded'. How the "
    "song was sung, 'cot' and the sampling among it, comes from the song itself."
)

DESCRIPTION = (
    "Changes part of a finished song and hands back the whole song; everything outside the "
    "edit stays as it was.\n\n"
    "Open 'Edit track...', select a stretch and press Retake, Cut, Notes..., Go on..., "
    "Break... or Style...; click a line of the words to sing it with other words, or drag a "
    "section to move it. Works on songs this pack has sung.\n\n"
    "Hearing the words takes Qwen3-ASR and its word aligner, and finding the bars takes the "
    "voice separator; each is downloaded the first time it is needed."
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

_VOICES = collections.OrderedDict()
"""The separated voice's loudness around the cuts of the moves made this session, by sound and stretches.

A second run of the same list separates nothing: the voice of a sound is the
same voice however often it is asked about."""

_VOICE_KEEP = 16

VOICE_CONTEXT = 1.0
"""Seconds of the song on each side of a stretch the separator hears besides it."""

VOICE_HOP = 0.005
"""The step the separated voice's loudness is taken at: five milliseconds."""


def _voice_near(sound, rate: int, windows, settings, progress):
    """The separated voice's loudness in ``windows`` of ``sound``: ``[(first second, [dB every VOICE_HOP])]``.

    Only the windows are separated, laid one after another with a second of
    silence between them, in one pass; the separator is let go afterwards
    unless the run keeps what it loads.
    """
    return _voices_in([(sound, start, stop) for start, stop in windows], rate, settings, progress)


def _voices_in(pieces, rate: int, settings, progress):
    """``_voice_near`` for ``(sound, start, stop)`` windows of several sounds at once, in one pass."""
    import torch

    from . import devices, download
    from .vocals import runtime as vocal_runtime

    path = download.ensure_vocals(settings, progress)
    parts, spans, at, gap = [], [], 0, None
    for sound, start, stop in pieces:
        samples = sound[0] if sound.dim() == 3 else sound
        samples = samples.detach().to(device="cpu", dtype=torch.float32)
        if gap is None:
            gap = torch.zeros((samples.shape[0], rate), dtype=torch.float32)
        first = max(0, int((start - VOICE_CONTEXT) * rate))
        last = min(int(samples.shape[-1]), int(math.ceil((stop + VOICE_CONTEXT) * rate)))
        if last <= first:
            continue
        parts += [samples[:, first:last], gap]
        spans.append((first, at, last - first))
        at += last - first + rate
    if not spans:
        return []
    try:
        voice = vocal_runtime.separate(path, devices.resolve(settings["device"]),
                                       torch.cat(parts, dim=-1)[None], rate,
                                       cancelled=interrupted)[0].mean(0)
    finally:
        if not settings.get("keep_model_loaded"):
            vocal_runtime.unload()
    hop = max(1, int(round(rate * VOICE_HOP)))
    found = []
    for first, placed, length in spans:
        count = length // hop
        piece = voice[placed:placed + count * hop].reshape(count, hop)
        loud = 20.0 * torch.log10(piece.pow(2).mean(dim=1).sqrt() + 1e-9)
        found.append((first / float(rate), [float(value) for value in loud]))
    return found


BREAK_AROUND = 3.0
"""Seconds on each side of a break whose singing the voice left in the break is measured against."""

BREAK_EDGE = 0.25
"""Seconds at each end of a break left out when the voice in it is measured: the seams, drawn again."""

BREAK_FLOOR = -30.0
"""The dB the singing around a break is taken to reach at the least.

A break can sit between two stretches of playing, and then there is little
singing around it to measure against; a break that is quiet by that measure
has to be quiet in dB as well, so the measure never stands on less than
this."""

BREAK_SHARE = 0.9
"""Of an edit's share of the bar, what a break's singing takes; listening for the voice left in it has the rest."""


BREAK_BLOCK = 0.1
"""Seconds the voice's loudness is taken over when a break is listened to, as the measure of 2026-09-23 took it."""


def _share(loud, first: float, spans) -> float | None:
    """The 95th percentile loudness over the ``(start, stop)`` seconds ``spans``, in blocks of ``BREAK_BLOCK``.

    ``loud`` is dB every VOICE_HOP from the second ``first``.
    """
    width = max(1, int(round(BREAK_BLOCK / VOICE_HOP)))
    found = []
    for start, stop in spans:
        low = max(0, int((start - first) / VOICE_HOP))
        high = min(len(loud), int((stop - first) / VOICE_HOP))
        for at in range(low, high - width + 1, width):
            power = sum(10.0 ** (value / 10.0) for value in loud[at:at + width]) / width
            found.append(10.0 * math.log10(power + 1e-18))
    if not found:
        return None
    found.sort()
    return found[min(len(found) - 1, int(0.95 * len(found)))]


def _voices_in_breaks(takes, start: int, rate: int, settings, progress) -> None:
    """``Take.voice`` for every take of a break put in at frame ``start``, in one pass of the separator.

    The voice in the break against the singing in the ``BREAK_AROUND``
    seconds on each side of it, never less than ``BREAK_FLOOR``; see
    ``core.VOICE_LEFT``. Left None for every take when the separator cannot
    be had -- no separator on the machine and 'download' off, say. Never
    fatal but for a cancel.
    """
    opening = start * FRAME_SECONDS
    pieces = [(take.waveform, opening - BREAK_AROUND, opening + take.count * FRAME_SECONDS + BREAK_AROUND)
              for take in takes]
    try:
        found = _voices_in(pieces, rate, settings, progress)
    except InterruptedError:
        raise
    except Exception:
        log.warning("[yue2_comfy.edit_track] the voice in the break could not be separated, so "
                    "the takes are kept by the join alone", exc_info=True)
        return
    for take, (first, loud) in zip(takes, found):
        closing = opening + take.count * FRAME_SECONDS
        inside = _share(loud, first, [(opening + BREAK_EDGE, closing - BREAK_EDGE)])
        around = _share(loud, first, [(opening - BREAK_AROUND, opening),
                                      (closing, closing + BREAK_AROUND)])
        if inside is not None:
            take.voice = inside - (BREAK_FLOOR if around is None else max(BREAK_FLOOR, around))


BREAK_VOICED = ("Every take of the break still has a voice in it, the quietest {:.0f} dB against the "
                "singing around it. Ask for more takes: the model plays some breaks with no voice and "
                "sings in others. A rap keeps its ad-libs even in its own bars without notes.")

BREAK_KEPT_VOICED = ("The take of the break kept still has a voice in it, {:.0f} dB against the "
                     "singing around it, and take {} has none.")

BREAK_UNHEARD = ("The takes of the break were not listened to for a voice left in them -- the voice "
                 "separator is not on this machine and 'download' is off -- so the one whose join "
                 "the model scored best was kept.")


def _break_notices(takes, pick: int) -> list:
    """What to say about the takes of a break once one was kept: a voice left in it, or no ear to tell."""
    from .inpaint import core

    sung = [take for take in takes if take is not None]
    kept = takes[pick]
    if kept is None or not sung:
        return []
    heard = [getattr(take, "voice", None) for take in takes]
    if all(voice is None for voice in heard):
        return [("notice", BREAK_UNHEARD)]
    if not core.voiced(kept):
        return []
    clean = [index for index, (take, voice) in enumerate(zip(takes, heard))
             if take is not None and voice is not None and not core.voiced(take)]
    if clean:
        return [("warn", BREAK_KEPT_VOICED.format(heard[pick], clean[0] + 1))]
    return [("warn", BREAK_VOICED.format(min(voice for voice in heard if voice is not None)))]


def _voice_level(sound, rate: int, windows, settings, label: str, progress):
    """``level(second)``: how loud the separated voice is within ``grid.QUIET_WINDOW`` of ``second``, in dB.

    None when the voice cannot be had -- no separator on the machine and
    'download' off, say -- and the move then cuts where the score says.
    Never fatal but for a cancel.
    """
    key = (label, tuple((round(start, 3), round(stop, 3)) for start, stop in windows))
    found = _VOICES.get(key)
    if found is None:
        try:
            found = _voice_near(sound, rate, windows, settings, progress)
        except InterruptedError:
            raise
        except Exception:
            log.warning("[yue2_comfy.edit_track] the voice around the move could not be separated, "
                        "so it cuts where the score says", exc_info=True)
            return None
        _VOICES[key] = found
        while len(_VOICES) > _VOICE_KEEP:
            _VOICES.popitem(last=False)
    if not found:
        return None

    def level(second):
        for first, loud in found:
            low = int((second - grid.QUIET_WINDOW - first) / VOICE_HOP)
            high = int((second + grid.QUIET_WINDOW - first) / VOICE_HOP) + 1
            if low >= 0 and high <= len(loud):
                return max(loud[low:high])
        return 0.0

    return level


WORDS_SUFFIX = ".words.json"
"""Where the word times of a song wait for the next session, beside the song they were measured on.

Timing a song reads 1.84 GB of weights and separates its voice; the answer
never changes, a song being remembered under the key of its own samples, and
what it is worth is exactly the second and third line somebody rewrites. The
words they were measured on are written with them, so a song whose words an
edit has changed is timed again rather than read wrong. A song that went on
writes the second its new lines were timed from as well (``_went_times``),
and one timed whole before that is timed again. A song heard a section at a
time says so (``BY_SECTIONS``), and one that can be heard so and was not is
timed again: until 0.13.0 every song was heard in one pass, which put 68 line
starts in 100 on their phrases against 94 (``inpaint.lines``)."""

BY_SECTIONS = "sections"
"""What a file of word times says under "heard" when its song was heard a section at a time."""

_PARTS = collections.OrderedDict()
"""When the words of a section are sung, counted from where its stretch opens, by the samples of that stretch and its lines.

A take is the song it was made on with a few seconds laid into it
(``core.laid``), so every section its edit did not reach is in it sample for
sample, moved by whole frames at most, and is not heard twice. What that
saves is the separator: on 2026-10-12 it took 7 seconds on a song of 193 and
11 on one of 312, where the aligner took 0.3 and 0.7 over all of either."""

_PARTS_KEEP = 512

PART_SLACK = 0.1
"""Seconds a word read beside a song may lie outside its section's stretch: a step of the aligner's clock and a little."""

PART_UNREAD = 0.2
"""The share of a section's time stamps out of order past which it is heard again with the section before it.

A section's stretch opens where its score says its singing does, and the
model does not always sing it there: with a phrase to spare at the end of a
chorus it sang the next verse's first line on it, 6.7 seconds ahead of the
verse, in all four songs sung on one ballad's score for the words-under-notes
stand. The verse's stretch then holds three of its four lines, the aligner is
given four, and it says so itself: 23 to 54 stamps in 100 came back out of
order (``asr.aligner.Times``), where the 118 sections of the stand's 28
songs had 7 in 100 at the median and nine sections in ten under 16. Heard in
one stretch with the chorus before, the verse came back with 0 to 14 in 100
and its first line where it is sung.

Measured on 2026-10-12 with no score in it -- Qwen3-ASR told the language,
hearing each line of the 28 songs where the times put it, 2300 words: 1731
of them heard for one pass over the whole song as mixed, 1889 for every
section alone, 1912 with this, and 1915 with every section heard twice, once
with each neighbour, which put 12 fewer lines on their phrases. Five sections
were heard again, the four verses and a last section of a kept song, and each
gained words."""

PART_BETTER = 0.1
"""How much smaller that share has to be when a section is heard with the one before it for those times to be taken.

Four more sections passed ``PART_UNREAD`` on the stand and were left as they
were. Three read worse with the section before, two of them in a song whose
words ran so far ahead that no hearing found them; the fourth read 6 stamps
in 100 better and was heard singing 5 of its 16 words that way where it sang
8 alone."""

_WHOLE_VOICES = collections.OrderedDict()
"""The separated voice of the songs whose grids were measured this session, by song key: mono at the aligner's 16 kHz.

Measuring a grid separates the whole voice, and the song's words are timed
on that voice straight after; kept so, a song of five minutes is 19 MB."""

_WHOLE_KEEP = 2

SEPARATING_SHARE = 0.8
"""Of the bar's share for timing a song's words, what separating the voice takes; the aligner has the rest."""

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
    """The aligner's folder, the file it reads words with, the device and 'low_vram', when it can light the words; None otherwise.

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
        return (folder, download.aligner_tokenizer(folder, {"download": "off"}), device,
                bool(settings.get("low_vram")))
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


def _ears_away(settings) -> None:
    """With 'low_vram', let both listeners go before the song model sings, kept or not. Never fatal.

    The song model makes its room from what is free on the card and does not
    send anything else away, and the word aligner is loaded first in a run, to
    light the words: on a small card its 1.75 GiB would sit beside the song
    model's 3.1 while it sings. Either comes back in a second or two when it
    is next asked.
    """
    if not settings.get("low_vram"):
        return
    try:
        from .asr import runtime as asr_runtime

        asr_runtime.unload()
        asr_runtime.unload_aligner()
    except Exception:
        log.debug("[yue2_comfy.edit_track] the listeners could not make way for the song model",
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
    _WHOLE_VOICES.clear()


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


def _one_take(step) -> bool:
    """Whether an edit sings nothing new: a cut or a move, one take with no seed."""
    from .inpaint import track

    return step.kind in track.ONE_TAKE


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
    """The takes of every edit sung in this session, the ones used longest ago dropped first.

    A run walks its list from the first edit, so an edit of the list a song
    was last run with is never dropped to make room: under plain
    least-recently-used a list whose takes outgrew the budget dropped its
    first edit for its last, the next run sang the first again and dropped the
    second for it, and so on down the list -- every run sang every edit
    (issue #13). A run holds what its song's last list walked and what it has
    walked itself (``walked``, ``put``), and once it is through (``settle``)
    the rest goes oldest first. If that is still past the budget, the edits
    before the last keep only the take that was kept: the others were there
    to compare, and comparing is over once another edit stands on that one.
    Their seeds are written down, so a run does not sing them again for a
    pick already made, and the window offers them as it offers the takes
    left after a restart. Past that the list holds every kept take over the
    budget, which a run singing all of it again would cost more than.
    """

    def __init__(self, budget: int = CACHE_BYTES):
        self._held = collections.OrderedDict()
        self._sizes = {}
        self._budget = int(budget)
        self._bytes = 0
        self._walked = collections.OrderedDict()

    def get(self, name: str):
        found = self._held.get(name)
        if found is not None:
            self._held.move_to_end(name)
        return found

    def put(self, name: str, entry, keep=()) -> None:
        """Hold ``entry`` under ``name``, measured now, dropping the oldest entries not named in ``keep``.

        An entry grown in place is measured again here.
        """
        if name in self._held:
            self._held.pop(name)
            self._bytes -= self._sizes.pop(name)
        self._held[name] = entry
        self._sizes[name] = _held_bytes(entry)
        self._bytes += self._sizes[name]
        self._fit(set(keep) | {name})

    def walked(self, song: str) -> frozenset:
        """The names of the edits the last run on ``song`` walked through: what a run on it must not drop."""
        return frozenset(self._walked.get(song, ()))

    def settle(self, song: str, names) -> None:
        """Write down the list a run on ``song`` walked, ``names`` in order, and fit the rest around it."""
        names = list(names)
        self._walked.pop(song, None)
        self._walked[song] = tuple(names)
        while len(self._walked) > WALKED_KEEP:
            self._walked.popitem(last=False)
        self._fit(set(names))
        for name in names[:-1]:
            if self._bytes <= self._budget:
                break
            self._trim(name)

    def _fit(self, keep) -> None:
        for name in [name for name in self._held if name not in keep]:
            if self._bytes <= self._budget:
                break
            self._held.pop(name)
            self._bytes -= self._sizes.pop(name)

    def _trim(self, name: str) -> None:
        """Let go of every take of one edit but the one kept, writing their seeds down as dropped."""
        entry = self._held.get(name)
        if entry is None or entry.get("kept") not in entry["takes"]:
            return
        others = [seed for seed in entry["takes"] if seed != entry["kept"]]
        if not others:
            return
        for seed in others:
            del entry["takes"][seed]
        entry["dropped"] = set(entry.get("dropped", ())) | set(others)
        self._bytes -= self._sizes[name]
        self._sizes[name] = _held_bytes(entry)
        self._bytes += self._sizes[name]

    def clear(self) -> None:
        self._held.clear()
        self._sizes.clear()
        self._bytes = 0
        self._walked.clear()

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
    _voice_keep(name, voice, rate)
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


def _measuring(settings, unique_id, path):
    """How a restyle finds the bars of a song it sang in the new style: ``clock(song, waveform, sheet, bars, progress)``.

    As a song's own grid is found (``_grid_of``): by its separated voice,
    with the separator's weights at ``path``. The song is sung only a little
    past the bars it is for, so its score is read as far as ``bars`` for the
    measuring and the grid then carries the whole score's bar starts. Nothing
    is kept: each take is a song of its own.
    """
    import dataclasses

    from .vocals_only import voice_of

    def clock(song, waveform, sheet, bars, progress):
        rate = int(song.sample_rate)
        heard = sheet if bars >= len(sheet["bars"]) else notation.read(notation.head(song.score,
                                                                                     bars))
        voice = voice_of({"waveform": waveform, "sample_rate": rate}, settings, unique_id, progress,
                         path=path)["waveform"]
        began = time.perf_counter()
        found = grid.measured(heard, waveform, rate, voice=voice)
        log.info("[yue2_comfy.edit_track] the song sung in the new style follows its score from "
                 "%.2f s at %.4f of its tempo, placed by %s, in %.1f s", found.offset, found.rate,
                 "the voice" if found.by_voice else "the mix", time.perf_counter() - began)
        return dataclasses.replace(found, starts=grid.starts_of(sheet))

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


def _times_read(name: str, text: str, since=None, sections: bool = False):
    """The word times measured for this song in some other session, or None. Never fatal.

    ``since`` is the second a song that went on has its new lines timed from,
    and times read for it must say they were measured so: a song timed whole
    before it was known to lose those lines is timed again. ``sections`` says
    the song can be heard a section at a time, and times read for it must say
    they were: the ones kept before that were heard in one pass over the
    whole song, which loses its place in a long one (``lines.sections``).
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
        if sections and kept.get("heard") != BY_SECTIONS:
            return None
        return [(str(word), float(start), float(stop)) for word, start, stop in kept["times"]]
    except (OSError, ValueError, KeyError, TypeError):
        log.debug("[yue2_comfy.edit_track] %s cannot be read and the words are timed again", path,
                  exc_info=True)
        return None


def _times_keep(name: str, text: str, times, since=None, sections: bool = False) -> None:
    """Write the word times beside their song. Never fatal: they only save time.

    ``since`` is written when the new lines were timed from there, and
    ``sections`` when the song was heard a section at a time.
    """
    path = _words_file(name)
    if path is None:
        return
    kept = {"words": _said(text),
            "times": [[word, round(float(start), 3), round(float(stop), 3)]
                      for word, start, stop in times]}
    if since is not None:
        kept["from"] = round(float(since), 3)
    if sections:
        kept["heard"] = BY_SECTIONS
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as handle:
            json.dump(kept, handle)
    except OSError:
        log.debug("[yue2_comfy.edit_track] the word times could not be written to %s", path,
                  exc_info=True)


def _band(progress, low: float, high: float):
    """A share of a bar that may not be there: None stays None."""
    return None if progress is None else Band(progress, low, high)


def _plain(sound):
    """A sound as ``[channels, samples]`` float32 on the CPU, which is how its sections are cut and known."""
    import torch

    samples = sound[0] if sound.dim() == 3 else sound
    return samples.detach().to(device="cpu", dtype=torch.float32)


def _windows_of(state, text: str):
    """``[(first frame, stop frame, lines)]`` for the sections the words ``text`` of a song are heard in, or None when it is heard whole.

    ``lines.sections`` on the song as ``state`` describes it, each stretch
    on whole frames: an edit moves what comes after it by whole frames, so a
    section it did not reach is the same samples in a take as in the song
    the take was made on, and is known by them (``_PARTS``). A stretch opens
    on the frame after where ``lines.sections`` has it, never before: the
    earlier it opens, the more often its first word is put on the playing
    (``lines.SECTION_EARLY``). None for a song with no score, for one whose
    lyrics cannot be paired with its score's sections, and when the
    sections' lines are not ``text``. None too for a song whose bars the
    separated voice did not place (``grid.Grid.by_voice``): the pitch
    classes of the mix alone put one song's bars 14 seconds off, and
    sections heard on such bars would be heard in the wrong seconds, where
    one pass over the song asks nothing of them.
    """
    from .inpaint import lines

    if state is None or state.sheet is None or state.clock is None or not str(text or "").strip():
        return None
    if not state.clock.by_voice:
        return None
    laid = grid.layout(state.sheet, state.clock, state.frames)["sections"]
    found = lines.sections(state.lyrics, laid, state.frames * FRAME_SECONDS)
    if not found or chr(10).join(said for _start, _stop, said in found) != text:
        return None
    return [(min(int(state.frames), int(math.ceil(start / FRAME_SECONDS - 1e-6))),
             min(int(state.frames), int(math.ceil(stop / FRAME_SECONDS - 1e-6))), said)
            for start, stop, said in found]


def _marks(samples, rate: int, windows) -> list:
    """``[first sample, stop sample, lines, key]`` for each of ``windows``; the key names the stretch by its samples and its lines."""
    import hashlib

    hop = rate * FRAME_SECONDS
    found = []
    for first, stop, said in windows:
        low = int(round(first * hop))
        high = max(low, min(int(samples.shape[-1]), int(round(stop * hop))))
        digest = hashlib.blake2b(digest_size=16)
        digest.update(samples[..., low:high].contiguous().numpy().data)
        digest.update(said.encode("utf-8"))
        found.append([low, high, said, digest.hexdigest()])
    return found


def _unread(times, first: int = 0, count=None) -> float:
    """The share of the time stamps of ``count`` words from word ``first`` that came back out of order.

    0.0 for times that do not say (``asr.aligner.Times.lost``), which is
    every one read from a file or handed over by something else.
    """
    lost = list(getattr(times, "lost", None) or ())
    chunk = lost[first:] if count is None else lost[first:first + count]
    return sum(chunk) / (2.0 * len(chunk)) if chunk else 0.0


def _pair_key(marks, index: int) -> str:
    """The name the times of section ``index`` are kept by when it was heard with the section before it: both stretches' keys."""
    return marks[index - 1][3] + "+" + marks[index][3]


def _part_keep(key: str, times) -> list:
    """Keep the word times of one section, counted from where its stretch opens, and hand them back."""
    kept = [(str(word), float(start), float(stop)) for word, start, stop in times]
    _PARTS[key] = kept
    _PARTS.move_to_end(key)
    while len(_PARTS) > _PARTS_KEEP:
        _PARTS.popitem(last=False)
    return kept


def _parts_learn(sound, rate: int, windows, times) -> None:
    """Remember a song's sections by their sound, from the word times read beside it.

    A take sung on the song this session then hears only the sections its
    edit reached. A section whose words begin in the stretch before its own
    was heard with that one (``_heard_after``) and is remembered by both. One
    whose words lie in neither is left out and heard when it is asked for:
    a grid measured again may have moved it. Never fatal.
    """
    try:
        at = 0
        marks = _marks(_plain(sound), rate, windows)
        for index, (low, high, said, key) in enumerate(marks):
            count = len(said.split())
            chunk = times[at:at + count]
            at += count
            opens, closes = low / float(rate), high / float(rate)
            if len(chunk) != count or any(float(stop) > closes + PART_SLACK
                                          for _word, _start, stop in chunk):
                continue
            first = min(float(start) for _word, start, _stop in chunk)
            if first < opens - PART_SLACK:
                if not index or first < marks[index - 1][0] / float(rate) - PART_SLACK:
                    continue
                key = _pair_key(marks, index)
            _part_keep(key, [(word, float(start) - opens, float(stop) - opens)
                             for word, start, stop in chunk])
    except Exception:
        log.debug("[yue2_comfy.edit_track] the sections of a song timed before were not "
                  "remembered", exc_info=True)


def _voice_keep(name: str, voice, rate: int) -> None:
    """Keep the separated voice a song's grid was measured by, for the timing of its words. Never fatal."""
    try:
        from .asr import model

        _WHOLE_VOICES[name] = model.mono_16k(voice, rate)
        _WHOLE_VOICES.move_to_end(name)
        while len(_WHOLE_VOICES) > _WHOLE_KEEP:
            _WHOLE_VOICES.popitem(last=False)
    except Exception:
        log.debug("[yue2_comfy.edit_track] the separated voice was not kept", exc_info=True)


def _voice_parts(samples, rate: int, spans, settings, progress) -> list:
    """The separated voice of the ``(first sample, stop sample)`` stretches of ``samples``, mono, in one pass.

    Each stretch is separated with ``VOICE_CONTEXT`` of the song on each side
    of it and a second of silence before the next, as ``_voices_in`` lays
    them, and comes back without that margin. The separator is let go
    afterwards unless the run keeps what it loads. Raises what looking for
    the separator raises when it is not on the machine.
    """
    import torch

    from . import devices, download
    from .vocals import runtime as vocal_runtime

    path = download.ensure_vocals(settings, progress)
    around = int(VOICE_CONTEXT * rate)
    gap = torch.zeros((samples.shape[0], rate), dtype=torch.float32)
    parts, placed, at = [], [], 0
    for low, high in spans:
        first, last = max(0, low - around), min(int(samples.shape[-1]), high + around)
        parts += [samples[:, first:last], gap]
        placed.append((at + low - first, high - low))
        at += last - first + rate
    try:
        voice = vocal_runtime.separate(path, devices.resolve(settings["device"]),
                                       torch.cat(parts, dim=-1)[None], rate, progress=progress,
                                       cancelled=interrupted)[0].mean(0)
    finally:
        if not settings.get("keep_model_loaded"):
            vocal_runtime.unload()
    return [voice[start:start + length] for start, length in placed]


def _voices_of(name: str, samples, rate: int, spans, settings, progress) -> list:
    """What to hear each ``(first sample, stop sample)`` stretch of a song in: ``[(samples, rate)]``, its voice where that can be had.

    The voice the song's grid was measured by, when ``name`` is that song
    and the voice is still held (``_WHOLE_VOICES``); otherwise the stretches
    are separated now. Without the separator -- not on the machine and
    'download' off, or no settings to find it by -- each stretch is the song
    as it is mixed, which the measurement in ``lines`` found right ten lines
    in a hundred less often. Never fatal but for a cancel.
    """
    from . import download

    whole = _WHOLE_VOICES.get(name) if name else None
    if whole is not None:
        from .asr import prompt

        per = prompt.SAMPLE_RATE / float(rate)
        if abs(int(whole.shape[-1]) - int(samples.shape[-1]) * per) <= prompt.SAMPLE_RATE:
            return [(whole[int(low * per):int(math.ceil(high * per))], prompt.SAMPLE_RATE)
                    for low, high in spans]
    voiced = None
    if settings is not None:
        try:
            voiced = _voice_parts(samples, rate, spans, settings, progress)
        except InterruptedError:
            raise
        except (FileNotFoundError, download.DownloadError):
            log.info("[yue2_comfy.edit_track] the voice separator is not on this machine, so the "
                     "words are timed on the song as it is mixed")
        except Exception:
            log.warning("[yue2_comfy.edit_track] the voice could not be separated, so the words "
                        "are timed on the song as it is mixed", exc_info=True)
    if voiced is None:
        return [(samples[..., low:high], rate) for low, high in spans]
    return [(clip, rate) for clip in voiced]


def _heard_after(place, name: str, samples, rate: int, marks, index: int, settings, alone: float):
    """The times of section ``index`` heard in one stretch with the section before it, or None when that reads no better.

    Counted from where the section's own stretch opens, so a line sung ahead
    of it has times below zero. ``alone`` is the share of its stamps that
    came back out of order when it was heard alone; see ``PART_UNREAD`` and
    ``PART_BETTER``.
    """
    from .asr import runtime as asr_runtime

    folder, reader, device, low_vram = place
    low, high = marks[index - 1][0], marks[index][1]
    before, said = marks[index - 1][2], marks[index][2]
    clip, clip_rate = _voices_of(name, samples, rate, [(low, high)], settings, None)[0]
    times = asr_runtime.word_times(folder, reader, device, clip, clip_rate, before + chr(10) + said,
                                   ("sections", _pair_key(marks, index)), None, interrupted,
                                   low_vram=low_vram)
    ahead, count = len(before.split()), len(said.split())
    if len(times) != ahead + count:
        return None
    together = _unread(times, ahead, count)
    if together > alone - PART_BETTER:
        return None
    log.info("[yue2_comfy.edit_track] a section came back with %.0f%% of its time stamps out of "
             "order and %.0f%% heard with the section before it, so it is timed there",
             100.0 * alone, 100.0 * together)
    shift = (low - marks[index][0]) / float(rate)
    return [(word, shift + start, shift + stop) for word, start, stop in times[ahead:]]


def _heard_by_sections(place, name: str, label: str, sound, rate: int, windows, settings,
                       progress, listened):
    """One ``(word, start, stop)`` for each word of a song, its sections heard one at a time; None when one cannot be.

    ``windows`` is what ``_windows_of`` found for the song. A section whose
    stretch was heard before -- in this sound or in the one it was made on
    -- is not heard again (``_PARTS``); the rest have their voice separated
    (``_voices_of``) and are given to the aligner in ``place`` with their own
    lines. One that comes back badly out of order is heard again with the
    section before it (``PART_UNREAD``, ``_heard_after``), and is then known
    by both stretches. None when a section has to be heard and the aligner
    is not at hand.
    """
    from .asr import runtime as asr_runtime

    samples = _plain(sound)
    marks = _marks(samples, rate, windows)
    known = {}
    for index, (_low, _high, _said, key) in enumerate(marks):
        for name_of in ([key] if not index else [key, _pair_key(marks, index)]):
            found = _PARTS.get(name_of)
            if found is not None:
                _PARTS.move_to_end(name_of)
                known[index] = found
                break
    missing = [index for index in range(len(marks)) if index not in known]
    if missing:
        if place is None:
            return None
        folder, reader, device, low_vram = place
        began = time.perf_counter()
        clips = _voices_of(name, samples, rate, [(marks[index][0], marks[index][1])
                                                 for index in missing],
                           settings, _band(progress, 0.0, SEPARATING_SHARE))
        listened.add("aligner")
        for at, index in enumerate(missing):
            _low, _high, said, key = marks[index]
            clip, clip_rate = clips[at]
            share = (1.0 - SEPARATING_SHARE) / len(missing)
            times = asr_runtime.word_times(
                folder, reader, device, clip, clip_rate, said, ("section", key),
                _band(progress, SEPARATING_SHARE + share * at, SEPARATING_SHARE + share * (at + 1)),
                interrupted, low_vram=low_vram)
            alone = _unread(times)
            together = None
            if index and alone >= PART_UNREAD:
                together = _heard_after(place, name, samples, rate, marks, index, settings, alone)
            known[index] = (_part_keep(key, times) if together is None
                            else _part_keep(_pair_key(marks, index), together))
        log.info("[yue2_comfy.edit_track] %d of the %d sections of %s heard for their words in "
                 "%.1f s", len(missing), len(marks), label[:8], time.perf_counter() - began)
    found = []
    for index, (low, _high, _said, _key) in enumerate(marks):
        opens = low / float(rate)
        found.extend((word, opens + start, opens + stop) for word, start, stop in known[index])
    return found


def _after(prior, step, take):
    """The song one take of an edit makes of ``prior``, or None when that cannot be worked out.

    Its bars and its length are the take's own, which is where its sections
    are heard; without them it is heard whole.
    """
    try:
        return track.after(prior, step, take.count,
                           take.song.score if step.kind in track.WRITTEN else None,
                           getattr(take, "sung", ()), getattr(take, "laid", None))
    except Exception:
        log.debug("[yue2_comfy.edit_track] the bars of a take could not be worked out, so it is "
                  "heard whole", exc_info=True)
        return None


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
    from .inpaint import core, grid, track

    entry = _write_wave(take.waveform, rate, "{}_{}.wav".format(name[:16], take.seed))
    drawn = _wave(take.waveform)
    own = track.after(prior, step, take.count,
                      take.song.score if step.kind in track.WRITTEN else None,
                      getattr(take, "sung", ()), getattr(take, "laid", None))
    laid = None
    if own.sheet is not None and own.clock is not None:
        laid = grid.layout(own.sheet, own.clock, own.frames)
    heard = getattr(take, "heard", None)
    voice = getattr(take, "voice", None)
    return {"seed": int(take.seed), "index": index, "kept": index == chosen, "sung": True,
            "voice": None if voice is None else round(voice, 1),
            "voiced": core.voiced(take),
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
            "voice": None, "voiced": False,
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


def _breaks(state) -> list:
    """Where the window may put a break before each section, as ``track.break_places`` gives it; never fatal."""
    try:
        return track.break_places(state)
    except (ValueError, IndexError, KeyError):
        log.debug("[yue2_comfy.edit_track] no break places for this song", exc_info=True)
        return []


def _moved(marks, step, count: int, sung=()) -> list:
    """Where the marks of the edits already made sit in the song after this one.

    An edit that put ``count`` frames where ``step.start`` to ``step.stop``
    were moves everything after it by the difference, and swallows whatever
    stood inside it. A move carries every mark along with the piece of the
    song it starts in, and then past the bars before its seams that were
    sung again, ``sung`` (``track.shifted``).
    """
    if step.kind == "move":
        carried = []
        for mark in marks:
            at = 0
            for start, stop in step.pieces:
                if start <= mark["start"] < stop:
                    carried.append(dict(mark, start=track.shifted(at + mark["start"] - start, sung)))
                    break
                at += stop - start
        return carried
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
            "voice": None, "voiced": False, "seconds": None, "total": None, "peaks": [], "rms": [], "grid": None, "lines": None,
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


SAID_STYLE = 60
"""How much of a song's style a refusal quotes to say which song it means."""


def _song_said(name: str, song=None) -> str:
    """A song as a person knows it in a refusal: its length, when it was made, and the start of its style.

    The key means nothing to anybody. The songs of one evening often share
    a style word for word, and two of them measured on 2026-10-18 were 3:43
    and 3:44 long, so the time each was made is what tells them apart, as
    it is in 'Saved songs...'. A song no longer remembered says so.
    """
    if song is None:
        try:
            song = songs.store().get(name)
        except ValueError:
            song = None
    if song is None:
        return "one no longer remembered"
    style = " ".join(str(song.style or "").split())
    if len(style) > SAID_STYLE:
        style = style[:SAID_STYLE].rstrip() + "..."
    made = (", made " + time.strftime("%Y-%m-%d %H:%M", time.localtime(song.created))
            if song.created else "")
    return "{}{}, '{}'".format(format_duration(song.frames * FRAME_SECONDS), made,
                               style or "no style given")


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

        A list that names the song it was made on (``track.song_of``) is
        refused on any other: its bars belong to that song.
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
        made_on = track.song_of(edits)
        if wanted and made_on:
            try:
                made_on = songs.canonical(made_on)
            except ValueError:
                refuse(unique_id, track.NOT_A_LIST)
            if made_on != name:
                refuse(unique_id, ANOTHER_SONG.format(_song_said(made_on), _song_said(name, song)))
        settings = _settings(song, options, unique_id)
        loras = adapters(settings, unique_id)
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
                _ears_away(settings)
                held = contextlib.ExitStack()
                stack.callback(held.close)
                models[0] = held.enter_context(session(settings, unique_id, progress, loras))
                singing[0] = held
            return models[0]

        def released():
            if singing[0] is not None:
                held, singing[0], models[0] = singing[0], None, None
                held.close()

        history, notices, picks, shown, marks = [], [], [], None, []
        clock, state, current, sound = None, None, song, waveform
        keyed, timed, ears, place = name, {}, None, None
        went, walked, held = {}, [], RESULTS.walked(name)
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
                        if step.kind == "move":
                            heard_by = opens + (closes - opens) * TIMES_SHARE
                            level = _voice_level(
                                sound, rate, grid.move_windows(state.sheet, state.clock,
                                                               grid.seams(state.sheet),
                                                               step.stretches),
                                settings, track.sound_name(name, history),
                                Band(progress, opens, heard_by))
                            opens = heard_by
                            if level is not None:
                                step = track.plan(state, edit, span, level)
                    except ValueError as error:
                        raise _Refused("Edit {}: {}".format(index + 1, error))
                    notices.extend(step.notices)
                    made = track.name(name, history, edit)
                    walked.append(made)
                    asked = ""
                    if step.kind in ("words", "extend"):
                        asked = chr(10).join(step.now)
                    elif step.kind in ("retake", "notes", "restyle") and ears is not None:
                        found_by = opens + (closes - opens) * TIMES_SHARE
                        asked = self._words_there(place, state, step, sound, rate, name, history,
                                                  Band(progress, opens, found_by), listened, went,
                                                  settings)
                        opens = found_by
                    hears = closes - (closes - opens) * HEARD_SHARE if asked else closes
                    entry = self._sung(loaded, current, sound, state, step, edit, made,
                                       settings, Band(progress, opens, hears), unique_id,
                                       held | set(walked))
                    seeds = (0,) if _one_take(step) else edit.seeds()
                    if asked:
                        self._heard(entry, seeds, step, sound, rate, settings, made, released,
                                    Band(progress, hears, closes), notices, listened, asked,
                                    ears if step.kind in ("retake", "notes", "restyle") else None)
                    ordered = [entry["takes"].get(seed) for seed in seeds]
                    pick = edit.take if edit.take is not None else core.best(ordered)
                    kept = ordered[pick]
                    entry["kept"] = kept.seed
                    history.append((edit, kept.seed))
                    picks.append(pick)
                    prior, earlier = state, sound
                    edge = _edge(step, state.frames)
                    state = track.after(state, step, kept.count,
                                        kept.song.score if step.kind in track.WRITTEN else None,
                                        getattr(kept, "sung", ()), getattr(kept, "laid", None))
                    current, sound = kept.song, kept.waveform
                    if edge and edit.fade:
                        sound = _faded(sound, edit.fade, edge == "head", rate)
                    marks = _moved(marks, step, kept.count, getattr(kept, "sung", ()))
                    landed = (track.placed(step, getattr(kept, "sung", ()))
                              if step.kind == "move" else (step.start, step.start + kept.count))
                    marks.append({"op": step.kind, "start": landed[0],
                                  "count": landed[1] - landed[0],
                                  "bars": list(edit.bars) if edit.bars else None,
                                  "seed": None if _one_take(step) else int(kept.seed),
                                  "took": round((step.stop - step.start) * FRAME_SECONDS, 3),
                                  "was": "\n".join(step.was), "now": "\n".join(step.now)})
                    shown = (step, ordered, pick, made, prior, seeds, earlier, entry)
                    before = {"label": track.sound_name(name, history[:-1]),
                              "disk": name if len(history) == 1 else "", "sound": earlier,
                              "lyrics": prior.lyrics, "state": prior}
                    if step.kind == "extend":
                        for take in ordered:
                            if take is not None and take.song is not None:
                                went[track.sound_name(name, history[:-1] + [(edit, take.seed)])] = (
                                    dict(before, since=track.sings_from(
                                        prior, step, notation.read(take.song.score))))
                    if step.kind == "move":
                        went[track.sound_name(name, history)] = dict(before, pieces=[
                            (start * FRAME_SECONDS, stop * FRAME_SECONDS)
                            for start, stop in step.pieces], sung=[
                            (start * FRAME_SECONDS, stop * FRAME_SECONDS, count * FRAME_SECONDS)
                            for start, stop, count in getattr(kept, "sung", ())])
                    if step.kind == "break":
                        at = step.start * FRAME_SECONDS
                        for take in ordered:
                            if take is not None:
                                went[track.sound_name(name, history[:-1] + [(edit, take.seed)])] = (
                                    dict(before, pieces=[(0.0, at), (at, prior.frames * FRAME_SECONDS)],
                                         sung=[(at, at, take.count * FRAME_SECONDS)]))
                        notices.extend(_break_notices(ordered, pick))
                if walked:
                    RESULTS.settle(name, walked)
                released()
                if wanted:
                    keyed = _remember(sound, _stamped(current, sound, name, song, marks))
                elif not given:
                    _also(sound, song, name)
                if place is None and "aligner" in listened:
                    place = _aligner_at_hand(settings)
                if place is not None:
                    timed = self._timed(place, name, keyed, history, shown, sound, rate, state,
                                        Band(progress, top, 1.0), listened, went, settings)
        except InterruptedError:
            translate_interrupt()
            raise
        except _Refused as stop:
            refuse(unique_id, str(stop))

        payload = self._payload(name, keyed, sound, rate, state, wanted, picks, shown,
                                settings, timed, ears is not None and place is not None, given,
                                current.style)
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
        when each is sung. It hears the song a section at a time
        (``_heard_by_sections``), the answer is kept beside the song, and the
        region opens at the last word of the line before --
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
        times = self._times_of(None, sound, saved, waveform, rate, text, progress, listened, went,
                               state, settings)
        if times is None:
            try:
                folder = download.ensure_aligner(settings, progress)
                reader = download.aligner_tokenizer(folder, settings, progress)
            except (FileNotFoundError, download.DownloadError) as error:
                raise _Refused(str(error))
            began = time.perf_counter()
            times = self._times_of((folder, reader, devices.resolve(settings["device"]),
                                    bool(settings.get("low_vram"))), sound,
                                   saved, waveform, rate, text, progress, listened, went,
                                   state, settings)
            log.info("[yue2_comfy.edit_track] %d words timed in %.1f s", len(times),
                     time.perf_counter() - began)
        try:
            start, stop = lines.region(state.lyrics, times, edit.lines[0], edit.lines[1])
        except ValueError as error:
            raise _Refused("{}: {}".format(where, error))
        return (int(round(start / FRAME_SECONDS)),
                state.frames if stop is None else int(round(stop / FRAME_SECONDS)))

    def _words_there(self, place, state, step, sound, rate: int, name: str, history, progress,
                     listened, went=None, settings=None) -> str:
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
                                   listened, went, state, settings)
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
                                       progress=progress, low_vram=bool(settings.get("low_vram")))
        except InterruptedError:
            raise
        except Exception as error:
            log.warning("[yue2_comfy.edit_track] the takes could not be heard", exc_info=True)
            _said_once(notices, "notice", DEAF.format(str(error).rstrip(".") or "no reason given"))
            return
        entry["said"] = answers[0]["text"]
        entry["asked"] = asked
        if step.kind in ("retake", "notes", "restyle"):
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
               progress, listened, went=None, settings=None) -> dict:
        """When each sung line is sung, in every sound the window can put on the track.

        ``{"song": lines, "takes": {index: lines}, "before": lines}``, where
        lines are ``[[line, start, stop]]`` in seconds, numbered as the lines
        of the words the window shows, and None where they could not be found
        -- the window then places them by the bars, as it did before there was
        anything better. The takes are the last edit's, each a song of its own
        length, the one kept being the song itself. The song as it was comes
        numbered as the lines after the edit, which a change of words may have
        moved (``lines.carried``). A song this run made go on, and each of its
        takes, is timed as ``_went_times`` says. Each sound goes with the
        song it is -- its own bars and length, a take's being its own
        (``_after``) -- which is what says where its sections are heard.
        """
        from .inpaint import lines

        found = {"song": None, "takes": {}, "before": None}
        jobs = [("song", None, sound, state.lyrics, track.sound_name(name, history), keyed, state)]
        if shown is not None:
            step, ordered, pick, earlier, prior = shown[0], shown[1], shown[2], shown[6], shown[4]
            edit = history[-1][0]
            for index, take in enumerate(ordered):
                if take is not None and index != pick:
                    jobs.append(("takes", index, take.waveform, state.lyrics,
                                 track.sound_name(name, history[:-1] + [(edit, take.seed)]), "",
                                 _after(prior, step, take)))
            if not _one_take(step):
                jobs.append(("before", None, earlier, prior.lyrics,
                             track.sound_name(name, history[:-1]),
                             name if len(history) == 1 else "", prior))
        for at, (where, index, waveform, lyrics, label, disk, song) in enumerate(jobs):
            spans = self._lines_of(place, label, disk, waveform, rate, lyrics,
                                   Band(progress, at / len(jobs), (at + 1) / len(jobs)), listened,
                                   went, song, settings)
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
                  progress, listened, went=None, state=None, settings=None):
        """``[[line, start, stop]]`` for every sung line of ``lyrics`` in ``waveform``, or None.

        ``label`` names the sound for this session and ``disk`` the song whose
        file keeps its times between sessions -- "" for a take that no song is
        remembered by. ``state`` is the song that sound is, for hearing it a
        section at a time. Never fatal but for a cancel: the lines are a
        picture, and a song that cannot be timed is still the song.
        """
        from .inpaint import lines

        text = lines.heard_text(lyrics)
        if not text.strip():
            return []
        try:
            times = self._times_of(place, label, disk, waveform, rate, text, progress, listened,
                                   went, state, settings)
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
                  listened, went=None, state=None, settings=None):
        """The aligner's ``(word, start, stop)`` for ``text`` in ``waveform``; None when they cannot be had.

        Read beside the song ``disk`` names when it was timed on these words
        before, and heard with the aligner in ``place`` otherwise, then kept
        there. ``label`` names the sound for this session's own memory of
        times, so a sound timed once in a run is not timed twice. Without the
        aligner at hand and with nothing on the disk, there is nothing to say.

        ``state`` is the song the sound is. With its score and its bars the
        song is heard a section at a time (``_windows_of``,
        ``_heard_by_sections``), with the separator ``settings`` finds; a
        song with no score, with lyrics its score's sections cannot be
        paired with, or with bars the voice did not place, is heard in one
        pass as it is mixed.

        A sound ``went`` names is a song going on, timed as ``_went_times``
        says, or a song whose sections moved, whose words keep the times they
        had and go with their sound (``lines.carried_along``), stretched over
        the bars before its seams that were sung again (``lines.stretched``).
        """
        from .asr import runtime as asr_runtime
        from .inpaint import lines

        windows = _windows_of(state, text)
        record = (went or {}).get(label)
        if record is not None and "pieces" in record:
            said = lines.heard_text(record["lyrics"])
            earlier = self._times_of(place, record["label"], record["disk"], record["sound"], rate,
                                     said, progress, listened, went, record.get("state"), settings)
            times = None if earlier is None else lines.carried_along(earlier, said,
                                                                     record["pieces"], text)
            if times is not None:
                times = lines.stretched(times, record.get("sung", ()))
                if disk:
                    _times_keep(disk, text, times, sections=windows is not None)
                return times
        elif record is not None:
            old = lines.heard_text(record["lyrics"])
            added = lines.added_after(old, text)
            if added is not None:
                return self._went_times(place, label, disk, waveform, rate, text, old, added,
                                        record, progress, listened, went, windows is not None,
                                        settings)
        times = _times_read(disk, text, sections=windows is not None) if disk else None
        if times is not None and windows is not None:
            _parts_learn(waveform, rate, windows, times)
        elif times is None and windows is not None:
            times = _heard_by_sections(place, disk, label, waveform, rate, windows, settings,
                                       progress, listened)
            if times is not None and disk:
                _times_keep(disk, text, times, sections=True)
        elif times is None and place is not None:
            folder, reader, device, low_vram = place
            listened.add("aligner")
            times = asr_runtime.word_times(folder, reader, device, waveform, rate, text, label,
                                           progress, interrupted, low_vram=low_vram)
            if disk:
                _times_keep(disk, text, times)
        return times

    def _went_times(self, place, label: str, disk: str, waveform, rate: int, text: str, old: str,
                    added: str, record, progress, listened, went, sections: bool = False,
                    settings=None):
        """The word times of a song going on: the song before it for its old lines, the new part for the new ones.

        ``record`` holds the ``label``, ``disk``, ``sound`` and ``lyrics`` of
        the sound the song went on from and the second this take is heard from,
        ``since`` (``track.sings_from``); ``old`` is that sound's words and ``added`` the
        ones it went on with. The old words are where they were -- the sound
        before the new part is the old one -- and the new ones are timed on
        the sound from ``since`` alone. Given the whole song, the aligner lost
        them among the old ones: a user's new outro of four lines, its first
        repeating the old outro's, came back lit six seconds early, and a
        ballad's chorus sung again came back inside the first one. None when
        the aligner is not at hand and the times are not beside the song.
        ``sections`` says the song can be heard a section at a time, as the
        song before it then was, and is what its times are kept and read by.
        """
        from .asr import runtime as asr_runtime
        from .inpaint import lines

        since = record["since"]
        times = _times_read(disk, text, since, sections) if disk else None
        if times is not None:
            return times
        halves = (None, None) if progress is None else (Band(progress, 0.0, 0.5),
                                                        Band(progress, 0.5, 1.0))
        earlier = self._times_of(place, record["label"], record["disk"], record["sound"], rate,
                                 old, halves[0], listened, went, record.get("state"), settings)
        if earlier is None:
            return None
        found, first = [], 0
        if added.strip():
            if place is None:
                return None
            folder, reader, device, low_vram = place
            listened.add("aligner")
            first = max(0, min(int(round(since * rate)), int(waveform.shape[-1]) - rate))
            found = asr_runtime.word_times(folder, reader, device, waveform[..., first:], rate,
                                           added, (label, first), halves[1], interrupted,
                                           low_vram=low_vram)
        times = lines.went_on(earlier, found, first / rate)
        if disk:
            _times_keep(disk, text, times, since, sections)
        return times

    def _sung(self, loaded, song, waveform, state, step, edit, name, settings, band,
              unique_id=None, keep=()):
        """The takes of one edit: the ones already sung under this name, and the ones still missing.

        Only the takes the result needs are sung. An edit that already says
        which take was kept needs that one: the others are there to compare,
        and comparing them is over. This is what a workflow saved with a
        chosen take costs when it is opened again -- the list still asks for
        the four takes it once compared, while the takes themselves live in
        this session's memory and a restart empties it. Their seeds are
        counted on from the edit's own, so any of them can be sung later and
        comes out the same take; the window offers them. An edit with nothing
        kept yet sings all of them, because the pick is made among them --
        all but the ones let go of to make room after the pick was made
        (``Results``), which the pick never needs again. ``keep`` names what
        the cache must not drop for these takes.
        """
        from .inpaint import core, ops

        settings = _edited(settings, edit)
        entry = RESULTS.get(name) or {"takes": {}, "natural": None}
        seeds = (0,) if _one_take(step) else edit.seeds()
        if not _one_take(step) and edit.take is not None:
            seeds = (seeds[edit.take],)
        dropped = ()
        if edit.take is None and entry.get("kept") in seeds:
            dropped = entry.get("dropped", ())
        missing = [seed for seed in seeds if seed not in entry["takes"] and seed not in dropped]
        if not missing:
            band.ratio(1.0)
            return entry
        try:
            if step.kind == "cut":
                region = ops.cut(step.start, step.stop, state.frames)
                entry["takes"][0] = core.cut(loaded(), song, waveform, region, step.lyrics,
                                             step.score, settings, band, interrupted)
            elif step.kind == "move":
                entry["takes"][0] = core.moved(loaded(), song, waveform, step.pieces,
                                               step.stop - step.start, step.lyrics, step.score,
                                               settings, band, interrupted, step.sung, step.pulse)
            elif step.kind == "extend":
                fresh = core.extended(loaded(), song, waveform, step.start, step.score, step.lyrics,
                                      missing, settings,
                                      lambda sheet: track.extension_frames(state, step, sheet),
                                      band, interrupted)
                for take in fresh:
                    entry["takes"][take.seed] = take
            elif step.kind == "restyle":
                from .vocals_only import separator_weights

                path = separator_weights(settings, unique_id, band)
                fresh = core.restyled(loaded(), song, waveform, state, step, edit.style,
                                      edit.octave, missing, settings,
                                      _measuring(settings, unique_id, path), band, interrupted)
                for take in fresh:
                    entry["takes"][take.seed] = take
            elif step.kind == "break":
                first, stop = step.bars
                fresh = core.broke(loaded(), song, waveform, step.start, step.score, step.lyrics,
                                   stop - first, missing, settings,
                                   lambda written: notation.interluded(state.score, written, first,
                                                                       stop - first, step.seam),
                                   lambda sheet: track.break_frames(state, step, sheet),
                                   Band(band, 0.0, BREAK_SHARE), interrupted)
                _voices_in_breaks(fresh, step.start, int(song.sample_rate), settings,
                                  Band(band, BREAK_SHARE, 1.0))
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
        if entry.get("dropped"):
            entry["dropped"] = set(entry["dropped"]) - set(entry["takes"])
        RESULTS.put(name, entry, keep)
        return entry

    def _payload(self, name: str, keyed: str, sound, rate: int, state, wanted, picks,
                 shown, settings, timed=None, hears=False, joined=False, style="") -> dict:
        """Everything the track window draws.

        ``song`` is the key the result is remembered under, empty when it could
        not be remembered; a run with no edits hands back the song it opened,
        so that key is the one it came in under. ``was`` is the key of the
        song that came in, which is the song's own even when the run was
        handed one of its other names. ``joined`` says it came in through
        'audio' rather than by 'song_key', and then the window writes ``was``
        into 'song_key': the list of edits is a list on that song, so with the
        input switched off the next run goes on with it. Until 2026-10-18 the
        field was written only by 'Saved songs...', and a user who edited a
        song from 'Load Audio' and then switched the input off was handed a
        song picked days before, with their edits laid on it.
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
        ``_goes_on``), and ``breaks`` where a break may go in before each
        section (see ``track.break_places``). ``hears`` says whether a retake's takes
        are heard on this machine (``_ears_at_hand``) or left to the join, so
        the window says which before anything is sung. After at least one edit there are also
        ``kind``, ``at`` (the seconds the last edit took in hand, on the song as
        it was before it), ``placed`` (for a move, the seconds the bars it
        moved lie at now), ``dropped`` (the section tags a cut took out),
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
        payload = {"song": keyed, "was": name, "joined": bool(joined), "style": style or "",
                   "sung": _sampled(settings), "hears": bool(hears),
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
        payload["breaks"] = _breaks(state)
        if shown is None:
            return payload
        step, ordered, pick, made, prior, seeds, earlier, entry = shown
        was = entry.get("was_heard")
        payload["chosen"] = pick
        payload["kind"] = step.kind
        payload["at"] = [round(step.start * FRAME_SECONDS, 3), round(step.stop * FRAME_SECONDS, 3)]
        if step.kind == "move":
            payload["placed"] = [round(frame * FRAME_SECONDS, 3) for frame in
                                 track.placed(step, getattr(ordered[pick], "sung", ()))]
        elif step.kind == "break" and ordered[pick] is not None:
            payload["placed"] = [round(step.start * FRAME_SECONDS, 3),
                                 round((step.start + ordered[pick].count) * FRAME_SECONDS, 3)]
        payload["dropped"] = list(step.dropped)
        payload["asked"] = entry.get("asked")
        timings = timed.get("takes") or {}
        payload["takes"] = [
            _take_facts(take, index, pick, made, rate, prior, step, timings.get(index), was)
            if take is not None else _take_gap(seeds[index], index)
            for index, take in enumerate(ordered)]
        if not _one_take(step):
            payload["before"] = _was_facts(earlier, rate, prior, step, made,
                                           timed.get("before"), entry.get("said"), was)
        return payload


EDIT_CLASSES = {"YuE2EditTrack": YuE2EditTrack}
EDIT_NAMES = {"YuE2EditTrack": ORIGIN}
