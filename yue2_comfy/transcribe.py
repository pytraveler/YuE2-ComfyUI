"""YuE2 Transcribe: a recording in, a score YuE2 can sing and its lyrics out.

This is how a song is covered. SheetSage2 listens to the recording and writes
what it hears -- the vocal line, the instrumental line, beats, key, sections
and, when asked, chords -- in the same two-voice score YuE2 sings from. Wired
into 'YuE2 Generate Song' with 'cot' at 'melody', the melody is kept and the
style, voice and instruments are whatever the style line says.

The lyrics output is the section tags of the transcription. With
'lyrics_auto_recognition' on, the words sung under them are recognised too:
Qwen3-ASR hears the whole recording, its words are cut into the sections (see
``asr.runtime``), and a language model breaks them into lines under a guard
that keeps the words as heard (see ``asr.layout``). Transcription and
recognition are kept per recording, so a new seed only lays the words out
again.

The score and lyrics editors sit on this node as they do on the song node, and
an edit is kept for the recording it was made on; an edit with no mark,
written before the node ever ran, is output as it is, the way the song node
sings one.

Nothing heavy is imported at module scope.
"""

from __future__ import annotations

import hashlib
import logging

from . import edits
from .constants import CATEGORY, OPTIONS_TYPE
from .progress import NodeProgress, announce, interrupted, refuse, translate_interrupt
from .staged import resolve

log = logging.getLogger(__name__)

MODE_CHOICES = ("melody", "full")
LAYOUT_TOKENS = 2048
LAYOUT_TEMPERATURE = 0.3
LAYOUT_TOP_P = 0.9

AUDIO_TOOLTIP = (
    "The recording to transcribe: a song with a singer works best. 'Load Audio' gives one.\n\n"
    "Only the first recording of a batch is used."
)
MODE_TOOLTIP = (
    "'melody' writes the vocal and instrumental lines without chords -- what a cover wants, "
    "sung with 'cot' set to 'melody' so the accompaniment follows the new style.\n\n"
    "'full' adds the chords it hears, for 'cot' set to 'full'. Switching between the two "
    "reuses the transcription; nothing is heard again."
)
RECOGNITION_TOOLTIP = (
    "Also recognise the words that are sung, and lay them out under the section tags, a line to a phrase.\n\n"
    "The first time, this downloads Qwen3-ASR-1.7B, a 3.8 GB speech model (Apache-2.0), into models/YuE2. "
    "The lines are laid out by the language model in 'model'; with none on this machine, the writer's "
    "own (2.7 GB) is downloaded too, as YuE2 Write Song does. "
    "Recognised words are close but not exact: read them through in 'Edit lyrics...' before singing them."
)
MODEL_TOOLTIP = (
    "The language model that breaks recognised words into lines. Used only with "
    "'lyrics_auto_recognition' on; the same list as on YuE2 Write Song, and 'auto' downloads the "
    "writer's own model (2.7 GB) when the machine has none. A section whose words "
    "it changes is broken into lines at its punctuation instead."
)
SEED_TOOLTIP = (
    "Changes only how the recognised words are broken into lines. The transcription and the words "
    "are the same whatever the seed, and a new seed does not listen to the recording again."
)
SCORE_EDIT_TOOLTIP = (
    "The edited score kept on this node. Empty at first, which means every run writes the "
    "transcription.\n\n'Edit score...' fills it after a run. The edit belongs to the recording "
    "and the mode it was made for: with another recording the node transcribes anew and says so. "
    "A score pasted before any run carries no recording and is output as it is."
)
LYRICS_EDIT_TOOLTIP = (
    "The edited lyrics kept on this node. Empty at first, which means the lyrics output is the "
    "section tags of the transcription, with the recognised words under them when "
    "'lyrics_auto_recognition' is on.\n\n'Edit lyrics...' fills it. An edit belongs to the "
    "recording it was written for; lyrics written before any run are output as they are."
)

OTHER_TRACK_SCORE = (
    "The edited score on this node was made for another recording, or for the other 'mode'. "
    "It was left out and this recording was transcribed anew; open 'Edit score...' to edit "
    "the new score, or press 'Reset score'."
)
OTHER_TRACK_LYRICS = (
    "The edited lyrics on this node were written for another recording, so this recording's "
    "own lyrics are given instead."
)
NO_SCORE = (
    "SheetSage2 listened to the recording but could not write a score from it: {reason}. "
    "That usually means it found no steady beat or no key -- a spoken or unpitched track, "
    "or one much shorter than a bar."
)
NO_WORDS = "No sung words were heard in this recording, so the lyrics are the section tags alone."
LAYOUT_FAILED = (
    "The language model could not lay the words out ({reason}), so they were broken into lines "
    "at their punctuation."
)
LAYOUT_CHANGED = (
    "The language model changed words in {changed} of {total} sections; those were broken into "
    "lines at their punctuation instead, with the words as heard."
)
BATCH_NOTE = "The audio input holds {count} recordings; only the first is transcribed."
CUT_SHORT = (
    "SheetSage2 ran out of room in {parts} of the recording's {total} parts: it writes at most {tokens} "
    "tokens for a part, and those filled before the part's end, so the last seconds of them may be "
    "missing from the score."
)
CANNOT_HEAR = "SheetSage2 could not transcribe this recording: {reason}."
CANNOT_RECOGNISE = "Qwen3-ASR could not hear this recording: {reason}."
WRITER_FETCHED = (
    "No language model was on this machine to lay the words out in lines, so the writer's own, "
    "{name} ({size}), is downloaded into the ComfyUI models folder, as YuE2 Write Song does on first use."
)


def track_of(audio) -> dict:
    """The first recording of a ComfyUI AUDIO value: samples, rate, how many there were, and its mark.

    The mark is taken from the samples as float32 and dropped with them: an
    hour of stereo is a gigabyte, not something to hold for the run.
    """
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("the audio input is not audio: connect 'Load Audio' or another audio output")
    batch = audio["waveform"]
    samples = batch[0] if batch.dim() == 3 else batch
    rate = int(audio["sample_rate"])
    mark = edits.audio_mark(samples.detach().float().cpu().contiguous().numpy(), rate)
    return {"samples": samples, "rate": rate, "count": int(batch.shape[0]) if batch.dim() == 3 else 1,
            "mark": mark, "seconds": samples.shape[-1] / float(rate)}


def _sections(score: str) -> list:
    from .sheetsage import sections

    try:
        return sections.sections(score)
    except ValueError:
        return []


class Band:
    """A share of the node's bar: a fraction reported here fills only ``low`` to ``high`` of it.

    A download or the language model reports counts against a total of its
    own; those are scaled into the share too, so the bar never runs ahead of
    the stage it is in.
    """

    def __init__(self, progress, low: float, high: float):
        self._progress = progress
        self._low = low
        self._high = high
        self._total = 1.0

    def ratio(self, fraction: float, text=None) -> None:
        self._progress.ratio(self._low + (self._high - self._low) * max(0.0, min(1.0, fraction)), text)

    def set_total(self, total: float) -> None:
        self._total = max(float(total), 1.0)

    def update(self, value: float, text=None) -> None:
        self.ratio(float(value) / self._total, text)

    def __getattr__(self, name):
        return getattr(self._progress, name)


class YuE2Transcribe:
    """A recording in, a cover-ready score and its lyrics out."""

    DESCRIPTION = (
        "Listens to a recording and writes its melody as a score YuE2 can sing -- the vocal and "
        "instrumental lines, and with 'full' the chords too -- so the song can be covered in "
        "another style. Wire 'score_abc' into YuE2 Generate Song with 'cot' set to 'melody', and "
        "write the words under the section tags on 'lyrics', or turn on 'lyrics_auto_recognition' "
        "to have them recognised.\n\n"
        "The transcription model, SheetSage2, is downloaded on first use (1.29 GB); its weights "
        "are licensed CC BY-NC 4.0, like YuE2's. Recognising the words downloads Qwen3-ASR-1.7B "
        "(3.8 GB, Apache-2.0) the first time it is turned on."
    )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        from . import llm

        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": AUDIO_TOOLTIP}),
                "mode": (list(MODE_CHOICES), {"default": MODE_CHOICES[0], "tooltip": MODE_TOOLTIP}),
                "lyrics_auto_recognition": ("BOOLEAN", {"default": False, "tooltip": RECOGNITION_TOOLTIP}),
                "model": (llm.choices(), {"tooltip": MODEL_TOOLTIP}),
                "seed": ("INT", {"default": 831001, "min": 0, "max": (1 << 63) - 1,
                                 "control_after_generate": "fixed", "tooltip": SEED_TOOLTIP}),
            },
            "optional": {
                "options": (OPTIONS_TYPE,),
                "score_abc": ("STRING", {"multiline": True, "default": "", "tooltip": SCORE_EDIT_TOOLTIP}),
                "lyrics": ("STRING", {"multiline": True, "default": "", "tooltip": LYRICS_EDIT_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("score_abc", "lyrics")
    FUNCTION = "transcribe"
    CATEGORY = CATEGORY

    def transcribe(self, audio, mode, lyrics_auto_recognition, model, seed, options=None,
                   score_abc="", lyrics="", unique_id=None):
        progress = NodeProgress(unique_id)
        settings = resolve(options)
        if mode not in MODE_CHOICES:
            refuse(unique_id, "'mode' must be one of {}.".format(", ".join(MODE_CHOICES)))
        try:
            track = track_of(audio)
        except (ValueError, AttributeError, TypeError) as error:
            refuse(unique_id, str(error))
        findings = []
        if track["count"] > 1:
            findings.append(("notice", BATCH_NOTE.format(count=track["count"])))
        recording = track["mark"]
        score_mark = edits.track_mark(recording, mode)

        kept = edits.read(score_abc)
        keep_score = bool(kept.score) and kept.words in (None, score_mark)
        heard = None
        if not keep_score or lyrics_auto_recognition:
            band = Band(progress, 0.0, 0.4) if lyrics_auto_recognition else progress
            heard = self._heard(track, recording, settings, band, unique_id)
            if heard.get("cut_short"):
                from .sheetsage import vocab

                findings.append(("notice", CUT_SHORT.format(parts=len(heard["cut_short"]), total=len(heard["windows"]),
                                                            tokens=vocab.MAX_TOKENS)))
        written = None
        if keep_score:
            score = kept.score
        else:
            if kept.score:
                findings.append(("warn", OTHER_TRACK_SCORE))
            written = self._written(heard, mode, unique_id)
            score = written

        from .sheetsage import sections

        found = _sections(score)
        own = sections.skeleton(found)
        if lyrics_auto_recognition:
            own = self._recognised(track, recording, heard, settings, model, seed, progress, unique_id,
                                   findings) or own
        kept_lyrics = edits.read(lyrics)
        if kept_lyrics.score and kept_lyrics.words in (None, recording):
            words_out = kept_lyrics.score
        else:
            if kept_lyrics.score:
                findings.append(("warn", OTHER_TRACK_LYRICS))
            words_out = own
        announce(unique_id, findings)
        ui = {edits.WORDS_UI: [score_mark], edits.TRACK_UI: [recording], edits.LYRICS_UI: [own],
              edits.MARKS_UI: [{chosen: edits.track_mark(recording, chosen) for chosen in MODE_CHOICES}]}
        if written is not None:
            ui[edits.SCORE_UI] = [written]
        progress.finish("{} sections, {:.0f} seconds".format(len(found), track["seconds"]))
        return {"ui": ui, "result": (score, words_out)}

    def _heard(self, track, recording, settings, progress, unique_id) -> dict:
        """SheetSage2's transcription of the recording, from the cache when it has been heard before."""
        from . import devices, download
        from .sheetsage import runtime

        try:
            settings["device"] = devices.validate(settings["device"])
        except (ValueError, RuntimeError) as error:
            refuse(unique_id, str(error))
        try:
            path = download.ensure_sheetsage(settings, progress)
        except (FileNotFoundError, download.DownloadError) as error:
            refuse(unique_id, str(error))
        try:
            return runtime.transcribe(path, devices.resolve(settings["device"]), track["samples"],
                                      track["rate"], key=(recording, runtime.stamp(path)),
                                      progress=progress, cancelled=interrupted)
        except InterruptedError:
            translate_interrupt()
            raise
        except ValueError as error:
            refuse(unique_id, CANNOT_HEAR.format(reason=str(error).rstrip(".")))
        finally:
            if not settings["keep_model_loaded"]:
                runtime.unload()

    def _written(self, heard, mode, unique_id) -> str:
        from .sheetsage import abc_rebuild, events

        try:
            return abc_rebuild.build(events.score_rows(heard["events"], heard["seconds"]),
                                     melody_only=(mode == "melody"))
        except ValueError as error:
            refuse(unique_id, NO_SCORE.format(reason=str(error).rstrip(".")))

    def _recognised(self, track, recording, heard, settings, model, seed, progress, unique_id, findings) -> str:
        """The recognised words laid out under the section tags, or "" when none were heard."""
        from . import devices, download
        from .asr import runtime as asr_runtime
        from .sheetsage import events, sections

        try:
            timed = sections.timed(events.score_rows(heard["events"], heard["seconds"]), heard["seconds"])
        except ValueError:
            timed = [{"label": "", "tag": "Verse", "start": 0.0, "end": float(heard["seconds"]), "notes": 1}]
        band = Band(progress, 0.4, 0.8)
        try:
            folder = download.ensure_asr(settings, band)
        except (FileNotFoundError, download.DownloadError) as error:
            refuse(unique_id, str(error))
        try:
            words = asr_runtime.recognise(folder, devices.resolve(settings["device"]), track["samples"],
                                          track["rate"], timed, key=(recording, asr_runtime.stamp(folder)),
                                          progress=band, cancelled=interrupted)
        except InterruptedError:
            translate_interrupt()
            raise
        except ValueError as error:
            refuse(unique_id, CANNOT_RECOGNISE.format(reason=str(error).rstrip(".")))
        finally:
            if not settings["keep_model_loaded"]:
                asr_runtime.unload()
        pieces = [{"tag": section["tag"], "text": part} for section, part in zip(timed, words["parts"])]
        wanted = sum(1 for piece in pieces if piece["text"].strip())
        if not wanted:
            findings.append(("notice", NO_WORDS))
            return ""
        answer = self._layout(pieces, words["language"], model, seed, settings, Band(progress, 0.8, 1.0), findings)
        from .asr import layout

        lyrics, kept = layout.lay_out(pieces, answer)
        if answer and kept < wanted:
            findings.append(("notice", LAYOUT_CHANGED.format(changed=wanted - kept, total=wanted)))
        return lyrics

    def _layout(self, pieces, language, model, seed, settings, progress, findings) -> str:
        """The language model's answer for these words, kept per words, model and seed; "" when there is none."""
        from . import llm, writer
        from .asr import layout, runtime as asr_runtime

        chat = layout.messages(pieces, language)
        if (model or llm.WRITER_AUTO).strip() == llm.WRITER_AUTO and not llm.catalogue() \
                and settings.get("download", "auto") != "off":
            from . import download
            from .constants import WRITER_BYTES, WRITER_NAME

            findings.append(("notice", WRITER_FETCHED.format(name=WRITER_NAME, size=download.human_size(WRITER_BYTES))))
        try:
            path = llm.resolve(model, settings, progress)
            digest = hashlib.sha256(repr(chat).encode("utf-8")).hexdigest()
            progress.ratio(0.0, "Laying the words out in lines")
            return asr_runtime.laid_out(
                (digest, path, int(seed)),
                lambda: llm.run(path, chat, int(seed), writer.context_needed(chat, LAYOUT_TOKENS),
                                settings["device"], keep_loaded=True, progress=progress, settings=settings,
                                greedy=False, max_new_tokens=LAYOUT_TOKENS, temperature=LAYOUT_TEMPERATURE,
                                top_p=LAYOUT_TOP_P))
        except Exception as error:
            log.warning("[yue2_comfy] laying the recognised words out failed", exc_info=True)
            findings.append(("notice", LAYOUT_FAILED.format(reason=str(error).strip().splitlines()[0][:200]
                                                             if str(error).strip() else type(error).__name__)))
            return ""
        finally:
            if not settings["keep_model_loaded"]:
                llm.unload()


TRANSCRIBE_CLASSES = {"YuE2Transcribe": YuE2Transcribe}
TRANSCRIBE_NAMES = {"YuE2Transcribe": "YuE2 Transcribe"}
