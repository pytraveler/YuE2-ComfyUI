"""YuE2 Load MIDI: a MIDI file in, a score YuE2 can sing and its lyrics out.

A song someone already has as MIDI -- from a sequencer, a karaoke collection
or a notation program -- becomes the two-voice score YuE2 sings from: the
voice from one track, the instrument line from another, and with 'full' chord
symbols guessed from what the rest play (see ``midi.score``). Wired into 'YuE2
Generate Song', the tune is sung in whatever style the style line asks for.

The file comes from ComfyUI's input folder, the way a Load Audio recording
does, and the node's own button and a file dropped on the node upload one
there. Reading a file takes milliseconds, so nothing is cached and nothing is
downloaded.

The lyrics output is a karaoke file's words under their section tags, or the
section tags alone. The score and lyrics editors sit on this node as they do
on 'YuE2 Transcribe': an edit belongs to the file it was made for -- a score
also to the mode and the tracks -- and an edit with no mark is output as it is.

The score goes out as a bare tune, without section comments, unless
'without_sections' is turned off. A file's tune comes without the words it will
be sung with, and the singing node lays the words it has along a bare tune
before singing it (see ``phrasing``): eight lines of words sung over the GTA
San Andreas intro as it stood lost their chorus, and laid along its phrases
they were heard 83 to 100 percent in order. Turned off, the score keeps the
sections the file names, or is one verse, and is sung as it arrives -- for a
karaoke file sung with its own words.
"""

from __future__ import annotations

import hashlib
import logging
import os

from . import edits, phrasing
from .constants import CATEGORY
from .midi import tracks
from .progress import NodeProgress, announce, refuse

log = logging.getLogger(__name__)

EXTENSIONS = (".mid", ".midi", ".kar", ".rmi")
MODE_CHOICES = ("melody", "full")
VOCAL_CHOICES = ["auto"] + [str(number) for number in range(1, tracks.TRACK_CHOICES + 1)]
INSTRUMENT_CHOICES = ["auto", "none"] + VOCAL_CHOICES[1:]
INPUT_ANNOTATION = " [input]"
LARGEST = 8 * 1024 * 1024
"""Bytes of MIDI the node reads. A long song is tens of kilobytes; the cap keeps a mislabelled file off the server."""

MIDI_TOOLTIP = (
    "A MIDI file in ComfyUI's input folder: .mid, .midi, .kar or .rmi. 'Choose MIDI file...' uploads one, "
    "and so does dropping a file on the node.\n\n"
    "The list on the node names the file's tracks and which of them the score takes."
)
MODE_TOOLTIP = (
    "'melody' writes the vocal and instrumental lines without chords, to be sung with 'cot' set to 'melody'.\n\n"
    "'full' adds chord symbols guessed from what the other tracks play together, for 'cot' set to 'full'. "
    "They are a guess: read them over in 'Edit score...'."
)
VOCAL_TOOLTIP = (
    "The track the voice sings, by its number in the list on the node. 'auto' takes the track the karaoke "
    "words fall on, then a track named as the voice or the melody, then the highest line that is not a bass. "
    "Chords in the track are sung as their top note."
)
INSTRUMENT_TOOLTIP = (
    "The track the score's instrument line takes, by its number in the list on the node, or 'none'. 'auto' "
    "takes a track named for it, then the busiest remaining track above G3 that is neither a bass nor mostly "
    "chords, and leaves the line empty when there is none."
)
SCORE_EDIT_TOOLTIP = (
    "The edited score kept on this node. Empty at first, which means every run writes the file's score.\n\n"
    "'Edit score...' fills it after a run. The edit belongs to the file, the mode and the tracks it was made "
    "for: with another file or another choice the node writes the score anew and says so. A score pasted "
    "before any run carries no file and is output as it is."
)
LYRICS_EDIT_TOOLTIP = (
    "The edited lyrics kept on this node. Empty at first, which means the lyrics output is the file's karaoke "
    "words under their section tags, or the section tags alone.\n\n'Edit lyrics...' fills it. An edit belongs "
    "to the file it was written for; lyrics written before any run are output as they are."
)
SECTIONS_TOOLTIP = (
    "On, as it starts: the score goes out without section comments, as a bare tune, and 'YuE2 Generate Song' "
    "lays the lyrics it sings along it -- each line on a phrase with about as many notes as the line has "
    "syllables, the tune repeated when the words outlast it, and the song ending with the words. A tune written "
    "without these words in mind is sung clearly that way when its phrases have room for them.\n\n"
    "Off: the score keeps the sections the file names -- its markers, or a karaoke file's paragraphs -- or is "
    "one verse, and is sung as it arrives. For a karaoke file sung with its own words, or lyrics written to "
    "this score's sections."
)

NO_FILE = (
    "Choose a MIDI file: put one into ComfyUI's input folder, press 'Choose MIDI file...' on the node, "
    "or drop the file on the node."
)
CANNOT_READ = "'{name}' could not be read as a MIDI file: {reason}."
NO_SCORE = "No score could be written from '{name}': {reason}."
OTHER_FILE_SCORE = (
    "The edited score on this node was made for another file, or for another 'mode' or choice of tracks. "
    "It was left out and this file's score is given instead; open 'Edit score...' to edit it, or press "
    "'Reset score'."
)
OTHER_FILE_LYRICS = (
    "The edited lyrics on this node were written for another file, so this file's own lyrics are given instead."
)


def input_folder() -> str:
    """ComfyUI's input folder; a ValueError outside ComfyUI, where there is none."""
    from . import paths

    folder_paths = paths._folder_paths()
    if folder_paths is None:
        raise ValueError("there is no ComfyUI input folder to read MIDI files from")
    return folder_paths.get_input_directory()


def midi_files() -> list:
    """The MIDI files at the top of the input folder, in the order a person reads them."""
    try:
        root = input_folder()
        names = os.listdir(root)
    except (OSError, ValueError):
        return []
    return sorted((name for name in names
                   if name.lower().endswith(EXTENSIONS) and os.path.isfile(os.path.join(root, name))),
                  key=str.lower)


def path_of(name) -> str:
    """Where a chosen file is; a ValueError says why it cannot be the file read.

    Only the input folder is read: a name that climbs out of it is refused,
    whatever is at the end of it. The ' [input]' that ComfyUI appends to some
    uploaded names is taken off.
    """
    name = str(name or "").strip()
    if name.endswith(INPUT_ANNOTATION):
        name = name[:-len(INPUT_ANNOTATION)]
    if not name:
        raise ValueError(NO_FILE)
    if not name.lower().endswith(EXTENSIONS):
        raise ValueError("'{}' is not a MIDI file: the node reads {}".format(name, ", ".join(EXTENSIONS)))
    root = os.path.realpath(input_folder())
    path = os.path.realpath(os.path.join(root, name))
    try:
        inside = os.path.commonpath([root, path]) == root
    except ValueError:
        inside = False
    if not inside:
        raise ValueError("'{}' lies outside ComfyUI's input folder".format(name))
    if not os.path.isfile(path):
        raise ValueError("'{}' is not in ComfyUI's input folder, {}".format(name, root))
    return path


def read_midi(name) -> bytes:
    """The bytes of a MIDI file in the input folder."""
    path = path_of(name)
    if os.path.getsize(path) > LARGEST:
        raise ValueError("'{}' is larger than {} MB, far more than the MIDI of any song".format(
            name, LARGEST // (1024 * 1024)))
    with open(path, "rb") as handle:
        return handle.read()


def summary(name, mode: str = "melody", vocal="auto", instrument="auto") -> dict:
    """What the list on the node shows before a run: the file's tracks, and the score's facts or why there is none.

    A file that cannot be read raises ValueError; a choice of tracks that
    cannot be sung is ``ok: false`` with the reason and the tracks, so the list
    still says what there is to choose from.
    """
    from .midi import score, smf

    song = smf.read(read_midi(name))
    try:
        out = score.convert(song, mode, vocal, instrument)
    except ValueError as error:
        return {"ok": False, "error": str(error), "parts": tracks.describe(tracks.parts(song), {})}
    return {"ok": True, "parts": out["parts"], "facts": out["facts"], "notices": out["notices"]}


class YuE2LoadMidi:
    """A MIDI file in, a score YuE2 sings and its lyrics out."""

    DESCRIPTION = (
        "Reads a MIDI file and writes its melody as a score YuE2 can sing -- the vocal line from one track, the "
        "instrumental line from another, and with 'full' chords guessed from the rest -- so a song can be sung "
        "from a file. Wire 'score_abc' into YuE2 Generate Song with 'cot' set to 'melody' (or 'full'): the words "
        "written on that node are laid along the tune's phrases. The 'lyrics' output is the karaoke words of a "
        ".kar file under their section tags, or the tags alone to write words under.\n\n"
        "Nothing is downloaded: the file is read on the spot."
    )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi": (midi_files(), {"tooltip": MIDI_TOOLTIP}),
                "mode": (list(MODE_CHOICES), {"default": MODE_CHOICES[0], "tooltip": MODE_TOOLTIP}),
                "vocal_track": (list(VOCAL_CHOICES), {"default": "auto", "tooltip": VOCAL_TOOLTIP}),
                "instrument_track": (list(INSTRUMENT_CHOICES), {"default": "auto", "tooltip": INSTRUMENT_TOOLTIP}),
            },
            "optional": {
                "score_abc": ("STRING", {"multiline": True, "default": "", "tooltip": SCORE_EDIT_TOOLTIP}),
                "lyrics": ("STRING", {"multiline": True, "default": "", "tooltip": LYRICS_EDIT_TOOLTIP}),
                "without_sections": ("BOOLEAN", {"default": True, "tooltip": SECTIONS_TOOLTIP}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("score_abc", "lyrics")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, midi, **_inputs):
        """The file's bytes, hashed: a file saved again under the same name runs again."""
        try:
            return hashlib.sha256(read_midi(midi)).hexdigest()
        except (OSError, ValueError):
            return str(midi)

    @classmethod
    def VALIDATE_INPUTS(cls, midi, **_inputs):
        """True for a MIDI file in the input folder, or the reason there is none -- before anything runs."""
        try:
            path_of(midi)
        except (OSError, ValueError) as error:
            return str(error)
        return True

    def load(self, midi, mode, vocal_track, instrument_track, score_abc="", lyrics="", without_sections=True,
             unique_id=None):
        from .midi import score, smf

        progress = NodeProgress(unique_id)
        if mode not in MODE_CHOICES:
            refuse(unique_id, "'mode' must be one of {}.".format(", ".join(MODE_CHOICES)))
        try:
            data = read_midi(midi)
            song = smf.read(data)
        except (OSError, ValueError) as error:
            refuse(unique_id, CANNOT_READ.format(name=midi, reason=str(error).rstrip(".")))
        try:
            out = score.convert(song, mode, vocal_track, instrument_track)
        except ValueError as error:
            refuse(unique_id, NO_SCORE.format(name=midi, reason=str(error).rstrip(".")))
        file_mark = edits.file_mark(data)
        marks = {chosen: edits.midi_mark(file_mark, chosen, vocal_track, instrument_track) for chosen in MODE_CHOICES}
        findings = []
        kept = edits.read(score_abc)
        if kept.score and kept.words in (None, marks[mode]):
            score_out = kept.score
        else:
            if kept.score:
                findings.append(("warn", OTHER_FILE_SCORE))
            findings.extend(("notice", text) for text in out["notices"])
            score_out = out["abc"]
        score_out = phrasing.bare(score_out) if without_sections else phrasing.labelled(score_out)
        kept_lyrics = edits.read(lyrics)
        if kept_lyrics.score and kept_lyrics.words in (None, file_mark):
            words_out = kept_lyrics.score
        else:
            if kept_lyrics.score:
                findings.append(("warn", OTHER_FILE_LYRICS))
            words_out = out["lyrics"]
        announce(unique_id, findings)
        facts = out["facts"]
        ui = {edits.SCORE_UI: [out["abc"]], edits.WORDS_UI: [marks[mode]], edits.TRACK_UI: [file_mark],
              edits.LYRICS_UI: [out["lyrics"]], edits.MARKS_UI: [marks],
              edits.MIDI_UI: [{"name": str(midi), "parts": out["parts"], "facts": facts}]}
        log.info("[yue2_comfy] %s: %d bars at %d BPM, voice track %s, instrument track %s", midi, facts["bars"],
                 facts["bpm"], facts["voice"], facts["instrument"])
        progress.finish("{} bars at {} BPM, {:.0f} seconds".format(facts["bars"], facts["bpm"], facts["seconds"]))
        return {"ui": ui, "result": (score_out, words_out)}


MIDI_CLASSES = {"YuE2LoadMidi": YuE2LoadMidi}
MIDI_NAMES = {"YuE2LoadMidi": "YuE2 Load MIDI"}
