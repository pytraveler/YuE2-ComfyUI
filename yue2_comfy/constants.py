"""Names, sizes, defaults and tooltips. No heavy imports live here.

Everything in this module is stdlib only, so it can be imported while ComfyUI
builds its node list and while the tests run outside ComfyUI entirely.

DEFAULT_OPTIONS is the single source of truth for the options node. Every widget
default in nodes.py reads from it, and every consumer starts from a copy of it,
so an unconnected options socket is never a special case.
"""

from __future__ import annotations

import re
import sys

PACK = "YuE2-ComfyUI"
CATEGORY = "YuE2"
ADVANCED_CATEGORY = CATEGORY + "/Advanced"
"""The staged nodes sit one level down, which is what makes them optional.

Someone who opens the YuE2 menu sees three nodes and can write a song. The four
that split a run into its stages are one click further in, where they cost
nothing to the person who does not want them."""

OPTIONS_TYPE = "YUE2_OPTIONS"
PLAN_TYPE = "YUE2_PLAN"
PLANS_TYPE = "YUE2_PLANS"
LATENTS_TYPE = "YUE2_LATENTS"

SAMPLE_RATE = 48000
LATENT_DIM = 64
FRAME_SECONDS = 1920 / SAMPLE_RATE

CONTEXT = 24576
MAX_SECONDS = 360.0

AUTO_BASE_SECONDS = 12.0
AUTO_SECONDS_PER_LINE = 12.0
AUTO_MIN_SECONDS = 40.0
AUTO_INSTRUMENTAL_SECONDS = 180.0
AUTO_WORDS_PER_LINE = 8

LM_REPO = "m-a-p/YuE2-3B"
VAE_REPO = "m-a-p/YuE2-Vae"
VAE_LEGACY_REPO = "m-a-p/YuE2-Vae-legacy"
REPACK_REPO = "Comfy-Org/YuE2"
WRITER_REPO = "unsloth/Qwen3.5-4B-GGUF"

MODELS_SUBDIR = "YuE2"
LM_DIRNAME = "YuE2-3B"
VAE_DIRNAME = "YuE2-Vae"
VAE_LEGACY_DIRNAME = "YuE2-Vae-legacy"
CHECKPOINTS_SUBDIR = "checkpoints"
WRITER_SUBDIR = "LLM"

WEIGHTS_NAME = "model.safetensors"
MERGES_NAME = "qwen.tiktoken"
MANIFEST_NAME = "weights_manifest.json"
CONFIG_NAME = "config.json"

REPACK_BF16_NAME = "yue2_3b_bf16.safetensors"
REPACK_INT8_NAME = "yue2_3b_int8_convrot.safetensors"
REPACK_BF16_PATH = "checkpoints/" + REPACK_BF16_NAME
REPACK_INT8_PATH = "checkpoints/" + REPACK_INT8_NAME

WRITER_NAME = "Qwen3.5-4B-Q4_K_M.gguf"
WRITER_PATH = WRITER_NAME
WRITER_BYTES = 2740937888

LM_BYTES = 7261441640
VAE_BYTES = 530512720
VAE_MARKERS = ("decoder.layers.0.weight_g", "decoder.layers.0.weight_v",
               "decoder.layers.1.layers.0.alpha")
"""Tensor names of the stable-audio Oobleck decoder family, which m-a-p's is one of.

A bare "decoder." prefix is not enough to recognise it: every video and image
VAE on the machine has one too, and so does ACE-Step 1.5's diffusion model. These
three are the weight-norm pair of the first convolution and the first SnakeBeta
slope; a diffusers-shaped VAE with its decoder.conv_in.conv.weight has none of
them. They do not single out YuE2's decoder, though: ACE-Step 1.5's own VAE has
all three with the same shapes, and VAE_SIGNATURE below is what tells the two
apart.
"""
VAE_SIGNATURE = ("decoder.layers.8.weight_v", [2, 64, 7])
"""The tensor, and its shape, that the names above cannot tell apart from a cousin.

VAE_MARKERS are the names of any Oobleck decoder in the stable-audio layout, and
ACE-Step 1.5's VAE (ace_1.5_vae.safetensors, which ComfyUI's own ACE-Step
templates put in models/vae) has all three, with the same shapes. Its decoder
has five upsampling blocks and stops at decoder.layers.7; m-a-p's has six, and
its output convolution -- 64 channels into stereo, kernel 7 -- is layers.8.
Measured on 2026-09-18 on the released YuE2-Vae, YuE2-Vae-legacy and
ace_1.5_vae headers.

Issue #1 was the other ACE-Step file: its diffusion model names itself
"decoder", which the prefix rule before 0.7.1 took for a VAE. VAE_MARKERS
stopped that one, and this stops the VAE beside it.
"""
MERGES_BYTES = 2561218
REPACK_BF16_BYTES = 7799983228
REPACK_INT8_BYTES = 3960938800

AUDIO_ENCODERS_SUBDIR = "audio_encoders"
SHEETSAGE_NAME = "sheetsage2_bf16.safetensors"
SHEETSAGE_PATH = AUDIO_ENCODERS_SUBDIR + "/" + SHEETSAGE_NAME
SHEETSAGE_BYTES = 1386868122
SHEETSAGE_MARKERS = ("encoder.feature_extractor.mel_mean", "decoder.layernorm_embedding.weight",
                     "layer_weight")
"""Comfy-Org's SheetSage2 file, beside the YuE2 checkpoint in the same repository, and the
tensor names no YuE2 file has: the mel statistics, the decoder's embedding norm, the layer mix."""

ASR_REPO = "Qwen/Qwen3-ASR-1.7B-hf"
ASR_DIRNAME = "Qwen3-ASR-1.7B"
ASR_TOKENIZER_NAME = "tokenizer.json"
ASR_FILES = (WEIGHTS_NAME, ASR_TOKENIZER_NAME, CONFIG_NAME)
ASR_BYTES = 4076193080
ASR_MARKER = "model.multi_modal_projector.linear_2.weight"
ASR_MARKER_SHAPE = [2048, 1024]
"""Qwen's own release of the speech model that recognises sung words, kept in
models/YuE2 under its size's name. The projector into a 2048-wide language
model is what tells the 1.7B build from the 0.6B one, which has the same files."""

VOCALS_REPO = "KimberleyJSN/melbandroformer"
VOCALS_REVISION = "ac9b0614ab3cd7f77219e18ba494dfd93956c348"
VOCALS_NAME = "MelBandRoformer.ckpt"
VOCALS_BYTES = 913106900
VOCALS_KNOWN_BYTES = (912885656, 456479072)
VOCALS_MARKERS = ("band_split.to_features.0.0.gamma", "mask_estimators.0.to_freqs.0.0.0.weight",
                  "layers.0.0.layers.0.0.to_gates.weight")
"""Kimberley Jensen's Mel-Band RoFormer vocal model (MIT), the one file that separates the voice,
fetched at the revision this pack was checked against and kept in models/YuE2. The two other sizes
are kijai's safetensors conversions of the same weights, fp32 and fp16: never downloaded, but used
when a machine already has one. The markers are tensor names only this network has."""

LM_ALLOW = (WEIGHTS_NAME, MERGES_NAME, CONFIG_NAME, MANIFEST_NAME)
VAE_ALLOW = (WEIGHTS_NAME, CONFIG_NAME, MANIFEST_NAME)

COT_CHOICES = ("full", "melody", "off")
VAE_CHOICES = ("standard", "legacy")
ATTENTION_CHOICES = ("sdpa", "cudnn")
DOWNLOAD_CHOICES = ("auto", "comfy-org", "original", "off")
QUANTIZATION_CHOICES = ("bf16", "int8")
OFFLOAD_CHOICES = ("auto", "on", "off")

WRITER_AUTO = "auto"
LANGUAGE_CHOICES = (
    "auto", "English", "Russian", "Chinese", "Japanese", "Korean",
    "Spanish", "French", "German", "Italian", "Portuguese",
)

WRITER_LENGTH_LINES = {
    "short": 8,
    "normal": 16,
    "long": 24,
    "very long": 32,
}
WRITER_LENGTH_CHOICES = tuple(WRITER_LENGTH_LINES)
WRITER_LENGTH_DEFAULT = "normal"

WRITER_LINES = WRITER_LENGTH_LINES[WRITER_LENGTH_DEFAULT]
WRITER_MIN_LINES = 2
WRITER_MAX_LINES = 40
WRITER_MAX_NEW_TOKENS = 900
WRITER_CONTEXT_MARGIN = 512

WRITER_TEMPERATURE = 0.9
WRITER_TOP_P = 0.95
WRITER_TOP_K = 40
WRITER_REPETITION_PENALTY = 1.05

ABC_TEMPERATURE = 0.7
ABC_TOP_P = 0.9
ABC_TOP_K = 30
ABC_REPETITION_PENALTY = 1.005
SEMANTIC_TEMPERATURE = 1.0
SEMANTIC_TOP_P = 0.95
SEMANTIC_TOP_K = 100
SEMANTIC_REPETITION_PENALTY = 1.2

DEFAULT_OPTIONS = {
    "cot": "full",
    "cfg_scale": 0.0,
    "max_seconds": 0.0,
    "vae": "standard",
    "device": "auto",
    "keep_model_loaded": False,
    "attention_backend": "sdpa",
    "download": "auto",
    "quantization": "bf16",
    "ode_steps": 32,
    "abc_temperature": ABC_TEMPERATURE,
    "abc_top_p": ABC_TOP_P,
    "abc_top_k": ABC_TOP_K,
    "temperature": SEMANTIC_TEMPERATURE,
    "top_p": SEMANTIC_TOP_P,
    "top_k": SEMANTIC_TOP_K,
    "repetition_penalty": SEMANTIC_REPETITION_PENALTY,
    "offload": "auto",
    "transpose": 0,
    "vocals_only": False,
    "low_vram": False,
}

TRANSPOSE_LIMIT = 12
"""How far 'transpose' reaches either way, in semitones: one octave.

Any key is at most six semitones from any other, so the rest of the range only
chooses which octave the tune moves into."""

DEFAULT_IDEA = "a quiet song about coming home in winter, female voice"
DEFAULT_STYLE = "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, unhurried phrasing, 88 BPM"
DEFAULT_LYRICS = "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n\n[Chorus]\nLet the day come into view\nEvery road begins with you"

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
"""These three describe the same inputs on the plain node and on the staged ones,
so they live here rather than in either module. A song and the score it grew from
answer to one seed, and saying so twice in two wordings is how they stop agreeing."""


def install_command(package: str) -> str:
    """The pip line for *this* interpreter, ready to paste into a terminal.

    "pip install X" is not advice a ComfyUI user can follow: the portable build
    runs an embedded Python that is not on PATH, a manual install has a venv, and
    the desktop app has its own environment again. Typing the bare command
    installs into whichever Python the shell happens to find, and the node keeps
    reporting the package as missing.
    """
    executable = sys.executable or "python"
    if " " in executable:
        executable = '"' + executable + '"'
    return executable + " -m pip install " + package


def sung_lines(lyrics: str) -> int:
    """How many lines are actually sung.

    Section markers -- [Verse], [Chorus] -- are directions, not words, and the
    model sings none of them, so they do not lengthen the song. The same goes
    for a bracketed direction inside a line: a line written as
    "[Only bass] I don't need a map ... [Guitar stab]" used to be taken for a
    marker because it starts and ends with a bracket, and its words went
    uncounted.

    A line counts by its words too, one line for every AUTO_WORDS_PER_LINE
    rounded to the nearest. People paste a whole verse onto one line, and
    counted as one line such a song got a ceiling of a minute and was cut off
    mid-verse. A lyric line of up to eleven words still counts as one. Text
    written without spaces between words counts one per line, as before.
    """
    count = 0
    for line in (lyrics or "").splitlines():
        words = _DIRECTION.sub(" ", line).split()
        if not any(ch.isalnum() for word in words for ch in word):
            continue
        count += max(1, (len(words) + AUTO_WORDS_PER_LINE // 2) // AUTO_WORDS_PER_LINE)
    return count


_DIRECTION = re.compile(r"\[[^\]]*\]")


def auto_seconds(lyrics: str) -> float:
    """The ceiling to use when the user leaves max_seconds at 0.

    A ceiling can only ever cut a song short; it cannot make the model write a
    shorter one. What it buys is this: on four lines of lyrics the model will
    sometimes keep going long after the words run out, and 180 seconds of that
    is worse than an honest warning at 60.
    """
    lines = sung_lines(lyrics)
    if lines == 0:
        return AUTO_INSTRUMENTAL_SECONDS
    seconds = AUTO_BASE_SECONDS + AUTO_SECONDS_PER_LINE * lines
    return min(MAX_SECONDS, max(AUTO_MIN_SECONDS, seconds))


def length_ceiling(max_seconds, lyrics: str) -> float:
    """The most seconds a run sings: 'max_seconds', or at 0 the ceiling the lyrics give.

    The singing stage stops there however far the score runs on, and the score
    editor draws the same line across its piano roll. Both read it from here, so
    the line on the screen is where the song really ends.
    """
    requested = float(max_seconds or 0)
    return requested if requested > 0 else auto_seconds(lyrics)


def length_lines(choice) -> int:
    """How many sung lines a length word asks for.

    This widget used to be a number of seconds, and it was measured to run
    backwards over most of its travel: 0 meant twelve lines, and every value
    from one second to a hundred and fifty-five asked for fewer than that, so
    someone who wanted a longer song and typed 120 got nine lines instead of
    twelve. Twelve seconds a line is also an invention -- near enough at 88 BPM,
    wrong by half at 140 -- which made the number on the widget one nobody could
    check against the song they got. Words cannot run backwards, and they
    promise nothing the tempo is able to break.

    An unknown word is the default rather than an error: the only way to get one
    is to hand-edit a workflow, and a song is a better answer than a refusal.
    """
    return WRITER_LENGTH_LINES.get(str(choice or "").strip().lower(),
                                   WRITER_LENGTH_LINES[WRITER_LENGTH_DEFAULT])


def seconds_to_tokens(seconds: float) -> int:
    """Song length in seconds -> codec tokens, the budget the sampler wants."""
    return max(1, int(round(float(seconds) / FRAME_SECONDS)))


def tokens_to_seconds(tokens: int) -> float:
    return int(tokens) * FRAME_SECONDS


def normalize_seed(seed) -> int:
    """Fold any integer into the range SongRequest accepts, rather than raising.

    A seed can arrive from a Primitive node or a saved workflow made on another
    pack's assumptions, where 0xffffffffffffffff is the usual ceiling. Refusing
    it would be correct and useless; masking it keeps the graph running and
    still gives the same song for the same number.
    """
    return int(seed) & ((1 << 63) - 1)
