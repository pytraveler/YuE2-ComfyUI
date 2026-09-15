"""The request Qwen3-ASR answers, and reading its answer back. Standard library only.

A request is a short chat: an empty system turn, a user turn holding the
audio, and the assistant turn left open. The audio is a run of placeholder
tokens, one for every frame the audio encoder gives, which the network
replaces with those frames. Left to itself the model answers
``language Russian<asr_text>`` and then the words; naming the language puts
that much of the answer in the request, and the model writes only the words.

The token ids are those of the released tokenizer, and the special tokens
split the text exactly where they stand, so the fixed parts are tokenized
around the placeholders rather than with thousands of them inside.
"""

from __future__ import annotations

SAMPLE_RATE = 16000
HOP = 160
MIN_SAMPLES = 8000
"""Shorter audio is padded with silence to half a second before it is heard."""
CHUNK_FRAMES = 100
FRAMES_PER_CHUNK_OUT = 13

END_OF_TEXT = 151643
IM_START = 151644
IM_END = 151645
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_PAD = 151676
STOP_TOKENS = (END_OF_TEXT, IM_END)
MARKER = "<asr_text>"

LANGUAGES = {
    "ar": "Arabic", "yue": "Cantonese", "zh": "Chinese", "cs": "Czech", "da": "Danish", "nl": "Dutch",
    "en": "English", "fil": "Filipino", "fi": "Finnish", "fr": "French", "de": "German", "el": "Greek",
    "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "mk": "Macedonian", "ms": "Malay", "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "es": "Spanish", "sv": "Swedish", "th": "Thai", "tr": "Turkish", "vi": "Vietnamese",
}
"""The languages the model was trained to name, by code; the name is what goes in the request."""

REPEAT_LIMIT = 20
LONGEST_REPEAT = 20


def language_name(language) -> str:
    """The name the model knows for a code or a name, or "" for none; ValueError for one it does not know."""
    if language is None or str(language).strip() in ("", "auto"):
        return ""
    text = str(language).strip()
    if text.lower() in LANGUAGES:
        return LANGUAGES[text.lower()]
    for name in LANGUAGES.values():
        if name.lower() == text.lower():
            return name
    raise ValueError("Qwen3-ASR does not know the language '{}'".format(text))


def padded_samples(samples: int) -> int:
    return max(int(samples), MIN_SAMPLES)


def mel_frames(samples: int) -> int:
    """Mel frames for this many samples at 16 kHz: one per hop, the frame past the end dropped."""
    return padded_samples(samples) // HOP


def _halved_three_times(frames: int) -> int:
    for _ in range(3):
        frames = (frames - 1) // 2 + 1 if frames > 0 else 0
    return frames


def chunk_lengths(frames: int) -> list:
    """Frames the encoder gives for each 100-frame chunk: 13 for a whole one, fewer for the last."""
    chunks = -(-frames // CHUNK_FRAMES)
    return [_halved_three_times(min(CHUNK_FRAMES, frames - index * CHUNK_FRAMES)) for index in range(chunks)]


def audio_tokens(frames: int) -> int:
    """Placeholder tokens a request holds for this many mel frames."""
    return sum(chunk_lengths(frames))


def attention_windows(frames: int, window_chunks: int = 8) -> list:
    """How the encoder's frames are grouped for attention: eight chunks' worth at a time, then the rest."""
    lengths = chunk_lengths(frames)
    total = sum(lengths)
    if not total:
        return []
    window = max(lengths) * window_chunks
    return [window] * (total // window) + ([total % window] if total % window else [])


def head_text(context: str = "") -> str:
    """Everything before the audio placeholders."""
    return "<|im_start|>system\n" + (context or "") + "<|im_end|>\n<|im_start|>user\n<|audio_start|>"


def tail_text(language: str = "") -> str:
    """Everything after them, ending in the open assistant turn and the named language if there is one."""
    text = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
    return text + ("language " + language + MARKER if language else "")


def request_ids(encode, frames: int, language: str = "", context: str = "") -> list:
    """The request's token ids, with ``encode(text)`` tokenizing the fixed parts."""
    return list(encode(head_text(context))) + [AUDIO_PAD] * audio_tokens(frames) + list(encode(tail_text(language)))


def _collapse_characters(text: str, limit: int) -> str:
    out = []
    index = 0
    while index < len(text):
        run = 1
        while index + run < len(text) and text[index + run] == text[index]:
            run += 1
        out.append(text[index] if run > limit else text[index:index + run])
        index += run
    return "".join(out)


def _collapse_patterns(text: str, limit: int, longest: int) -> str:
    out = []
    index = 0
    while index < len(text):
        for size in range(1, longest + 1):
            pattern = text[index:index + size]
            if len(pattern) < size or text[index:index + size * (limit + 1)] != pattern * (limit + 1):
                continue
            end = index + size * (limit + 1)
            while text[end:end + size] == pattern:
                end += size
            out.append(pattern)
            index = end
            break
        else:
            out.append(text[index])
            index += 1
    return "".join(out)


def collapse_repeats(text: str, limit: int = REPEAT_LIMIT, longest: int = LONGEST_REPEAT) -> str:
    """A character or a short phrase said more than ``limit`` times in a row, said once.

    A speech model that loses its place can write the same syllable until it
    runs out of tokens; Qwen's own library cleans that up the same way.
    """
    return _collapse_patterns(_collapse_characters(text, limit), limit, longest)


def read_answer(text: str, language: str = "") -> dict:
    """``{"language", "text"}`` from the decoded answer; the named language when the request named one.

    An answer cut off after its language line, before the marker, heard no
    words: the line is the model's header, not lyrics.
    """
    text = (text or "").strip()
    if not text:
        return {"language": language or None, "text": ""}
    text = collapse_repeats(text)
    if language:
        return {"language": language, "text": text.split(MARKER, 1)[-1].strip()}
    if MARKER not in text:
        if text.lower().startswith("language ") and len(text.splitlines()[0].split()) <= 2:
            return {"language": text.splitlines()[0][len("language "):].strip() or None, "text": ""}
        return {"language": None, "text": text}
    prefix, words = text.split(MARKER, 1)
    found = None
    for line in prefix.splitlines():
        line = line.strip()
        if line:
            found = line[len("language "):].strip() if line.lower().startswith("language ") else line
            break
    if found is not None and found.lower() == "none":
        found = None
    return {"language": found or None, "text": words.strip()}
