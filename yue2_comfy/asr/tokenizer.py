"""Qwen3-ASR's tokenizer, read with the ``tokenizers`` library ComfyUI already installs."""

from __future__ import annotations


class Tokenizer:
    """Text to ids and back, from the release's ``tokenizer.json``."""

    def __init__(self, path: str):
        from tokenizers import Tokenizer as Backend

        self._backend = Backend.from_file(path)

    def encode(self, text: str) -> list:
        return self._backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids) -> str:
        """The words of ``ids``, with the chat's special tokens left out."""
        return self._backend.decode([int(token) for token in ids], skip_special_tokens=True)
