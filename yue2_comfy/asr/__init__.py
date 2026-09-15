"""Qwen3-ASR in plain torch: the words sung in a recording, for YuE2 Transcribe.

``prompt`` is the request around the audio and the reading of the answer, with
no dependencies. ``network`` and ``model`` need torch and are imported only
when words are actually recognised. The weights are Qwen's own
``Qwen/Qwen3-ASR-1.7B-hf`` release, Apache-2.0.
"""
