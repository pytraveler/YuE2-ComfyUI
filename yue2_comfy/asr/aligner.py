"""Qwen3-ForcedAligner-0.6B on this pack's Qwen3-ASR network: when each word of a song is sung.

The aligner is the speech model's own audio tower with a smaller Qwen3
language model behind it, and a head that sorts every ``<timestamp>`` token
into one of 5000 steps of 80 ms. The request is the audio's placeholders
followed by the words, each word followed by two of those tokens -- its start
and its end -- and one causal pass reads them all: nothing is generated, so
the whole song is timed in a single forward.

It is not a recogniser. The words are given, and what comes back is where they
are sung, which is what changing one line of a song needs: that line's own
stretch, rather than the verse it sits in. ``inpaint.lines`` does the
arithmetic on the answer.

The network module is written for the 1.7B speech model and reads its widths
as module constants when a layer is made, so the same file builds this one
once ``_shaped`` has set them; the audio tower, the head count and the head
size are the same in both. Qwen's release is Apache-2.0, 1.84 GB, and it
carries its tokenizer as a vocabulary and a merge list rather than the one
file this pack's reader reads -- the speech model's ``tokenizer.json`` holds
the same 151643 entries and the same 151387 merges, so that is what is used,
with the timestamp token taken from the aligner's own config.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os

import torch
from torch import nn

from . import network, prompt

log = logging.getLogger(__name__)

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
TOKENIZER_NAME = "tokenizer.json"

HEAD_KEY = "thinker.lm_head.weight"
RENAME = (("thinker.audio_tower.proj1.", "multi_modal_projector.linear_1."),
          ("thinker.audio_tower.proj2.", "multi_modal_projector.linear_2."),
          ("thinker.audio_tower.", "audio_tower."),
          ("thinker.model.", "language_model."))

SIZES = (("WIDTH", "hidden_size"), ("FFN", "intermediate_size"), ("VOCAB", "vocab_size"),
         ("LAYERS", "num_hidden_layers"))


class Aligner:
    """The two halves of the aligner, its tokenizer, and the numbers its answer is read with."""

    def __init__(self, net, head, tokenizer, stamp: int, step_ms: int):
        self.net = net
        self.head = head
        self.tokenizer = tokenizer
        self.stamp = int(stamp)
        self.step_ms = int(step_ms)


def is_loaded() -> bool:
    """Whether the aligner is held, for the report Unload Models writes; it is kept in ``runtime``."""
    from . import runtime

    return runtime.aligner_loaded()


def unload() -> None:
    """Let go of the held aligner, when someone asks for the memory back."""
    from . import runtime

    runtime.unload_aligner()


def sizes_of(folder: str) -> dict:
    """The released config as this port reads it: the widths to build, and the clock to read by."""
    with open(os.path.join(folder, CONFIG_NAME), "r", encoding="utf-8") as handle:
        whole = json.load(handle)
    inner = whole.get("thinker_config") or {}
    text = inner.get("text_config") or {}
    missing = [key for _name, key in SIZES if not isinstance(text.get(key), int)]
    if missing or not isinstance(inner.get("classify_num"), int):
        raise ValueError("this is not the forced aligner's config: it does not say {}".format(
            ", ".join(missing) or "classify_num"))
    shape = (text.get("num_attention_heads"), text.get("num_key_value_heads"), text.get("head_dim"))
    if shape != (network.HEADS, network.KV_HEADS, network.HEAD):
        raise ValueError("the aligner's attention shape {} is not the one this port builds".format(
            shape))
    found = {name: int(text[key]) for name, key in SIZES}
    found["classify"] = int(inner["classify_num"])
    found["stamp"] = int(whole.get("timestamp_token_id", 151705))
    found["step_ms"] = int(whole.get("timestamp_segment_time", 80))
    return found


@contextlib.contextmanager
def _shaped(sizes: dict):
    """The network module built to the aligner's widths for as long as this is open.

    Nothing else builds a network while it is: both halves are made inside
    ``load``, on the meta device, and the widths go back before any weight is
    read.
    """
    saved = {name: getattr(network, name) for name, _key in SIZES}
    try:
        for name, _key in SIZES:
            setattr(network, name, sizes[name])
        yield
    finally:
        for name, value in saved.items():
            setattr(network, name, value)


def load(folder: str, tokenizer_path: str, device, dtype=torch.bfloat16,
         low_vram: bool = False) -> Aligner:
    """The aligner with the folder's weights, on ``device``, reading words with ``tokenizer_path``.

    With ``low_vram`` the audio's convolutions go ``network.SMALL_CONV_CHUNKS``
    seconds at a time, which is where its pass over a song spent its memory:
    measured on 2026-09-23, 1.72 GiB above the weights for four minutes and
    2.34 for six, against 0.43 and 0.65 in parts. Its weights stay BF16: at
    1.75 GiB they are not what a small card runs out of.
    """
    from safetensors import safe_open

    from .tokenizer import Tokenizer

    sizes = sizes_of(folder)
    with _shaped(sizes), torch.device("meta"):
        net = network.Network()
        head = nn.Linear(sizes["WIDTH"], sizes["classify"], bias=False)
    state, crown = {}, {}
    with safe_open(os.path.join(folder, WEIGHTS_NAME), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key == HEAD_KEY:
                crown["weight"] = handle.get_tensor(key).to(dtype)
                continue
            for old, new in RENAME:
                if key.startswith(old):
                    state[new + key[len(old):]] = handle.get_tensor(key).to(dtype)
                    break
            else:
                raise ValueError(
                    "the aligner's weights do not fit: unexpected tensor {}".format(key))
    missing, unexpected = net.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected or "weight" not in crown:
        raise ValueError("the aligner's weights do not fit: missing {} unexpected {}".format(
            missing[:5], unexpected[:5]))
    head.load_state_dict(crown, assign=True)
    net.audio_tower.positions = network.sinusoids(network.POSITIONS, network.AUDIO_WIDTH)
    if low_vram:
        net.audio_tower.conv_chunks = network.SMALL_CONV_CHUNKS
    net = net.to(device).eval().requires_grad_(False)
    head = head.to(device).eval().requires_grad_(False)
    return Aligner(net, head, Tokenizer(tokenizer_path), sizes["stamp"], sizes["step_ms"])


def repair(values) -> list:
    """Times sorted into a rising run, the way Qwen's own reader sorts them.

    The head answers each timestamp on its own, so a word here and there comes
    back out of order. The longest rising run is taken as read; a short gap
    between two of them is filled from whichever side is nearer, and a longer
    one is spread evenly across it.
    """
    data = [int(value) for value in values]
    count = len(data)
    if not count:
        return []
    best = [1] * count
    parent = [-1] * count
    for index in range(1, count):
        for earlier in range(index):
            if data[earlier] <= data[index] and best[earlier] + 1 > best[index]:
                best[index] = best[earlier] + 1
                parent[index] = earlier
    rising = [False] * count
    index = best.index(max(best))
    while index != -1:
        rising[index] = True
        index = parent[index]
    found = list(data)
    at = 0
    while at < count:
        if rising[at]:
            at += 1
            continue
        stop = at
        while stop < count and not rising[stop]:
            stop += 1
        left = next((found[k] for k in range(at - 1, -1, -1) if rising[k]), None)
        right = next((found[k] for k in range(stop, count) if rising[k]), None)
        wide = stop - at
        for k in range(at, stop):
            if wide <= 2:
                if left is None:
                    found[k] = right
                elif right is None:
                    found[k] = left
                else:
                    found[k] = left if (k - (at - 1)) <= (stop - k) else right
            elif left is not None and right is not None:
                found[k] = left + (right - left) / (wide + 1) * (k - at + 1)
            else:
                found[k] = left if left is not None else right
        at = stop
    return [int(value) for value in found]


@torch.inference_mode()
def align(aligner: Aligner, audio: torch.Tensor, text: str, cancelled=None, progress=None) -> list:
    """``[(word, start, stop)]`` in seconds for the words of ``text`` in ``[samples]`` of mono 16 kHz audio."""
    from .align import words_of

    words = words_of(text)
    if not words:
        return []
    if audio.numel() and not torch.isfinite(audio).all():
        raise ValueError("the audio holds samples that are not finite numbers")
    net, head = aligner.net, aligner.head
    mel = network.log_mel(audio)
    ids = [prompt.AUDIO_START] + [prompt.AUDIO_PAD] * prompt.audio_tokens(mel.shape[1])
    ids = ids + [prompt.AUDIO_END]
    for word in words:
        ids = ids + list(aligner.tokenizer.encode(word)) + [aligner.stamp, aligner.stamp]
    heard = net.hear(mel)
    del mel
    device, dtype = net.device, net.dtype
    request = torch.tensor([ids], device=device)
    length = len(ids)
    with network.exact_float32():
        x = net.language_model.embed_tokens(request)
        slots = (request[0] == prompt.AUDIO_PAD).nonzero().squeeze(-1)
        if int(slots.numel()) != int(heard.shape[0]):
            raise ValueError("the request holds {} audio placeholders for {} audio frames".format(
                int(slots.numel()), int(heard.shape[0])))
        x[0, slots] = heard.to(dtype)
        del heard
        cos, sin = network.rotary(length, device, dtype)
        layers = net.language_model.layers
        for index, layer in enumerate(layers):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled while the words were being timed")
            keys = torch.empty(1, network.KV_HEADS, length, network.HEAD, device=device,
                               dtype=dtype)
            values = torch.empty_like(keys)
            x = x + layer.self_attn(layer.input_layernorm(x), cos, sin, keys, values, 0)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            del keys, values
            if progress is not None:
                progress.ratio((index + 1) / len(layers))
        x = net.language_model.norm(x)
        stamps = (request[0] == aligner.stamp).nonzero().squeeze(-1)
        steps = head(x[0, stamps]).float().argmax(-1).cpu().tolist()
    times = repair([step * aligner.step_ms for step in steps])
    log.info("[yue2_comfy.asr.aligner] %d words timed in one pass of %d tokens", len(words), length)
    return [(word, times[2 * index] / 1000.0, times[2 * index + 1] / 1000.0)
            for index, word in enumerate(words)]
