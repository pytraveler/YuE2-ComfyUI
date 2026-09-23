"""Loading Qwen3-ASR's weights and recognising the words of one stretch of audio.

The weights are the ``-hf`` release: one ``model.safetensors`` beside the
``tokenizer.json`` it was trained with. Everything runs in bfloat16 except the
log-mel front end, which is worked out in float32 on the CPU.
"""

from __future__ import annotations

import os

import torch

from . import network, prompt

PREFIX = "model."
TIED_HEAD = "lm_head.weight"


def load(folder: str, device, dtype=torch.bfloat16, low_vram: bool = False) -> network.Network:
    """The network with the folder's weights, on ``device``; with ``low_vram``, shaped for a small card.

    That is the language model's layers packed (``pack``) and the audio's
    convolutions taken ``network.SMALL_CONV_CHUNKS`` seconds at a time.
    """
    from safetensors import safe_open

    with torch.device("meta"):
        net = network.Network()
    state = {}
    with safe_open(os.path.join(folder, "model.safetensors"), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if name == TIED_HEAD:
                continue
            if not name.startswith(PREFIX):
                raise ValueError("Qwen3-ASR weights do not fit: unexpected tensor '{}'".format(name))
            state[name[len(PREFIX):]] = handle.get_tensor(name).to(dtype)
    missing, unexpected = net.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError("Qwen3-ASR weights do not fit: missing {} unexpected {}".format(missing[:5], unexpected[:5]))
    net.audio_tower.positions = network.sinusoids(network.POSITIONS, network.AUDIO_WIDTH)
    if low_vram:
        net.audio_tower.conv_chunks = network.SMALL_CONV_CHUNKS
        pack(net, device)
    return net.to(device).eval().requires_grad_(False)


def pack(net: network.Network, card=None) -> int:
    """The language model's 28 layers as INT8 rows, and how many matrices were packed.

    The same rows as the song model's under 'low_vram' (``quantized.Packed``),
    packed on ``card`` a matrix at a time while the weights wait in system
    RAM, so the card never holds the BF16 model. The audio encoder, the
    projector and the embedding the answer is read with stay as they are.

    Measured on 2026-09-23 on twelve recordings, five with known lyrics, with
    the convolutions in parts as well: the weights went 3.80 -> 2.49 GiB, and
    the lyrics were heard as well as before -- WER 0.061, 0.066, 0.048, 0.155,
    0.178 against 0.061, 0.073, 0.048, 0.168, 0.178 -- and the words of twenty
    8-second clips, which is what picks a take, 144 of 172 either way.
    Decoding was a fifth slower. Packing the encoder too saved another 0.28
    GiB and dropped the whole first verse of one of the user's songs (WER
    0.178 -> 0.310); scaled every 128 values instead of every row it kept that
    verse and heard the user's other song worse (0.168 -> 0.217). So the
    encoder is left alone.
    """
    from .. import quantized

    packed = 0
    for layer in net.language_model.layers:
        for block in (layer.self_attn, layer.mlp):
            for leaf, child in list(block.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(block, leaf, quantized.Packed(child, card))
                    packed += 1
    return packed


def mono_16k(waveform: torch.Tensor, rate: int) -> torch.Tensor:
    """``[channels, samples]`` at any rate as mono float32 at 16 kHz, on the CPU.

    A whole ComfyUI AUDIO value, ``[recordings, channels, samples]``, is heard
    as its first recording, which is the one every node here works on.
    """
    mono = waveform.detach().float().cpu()
    if mono.dim() == 3:
        mono = mono[0]
    if mono.dim() == 2:
        mono = mono.mean(dim=0)
    if int(rate) != prompt.SAMPLE_RATE:
        import torchaudio

        mono = torchaudio.functional.resample(mono, int(rate), prompt.SAMPLE_RATE)
    return mono


def token_limit(seconds: float) -> int:
    """Room for the answer: far more than anyone sings in that time, so only a lost model reaches it."""
    return 64 + int(seconds * 12)


@torch.inference_mode()
def recognise(net: network.Network, tokenizer, audio: torch.Tensor, language: str = "", context: str = "",
              limit=None, cancelled=None, progress=None) -> dict:
    """``{"language", "text", "tokens"}`` for ``[samples]`` of mono 16 kHz audio.

    ``language`` is a name ``prompt.language_name`` accepts, or "" to let the
    model tell. ``context`` goes in the system turn, where the model reads it
    as words it may hear.
    """
    if audio.numel() and not torch.isfinite(audio).all():
        raise ValueError("the audio holds samples that are not finite numbers")
    name = prompt.language_name(language)
    mel = network.log_mel(audio)
    frames = mel.shape[1]
    ids = prompt.request_ids(tokenizer.encode, frames, name, context)
    heard = net.hear(mel)
    del mel
    seconds = audio.numel() / prompt.SAMPLE_RATE
    tokens = net.generate(ids, heard, limit or token_limit(seconds), cancelled=cancelled, progress=progress)
    del heard
    answer = prompt.read_answer(tokenizer.decode(tokens), name)
    return {"language": answer["language"], "text": answer["text"], "tokens": tokens, "request": ids}
