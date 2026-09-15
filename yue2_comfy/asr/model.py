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


def load(folder: str, device, dtype=torch.bfloat16) -> network.Network:
    """The network with the folder's weights, on ``device``."""
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
    return net.to(device).eval().requires_grad_(False)


def mono_16k(waveform: torch.Tensor, rate: int) -> torch.Tensor:
    """``[channels, samples]`` at any rate as mono float32 at 16 kHz, on the CPU."""
    mono = waveform.detach().float().cpu()
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
