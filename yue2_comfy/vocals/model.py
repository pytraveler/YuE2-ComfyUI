"""Loading Mel-Band RoFormer's weights and separating a whole recording in overlapping chunks.

The network was trained on eight-second stereo chunks at 44.1 kHz, so a
recording is resampled to that, cut into eight-second windows that overlap by
half, and put back together with linear cross-fades a tenth of a window long,
the way the model's authors run it: the recording is first mirrored outwards by
the overlap at both ends, so its first and last seconds are heard in the middle
of a window too. The voice then goes back to the recording's own rate, channel
count and length.
"""

from __future__ import annotations

import os

import torch

from . import bands, network

CHUNK = 352800
"""Eight seconds at 44.1 kHz: the length the model was trained on."""
OVERLAPS = 2
STEP = CHUNK // OVERLAPS
FADE = CHUNK // 10
BORDER = CHUNK - STEP


def read_weights(path: str) -> dict:
    """The tensors of a ``.ckpt`` state dict or a ``.safetensors`` file, on the CPU.

    A ``.ckpt`` is a pickle, so it is read with ``weights_only``: tensors and
    plain containers load, code does not run.
    """
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Mel-Band RoFormer weights do not fit: {} holds no tensors by name".format(
            os.path.basename(path)))
    return state


def load(path: str, device, dtype=torch.float32) -> network.Network:
    """The network with the file's weights, on ``device``; every tensor must be there with its shape."""
    state = read_weights(path)
    with torch.device("meta"):
        net = network.Network()
    wanted = {name: tuple(tensor.shape) for name, tensor in net.state_dict().items()}
    found = {name: tuple(tensor.shape) for name, tensor in state.items()}
    missing = sorted(set(wanted) - set(found))
    unexpected = sorted(set(found) - set(wanted))
    misshapen = sorted(name for name in set(wanted) & set(found) if wanted[name] != found[name])
    if missing or unexpected or misshapen:
        raise ValueError("Mel-Band RoFormer weights do not fit: missing {} unexpected {} misshapen {}".format(
            missing[:5], unexpected[:5], misshapen[:5]))
    net.load_state_dict({name: tensor.to(dtype) for name, tensor in state.items()}, strict=True, assign=True)
    net = net.to(device).eval().requires_grad_(False)
    net.index_tables(device)
    return net


def fades(device) -> torch.Tensor:
    """The window every chunk is weighted by: a linear rise, a flat top, a linear fall."""
    window = torch.ones(CHUNK, device=device)
    window[:FADE] = torch.linspace(0.0, 1.0, FADE, device=device)
    window[-FADE:] = torch.linspace(1.0, 0.0, FADE, device=device)
    return window


def chunk_starts(length: int) -> list:
    """Where each window starts in a padded recording of ``length`` samples."""
    return list(range(0, length, STEP))


@torch.inference_mode()
def demix(net, audio: torch.Tensor, cancelled=None, progress=None) -> torch.Tensor:
    """``[2, samples]`` stereo at 44.1 kHz on the network's device to the vocals, the same shape.

    ``progress`` is called with the share done after every window, and
    ``cancelled`` is asked before every window; a yes raises ``InterruptedError``.
    """
    length = audio.shape[-1]
    padded = length > 2 * BORDER
    if padded:
        audio = torch.nn.functional.pad(audio[None], (BORDER, BORDER), mode="reflect")[0]
    total = audio.shape[-1]
    window = fades(audio.device)
    summed = torch.zeros_like(audio)
    weight = torch.zeros(total, device=audio.device)
    starts = chunk_starts(total)
    for number, start in enumerate(starts):
        if cancelled is not None and cancelled():
            raise InterruptedError("voice separation cancelled")
        part = audio[:, start:start + CHUNK]
        size = part.shape[-1]
        if size < CHUNK:
            mode = "reflect" if size > CHUNK // 2 + 1 else "constant"
            part = torch.nn.functional.pad(part[None], (0, CHUNK - size), mode=mode)[0]
        voice = net(part[None])[0]
        weights = window.clone()
        if start == 0:
            weights[:FADE] = 1.0
        elif start + CHUNK >= total:
            weights[-FADE:] = 1.0
        summed[:, start:start + size] += voice[:, :size] * weights[:size]
        weight[start:start + size] += weights[:size]
        if progress is not None:
            progress((number + 1) / len(starts))
    vocals = torch.nan_to_num(summed / weight)
    if padded:
        vocals = vocals[:, BORDER:-BORDER]
    return vocals


def separate(net, waveform: torch.Tensor, rate: int, cancelled=None, progress=None) -> torch.Tensor:
    """ComfyUI audio ``[batch, channels, samples]`` at any rate to its vocals, same shape and rate, float32 on the CPU.

    Mono is heard as two equal channels and folded back; more than two channels
    are refused, since the model knows only stereo.
    """
    import torchaudio

    batch, channels, length = waveform.shape
    if channels not in (1, 2):
        raise ValueError("Only mono or stereo audio can be separated; this has {} channels.".format(channels))
    if length == 0:
        return waveform.detach().float().cpu().clone()
    device = next(net.parameters()).device
    outputs = []
    for item in range(batch):
        audio = waveform[item].detach().float().cpu()
        if channels == 1:
            audio = audio.expand(2, -1)
        if int(rate) != bands.SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, int(rate), bands.SAMPLE_RATE)

        def report(share, item=item):
            if progress is not None:
                progress((item + share) / batch)

        vocals = demix(net, audio.to(device), cancelled=cancelled, progress=report).float().cpu()
        if int(rate) != bands.SAMPLE_RATE:
            vocals = torchaudio.functional.resample(vocals, bands.SAMPLE_RATE, int(rate))
        if vocals.shape[-1] >= length:
            vocals = vocals[:, :length]
        else:
            vocals = torch.nn.functional.pad(vocals, (0, length - vocals.shape[-1]))
        if channels == 1:
            vocals = vocals.mean(dim=0, keepdim=True)
        outputs.append(vocals)
    return torch.stack(outputs).contiguous()
