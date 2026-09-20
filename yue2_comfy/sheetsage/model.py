"""Loading SheetSage2's weights and transcribing a whole recording.

The weights are Comfy-Org's single file, ``sheetsage2_bf16.safetensors``: the
MERT-v2 encoder with m-a-p's adapters already merged, the decoder, and the mel
filters and statistics the front end needs. A song is read in 300-second
windows (see ``events``), each padded with silence to the full 300 seconds,
because the encoder was trained to hear a whole window, silence included.

``transcribe`` takes the window as a parameter, because how much the model
hears at a time decides the beat it settles on (see ``events.MINUTE``): a
minute at a time is written on the recording's own pulse where the whole song
is not. The encoder is handed 300 seconds either way, so the short window is
the same pass over less music, not a cheaper one.
"""

from __future__ import annotations

import torch

from . import events, network, vocab

FRONTEND_KEYS = {"encoder.feature_extractor.spectrogram.window": "frontend.window",
                 "encoder.feature_extractor.mel_scale.fb": "frontend.filters",
                 "encoder.feature_extractor.mel_mean": "frontend.mean",
                 "encoder.feature_extractor.mel_std": "frontend.std"}


def load(path: str, device, dtype=torch.bfloat16) -> network.Network:
    """The network with the file's weights, on ``device``; the front end stays in float32."""
    from safetensors import safe_open

    with torch.device("meta"):
        net = network.Network()
    state = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            state[FRONTEND_KEYS.get(name, name)] = handle.get_tensor(name)
    frontend = {name[len("frontend."):]: state.pop(name).float() for name in list(state)
                if name.startswith("frontend.")}
    missing, unexpected = net.load_state_dict({name: tensor.to(dtype) for name, tensor in state.items()},
                                              strict=False, assign=True)
    missing = [name for name in missing if not name.startswith("frontend.")]
    if missing or unexpected:
        raise ValueError("SheetSage2 weights do not fit: missing {} unexpected {}".format(missing[:5], unexpected[:5]))
    for name, tensor in frontend.items():
        setattr(net.frontend, name, tensor)
    return net.to(device).eval().requires_grad_(False)


def mono_24k(waveform: torch.Tensor, rate: int) -> torch.Tensor:
    """``[channels, samples]`` at any rate as mono float32 at 24 kHz, on the CPU."""
    mono = waveform.detach().float().cpu().mean(dim=0)
    if int(rate) != network.SAMPLE_RATE:
        import torchaudio

        mono = torchaudio.functional.resample(mono, int(rate), network.SAMPLE_RATE)
    return mono


def _slice(audio: torch.Tensor, start: float, length: float = vocab.WINDOW_SECONDS) -> torch.Tensor:
    """``length`` seconds of the recording from ``start``, padded with silence to a full window.

    The encoder is always handed 300 seconds because that is what it was
    trained on, silence included. A shorter window is therefore not a cheaper
    one: it is the same pass over less music.
    """
    offset = round(start * network.SAMPLE_RATE)
    piece = audio[offset:offset + int(round(length * network.SAMPLE_RATE))]
    return torch.nn.functional.pad(piece, (0, network.WINDOW_SAMPLES - piece.numel()))


@torch.inference_mode()
def transcribe(net: network.Network, waveform: torch.Tensor, rate: int, cancelled=None, progress=None,
               length: float = vocab.WINDOW_SECONDS, carry: bool = True) -> dict:
    """A recording's events, in song order, with the tokens of every window.

    ``progress(stage, window, windows, tokens)`` hears about each window's
    encoding and every 64 decoded tokens. ``warnings`` are the lenient
    decodings; ``cut_short`` numbers the windows, from 1, whose decoding
    filled ``network.MAX_TOKENS`` before their end. ``handed_over`` holds the
    seconds such a window never reached, which the next one takes over (see
    ``events.resume_point``): the index of that next window, and the span it
    gained as ``from`` and ``to``.
    """
    audio = mono_24k(waveform, rate)
    if audio.numel() < 1025 or not torch.isfinite(audio).all():
        raise ValueError("the recording needs at least 1025 finite samples at 24 kHz")
    duration = audio.numel() / network.SAMPLE_RATE
    device = net.encoder_projection.weight.device
    plan = events.plan_for(duration, length)
    stitched = []
    windows = []
    warnings = []
    cut_short = []
    handed_over = []
    for index, window in enumerate(plan):
        if cancelled is not None and cancelled():
            raise InterruptedError("transcription cancelled")
        if progress is not None:
            progress("encode", index, len(plan), 0)
        prefix, base = None, 0
        if index and carry:
            prefix, base = events.carried_prefix(stitched, vocab.FULL_PROMPTS, window)
            if prefix is not None and len(prefix) >= network.MAX_TOKENS - 128:
                raise ValueError("the overlap between windows fills the decoder's context")
        memory = net.encode(_slice(audio, window["start"], length)[None].to(device))
        tokens, cut = net.generate(memory, prefix or vocab.prompt_prefix(),
                                   events.stop_seconds(window, duration, length=length),
                                   cancelled=cancelled,
                                   progress=(lambda count, i=index: progress("decode", i, len(plan), count))
                                   if progress is not None else None)
        del memory
        windows.append(tokens)
        if cut:
            cut_short.append(index + 1)
        decoded, warning = vocab.decode_window(tokens)
        if warning:
            warnings.append(warning)
        kept = events.stitch(decoded, events.time_map(decoded, target=length), window, duration, index, base or 0)
        stitched.extend(kept)
        resume = events.resume_point(window, kept) if index + 1 < len(plan) else 0.0
        if resume:
            handed_over.append({"window": index + 1, "from": resume, "to": float(window["accept_end"])})
            plan[index + 1]["accept_start"] = resume
            plan[index + 1]["prefix_end"] = resume
    return {"events": events.sort_song(stitched), "seconds": duration, "windows": windows, "warnings": warnings,
            "cut_short": cut_short, "handed_over": handed_over}
