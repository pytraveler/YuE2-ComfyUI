"""Qwen3-ASR-1.7B's network in plain torch: an audio encoder feeding a Qwen3 language model.

The shapes are those of the released weights, and module names follow the
file's keys under ``model.``, so the weights load without a rename table.

Audio is heard as 128-bin log-mel frames, 100 a second. The encoder cuts them
into one-second chunks, and three halving 3x3 convolutions turn each chunk
into 13 frames with a sinusoidal position added. Frames past the end of the
audio are dropped, and 24 pre-norm transformer layers attend within windows of
eight chunks. A two-layer projector widens the result to the language
model's 2048.

The language model is Qwen3: 28 layers, 16 query and 8 key-value heads of 128,
query and key RMS norms, rotary positions with base one million and a gated
SiLU feed-forward, its output head tied to the embedding. The audio frames
take the place of the request's placeholder tokens, and the answer is decoded
greedily into a preallocated cache.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from . import decode, prompt

N_FFT = 400
MELS = 128
MEL_TOP_HZ = 8000.0
AUDIO_WIDTH = 1024
AUDIO_HEADS = 16
AUDIO_LAYERS = 24
AUDIO_FFN = 4096
CONV_WIDTH = 480
POSITIONS = 13
WIDTH = 2048
HEADS = 16
KV_HEADS = 8
HEAD = 128
LAYERS = 28
FFN = 6144
VOCAB = 151936
ROPE_BASE = 1000000
RMS_EPS = 1e-6
AUDIO_SCALE = (AUDIO_WIDTH // AUDIO_HEADS) ** -0.5
SCALE = HEAD ** -0.5
ROOM_STEP = 1024
"""The cache is sized in whole steps of this, so requests of nearby lengths share one cache and one graph."""
CONV_CHUNKS = 320
"""Chunks, a second of audio each, the audio encoder's convolutions take at once: a longer recording goes in parts."""
NEAR_TIE = 2.0
"""How far below the plain step's best logit a replayed pick may fall at a check and still stand."""
GRAPHS = True


def _hz_to_mel(hz: float) -> float:
    return 15.0 + math.log(hz / 1000.0) * (27.0 / math.log(6.4)) if hz >= 1000.0 else 3.0 * hz / 200.0


def mel_filters() -> torch.Tensor:
    """Slaney-scale triangular filters from 0 to 8 kHz, area-normalised: ``[201, 128]`` float32."""
    bins = N_FFT // 2 + 1
    top = _hz_to_mel(MEL_TOP_HZ)
    step = top / (MELS + 1)
    mels = torch.arange(MELS + 2, dtype=torch.float64) * step
    mels[-1] = top
    centres = torch.where(mels >= 15.0, 1000.0 * torch.exp((math.log(6.4) / 27.0) * (mels - 15.0)), 200.0 * mels / 3.0)
    fft = torch.arange(bins, dtype=torch.float64) * ((prompt.SAMPLE_RATE // 2) / (bins - 1))
    fft[-1] = float(prompt.SAMPLE_RATE // 2)
    gaps = centres[1:] - centres[:-1]
    slopes = centres[None, :] - fft[:, None]
    falling = -slopes[:, :-2] / gaps[:-1]
    rising = slopes[:, 2:] / gaps[1:]
    filters = torch.clamp(torch.minimum(falling, rising), min=0.0)
    filters = filters * (2.0 / (centres[2:MELS + 2] - centres[:MELS]))[None, :]
    return filters.float()


def log_mel(audio: torch.Tensor) -> torch.Tensor:
    """``[samples]`` float32 at 16 kHz on the CPU to ``[128, frames]`` log-mel, clipped 8 below its loudest."""
    audio = audio.float()
    if audio.numel() < prompt.MIN_SAMPLES:
        audio = F.pad(audio, (0, prompt.MIN_SAMPLES - audio.numel()))
    spectrum = torch.stft(audio, N_FFT, prompt.HOP, window=torch.hann_window(N_FFT), return_complex=True)
    power = spectrum[..., :-1].abs() ** 2
    mel = mel_filters().T @ power
    logs = torch.clamp(mel, min=1e-10).log10()
    logs = torch.maximum(logs, logs.max() - 8.0)
    return (logs + 4.0) / 4.0


def sinusoids(length: int, channels: int) -> torch.Tensor:
    """Whisper's fixed positions: sines then cosines over geometric timescales up to 10000."""
    increment = math.log(10000) / (channels // 2 - 1)
    inverse = torch.exp(-increment * torch.arange(channels // 2).float())
    scaled = torch.arange(length)[:, None] * inverse[None, :]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


class AudioAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(AUDIO_WIDTH, AUDIO_WIDTH)
        self.k_proj = nn.Linear(AUDIO_WIDTH, AUDIO_WIDTH)
        self.v_proj = nn.Linear(AUDIO_WIDTH, AUDIO_WIDTH)
        self.out_proj = nn.Linear(AUDIO_WIDTH, AUDIO_WIDTH)

    def forward(self, x, windows):
        frames = x.shape[0]
        shape = (frames, AUDIO_HEADS, AUDIO_WIDTH // AUDIO_HEADS)
        q, k, v = (projection(x).reshape(shape).transpose(0, 1).unsqueeze(0)
                   for projection in (self.q_proj, self.k_proj, self.v_proj))
        parts = []
        for qw, kw, vw in zip(q.split(windows, dim=2), k.split(windows, dim=2), v.split(windows, dim=2)):
            parts.append(F.scaled_dot_product_attention(qw, kw, vw, scale=AUDIO_SCALE).transpose(1, 2).contiguous())
        return self.out_proj(torch.cat(parts, dim=1).reshape(frames, -1))


class AudioLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = AudioAttention()
        self.self_attn_layer_norm = nn.LayerNorm(AUDIO_WIDTH)
        self.fc1 = nn.Linear(AUDIO_WIDTH, AUDIO_FFN)
        self.fc2 = nn.Linear(AUDIO_FFN, AUDIO_WIDTH)
        self.final_layer_norm = nn.LayerNorm(AUDIO_WIDTH)

    def forward(self, x, windows):
        x = x + self.self_attn(self.self_attn_layer_norm(x), windows)
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class AudioTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(1, CONV_WIDTH, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(CONV_WIDTH, CONV_WIDTH, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(CONV_WIDTH, CONV_WIDTH, 3, 2, padding=1)
        self.conv_out = nn.Linear(CONV_WIDTH * 16, AUDIO_WIDTH, bias=False)
        self.layers = nn.ModuleList([AudioLayer() for _ in range(AUDIO_LAYERS)])
        self.ln_post = nn.LayerNorm(AUDIO_WIDTH)
        self.register_buffer("positions", sinusoids(POSITIONS, AUDIO_WIDTH), persistent=False)

    def _convolved(self, x: torch.Tensor) -> torch.Tensor:
        """``[chunks, 1, 128, 100]`` mel through the three convolutions into ``[chunks, 13, 1024]``.

        Each chunk is convolved on its own, so a long recording can go
        through in parts: the first convolution's output is three megabytes a
        second of audio, and a ten-minute recording taken at once wanted more
        of the card than the language model does. A recording of up to
        ``CONV_CHUNKS`` seconds still goes at once, because the convolution
        kernels chosen for another batch size round differently, and the
        tokens of a song were measured to change with them.
        """
        x = F.gelu(self.conv2d3(F.gelu(self.conv2d2(F.gelu(self.conv2d1(x))))))
        count, channels, bins, steps = x.shape
        return self.conv_out(x.permute(0, 3, 1, 2).contiguous().view(count, steps, channels * bins))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """``[128, frames]`` to ``[tokens, 1024]``, one row for every placeholder of the request."""
        frames = mel.shape[1]
        lengths = prompt.chunk_lengths(frames)
        chunks = len(lengths)
        padded = F.pad(mel, (0, chunks * prompt.CHUNK_FRAMES - frames))
        x = padded[None].view(1, MELS, chunks, prompt.CHUNK_FRAMES).permute(0, 2, 1, 3)
        x = x.reshape(chunks, 1, MELS, prompt.CHUNK_FRAMES).to(self.conv_out.weight.dtype)
        x = self._convolved(x) if chunks <= CONV_CHUNKS else torch.cat(
            [self._convolved(part) for part in x.split(CONV_CHUNKS, dim=0)])
        steps = x.shape[1]
        x = x + self.positions[:steps].to(x.dtype)
        keep = torch.arange(steps, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        x = torch.index_select(x.reshape(-1, x.shape[-1]), 0, keep.flatten().nonzero().squeeze(-1))
        windows = prompt.attention_windows(frames)
        for layer in self.layers:
            x = layer(x, windows)
        return self.ln_post(x)


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(AUDIO_WIDTH, AUDIO_WIDTH)
        self.linear_2 = nn.Linear(AUDIO_WIDTH, WIDTH)

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(x)))


class RMSNorm(nn.Module):
    """Root-mean-square norm taken in float32, scaled in the model's own type."""

    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        kind = x.dtype
        wide = x.float()
        wide = wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return self.weight * wide.to(kind)


def _rotate(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class TextAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(WIDTH, HEADS * HEAD, bias=False)
        self.k_proj = nn.Linear(WIDTH, KV_HEADS * HEAD, bias=False)
        self.v_proj = nn.Linear(WIDTH, KV_HEADS * HEAD, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD, WIDTH, bias=False)
        self.q_norm = RMSNorm(HEAD)
        self.k_norm = RMSNorm(HEAD)

    def forward(self, x, cos, sin, keys, values, position):
        batch, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(batch, length, HEADS, HEAD)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(batch, length, KV_HEADS, HEAD)).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, KV_HEADS, HEAD).transpose(1, 2)
        q = q * cos + _rotate(q) * sin
        k = k * cos + _rotate(k) * sin
        end = position + length
        keys[:, :, position:end] = k
        values[:, :, position:end] = v
        out = F.scaled_dot_product_attention(q, keys[:, :, :end], values[:, :, :end],
                                             is_causal=length > 1, scale=SCALE, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1))

    def step(self, x, cos, sin, keys, values, position, mask):
        """One token at the position held in ``position``, attending over the whole cache through ``mask``.

        Every shape is fixed, so the step can be captured as a CUDA graph.
        """
        q = self.q_norm(self.q_proj(x).view(1, 1, HEADS, HEAD)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(1, 1, KV_HEADS, HEAD)).transpose(1, 2)
        v = self.v_proj(x).view(1, 1, KV_HEADS, HEAD).transpose(1, 2)
        q = q * cos + _rotate(q) * sin
        k = k * cos + _rotate(k) * sin
        keys.index_copy_(2, position, k)
        values.index_copy_(2, position, v)
        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, scale=SCALE, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(1, 1, -1))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(WIDTH, FFN, bias=False)
        self.up_proj = nn.Linear(WIDTH, FFN, bias=False)
        self.down_proj = nn.Linear(FFN, WIDTH, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(WIDTH)
        self.self_attn = TextAttention()
        self.post_attention_layernorm = RMSNorm(WIDTH)
        self.mlp = FeedForward()


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, WIDTH)
        self.layers = nn.ModuleList([TextLayer() for _ in range(LAYERS)])
        self.norm = RMSNorm(WIDTH)


def rotary(length: int, device, dtype) -> tuple:
    """Cosine and sine for positions ``0..length-1``, worked out in float32 and cast to ``dtype``."""
    inverse = 1.0 / (ROPE_BASE ** (torch.arange(0, HEAD, 2, dtype=torch.float) / HEAD))
    angles = torch.outer(torch.arange(length, dtype=torch.float32, device=device), inverse.to(device))
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype)[None, None], angles.sin().to(dtype)[None, None]


class Network(nn.Module):
    """The whole model. ``hear`` gives the audio's frames, ``generate`` the answer's tokens."""

    def __init__(self):
        super().__init__()
        self.audio_tower = AudioTower()
        self.multi_modal_projector = Projector()
        self.language_model = LanguageModel()

    @property
    def device(self):
        return self.multi_modal_projector.linear_2.weight.device

    @property
    def dtype(self):
        return self.multi_modal_projector.linear_2.weight.dtype

    def hear(self, mel: torch.Tensor) -> torch.Tensor:
        """``[128, frames]`` log-mel to ``[tokens, 2048]`` frames in the language model's width."""
        with exact_float32():
            return self.multi_modal_projector(self.audio_tower(mel.to(self.device)))

    def _forward(self, x, cos, sin, cache, position):
        length = x.shape[1]
        cos = cos[:, :, position:position + length]
        sin = sin[:, :, position:position + length]
        for layer, (keys, values) in zip(self.language_model.layers, cache):
            x = x + layer.self_attn(layer.input_layernorm(x), cos, sin, keys, values, position)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = self.language_model.norm(x[:, -1:])
        return F.linear(x, self.language_model.embed_tokens.weight)[:, -1].float()

    def generate(self, ids: list, audio: torch.Tensor, limit: int, cancelled=None, progress=None) -> list:
        """The answer's tokens after ``ids``, greedy, up to a stop token or ``limit`` of them.

        ``audio`` fills the placeholders in order. ``cancelled`` is asked and
        ``progress`` told the count every 32 tokens.
        """
        with exact_float32():
            return self._generate(ids, audio, limit, cancelled, progress)

    def _generate(self, ids, audio, limit, cancelled, progress):
        device, dtype = self.device, self.dtype
        request = torch.tensor([ids], device=device)
        x = self.language_model.embed_tokens(request)
        slots = request[0] == prompt.AUDIO_PAD
        if int(slots.sum()) != audio.shape[0]:
            raise ValueError("the request holds {} audio placeholders for {} audio frames"
                             .format(int(slots.sum()), audio.shape[0]))
        x[0, slots] = audio.to(dtype)
        del request, slots
        room = -(-(len(ids) + limit) // ROOM_STEP) * ROOM_STEP
        static = self._static_for(room, device, dtype)
        cos, sin, cache = static["cos"], static["sin"], static["cache"]
        logits = self._forward(x, cos, sin, cache, 0)
        del x
        stepper = self._stepper(static, len(ids)) if device.type == "cuda" and GRAPHS else None

        checked = [len(ids)]

        def plain(token, position):
            step = self.language_model.embed_tokens(torch.tensor([[token]], device=device))
            return int(torch.argmax(self._forward(step, cos, sin, cache, position)[0]))

        def check(token, position, fast):
            replayed_kv = [(keys[:, :, position].clone(), values[:, :, position].clone()) for keys, values in cache]
            step = self.language_model.embed_tokens(torch.tensor([[token]], device=device))
            logits = self._forward(step, cos, sin, cache, position)[0]
            if not cache_agrees(cache, replayed_kv, checked[0], position):
                return False
            checked[0] = position + 1
            return near_tie(logits, fast, position)

        def disagreed():
            import logging

            logging.getLogger(__name__).warning(
                "[yue2_comfy.asr] the captured step disagreed with the plain one; decoding without it")
            static["step"] = None

        replayed = None if stepper is None else (lambda token, position: int(stepper(token, position)))
        return decode.greedy(int(torch.argmax(logits[0])), len(ids), limit, prompt.STOP_TOKENS, plain, replayed,
                             cancelled, progress, disagreed, check)

    def _static_for(self, room: int, device, dtype) -> dict:
        """The cache and rotary tables for at least ``room`` positions, kept until a request needs more.

        A smaller request reuses the larger cache and its captured step, the
        mask hiding the rest. Only growing records a step again: recording a
        smaller step after a larger one was let go gave a step that answered
        zeros, measured on torch 2.11, while every recording on growth matched.

        The cache starts as zeros, not as whatever the memory held: the
        captured step attends over the whole cache and hides the unfilled
        part with a mask, and a stray infinity there turns the masked sum into
        NaN. Left unfilled, the step disagreed with the plain one on every
        190 s song with one allocator and on none with the other.
        """
        kept = getattr(self, "_static", None)
        if kept is not None and kept["room"] >= room and kept["device"] == device:
            return kept
        self.forget_steps()
        cos, sin = rotary(room, device, dtype)
        cache = [(torch.zeros(1, KV_HEADS, room, HEAD, device=device, dtype=dtype),
                  torch.zeros(1, KV_HEADS, room, HEAD, device=device, dtype=dtype)) for _ in range(LAYERS)]
        self._static = {"room": room, "device": device, "cos": cos, "sin": sin, "cache": cache, "step": None}
        return self._static

    def forget_steps(self) -> None:
        """Let go of the kept cache and the captured step."""
        self._static = None

    def _stepper(self, static: dict, start: int):
        """The captured decoding step for this room, reset for a request of ``start`` tokens; None when capture fails."""
        try:
            if static["step"] is None:
                static["step"] = GraphStep(self, static, start)
            else:
                static["step"].reset(start)
            return static["step"]
        except Exception:
            import logging

            logging.getLogger(__name__).warning("[yue2_comfy.asr] decoding without a CUDA graph", exc_info=True)
            static["step"] = None
            return None

    def _graph_step(self, token, position, mask, static):
        x = self.language_model.embed_tokens(token)
        cos = static["cos"].index_select(2, position)
        sin = static["sin"].index_select(2, position)
        for layer, (keys, values) in zip(self.language_model.layers, static["cache"]):
            x = x + layer.self_attn.step(layer.input_layernorm(x), cos, sin, keys, values, position, mask)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = self.language_model.norm(x)
        return torch.argmax(F.linear(x, self.language_model.embed_tokens.weight)[:, -1].float(), dim=-1)


def near_tie(logits: torch.Tensor, fast: int, position: int, margin: float = NEAR_TIE) -> bool:
    """Whether the plain step's logits let the replayed step's pick ``fast`` stand.

    Equal picks always do. Otherwise ``fast`` must score within ``margin`` of
    the plain step's best: the two steps differ by bfloat16 rounding, which
    flips a near tie now and then, and between checks such flips are taken
    on trust anyway. A pick far below the best is what garbage in the cache
    looks like when the cache check itself has not caught it.
    """
    best = int(torch.argmax(logits))
    if best == fast:
        return True
    gap = float(logits[best] - logits[fast])
    import logging

    logging.getLogger(__name__).info("[yue2_comfy.asr] at position %d the replayed step picked a token %.2f below "
                                     "the plain step's best (a near tie is under %.1f)", position, gap, margin)
    return gap <= margin


def cache_agrees(cache: list, replayed_kv: list, since: int, position: int, tolerance: float = 0.25) -> bool:
    """Whether the replayed steps left the cache as the plain step would have.

    Called at a check, after the plain step has written ``position``: every
    key and value the replays wrote from ``since`` up to it must be finite,
    and what they wrote at ``position`` itself, saved in ``replayed_kv``
    before the plain step overwrote it, must lie within ``tolerance`` of the
    plain step's, taken as a share of the layer's largest value. A step that
    goes wrong leaves NaN, zeros or garbage behind, and that is what the pick
    alone would miss whenever the garbage happens not to change the pick at
    the checked position. The two steps do not agree to the bit: the replayed
    one attends over the whole masked cache and the plain one over a slice,
    and in bfloat16 the deepest layers' values were measured 3 percent of
    their largest apart on a sound run. Garbage is a hundred.
    """
    finite = []
    shares = []
    for (keys, values), (replayed_keys, replayed_values) in zip(cache, replayed_kv):
        finite.append(torch.isfinite(keys[:, :, since:position]).all())
        finite.append(torch.isfinite(values[:, :, since:position]).all())
        for plain_side, replayed_side in ((keys[:, :, position], replayed_keys), (values[:, :, position], replayed_values)):
            plain_side, replayed_side = plain_side.float(), replayed_side.float()
            shares.append((plain_side - replayed_side).abs().max() / (plain_side.abs().max() + 1e-3))
    sound = bool(torch.stack(finite).all().item())
    worst = float(torch.stack(shares).max().item())
    if not sound or worst > tolerance:
        import logging

        logging.getLogger(__name__).info(
            "[yue2_comfy.asr] at position %d the replayed cache %s (the check allows %.0f%%)", position,
            "holds values that are not finite" if not sound else "is off by %.0f%% of a layer's largest value" % (100 * worst),
            100 * tolerance)
        return False
    return True


class GraphStep:
    """One decoding step recorded as a CUDA graph and replayed for every token.

    A step of this model is some two thousand small kernels, and launching them
    one by one costs far more than running them: measured inside a ComfyUI
    process, where every launch takes twice as long as in a bare one, 60 ms a
    token of which the card spent 5. Replayed from a graph the launches go in
    one call. The inputs -- the token, its position and the mask of the cache
    filled so far -- are tensors the graph reads, written before each replay.
    """

    def __init__(self, net: Network, static: dict, start: int):
        """Recorded right after a request's prefill: the rehearsals write only position ``start``, which its first step writes again."""
        device, dtype = net.device, net.dtype
        room = static["room"]
        self.token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.mask = torch.full((1, 1, 1, room), float("-inf"), device=device, dtype=dtype)
        self.reset(start)
        self.position.fill_(start)
        run = lambda: net._graph_step(self.token, self.position, self.mask, static)
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = run()

    def reset(self, start: int) -> None:
        """Ready for a request whose first ``start`` positions the cache already holds."""
        self.mask.fill_(float("-inf"))
        self.mask[..., :start + 1] = 0

    def __call__(self, token: int, position: int) -> torch.Tensor:
        self.token.fill_(token)
        self.position.fill_(position)
        self.mask[..., position] = 0
        self.graph.replay()
        return self.out


class exact_float32:
    """Full-precision float32 matrix products while the model runs, restored after.

    The same guard as SheetSage2's (see ``sheetsage.network.exact_float32``): a
    node pack elsewhere in the process can leave TF32 on, and the norms and the
    rotary tables are float32 products whose last bits choose between
    near-tied words.
    """

    def __enter__(self):
        self.saved = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        return self

    def __exit__(self, *exc):
        precision, allow = self.saved
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = allow
        return False
