"""SheetSage2's network in plain torch: a MERT-v2 encoder and a BART-style decoder.

The shapes come from the released weights and the published architecture.
Module names follow the weight file's own keys, so the weights load without a
rename table.

The encoder turns 300 seconds of 24 kHz audio into 7500 frames. It takes
128-bin log-mel frames at 100 per second, has three ConvNeXt stages halve
them twice, and passes the result through 24 Conformer blocks with rotary
positions. What the decoder reads is a softmax-weighted mix of the stage
output and every block's output, projected from 1024 to 512 wide.

The decoder is six post-norm layers of self-attention, attention over those
frames and a feed-forward, with learned positions offset by two. It writes
tokens greedily under ``grammar.Grammar``, one step at a time, with its own
keys and values kept in a preallocated cache.

Nothing here keeps a layer's output once the next layer has it. That, and
running under inference mode, is what keeps a 300-second window in a few
gigabytes where a graph-keeping run needs sixteen.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import grammar, vocab

SAMPLE_RATE = 24000
WINDOW_SAMPLES = int(vocab.WINDOW_SECONDS) * SAMPLE_RATE
N_FFT = 2048
HOP = 240
MELS = 128
WIDTH = 1024
HEADS = 16
BLOCKS = 24
DECODER_WIDTH = 512
DECODER_HEADS = 8
DECODER_LAYERS = 6
MAX_TOKENS = 5120


class exact_float32:
    """Full-precision float32 matrix products for the length of a transcription, restored after.

    ComfyUI processes often run with TF32 allowed for float32 matmul -- a node
    pack sets ``float32_matmul_precision('high')`` and it stays set for every
    node after it. The mel front end multiplies in float32, and under TF32 its
    frames already differ in their last bits, which the encoder then carries
    into different near-tie tokens. Measured in a live ComfyUI: 88 percent of
    mel values identical, and a score that no longer matched. The setting is
    the process's, so it is put back exactly as it was found.
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


def attention_kernels():
    """Prefer cuDNN's attention kernel, then flash, efficient and math.

    The kernels agree only to the last bits in bfloat16, and near-ties between
    two tokens are decided by those bits. For the encoder's 7500-frame
    attention, cuDNN first is what the reference transcriptions ran with: with
    it this network's frames match them exactly, with the default order they
    drift by a few steps of bfloat16. Older torch without the priority API
    keeps its own choice.
    """
    import contextlib

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        order = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                 SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        return sdpa_kernel(order, set_priority=True)
    except (ImportError, AttributeError, TypeError):
        return contextlib.nullcontext()


class Frontend(nn.Module):
    """Power log-mel frames, normalised per bin with the statistics stored in the weights."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.zeros(N_FFT), persistent=False)
        self.register_buffer("filters", torch.zeros(N_FFT // 2 + 1, MELS), persistent=False)
        self.register_buffer("mean", torch.zeros(MELS), persistent=False)
        self.register_buffer("std", torch.ones(MELS), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(waveform.float(), N_FFT, hop_length=HOP, win_length=N_FFT,
                              window=self.window, center=True, pad_mode="reflect",
                              normalized=False, onesided=True, return_complex=True)
        power = spectrum.abs().pow(2.0)
        mel = torch.matmul(power.transpose(-1, -2), self.filters)
        mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        mel = mel[:, :-1, :]
        return (mel - self.mean) / self.std.clamp_min(1e-5)


class ResponseNorm(nn.Module):
    """Global response normalisation across time, scaled by the channel mean."""

    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, width))
        self.bias = nn.Parameter(torch.zeros(1, 1, width))

    def forward(self, x):
        magnitude = torch.norm(x, p=2, dim=1, keepdim=True)
        scale = magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-6)
        return self.weight * (x * scale) + self.bias + x


class ConvNextLayer(nn.Module):
    """A depthwise kernel-7 convolution, then a widening feed-forward, added back."""

    def __init__(self, width: int):
        super().__init__()
        self.depthwise_block = nn.ModuleList([nn.Identity(), nn.Conv1d(width, width, 7, padding=3, groups=width)])
        self.pointwise_block = nn.ModuleList([nn.LayerNorm(width, eps=1e-6), nn.Linear(width, 4 * width),
                                              nn.Identity(), ResponseNorm(4 * width), nn.Linear(4 * width, width)])

    def forward(self, x):
        y = self.depthwise_block[1](x.transpose(1, 2)).transpose(1, 2)
        block = self.pointwise_block
        y = block[4](block[3](F.gelu(block[1](block[0](y)))))
        return x + y


class ConvNextStage(nn.Module):
    """An optional halving convolution, then ConvNeXt layers at the new width."""

    def __init__(self, width_in: int, width_out: int, depth: int, halve: bool):
        super().__init__()
        self.resampling_layer = (nn.ModuleList([nn.LayerNorm(width_in, eps=1e-6), nn.Identity(),
                                                nn.Conv1d(width_in, width_out, 2, stride=2)])
                                 if halve else None)
        self.convnext_layers = nn.ModuleList([ConvNextLayer(width_out) for _ in range(depth)])

    def forward(self, x):
        if self.resampling_layer is not None:
            x = self.resampling_layer[0](x)
            x = self.resampling_layer[2](x.transpose(1, 2)).transpose(1, 2)
        for layer in self.convnext_layers:
            x = layer(x)
        return x


def rotary(frames: int, device, dtype) -> tuple:
    """Cosine and sine tables for rotary positions, shaped to broadcast over ``[batch, time, heads, dim]``.

    The tables hold values rounded to ``dtype`` but are returned as float32:
    the rotation is applied in float32 and the result cast back, which is the
    arithmetic the released model's reference runs with.
    """
    head = WIDTH // HEADS
    inverse = 1.0 / (10000 ** (torch.arange(0, head, 2, dtype=torch.float32, device=device) / head))
    positions = torch.arange(frames, dtype=torch.float32, device=device)
    angles = torch.outer(positions, inverse)
    angles = torch.cat((angles, angles), dim=-1)
    return (angles.cos().to(dtype).float()[None, :, None, :],
            angles.sin().to(dtype).float()[None, :, None, :])


def _rotate(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_1 = nn.Linear(WIDTH, 4 * WIDTH)
        self.w_2 = nn.Linear(4 * WIDTH, WIDTH)

    def forward(self, x):
        return self.w_2(F.gelu(self.w_1(x)))


class EncoderAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(WIDTH, WIDTH)
        self.key_proj = nn.Linear(WIDTH, WIDTH)
        self.value_proj = nn.Linear(WIDTH, WIDTH)
        self.out_proj = nn.Linear(WIDTH, WIDTH)

    def forward(self, x, cos, sin):
        batch, frames, _ = x.shape
        shape = (batch, frames, HEADS, WIDTH // HEADS)
        q = self.query_proj(x).reshape(shape).float()
        k = self.key_proj(x).reshape(shape).float()
        v = self.value_proj(x).reshape(shape)
        q = (q * cos + _rotate(q) * sin).to(v.dtype)
        k = (k * cos + _rotate(k) * sin).to(v.dtype)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out_proj(out.transpose(1, 2).reshape(batch, frames, WIDTH))


class ConvolutionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(WIDTH, eps=1e-5)
        self.conv_block = nn.ModuleList([
            nn.Identity(), nn.Conv1d(WIDTH, 2 * WIDTH, 1, bias=False), nn.Identity(),
            nn.Conv1d(WIDTH, WIDTH, 31, padding=15, groups=WIDTH, bias=False),
            nn.ModuleList([nn.Identity(), nn.LayerNorm(WIDTH, eps=1e-5)]), nn.Identity(),
            nn.Conv1d(WIDTH, WIDTH, 1, bias=False)])

    def forward(self, x):
        block = self.conv_block
        y = block[1](self.layer_norm(x).transpose(1, 2))
        y = block[3](F.glu(y, dim=1))
        y = F.gelu(block[4][1](y.transpose(1, 2)))
        return block[6](y.transpose(1, 2)).transpose(1, 2)


class ConformerBlock(nn.Module):
    """Half a feed-forward, attention, convolution, the other half, then a final norm."""

    def __init__(self):
        super().__init__()
        self.ffn1_layer_norm = nn.LayerNorm(WIDTH, eps=1e-5)
        self.ffn1 = FeedForward()
        self.attn_layer_norm = nn.LayerNorm(WIDTH, eps=1e-5)
        self.attn = EncoderAttention()
        self.conv_module = ConvolutionModule()
        self.ffn2_layer_norm = nn.LayerNorm(WIDTH, eps=1e-5)
        self.ffn2 = FeedForward()
        self.final_layer_norm = nn.LayerNorm(WIDTH, eps=1e-5)

    def forward(self, x, cos, sin):
        x = x + 0.5 * self.ffn1(self.ffn1_layer_norm(x))
        x = self.attn(self.attn_layer_norm(x), cos, sin) + x
        x = self.conv_module(x) + x
        x = x + 0.5 * self.ffn2(self.ffn2_layer_norm(x))
        return self.final_layer_norm(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling_module = nn.ModuleList([ConvNextStage(MELS, MELS, 3, False),
                                                 ConvNextStage(MELS, 512, 4, True),
                                                 ConvNextStage(512, WIDTH, 5, True)])
        self.layers = nn.ModuleList([ConformerBlock() for _ in range(BLOCKS)])


class DecoderAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(DECODER_WIDTH, DECODER_WIDTH)
        self.k_proj = nn.Linear(DECODER_WIDTH, DECODER_WIDTH)
        self.v_proj = nn.Linear(DECODER_WIDTH, DECODER_WIDTH)
        self.out_proj = nn.Linear(DECODER_WIDTH, DECODER_WIDTH)

    def heads(self, x):
        return x.view(x.shape[0], x.shape[1], DECODER_HEADS, DECODER_WIDTH // DECODER_HEADS).transpose(1, 2)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DecoderAttention()
        self.self_attn_layer_norm = nn.LayerNorm(DECODER_WIDTH)
        self.encoder_attn = DecoderAttention()
        self.encoder_attn_layer_norm = nn.LayerNorm(DECODER_WIDTH)
        self.fc1 = nn.Linear(DECODER_WIDTH, 4 * DECODER_WIDTH)
        self.fc2 = nn.Linear(4 * DECODER_WIDTH, DECODER_WIDTH)
        self.final_layer_norm = nn.LayerNorm(DECODER_WIDTH)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_positions = nn.Embedding(MAX_TOKENS + 2, DECODER_WIDTH)
        self.layernorm_embedding = nn.LayerNorm(DECODER_WIDTH)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(DECODER_LAYERS)])


class Network(nn.Module):
    """The whole model. ``encode`` gives a window's frames, ``generate`` its tokens."""

    def __init__(self):
        super().__init__()
        self.frontend = Frontend()
        self.encoder = Encoder()
        self.layer_weight = nn.Parameter(torch.zeros(BLOCKS + 1))
        self.encoder_projection = nn.Linear(WIDTH, DECODER_WIDTH)
        self.decoder = Decoder()
        self.token_embedding = nn.Embedding(vocab.SIZE, DECODER_WIDTH)
        self.output_projection = nn.Linear(DECODER_WIDTH, vocab.SIZE, bias=False)
        self._masks = {}

    @property
    def dtype(self):
        return self.encoder_projection.weight.dtype

    def encode(self, window: torch.Tensor) -> torch.Tensor:
        """``[1, 7200000]`` samples to ``[1, 7500, 512]`` frames for the decoder."""
        with exact_float32(), attention_kernels():
            return self._encode(window)

    def _encode(self, window: torch.Tensor) -> torch.Tensor:
        mel = self.frontend(window)
        x = mel.to(self.dtype)
        del mel
        for stage in self.encoder.subsampling_module:
            x = stage(x)
        weights = torch.softmax(self.layer_weight, dim=0)
        cos, sin = rotary(x.shape[1], x.device, x.dtype)
        mixed = x * weights[0]
        for weight, block in zip(weights[1:], self.encoder.layers):
            x = block(x, cos, sin)
            mixed.add_(x * weight)
        del x
        return self.encoder_projection(mixed)

    def _mask(self, ranges: list, device) -> torch.Tensor:
        key = tuple(ranges)
        mask = self._masks.get(key)
        if mask is None or mask.device != device:
            mask = torch.full((vocab.SIZE,), float("-inf"), device=device)
            for start, end in ranges:
                mask[start:end] = 0.0
            self._masks[key] = mask
        return mask

    def _cross(self, memory: torch.Tensor) -> list:
        cache = []
        for layer in self.decoder.layers:
            attention = layer.encoder_attn
            cache.append((attention.heads(attention.k_proj(memory)), attention.heads(attention.v_proj(memory))))
        return cache

    def _step(self, ids: torch.Tensor, position: int, keys: list, values: list, cross: list) -> torch.Tensor:
        length = ids.shape[1]
        x = self.token_embedding(ids) + self.decoder.embed_positions.weight[position + 2:position + 2 + length]
        x = self.decoder.layernorm_embedding(x)
        end = position + length
        for index, layer in enumerate(self.decoder.layers):
            attention = layer.self_attn
            q = attention.heads(attention.q_proj(x))
            keys[index][:, :, position:end] = attention.heads(attention.k_proj(x))
            values[index][:, :, position:end] = attention.heads(attention.v_proj(x))
            out = F.scaled_dot_product_attention(q, keys[index][:, :, :end], values[index][:, :, :end],
                                                 is_causal=length > 1 and position == 0)
            x = layer.self_attn_layer_norm(x + attention.out_proj(out.transpose(1, 2).reshape(x.shape)))
            attention = layer.encoder_attn
            q = attention.heads(attention.q_proj(x))
            out = F.scaled_dot_product_attention(q, cross[index][0], cross[index][1])
            x = layer.encoder_attn_layer_norm(x + attention.out_proj(out.transpose(1, 2).reshape(x.shape)))
            x = layer.final_layer_norm(x + layer.fc2(F.gelu(layer.fc1(x))))
        return self.output_projection(x[:, -1])

    def generate(self, memory: torch.Tensor, prefix: list, stop_seconds, cancelled=None, progress=None) -> list:
        """A window's tokens: the prefix, then greedy choices under the grammar, ending in eos.

        Decoding ends at eos, at the token limit, or as soon as a time stamp
        reaches ``stop_seconds`` -- the part after it belongs to the next window.
        ``cancelled`` is asked every 64 tokens and ``progress`` told the count.

        The decoder keeps torch's own choice of attention kernel: unlike the
        encoder's, its reference steps were taken with that choice, and cuDNN
        would also rebuild its plan for every new cache length.
        """
        with exact_float32():
            return self._generate(memory, prefix, stop_seconds, cancelled, progress)

    def _generate(self, memory, prefix, stop_seconds, cancelled, progress) -> tuple:
        state = grammar.Grammar().follow(prefix)
        tokens = [int(token) for token in prefix]
        device = memory.device
        head = DECODER_WIDTH // DECODER_HEADS
        keys = [torch.empty(1, DECODER_HEADS, MAX_TOKENS, head, device=device, dtype=memory.dtype)
                for _ in range(DECODER_LAYERS)]
        values = [torch.empty_like(key) for key in keys]
        cross = self._cross(memory)
        logits = self._step(torch.tensor([tokens], device=device), 0, keys, values, cross)
        time_first, time_end = vocab.BLOCKS["time"]
        while len(tokens) < MAX_TOKENS:
            token = int(torch.argmax(logits.float()[0] + self._mask(state.allowed(), device)))
            tokens.append(token)
            finished = state.update(token)
            if (not finished and stop_seconds is not None and time_first <= token < time_end
                    and (token - time_first) / vocab.TIME_HZ >= stop_seconds):
                tokens.append(vocab.EOS)
                finished = True
            if finished:
                break
            if len(tokens) % 64 == 0:
                if cancelled is not None and cancelled():
                    raise InterruptedError("transcription cancelled")
                if progress is not None:
                    progress(len(tokens))
            logits = self._step(torch.tensor([[token]], device=device), len(tokens) - 1, keys, values, cross)
        cut = tokens[-1] != vocab.EOS
        if cut:
            tokens.append(vocab.EOS)
        return tokens, cut
