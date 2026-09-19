"""Small safetensors files in every spelling YuE2 LoRAs are published in, written without torch.

The tensors are zeros except for the scalars a test sets, because what is
checked here is what a header says. Shapes are the real ones, at a small rank,
so a file weighs kilobytes.
"""

from __future__ import annotations

import json
import math
import struct

SIZES = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I64": 8}

HIDDEN, FFN, KV, LATENT = 2048, 6144, 1024, 64
ATTENTION = {"q_proj": (HIDDEN, HIDDEN), "k_proj": (KV, HIDDEN), "v_proj": (KV, HIDDEN),
             "o_proj": (HIDDEN, HIDDEN)}
MLP = {"gate_proj": (FFN, HIDDEN), "up_proj": (FFN, HIDDEN), "down_proj": (HIDDEN, FFN)}


def scalar_bytes(dtype: str, value: float) -> bytes:
    if dtype == "F32":
        return struct.pack("<f", value)
    if dtype == "F64":
        return struct.pack("<d", value)
    if dtype == "F16":
        return struct.pack("<e", value)
    if dtype == "BF16":
        return struct.pack("<f", value)[2:]
    if dtype == "I64":
        return struct.pack("<q", int(value))
    raise ValueError(dtype)


def write(path, tensors: dict, metadata: dict | None = None) -> None:
    """tensors: name -> (dtype, shape) or (dtype, shape, value) for a one-element tensor."""
    header, blobs, offset = {}, [], 0
    for name, spec in tensors.items():
        dtype, shape = spec[0], list(spec[1])
        size = SIZES[dtype] * math.prod(shape)
        blob = scalar_bytes(dtype, spec[2]) if len(spec) > 2 else bytes(size)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        blobs.append(blob)
        offset += size
    if metadata:
        header["__metadata__"] = {str(key): str(value) for key, value in metadata.items()}
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for blob in blobs:
            handle.write(blob)


def pair(name, shape, rank, down="lora_A", up="lora_B", dtype="F32", alpha=None, alpha_dtype="F32"):
    """The two (or three) tensors of one low-rank module."""
    out, width = shape
    found = {name + "." + down: (dtype, (rank, width)), name + "." + up: (dtype, (out, rank))}
    if alpha is not None:
        found[name + ".alpha"] = (alpha_dtype, (), alpha)
    return found


def map_layer(index, half="ar", prefix="model.", rank=4, down="lora_A", up="lora_B", alpha=None,
              separator="."):
    """One layer's seven projections in m-a-p's layout."""
    attn, ffn = ("self_attn", "mlp") if half == "ar" else ("nar_self_attn", "nar_mlp")
    found = {}
    for block, table in ((attn, ATTENTION), (ffn, MLP)):
        for name, shape in table.items():
            module = prefix + "layers.{}.{}.{}".format(index, block, name)
            if separator != ".":
                module = prefix + separator.join(("model", "layers", str(index), block, name))
            found.update(pair(module, shape, rank, down, up, alpha=alpha))
    return found


def comfy_layer(index, tree="diffusion_model.", rank=4, down="lora_down.weight",
                up="lora_up.weight", alpha=None, block_diagonal=True, dtype="BF16"):
    """One layer in ComfyUI's layout: fused qkv and gate_up, then o and down."""
    base = tree + "model.layers.{}.".format(index)
    qkv_rank = 3 * rank if block_diagonal else rank
    gate_rank = 2 * rank if block_diagonal else rank
    found = {}
    for module, shape, r in ((base + "self_attn.qkv_proj", (HIDDEN + 2 * KV, HIDDEN), qkv_rank),
                             (base + "self_attn.o_proj", (HIDDEN, HIDDEN), rank),
                             (base + "mlp.gate_up_proj", (2 * FFN, HIDDEN), gate_rank),
                             (base + "mlp.down_proj", (HIDDEN, FFN), rank)):
        found.update(pair(module, shape, r, down, up, dtype=dtype,
                          alpha=None if alpha is None else alpha * r / rank))
    return found
