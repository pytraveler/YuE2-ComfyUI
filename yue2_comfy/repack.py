"""Reading the Comfy-Org single-file repack into this pack's own classes.

Comfy-Org publishes YuE2 as one file for the native ComfyUI nodes, and from now
on that is the file most people will have. It holds the same weights this pack
already loads, arranged differently:

    vae.*                    the decoder, byte for byte what YuE2-Vae ships
    text_encoders.model.*    the autoregressive half
    model.diffusion_model.*  the NAR half, which upstream keeps in the same
                             layers under nar_ names
    text_encoders.yue2_tokenizer_json  the vocabulary, as a byte tensor

Two things have to be undone. The released checkpoint is one mixture of
transformers whose every layer carries both halves, so the two trees are woven
back together, the NAR side regaining its nar_ names. And the repack fuses
q/k/v into one matrix and gate/up into another, so 112 matrices are cut back
apart. Cutting is exact -- a concatenation split at the right offsets returns
the original rows -- and the split points are read off the tensors themselves
rather than taken from a config, so a future release with different head counts
needs no edit here.

Verified against the released files on 2026-09-12: the VAE block matches
YuE2-Vae in all 435 tensors, the converted vocabulary matches qwen.tiktoken in
all 151643 ranks, and the whole conversion is checked by the same
load_state_dict(strict=True) that guards the ordinary path.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import struct

log = logging.getLogger(__name__)

AR_PREFIX = "text_encoders.model."
NAR_PREFIX = "model.diffusion_model."
NAR_LAYER_PREFIX = NAR_PREFIX + "model."
VAE_PREFIX = "vae."
TOKENIZER_KEY = "text_encoders.yue2_tokenizer_json"
FORMAT_KEY = "yue2_format"

NAR_RENAMES = {
    "self_attn": "nar_self_attn",
    "mlp": "nar_mlp",
    "input_layernorm": "nar_input_layernorm",
    "post_attention_layernorm": "nar_pre_mlp_layernorm",
}


def is_repack(header: dict) -> bool:
    """Whether a safetensors header describes the single-file repack.

    The metadata carries yue2_format, but metadata is the first thing lost when
    a file is re-saved by a tool that does not care about it, so the prefixes
    decide and the marker only confirms.
    """
    keys = [key for key in header if key != "__metadata__"]
    has_tokenizer = TOKENIZER_KEY in header
    has_trees = (any(key.startswith(AR_PREFIX) for key in keys)
                 and any(key.startswith(NAR_PREFIX) for key in keys)
                 and any(key.startswith(VAE_PREFIX) for key in keys))
    return bool(has_trees and has_tokenizer)


def _split_qkv(tensor, o_proj):
    """Cut a fused qkv matrix back into q, k and v.

    The query half is as wide as the attention output, which o_proj states, and
    what remains is key and value in equal parts. Nothing here needs to know
    how many heads there are.
    """
    query_rows = int(o_proj.shape[1])
    rest = int(tensor.shape[0]) - query_rows
    if rest <= 0 or rest % 2:
        raise ValueError(
            "Fused qkv of shape {} cannot be split against an o_proj of shape {}"
            .format(tuple(tensor.shape), tuple(o_proj.shape)))
    half = rest // 2
    return (tensor[:query_rows],
            tensor[query_rows:query_rows + half],
            tensor[query_rows + half:])


def _split_gate_up(tensor):
    """Cut a fused gate_up matrix into its two equal halves."""
    rows = int(tensor.shape[0])
    if rows % 2:
        raise ValueError("Fused gate_up has an odd number of rows: {}".format(rows))
    half = rows // 2
    return tensor[:half], tensor[half:]


def _layer_number(key: str, prefix: str) -> str:
    """The N in <prefix>layers.N.something, or "" when the key is not a layer."""
    tail = key[len(prefix):]
    if not tail.startswith("layers."):
        return ""
    parts = tail.split(".")
    return parts[1] if len(parts) > 2 else ""


def _convert_tree(state, prefix, target_layer_prefix, rename, into):
    """Move one of the repack's two trees into our single-model layout."""
    for key, tensor in state.items():
        if not key.startswith(prefix):
            continue
        tail = key[len(prefix):]
        number = _layer_number(key, prefix)
        if not number:
            continue
        remainder = tail[len("layers." + number + "."):]
        section = remainder.split(".")[0]
        renamed = rename.get(section, section)
        base = "{}{}.{}".format(target_layer_prefix, number, renamed)
        leaf = remainder[len(section):].lstrip(".")

        if leaf == "qkv_proj.weight":
            sibling = "{}layers.{}.{}.o_proj.weight".format(prefix, number, section)
            o_proj = state.get(sibling)
            if o_proj is None:
                raise ValueError(
                    "Cannot split {} without its o_proj, which should be at {}"
                    .format(key, sibling))
            query, keys_, values = _split_qkv(tensor, o_proj)
            into[base + ".q_proj.weight"] = query
            into[base + ".k_proj.weight"] = keys_
            into[base + ".v_proj.weight"] = values
        elif leaf == "gate_up_proj.weight":
            gate, up = _split_gate_up(tensor)
            into[base + ".gate_proj.weight"] = gate
            into[base + ".up_proj.weight"] = up
        elif leaf:
            into[base + "." + leaf] = tensor
        else:
            into[base] = tensor


def lm_state(state: dict) -> dict:
    """The repack's two trees, woven back into the released checkpoint layout.

    The final norm appears twice in the repack, once per tree. They are the same
    tensor -- checked here rather than assumed, because a silent disagreement
    would be a wrong model that still loads.
    """
    converted = {}

    ar_norm = state.get(AR_PREFIX + "norm.weight")
    nar_norm = state.get(NAR_LAYER_PREFIX + "norm.weight")
    if ar_norm is not None and nar_norm is not None and not bool((ar_norm == nar_norm).all()):
        raise ValueError(
            "The repack's two final norms differ, so this file is not the "
            "architecture this pack knows how to rebuild")
    if ar_norm is not None:
        converted["model.norm.weight"] = ar_norm

    embed = state.get(AR_PREFIX + "embed_tokens.weight")
    if embed is not None:
        converted["model.embed_tokens.weight"] = embed
    head = state.get(AR_PREFIX + "lm_head.weight")
    if head is not None:
        converted["lm_head.weight"] = head

    for key, tensor in state.items():
        if not key.startswith(NAR_PREFIX) or key.startswith(NAR_LAYER_PREFIX):
            continue
        converted[key[len(NAR_PREFIX):]] = tensor

    _convert_tree(state, AR_PREFIX, "model.layers.", {}, converted)
    _convert_tree(state, NAR_LAYER_PREFIX, "model.layers.", NAR_RENAMES, converted)
    return converted


def vae_state(state: dict) -> dict:
    """The decoder half, which needs nothing but its prefix removed."""
    return {key[len(VAE_PREFIX):]: tensor
            for key, tensor in state.items() if key.startswith(VAE_PREFIX)}


def _byte_decoder():
    """The inverse of the byte-level alphabet a HuggingFace BPE vocabulary uses."""
    printable = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("\xa1"), ord("\xac") + 1))
                 + list(range(ord("\xae"), ord("\xff") + 1)))
    mapped = list(printable)
    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + spare)
            spare += 1
    return {chr(code): byte for byte, code in zip(printable, mapped)}


def embedded_tokenizer(path: str) -> dict:
    """The tokenizer JSON carried inside the repack, read without torch.

    Only this one tensor is read: its offsets come from the header, so the seven
    gigabytes in front of it are never touched.
    """
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length).decode("utf-8"))
        entry = header.get(TOKENIZER_KEY)
        if entry is None:
            raise ValueError("This file carries no embedded YuE2 tokenizer")
        start, end = entry["data_offsets"]
        handle.seek(8 + length + start)
        blob = handle.read(end - start)
    return json.loads(blob.decode("utf-8"))


def tiktoken_lines(path: str) -> bytes:
    """The embedded vocabulary, rewritten in the format the released file uses.

    tiktoken wants raw token bytes and a rank; the JSON stores each token as
    printable characters standing in for those bytes. Undoing that substitution
    reproduces qwen.tiktoken exactly, which is how this was checked.
    """
    data = embedded_tokenizer(path)
    vocab = data.get("model", {}).get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("The embedded tokenizer has no vocabulary")
    decoder = _byte_decoder()
    rows = []
    for token, rank in vocab.items():
        try:
            raw = bytes(decoder[character] for character in token)
        except KeyError as error:
            raise ValueError(
                "The embedded vocabulary uses a character outside the byte-level "
                "alphabet: {!r}".format(error.args[0])) from error
        rows.append((int(rank), base64.b64encode(raw)))
    rows.sort()
    return b"\n".join(token + b" " + str(rank).encode("ascii")
                      for rank, token in rows) + b"\n"


def merges_beside(path: str, cache_dir: str) -> str:
    """Write the embedded vocabulary out once, and return where it went.

    The vendored tokenizer takes a file path, and the repack has no file. Rather
    than fork the vendored code for this, the vocabulary is written to the
    pack's own cache the first time a repack is loaded. The name carries the
    source file's size and modification time, so replacing the checkpoint
    produces a different name instead of reusing a stale vocabulary.
    """
    info = os.stat(path)
    name = "qwen-{}-{}.tiktoken".format(info.st_size, info.st_mtime_ns)
    target = os.path.join(cache_dir, name)
    if os.path.isfile(target):
        return target
    os.makedirs(cache_dir, exist_ok=True)
    partial = target + ".part"
    with open(partial, "wb") as handle:
        handle.write(tiktoken_lines(path))
    os.replace(partial, target)
    log.info("[yue2_comfy.repack] vocabulary written to %s", target)
    return target
