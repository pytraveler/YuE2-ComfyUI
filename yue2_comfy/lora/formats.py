"""What a LoRA file for YuE2 holds, read from its header alone.

Two layouts are in use, and a file of either works here. ComfyUI's own is the
one Comfy-Org's checkpoint defines: q/k/v fused into ``qkv_proj`` and gate/up
into ``gate_up_proj``, the acoustic half under ``diffusion_model.`` (the MODEL
slot of ComfyUI's LoraLoader) and the half that writes the score and the
performance under ``text_encoders.`` (the CLIP slot). ai-toolkit trains YuE2
into it, and so do several authors on the Hub. m-a-p's layout is the one the
released checkpoint and this pack's model use: separate projections, and the
acoustic half named ``nar_self_attn`` and ``nar_mlp`` inside the same layers.
Trainers write it in at least five spellings.

Measured on 2026-09-19 over the files people had published by then: ComfyUI's
LoraLoader applied three of eight local files and none written in m-a-p's
layout -- those it loads without a word beyond a console line, and the song
comes out as if no adapter were connected. Nothing here is skipped that way. A
part of a file that does not land on the model is a refusal that names it.

A fused projection is cut back into its three or two by rows, exactly as
``repack`` cuts Comfy-Org's checkpoint: q, k and v take 2048, 1024 and 1024
rows, gate and up 6144 each. The file's own matrix is what gets multiplied, and
only the product is cut, so the arithmetic is the same as ComfyUI's.

The scale of a low-rank part is ``alpha / rank`` when the file carries alpha.
When it does not, an ``adapter_config.json`` beside the file may: PEFT writes
``lora_alpha`` and ``r`` there, and one trainer writes ``alpha`` and
``rank``. Measured: an industrial-rock adapter keeps 48 over 32 there, and read
without it, its acoustic half would play at two thirds of the strength it was
trained at. Without either, the scale is 1.

Nothing here imports torch. Alpha is a scalar in the file, and its two or four
bytes are read at the offset the header gives.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import struct

AR, NAR = "ar", "nar"
HALVES = (AR, NAR)

COMFYUI, MAP = "ComfyUI", "m-a-p"

LAYERS = 28
HIDDEN = 2048
FFN = 6144
KV = 1024
HEAD_DIM = 128
LATENT = 64
FREQUENCIES = 256
"""The released architecture, the only one a file is checked against."""

HEADER_LIMIT = 64 * 1024 * 1024

SIDECAR = "adapter_config.json"

LOWRANK, DIFF, REPLACE = "lowrank", "diff", "replace"

ROLES = (
    (".lora_A.default.weight", "down"), (".lora_B.default.weight", "up"),
    (".lora_down.weight", "down"), (".lora_up.weight", "up"),
    (".lora_A.weight", "down"), (".lora_B.weight", "up"),
    (".lora_mid.weight", "lora_mid"),
    (".lora_A", "down"), (".lora_B", "up"),
    (".A", "down"), (".B", "up"),
    (".alpha", "alpha"), (".diff_b", "diff_b"), (".diff", "diff"),
    (".weight", "weight"), (".bias", "bias"),
)
"""Suffix to role, longest first so that ``.lora_A.weight`` is not read as ``.weight``."""

UNSUPPORTED = ("lora_mid", "dora_scale", "hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b",
               "lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b",
               "reshape_weight", "set_weight")
"""Roles of adapters that are not a plain low-rank pair or a difference: Tucker
middles, DoRA, LoHa, LoKr. None was found on a YuE2 file, and each needs
arithmetic of its own, so a file with one is refused by name rather than folded
wrongly."""

NAR_RENAMES = {
    "self_attn": "nar_self_attn",
    "mlp": "nar_mlp",
    "input_layernorm": "nar_input_layernorm",
    "post_attention_layernorm": "nar_pre_mlp_layernorm",
}
"""The acoustic half's names in m-a-p's layout, for a module ComfyUI's layout
keeps under ``diffusion_model.`` -- the table ``repack`` weaves the checkpoint
back together with."""

SHARED = ("model.norm", "model.embed_tokens", "lm_head", "latent_pos_embed")
"""Parts this pack does not change: the final norm both halves read, the two
vocabulary tables, which live in system RAM with only a window on the card, and
the position table. No published file touches them."""

KNOWN_COMPANIONS = {
    "df175dbf9405a8e15b2c3f8dbdcc97303575f763787f227b03029020e28102fe":
        "nar_lora_joint_v4 (Mothersuperior)",
}
"""Acoustic adapters named by the SHA-256 of their ``.pt`` release in the
metadata of artist adapters trained with them, checked on 2026-09-19."""

TRIGGER_KEYS = ("trigger_word", "trigger_words", "trigger", "modelspec.trigger_phrase")

BASE_KEYS = ("base_model", "base_model_name_or_path", "ss_base_model_version",
             "modelspec.architecture")

_LAYER = re.compile(r"^model\.layers\.(\d+)\.([a-z_]+)(?:\.(.*))?$")


def _targets() -> dict:
    """Every tensor a file may change, as name -> (shape, half)."""
    table = {}
    attention = {"q_proj": (HIDDEN, HIDDEN), "k_proj": (KV, HIDDEN), "v_proj": (KV, HIDDEN),
                 "o_proj": (HIDDEN, HIDDEN)}
    mlp = {"gate_proj": (FFN, HIDDEN), "up_proj": (FFN, HIDDEN), "down_proj": (HIDDEN, FFN)}
    for index in range(LAYERS):
        base = "model.layers.{}.".format(index)
        for half, attn, ffn, norms in (
                (AR, "self_attn", "mlp", ("input_layernorm", "post_attention_layernorm")),
                (NAR, "nar_self_attn", "nar_mlp", ("nar_input_layernorm", "nar_pre_mlp_layernorm"))):
            for name, shape in attention.items():
                table[base + attn + "." + name + ".weight"] = (shape, half)
            for name in ("q_norm", "k_norm"):
                table[base + attn + "." + name + ".weight"] = ((HEAD_DIM,), half)
            for name, shape in mlp.items():
                table[base + ffn + "." + name + ".weight"] = (shape, half)
            for name in norms:
                table[base + name + ".weight"] = ((HIDDEN,), half)
    table["llm2vae.weight"] = ((LATENT, HIDDEN), NAR)
    table["llm2vae.bias"] = ((LATENT,), NAR)
    table["vae2llm.weight"] = ((HIDDEN, LATENT), NAR)
    table["vae2llm.bias"] = ((HIDDEN,), NAR)
    table["time_embedder.mlp.0.weight"] = ((HIDDEN, FREQUENCIES), NAR)
    table["time_embedder.mlp.0.bias"] = ((HIDDEN,), NAR)
    table["time_embedder.mlp.2.weight"] = ((HIDDEN, HIDDEN), NAR)
    table["time_embedder.mlp.2.bias"] = ((HIDDEN,), NAR)
    return table


TARGETS = _targets()


def half_of(name: str) -> str:
    """Which half a tensor of the model belongs to, AR or NAR."""
    return TARGETS[name][1]


def in_layers(name: str) -> bool:
    """Whether a tensor sits in the 28 layers, which move with their half, or outside them."""
    return name.startswith("model.layers.")


def split_role(key: str) -> tuple:
    """``(module, role)`` for one tensor name, role None when no suffix is known."""
    for suffix, role in ROLES:
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[:-len(suffix)], role
    for role in UNSUPPORTED:
        if key.endswith("." + role):
            return key[:-len(role) - 1], role
    return key, None


def normalize(module: str) -> tuple:
    """``(layout, path)``: the layout a module name is written in, and its m-a-p path.

    The layout is ComfyUI's when the name carries one of its two tree prefixes,
    m-a-p's otherwise. The path is the module's name in the released
    checkpoint, fused projections still fused -- ``cuts`` takes them apart.
    """
    layout = MAP
    context = None
    for prefix, half in (("model.diffusion_model.", NAR), ("diffusion_model.", NAR),
                         ("text_encoders.", AR)):
        if module.startswith(prefix):
            layout, context, module = COMFYUI, half, module[len(prefix):]
            break
    for prefix in ("base_model.model.", "adapters."):
        if module.startswith(prefix):
            module = module[len(prefix):]
    if "-" in module and "." not in module:
        module = module.replace("-", ".")
    if module.startswith("layers."):
        module = "model." + module
    if context == NAR:
        match = _LAYER.match(module)
        if match:
            section = NAR_RENAMES.get(match.group(2), match.group(2))
            rest = match.group(3)
            module = "model.layers.{}.{}{}".format(match.group(1), section,
                                                   "." + rest if rest else "")
    return layout, module


def cuts(path: str) -> tuple:
    """The modules a path stands for, with the rows of the file's tensor each one takes.

    ``((module, start, stop), ...)``. A module that is not fused is one entry
    with start and stop None.
    """
    if path.endswith(".qkv_proj"):
        base = path[:-len("qkv_proj")]
        return ((base + "q_proj", 0, HIDDEN), (base + "k_proj", HIDDEN, HIDDEN + KV),
                (base + "v_proj", HIDDEN + KV, HIDDEN + 2 * KV))
    if path.endswith(".gate_up_proj"):
        base = path[:-len("gate_up_proj")]
        return ((base + "gate_proj", 0, FFN), (base + "up_proj", FFN, 2 * FFN))
    return ((path, None, None),)


def fused_rows(path: str):
    """How many rows the fused matrix a path names has, or None for one that is not fused."""
    spans = cuts(path)
    return spans[-1][2] if len(spans) > 1 else None


@dataclasses.dataclass(frozen=True)
class Piece:
    """One tensor of the model a part changes, and the rows of the part's tensor it takes."""

    target: str
    start: int | None = None
    stop: int | None = None

    @property
    def half(self) -> str:
        return half_of(self.target)


@dataclasses.dataclass(frozen=True)
class Part:
    """One module of a file, and what it does to the model.

    'kind' is ``lowrank`` (a pair: ``up`` of out x rank, ``down`` of rank x in),
    ``diff`` (a difference added to the weight, and at ``bias_key`` one for the
    bias) or ``replace`` (a whole new weight or bias, turned into a difference
    against the model's own when it is folded). 'scale' multiplies the low-rank
    product before any strength does: alpha over rank, the sidecar's, or 1.
    """

    module: str
    kind: str
    pieces: tuple
    up_key: str = ""
    down_key: str = ""
    weight_key: str = ""
    bias_key: str = ""
    rank: int = 0
    scale: float = 1.0

    @property
    def halves(self) -> tuple:
        return tuple(half for half in HALVES if any(piece.half == half for piece in self.pieces))


@dataclasses.dataclass(frozen=True)
class Reading:
    """What one file is, as far as its header can say.

    'problem' is empty for a file the pack can fold and says why not
    otherwise. 'parts' are in the file's own order. 'metadata' is the file's
    metadata as text; 'triggers', 'intended_cot' and 'companion' are what was
    worth showing from it. 'scale_source' says where the low-rank scale came
    from: ``alpha``, the sidecar's name, or ``1.0``.
    """

    path: str
    layout: str
    parts: tuple
    problem: str = ""
    scale_source: str = ""
    metadata: dict = dataclasses.field(default_factory=dict)
    triggers: tuple = ()
    intended_cot: str = ""
    companion: str = ""

    @property
    def halves(self) -> tuple:
        return tuple(half for half in HALVES
                     if any(half in part.halves for part in self.parts))

    def ranks(self, half: str) -> tuple:
        return tuple(sorted({part.rank for part in self.parts
                             if part.kind == LOWRANK and half in part.halves}))

    def tensors(self, half: str) -> int:
        """How many tensors of the model this file changes in one half."""
        return len({piece.target for part in self.parts for piece in part.pieces
                    if piece.half == half})

    @property
    def usable(self) -> bool:
        return not self.problem and bool(self.parts)


class NotYuE2(ValueError):
    """A file that is not a LoRA for YuE2 at all, as opposed to one that cannot be folded."""


def header(path: str) -> tuple:
    """``(header, data_start)`` of a safetensors file, without reading a tensor."""
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise NotYuE2("shorter than a safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= HEADER_LIMIT:
            raise NotYuE2("not a safetensors file")
        body = handle.read(length)
    if len(body) != length:
        raise NotYuE2("the header is cut short")
    try:
        parsed = json.loads(body.decode("utf-8"))
    except ValueError as error:
        raise NotYuE2("the header is not JSON") from error
    if not isinstance(parsed, dict):
        raise NotYuE2("the header is not a table")
    return parsed, 8 + length


_SCALARS = {"F32": ("<f", 4), "F64": ("<d", 8), "F16": ("<e", 2), "I64": ("<q", 8),
            "I32": ("<i", 4), "I16": ("<h", 2), "BF16": ("", 2)}


def scalar(path: str, entry: dict, data_start: int) -> float:
    """The one number a tensor of one element holds, read straight from the file."""
    dtype = str(entry.get("dtype", ""))
    shape = entry.get("shape") or []
    if math.prod(int(size) for size in shape) != 1 or dtype not in _SCALARS:
        raise ValueError("alpha is not a single number of a known type")
    code, size = _SCALARS[dtype]
    begin, end = (int(value) for value in entry["data_offsets"])
    if end - begin != size:
        raise ValueError("alpha has {} bytes, not {}".format(end - begin, size))
    with open(path, "rb") as handle:
        handle.seek(data_start + begin)
        raw = handle.read(size)
    if len(raw) != size:
        raise ValueError("alpha lies past the end of the file")
    if dtype == "BF16":
        return struct.unpack("<f", b"\x00\x00" + raw)[0]
    return float(struct.unpack(code, raw)[0])


def sidecar(path: str) -> tuple:
    """``(scale, config)`` from an adapter_config.json beside the file, scale None without one.

    PEFT writes ``r`` and ``lora_alpha``, and with rank-stabilised LoRA
    divides by the square root of the rank instead.
    """
    candidate = os.path.join(os.path.dirname(os.path.abspath(path)), SIDECAR)
    try:
        with open(candidate, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return None, {}
    if not isinstance(config, dict):
        return None, {}
    rank = config.get("r", config.get("rank"))
    alpha = config.get("lora_alpha", config.get("alpha"))
    try:
        rank, alpha = float(rank), float(alpha)
    except (TypeError, ValueError):
        return None, config
    if rank <= 0:
        return None, config
    if config.get("use_rslora"):
        return alpha / math.sqrt(rank), config
    return alpha / rank, config


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _made_for_other(metadata: dict, config: dict) -> str:
    """The base model a file names when it names one and none of them is YuE2, else ""."""
    named = [str(source.get(key)) for source in (metadata, config) for key in BASE_KEYS
             if source.get(key)]
    if named and not any("yue" in name.lower() for name in named):
        return named[0]
    return ""


def _triggers(metadata: dict) -> tuple:
    """The trigger words a file's metadata states, in order and once each."""
    found = []
    for key in TRIGGER_KEYS:
        value = metadata.get(key)
        if not value:
            continue
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = value
        items = parsed if isinstance(parsed, list) else str(parsed).split(",")
        for item in items:
            word = str(item).strip()
            if word and word not in found:
                found.append(word)
    return tuple(found)


def _companion(metadata: dict) -> str:
    """The acoustic adapter an artist adapter was trained with, when its metadata names one.

    Two ways were found: a SHA-256 of the adapter, at the top of the metadata
    or inside a ``metadata`` entry that holds JSON of its own, and a file name
    under ``acoustic_adapter``.
    """
    sources = [metadata]
    nested = metadata.get("metadata")
    if nested:
        try:
            inner = json.loads(nested)
        except ValueError:
            inner = None
        if isinstance(inner, dict):
            sources.append(inner)
    for source in sources:
        value = str(source.get("nar_companion_sha256") or "").strip().lower()
        if value:
            return KNOWN_COMPANIONS.get(value, "sha256 " + value[:12])
    adapter = str(metadata.get("acoustic_adapter") or "").strip().replace("\\", "/")
    if adapter:
        stem = adapter.rsplit("/", 1)[-1]
        return stem.split(".", 1)[0] or stem
    return ""


def _shape(entry) -> tuple:
    return tuple(int(size) for size in (entry.get("shape") or []))


def read(path: str) -> Reading:
    """What the LoRA file at ``path`` does to YuE2.

    Raises NotYuE2 when the file is not a LoRA for this model at all -- when
    not one of its tensors lands on YuE2 in name and shape -- which is what
    keeps the image LoRAs of an ordinary install out of the list. Returns a
    Reading with ``problem`` set when it is one but cannot be folded.
    """
    table, data_start = header(path)
    metadata = {str(key): _text(value)
                for key, value in (table.pop("__metadata__", None) or {}).items()}
    side_scale, config = sidecar(path)

    groups = {}
    for key, entry in table.items():
        if isinstance(entry, dict):
            module, role = split_role(key)
            groups.setdefault(module, {})[role] = (key, entry)

    parts, refused, strangers, layouts, sources = [], [], [], set(), set()
    landed = 0
    for module, roles in groups.items():
        layout, mapped = normalize(module)
        odd = sorted(str(role) for role in roles if role is None or role in UNSUPPORTED)
        plain = {role: value for role, value in roles.items()
                 if role is not None and role not in UNSUPPORTED}
        if mapped in SHARED or any(mapped.startswith(name + ".") for name in SHARED):
            refused.append(module + " (a part this pack does not change)")
            continue
        found = _part(path, module, mapped, plain, data_start, side_scale) if plain else None
        if any(role in UNSUPPORTED for role in roles):
            landed += isinstance(found, Part)
            refused.append("{} ({})".format(module, ", ".join(odd)))
        elif odd or found is None:
            strangers.append(module)
        elif isinstance(found, str):
            refused.append(found)
        else:
            landed += 1
            layouts.add(layout)
            parts.append(found)
            if found.kind == LOWRANK:
                sources.add("alpha" if "alpha" in roles
                            else SIDECAR if side_scale is not None else "1.0")

    if not landed:
        raise NotYuE2("no tensor lands on YuE2")

    problem = ""
    other = _made_for_other(metadata, config)
    if other:
        problem = "It was made for {}, not for YuE2.".format(other)
    elif config.get("rank_pattern") or config.get("alpha_pattern"):
        problem = ("Its adapter_config.json gives some modules a rank of their own, which this "
                   "pack does not read.")
    elif refused or strangers:
        names = refused + [name + " (not a part of YuE2)" for name in strangers]
        problem = "{} of its parts do not land on YuE2: {}{}".format(
            len(names), ", ".join(names[:4]), " ..." if len(names) > 4 else "")
    return Reading(path=path, layout=" + ".join(sorted(layouts)), parts=tuple(parts),
                   problem=problem, scale_source=" + ".join(sorted(sources)),
                   metadata=metadata, triggers=_triggers(metadata),
                   intended_cot=str(metadata.get("intended_cot") or "").strip(),
                   companion=_companion(metadata))


def _part(path, module, mapped, roles, data_start, side_scale):
    """One module as a Part, None for a module YuE2 does not have, or a refusal as text."""
    if "up" in roles or "down" in roles:
        return _pair(path, module, mapped, roles, data_start, side_scale)
    if "diff" in roles or "diff_b" in roles:
        return _whole(module, mapped, roles, DIFF, "diff", "diff_b")
    if "weight" in roles or "bias" in roles:
        return _whole(module, mapped, roles, REPLACE, "weight", "bias")
    return None


def _pair(path, module, mapped, roles, data_start, side_scale):
    """A low-rank pair, checked against the shape of every matrix it lands on."""
    spans = cuts(mapped)
    names = [target + ".weight" for target, _start, _stop in spans]
    if any(name not in TARGETS or len(TARGETS[name][0]) != 2 for name in names):
        return None
    if not ("up" in roles and "down" in roles):
        return "{} (half of a pair)".format(module)
    up_key, up = roles["up"]
    down_key, down = roles["down"]
    up_shape, down_shape = _shape(up), _shape(down)
    if len(up_shape) != 2 or len(down_shape) != 2 or up_shape[1] != down_shape[0]:
        return "{} (a pair that does not multiply)".format(module)
    total = fused_rows(mapped)
    if total is not None and up_shape[0] != total:
        return "{} ({} rows, where the fused matrix has {})".format(module, up_shape[0], total)
    pieces = []
    for name, (_target, start, stop) in zip(names, spans):
        rows = up_shape[0] if start is None else stop - start
        want = TARGETS[name][0]
        if (rows, down_shape[1]) != want:
            return "{} (shape {} x {}, where YuE2 has {} x {})".format(
                module, rows, down_shape[1], want[0], want[1])
        pieces.append(Piece(name, start, stop))
    rank = down_shape[0]
    if "alpha" in roles:
        try:
            scale = scalar(path, roles["alpha"][1], data_start) / float(rank)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return "{} (its alpha: {})".format(module, error)
    else:
        scale = side_scale if side_scale is not None else 1.0
    return Part(module=module, kind=LOWRANK, pieces=tuple(pieces), up_key=up_key,
                down_key=down_key, rank=int(rank), scale=float(scale))


def _whole(module, mapped, roles, kind, weight_role, bias_role):
    """A difference or a replacement, checked against the shapes of what it lands on."""
    spans = cuts(mapped)
    total = fused_rows(mapped)
    if total is not None and kind == REPLACE:
        return "{} (a whole fused matrix to replace)".format(module)
    pieces, keys = [], {}
    for role, suffix in ((weight_role, ".weight"), (bias_role, ".bias")):
        if role not in roles:
            continue
        key, entry = roles[role]
        shape = _shape(entry)
        for target, start, stop in spans:
            name = target + suffix
            if name not in TARGETS:
                return None
            if total is not None and shape[:1] != (total,):
                return "{} ({} rows, where the fused matrix has {})".format(
                    module, shape[:1], total)
            took = shape if start is None else (stop - start,) + shape[1:]
            want = TARGETS[name][0]
            if took != want:
                return "{} (shape {}, where YuE2 has {})".format(module, took, want)
            pieces.append(Piece(name, start, stop))
        keys[role] = key
    return Part(module=module, kind=kind, pieces=tuple(pieces),
                weight_key=keys.get(weight_role, ""), bias_key=keys.get(bias_role, ""))


def summary(reading: Reading) -> dict:
    """The facts the node shows under a row, as plain JSON.

    The rank shown is the smallest in the half. A file in ComfyUI's layout
    that was trained on separate projections packs them block-diagonally, so
    its fused q/k/v pair has three times the rank of each projection and its
    gate/up pair twice: jpop-t4 reads 16, 32 and 48 and was trained at 16.
    """
    return {
        "layout": reading.layout,
        "halves": {half: {"tensors": reading.tensors(half),
                          "rank": min(reading.ranks(half)) if reading.ranks(half) else 0}
                   for half in reading.halves},
        "problem": reading.problem,
        "triggers": list(reading.triggers),
        "intended_cot": reading.intended_cot,
        "companion": reading.companion,
        "scale": reading.scale_source,
    }
