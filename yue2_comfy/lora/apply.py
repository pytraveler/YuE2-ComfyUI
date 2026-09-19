"""Folding LoRA files into the half of YuE2 on the card, and taking them out again.

A fold writes ``W + strength * scale * (up @ down)`` into the weights of the
half a stage is about to use, once, when that half is on the card, and every
token after it costs what it cost without an adapter: the CUDA graph, the fused
attention and the sampler do not know anything happened. Measured on an RTX
5090 on 2026-09-19: a half of 196 matrices, 2.62 GiB, folds in 10 to 30 ms,
the same to the bit every time.

The arithmetic is ComfyUI's: the file's own pair is multiplied in float32, the
product scaled by strength times alpha over rank and added to a float32 copy
of the weight, and the sum rounded to the model's dtype once. For a fused pair
the whole fused product is made and then cut into its rows, which is what
ComfyUI computes before it cuts nothing. Several adapters on one weight are
added in the order ``choices`` sorts them into, then rounded once. Measured on
the published files: rounding keeps the adapter's whole difference for typical
ones (1.5 to 4 percent of the weight) at a noise of 2 to 6 percent of it, and
97.4 percent of the gentlest, jpop-t4, whose noise is 0.13 percent of the
weight -- as much as bf16's own rounding of it.

Taking an adapter out needs the weights it was folded into, and those are the
checkpoint itself. A half on the card under 'auto' or 'on' still has its
file-backed homes, so sending it off the card and back leaves its own weights
there, in a second or less. Where there are no homes -- under 'off', or for
weights that are not file-backed -- ``placement`` keeps the CPU copy a half
came from as it goes onto the card for a run with an adapter, and that copy
is the way back. A half resident since a run without one has no such copy,
and gets one here, once, before the first fold.

Two kinds of weight are not folded. The acoustic modules outside the layers --
llm2vae, vae2llm and the time embedder -- never leave the card, so they keep a
small copy of their own on the side. Layers packed as INT8 rows by
``low_vram`` are not folded at all: the adapter is held beside the rows, see
``quantized.Packed``.

On a run on the CPU nothing is written in place, since the CPU weights are
views of the checkpoint file: a folded tensor takes the place of the view, and
the view is what taking the adapter out puts back.
"""

from __future__ import annotations

import contextlib
import logging
import time

import torch

from .. import placement, quantized, runtime
from . import catalogue, choices, formats

log = logging.getLogger(__name__)

_READINGS: dict = {}


def reading(path: str) -> formats.Reading:
    """The header of a chosen file, read once per file identity, refused when it cannot fold."""
    mark = catalogue.stamp(path)
    found = _READINGS.get(mark) if mark else None
    if found is None:
        found = formats.read(path)
        if mark:
            _READINGS[mark] = found
    if not found.usable:
        raise ValueError("{} cannot be used: {}".format(path, found.problem))
    return found


def _locate(lm, name: str):
    """The module holding one tensor of the released checkpoint, and the tensor's name in it."""
    owner, attribute = name.rsplit(".", 1)
    module = lm
    for step in owner.split("."):
        module = getattr(module, step)
    return module, attribute


class Plan:
    """What one half gets from the chosen adapters, target by target.

    'plain' holds the tensors to fold, 'packed' the INT8 layers to hold an
    adapter beside, 'outside' the acoustic modules outside the layers. Each
    maps a target to its ``(choice, part, piece)`` items in folding order;
    'modules' maps a packed target to its layer.
    """

    def __init__(self, lm, half: str, chosen):
        self.plain, self.packed, self.outside, self.modules = {}, {}, {}, {}
        for choice in chosen:
            for part in reading(choice.path).parts:
                for piece in part.pieces:
                    if piece.half != half:
                        continue
                    item = (choice, part, piece)
                    if not formats.in_layers(piece.target):
                        self.outside.setdefault(piece.target, []).append(item)
                        continue
                    module, _attribute = _locate(lm, piece.target)
                    if isinstance(module, quantized.Packed):
                        self.modules[piece.target] = module
                        self.packed.setdefault(piece.target, []).append(item)
                    else:
                        self.plain.setdefault(piece.target, []).append(item)


class Sources:
    """The chosen files, opened once for one fold and closed after it."""

    def __init__(self, stack):
        self.stack = stack
        self.handles = {}

    def tensor(self, path: str, key: str):
        handle = self.handles.get(path)
        if handle is None:
            from safetensors import safe_open

            handle = self.stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            self.handles[path] = handle
        return handle.get_tensor(key)


class Products:
    """The last low-rank product made, since a fused pair feeds two or three targets in a row."""

    def __init__(self):
        self.key = None
        self.value = None

    def get(self, key, make):
        if key != self.key:
            self.value = None
            self.value = make()
            self.key = key
        return self.value


def _difference(half, item, base, sources, products, device):
    """One item's difference to its target, float32 on ``device``, the target's shape."""
    choice, part, piece = item
    strength = float(choice.strength(half))
    rows = slice(piece.start, piece.stop) if piece.start is not None else slice(None)
    if part.kind == formats.LOWRANK:
        def make():
            up = sources.tensor(choice.path, part.up_key).to(device=device, dtype=torch.float32)
            down = sources.tensor(choice.path, part.down_key).to(device=device, dtype=torch.float32)
            product = torch.mm(up, down)
            product.mul_(strength * float(part.scale))
            return product

        return products.get((choice.path, part.module, strength), make)[rows]
    key = part.weight_key if piece.target.endswith(".weight") else part.bias_key
    value = sources.tensor(choice.path, key).to(device=device, dtype=torch.float32)[rows]
    if part.kind == formats.DIFF:
        return value * strength
    return (value - base) * strength


def _folded(half, items, current, sources, products, device):
    """The target's new value: its own weight plus every item's difference, rounded once."""
    base = current.to(device=device, dtype=torch.float32, copy=True)
    total = base.clone() if any(part.kind == formats.REPLACE for _c, part, _p in items) else base
    for item in items:
        total.add_(_difference(half, item, base, sources, products, device))
    return total.to(current.dtype)


def _keep_pristine(owner, attribute: str, parameter: bool, current) -> None:
    """Make sure the tensor about to be folded has a way back, cloning it off the card if it must."""
    if placement._home(owner, attribute, current) is not None:
        return
    if current.device.type == "cpu":
        placement._pristines(owner)[attribute] = current
    else:
        placement._pristines(owner)[attribute] = current.detach().to("cpu")


def _fold_plain(lm, half, plan, sources, device) -> list:
    """Fold into the plain tensors of the half, in place on the card, swapped in on the CPU."""
    touched = []
    products = Products()
    for target, items in plan.plain.items():
        owner, attribute = _locate(lm, target)
        parameter = attribute in owner._parameters
        current = placement._held(owner, attribute, parameter)
        _keep_pristine(owner, attribute, parameter, current)
        new = _folded(half, items, current, sources, products, device)
        if current.device.type == "cpu":
            placement._hold(owner, attribute, parameter, new)
        else:
            current.copy_(new)
        touched.append((owner, attribute, parameter))
    return touched


def _factor(half, plan, sources) -> None:
    """Hold the chosen adapters beside the packed layers of the half, stacked along the rank."""
    for target, items in plan.packed.items():
        module = plan.modules[target]
        device = module.rows.device
        downs, ups = [], []
        for choice, part, piece in items:
            if part.kind != formats.LOWRANK:
                raise ValueError(
                    "{} changes whole matrices ({}), and 'low_vram' keeps them packed as INT8. "
                    "Turn 'low_vram' off in YuE2 Options to sing with it.".format(choice.name, part.module))
            down = sources.tensor(choice.path, part.down_key).to(device=device, dtype=torch.float32)
            up = sources.tensor(choice.path, part.up_key).to(device=device, dtype=torch.float32)
            if piece.start is not None:
                up = up[piece.start:piece.stop]
            downs.append(down)
            ups.append(up * (float(choice.strength(half)) * float(part.scale)))
        dtype = module.scales.dtype
        module.adapt(torch.cat(downs, 0).to(dtype), torch.cat(ups, 1).to(dtype))


def _packed_in(lm, half) -> list:
    blocks = quantized.AR_BLOCKS if half == formats.AR else quantized.NAR_BLOCKS
    return [child for _block, _leaf, child in quantized._packed(lm, blocks)]


def _take_out(keeper, lm, half, homes: bool, touched) -> None:
    """Put a half's own weights back: off the card and on again, or swapped back on the CPU."""
    if getattr(keeper.device, "type", "cpu") != "cpu":
        keeper.cycle(half, homes)
        return
    for owner, attribute, parameter in touched:
        pristine = owner.__dict__.get("_yue2_pristine", {}).get(attribute)
        if pristine is not None:
            placement._hold(owner, attribute, parameter, pristine)


def _put(lm, target: str, value) -> None:
    """Write a whole tensor of the model: in place on the card, swapped in on the CPU."""
    owner, attribute = _locate(lm, target)
    parameter = attribute in owner._parameters
    current = placement._held(owner, attribute, parameter)
    if current.device.type == "cpu":
        placement._hold(owner, attribute, parameter, value.to("cpu"))
    else:
        current.copy_(value)


def _restore_outside(keeper, lm) -> None:
    """Put back every acoustic module outside the layers an adapter was folded into."""
    for target, pristine in keeper.lora.outside_pristine.items():
        _put(lm, target, pristine)


def _outside(keeper, lm, plan, sources, half) -> None:
    """Fold into the acoustic modules outside the layers, from the copy each keeps on the side.

    On the card the copy is a clone in RAM; on the CPU it is the model's own
    tensor, which the folded one only takes the place of.
    """
    _restore_outside(keeper, lm)
    products = Products()
    for target, items in plan.outside.items():
        owner, attribute = _locate(lm, target)
        parameter = attribute in owner._parameters
        current = placement._held(owner, attribute, parameter)
        if target not in keeper.lora.outside_pristine:
            keeper.lora.outside_pristine[target] = (
                current if current.device.type == "cpu" else current.detach().to("cpu", copy=True))
        _put(lm, target, _folded(half, items, current, sources, products, current.device))


def ensure(models, keeper, half: str, homes: bool = True) -> None:
    """Leave exactly the run's adapters in ``half``, folding or taking out only what changed.

    Called by ``placement.arrange`` after every move a stage asked for. The
    record of what is folded lives on the model (``placement.LoraState``), so a
    model kept loaded between runs is folded again only when the adapters or
    their strengths changed.
    """
    lm = models.lm
    chosen = choices.touching(getattr(models, "loras", None) or (), half)
    wanted = choices.signature(chosen, half)
    plan = Plan(lm, half, chosen) if chosen else None
    plain = wanted if plan and plan.plain else ()
    packed = wanted if plan and plan.packed else ()
    outside = wanted if plan and plan.outside else ()
    device = keeper.device if getattr(keeper.device, "type", "cpu") != "cpu" else torch.device("cpu")
    work = (keeper.lora.folded.get(half, ()) != plain or keeper.lora.factored.get(half, ()) != packed
            or (half == formats.NAR and keeper.lora.outside != outside))
    if not work:
        return
    started = time.perf_counter()
    with torch.no_grad(), runtime.deterministic_math(), contextlib.ExitStack() as stack:
        sources = Sources(stack)
        if keeper.lora.folded.get(half, ()) != plain:
            if keeper.lora.folded.get(half):
                _take_out(keeper, lm, half, homes, keeper.lora.touched.get(half, ()))
            keeper.lora.folded.pop(half, None)
            keeper.lora.touched.pop(half, None)
            if plain:
                keeper.lora.touched[half] = _fold_plain(lm, half, plan, sources, device)
                keeper.lora.folded[half] = plain
        if keeper.lora.factored.get(half, ()) != packed:
            for module in _packed_in(lm, half):
                module.adapt(None, None)
            keeper.lora.factored.pop(half, None)
            if packed:
                _factor(half, plan, sources)
                keeper.lora.factored[half] = packed
        if half == formats.NAR and keeper.lora.outside != outside:
            if outside:
                _outside(keeper, lm, plan, sources, half)
            else:
                _restore_outside(keeper, lm)
            keeper.lora.outside = outside
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    names = ", ".join(choice.name or choice.path for choice in chosen) or "none"
    log.info("[yue2_comfy.lora] the %s half now holds %s (%.2f s)", half.upper(), names,
             time.perf_counter() - started)
