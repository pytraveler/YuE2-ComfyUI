"""Which half of YuE2 sits on the card, decided stage by stage.

YuE2-3B is a mixture of transformers. Each of its 28 layers holds two sets of
weights: one for the autoregressive half, which writes the score and then the
codec tokens, and one for the non-autoregressive half, which turns those
tokens into acoustic latents by flow matching. No stage uses both. The AR half
with the embeddings is 4.03 GiB, the NAR layers are 2.63 GiB, and the modules
the acoustic stage needs besides its layers come to 0.10 GiB.

So the half a stage does not use can wait on the CPU. Measured on an RTX 5090
on 2026-09-13, with the halves swapped around the acoustic stage: a 40-second
song peaked at 4.94 GiB instead of 7.55 before its decode, a 240-second song
at 11.53 GiB instead of 14.94, a swap took about a second, and the audio was
the same to the last byte in both. The weights are the same tensors wherever
they are kept, so there is nothing for the song to differ by.

'off' keeps both halves on the card, which is what the pack always did. 'on'
keeps only the half the running stage needs, and neither during the decode.
'auto' moves a half off only when the stage would not fit beside it, judged
from the memory free just before the stage, and never moves one back for its
own sake: a card with room ends up holding the whole model and moving nothing.

The judgement is made in advance rather than by running out of memory and
trying again. On Windows the NVIDIA driver can put an allocation that does not
fit into shared system memory instead of failing it, and a run that is out of
VRAM then does not stop, it crawls. The retry in ``guarded`` is for the drivers
that do raise.
"""

from __future__ import annotations

import contextlib
import functools
import gc
import logging
import time

log = logging.getLogger(__name__)

AUTO, ON, OFF = "auto", "on", "off"
MODES = (AUTO, ON, OFF)

AR, NAR, SHARED = "ar", "nar", "shared"
HALVES = (AR, NAR)
CARD, CPU, MIXED = "card", "cpu", "mixed"

AR_TOP = ("model.embed_tokens", "lm_head")
AR_LAYER = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
NAR_LAYER = ("nar_input_layernorm", "nar_self_attn", "nar_pre_mlp_layernorm", "nar_mlp")
SHARED_TOP = ("model.norm", "llm2vae", "vae2llm", "time_embedder", "latent_pos_embed")
"""Module names as the vendored model spells them.

AR_TOP and AR_LAYER are exactly what upstream's own ``nar._offload_ar`` moves
off the card, so the split is one upstream already runs. SHARED_TOP is what
the acoustic stage reads outside its layers. A test builds the released model
on the meta device and checks that the three groups hold every parameter and
buffer exactly once."""

GIB = 1024 ** 3

LAYERS, HEADS, KV_HEADS, HEAD_DIM = 28, 16, 8, 128
ELEMENT_BYTES = 2
"""The released architecture, in BF16. The strict load refuses anything else."""

ATTENTION_COPIES = 3
"""Attention matrices of heads x queries x keys counted at a peak.

On this pack's torch the stages pay for attention in whole matrices: the peaks
measured on 2026-09-13 fit about 2.65 of them, for the acoustic prefill of a
240-second song and for its flow matching alike. Three leaves a little room."""

WORK_BASE = int(0.35 * GIB)
"""cuBLAS workspace, sampling buffers and the like. The resident allocation sat
0.29 to 0.34 GiB above the weights in every measured run."""

MARGIN = int(0.5 * GIB)
"""Room for the allocator's rounding, on top of every estimate."""

DECODE_BYTES = 3 * GIB
"""The VAE and its tiles at the released 1024-frame core: 2.78 GiB measured for
a 40-second song, 2.92 GiB for a 240-second one."""


def kv_bytes(tokens: int, branches: int = 1) -> int:
    """The cache for ``tokens`` positions: two BF16 tensors a layer, per branch.

    The same size GraphAR allocates for its capacity, and the same size the
    acoustic prefill keeps for the tokens the flow matching attends to.
    """
    return 2 * LAYERS * int(branches) * int(tokens) * KV_HEADS * HEAD_DIM * ELEMENT_BYTES


def attention_bytes(queries: int, keys: int) -> int:
    """The attention matrices alive at the peak of one pass."""
    return ATTENTION_COPIES * HEADS * int(queries) * int(keys) * ELEMENT_BYTES


def ar_stage_bytes(prefix_tokens: int, budget: int, branches: int = 1) -> int:
    """What the score or the performance allocates beyond the weights.

    The cache for the prompt and the whole budget, the attention of the prompt's
    prefill, and the fixed rest. Measured against it: the score of a 40-second
    song took 0.50 GiB, the performance of a 240-second song 1.61 GiB.
    """
    return (kv_bytes(int(prefix_tokens) + int(budget), branches)
            + attention_bytes(prefix_tokens, prefix_tokens) + WORK_BASE)


def prefill_bytes(ar_tokens: int) -> int:
    """The acoustic prefill: the cache it keeps and the attention it passes through."""
    return kv_bytes(ar_tokens) + attention_bytes(ar_tokens, ar_tokens) + WORK_BASE


def solve_bytes(ar_tokens: int, frames: int) -> int:
    """Flow matching over ``frames``, beyond the prefill cache, which exists by then."""
    positions = int(frames) + 2
    return attention_bytes(positions, int(ar_tokens) + positions) + WORK_BASE


def moves(mode: str, required, where: dict, sizes: dict, available, work: int,
          margin: int = MARGIN) -> list:
    """The moves one stage needs, as ``(half, place)`` pairs, moves off the card first.

    ``required`` is AR, NAR, or None for the decode, which uses neither half.
    ``where`` maps each half to CARD, CPU or MIXED, the last for a move that was
    interrupted part way and is finished rather than trusted. ``available`` is
    None when the device cannot say how much is free, which counts as room.

    Every move off the card comes before every move onto it, so the two halves
    are never on the card together unless the mode means them to be.
    """
    mode = mode if mode in MODES else AUTO
    others = [half for half in HALVES if half != required]
    bring = [required] if required is not None and where[required] != CARD else []
    occupying = [half for half in others if where[half] != CPU]
    if mode == ON:
        return [(half, CPU) for half in occupying] + [(half, CARD) for half in bring]
    if mode == OFF and required is not None:
        return [(half, CARD) for half in bring + [h for h in others if where[h] != CARD]]
    needed = int(work) + int(margin) + sum(sizes[half] for half in bring)
    if available is not None and available < needed:
        return [(half, CPU) for half in occupying] + [(half, CARD) for half in bring]
    return [(half, CARD) for half in bring]


def free_bytes(device):
    """Bytes this process could still allocate on ``device``, or None off CUDA.

    The driver's free memory plus what the caching allocator holds unused, and
    no more than a per-process memory fraction allows when one is set.
    """
    if getattr(device, "type", None) != "cuda":
        return None
    try:
        import torch

        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        available = free + torch.cuda.memory_reserved(device) - allocated
        getter = getattr(torch.cuda.memory, "get_per_process_memory_fraction", None)
        if getter is not None:
            fraction = float(getter(device))
            if fraction < 1.0:
                available = min(available, int(fraction * total) - allocated)
        return max(0, int(available))
    except Exception:
        log.debug("[yue2_comfy.placement] cannot read free VRAM", exc_info=True)
        return None


def _resolve(root, dotted: str):
    return functools.reduce(getattr, dotted.split("."), root)


def _tensors(module) -> list:
    return list(module.parameters()) + list(module.buffers())


def groups(lm) -> dict:
    """The AR half, the NAR layers and the shared modules of one model.

    Raises AttributeError on a model that is not laid out this way, which every
    caller takes as a reason to keep the model whole rather than half moved.
    """
    ar = [_resolve(lm, name) for name in AR_TOP]
    nar = []
    for layer in lm.model.layers:
        ar.extend(getattr(layer, name) for name in AR_LAYER)
        nar.extend(getattr(layer, name) for name in NAR_LAYER)
    return {AR: ar, NAR: nar, SHARED: [_resolve(lm, name) for name in SHARED_TOP]}


class Placement:
    """The groups of one loaded model, where they are, and how to move them."""

    def __init__(self, lm, device):
        self.device = device
        self.groups = groups(lm)
        self.sizes = {half: sum(t.numel() * t.element_size()
                                for module in self.groups[half] for t in _tensors(module))
                      for half in HALVES}

    def _place(self, device) -> str:
        if device.type == "cpu":
            return CPU
        index = getattr(self.device, "index", None)
        if device.type == self.device.type and (index is None or device.index == index):
            return CARD
        return MIXED

    def where(self, half: str) -> str:
        places = {self._place(tensor.device)
                  for module in self.groups[half] for tensor in _tensors(module)}
        return places.pop() if len(places) == 1 else MIXED

    def _move(self, half: str, place: str) -> float:
        import torch

        target = self.device if place == CARD else torch.device("cpu")
        start = time.perf_counter()
        for module in self.groups[half]:
            module.to(target)
        if place == CPU and self.device.type == "cuda":
            torch.cuda.empty_cache()
        return time.perf_counter() - start

    def arrange(self, mode: str, required, work: int, stage: str) -> list:
        """Move what this stage needs moved, and say so in the log."""
        if getattr(self.device, "type", "cpu") == "cpu":
            return []
        where = {half: self.where(half) for half in HALVES}
        available = free_bytes(self.device)
        done = []
        for half, place in moves(mode, required, where, self.sizes, available, work):
            seconds = self._move(half, place)
            done.append("{}->{}".format(half, place))
            reason = ""
            if available is not None:
                reason = ", with {:.1f} GiB free for about {:.1f} GiB of work".format(
                    available / GIB, work / GIB)
            log.info("[yue2_comfy.placement] %s: the %s half %s (%.2f GiB) in %.1f s%s",
                     stage, half.upper(), "onto the card" if place == CARD else "to the CPU",
                     self.sizes[half] / GIB, seconds, reason)
        return done


def placement_for(lm, device) -> Placement:
    """The Placement kept on the model, made on first use."""
    held = getattr(lm, "_yue2_placement", None)
    if held is None or held.device != device:
        held = Placement(lm, device)
        lm._yue2_placement = held
    return held


def load(model, device):
    """A freshly built model, ready for its first stage: the shared modules on the card.

    Both halves stay where the checkpoint was read, on the CPU, and move when a
    stage asks for them. Loading onto the card whole would put all 6.8 GiB
    there at once, which is the one peak no stage could take back.
    """
    try:
        shared = groups(model)[SHARED]
    except AttributeError:
        log.warning("[yue2_comfy.placement] this model is not laid out in two halves; "
                    "loading it onto the card whole")
        return model.to(device)
    for module in shared:
        module.to(device)
    return model


def arrange(models, required, work: int, stage: str, mode=None) -> list:
    """Put the halves where one stage of this run wants them.

    A models object without a real backbone -- the stand-ins the tests use -- is
    left alone, and so is a model that is not laid out in two halves, which the
    loader has put on the card whole.
    """
    lm = getattr(models, "lm", None)
    device = getattr(models, "device", None)
    if lm is None or device is None or isinstance(device, str):
        return []
    try:
        keeper = placement_for(lm, device)
    except AttributeError:
        return []
    return keeper.arrange(mode or getattr(models, "offload", AUTO), required, work, stage)


@contextlib.contextmanager
def acoustic(models, mode=None):
    """Arrange the halves inside upstream's acoustic stage, without editing it.

    ``nar.synthesize`` runs a prefill through the AR layers and then flow
    matching through the NAR ones, chunk by chunk. Its only seams between the
    two are ``CachedNAR``, whose constructor is the prefill, and its own
    ``_offload_ar`` context, entered after the prefill. Both are looked up in
    the module when the function runs, so replacing them for the length of one
    call is enough -- the way ``runtime.pinned_attention`` replaces GraphAR --
    and both are put back afterwards.

    The replacement context moves nothing back on the way out. The next chunk's
    prefill asks for the AR half itself, and so does the next run.
    """
    lm = getattr(models, "lm", None)
    if lm is None:
        yield
        return

    from .vendor.yue2 import nar

    original_engine, original_offload = nar.CachedNAR, nar._offload_ar
    pending = {"solve": WORK_BASE}

    class PlacedNAR(original_engine):
        def __init__(self, model, chunk, *args, **kwargs):
            tokens, frames = len(chunk.ar_tokens), len(chunk.noise)
            pending["solve"] = solve_bytes(tokens, frames)
            arrange(models, AR, prefill_bytes(tokens), "the acoustic prefill", mode)
            super().__init__(model, chunk, *args, **kwargs)

    @contextlib.contextmanager
    def placed(model, enabled):
        arrange(models, NAR, pending["solve"], "flow matching", mode)
        yield

    nar.CachedNAR, nar._offload_ar = PlacedNAR, placed
    try:
        yield
    finally:
        nar.CachedNAR, nar._offload_ar = original_engine, original_offload


def advice(mode: str) -> str:
    """What to change, added to a message that otherwise only says what was missing."""
    if mode == OFF:
        return ("\n\nYuE2 ran out of video memory with the whole model on the card. Set "
                "'offload' to 'auto' or 'on' in YuE2 Options to keep only the half each "
                "stage needs, or lower 'max_seconds'.")
    return ("\n\nYuE2 ran out of video memory with only the half each stage needs on the "
            "card. Lowering 'max_seconds' is what shrinks the rest: the acoustic stage "
            "needs more memory the longer the song is allowed to run.")


def guarded(models, stage: str, call):
    """Run one stage, and in 'auto' once more with offload on if VRAM ran out.

    ``call`` takes the mode to arrange the halves with. Running a stage again is
    safe because every stage starts from its seed: the second attempt produces
    the tokens or latents the first would have. Outside 'auto', or out of memory
    a second time, the error goes up with a sentence saying what to change.
    """
    mode = getattr(models, "offload", AUTO)
    mode = mode if mode in MODES else AUTO
    try:
        import torch

        memory_error = torch.cuda.OutOfMemoryError
    except Exception:
        return call(mode)
    try:
        return call(mode)
    except memory_error as error:
        if mode != AUTO:
            raise memory_error(str(error) + advice(mode)) from error
    log.warning("[yue2_comfy.placement] %s ran out of VRAM; running it again with only "
                "the half it needs on the card", stage)
    gc.collect()
    torch.cuda.empty_cache()
    try:
        return call(ON)
    except memory_error as error:
        raise memory_error(str(error) + advice(ON)) from error
