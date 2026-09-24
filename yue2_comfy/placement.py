"""Which half of YuE2 sits on the card, decided stage by stage.

YuE2-3B is a mixture of transformers. Each of its 28 layers holds two sets of
weights: one for the autoregressive half, which writes the score and then the
codec tokens, and one for the non-autoregressive half, which turns those
tokens into acoustic latents by flow matching. No stage uses both. The AR half
with the embeddings is 4.03 GiB, the NAR layers are 2.63 GiB, and the modules
the acoustic stage needs besides its layers come to 0.10 GiB.

So the half a stage does not use can wait on the CPU. Measured on an RTX 5090
on 2026-09-17, with the halves swapped around the acoustic stage and the
vocabulary narrowed to the phase: a 40-second song peaked at 4.43 GiB against
9.81 with everything on the card, a 233-second song at 4.50 against 9.95, a swap
took about a second, and the audio was the same to the last byte in both. The
weights are the same tensors wherever they are kept, so there is nothing for the
song to differ by.

Where the peak of a stage falls moved with 0.7.0. Now that the acoustic stage
no longer builds attention matrices, the largest moment of a short song is the
swap itself: the copy runs about 0.13 GiB above the side it is moving, and the
stage that follows it computes in less. The estimates below are of the
computing, and MARGIN is what covers the moving.

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

The allocator itself is left alone. Upstream calls
``torch.cuda.set_per_process_memory_fraction`` to hold itself inside a share of
the card; this pack deliberately does not, because the fraction belongs to the
process and the process is ComfyUI -- a cap taken for one song would also cap
the video model queued behind it, and every node that ran afterwards would pay
for a number chosen here. The plan above is what keeps a stage inside the card
instead. If a report ever shows a prefill or a decode overrunning on a loaded
card, the cap is one line inside ``guarded``, set for the stage and dropped
with it; until then there is nothing to fix.
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

The AR half still pays for attention in whole matrices: its prefill asks for
grouped-query attention the same way, and the peaks measured on 2026-09-13 fit
about 2.65 of them. Three leaves a little room. The acoustic half stopped
paying that in 0.7.0 -- see ``fused_bytes``."""

WIDENED_KV_BYTES = 2 * HEADS * HEAD_DIM * ELEMENT_BYTES
"""One key position of the copy ``runtime.fused_attention`` makes.

It repeats the 8 key heads into 16 so that a kernel which streams takes the
call, which costs K and V once more at the full head count: 8 KiB a position,
against the matrix it no longer builds. Since 2026-09-24 the copy is made only
where cuDNN turns the grouped call down; the estimate keeps it, which is the
safe side, and cuDNN's own work measured no higher than the copy's."""

ROW_BYTES = 48 * 1024
"""What one row the acoustic half computes costs beyond attention.

Hidden states, projections and the MLP alive at a peak. Fitted to the work
measured on 2026-09-17, with the weights and the caches subtracted: 47.0 and
47.7 KiB a row for the prefill of a 40-second and a 233-second song, 42.5 and
42.8 for their flow matching. At 48 the model lands within a percent of the
prefill and about eight percent above the solve, which is the safe side."""

WORK_BASE = int(0.35 * GIB)
"""cuBLAS workspace, sampling buffers and the like. The resident allocation sat
0.29 to 0.34 GiB above the weights in every measured run."""

MARGIN = int(0.5 * GIB)
"""Room for the allocator's rounding, on top of every estimate."""

DECODE_BYTES = 5 * GIB
"""What the decode at the released 1024-frame core takes from the card, as the
driver counts it rather than as torch allocates it.

It used to be 3 GiB, the allocated peak, and that was the wrong number to judge
by. Measured on 2026-09-18 on an RTX 5090 over 4500 frames of the legacy
decoder (the standard one, and every song length, allocate the same): 3.15 GiB
allocated under ComfyUI's own flags -- 0.25 of weights, 0.25 of the weights
weight_norm computes from them, 2.18 of activations and 0.48 of cuDNN workspace
-- but 4.0-4.5 GiB held by the cudaMallocAsync pool ComfyUI runs on CUDA 13
(4.7 over 8700 frames), and 8.07 by the native allocator, 5.07 once
``generate._decode_tiled`` empties the cache after every tile. What has to fit
is what is held.

An RTX 3070 Ti found the difference on 2026-09-18. With 4.05 GiB free and the
3.5 this used to ask for, 'auto' left the NAR half on the card through the
decode, and the process reached 7.1-7.8 GiB of the 6.8 the card had left for it
(the same state reproduced on the 5090); moved off first, it peaked at
4.2-4.9, and the song was the same to the byte. A half whose weights are still
in the checkpoint file leaves the card without a copy, so asking for the room
costs nothing."""

SMALL_DECODE_BYTES = 2 * GIB
"""The same decode in the 256-frame tiles ``low_vram`` uses: 1.22 GiB allocated
and 1.79-1.82 held, cache emptied after every tile, under either allocator,
measured on 2026-09-18 on the same 4500 frames."""


def decode_bytes(low_vram: bool = False) -> int:
    """What to keep free for the decode, which depends on the tile it runs in."""
    return SMALL_DECODE_BYTES if low_vram else DECODE_BYTES


def kv_bytes(tokens: int, branches: int = 1) -> int:
    """The cache for ``tokens`` positions: two BF16 tensors a layer, per branch.

    The same size GraphAR allocates for its capacity, and the same size the
    acoustic prefill keeps for the tokens the flow matching attends to.
    """
    return 2 * LAYERS * int(branches) * int(tokens) * KV_HEADS * HEAD_DIM * ELEMENT_BYTES


def attention_bytes(queries: int, keys: int) -> int:
    """The attention matrices alive at the peak of one pass."""
    return ATTENTION_COPIES * HEADS * int(queries) * int(keys) * ELEMENT_BYTES


def fused_bytes(queries: int, keys: int) -> int:
    """What one acoustic pass holds now that no kernel builds the matrix.

    The widened K/V it attends to, and the rows it computes. Measured against
    it on 2026-09-17: the acoustic prefill of a 40-second song took 0.350 GiB
    and this says 0.352; of a 233-second song 1.348 against 1.350; their flow
    matching 0.065 and 0.347 against 0.070 and 0.376.
    """
    return WIDENED_KV_BYTES * int(keys) + ROW_BYTES * int(queries)


def ar_stage_bytes(prefix_tokens: int, budget: int, branches: int = 1) -> int:
    """What the score or the performance allocates beyond the weights.

    The cache for the prompt and the whole budget, the attention of the prompt's
    prefill, and the fixed rest. Measured against it: the score of a 40-second
    song took 0.50 GiB, the performance of a 240-second song 1.61 GiB.
    """
    return (kv_bytes(int(prefix_tokens) + int(budget), branches)
            + attention_bytes(prefix_tokens, prefix_tokens) + WORK_BASE)


def prefill_bytes(ar_tokens: int) -> int:
    """The acoustic prefill: the cache it keeps and the attention it passes through.

    WORK_BASE is not in here, and was until 0.7.0: the workspace it stands for
    is allocated by the stage that runs first and is still held, so counting it
    again as memory to find would ask a card for what it already gave.
    """
    return kv_bytes(ar_tokens) + fused_bytes(ar_tokens, ar_tokens)


def solve_bytes(ar_tokens: int, frames: int) -> int:
    """Flow matching over ``frames``, beyond the prefill cache, which exists by then."""
    positions = int(frames) + 2
    return fused_bytes(positions, int(ar_tokens) + positions)


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


def narrows(mode: str) -> bool:
    """Whether this mode keeps the two vocabulary tables out of the card.

    Every mode but ``off``, which means what it says: the whole model on the
    card. It costs nothing measurable and takes the same song out -- see
    ``vocabulary`` -- so there is no reason for a saving mode to decline it.
    """
    return (mode if mode in MODES else AUTO) != OFF


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


def _self_placed(module) -> bool:
    """Whether a module decides its own residency, so no half owns it.

    ``vocabulary.Sliced`` is the one that does: while it holds the two tables,
    they live in system RAM with one window on the card, and a half that moved
    them would undo exactly that.
    """
    return bool(getattr(module, "yue2_self_placed", False))


def groups(lm) -> dict:
    """The AR half, the NAR layers and the shared modules of one model.

    Raises AttributeError on a model that is not laid out this way, which every
    caller takes as a reason to keep the model whole rather than half moved.
    """
    ar = [module for module in (_resolve(lm, name) for name in AR_TOP)
          if not _self_placed(module)]
    nar = []
    for layer in lm.model.layers:
        ar.extend(getattr(layer, name) for name in AR_LAYER)
        nar.extend(getattr(layer, name) for name in NAR_LAYER)
    return {AR: ar, NAR: nar, SHARED: [_resolve(lm, name) for name in SHARED_TOP]}


def _slots(modules) -> list:
    """Every tensor a group holds, once each, as ``(owner, name, is_parameter)``.

    Walked by hand rather than left to ``module.to`` because a move has to know
    which module owns a tensor under which name: that is where its file-backed
    copy is kept, and where it is put back. A tensor two modules share is
    listed once, under the first of them.
    """
    found, seen = [], set()
    for module in modules:
        for owner in module.modules():
            for table, parameter in ((owner._parameters, True), (owner._buffers, False)):
                for name, tensor in table.items():
                    if tensor is None or id(tensor) in seen:
                        continue
                    seen.add(id(tensor))
                    found.append((owner, name, parameter))
    return found


def _held(owner, name: str, parameter: bool):
    """The plain tensor behind one slot, never the Parameter wrapped around it.

    A home has to be the storage the parameter held before the move, not the
    Parameter itself: the move reassigns ``.data`` on that object, and a home
    that was the object would follow it onto the card.
    """
    tensor = owner._parameters[name] if parameter else owner._buffers[name]
    return tensor.data if parameter else tensor


def _hold(owner, name: str, parameter: bool, tensor) -> None:
    """Put a tensor in one slot. A parameter keeps its identity; only its data changes.

    Unless torch will not swap the data across the two kinds of tensor -- the
    CPU and the meta device, say -- in which case a new Parameter takes the
    slot, which is what ``Module.to`` itself does there.
    """
    import torch

    if not parameter:
        owner._buffers[name] = tensor
        return
    held = owner._parameters[name]
    if torch._has_compatible_shallow_copy_type(held, tensor):
        held.data = tensor
    else:
        owner._parameters[name] = torch.nn.Parameter(tensor, requires_grad=held.requires_grad)


def _homes(owner) -> dict:
    """The file-backed copies one module's tensors had before they went to the card."""
    homes = owner.__dict__.get("_yue2_homes")
    if homes is None:
        homes = {}
        owner._yue2_homes = homes
    return homes


def _pristines(owner) -> dict:
    """The CPU copies one module's tensors had before a LoRA was folded into them on the card.

    Only kept when the weights have no home to go back to -- under 'off', or for
    weights that are not file-backed -- and only for a half some adapter of the
    run changes: see ``lora.apply``. The weights on the card are then the only
    ones a fold writes to, and this is what leaving the card puts back.
    """
    kept = owner.__dict__.get("_yue2_pristine")
    if kept is None:
        kept = {}
        owner._yue2_pristine = kept
    return kept


def _home(owner, name: str, tensor):
    """The copy to put back instead of copying ``tensor`` off the card, or None.

    A file-backed home first, a pristine copy kept for a LoRA fold second.
    """
    for table in ("_yue2_homes", "_yue2_pristine"):
        home = owner.__dict__.get(table, {}).get(name)
        if home is None or home.device.type != "cpu":
            continue
        if home.shape != tensor.shape or home.dtype != tensor.dtype:
            continue
        return home
    return None


STAGE_BYTES = 64 * 1024 ** 2
"""The size of each of the two pinned buffers a half goes through onto the card.

Measured on 2026-09-18 with the NAR half of the released checkpoint, 2.63 GiB,
on an RTX 5090: ``tensor.to(card)`` moved it at 1.64 GB/s from a copy of the
file the operating system had not cached and 4.18 GB/s from one it had; these
buffers moved it at 4.65 and 7.55. Two are what lets the processor fill one
while the card empties the other. 128 MiB of pinned memory is held for the
life of the process, which is a sixtieth of the model it moves."""

_STAGING = {}


def _staging(device):
    """The two pinned buffers and the copy stream for one card, made on first use.

    Made outside inference mode whatever the first use is inside: a tensor
    made in it can never again be written outside it, and these are written
    at every move. Found on 2026-09-24 on a card left 4 GiB free: the first
    move, the AR half onto the card for an edit's join, made them inside
    inference mode, and a change of words failed at the next, the NAR half's
    for the flow matching, outside it. On a large card nothing moves, so
    nothing was seen.
    """
    import torch

    held = _STAGING.get(device)
    if held is None:
        with torch.inference_mode(False):
            held = ([torch.empty(STAGE_BYTES, dtype=torch.uint8, pin_memory=True)
                     for _ in range(2)], torch.cuda.Stream(device))
        _STAGING[device] = held
    return held


def _to_card(tensors, device) -> list:
    """Copies of CPU tensors on ``device``, through the two pinned buffers.

    ``tensor.to(card)`` from ordinary memory goes through the driver's own
    staging, one small piece at a time, and when the tensor is a view of a
    mapped checkpoint -- which is what the loader leaves behind, see
    ``loader._read_state`` -- each of those pieces is also a page fault on the
    file. Copying a buffer's worth at a time on the processor turns the faults
    into one long sequential read, and the card copies out of pinned memory in
    the background while the next buffer fills.

    The bytes are the bytes: a test checks the copies against the originals.
    A tensor that is not contiguous, and every tensor when the device is not
    CUDA, goes the ordinary way.
    """
    import torch

    if getattr(device, "type", None) != "cuda":
        return [tensor.to(device) for tensor in tensors]
    buffers, stream = _staging(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    pending = [None, None]
    slot = 0
    copies = []
    for tensor in tensors:
        if not tensor.is_contiguous():
            copies.append(tensor.to(device))
            continue
        copy = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
        source = tensor.reshape(-1).view(torch.uint8)
        target = copy.view(-1).view(torch.uint8)
        for start in range(0, source.numel(), STAGE_BYTES):
            count = min(STAGE_BYTES, source.numel() - start)
            if pending[slot] is not None:
                pending[slot].synchronize()
            buffers[slot][:count].copy_(source[start:start + count])
            with torch.cuda.stream(stream):
                target[start:start + count].copy_(buffers[slot][:count], non_blocking=True)
                pending[slot] = torch.cuda.Event()
                pending[slot].record(stream)
            slot ^= 1
        copies.append(copy)
    stream.synchronize()
    return copies


def _onto(slots, copies, keep: bool, pristine: bool = False) -> None:
    """Put the card copies in their slots, leaving each file-backed view as a home or not.

    ``pristine`` keeps the CPU tensor all the same when it is not kept as a
    home: a LoRA will be folded into the card copy, and this is its way back.
    """
    for (owner, name, parameter, tensor), copy in zip(slots, copies):
        if keep and tensor.device.type == "cpu":
            _homes(owner)[name] = tensor
        else:
            owner.__dict__.get("_yue2_homes", {}).pop(name, None)
            if pristine and tensor.device.type == "cpu":
                _pristines(owner)[name] = tensor
            else:
                owner.__dict__.get("_yue2_pristine", {}).pop(name, None)
        _hold(owner, name, parameter, copy)


def _off(slots) -> int:
    """Take tensors off the card, and say how many bytes had to be copied to do it.

    A function of its own so that no name in it outlives the loop: a card
    tensor still referenced when ``empty_cache`` runs keeps its block.
    """
    import torch

    cpu = torch.device("cpu")
    copied = 0
    for owner, name, parameter, tensor in slots:
        home = _home(owner, name, tensor)
        if home is None:
            home = tensor.to(cpu)
            copied += tensor.numel() * tensor.element_size()
        _hold(owner, name, parameter, home)
    return copied


class LoraState:
    """What ``lora.apply`` knows about the adapters in one model, kept on the model itself.

    Not on the Placement: that is made again whenever what it measured goes
    stale -- ``vocabulary.narrowed`` drops it on the way into and out of every
    stage -- while the weights it describes stay folded. Kept there, the
    record was lost between the score and the performance, and the next stage
    folded the adapter a second time on top of the first; measured on
    2026-09-19, before this class existed.

    'folded' says which adapters are folded into the weights of a half on the
    card, and 'touched' which tensors took them; a half that leaves the card
    holds its own weights again, so leaving forgets both. 'factored' is the same
    for adapters held beside packed layers, which travel with the half and are
    not forgotten. 'outside' and 'outside_pristine' are for the acoustic
    modules outside the layers -- llm2vae, vae2llm, the time embedder -- which
    never leave the card and keep a copy of their own.
    """

    def __init__(self):
        self.folded = {}
        self.touched = {}
        self.factored = {}
        self.outside = ()
        self.outside_pristine = {}


def lora_state(lm) -> LoraState:
    """The model's LoraState, made on first use."""
    state = lm.__dict__.get("_yue2_lora")
    if state is None:
        state = LoraState()
        lm._yue2_lora = state
    return state


class Placement:
    """The groups of one loaded model, where they are, and how to move them.

    ``keeps_homes`` is whether the model's weights are still views of the
    checkpoint file, which the loader says when it builds the model. When they
    are, a tensor going to the card leaves its file-backed copy behind on the
    module that owns it, and going back to the CPU is putting that copy back:
    nothing crosses the bus and nothing is allocated. The pages belong to the
    file, so the operating system can drop them whenever it needs the memory
    and read them again on the next trip to the card.

    That is the move the reporter's RTX 3070 Ti paid 4.7 of a 60-second run for
    on 2026-09-18: both halves copied to the CPU before the decode, and freed a
    second later when the run unloaded the model. When the weights are not
    file-backed -- the INT8 repack restored to BF16, the layers ``low_vram``
    packs -- a copy left behind would be memory the model holds twice, so they
    are copied off the card as before.

    Nor under 'off'. A home keeps the whole mapping of the checkpoint alive,
    and on Windows a copy-on-write mapping is charged to the process in full:
    with every tensor on the card, 16.98 GiB of commit against 10.06 without
    homes, measured on 2026-09-18. 'auto' and 'on' hold that mapping anyway,
    through the vocabulary tables they keep in RAM; 'off' narrows nothing and
    used to let it go, so there a move onto the card forgets the home, and the
    rare move off copies, as it always did.

    'pristine' is the halves whose CPU copies are kept when they go to the
    card, set before a stage's moves, for a half an adapter of the run changes;
    'lora' is the model's LoraState, which a half leaving the card updates.
    """

    def __init__(self, lm, device):
        self.device = device
        self.groups = groups(lm)
        self.sizes = {half: sum(t.numel() * t.element_size()
                                for module in self.groups[half] for t in _tensors(module))
                      for half in HALVES}
        self.keeps_homes = bool(getattr(lm, "_yue2_file_backed", False))
        self.pristine = set()
        self.lora = lora_state(lm)

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

    def _pending(self, half: str, place: str) -> list:
        """The slots of a half not yet where ``place`` is, with the tensor each holds.

        Its own function so that no loop variable in ``_move`` still holds a
        card tensor when the cache is emptied: one did, and kept a 24 MiB block.
        """
        pending = []
        for owner, name, parameter in _slots(self.groups[half]):
            tensor = _held(owner, name, parameter)
            if (self._place(tensor.device) == CARD) != (place == CARD):
                pending.append((owner, name, parameter, tensor))
        return pending

    def _move(self, half: str, place: str, homes: bool = True) -> tuple:
        """Move one half, and say how long it took, how much moved and how much was copied.

        Only tensors not already where they are going count, so the size in the
        log is what the move handled rather than the size of the half: while
        the vocabulary is narrowed the two tables never leave the CPU, and the
        AR half that goes to the card is 2.63 GiB of its 4.03. ``homes`` False
        is 'off': see the class.
        """
        import torch

        start = time.perf_counter()
        slots = self._pending(half, place)
        moved = sum(entry[3].numel() * entry[3].element_size() for entry in slots)
        copied = 0
        if place == CARD:
            copies = _to_card([entry[3] for entry in slots], self.device)
            _onto(slots, copies, self.keeps_homes and homes, half in self.pristine)
            del copies
            copied = moved
        else:
            copied = _off(slots)
            del slots
            self.lora.folded.pop(half, None)
            self.lora.touched.pop(half, None)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return time.perf_counter() - start, moved, copied

    def cycle(self, half: str, homes: bool = True) -> None:
        """Send a half off the card and back, which leaves its own weights on the card.

        How a LoRA is taken out: going off puts back each tensor's home or its
        pristine copy without copying anything, and coming back fetches them
        through the pinned buffers, a second or less for a half. Both moves
        free before they allocate, so the card never holds the half twice.
        """
        if getattr(self.device, "type", "cpu") == "cpu":
            return
        self._move(half, CPU, homes)
        self._move(half, CARD, homes)

    def arrange(self, mode: str, required, work: int, stage: str) -> list:
        """Move what this stage needs moved, and say so in the log."""
        if getattr(self.device, "type", "cpu") == "cpu":
            return []
        where = {half: self.where(half) for half in HALVES}
        available = free_bytes(self.device)
        done = []
        for half, place in moves(mode, required, where, self.sizes, available, work):
            seconds, moved, copied = self._move(half, place, homes=mode != OFF)
            done.append("{}->{}".format(half, place))
            reason = ""
            if available is not None:
                reason = ", with {:.1f} GiB free for about {:.1f} GiB of work".format(
                    available / GIB, work / GIB)
            if place == CARD:
                log.info("[yue2_comfy.placement] %s: the %s half onto the card (%.2f GiB) "
                         "in %.1f s%s", stage, half.upper(), moved / GIB, seconds, reason)
            elif copied:
                log.info("[yue2_comfy.placement] %s: the %s half to the CPU (%.2f GiB, %.2f "
                         "copied) in %.1f s%s", stage, half.upper(), moved / GIB,
                         copied / GIB, seconds, reason)
            else:
                log.info("[yue2_comfy.placement] %s: the %s half off the card (%.2f GiB, "
                         "nothing copied: its weights are still in the checkpoint file) "
                         "in %.1f s%s", stage, half.upper(), moved / GIB, seconds, reason)
        return done


def placement_for(lm, device) -> Placement:
    """The Placement kept on the model, made on first use."""
    held = getattr(lm, "_yue2_placement", None)
    if held is None or held.device != device:
        held = Placement(lm, device)
        lm._yue2_placement = held
    return held


def load(model, device, file_backed: bool = False):
    """A freshly built model, ready for its first stage: the shared modules on the card.

    Both halves stay where the checkpoint was read, on the CPU, and move when a
    stage asks for them. Loading onto the card whole would put all 6.8 GiB
    there at once, which is the one peak no stage could take back.

    ``file_backed`` is the loader saying that the halves are still views of the
    checkpoint file, which is what lets a move off the card put them back there
    instead of copying them -- see ``Placement``.
    """
    model._yue2_file_backed = bool(file_backed)
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
    """Put the halves where one stage of this run wants them, with the run's LoRAs in the one it needs.

    A models object without a real backbone -- the stand-ins the tests use -- is
    left alone, and so is a model that is not laid out in two halves, which the
    loader has put on the card whole.

    The halves an adapter of the run changes keep their CPU copies as they go
    onto the card, and once the moves are done the required half gets exactly
    the run's adapters -- see ``lora.apply``. A run without adapters on a model
    that has none folded in pays nothing here, not even the import.
    """
    lm = getattr(models, "lm", None)
    device = getattr(models, "device", None)
    if lm is None or device is None or isinstance(device, str):
        return []
    try:
        keeper = placement_for(lm, device)
    except AttributeError:
        return []
    chosen = getattr(models, "loras", None) or ()
    if chosen:
        from .lora import choices

        keeper.pristine = choices.halves(chosen)
    else:
        keeper.pristine = set()
    mode = mode or getattr(models, "offload", AUTO)
    done = keeper.arrange(mode, required, work, stage)
    state = keeper.lora
    if required in HALVES and (chosen or state.folded.get(required) or state.factored.get(required)
                               or (required == NAR and state.outside)):
        from .lora import apply

        apply.ensure(models, keeper, required, homes=mode != OFF)
    return done


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
