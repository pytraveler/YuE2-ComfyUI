"""The attention of one token-loop step, answered by the engine 'attention_backend' names.

Upstream's decode step attends one query per branch over a cache allocated for
the whole song, through public SDPA with a mask of the slots filled so far. On
the Windows wheels that call lands in the math kernel, which reads every slot
of the cache, filled or not, and copies K/V to widen the grouped heads: on a
5090, 0.77 ms a layer for two branches over 9216 slots. The token loops are
most of a song's time -- 83% of a 100-second pop song on 2026-09-24 -- so this
one call is where the speed is.

Two engines answer it here, each reading only the filled slots:

  'fast'   torch's own memory-efficient kernel, no package. The two query heads
           that share a key head are folded into two query rows, the cache is
           cut into PARTS pieces packed as one batch with each piece's filled
           length, and the pieces are joined by their log-sum-exp in a fixed
           order. Measured on 2026-09-24 over three songs: 137-142 tokens a
           second against sdpa's 112-129, the same bytes on every run and
           across processes.
  'flash'  flash-attn's ``flash_attn_with_kvcache`` with the filled length per
           branch -- what upstream's own 'flash' backend does with the kernel
           the Linux wheels carry. 161-170 tokens a second, the same bytes on
           every run and across processes. Needs the package, which has no
           official Windows wheels.

cuDNN's attention was the old fast choice and is gone: stepped over a filling
cache, the same inputs gave different bytes in 2-4 steps of 400, so no seed
ever came back. A song or workflow that still says 'cudnn' gets 'fast'.

Each engine sings a seed its own way: a take made with one is not the take
another makes, so 'sdpa' stays the default and keeps every earlier seed.
"""

from __future__ import annotations

import logging

from .constants import LEGACY_ATTENTION as LEGACY

log = logging.getLogger(__name__)

SDPA, FAST, FLASH = "sdpa", "fast", "flash"

LABELS = {FAST: "fast", FLASH: "flash-attn"}
"""What GraphAR is told its backend is, which the timing log reports.

Not 'flash': upstream's step tests for that name and would take its own branch,
the one whose kernel the Windows wheels lack."""

PARTS = 16
"""How many pieces 'fast' cuts the cache into.

The kernel runs one block of threads per query block and head, so with two
query rows a head the whole cache would fall to 16 blocks on a card with 170
places to put them. Measured in the stepped probe on 2026-09-24: 0.207 ms a
step whole, 0.177 at 4 pieces, 0.114 at 8, 0.083 at 16, 0.090 at 32."""

NO_FLASH = (
    "'flash' in the options needs the flash-attn package, and this Python does not have it. "
    "Choose 'fast', which needs nothing, or install a flash-attn build made for this torch."
)
OLD_CARD = (
    "'flash' in the options needs an Ampere card or newer, and this one is older. "
    "Choose 'fast', which runs on it."
)
NO_FAST = (
    "'fast' in the options needs a newer torch than this one. Choose 'sdpa'."
)


def resolve(backend) -> str:
    """The engine a setting names, with an old name read as what replaced it."""
    name = str(backend or SDPA)
    if name in LEGACY:
        log.info("[yue2_comfy.attention] attention_backend '%s' is gone -- it could not repeat a "
                 "seed -- so '%s' is used", name, LEGACY[name])
        return LEGACY[name]
    return name


def _efficient_takes_lengths() -> bool:
    """Whether this torch's efficient kernel takes a filled length per sequence."""
    try:
        import torch

        op = getattr(torch.ops.aten, "_efficient_attention_forward", None)
        return op is not None and "seqlen_k" in str(op.default._schema)
    except Exception:
        return False


def engine_for(name: str, device):
    """The engine for ``name`` on ``device``, None for upstream's own call; ValueError when it cannot run here."""
    if name == SDPA:
        return None
    if name == FAST:
        if not _efficient_takes_lengths():
            raise ValueError(NO_FAST)
        return Split()
    if name == FLASH:
        try:
            import flash_attn
        except Exception:
            raise ValueError(NO_FLASH) from None
        if not hasattr(flash_attn, "flash_attn_with_kvcache"):
            raise ValueError(NO_FLASH)
        import torch

        if torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError(OLD_CARD)
        return Flash(flash_attn.flash_attn_with_kvcache)
    raise ValueError("attention_backend must be one of sdpa, fast or flash, not {!r}".format(name))


class Split:
    """'fast': the efficient kernel over the cache in pieces, joined in a fixed order."""

    name = FAST

    def prepare(self, graph) -> None:
        """The piece layout for one GraphAR, made once, before any step is captured."""
        import torch

        capacity, branches, device = graph.capacity, graph.branches, graph.device
        size = -(-capacity // PARTS)
        starts = [min(part * size, capacity) for part in range(PARTS)]
        ends = [min((part + 1) * size, capacity) for part in range(PARTS)]
        config = graph.model.config
        group = config.num_attention_heads // config.num_key_value_heads
        graph.split_size = size
        graph.split_starts = torch.tensor(starts, dtype=torch.long, device=device)
        graph.split_lengths = torch.tensor([end - start for start, end in zip(starts, ends)],
                                          dtype=torch.long, device=device)
        graph.split_cu_k = torch.tensor([branch * capacity + start for branch in range(branches)
                                         for start in starts] + [branches * capacity],
                                        dtype=torch.int32, device=device)
        graph.split_cu_q = torch.arange(branches * PARTS + 1, dtype=torch.int32, device=device) * group
        graph.split_seen = torch.ones(branches * PARTS, dtype=torch.int32, device=device)
        graph.split_empty = torch.zeros(branches, PARTS, 1, 1, dtype=torch.float32, device=device)

    def begin(self, graph) -> None:
        """Each piece's filled length for this step, and a -inf for the pieces nothing has reached.

        Worked out once a step rather than once a layer; inside the graph like
        the rest of the step. Written into the buffers ``prepare`` made, never
        into new ones: a tensor made here during the rehearsals and replaced
        during the capture is memory the capture frees, and one made during the
        capture is memory freed outside it when the graph goes -- both of which
        the asynchronous allocator warns about, as it did in the user's ComfyUI
        on 2026-09-24 before this was changed.
        """
        import torch

        filled = graph.positions[:, None] + 1 - graph.split_starts[None, :]
        seen = torch.minimum(filled.clamp(min=0), graph.split_lengths[None, :])
        graph.split_seen.copy_(seen.clamp(min=1).flatten())
        graph.split_empty.zero_()
        graph.split_empty.masked_fill_((seen == 0)[:, :, None, None], float("-inf"))

    def attend(self, graph, query, keys, values):
        """Upstream's grouped call, [B, H, 1, D] over [B, Hk, slots, D], answered piece by piece."""
        import torch

        branches, heads, _one, dim = query.shape
        kv_heads, capacity = keys.shape[1], keys.shape[2]
        group = heads // kv_heads
        folded = query.transpose(1, 2).reshape(branches, kv_heads, group, dim).transpose(1, 2)
        queries = folded[:, None].expand(branches, PARTS, group, kv_heads, dim)
        queries = queries.reshape(1, branches * PARTS * group, kv_heads, dim)
        packed = (keys.transpose(1, 2).reshape(1, branches * capacity, kv_heads, dim),
                  values.transpose(1, 2).reshape(1, branches * capacity, kv_heads, dim))
        out, lse = torch.ops.aten._efficient_attention_forward(
            queries, packed[0], packed[1], None, graph.split_cu_q, graph.split_cu_k, group,
            graph.split_size, 0.0, 0, True, seqlen_k=graph.split_seen)[:2]
        out = out.view(branches, PARTS, group, kv_heads, dim).float()
        lse = lse[:, :, :group].reshape(branches, PARTS, kv_heads, group).transpose(2, 3)
        lse = lse + graph.split_empty
        weight = torch.exp(lse - lse.amax(dim=1, keepdim=True))
        joined = (out * weight[..., None]).sum(dim=1) / weight.sum(dim=1)[..., None]
        joined = joined.to(query.dtype).transpose(1, 2).reshape(branches, 1, heads, dim)
        return joined.transpose(1, 2)


class Flash:
    """'flash': flash-attn's kernel for a cache, told how much of it each branch has filled."""

    name = FLASH

    def __init__(self, kernel):
        self.kernel = kernel

    def prepare(self, graph) -> None:
        import torch

        graph.flash_filled = torch.ones(graph.branches, dtype=torch.int32, device=graph.device)

    def begin(self, graph) -> None:
        """Each branch's filled length, into the buffer ``prepare`` made; see ``Split.begin``."""
        graph.flash_filled.copy_(graph.positions + 1)

    def attend(self, graph, query, keys, values):
        out = self.kernel(query.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                          cache_seqlens=graph.flash_filled)
        return out.transpose(1, 2)


class Functional:
    """torch.nn.functional as the decode step sees it, its attention answered by an engine.

    Everything else is the real module. Only a call made while a step is being
    decoded, with one query per branch, goes to the engine; anything else --
    and anything upstream might add later -- reaches SDPA untouched.
    """

    def __init__(self, functional, engine):
        self._functional = functional
        self._engine = engine
        self.decoding = None

    def __getattr__(self, name):
        return getattr(self._functional, name)

    def scaled_dot_product_attention(self, query, key, value, *args, **kwargs):
        graph = self.decoding
        if graph is None or query.ndim != 4 or query.shape[2] != 1 or key.shape[2] != graph.capacity:
            return self._functional.scaled_dot_product_attention(query, key, value, *args, **kwargs)
        return self._engine.attend(graph, query, key, value)
