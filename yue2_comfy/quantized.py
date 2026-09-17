"""The layers of both halves kept on the card as INT8 rows.

The 28 layers are the memory nothing else could take away. The AR half is 2.62
GiB of matrices and the NAR half 2.63, every token reads all of one of them,
and neither can wait in system RAM: fetching them over the bus costs more per
token than the whole model costs now. Kept as INT8 rows with one scale each
they are half that, and the matrix a multiply needs becomes BF16 for the length
of that multiply and is given back.

What it costs is time and the song. Measured on an RTX 5090 on 2026-09-17, on a
28-layer tower of the released shapes at batch one, captured in a CUDA graph:
2.56 ms a step with BF16 weights and 5.66 ms with packed ones, so the token
loop takes about twice as long. On whole songs it costs less than that, because
the acoustic stage gets faster -- there is half as much to carry onto the card,
and a multiply over a few thousand rows pays for its unpacking once: a
four-minute song took 110 seconds instead of 104 and peaked at 3.14 GiB instead
of 4.50, a 40-second one 27.6 seconds instead of 20.5 at 3.11 GiB instead of
4.43. Both sang with the allocator capped at 3.5 GiB and failed at 3.25, where
before this they had needed 4.75.

And an INT8 round trip is lossy, so the same seed writes the same score and a
different performance of it. Both takes were read back by this pack's own speech
recognition: the four-minute one sang the lyric through, and 98.6 percent of the
words heard were words of the lyrics, which is what the BF16 take scored too.
Nothing here is on unless ``low_vram`` asks for it.

The scale is one per row, and it is folded into the answer rather than into the
weight. A row of scales multiplied into an output of a few rows costs nothing,
while multiplying it into the matrix would write the whole matrix a second time
-- 7.13 ms a step measured that way against 5.66.

Upstream's GraphAR refuses a model whose AR matrices are not plain ``nn.Linear``
of the model's dtype, because its own FP8 modules cannot be captured. These can:
unpacking is CUDA work, and the capture records it like the multiply that
follows. So ``captured`` shows that constructor a plain linear for the length of
its check and puts the packed ones back before a token is decoded.
"""

from __future__ import annotations

import contextlib
import logging

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

LEVELS = 127
"""The largest magnitude a packed row holds.

-128 exists and is not used: a symmetric range keeps zero at zero, which the
attention masks and the silences depend on."""

AR_BLOCKS = ("self_attn", "mlp")
NAR_BLOCKS = ("nar_self_attn", "nar_mlp")
BLOCKS = AR_BLOCKS + NAR_BLOCKS
"""The blocks of a layer whose matrices are packed: both halves, attention and
MLP, which is where all 5.25 GiB of them are. The norms are kilobytes, and the
two vocabulary tables are kept off the card another way -- see ``vocabulary``."""


def pack(weight):
    """One matrix as INT8 rows, and the scale each row was divided by.

    Symmetric and per row: a row whose largest magnitude is far from the
    matrix's own loses nothing to its neighbours. Measured on all 392 matrices
    of the released checkpoint, this reconstructs them to a cosine of 0.99990
    or better, 0.99996 on average, for about one percent of relative error --
    which is what INT8 costs at this width however the scales are arranged, and
    is why the setting that turns it on says the song changes.
    """
    values = weight.detach().float()
    scales = values.abs().amax(dim=1).clamp(min=1e-12) / LEVELS
    rows = torch.round(values / scales[:, None]).clamp_(-LEVELS, LEVELS)
    return rows.to(torch.int8), scales.to(weight.dtype)


class Packed(torch.nn.Module):
    """One linear layer as INT8 rows, unpacked for a single multiply."""

    def __init__(self, linear, card=None):
        super().__init__()
        weight = linear.weight.detach()
        home = weight.device
        rows, scales = pack(weight if card is None else weight.to(card))
        self.register_buffer("rows", rows.to(home))
        self.register_buffer("scales", scales.to(home))
        bias = getattr(linear, "bias", None)
        self.register_buffer("bias", None if bias is None else bias.detach().to(home))
        self.out_features, self.in_features = int(weight.shape[0]), int(weight.shape[1])

    def forward(self, hidden):
        answer = F.linear(hidden, self.rows.to(hidden.dtype)) * self.scales
        return answer if self.bias is None else answer + self.bias

    def extra_repr(self):
        return "in_features={}, out_features={}, int8".format(
            self.in_features, self.out_features)


def compress(lm, card=None) -> int:
    """Pack every matrix of both halves, and say how many were packed.

    ``card`` is where the arithmetic runs. The weights sit in system RAM at
    this point and packing them there takes the best part of a minute, while a
    matrix at a time on the card takes about a second for all of them.

    ``placement`` is told to forget what it measured: the halves are half the
    size now, and that size is what decides which of them it moves.

    One consequence worth knowing: a packed ``mlp`` block holds buffers and no
    parameters at all, so ``next(module.parameters())`` raises on it. Upstream's
    ``nar._offload_ar`` reads a device that way, which is why it matters that
    ``placement.acoustic`` replaces that function for the whole stage.
    """
    packed = 0
    saved = 0
    for layer in getattr(getattr(lm, "model", None), "layers", []):
        for name in BLOCKS:
            block = getattr(layer, name, None)
            if block is None:
                continue
            for leaf, child in list(block.named_children()):
                if not isinstance(child, torch.nn.Linear):
                    continue
                saved += child.weight.numel() * (child.weight.element_size() - 1)
                setattr(block, leaf, Packed(child, card))
                packed += 1
    if packed:
        lm._yue2_placement = None
        log.info("[yue2_comfy.quantized] %d matrices packed as INT8, %.2f GiB less to "
                 "keep on the card", packed, saved / float(1024 ** 3))
    return packed


def compressed(lm) -> bool:
    """Whether this model's layers are packed."""
    return bool(_packed(lm, BLOCKS))


@contextlib.contextmanager
def captured(models, enabled: bool = True):
    """Let GraphAR build itself around packed layers.

    Upstream checks that every AR matrix is a plain Linear of the model's dtype
    and refuses otherwise, because the FP8 modules it wrote that check for
    cannot be captured. Packed ones can, which is what this pack measured
    before defeating the check: unpacking is CUDA work like the multiply after
    it, and both go into the graph. Only the constructor sees the stand-in;
    every token is decoded through the packed layers themselves.
    """
    lm = getattr(models, "lm", None)
    if not enabled or lm is None or not compressed(lm):
        yield
        return

    from .vendor.yue2 import cuda_graph

    original = cuda_graph.GraphAR

    class PackedGraphAR(original):
        """Upstream's graph, built while the AR matrices look ordinary."""

        def __init__(self, model, *args, **kwargs):
            with _plain(model):
                super().__init__(model, *args, **kwargs)

    cuda_graph.GraphAR = PackedGraphAR
    try:
        yield
    finally:
        cuda_graph.GraphAR = original


def _packed(lm, blocks) -> list:
    """Every packed matrix of a model, as (block, name, module)."""
    found = []
    for layer in getattr(getattr(lm, "model", None), "layers", []):
        for name in blocks:
            block = getattr(layer, name, None)
            for leaf, child in (block.named_children() if block is not None else ()):
                if isinstance(child, Packed):
                    found.append((block, leaf, child))
    return found


@contextlib.contextmanager
def _plain(lm):
    """The AR matrices shown as one empty Linear of the model's dtype.

    One module stands in for all 196 of them: what reads them asks only whether
    they are Linear and what dtype they hold, and both answers are the same for
    every one.
    """
    swapped = _packed(lm, AR_BLOCKS)
    if not swapped:
        yield
        return
    dtype = lm.model.embed_tokens.weight.dtype
    stand_in = torch.nn.Linear(1, 1, bias=False).to(dtype)
    for block, leaf, _child in swapped:
        setattr(block, leaf, stand_in)
    try:
        yield
    finally:
        for block, leaf, child in swapped:
            setattr(block, leaf, child)
