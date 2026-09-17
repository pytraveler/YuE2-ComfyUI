"""What upstream does to the whole process, done to one run instead.

Three things are unavoidable -- YuE2 needs the deterministic math settings to
reproduce a seed, it needs an attention backend that this machine actually has,
and its acoustic attention needs a kernel that does not build the whole matrix
-- and all three are process-wide in the upstream code. Inside ComfyUI that is
not acceptable: a flag left flipped changes every other model in the graph
until the user restarts. Each is a context manager that puts back what it
found, including on failure.
"""

from __future__ import annotations

import contextlib
import logging

log = logging.getLogger(__name__)

DETERMINISTIC_FLAGS = (
    ("backends.cudnn.benchmark", False),
    ("backends.cudnn.deterministic", True),
    ("backends.cuda.matmul.allow_tf32", False),
    ("backends.cudnn.allow_tf32", False),
    ("backends.cuda.matmul.allow_fp16_reduced_precision_reduction", False),
)

_MISSING = object()


def _lookup(root, path: str):
    """The owner object and final attribute name for a dotted path."""
    parts = path.split(".")
    owner = root
    for part in parts[:-1]:
        owner = getattr(owner, part, None)
        if owner is None:
            return None, parts[-1]
    return owner, parts[-1]


@contextlib.contextmanager
def deterministic_math():
    """Upstream's six torch globals for the length of one run, then back.

    A flag that does not exist on this torch version is skipped rather than
    created: these names have moved before, and inventing an attribute that
    torch does not read would be a silent no-op that looks like it worked.
    """
    import torch

    saved = []
    precision = torch.get_float32_matmul_precision()
    try:
        for path, wanted in DETERMINISTIC_FLAGS:
            owner, name = _lookup(torch, path)
            if owner is None or not hasattr(owner, name):
                log.debug("[yue2_comfy.runtime] torch has no %s, skipping", path)
                continue
            saved.append((owner, name, getattr(owner, name)))
            setattr(owner, name, wanted)
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        try:
            torch.set_float32_matmul_precision(precision)
        except Exception:
            log.debug("[yue2_comfy.runtime] could not restore matmul precision",
                      exc_info=True)
        for owner, name, value in reversed(saved):
            try:
                setattr(owner, name, value)
            except Exception:
                log.debug("[yue2_comfy.runtime] could not restore %s", name, exc_info=True)


@contextlib.contextmanager
def pinned_attention(backend: str):
    """Force one attention backend on the AR stages, without forking upstream.

    GraphAR's 'auto' asks whether variable-length FlashAttention is available by
    checking that torch.ops.aten._flash_attention_forward exists and that its
    schema mentions seqused_k. On Windows wheels both are true and the kernel is
    still not built, so 'auto' picks flash and graph capture dies with
    'USE_FLASH_ATTENTION was not enabled for build'. The honest question is
    torch.backends.cuda.is_flash_attention_available(), which answers False.

    Rather than edit the vendored file, this swaps the class the sampler will
    import. sampling.generate_tokens imports GraphAR inside the function body,
    so replacing the module attribute is enough, and it is put back afterwards.

    This covers the AR stages only. The NAR stage takes its backend as an
    argument and accepts sdpa, math or flash -- never cudnn -- so it is always
    called with sdpa.
    """
    from .vendor.yue2 import cuda_graph

    original = cuda_graph.GraphAR

    class PinnedGraphAR(original):
        def __init__(self, *args, **kwargs):
            kwargs["attention_backend"] = backend
            super().__init__(*args, **kwargs)

    cuda_graph.GraphAR = PinnedGraphAR
    try:
        yield
    finally:
        cuda_graph.GraphAR = original


FUSED_BACKENDS = ("EFFICIENT_ATTENTION", "MATH")
"""Which SDPA kernels the widened acoustic call is allowed to land in.

The fused one first, math kept enabled underneath it so that a shape the fused
kernel refuses still runs instead of raising.
"""


def _grouped(q, k, v) -> bool:
    """Whether this call has the grouped-query shape the fused kernels refuse.

    Anything else -- a wrong rank, mismatched K/V, a head count that does not
    divide -- is left exactly as upstream got it, so upstream raises its own
    error about it rather than a confusing one from here.
    """
    try:
        if q.device.type != "cuda" or q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
            return False
        heads, kv_heads = int(q.shape[1]), int(k.shape[1])
    except (AttributeError, IndexError, TypeError):
        return False
    return kv_heads > 0 and heads > kv_heads and heads % kv_heads == 0


@contextlib.contextmanager
def _fused_kernel(device_type: str):
    """FUSED_BACKENDS for the length of one call, on CUDA only."""
    if device_type != "cuda":
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        log.debug("[yue2_comfy.runtime] this torch has no sdpa_kernel", exc_info=True)
        yield
        return
    wanted = [getattr(SDPBackend, name) for name in FUSED_BACKENDS if hasattr(SDPBackend, name)]
    try:
        with sdpa_kernel(wanted):
            yield
    except TypeError:
        log.debug("[yue2_comfy.runtime] this torch takes one backend at a time")
        with sdpa_kernel(wanted[0]):
            yield


@contextlib.contextmanager
def fused_attention():
    """Widen grouped K/V so the acoustic attention gets a kernel that streams.

    ``nar.attention`` asks SDPA for grouped-query attention with
    ``enable_gqa=True``: 16 query heads against 8 key heads. The memory-efficient
    kernel refuses that argument outright, and on the Windows wheels flash is not
    compiled in, so the call lands in the math kernel -- which materializes the
    whole heads x queries x keys matrix. Measured on 2026-09-17 on the shapes of
    a four-minute song: flow matching cost 10.03 GiB that way against 0.023 GiB
    with K/V repeated to 16 heads and no ``enable_gqa``; the acoustic prefill
    6.07 against 0.025 GiB. End to end the song's peak went 10.72 -> 5.66 GiB and
    the run 122 -> 98 s, which is why this is not optional: it is cheaper and
    faster at once. The copy it makes costs 8 KiB a token, against the matrix it
    avoids.

    The same trick, for the same reason, is what ComfyUI's own
    ``comfy/ops.py:scaled_dot_product_attention`` does for its models, and what
    upstream itself does on MPS two lines above the call this wraps. CUDA only
    escaped it because the wheels upstream was written against had flash.

    Upstream is not edited: ``CachedNAR`` looks the function up in its module
    when it runs, the way ``pinned_attention`` replaces GraphAR.

    It is not byte-identical with the songs this pack made before 0.7.0. The
    score and the semantic tokens come out the same; only the last stage renders
    differently, about 30 dB below the song's own level. Two runs of one seed
    still agree with each other to the byte.
    """
    from .vendor.yue2 import nar

    original = nar.attention

    def attention(q, k, v, **kwargs):
        """Upstream's attention, with the group repeated into the key heads."""
        if _grouped(q, k, v):
            groups = int(q.shape[1]) // int(k.shape[1])
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        device_type = getattr(getattr(q, "device", None), "type", "cpu")
        with _fused_kernel(device_type):
            return original(q, k, v, **kwargs)

    nar.attention = attention
    try:
        yield
    finally:
        nar.attention = original


def flash_attention_available() -> bool:
    """Whether this torch build can actually run variable-length FlashAttention."""
    try:
        import torch

        return bool(torch.backends.cuda.is_flash_attention_available())
    except Exception:
        return False
