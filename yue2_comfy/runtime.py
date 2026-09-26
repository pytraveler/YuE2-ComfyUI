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
import functools
import logging
import warnings

from .constants import CAPTURE_MODE

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

    Upstream's step always runs its public-SDPA branch here. With 'fast' or
    'flash' the one attention call in that branch is answered by the engine in
    ``attention.py``: the module's ``F`` is swapped for one whose SDPA knows
    which graph is decoding, and the graph says so around each step, so the
    rest of upstream's step -- projections, rotary, the cache writes, the fused
    matrices -- runs as it was written. Whether the engine can run is asked when
    a graph is built, so a run on the processor, which builds none, is never
    refused for it. An old 'cudnn' is read as 'fast'; see ``attention.LEGACY``.

    The graph each GraphAR records into is swapped as well. Upstream makes a
    plain ``torch.cuda.CUDAGraph`` just before recording and records with
    torch's default mode, under which another thread's memory query can abort
    the process; the one this class hands out records with ``CAPTURE_MODE``.
    See ``guarded_graph``.

    This covers the AR stages only. The acoustic stage has its own kernel; see
    ``fused_attention``.
    """
    from . import attention
    from .vendor.yue2 import cuda_graph

    name = attention.resolve(backend)
    original, functional = cuda_graph.GraphAR, cuda_graph.F
    answering = [None]

    class PinnedGraphAR(original):
        _guarded = None

        @property
        def graph(self):
            return self._guarded

        @graph.setter
        def graph(self, value):
            self._guarded = None if value is None else guarded_graph()

        def __init__(self, *args, **kwargs):
            kwargs["attention_backend"] = attention.SDPA
            super().__init__(*args, **kwargs)
            self.engine = attention.engine_for(name, self.device, getattr(self, "dtype", None))
            if self.engine is not None:
                if answering[0] is None:
                    answering[0] = attention.Functional(functional, self.engine)
                    cuda_graph.F = answering[0]
                self.engine.prepare(self)
                self.attention_backend = attention.LABELS[name]

        def _decode(self):
            if self.engine is None:
                return super()._decode()
            import torch

            with torch.inference_mode():
                self.engine.begin(self)
            answering[0].decoding = self
            try:
                return super()._decode()
            finally:
                answering[0].decoding = None

    cuda_graph.GraphAR = PinnedGraphAR
    try:
        yield
    finally:
        cuda_graph.GraphAR = original
        cuda_graph.F = functional


def guarded_graph():
    """A CUDA graph that records with ``CAPTURE_MODE``, whatever mode its capture asks for.

    ``torch.cuda.graph`` passes its mode to the graph's own ``capture_begin``,
    so a graph that answers for the mode changes nothing else in the code that
    records into it -- upstream's GraphAR among them, which is left as written.
    Made fresh for every capture, the way upstream makes a plain one.

    Every other argument goes through as it came. From torch 2.13 on,
    ``torch.cuda.graph`` also passes ``check_input_liveness``; a ``capture_begin``
    that named only the pool and the mode refused it, and every song on such a
    torch failed at its first capture (issue #8).
    """
    return _guarded_class()()


@functools.lru_cache(maxsize=None)
def _guarded_class():
    """The graph class ``guarded_graph`` makes, built once torch is wanted."""
    import torch

    class GuardedGraph(torch.cuda.CUDAGraph):
        def capture_begin(self, pool=None, capture_error_mode=CAPTURE_MODE, **kwargs):
            super().capture_begin(pool=pool, capture_error_mode=CAPTURE_MODE, **kwargs)

    return GuardedGraph


GROUPED_BACKEND = "CUDNN_ATTENTION"
"""The SDPA kernel the acoustic call asks for first, with K/V grouped as they come.

cuDNN takes ``enable_gqa`` itself, so nothing is copied and no matrix is built.
Measured on 2026-09-24 on an RTX 5090 (torch 2.11, cuDNN 9.19) against the
widened call through the efficient kernel: the acoustic stage of a 100-second
song 5.8 -> 4.4 s, of a 180-second one 12.9 -> 8.6 s, of a 236-second one
21.2 -> 12.9 s, at the same peak. Every run gave the same latents to the bit, in
one process, across processes and with 3 GiB of the card left free. The latents
lie 25-40 dB from the efficient kernel's, which is as far as those lie from the
math kernel's; Qwen3-ASR heard the same words in three songs of four and a WER
of 0.155 against 0.138 in the fourth.

cuDNN's attention is not reproducible everywhere: stepped one query at a time
over a filling cache, as the token loops use it, the same inputs gave different
bytes in 2-4 steps of 400. The acoustic calls are whole blocks of queries and
never did.
"""

FUSED_BACKENDS = ("EFFICIENT_ATTENTION", "MATH")
"""Which SDPA kernels the widened acoustic call is allowed to land in.

The fused one first, math kept enabled underneath it so that a shape the fused
kernel refuses still runs instead of raising. Only a call cuDNN turns down comes
this way -- on a card older than Ampere, say.
"""

_REFUSED = set()
"""Acoustic calls cuDNN has turned down in this process, by shape, dtype and causality.

Whether a kernel takes a call depends on nothing else, so remembering the answer
sends a call the way asking again would -- one song's calls go the same way on
every run -- without paying for the refusal and its warnings each time.
"""


def _call_key(q, k, kwargs) -> tuple:
    """What decides whether cuDNN takes an acoustic call."""
    device = getattr(q, "device", None)
    return (getattr(device, "type", None), getattr(device, "index", None), str(getattr(q, "dtype", "")),
            tuple(q.shape), tuple(k.shape), bool(kwargs.get("causal", False)))


def _is_refusal(error: Exception) -> bool:
    """Whether an error is SDPA saying that no kernel it may use takes the call, and nothing else.

    Out of memory is a RuntimeError too, and must not be read as a refusal: the
    widened path would only ask for more.
    """
    return "No available kernel" in str(error)


@contextlib.contextmanager
def _only_grouped_kernel():
    """GROUPED_BACKEND alone for the length of one call, so a refusal raises instead of landing in math.

    A torch without that kernel, or without ``sdpa_kernel``, refuses the same way.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        raise RuntimeError("No available kernel: this torch has no sdpa_kernel") from None
    backend = getattr(SDPBackend, GROUPED_BACKEND, None)
    if backend is None:
        raise RuntimeError("No available kernel: this torch has no %s" % GROUPED_BACKEND)
    with sdpa_kernel(backend):
        yield


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
    """Give the acoustic attention a kernel that streams: cuDNN, or the efficient one on widened K/V.

    Since 2026-09-24 a grouped call goes to cuDNN first, K/V as they are -- see
    GROUPED_BACKEND for what that measured. Only where cuDNN turns a call down
    does it take the path below, which was the only one before.

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

    It is not byte-identical with the songs this pack made before 0.7.0, and the
    cuDNN path is not with the songs made between then and 2026-09-24. The score
    and the semantic tokens come out the same; only the last stage renders
    differently, 20-40 dB below the song's own level. Two runs of one seed still
    agree with each other to the byte.
    """
    from .vendor.yue2 import nar

    original = nar.attention

    def attention(q, k, v, **kwargs):
        """Upstream's attention: grouped through cuDNN, or with the group repeated into the key heads."""
        if _grouped(q, k, v):
            key = _call_key(q, k, kwargs)
            if key not in _REFUSED:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with _only_grouped_kernel():
                            return original(q, k, v, **kwargs)
                except RuntimeError as error:
                    if not _is_refusal(error):
                        raise
                    if not _REFUSED:
                        log.info("[yue2_comfy.runtime] cuDNN does not take the acoustic attention on this "
                                 "card, so its K/V are widened for the efficient kernel instead")
                    _REFUSED.add(key)
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
