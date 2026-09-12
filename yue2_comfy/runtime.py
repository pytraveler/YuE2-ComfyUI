"""The two things upstream does to the whole process, done to one run instead.

Both are unavoidable -- YuE2 needs the deterministic math settings to reproduce
a seed, and it needs an attention backend that this machine actually has -- and
both are process-wide in the upstream code. Inside ComfyUI that is not
acceptable: a flag left flipped changes every other model in the graph until
the user restarts.
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


def flash_attention_available() -> bool:
    """Whether this torch build can actually run variable-length FlashAttention."""
    try:
        import torch

        return bool(torch.backends.cuda.is_flash_attention_available())
    except Exception:
        return False
