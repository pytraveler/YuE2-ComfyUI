"""Reading a few keys out of a GGUF header without parsing the whole thing.

``gguf.GGUFReader`` materialises every metadata value the moment it is
constructed, and one of those values is ``tokenizer.ggml.tokens`` -- a quarter
of a million strings turned into Python objects. On a folder of large quants
that is seconds per file, paid the first time ComfyUI answers ``/object_info``,
which is to say paid as "ComfyUI takes forever to start".

So the header is walked by hand instead. Values that were not asked for are
skipped by seeking past them: a fixed-width array is one seek, and even an
array of strings costs one length read per element with nothing allocated.
Parsing stops as soon as every wanted key has been seen, which for llama.cpp's
own writing order means the tokenizer is usually never reached at all.

Anything unexpected -- a version this does not know, a truncated file, a type id
from the future -- falls back to ``GGUFReader``, so the worst case is the speed
we had before rather than a model that stops being listed.

Ported from the MiniMax-H3 Prompt Rewriter pack.
"""

from __future__ import annotations

import logging
import os
import struct

log = logging.getLogger(__name__)

MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

_WIDTH = {
    UINT8: 1, INT8: 1,
    UINT16: 2, INT16: 2,
    UINT32: 4, INT32: 4, FLOAT32: 4,
    BOOL: 1,
    UINT64: 8, INT64: 8, FLOAT64: 8,
}

_FORMAT = {
    UINT8: "<B", INT8: "<b",
    UINT16: "<H", INT16: "<h",
    UINT32: "<I", INT32: "<i", FLOAT32: "<f",
    BOOL: "<?",
    UINT64: "<Q", INT64: "<q", FLOAT64: "<d",
}

MAX_STRING = 1 << 20

DEFAULT_ALIGNMENT = 32
ALIGNMENT_KEY = "general.alignment"


class MalformedGGUF(ValueError):
    pass


def _exact(handle, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise MalformedGGUF("wanted {} bytes, got {}".format(count, len(data)))
    return data


def _scalar(handle, kind: int):
    fmt = _FORMAT.get(kind)
    if fmt is None:
        raise MalformedGGUF("unknown value type {}".format(kind))
    return struct.unpack(fmt, _exact(handle, _WIDTH[kind]))[0]


def _string(handle) -> str:
    length = struct.unpack("<Q", _exact(handle, 8))[0]
    if length > MAX_STRING:
        raise MalformedGGUF("implausible string length {}".format(length))
    return _exact(handle, length).decode("utf-8", errors="replace")


def _skip(handle, kind: int) -> None:
    """Step over one value without building anything from it."""
    if kind in _WIDTH:
        handle.seek(_WIDTH[kind], 1)
        return
    if kind == STRING:
        length = struct.unpack("<Q", _exact(handle, 8))[0]
        handle.seek(length, 1)
        return
    if kind == ARRAY:
        element, count = struct.unpack("<IQ", _exact(handle, 12))
        if element in _WIDTH:
            handle.seek(_WIDTH[element] * count, 1)
            return
        for _ in range(count):
            _skip(handle, element)
        return
    raise MalformedGGUF("unknown value type {}".format(kind))


def _value(handle, kind: int):
    if kind == STRING:
        return _string(handle)
    if kind == ARRAY:
        element, count = struct.unpack("<IQ", _exact(handle, 12))
        return [_string(handle) if element == STRING else _scalar(handle, element)
                for _ in range(count)]
    return _scalar(handle, kind)


def _check_complete(handle, tensor_count: int, alignment: int, size: int) -> None:
    """Refuse a file whose tensor data does not fit in it.

    An interrupted browser download is a plain ``.gguf`` with a perfectly valid
    header, and offering it in the model list only moves the failure to the
    point where someone has already waited for a load. Reading the offsets back
    is the cheap way to catch it: no allocation, and the table is a few hundred
    fixed-size records.

    The last tensor's *offset* is checked rather than its end, because knowing
    where it ends means knowing the block size of every quantisation format
    llama.cpp has ever shipped. The offset alone catches every truncation that
    happens in practice and needs no such table.
    """
    furthest = 0
    for _ in range(tensor_count):
        _string(handle)
        dimensions = struct.unpack("<I", _exact(handle, 4))[0]
        if dimensions > 8:
            raise MalformedGGUF("implausible tensor rank {}".format(dimensions))
        handle.seek(8 * dimensions, 1)
        _kind, offset = struct.unpack("<IQ", _exact(handle, 12))
        furthest = max(furthest, offset)

    alignment = alignment if alignment and alignment > 0 else DEFAULT_ALIGNMENT
    position = handle.tell()
    data_start = position + (-position % alignment)
    if data_start + furthest >= size:
        raise MalformedGGUF(
            "tensor data starts at {} and reaches at least {}, past the end of a "
            "{}-byte file".format(data_start, data_start + furthest, size)
        )


def read_keys(path: str, wanted: tuple, probe=None, verify: bool = False) -> dict:
    """Return ``{key: value}`` for the wanted keys present in the header.

    ``probe`` is called with each key already collected and may return further
    keys to look for -- which is how ``<arch>.block_count`` is fetched without
    knowing the architecture until ``general.architecture`` has been read.

    ``verify`` walks the tensor table afterwards and raises when the file is too
    small to hold what it declares. It costs a few hundred fixed-size reads and
    gives up the early exit, so it is for deciding whether to *offer* a file
    rather than for reading a template out of one already chosen.
    """
    found: dict = {}
    remaining = set(wanted)
    alignment = DEFAULT_ALIGNMENT

    with open(path, "rb") as handle:
        if _exact(handle, 4) != MAGIC:
            raise MalformedGGUF("not a GGUF file")
        version = struct.unpack("<I", _exact(handle, 4))[0]
        if version not in SUPPORTED_VERSIONS:
            raise MalformedGGUF("GGUF version {} is not one this parser knows".format(version))
        tensor_count, kv_count = struct.unpack("<QQ", _exact(handle, 16))

        for _ in range(kv_count):
            key = _string(handle)
            kind = struct.unpack("<I", _exact(handle, 4))[0]
            if key in remaining:
                found[key] = _value(handle, kind)
                remaining.discard(key)
                if probe is not None:
                    remaining |= {name for name in probe(found) if name not in found}
                if not remaining and not verify:
                    break
            elif verify and key == ALIGNMENT_KEY:
                alignment = int(_value(handle, kind) or DEFAULT_ALIGNMENT)
            else:
                _skip(handle, kind)

        if verify:
            _check_complete(handle, tensor_count, alignment, os.path.getsize(path))

    return found


def keys(path: str, wanted: tuple, probe=None, verify: bool = False) -> dict:
    """:func:`read_keys`, falling back to ``gguf.GGUFReader`` if it cannot cope."""
    try:
        return read_keys(path, wanted, probe, verify)
    except MalformedGGUF:
        raise
    except Exception as error:
        log.info("[yue2_comfy.gguf_meta] fast header read failed on %s (%s), "
                 "falling back to GGUFReader", path, error)

    from gguf import GGUFReader

    reader = GGUFReader(path, "r")
    found: dict = {}
    remaining = set(wanted)
    while remaining:
        for name in list(remaining):
            field = reader.fields.get(name)
            remaining.discard(name)
            if field is None:
                continue
            try:
                found[name] = field.contents()
            except Exception:
                log.debug("[yue2_comfy.gguf_meta] %s unreadable in %s", name, path)
        if probe is not None:
            remaining |= {name for name in probe(found) if name not in found}
    return found
