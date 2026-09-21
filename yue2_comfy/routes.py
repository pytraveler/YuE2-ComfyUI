"""HTTP routes behind the song editor's and the score editor's windows, the MIDI and LoRA nodes' lists, and the Edit Track sounds.

Registered on import. A failure here must never stop the nodes from loading --
both editors are conveniences on top of widgets that work without them -- so
the registration is guarded and logged rather than raised.

The score routes answer 200 with ``ok: false`` and a message when a score or an
edit cannot be used, because that is something the person editing needs to
read, and 400 only when the request itself is not what the editor sends.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

log = logging.getLogger(__name__)

PREFIX = "/yue2"
SOUND_SUBFOLDER = "yue2_edit"
"""The folder under ComfyUI's temp where 'YuE2 Edit Track' writes the song and its takes."""
SOUND_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,120}\.wav$")
"""What the sound route will serve: one file name as the node spells them, nothing that walks."""
SOUND_CHUNK = 1024 * 1024
"""Bytes read and written per turn of the loop while a sound goes out."""
LONGEST = 200000
"""Characters of style plus lyrics the token route will read.

A song's lyrics are a few thousand characters at most. The cap is there so a
pasted book cannot tie up a server thread, not to shape any real use."""


def answer_tokens(body) -> tuple:
    """``(payload, status)`` for one request to the token route.

    Kept apart from the aiohttp handler so it can be tested without a server.
    """
    if not isinstance(body, dict):
        return {"ok": False, "error": "Send a JSON object with style and lyrics."}, 400
    style = body.get("style", "")
    lyrics = body.get("lyrics", "")
    if not isinstance(style, str) or not isinstance(lyrics, str):
        return {"ok": False, "error": "style and lyrics must be strings."}, 400
    if len(style) + len(lyrics) > LONGEST:
        return {"ok": False, "error": "That is far longer than any song."}, 413

    from . import tokens

    try:
        return tokens.lyric_cuts(style, lyrics), 200
    except Exception as error:
        log.warning("[yue2_comfy.routes] token cuts failed: %s", error, exc_info=True)
        return {"ok": True, "available": False, "problem": str(error)}, 200


def _score_problem(error) -> dict:
    """A failure the editor shows as it is, logged when it is not the person's doing."""
    if not isinstance(error, ValueError):
        log.warning("[yue2_comfy.routes] the score editor failed: %s", error, exc_info=True)
        return {"ok": False, "error": "The score editor hit an error it did not expect: "
                                      "{}".format(error)}
    return {"ok": False, "error": str(error)}


def answer_score_read(body) -> tuple:
    """``(payload, status)`` for the score editor asking to draw a score."""
    if not isinstance(body, dict) or not isinstance(body.get("abc"), str):
        return {"ok": False, "error": "Send a JSON object with the score as 'abc'."}, 400

    from . import notation

    try:
        return {"ok": True, "sheet": notation.read(body["abc"])}, 200
    except Exception as error:  # noqa: BLE001 - the window shows what went wrong
        return _score_problem(error), 200


def answer_score_write(body) -> tuple:
    """``(payload, status)`` for the score editor writing an edit into a score.

    The answer carries the new text, the bars that were written again, and the
    new text read back, so the window redraws from what the score now says
    rather than from what it asked for.
    """
    if not isinstance(body, dict) or not isinstance(body.get("abc"), str) \
            or "sheet" not in body:
        return {"ok": False, "error": "Send a JSON object with the score as 'abc' and "
                                      "the edit as 'sheet'."}, 400

    from . import notation

    try:
        written = notation.write(body["abc"], body["sheet"])
        return {"ok": True, "abc": written["abc"], "bars": written["bars"],
                "sheet": notation.read(written["abc"])}, 200
    except Exception as error:  # noqa: BLE001 - the window shows what went wrong
        return _score_problem(error), 200


def answer_score_length(body) -> tuple:
    """``(payload, status)`` for the editor making a score longer, or making one from nothing.

    An empty ``abc`` asks for a blank score of that many bars, which is how the
    window offers a grid to someone with nothing to edit yet; anything else is
    the score to add empty bars to. The answer is read back before it is sent,
    so a text the roll could not draw never reaches the window.
    """
    if not isinstance(body, dict) or not isinstance(body.get("abc"), str) \
            or not isinstance(body.get("bars"), int) or isinstance(body.get("bars"), bool):
        return {"ok": False, "error": "Send a JSON object with the score as 'abc' and "
                                      "the bars wanted as a whole number 'bars'."}, 400

    from . import notation

    bpm = body.get("bpm", notation.BLANK_BPM)
    if not isinstance(bpm, int) or isinstance(bpm, bool):
        return {"ok": False, "error": "'bpm' must be a whole number."}, 400
    try:
        text = notation.blank(body["bars"], bpm) if not body["abc"].strip() \
            else notation.lengthened(body["abc"], body["bars"])
        notation.read(text)
        return {"ok": True, "abc": text}, 200
    except Exception as error:  # noqa: BLE001 - the window shows what went wrong
        return _score_problem(error), 200


def answer_midi_tracks(body) -> tuple:
    """``(payload, status)`` for 'YuE2 Load MIDI' asking what a file in the input folder holds.

    Asked before any run, so the list on the node names the tracks while they
    are still being chosen. A file that cannot be read, or a choice of tracks
    that cannot be sung, is 200 with ``ok: false`` and the reason -- with the
    tracks, when the file itself could be read.
    """
    if not isinstance(body, dict) or not isinstance(body.get("name"), str):
        return {"ok": False, "error": "Send a JSON object with the file's name as 'name'."}, 400

    from . import load_midi

    choice = [str(body.get(key) or default)
              for key, default in (("mode", "melody"), ("vocal_track", "auto"), ("instrument_track", "auto"))]
    try:
        return load_midi.summary(body["name"], *choice), 200
    except (OSError, ValueError) as error:
        return {"ok": False, "error": str(error), "parts": []}, 200
    except Exception as error:  # noqa: BLE001 - the node shows what went wrong
        log.warning("[yue2_comfy.routes] reading a MIDI file failed: %s", error, exc_info=True)
        return {"ok": False, "parts": [],
                "error": "Reading the file hit an error it did not expect: {}".format(error)}, 200


def answer_score_midi(body) -> tuple:
    """``(payload, status)`` for the score editor saving a score as a MIDI file, handed back as base64."""
    if not isinstance(body, dict) or not isinstance(body.get("abc"), str):
        return {"ok": False, "error": "Send a JSON object with the score as 'abc'."}, 400
    if len(body["abc"]) > LONGEST:
        return {"ok": False, "error": "That is far longer than any score."}, 413

    import base64

    from . import edits
    from .midi import export

    try:
        data = export.midi_of(edits.read(body["abc"]).score)
    except Exception as error:  # noqa: BLE001 - the window shows what went wrong
        return _score_problem(error), 200
    return {"ok": True, "data": base64.b64encode(data).decode("ascii")}, 200


def answer_loras() -> tuple:
    """``(payload, status)`` for 'YuE2 LoRA' asking which LoRA files are for YuE2.

    Every file in ComfyUI's LoRA folders whose header lands on YuE2, with what
    the node shows under a row: the halves it changes, its rank, its trigger
    words, and the reason when it cannot be used. Each header is read once per
    file; see ``lora.catalogue``.
    """
    from .lora import catalogue

    try:
        return {"ok": True, "loras": catalogue.listing()}, 200
    except Exception as error:  # noqa: BLE001 - the node shows what went wrong
        log.warning("[yue2_comfy.routes] listing the LoRA files failed: %s", error, exc_info=True)
        return {"ok": False, "loras": [],
                "error": "Listing the LoRA files hit an error it did not expect: {}".format(error)}, 200


def _byte_span(wanted, size: int):
    """The one byte range a Range header asks for, as ``(first, last)`` within a file of ``size``.

    None when there is no usable single range, so the whole file goes out with
    200 as the RFC allows; ``()`` when the range lies beyond the file, for 416.
    """
    if not isinstance(wanted, str):
        return None
    match = re.match(r"^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$", wanted)
    if match is None:
        return None
    first, last = match.group(1), match.group(2)
    if not first and not last:
        return None
    if not first:
        count = min(int(last), size)
        return () if count == 0 else (size - count, size - 1)
    start = int(first)
    if start >= size:
        return ()
    end = size - 1 if not last else min(int(last), size - 1)
    return () if end < start else (start, end)


def sound_answer(name, wanted, folder) -> tuple:
    """``(status, headers, path, offset, count)`` for one request to the sound route.

    The route exists because of how the file used to go out. ComfyUI's /view
    sends a file through aiohttp's FileResponse, which hands the body to the
    loop's sendfile -- TransmitFile on Windows -- and client editions of Windows
    allow only a couple of those at a time for the whole machine. A browser's
    audio element keeps its request open while it sits on a full buffer, so two
    takes warmed for switching were enough to leave the playing one, the other
    takes and even the index page with headers and no body. Measured 2026-09-22
    on the user's own server: one slow reader was already enough there.

    Kept apart from the aiohttp handler so it can be tested without a server:
    it decides status, headers and which bytes of which file to send. Only a
    plain file name under the node's own folder is served; a range asks for one
    span of bytes as the browser does when it seeks.
    """
    if not folder or not isinstance(name, str) or SOUND_NAME.match(name) is None:
        return 400, {"Content-Length": "0"}, None, 0, 0
    path = os.path.join(folder, name)
    try:
        size = os.path.getsize(path)
    except OSError:
        return 404, {"Content-Length": "0"}, None, 0, 0
    headers = {"Accept-Ranges": "bytes", "Content-Type": "audio/wav", "Cache-Control": "no-cache"}
    span = _byte_span(wanted, size)
    if span is None:
        headers["Content-Length"] = str(size)
        return 200, headers, path, 0, size
    if span == ():
        headers["Content-Range"] = "bytes */{}".format(size)
        headers["Content-Length"] = "0"
        return 416, headers, None, 0, 0
    start, end = span
    headers["Content-Range"] = "bytes {}-{}/{}".format(start, end, size)
    headers["Content-Length"] = str(end - start + 1)
    return 206, headers, path, start, end - start + 1


def sound_folder():
    """Where the node's sounds are on this server; None outside ComfyUI."""
    from . import paths

    folder_paths = paths._folder_paths()
    if folder_paths is None:
        return None
    return os.path.join(folder_paths.get_temp_directory(), SOUND_SUBFOLDER)


async def send_sound(request, folder):
    """The aiohttp half of the sound route: the answer of ``sound_answer`` written in plain chunks.

    Reads happen off the loop, writes wait for the socket, and nothing is
    handed to sendfile, so a browser holding one take open costs nobody else
    their file. A client that goes away mid-file ends the loop quietly.
    """
    from aiohttp import web

    status, headers, path, offset, count = sound_answer(
        request.query.get("name", ""), request.headers.get("Range"), folder)
    response = web.StreamResponse(status=status, headers=headers)
    await response.prepare(request)
    if path is not None and request.method == "GET":
        try:
            with open(path, "rb") as handle:
                handle.seek(offset)
                left = count
                while left > 0:
                    piece = await asyncio.to_thread(handle.read, min(SOUND_CHUNK, left))
                    if not piece:
                        break
                    left -= len(piece)
                    await response.write(piece)
        except (ConnectionResetError, ConnectionAbortedError):
            return response
    await response.write_eof()
    return response


def register() -> None:
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

    async def body_of(request):
        try:
            return await request.json()
        except ValueError:
            return None

    @routes.post(PREFIX + "/score/read")
    async def score_read(request):
        """A score as notes, for the piano roll."""
        payload, status = await asyncio.to_thread(answer_score_read, await body_of(request))
        return web.json_response(payload, status=status)

    @routes.post(PREFIX + "/score/write")
    async def score_write(request):
        """Edited notes written back into the score."""
        payload, status = await asyncio.to_thread(answer_score_write, await body_of(request))
        return web.json_response(payload, status=status)

    @routes.post(PREFIX + "/score/length")
    async def score_length(request):
        """A blank score, or the one sent with empty bars added at its end."""
        payload, status = await asyncio.to_thread(answer_score_length, await body_of(request))
        return web.json_response(payload, status=status)

    @routes.post(PREFIX + "/score/midi")
    async def score_midi(request):
        """The score as a MIDI file, for 'Save as MIDI...'."""
        payload, status = await asyncio.to_thread(answer_score_midi, await body_of(request))
        return web.json_response(payload, status=status)

    @routes.post(PREFIX + "/midi/tracks")
    async def midi_tracks(request):
        """What a MIDI file in the input folder holds, for the list on 'YuE2 Load MIDI'."""
        payload, status = await asyncio.to_thread(answer_midi_tracks, await body_of(request))
        return web.json_response(payload, status=status)

    @routes.get(PREFIX + "/loras")
    async def loras(request):
        """The LoRA files for YuE2, for the rows on 'YuE2 LoRA'."""
        payload, status = await asyncio.to_thread(answer_loras)
        return web.json_response(payload, status=status)

    @routes.get(PREFIX + "/sound")
    async def sound(request):
        """The song or a take from 'YuE2 Edit Track', for the window's players; see ``sound_answer``."""
        return await send_sound(request, sound_folder())

    @routes.post(PREFIX + "/tokens")
    async def lyric_tokens(request):
        """Where the tokenizer cuts the lyrics, drawn under the letters in the editor."""
        try:
            body = await request.json()
        except ValueError:
            body = None
        payload, status = await asyncio.to_thread(answer_tokens, body)
        return web.json_response(payload, status=status)


try:
    register()
    log.info("[yue2_comfy] routes registered")
except Exception as error:  # noqa: BLE001 - the nodes must still load
    log.debug("[yue2_comfy] routes not registered: %s", error)
