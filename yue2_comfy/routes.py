"""HTTP routes behind the song editor's and the score editor's windows, and the MIDI and LoRA nodes' lists.

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

log = logging.getLogger(__name__)

PREFIX = "/yue2"
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
