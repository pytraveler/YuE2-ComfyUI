"""HTTP routes behind the song editor's window.

Registered on import. A failure here must never stop the nodes from loading --
the editor is a convenience on top of widgets that work without it -- so the
registration is guarded and logged rather than raised.
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


def register() -> None:
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

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
