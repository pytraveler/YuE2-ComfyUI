"""Which requests may change what the pack keeps on this machine.

ComfyUI's routes answer anyone who can reach its port: they carry no login, and
on a server started with --listen that is the whole network. Most of what the
pack's routes do is harmless from anywhere -- draw a score, list the songs,
play a take. A few change what the pack keeps or will fetch, or reach the
desktop: the model list's edits and its 'Check it' (which reads a file named in
the request), a song or a note deleted or written, the sounds switch, and the
two buttons that open a file manager. Those answer a browser on the machine
ComfyUI runs on and nobody else, the rule ComfyUI-Manager applies to its own
risky features. YUE2_ALLOW_REMOTE_EDITS=1 lifts it, for a server someone has
put on their network on purpose.

A request counts as this machine's when it comes from a loopback address,
names a loopback host, carries none of the headers a proxy adds, and was not
sent by a page of another site. A tunnel or reverse proxy on the same machine
makes the outside look local by its address; the Host it passes or the headers
it adds usually give it away. One that rewrites both cannot be told from the
person at the keyboard, and whoever set it up has opened /prompt to the same
people anyway.

An SSH tunnel to the server (``ssh -L 8188:127.0.0.1:8188 server``, then
http://127.0.0.1:8188 in the browser) counts as this machine on purpose: sshd
opens the connection from the server itself, the browser names a loopback
host, and only someone who can log in to the server can open one. That is the
way in from afar that needs no switch, and the refusal says so.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlsplit

SWITCH = "YUE2_ALLOW_REMOTE_EDITS"
SWITCH_ON = ("1", "true", "yes", "on")
LOCAL_NAMES = ("localhost",)
FORWARDED = ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-real-ip")
FOREIGN_SITES = ("cross-site", "same-site")

REMOTE = (
    "This can be changed only from a browser on the machine ComfyUI runs on, with ComfyUI opened as "
    "127.0.0.1 or localhost. From another computer, an SSH tunnel counts as that machine: "
    "ssh -L 8188:127.0.0.1:8188 <server>, then open http://127.0.0.1:8188. To allow it from the network "
    "instead, start ComfyUI with {}=1 in its environment.".format(SWITCH)
)


def switched_on(environ=None) -> bool:
    """Whether the operator has opened these routes to the network."""
    value = (os.environ if environ is None else environ).get(SWITCH, "")
    return str(value).strip().lower() in SWITCH_ON


def _host(value) -> str:
    """The host part of a Host header or an Origin's netloc, lowercased, without a port or brackets."""
    text = str(value or "").strip().lower()
    if text.startswith("["):
        return text[1:text.index("]")] if "]" in text else ""
    if text.count(":") == 1:
        return text.split(":", 1)[0]
    return text


def loopback(host) -> bool:
    """Whether a host or an address names this machine itself."""
    text = _host(host)
    if text in LOCAL_NAMES:
        return True
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_loopback


def refusal(remote, headers, environ=None) -> str:
    """Why a request may not change what the pack keeps, or "" when it may.

    ``remote`` is the peer's address and ``headers`` the request's headers,
    any mapping; names are compared without case, as HTTP compares them.
    """
    if switched_on(environ):
        return ""
    named = {str(key).lower(): value for key, value in dict(headers or {}).items()}
    if not loopback(remote) or not loopback(named.get("host")):
        return REMOTE
    if any(named.get(key) for key in FORWARDED):
        return REMOTE
    if str(named.get("sec-fetch-site") or "").strip().lower() in FOREIGN_SITES:
        return REMOTE
    origin = named.get("origin")
    if origin is not None and not loopback(urlsplit(str(origin)).netloc):
        return REMOTE
    return ""


def refusal_of(request) -> str:
    """``refusal`` for an aiohttp request."""
    return refusal(request.remote, request.headers)
