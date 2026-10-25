"""Which requests may change what the pack keeps: a browser on this machine, and nobody else unless the operator says so.

The rule itself is checked on headers, and then on the real routes, registered
on a test server the way ComfyUI registers them: every route that writes,
deletes, opens or reads a path from the request answers a request from
elsewhere with 403 and does none of its work.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from yue2_comfy import access, model_list, routes, songs

LOCAL = {"Host": "127.0.0.1:8188", "Origin": "http://127.0.0.1:8188", "Sec-Fetch-Site": "same-origin"}
KEY = "ab" * 32


@pytest.fixture(autouse=True)
def switch_off(monkeypatch):
    monkeypatch.delenv(access.SWITCH, raising=False)


@pytest.mark.parametrize("remote, headers", [
    ("127.0.0.1", LOCAL),
    ("::1", {"Host": "[::1]:8188", "Origin": "http://[::1]:8188"}),
    ("::ffff:127.0.0.1", {"Host": "localhost:8188"}),
    ("127.0.0.1", {"host": "LOCALHOST"}),
    ("127.0.0.1", {"Host": "127.0.0.1:8188", "Sec-Fetch-Site": "none"}),
    ("127.0.0.1", {"Host": "127.0.0.1:9000", "Origin": "http://127.0.0.1:9000", "Sec-Fetch-Site": "same-origin"}),
])
def test_a_browser_on_this_machine_may(remote, headers):
    """The last case is an SSH tunnel from another computer on a port of its own: sshd connects from the server."""
    assert access.refusal(remote, headers) == ""


@pytest.mark.parametrize("remote, headers", [
    ("192.168.1.20", LOCAL),
    ("127.0.0.1", dict(LOCAL, Host="comfy.example:8188")),
    ("127.0.0.1", dict(LOCAL, Host="192.168.1.10:8188")),
    ("127.0.0.1", {"Host": "127.0.0.1:8188"} | {"X-Forwarded-For": "203.0.113.9"}),
    ("127.0.0.1", {"Host": "127.0.0.1:8188", "Forwarded": "for=203.0.113.9"}),
    ("127.0.0.1", {"Host": "127.0.0.1:8188", "X-Real-IP": "203.0.113.9"}),
    ("127.0.0.1", dict(LOCAL, **{"Sec-Fetch-Site": "cross-site"})),
    ("127.0.0.1", dict(LOCAL, **{"Sec-Fetch-Site": "same-site"})),
    ("127.0.0.1", dict(LOCAL, Origin="http://evil.example")),
    ("127.0.0.1", dict(LOCAL, Origin="null")),
    ("127.0.0.1", {}),
    (None, LOCAL),
])
def test_anyone_else_is_refused_and_told_how_to_allow_it(remote, headers):
    """Another address, a host that is not this machine, a proxy, another site's page."""
    said = access.refusal(remote, headers)
    assert said == access.REMOTE and access.SWITCH in said and "ssh -L" in said


def test_the_operator_can_open_them_to_the_network():
    assert access.refusal("192.168.1.20", {"Host": "comfy.example"}, {access.SWITCH: "1"}) == ""
    assert access.refusal("192.168.1.20", {"Host": "comfy.example"}, {access.SWITCH: "0"}) == access.REMOTE


def test_an_aiohttp_request_is_read_by_its_peer_and_headers():
    request = types.SimpleNamespace(remote="127.0.0.1", headers={"Host": "127.0.0.1:8188"})
    assert access.refusal_of(request) == ""
    request.remote = "10.0.0.7"
    assert access.refusal_of(request) == access.REMOTE


GUARDED = [
    ("/yue2/writers/save", {"entry": {"name": "N", "repo": "o/n", "file": "n.gguf"}}),
    ("/yue2/writers/delete", {"name": "N"}),
    ("/yue2/writers/restore", {}),
    ("/yue2/writers/check", {"entry": {"name": "N", "file": "C:\\Windows\\win.ini.gguf"}}),
    ("/yue2/writers/open", {"what": "folder"}),
    ("/yue2/songs/note", {"key": KEY, "note": "x"}),
    ("/yue2/songs/drop", {"key": KEY}),
    ("/yue2/songs/sounds", {"sweep": True}),
]


@pytest.fixture
def server(monkeypatch):
    """The pack's routes registered on a test server; ``done`` names the work each one reached."""
    pytest.importorskip("aiohttp")
    from aiohttp import web

    table = web.RouteTableDef()
    fake = types.ModuleType("server")
    fake.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=table))
    monkeypatch.setitem(sys.modules, "server", fake)
    done = []

    def did(what, answer=None):
        def work(*args, **kwargs):
            done.append((what, args, kwargs))
            return dict(answer or {"ok": True})
        return work

    monkeypatch.setattr(model_list, "save", did("save"))
    monkeypatch.setattr(model_list, "delete", did("delete"))
    monkeypatch.setattr(model_list, "restore", did("restore"))
    monkeypatch.setattr(model_list, "check", did("check"))
    monkeypatch.setattr(model_list, "reveal", did("open"))
    monkeypatch.setattr(model_list, "listing", did("listing"))
    monkeypatch.setattr(songs, "remark", did("note", ""))
    monkeypatch.setattr(songs, "drop", did("drop", {"keys": [], "bytes": 0}))
    monkeypatch.setattr(songs, "sweep_sounds", did("sweep"))
    monkeypatch.setattr(songs, "sounds", lambda: {"on": False, "count": 0, "bytes": 0, "budget": 0,
                                                  "can": True, "why": ""})
    monkeypatch.setattr(songs, "listing", lambda limit=0: [])
    routes.register()

    def run(calls):
        from aiohttp.test_utils import TestClient, TestServer

        async def main():
            app = web.Application()
            app.add_routes(table)
            async with TestClient(TestServer(app)) as client:
                return await calls(client)

        return asyncio.run(main())

    return run, done


def test_every_route_that_changes_something_refuses_a_request_from_elsewhere(server):
    run, done = server

    async def calls(client):
        found = []
        for path, body in GUARDED:
            answer = await client.post(path, json=body, headers={"Host": "comfy.example:8188"})
            found.append((path, answer.status, json.loads(await answer.text())))
        return found

    for path, status, payload in run(calls):
        assert status == 403, path
        assert payload == {"ok": False, "error": access.REMOTE, "local": False}, path
    assert done == [], "none of them did any of its work"


def test_the_same_routes_answer_a_browser_on_this_machine(server):
    run, done = server

    async def calls(client):
        return [(await client.post(path, json=body)).status for path, body in GUARDED]

    assert run(calls) == [200] * len(GUARDED)
    assert [what for what, _args, _kwargs in done] == ["save", "delete", "restore", "check", "open", "note",
                                                       "drop", "sweep"]


def test_the_lists_say_whether_the_browser_may_change_them(server):
    run, done = server

    async def calls(client):
        far = {"Host": "comfy.example:8188"}
        writers_far = json.loads(await (await client.get("/yue2/writers", headers=far)).text())
        writers_here = json.loads(await (await client.get("/yue2/writers")).text())
        songs_far = json.loads(await (await client.get("/yue2/songs", headers=far)).text())
        songs_here = json.loads(await (await client.get("/yue2/songs")).text())
        return writers_far, writers_here, songs_far, songs_here

    writers_far, writers_here, songs_far, songs_here = run(calls)
    assert [args for what, args, _kwargs in done if what == "listing"] == [(False,), (True,)]
    assert writers_far["local"] is False and writers_far["refused"] == access.REMOTE
    assert writers_here["local"] is True and writers_here["refused"] == ""
    assert songs_far["local"] is False and songs_far["sounds"]["can"] is False
    assert songs_far["sounds"]["why"] == access.REMOTE
    assert songs_here["local"] is True and songs_here["sounds"]["can"] is True
