"""The sound route behind the Edit Track window: what it serves, which bytes, and that one held take starves no other.

ComfyUI's /view hands file bodies to the loop's sendfile, which on Windows is
TransmitFile, and client editions of Windows run only a couple of those at a
time for the whole machine. A browser's audio element keeps its request open
while it sits on a full buffer, so the takes warmed for switching left the
playing take with headers and no body. The pack's route writes plain chunks.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from yue2_comfy import routes


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "song_0123abcd.wav").write_bytes(bytes(range(256)) * 4)
    (tmp_path / "empty.wav").write_bytes(b"")
    return str(tmp_path)


def test_a_whole_file_goes_out_with_its_size_and_kind(folder):
    status, headers, path, offset, count = routes.sound_answer("song_0123abcd.wav", None, folder)
    assert status == 200
    assert path == os.path.join(folder, "song_0123abcd.wav")
    assert (offset, count) == (0, 1024)
    assert headers["Content-Length"] == "1024"
    assert headers["Content-Type"] == "audio/wav"
    assert headers["Accept-Ranges"] == "bytes", "the browser seeks by asking for byte ranges"


@pytest.mark.parametrize("wanted, span, told", [
    ("bytes=0-", (0, 1024), "bytes 0-1023/1024"),
    ("bytes=100-199", (100, 100), "bytes 100-199/1024"),
    ("bytes=1000-5000", (1000, 24), "bytes 1000-1023/1024"),
    ("bytes=-24", (1000, 24), "bytes 1000-1023/1024"),
    ("bytes=-5000", (0, 1024), "bytes 0-1023/1024"),
    ("bytes = 8 - 15", (8, 8), "bytes 8-15/1024"),
])
def test_a_range_answers_with_just_those_bytes(folder, wanted, span, told):
    status, headers, path, offset, count = routes.sound_answer("song_0123abcd.wav", wanted, folder)
    assert status == 206
    assert path is not None
    assert (offset, count) == span
    assert headers["Content-Range"] == told
    assert headers["Content-Length"] == str(span[1])


@pytest.mark.parametrize("wanted", ["bytes=1024-", "bytes=2000-3000", "bytes=-0", "bytes=9-8"])
def test_a_range_past_the_file_is_refused_with_the_size(folder, wanted):
    status, headers, path, offset, count = routes.sound_answer("song_0123abcd.wav", wanted, folder)
    assert status == 416
    assert path is None and count == 0
    assert headers["Content-Range"] == "bytes */1024"


@pytest.mark.parametrize("wanted", ["", "bytes=", "bytes=-", "bytes=0-1,5-9", "items=0-1", "0-100", 7])
def test_a_range_the_route_cannot_read_means_the_whole_file(folder, wanted):
    status, headers, path, offset, count = routes.sound_answer("song_0123abcd.wav", wanted, folder)
    assert status == 200
    assert (offset, count) == (0, 1024)


def test_an_empty_file_can_only_go_out_whole(folder):
    assert routes.sound_answer("empty.wav", None, folder)[0] == 200
    assert routes.sound_answer("empty.wav", "bytes=0-", folder)[0] == 416


@pytest.mark.parametrize("name", [
    "", "song_0123abcd", "song_0123abcd.flac", "../song_0123abcd.wav", "..\\x.wav", "a/b.wav",
    ".wav", "-x.wav", "x y.wav", "x" * 130 + ".wav", None, 3,
])
def test_only_a_plain_name_the_node_would_write_is_served(folder, name):
    status, headers, path, offset, count = routes.sound_answer(name, None, folder)
    assert status == 400
    assert path is None


def test_a_file_that_is_not_there_is_404_and_no_folder_is_400(folder):
    assert routes.sound_answer("song_ffffffff.wav", None, folder)[0] == 404
    assert routes.sound_answer("song_0123abcd.wav", None, None)[0] == 400
    assert routes.sound_answer("song_0123abcd.wav", None, "")[0] == 400


def test_the_names_the_node_writes_pass(folder):
    for name in ("song_20f27503a63e0f8d.wav", "444dbdbbf54db305_520452821.wav", "fb2abc4bd804271a_0.wav"):
        assert routes.SOUND_NAME.match(name), name


def _served(folder, calls):
    """Runs ``calls(client)`` against the real route on a test server, without a running ComfyUI."""
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    async def sound(request):
        return await routes.send_sound(request, folder)

    async def main():
        app = web.Application()
        app.router.add_get("/yue2/sound", sound)
        async with TestClient(TestServer(app)) as client:
            return await calls(client)

    return asyncio.run(main())


def test_the_route_sends_the_bytes_a_range_names(folder):
    async def calls(client):
        whole = await client.get("/yue2/sound", params={"name": "song_0123abcd.wav"})
        part = await client.get("/yue2/sound", params={"name": "song_0123abcd.wav"},
                                headers={"Range": "bytes=250-259"})
        head = await client.head("/yue2/sound", params={"name": "song_0123abcd.wav"})
        gone = await client.get("/yue2/sound", params={"name": "song_ffffffff.wav"})
        bad = await client.get("/yue2/sound", params={"name": "../song_0123abcd.wav"})
        return (whole.status, await whole.read(), part.status, part.headers.get("Content-Range"),
                await part.read(), head.status, head.headers.get("Content-Length"), await head.read(),
                gone.status, bad.status)

    told = _served(folder, calls)
    assert told[0] == 200 and told[1] == bytes(range(256)) * 4
    assert told[2] == 206 and told[3] == "bytes 250-259/1024"
    assert told[4] == bytes(range(250, 256)) + bytes(range(4))
    assert told[5] == 200 and told[6] == "1024" and told[7] == b""
    assert told[8] == 404 and told[9] == 400


def test_a_take_held_open_by_one_reader_does_not_starve_another(tmp_path):
    """Two readers sit on the first bytes of a take; a third asks for a range and gets it at once.

    This is the shape the window makes: the warmed takes hold their requests,
    the player comes back for more. Through /view on Windows the third read
    stalled for as long as anyone waited.
    """
    aiohttp = pytest.importorskip("aiohttp")
    (tmp_path / "song_0123abcd.wav").write_bytes(os.urandom(6 * 1024 * 1024))

    async def calls(client):
        holders = [await client.get("/yue2/sound", params={"name": "song_0123abcd.wav"},
                                    headers={"Range": "bytes=0-"}) for _ in range(2)]
        for held in holders:
            await held.content.read(1024)
        began = time.perf_counter()
        part = await client.get("/yue2/sound", params={"name": "song_0123abcd.wav"},
                                headers={"Range": "bytes=5000000-5000999"})
        body = await asyncio.wait_for(part.read(), timeout=5.0)
        took = time.perf_counter() - began
        for held in holders:
            held.close()
        return part.status, len(body), took

    status, length, took = _served(str(tmp_path), calls)
    assert status == 206 and length == 1000
    assert took < 2.0, "the third reader waited {:.1f} s behind two held takes".format(took)
