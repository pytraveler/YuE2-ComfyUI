"""The Model list window's server half: what it shows, the edits it makes, Check it, and its routes.

The machine is test_catalog's: a two-entry seed (Alpha 1B, Beta 2B), one model
folder, no Ollama store, and a download that must never happen. The Hub is a
function returning a listing.
"""

import json
import os

import pytest

from test_catalog import machine, seeded  # noqa: F401 - fixtures used by name
from test_llm import write_gguf
from yue2_comfy import catalog, download, model_list, routes

HUB = {
    "Beta-2B.gguf": int(1.5 * 1024 ** 3),
    "Beta-2B-Q8_0.gguf": 3 * 1024 ** 3,
    "mmproj-F16.gguf": 900 * 1024 ** 2,
    "big/Beta-big-00001-of-00002.gguf": 4 * 1024 ** 3,
    "big/Beta-big-00002-of-00002.gguf": 4 * 1024 ** 3,
    "README.md": 900,
}


@pytest.fixture(autouse=True)
def rendered(monkeypatch):
    """Templates render without jinja2, which CI does not install; ComfyUI has it through torch.

    ``rendered.error`` set to an exception makes the next render raise it.
    """
    state = {"error": None}

    def render(template, messages, **kwargs):
        if state["error"] is not None:
            raise state["error"]
        return str(messages)

    monkeypatch.setattr(model_list.chat_template, "render", render)
    return state


@pytest.fixture
def hub(monkeypatch):
    """Hugging Face as a listing; ``asked`` records the repositories asked about."""
    asked = []

    def listing(repo, revision="main", token=None):
        asked.append(repo)
        if repo == "owner/Missing-GGUF":
            raise download.DownloadError("HTTP 404 for 'owner/Missing-GGUF' (main) -- no such repository, "
                                         "revision, or file.")
        return dict(HUB)

    monkeypatch.setattr(download, "list_repo_files", listing)
    monkeypatch.setattr(download, "access_token", lambda: None)
    return asked


def row(state, name):
    return next(one for one in state["entries"] if one.get("name") == name)


def test_the_window_shows_each_entry_where_it_is_and_the_files_found(machine):
    folder, fetched = machine
    wanted = write_gguf(folder / "Alpha-1B.gguf")
    write_gguf(folder / "own.gguf")
    write_gguf(folder / "embed.gguf", arch="qwen3", ints={"qwen3.pooling_type": 3})
    state = model_list.listing()
    alpha, beta = row(state, "Alpha 1B"), row(state, "Beta 2B")
    assert alpha["label"] == "Alpha 1B (on disk, ~3 GB VRAM)" and alpha["on_disk"] and alpha["where"] == wanted
    assert beta["label"] == "Beta 2B (download 1.5 GB, ~3 GB VRAM)" and not beta["on_disk"]
    assert alpha["from_pack"] and beta["from_pack"] and alpha["usable"]
    assert [found["label"].split(" (")[0] for found in state["found"]] == ["own.gguf"]
    assert state["writable"] and state["problem"] == "" and state["restorable"] == []
    assert state["choices"][0] == "auto" and state["folder"] == str(folder)
    assert fetched == [] and not os.path.exists(catalog.live_file())


def test_an_entry_the_widget_cannot_use_is_shown_to_be_mended(machine):
    os.makedirs(os.path.dirname(catalog.live_file()), exist_ok=True)
    with open(catalog.live_file(), "wb") as handle:
        handle.write(json.dumps({"writers": [{"name": "Half", "repo": "o/h"}]}).encode("utf-8"))
    catalog.forget()
    state = model_list.listing()
    half = row(state, "Half")
    assert half["usable"] is False and half["label"] == "Half"


def test_a_list_that_does_not_parse_is_shown_but_locked(machine):
    os.makedirs(os.path.dirname(catalog.live_file()), exist_ok=True)
    with open(catalog.live_file(), "wb") as handle:
        handle.write(b"{ not json")
    catalog.forget()
    state = model_list.listing()
    assert not state["writable"] and "not valid JSON" in state["problem"]
    payload, status = routes.answer_writers_save({"entry": {"name": "N", "repo": "o/n", "file": "n.gguf"}})
    assert status == 200 and payload["ok"] is False and "has to parse" in payload["error"]


def test_saving_adds_edits_and_says_when_a_name_changed(machine):
    added = model_list.save("", {"name": "Gamma 3B", "repo": "owner/Gamma-GGUF", "file": "g.gguf",
                                 "download_gb": "2", "vram": "~4 GB VRAM"})
    assert added["label"] == "Gamma 3B (download 2 GB, ~4 GB VRAM)" and added["renamed"] == ""
    assert added["label"] in added["choices"]
    edited = model_list.save("Gamma 3B", {"name": "Gamma 3B", "repo": "owner/Gamma-GGUF", "file": "g.gguf",
                                          "vram": "~5 GB VRAM"})
    assert edited["label"] == "Gamma 3B (download, ~5 GB VRAM)" and edited["renamed"] == ""
    renamed = model_list.save("Gamma 3B", {"name": "Gamma Three", "repo": "owner/Gamma-GGUF", "file": "g.gguf"})
    assert renamed["renamed"] == "Gamma 3B" and renamed["label"] == "Gamma Three (download)"
    recased = model_list.save("Gamma Three", {"name": "gamma three", "repo": "owner/Gamma-GGUF", "file": "g.gguf"})
    assert recased["renamed"] == ""


def test_a_refused_edit_is_read_out_not_raised(machine):
    payload, status = routes.answer_writers_save({"entry": {"name": "Alpha 1B", "repo": "o/a", "file": "a.gguf"}})
    assert status == 200 and payload == {"ok": False, "error": "The list already has a model called 'Alpha 1B'."}
    payload, status = routes.answer_writers_save({"was": "Nobody", "entry": {"name": "X", "repo": "o/x",
                                                                            "file": "x.gguf"}})
    assert status == 200 and "no longer in the list" in payload["error"]


def test_a_deleted_entry_hands_its_nodes_to_its_own_file(machine):
    """The same model under its file name; nothing else is picked for them."""
    folder, _fetched = machine
    write_gguf(folder / "Alpha-1B.gguf")
    gone = model_list.delete("Alpha 1B")
    assert gone["gone"] and gone["moved_to"].startswith("Alpha-1B.gguf (")
    assert gone["moved_to"] in gone["choices"]
    kept = model_list.delete("Beta 2B")
    assert kept["gone"] and kept["moved_to"] == ""
    assert model_list.delete("Nobody")["gone"] is False


def test_the_packaged_entries_come_back(machine):
    model_list.delete("Beta 2B")
    assert model_list.listing()["restorable"] == ["Beta 2B"]
    back = model_list.restore()
    assert back["restored"] == ["Beta 2B"] and any(choice.startswith("Beta 2B (") for choice in back["choices"])


def test_check_reads_a_file_on_this_machine(machine, hub):
    folder, _fetched = machine
    path = write_gguf(folder / "mine.gguf")
    found = model_list.check({"name": "", "file": path})
    assert found["verdict"] == "good" and found["here"] and hub == []
    assert "Its chat template renders" in found["lines"][-1]["text"]
    assert found["download_gb"] == round(os.path.getsize(path) / 1024 ** 3, 2)


def test_check_says_when_the_template_would_stop_the_first_run(machine, hub, rendered):
    folder, _fetched = machine
    rendered["error"] = ValueError("unknown tag 'generation'")
    found = model_list.check({"file": write_gguf(folder / "odd.gguf")})
    assert found["verdict"] == "bad" and "does not render here (unknown tag 'generation')" in found["lines"][-1]["text"]


def test_check_says_why_a_file_cannot_write(machine, hub):
    folder, _fetched = machine
    path = write_gguf(folder / "embed.gguf", arch="qwen3", ints={"qwen3.pooling_type": 3})
    found = model_list.check({"file": path})
    assert found["verdict"] == "bad" and "embedding model" in found["lines"][0]["text"]
    gone = model_list.check({"file": str(folder / "never.gguf")})
    assert gone["verdict"] == "bad" and "is not on this machine" in gone["lines"][0]["text"]


def test_check_reads_a_listed_file_already_here_instead_of_asking_the_hub(machine, hub):
    folder, _fetched = machine
    write_gguf(folder / "Beta-2B.gguf")
    found = model_list.check({"name": "Beta 2B", "repo": "owner/Beta-2B-GGUF", "file": "Beta-2B.gguf"})
    assert found["here"] and found["verdict"] == "good" and hub == []


def test_check_asks_the_hub_whether_the_file_is_there_and_how_big(machine, hub):
    found = model_list.check({"repo": "owner/Beta-2B-GGUF", "file": "Beta-2B.gguf"})
    assert found["verdict"] == "good" and found["download_gb"] == 1.5 and not found["here"]
    assert found["files"] == [] and hub == ["owner/Beta-2B-GGUF"]
    split = model_list.check({"repo": "owner/Beta-2B-GGUF", "file": "big/Beta-big-00001-of-00002.gguf"})
    assert split["download_gb"] == 8.0 and "and its other parts" in split["lines"][0]["text"]


def test_a_misspelt_file_is_named_with_the_models_the_repository_has(machine, hub):
    found = model_list.check({"repo": "owner/Beta-2B-GGUF", "file": "Beta-2B-Q4.gguf"})
    assert found["verdict"] == "bad" and "has no file 'Beta-2B-Q4.gguf'" in found["lines"][0]["text"]
    assert found["files"] == [{"file": "Beta-2B-Q8_0.gguf", "gb": 3.0}, {"file": "Beta-2B.gguf", "gb": 1.5},
                              {"file": "big/Beta-big-00001-of-00002.gguf", "gb": 8.0}]


def test_a_repository_alone_lists_its_models(machine, hub):
    found = model_list.check({"repo": "owner/Beta-2B-GGUF", "file": ""})
    assert found["verdict"] == "good" and len(found["files"]) == 3
    payload, status = routes.answer_writers_check({"entry": {"repo": "not a repo"}})
    assert status == 200 and "not a Hugging Face repository id" in payload["error"]


def test_a_repository_the_hub_does_not_know_is_said_plainly(machine, hub):
    found = model_list.check({"repo": "owner/Missing-GGUF", "file": "m.gguf"})
    assert found["verdict"] == "bad" and "HTTP 404" in found["lines"][0]["text"]


def test_opening_the_list_writes_it_first_and_opens_only_two_things(machine, monkeypatch):
    folder, _fetched = machine
    opened = []
    monkeypatch.setattr(model_list, "_open", opened.append)
    assert not os.path.exists(catalog.live_file())
    shown = model_list.reveal("list")
    assert shown == {"ok": True, "path": catalog.live_file()} and opened == [catalog.live_file()]
    with open(catalog.live_file(), "rb") as handle:
        assert [raw["name"] for raw in json.loads(handle.read().decode("utf-8"))["writers"]] == ["Alpha 1B",
                                                                                               "Beta 2B"]
    assert model_list.reveal("folder") == {"ok": True, "path": str(folder)}
    payload, status = routes.answer_writers_open({"what": "C:\\Windows"})
    assert status == 400 and len(opened) == 2


def test_a_desktop_that_cannot_open_it_still_says_where_it_is(machine, monkeypatch):
    def refuse(path):
        raise OSError("no application is associated")

    monkeypatch.setattr(model_list, "_open", refuse)
    shown = model_list.reveal("list")
    assert shown["ok"] is False and shown["path"] == catalog.live_file() and "no application" in shown["error"]


@pytest.mark.parametrize("answer, body", [
    (routes.answer_writers_save, None),
    (routes.answer_writers_save, {"entry": "not a dict"}),
    (routes.answer_writers_save, {"entry": {}, "was": 3}),
    (routes.answer_writers_delete, {"name": 3}),
    (routes.answer_writers_check, {"entry": []}),
    (routes.answer_writers_open, {}),
])
def test_a_request_the_window_does_not_send_is_a_400(answer, body):
    payload, status = answer(body)
    assert status == 400 and payload["ok"] is False


def test_the_routes_answer_the_window(machine):
    payload, status = routes.answer_writers()
    assert status == 200 and payload["ok"] and [one["name"] for one in payload["entries"]] == ["Alpha 1B",
                                                                                              "Beta 2B"]
    payload, status = routes.answer_writers_delete({"name": "Beta 2B"})
    assert status == 200 and payload["gone"]
    payload, status = routes.answer_writers_restore()
    assert status == 200 and payload["restored"] == ["Beta 2B"]


def test_an_unexpected_failure_is_logged_and_read_out(machine, monkeypatch):
    def broken():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(model_list, "listing", broken)
    payload, status = routes.answer_writers()
    assert status == 200 and payload["ok"] is False and "disk on fire" in payload["error"]
