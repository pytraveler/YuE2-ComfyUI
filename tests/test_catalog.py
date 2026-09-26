"""The model list: the pack's seed, the live copy, what an update merges in, and what the widget offers.

The live file lives in each test's own folder (see conftest). Tests that need
a different seed write one and point ``catalog.SEED_FILE`` at it.
"""

import json
import os

import pytest

from test_llm import write_gguf
from yue2_comfy import catalog, download, llm, paths


def seed_file(tmp_path, names, **extra):
    """A seed with one Hub entry per name, and whatever other keys a test wants."""
    data = {"writers": [{"name": name, "repo": "owner/" + name.replace(" ", "-") + "-GGUF",
                         "file": name.replace(" ", "-") + ".gguf", "download_gb": 1.5,
                         "vram": "~3 GB VRAM"} for name in names]}
    data.update(extra)
    path = tmp_path / ("seed_" + "_".join(n.replace(" ", "") for n in names) + ".json")
    path.write_bytes(json.dumps(data).encode("utf-8"))
    return str(path)


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    """A two-entry seed, so the tests do not hang on the pack's own list."""
    monkeypatch.setattr(catalog, "SEED_FILE", seed_file(tmp_path, ["Alpha 1B", "Beta 2B"]))
    catalog.forget()
    return tmp_path


def names() -> list:
    return [entry.name for entry in catalog.entries()]


def live() -> dict:
    with open(catalog.live_file(), "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def new_seed(tmp_path, monkeypatch, entry_names):
    monkeypatch.setattr(catalog, "SEED_FILE", seed_file(tmp_path, entry_names))
    catalog.forget()


def test_the_packaged_list_is_clean():
    """Every entry the pack ships passes the same check an edit does, and none is offered twice."""
    with open(catalog.SEED_FILE, "rb") as handle:
        data = json.loads(handle.read().decode("utf-8"))
    assert catalog.OFFERED_KEY not in data
    listed = data["writers"]
    assert listed
    for raw in listed:
        cleaned = catalog.clean_entry(raw)
        assert cleaned == {key: raw[key] for key in cleaned}
        assert raw["repo"] and raw["file"].endswith(".gguf") and raw["download_gb"] > 0
    labels = [catalog.label_of(raw) for raw in listed]
    assert len(set(labels)) == len(labels)
    assert len({raw["name"].casefold() for raw in listed}) == len(listed)


def test_the_default_writer_is_in_the_packaged_list():
    from yue2_comfy.constants import WRITER_NAME, WRITER_REPO

    assert any(entry.repo == WRITER_REPO and entry.file == WRITER_NAME for entry in catalog.entries())


def test_reading_the_list_writes_nothing(seeded):
    """Listing models happens on every page load; it must not create a file in somebody's user folder."""
    assert names() == ["Alpha 1B", "Beta 2B"]
    assert catalog.problem() == ""
    assert not os.path.exists(catalog.live_file())


def test_the_label_is_the_name_with_size_and_card():
    entry = catalog.Entry(name="Qwen 9B", file="q.gguf", repo="a/b", download_gb=5.68, vram="~7 GB VRAM",
                          note="a note that is not in the label")
    assert entry.label == "Qwen 9B (5.68 GB, ~7 GB VRAM)"
    assert catalog.Entry(name="Bare", file="b.gguf", repo="a/b").label == "Bare"


def test_the_first_edit_writes_the_whole_list_and_what_it_was_offered(seeded):
    catalog.add({"name": "Mine 3B", "repo": "me/mine-GGUF", "file": "mine.gguf"})
    data = live()
    assert [raw["name"] for raw in data["writers"]] == ["Alpha 1B", "Beta 2B", "Mine 3B"]
    assert data[catalog.OFFERED_KEY] == ["Alpha 1B", "Beta 2B"]
    assert names() == ["Alpha 1B", "Beta 2B", "Mine 3B"]


def test_a_deleted_packaged_entry_stays_deleted_through_an_update(seeded, monkeypatch):
    assert catalog.remove("Alpha 1B")
    new_seed(seeded, monkeypatch, ["Alpha 1B", "Beta 2B", "Gamma 3B"])
    assert names() == ["Beta 2B", "Gamma 3B"]
    catalog.forget()
    assert names() == ["Beta 2B", "Gamma 3B"]


def test_an_update_adds_the_new_entry_and_keeps_a_backup(seeded, monkeypatch):
    catalog.add({"name": "Mine 3B", "repo": "me/mine-GGUF", "file": "mine.gguf"})
    assert not os.path.exists(catalog.live_file() + catalog.BACKUP_SUFFIX)
    new_seed(seeded, monkeypatch, ["Alpha 1B", "Beta 2B", "Gamma 3B"])
    assert names() == ["Alpha 1B", "Beta 2B", "Mine 3B", "Gamma 3B"]
    assert os.path.exists(catalog.live_file() + catalog.BACKUP_SUFFIX)
    assert "Gamma 3B" in live()[catalog.OFFERED_KEY]


def test_a_renamed_packaged_entry_does_not_come_back_under_its_old_name(seeded, monkeypatch):
    raw = dict(catalog.raw_entries()[0], name="Alpha renamed")
    catalog.update("Alpha 1B", raw)
    new_seed(seeded, monkeypatch, ["Alpha 1B", "Beta 2B"])
    assert names() == ["Alpha renamed", "Beta 2B"]


def test_a_list_removed_from_the_file_on_purpose_stays_removed(seeded):
    catalog.add({"name": "Mine 3B", "repo": "me/mine-GGUF", "file": "mine.gguf"})
    data = live()
    del data["writers"]
    with open(catalog.live_file(), "wb") as handle:
        handle.write(json.dumps(data).encode("utf-8"))
    catalog.forget()
    assert names() == []


def test_the_merge_alone_leaves_a_file_with_nothing_new_untouched():
    seed = {"writers": [{"name": "A", "file": "a.gguf", "repo": "o/a"}]}
    live_data = {"writers": [{"name": "A", "file": "a.gguf", "repo": "o/a"}], "seed_offered": ["A"]}
    merged, added = catalog.merge(live_data, seed)
    assert merged is live_data and added == []


def test_a_file_that_does_not_parse_is_not_written_over(seeded):
    catalog.add({"name": "Mine 3B", "repo": "me/mine-GGUF", "file": "mine.gguf"})
    broken = b'{"writers": [ {"name": "Mine 3B", '
    with open(catalog.live_file(), "wb") as handle:
        handle.write(broken)
    catalog.forget()
    assert "not valid JSON" in catalog.problem()
    assert names() == ["Alpha 1B", "Beta 2B"]
    for edit in (lambda: catalog.add({"name": "X", "repo": "o/x", "file": "x.gguf"}),
                 lambda: catalog.remove("Alpha 1B"), catalog.restore_packaged, catalog.writable):
        with pytest.raises(catalog.CatalogWriteError):
            edit()
    with open(catalog.live_file(), "rb") as handle:
        assert handle.read() == broken


def test_two_edits_in_one_second_are_both_read(seeded):
    """The cache is keyed on the file's size and time; an edit of the same size in the same second must not hide."""
    catalog.add({"name": "Mine AB", "repo": "me/mine-GGUF", "file": "mine.gguf"})
    stamp = os.stat(catalog.live_file()).st_mtime_ns
    assert "Mine AB" in names()
    catalog.update("Mine AB", {"name": "Mine CD", "repo": "me/mine-GGUF", "file": "mine.gguf"})
    os.utime(catalog.live_file(), ns=(stamp, stamp))
    assert "Mine CD" in names() and "Mine AB" not in names()


def test_a_name_or_a_label_already_listed_is_refused(seeded):
    with pytest.raises(catalog.CatalogWriteError, match="already has"):
        catalog.add({"name": "alpha 1b", "repo": "o/other-GGUF", "file": "other.gguf"})
    catalog.add({"name": "Twin (1 GB)", "repo": "o/t-GGUF", "file": "t.gguf"})
    with pytest.raises(catalog.CatalogWriteError, match="already in the list"):
        catalog.update("Beta 2B", {"name": "Twin", "repo": "o/t-GGUF", "file": "u.gguf", "download_gb": 1})


def test_an_entry_can_keep_its_own_name_when_edited(seeded):
    raw = dict(catalog.raw_entries()[1], note="changed")
    catalog.update("Beta 2B", raw)
    assert catalog.entries()[1].note == "changed"


def test_editing_or_removing_a_missing_entry(seeded):
    with pytest.raises(catalog.CatalogWriteError, match="no longer in the list"):
        catalog.update("Nobody", {"name": "Nobody", "repo": "o/n", "file": "n.gguf"})
    assert catalog.remove("Nobody") is False


def test_the_pack_entries_come_back_and_your_own_stay(seeded):
    catalog.add({"name": "Mine 3B", "repo": "me/mine-GGUF", "file": "mine.gguf"})
    catalog.remove("Alpha 1B")
    assert catalog.restorable() == ["Alpha 1B"]
    assert catalog.restore_packaged() == ["Alpha 1B"]
    assert names() == ["Beta 2B", "Mine 3B", "Alpha 1B"]
    assert catalog.restorable() == [] and catalog.restore_packaged() == []


@pytest.mark.parametrize("raw, reason", [
    ({"repo": "o/n", "file": "n.gguf"}, "name"),
    ({"name": "auto", "repo": "o/n", "file": "n.gguf"}, "mistaken"),
    ({"name": "model.gguf", "repo": "o/n", "file": "n.gguf"}, "mistaken"),
    ({"name": "!! broken", "repo": "o/n", "file": "n.gguf"}, "mistaken"),
    ({"name": "ollama: x", "repo": "o/n", "file": "n.gguf"}, "mistaken"),
    ({"name": "N", "repo": "o/n"}, ".gguf"),
    ({"name": "N", "repo": "o/n", "file": "n.safetensors"}, "has to be a .gguf"),
    ({"name": "N", "repo": "not a repo", "file": "n.gguf"}, "repository id"),
    ({"name": "N", "repo": "o/n", "file": "../n.gguf"}, "inside the repository"),
    ({"name": "N", "repo": "o/n", "file": "C:/models/n.gguf"}, "inside the repository"),
    ({"name": "N", "file": "relative/n.gguf"}, "full path"),
    ({"name": "N", "file": "\\\\server\\share\\n.gguf"}, "Network"),
    ({"name": "N", "repo": "o/n", "file": "n.gguf", "download_gb": -1}, "0 or more"),
    ({"name": "N", "repo": "o/n", "file": "n.gguf", "download_gb": "big"}, "number"),
    ({"name": "N", "repo": "o/n", "file": "n.gguf", "note": "two\nlines"}, "one line"),
])
def test_what_the_list_refuses(raw, reason):
    with pytest.raises(catalog.CatalogWriteError, match=reason):
        catalog.clean_entry(raw)


def test_a_file_on_this_machine_needs_no_repository(tmp_path):
    path = str(tmp_path / "local.gguf")
    assert catalog.clean_entry({"name": "Local", "file": path, "download_gb": "", "vram": " "}) == {
        "name": "Local", "file": path}


def test_an_entry_is_found_by_label_and_by_name(seeded):
    first = catalog.entries()[0]
    assert catalog.find(first.label) == first
    assert catalog.find("Alpha 1B (9.9 GB, another note)") == first
    assert catalog.find("alpha 1b") == first
    assert catalog.find("Nobody (1 GB)") is None


@pytest.fixture
def machine(seeded, monkeypatch):
    """A models folder, no Ollama store, fresh header caches, and a download that must be asked for."""
    from yue2_comfy import ollama

    folder = seeded / "LLM"
    folder.mkdir()
    monkeypatch.setattr(ollama, "roots", lambda: [])
    monkeypatch.setattr(paths, "gguf_roots", lambda: [str(folder)])
    monkeypatch.setattr(download, "writer_root", lambda: str(folder))
    fetched = []

    def fetch_listed(name, repo, file, settings, progress=None):
        fetched.append((name, repo, file))
        return str(folder / file)

    monkeypatch.setattr(download, "fetch_listed", fetch_listed)
    llm._HEADERS.clear()
    llm._CATALOGUE.update({"roots": None, "entries": [], "at": 0.0})
    yield folder, fetched
    llm._HEADERS.clear()
    llm._CATALOGUE.update({"roots": None, "entries": [], "at": 0.0})


def test_the_widget_offers_auto_then_the_list_then_the_disk(machine):
    folder, _fetched = machine
    write_gguf(folder / "own.gguf")
    write_gguf(folder / "Alpha-1B.gguf")
    offered = [choice.split(" (")[0] for choice in llm.choices()]
    assert offered == ["auto", "Alpha 1B", "Beta 2B", "own.gguf"]


def test_a_listed_model_already_on_disk_is_used_without_a_download(machine):
    folder, fetched = machine
    wanted = write_gguf(folder / "Alpha-1B.gguf")
    assert llm.resolve(catalog.entries()[0].label, {}, None) == wanted
    assert fetched == []


def test_a_listed_model_in_another_folder_is_not_fetched_twice(machine, monkeypatch):
    folder, fetched = machine
    other = folder.parent / "elsewhere"
    other.mkdir()
    wanted = write_gguf(other / "Beta-2B.gguf")
    monkeypatch.setattr(paths, "gguf_roots", lambda: [str(folder), str(other)])
    assert llm.resolve("Beta 2B", {}, None) == wanted
    assert fetched == []


def test_a_listed_model_not_on_disk_is_downloaded(machine):
    folder, fetched = machine
    assert llm.resolve("Beta 2B (1.5 GB, ~3 GB VRAM)", {}, None) == str(folder / "Beta-2B.gguf")
    assert fetched == [("Beta 2B", "owner/Beta-2B-GGUF", "Beta-2B.gguf")]


def test_a_workflow_saved_with_the_file_label_still_runs(machine):
    """Once the list names a file, the file's own row is hidden; a graph saved with that row keeps working."""
    folder, fetched = machine
    wanted = write_gguf(folder / "Alpha-1B.gguf")
    saved = [label for label, path in llm.catalogue() if path == wanted][0]
    assert saved not in llm.choices()
    assert llm.known(saved) == ""
    assert llm.resolve(saved, {}, None) == wanted and fetched == []


def test_a_choice_that_names_nothing_is_refused_with_a_reason(machine):
    assert "Pick another model" in llm.known("Gone 5B (3 GB)")
    assert llm.known("auto") == "" and llm.known("Beta 2B") == ""
    with pytest.raises(FileNotFoundError, match="neither in the model list"):
        llm.resolve("Gone 5B (3 GB)", {}, None)


def test_a_list_that_does_not_parse_says_so_first_and_cannot_be_run(machine):
    os.makedirs(os.path.dirname(catalog.live_file()), exist_ok=True)
    with open(catalog.live_file(), "wb") as handle:
        handle.write(b"not json")
    catalog.forget()
    choices = llm.choices()
    assert choices[0] == "auto" and choices[1].startswith(llm.PROBLEM_PREFIX)
    assert "not valid JSON" in llm.known(choices[1])
    with pytest.raises(FileNotFoundError, match="only reports the problem"):
        llm.resolve(choices[1], {}, None)


def test_a_model_copied_in_while_running_appears_without_a_restart(machine, monkeypatch):
    folder, _fetched = machine
    assert [label.split(" (")[0] for label, _path in llm.catalogue()] == []
    write_gguf(folder / "copied-in.gguf")
    monkeypatch.setattr(llm, "SCAN_SECONDS", 0.0)
    assert [label.split(" (")[0] for label, _path in llm.catalogue()] == ["copied-in.gguf"]


def test_write_song_refuses_only_a_model_that_is_gone(machine):
    from yue2_comfy.nodes import YuE2WriteSong

    assert YuE2WriteSong.VALIDATE_INPUTS(model="auto") is True
    assert YuE2WriteSong.VALIDATE_INPUTS(model="Beta 2B (7 GB)") is True
    assert "Pick another model" in YuE2WriteSong.VALIDATE_INPUTS(model="Gone 5B")


def test_transcribe_checks_the_model_only_when_it_recognises_words(machine):
    from yue2_comfy.transcribe import YuE2Transcribe

    assert YuE2Transcribe.VALIDATE_INPUTS(model="Gone 5B", lyrics_auto_recognition=False) is True
    assert "Pick another model" in YuE2Transcribe.VALIDATE_INPUTS(model="Gone 5B", lyrics_auto_recognition=True)


def test_a_listed_download_with_downloading_off_gives_the_link(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "writer_root", lambda: str(tmp_path))
    with pytest.raises(FileNotFoundError) as error:
        download.fetch_listed("Beta 2B", "owner/Beta-GGUF", "sub/Beta.gguf", {"download": "off"})
    message = str(error.value)
    assert "owner/Beta-GGUF/resolve/main/sub/Beta.gguf" in message
    assert os.path.join(str(tmp_path), "Beta.gguf") in message


def test_a_misspelt_listed_file_is_named_before_anything_is_fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "writer_root", lambda: str(tmp_path))
    monkeypatch.setattr(download, "list_repo_files", lambda repo, revision="main", token=None: {"Beta.gguf": 9})
    monkeypatch.setattr(download, "fetch", lambda *args, **kwargs: pytest.fail("fetched a file that is not there"))
    with pytest.raises(download.DownloadError, match="has no file 'Beat.gguf'"):
        download.fetch_listed("Beta 2B", "owner/Beta-GGUF", "Beat.gguf", {})
