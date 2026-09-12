"""Reading a store Ollama filled, without asking anyone to download it twice.

The store is built here rather than found, because what is worth testing is the
layout -- a manifest naming a blob by digest -- and that is three files.
"""

import hashlib
import json
import os

import pytest

from test_llm import write_gguf
from yue2_comfy import llm, ollama, paths


@pytest.fixture(autouse=True)
def forget_models():
    llm._HEADERS.clear()
    llm._CATALOGUE.update({"roots": None, "entries": []})
    yield
    llm._HEADERS.clear()
    llm._CATALOGUE.update({"roots": None, "entries": []})


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "store"
    (root / "blobs").mkdir(parents=True)
    (root / "manifests").mkdir(parents=True)
    monkeypatch.setenv(ollama.STORE_ENV, str(root))
    monkeypatch.delenv(ollama.OLLAMA_ENV, raising=False)
    monkeypatch.setattr(ollama, "DEFAULT_ROOTS", ())
    return root


def pull(root, name, tag="latest", registry="registry.ollama.ai", namespace="library",
         kind="model", chat=True, layers=None):
    """Write one blob and the manifest that names it, the way Ollama would."""
    digest = hashlib.sha256((name + tag).encode()).hexdigest()
    blob = root / "blobs" / ("sha256-" + digest)
    write_gguf(blob, kind=kind, chat=chat)
    manifest = root / "manifests" / registry / namespace / name / tag
    manifest.parent.mkdir(parents=True, exist_ok=True)
    body = {"layers": layers if layers is not None else [
        {"mediaType": ollama.MODEL_LAYER, "digest": "sha256:" + digest},
        {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:" + "0" * 64},
    ]}
    manifest.write_text(json.dumps(body), encoding="utf-8")
    return str(blob)


def test_a_pulled_model_is_found_by_its_ollama_name(store):
    blob = pull(store, "qwen3", "8b")
    assert ollama.entries() == [("qwen3:8b", blob)]


def test_the_library_prefix_is_dropped_but_another_registry_is_not(store):
    pull(store, "qwen3", "8b")
    pull(store, "mine", "v1", registry="hf.co", namespace="someone")
    assert sorted(name for name, _path in ollama.entries()) == [
        "hf.co/someone/mine:v1", "qwen3:8b"]


def test_two_tags_of_one_blob_are_one_entry(store):
    """qwen3:8b and qwen3:latest usually point at the same file."""
    digest = hashlib.sha256(b"shared").hexdigest()
    blob = store / "blobs" / ("sha256-" + digest)
    write_gguf(blob)
    for tag in ("8b", "latest"):
        manifest = store / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / tag
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"layers": [
            {"mediaType": ollama.MODEL_LAYER, "digest": "sha256:" + digest}]}),
            encoding="utf-8")
    assert [name for name, _path in ollama.entries()] == ["qwen3:8b"]


def test_a_manifest_with_no_model_layer_is_skipped(store):
    pull(store, "embed", "v1", layers=[
        {"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:" + "1" * 64}])
    assert ollama.entries() == []


def test_a_digest_that_climbs_out_of_the_store_is_refused(store):
    """A manifest is a file like any other, so its digest is checked, not trusted."""
    assert ollama._blob(str(store), "sha256:../../../../etc/passwd") == ""
    assert ollama._blob(str(store), "sha256:" + "z" * 64) == ""


def test_a_manifest_that_is_not_json_is_skipped(store):
    manifest = store / "manifests" / "registry.ollama.ai" / "library" / "broken" / "latest"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("not json at all", encoding="utf-8")
    assert ollama.entries() == []


def test_a_folder_without_manifests_is_not_a_store(tmp_path, monkeypatch):
    monkeypatch.setenv(ollama.STORE_ENV, str(tmp_path))
    monkeypatch.delenv(ollama.OLLAMA_ENV, raising=False)
    monkeypatch.setattr(ollama, "DEFAULT_ROOTS", ())
    assert ollama.roots() == []


def test_a_network_store_is_only_scanned_when_it_was_asked_for(tmp_path, monkeypatch):
    """Reaching into WSL starts a stopped distribution; a dropdown must not."""
    monkeypatch.setattr(paths, "is_network_path", lambda value: True)
    monkeypatch.setattr(ollama, "DEFAULT_ROOTS", ())
    root = tmp_path / "store"
    (root / "manifests").mkdir(parents=True)

    monkeypatch.setenv(ollama.OLLAMA_ENV, str(root))
    monkeypatch.delenv(ollama.STORE_ENV, raising=False)
    assert ollama.roots() == []

    monkeypatch.setenv(ollama.STORE_ENV, str(root))
    assert ollama.roots() == [str(root)]


def test_an_ollama_model_is_offered_in_the_catalogue(store, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "gguf_roots", lambda: [])
    blob = pull(store, "qwen3", "8b")
    entries = llm.catalogue()
    assert len(entries) == 1
    label, path = entries[0]
    assert path == blob
    assert label.startswith("ollama: qwen3:8b (")


def test_an_ollama_blob_that_cannot_write_is_not_offered(store, monkeypatch):
    """A store holds whatever was pulled, embedding models included."""
    monkeypatch.setattr(paths, "gguf_roots", lambda: [])
    pull(store, "nomic-embed", "v1", chat=False)
    assert llm.catalogue() == []


def test_a_store_inside_a_scanned_folder_is_still_listed_once(store, monkeypatch):
    """Ollama names blobs by digest, so a sweep for '*.gguf' walks straight past.

    Which is why the store has to be read at all rather than left to the folder
    scan, and why pointing a model folder at the blobs changes nothing.
    """
    blob = pull(store, "qwen3", "8b")
    monkeypatch.setattr(paths, "gguf_roots", lambda: [os.path.dirname(blob)])
    labels = [label for label, _path in llm.catalogue()]
    assert len(labels) == 1
    assert labels[0].startswith("ollama: qwen3:8b (")
