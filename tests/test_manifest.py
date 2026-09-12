"""Checksum checks built from real files, small enough to hash in a test.

The schema is the one m-a-p ships: read off the three manifests on this machine
on 2026-09-13, all of them ``{"schema": 1, "files": {name: {bytes, sha256}}}``
with a single ``model.safetensors`` entry each.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from yue2_comfy import manifest
from yue2_comfy.constants import MANIFEST_NAME

BODY = b"not really seven gigabytes, but it hashes the same way"


def write_model(folder, name="model.safetensors", body=BODY):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(body)
    return path


def write_manifest(folder, files, schema=1):
    document = {"schema": schema, "files": files}
    (folder / MANIFEST_NAME).write_text(json.dumps(document), encoding="utf-8")


def entry(body=BODY):
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def test_a_good_folder_passes(tmp_path):
    write_model(tmp_path)
    write_manifest(tmp_path, {"model.safetensors": entry()})
    assert manifest.verify(str(tmp_path)) == []


def test_a_file_that_was_rewritten_in_flight_is_caught(tmp_path):
    """The whole point: right length, wrong bytes, and size cannot see it."""
    other = b"x" * len(BODY)
    write_model(tmp_path, body=other)
    write_manifest(tmp_path, {"model.safetensors": entry()})
    assert manifest.verify(str(tmp_path)) == ["model.safetensors"]


def test_a_file_of_the_wrong_length_is_caught_without_hashing_it(tmp_path):
    write_model(tmp_path, body=BODY + b"tail")
    write_manifest(tmp_path, {"model.safetensors": entry()})
    assert manifest.verify(str(tmp_path)) == ["model.safetensors"]


def test_a_folder_with_no_manifest_is_not_a_failure(tmp_path):
    """Comfy-Org's repack ships none, and inventing a verdict would be worse."""
    write_model(tmp_path)
    assert manifest.read(str(tmp_path)) == {}
    assert manifest.verify(str(tmp_path)) == []


def test_an_unknown_schema_is_declined_rather_than_guessed_at(tmp_path):
    write_model(tmp_path)
    write_manifest(tmp_path, {"model.safetensors": entry()}, schema=2)
    assert manifest.read(str(tmp_path)) == {}
    assert manifest.verify(str(tmp_path)) == []


def test_a_manifest_that_is_not_json_is_declined(tmp_path):
    write_model(tmp_path)
    (tmp_path / MANIFEST_NAME).write_bytes(b"<html>404</html>")
    assert manifest.read(str(tmp_path)) == {}


def test_a_file_the_manifest_names_but_the_folder_lacks_is_not_our_question(tmp_path):
    """Whether it should be there is the loader's call, answered in its words."""
    write_manifest(tmp_path, {"model.safetensors": entry()})
    assert manifest.verify(str(tmp_path)) == []


def test_an_entry_that_escapes_the_folder_is_ignored(tmp_path):
    write_model(tmp_path)
    write_manifest(tmp_path, {"../outside.safetensors": entry(),
                              "model.safetensors": entry()})
    assert list(manifest.read(str(tmp_path))) == ["model.safetensors"]


def test_a_checksum_that_is_not_a_checksum_is_ignored(tmp_path):
    write_model(tmp_path)
    write_manifest(tmp_path, {"model.safetensors": {"bytes": len(BODY),
                                                    "sha256": "nonsense"}})
    assert manifest.read(str(tmp_path)) == {}


def test_check_refuses_and_says_what_to_delete(tmp_path):
    write_model(tmp_path, body=b"y" * len(BODY))
    write_manifest(tmp_path, {"model.safetensors": entry()})
    with pytest.raises(manifest.Corrupt) as error:
        manifest.check(str(tmp_path))
    message = str(error.value)
    assert "model.safetensors" in message
    assert "Delete it" in message


def test_hashing_can_be_cancelled_between_chunks(tmp_path):
    """Seven gigabytes is long enough that Cancel has to reach in here too."""
    path = write_model(tmp_path, body=b"z" * (manifest.READ_CHUNK * 3))
    with pytest.raises(InterruptedError):
        manifest.digest(str(path), cancelled=lambda: True)


def test_the_digest_is_the_one_hashlib_gives(tmp_path):
    path = write_model(tmp_path)
    assert manifest.digest(str(path)) == hashlib.sha256(BODY).hexdigest()


def test_progress_is_reported_as_it_reads(tmp_path):
    seen = []

    class Bar:
        def text(self, message, force=False):
            seen.append(("text", message))

        def ratio(self, fraction, text=None):
            seen.append(("ratio", fraction))

    write_model(tmp_path)
    write_manifest(tmp_path, {"model.safetensors": entry()})
    assert manifest.verify(str(tmp_path), Bar()) == []
    assert ("text", "Checking model.safetensors") in seen
    assert any(kind == "ratio" for kind, _value in seen)
    assert all(0.0 <= value <= 1.0 for kind, value in seen if kind == "ratio")
