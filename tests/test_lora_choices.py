"""A run's adapters as the settings carry them: sorted for folding, zero rows dropped, comparable between runs."""

from __future__ import annotations

import os

import lora_files as lf
from yue2_comfy.lora import choices


def row(path, name, sha, ar=1.0, nar=1.0):
    return {"name": name, "path": str(path), "sha256": sha, "ar": ar, "nar": nar}


def test_rows_fold_in_hash_order_whatever_order_they_came_in(tmp_path):
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    lf.write(a, lf.map_layer(0))
    lf.write(b, lf.map_layer(1))
    one = choices.from_settings([row(b, "b", "22"), row(a, "a", "11")])
    two = choices.from_settings([row(a, "a", "11"), row(b, "b", "22")])
    assert one == two and [choice.name for choice in one] == ["a", "b"]


def test_a_row_at_zero_on_both_halves_or_without_a_file_is_left_out(tmp_path):
    a = tmp_path / "a.safetensors"
    lf.write(a, lf.map_layer(0))
    found = choices.from_settings([row(a, "a", "1", ar=0.0, nar=0.0), {"name": "x", "ar": 1.0},
                                   "not a row", row(a, "b", "2", ar=float("nan"), nar=float("inf")),
                                   row(a, "c", "3", ar=-0.5, nar=0.0)])
    assert [(choice.name, choice.ar, choice.nar) for choice in found] == [("c", -0.5, 0.0)]


def test_a_half_is_touched_only_by_files_that_change_it(tmp_path):
    ar, nar = tmp_path / "ar.safetensors", tmp_path / "nar.safetensors"
    lf.write(ar, lf.map_layer(0))
    lf.write(nar, lf.comfy_layer(0))
    picked = choices.from_settings([row(ar, "ar", "1"), row(nar, "nar", "2")])
    assert [choice.name for choice in choices.touching(picked, "ar")] == ["ar"]
    assert [choice.name for choice in choices.touching(picked, "nar")] == ["nar"]
    assert choices.halves(picked) == {"ar", "nar"}
    assert choices.halves(choices.from_settings([row(ar, "ar", "1", ar=0.0)])) == set()


def test_the_signature_moves_with_the_strength_and_with_the_file(tmp_path):
    path = tmp_path / "ar.safetensors"
    lf.write(path, lf.map_layer(0))
    first = choices.signature(choices.from_settings([row(path, "ar", "1", ar=0.5)]), "ar")
    assert first == choices.signature(choices.from_settings([row(path, "ar", "1", ar=0.5, nar=0.9)]), "ar")
    assert first != choices.signature(choices.from_settings([row(path, "ar", "1", ar=0.6)]), "ar")
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    assert first != choices.signature(choices.from_settings([row(path, "ar", "1", ar=0.5)]), "ar")
    assert choices.signature((), "ar") == ()
