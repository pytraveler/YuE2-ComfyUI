"""The 'YuE2 LoRA' node and the nodes that sing with its adapters, without a model."""

from __future__ import annotations

import contextlib
import json
import os

import pytest

import lora_files as lf
from yue2_comfy import constants, generate, nodes, routes, songs, staged
from yue2_comfy.lora import catalogue
from yue2_comfy.lora import node as lora_node


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """One LoRA folder: an AR file, a NAR file with a trigger word, and one that cannot be used."""
    root = tmp_path / "loras"
    (root / "rock").mkdir(parents=True)
    lf.write(root / "rock" / "ar.safetensors", lf.map_layer(0))
    lf.write(root / "jpop.safetensors", lf.comfy_layer(0), {"trigger_word": "jpstyle26"})
    hum = lf.map_layer(0, half="nar", prefix="")
    hum["hum_proj.0.weight"] = ("F32", (2048, 64))
    lf.write(root / "hum.safetensors", hum)
    lf.write(root / "inst.safetensors", lf.map_layer(1), {"intended_cot": "full"})
    monkeypatch.setattr(catalogue, "roots", lambda: [str(root)])
    return root


def rows(*items):
    return json.dumps(list(items))


def test_the_node_is_one_text_widget_one_socket_and_one_output():
    spec = lora_node.YuE2LoRA.INPUT_TYPES()
    assert list(spec["required"]) == ["loras"]
    assert spec["required"]["loras"][0] == "STRING" and spec["required"]["loras"][1]["default"] == "[]"
    assert list(spec["optional"]) == ["lora"] and spec["optional"]["lora"][0] == constants.LORA_TYPE
    assert lora_node.YuE2LoRA.RETURN_TYPES == (constants.LORA_TYPE,)
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["YuE2LoRA"] == "YuE2 LoRA"


def test_rows_are_read_with_their_defaults():
    found = lora_node.rows(rows({"name": "a"}, {"name": "b", "ar": 0.5, "nar": -1, "on": False}))
    assert found == [{"name": "a", "ar": 1.0, "nar": 1.0, "on": True},
                     {"name": "b", "ar": 0.5, "nar": -1.0, "on": False}]
    assert lora_node.rows("") == [] and lora_node.rows("[]") == []


@pytest.mark.parametrize("text,word", [
    ("{not json", "JSON"), ('{"name": "a"}', "list"), ('["a"]', "not a row"),
    ('[{"name": "a", "ar": "loud"}]', "not a number"), ('[{"name": "a", "nar": 11}]', "from -10"),
])
def test_rows_that_cannot_be_read_say_why(text, word):
    with pytest.raises(ValueError, match=word):
        lora_node.rows(text)


def test_picked_rows_are_found_hashed_and_keep_only_the_halves_their_file_has(folder):
    chosen = lora_node.picked(rows({"name": "rock\\ar.safetensors", "ar": 0.5, "nar": 0.7},
                                   {"name": "jpop.safetensors", "ar": 0.3, "nar": 0.9},
                                   {"name": "jpop.safetensors", "on": False},
                                   {"name": "jpop.safetensors", "ar": 0, "nar": 0},
                                   {"name": ""}))
    assert [(item["name"], item["ar"], item["nar"]) for item in chosen] == [
        ("rock/ar.safetensors", 0.5, 0.0), ("jpop.safetensors", 0.0, 0.9)]
    assert chosen[0]["path"] == str(folder / "rock" / "ar.safetensors")
    assert chosen[0]["sha256"] == catalogue.identity(chosen[0]["path"]) and len(chosen[0]["sha256"]) == 64


def test_a_file_that_is_gone_or_cannot_be_folded_is_refused(folder):
    with pytest.raises(FileNotFoundError, match="gone.safetensors"):
        lora_node.picked(rows({"name": "gone.safetensors"}))
    with pytest.raises(ValueError, match="hum_proj"):
        lora_node.picked(rows({"name": "hum.safetensors"}))
    with pytest.raises(ValueError, match="hum_proj"):
        lora_node.YuE2LoRA().pick(rows({"name": "hum.safetensors"}))


def test_a_chained_node_adds_its_rows_after_the_ones_it_was_handed(folder):
    first, = lora_node.YuE2LoRA().pick(rows({"name": "jpop.safetensors"}))
    both, = lora_node.YuE2LoRA().pick(rows({"name": "rock/ar.safetensors"}), lora=first)
    assert [item["name"] for item in both] == ["jpop.safetensors", "rock/ar.safetensors"]
    assert json.loads(json.dumps(both)) == both


def test_a_file_replaced_under_its_name_makes_the_node_run_again(folder):
    text = rows({"name": "jpop.safetensors"})
    before = lora_node.YuE2LoRA.IS_CHANGED(text)
    assert lora_node.YuE2LoRA.IS_CHANGED(text) == before
    target = folder / "jpop.safetensors"
    stat = os.stat(target)
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    assert lora_node.YuE2LoRA.IS_CHANGED(text) != before


def test_every_singing_node_takes_adapters_on_its_last_input():
    """A socket holds no widget value, and last is where a new one cannot shift an old workflow."""
    for cls in (nodes.YuE2GenerateSong, staged.YuE2Plan, staged.YuE2PlanBatch, staged.YuE2RenderPlan):
        optional = cls.INPUT_TYPES()["optional"]
        assert list(optional)[-1] == "lora", cls.__name__
        assert optional["lora"][0] == constants.LORA_TYPE


def test_the_song_node_sings_with_the_adapters_it_is_handed(folder, monkeypatch):
    chosen, = lora_node.YuE2LoRA().pick(rows({"name": "rock/ar.safetensors", "ar": 0.4}))
    seen, kept = [], []

    @contextlib.contextmanager
    def session(settings, unique_id, progress):
        seen.append(dict(settings))
        yield object()

    monkeypatch.setattr(nodes, "session", session)
    monkeypatch.setattr(generate, "run", lambda *args, **kwargs: (
        "waveform", "X:1", "X:1", {"seconds_of_audio": 1.0, "total_seconds": 1.0,
                                   "semantic": {"output_tokens": 1, "output_tps": 1.0,
                                                "execution": "x", "attention": "sdpa"}}, "performed"))
    monkeypatch.setattr(songs, "keep", lambda *args: kept.append(args))
    nodes.YuE2GenerateSong().generate(style="s", lyrics="l", seed=1, lora=chosen)
    assert seen[0]["loras"] == chosen
    assert kept[0][5]["loras"] == chosen
    nodes.YuE2GenerateSong().generate(style="s", lyrics="l", seed=1)
    assert seen[1]["loras"] == []


def test_a_plan_carries_its_adapters_and_the_render_node_keeps_or_replaces_them(folder, monkeypatch):
    first, = lora_node.YuE2LoRA().pick(rows({"name": "rock/ar.safetensors"}))
    second, = lora_node.YuE2LoRA().pick(rows({"name": "jpop.safetensors"}))
    sung = []

    @contextlib.contextmanager
    def session(settings, unique_id, progress):
        sung.append(settings.get("loras"))
        yield object()

    class Latents:
        def cpu(self):
            return self

    monkeypatch.setattr(staged, "session", session)
    monkeypatch.setattr(generate, "write_score", lambda *args, **kwargs: ("X:1", [1], {}))
    monkeypatch.setattr(generate, "sing", lambda *args, **kwargs: (Latents(), {}, None))
    monkeypatch.setattr(generate, "decode", lambda *args, **kwargs: ("wave", {"seconds_of_audio": 1.0}))
    plan, _score = staged.YuE2Plan().plan(style="s", lyrics="l", seed=3, lora=first)["result"]
    assert plan["settings"]["loras"] == first
    assert json.loads(json.dumps(plan)) == plan
    staged.YuE2RenderPlan().render(plan)
    staged.YuE2RenderPlan().render(plan, lora=second)
    staged.YuE2RenderPlan().render(plan, lora=[])
    assert sung[1:] == [first, second, []]


def test_a_file_gone_since_it_was_chosen_is_refused_before_anything_loads(folder, monkeypatch):
    chosen, = lora_node.YuE2LoRA().pick(rows({"name": "jpop.safetensors"}))
    os.remove(folder / "jpop.safetensors")
    catalogue.forget()
    loaded = []
    monkeypatch.setattr(nodes, "session", lambda *args: loaded.append(True))
    with pytest.raises(ValueError, match="jpop.safetensors is no longer where it was chosen"):
        nodes.YuE2GenerateSong().generate(style="s", lyrics="l", seed=1, lora=chosen)
    assert loaded == []


def test_an_adapter_made_for_another_cot_is_warned_about(folder):
    chosen, = lora_node.YuE2LoRA().pick(rows({"name": "inst.safetensors"}))
    assert lora_node.notices(chosen, "full") == []
    [(level, message)] = lora_node.notices(chosen, "off")
    assert level == "warn" and "'full'" in message and "'off'" in message


def test_the_song_memory_writes_the_adapters_down_and_nothing_live():
    settings = {"cot": "full", "loras": [{"name": "a", "path": "p", "sha256": "h", "ar": 1.0, "nar": 0.5}],
                "device": object(), "rows": [{"live": object()}]}
    assert songs._plain(settings) == {"cot": "full", "loras": settings["loras"]}


def test_the_route_lists_the_files_with_what_the_rows_show(folder):
    payload, status = routes.answer_loras()
    assert status == 200 and payload["ok"]
    names = {item["name"]: item for item in payload["loras"]}
    assert set(names) == {"hum.safetensors", "inst.safetensors", "jpop.safetensors", "rock/ar.safetensors"}
    assert names["jpop.safetensors"]["triggers"] == ["jpstyle26"]
    assert names["hum.safetensors"]["problem"]
    json.dumps(payload)
