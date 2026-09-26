"""Finding a writer model, and refusing the files that cannot be one.

The GGUFs here are built byte by byte rather than downloaded, so the tests say
what the header reader does with an adapter, an mmproj and a truncated file
without keeping three gigabytes in the repository.

``general.type`` really is the separator: measured across seventeen files in a
working model folder it read ``model``, ``adapter`` or ``mmproj`` outright, and
only the models carried a chat template.
"""

import os
import struct

import pytest

from yue2_comfy import gguf_meta, llm, paths

STRING = 8
UINT32 = 4
PADDING = 4096


def _kv(key: str, value: str) -> bytes:
    name = key.encode("utf-8")
    text = value.encode("utf-8")
    return (struct.pack("<Q", len(name)) + name + struct.pack("<I", STRING)
            + struct.pack("<Q", len(text)) + text)


def _kv_int(key: str, value: int) -> bytes:
    name = key.encode("utf-8")
    return struct.pack("<Q", len(name)) + name + struct.pack("<I", UINT32) + struct.pack("<I", value)


def _tensor(number: int) -> bytes:
    """One tensor-table record: a name, one dimension of one element, type 0, offset 0."""
    name = "blk.{}.weight".format(number).encode("utf-8")
    return (struct.pack("<Q", len(name)) + name + struct.pack("<I", 1) + struct.pack("<Q", 1)
            + struct.pack("<I", 0) + struct.pack("<Q", 0))


def write_gguf(path, kind="model", arch="qwen35", chat=True, truncated=False, ints=None, tensors=0):
    """A GGUF with a valid header, ``tensors`` one-element tensors, and nothing else in it.

    ``ints`` are extra keys written as uint32, such as a block count or a pooling type.
    """
    pairs = [_kv("general.architecture", arch), _kv("general.type", kind)]
    pairs += [_kv_int(key, value) for key, value in (ints or {}).items()]
    if chat:
        pairs.append(_kv("tokenizer.chat_template", "{{ messages }}"))
    header = (gguf_meta.MAGIC + struct.pack("<I", 3)
              + struct.pack("<QQ", tensors, len(pairs)) + b"".join(pairs)
              + b"".join(_tensor(number) for number in range(tensors)))
    with open(path, "wb") as handle:
        handle.write(header)
        if not truncated:
            handle.write(b"\0" * PADDING)
    return str(path)


@pytest.fixture(autouse=True)
def forget_models(monkeypatch, tmp_path):
    """No cache between tests, no Ollama store on the machine running them, and an empty model list.

    The model list is test_catalog's; here the widget offers what is on disk.
    """
    from yue2_comfy import catalog, ollama

    empty = tmp_path / "empty_writers.json"
    empty.write_bytes(b'{"writers": []}')
    monkeypatch.setattr(catalog, "SEED_FILE", str(empty))
    catalog.forget()
    monkeypatch.setattr(ollama, "roots", lambda: [])
    llm._HEADERS.clear()
    llm._CATALOGUE.update({"roots": None, "entries": []})
    yield
    llm._HEADERS.clear()
    llm._CATALOGUE.update({"roots": None, "entries": []})


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "gguf_roots", lambda: [str(tmp_path)])
    return tmp_path


def labels() -> list:
    """Catalogue labels without their sizes, which most of these do not test."""
    return [name.split(" (")[0] for name, _path in llm.catalogue()]


def test_a_model_with_a_chat_template_is_runnable(tmp_path):
    assert llm.runnable(write_gguf(tmp_path / "writer.gguf"))


def test_an_adapter_is_not_offered(tmp_path):
    """A LoRA would load, answer as the base model, and look like a bad writer."""
    assert not llm.runnable(write_gguf(tmp_path / "lora.gguf", kind="adapter", chat=False))


def test_an_mmproj_is_not_offered(tmp_path):
    assert not llm.runnable(
        write_gguf(tmp_path / "mmproj.gguf", kind="mmproj", arch="clip", chat=False))


def test_a_model_without_a_chat_template_is_not_offered(tmp_path):
    assert not llm.runnable(write_gguf(tmp_path / "bare.gguf", chat=False))


def test_a_half_downloaded_file_is_not_offered(tmp_path):
    assert not llm.runnable(write_gguf(tmp_path / "cut.gguf", truncated=True))


def test_something_that_is_not_a_gguf_is_not_offered(tmp_path):
    path = tmp_path / "notes.gguf"
    path.write_bytes(b"this is not a model")
    assert not llm.runnable(str(path))


def test_a_missing_file_is_not_offered(tmp_path):
    assert not llm.runnable(str(tmp_path / "never-existed.gguf"))


def test_the_catalogue_lists_only_what_can_write(folder):
    write_gguf(folder / "writer.gguf")
    write_gguf(folder / "lora.gguf", kind="adapter", chat=False)
    write_gguf(folder / "mmproj.gguf", kind="mmproj", arch="clip", chat=False)
    assert labels() == ["writer.gguf"]


def test_the_catalogue_looks_inside_subfolders(folder):
    (folder / "nested").mkdir()
    write_gguf(folder / "nested" / "writer.gguf")
    assert labels() == ["writer.gguf"]


def test_two_files_of_the_same_name_stay_tellable_apart(folder):
    for parent in ("q4", "q8"):
        (folder / parent).mkdir()
        write_gguf(folder / parent / "same.gguf")
    assert sorted(labels()) == ["q4/same.gguf", "q8/same.gguf"]


def test_the_search_is_bounded_in_depth(folder):
    """`checkpoints` is in the search now, and it is a large tree."""
    deep = folder / "one" / "two" / "three"
    deep.mkdir(parents=True)
    write_gguf(deep / "buried.gguf")
    write_gguf(folder / "one" / "shallow.gguf")
    assert labels() == ["shallow.gguf"]


def test_only_the_first_shard_of_a_split_model_is_offered(folder):
    """Measured: llama.cpp loads the siblings itself, given the first shard.

    The later shards carry no chat template, which is what keeps them out --
    offering them would hand somebody half a model.
    """
    write_gguf(folder / "model-00001-of-00002.gguf")
    write_gguf(folder / "model-00002-of-00002.gguf", chat=False)
    assert labels() == ["model-00001-of-00002.gguf"]


def test_an_embedding_model_is_not_offered(tmp_path):
    """A whole model with a chat template, found on a real disk; only its pooling type says what it is."""
    path = write_gguf(tmp_path / "qwen3-embed.gguf", arch="qwen3", ints={"qwen3.pooling_type": 3})
    assert "embedding" in llm.unfit(path)
    assert llm.runnable(write_gguf(tmp_path / "chat.gguf", arch="qwen3", ints={"qwen3.pooling_type": 0}))


def test_a_draft_head_is_not_offered_and_a_whole_model_with_one_inside_is(tmp_path):
    """Measured 2026-09-26: the FastMTP file holds 19 tensors for 65 layers, its model 866 for the same 65."""
    head = write_gguf(tmp_path / "FastMTP.gguf", ints={"qwen35.block_count": 65, "qwen35.nextn_predict_layers": 1},
                      tensors=19)
    assert "only part of a model (19 tensors for 65 layers)" in llm.unfit(head)
    whole = write_gguf(tmp_path / "whole.gguf", ints={"qwen35.block_count": 4, "qwen35.nextn_predict_layers": 1},
                       tensors=12)
    assert llm.unfit(whole) == ""


def test_a_split_model_whose_first_part_holds_no_tensors_is_offered(folder):
    """Three models in a real folder were never offered: their first part is the header alone.

    It ends where its header ends, which the truncation check read as a cut
    download. The later parts are named as parts, not as models without a template.
    """
    first = write_gguf(folder / "big-00001-of-00003.gguf", truncated=True,
                       ints={"split.count": 3, "split.tensors.count": 900, "qwen35.block_count": 64})
    for number in (2, 3):
        write_gguf(folder / "big-0000{}-of-00003.gguf".format(number), chat=False, tensors=4,
                   ints={"split.count": 3, "split.no": number - 1})
    assert llm.unfit(first) == ""
    assert llm.unfit(str(folder / "big-00002-of-00003.gguf")).startswith("part 2 of 3 of a split model")
    assert labels() == ["big-00001-of-00003.gguf"]


def test_a_split_model_is_offered_with_the_size_of_all_its_parts(folder):
    parts = [write_gguf(folder / "big-0000{}-of-00003.gguf".format(number)) for number in (1, 2, 3)]
    assert llm.size_of(parts[0]) == sum(os.path.getsize(part) for part in parts)
    assert llm.size_of(parts[1]) == os.path.getsize(parts[1])
    assert llm.size_of(str(folder / "gone-00001-of-00002.gguf")) == 0


def test_a_cut_download_is_still_refused_when_it_holds_tensors(tmp_path):
    path = write_gguf(tmp_path / "cut.gguf", truncated=True, tensors=3, ints={"split.count": 2})
    assert "not a GGUF this can read" in llm.unfit(path)


def test_the_parts_of_a_split_model_are_named_from_its_first(tmp_path):
    assert gguf_meta.split_names("Q4_K_M/big-00001-of-00003.gguf") == [
        "Q4_K_M/big-00001-of-00003.gguf", "Q4_K_M/big-00002-of-00003.gguf", "Q4_K_M/big-00003-of-00003.gguf"]
    assert gguf_meta.split_names("C:\\m\\big-00001-of-00002.GGUF")[1] == "C:\\m\\big-00002-of-00002.GGUF"
    assert gguf_meta.split_names("big-00002-of-00003.gguf") == ["big-00002-of-00003.gguf"]
    assert gguf_meta.split_names("plain.gguf") == ["plain.gguf"]


def test_the_header_tells_how_many_tensors_the_file_holds(tmp_path):
    path = write_gguf(tmp_path / "t.gguf", tensors=5)
    assert gguf_meta.read_keys(path, (gguf_meta.TENSOR_COUNT, "general.type")) == {
        gguf_meta.TENSOR_COUNT: 5, "general.type": "model"}


def test_a_cache_copy_is_named_by_repository_not_by_commit(tmp_path, monkeypatch):
    """A snapshot directory is a commit hash, which is no use as a label."""
    snapshot = tmp_path / "models--Qwen--Qwen3-VL-8B-Instruct-GGUF" / "snapshots" / "f982a075"
    snapshot.mkdir(parents=True)
    write_gguf(snapshot / "same.gguf")
    (tmp_path / "LLM").mkdir()
    write_gguf(tmp_path / "LLM" / "same.gguf")
    monkeypatch.setattr(paths, "gguf_roots", lambda: [str(tmp_path / "LLM"), str(snapshot)])
    assert sorted(labels()) == [
        "LLM/same.gguf", "Qwen/Qwen3-VL-8B-Instruct-GGUF/same.gguf"]


def test_choices_start_with_auto(folder):
    write_gguf(folder / "writer.gguf")
    assert [choice.split(" (")[0] for choice in llm.choices()] == ["auto", "writer.gguf"]


def test_the_label_says_how_big_the_file_is(folder):
    """A dropdown of names asks somebody with an 8 GB card to guess."""
    write_gguf(folder / "writer.gguf")
    assert llm.catalogue()[0][0].endswith(" KB)")


def test_a_workflow_saved_against_another_quant_of_the_same_name_still_runs(folder):
    """The size is cosmetic, so it must not be what a saved graph hangs on."""
    wanted = write_gguf(folder / "writer.gguf")
    assert llm.resolve("writer.gguf (7.77 GB)", {}, None) == wanted


def test_auto_prefers_the_default_model(folder):
    from yue2_comfy.constants import WRITER_NAME

    write_gguf(folder / "aaa-first-alphabetically.gguf")
    wanted = write_gguf(folder / WRITER_NAME)
    assert llm.resolve("auto", {}, None) == wanted


def test_auto_takes_what_is_there_when_the_default_is_absent(folder):
    only = write_gguf(folder / "something-else.gguf")
    assert llm.resolve("auto", {}, None) == only


def test_a_named_model_resolves_to_its_path(folder):
    wanted = write_gguf(folder / "writer.gguf")
    write_gguf(folder / "other.gguf")
    assert llm.resolve("writer.gguf", {}, None) == wanted


def test_a_deleted_model_says_what_is_there_instead(folder):
    write_gguf(folder / "still-here.gguf")
    with pytest.raises(FileNotFoundError) as error:
        llm.resolve("deleted.gguf", {}, None)
    message = str(error.value)
    assert "deleted.gguf" in message
    assert "still-here.gguf" in message
    assert "auto" in message


def test_auto_with_nothing_on_disk_and_downloading_off_gives_the_link(folder):
    with pytest.raises(FileNotFoundError) as error:
        llm.resolve("auto", {"download": "off"}, None)
    message = str(error.value)
    assert "huggingface.co/unsloth/Qwen3.5-4B-GGUF" in message
    assert "Qwen3.5-4B-Q4_K_M.gguf" in message


def test_the_install_hint_names_this_interpreter():
    import sys

    hint = llm.install_hint()
    assert "llama-cpp-python" in hint
    assert os.path.basename(sys.executable or "python") in hint


def test_headers_are_read_once_per_file(folder, monkeypatch):
    """A folder of large quants should cost a stat each after the first pass."""
    path = write_gguf(folder / "writer.gguf")
    reads = []
    real = gguf_meta.keys
    monkeypatch.setattr(gguf_meta, "keys",
                        lambda *args, **kwargs: reads.append(args[0]) or real(*args, **kwargs))
    assert llm.runnable(path)
    assert llm.runnable(path)
    assert reads == [path]
