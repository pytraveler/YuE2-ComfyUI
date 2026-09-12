"""The Comfy-Org single file, rebuilt into the layout our classes expect.

The real proof of this conversion is not here: it is that all 628 language
model tensors and all 435 VAE tensors come back bit for bit against the
released checkpoints, and that a song generated from the repack has the same
waveform hash as one generated from the released files. That needs 8 GB of
weights and a GPU. What these tests hold is the shape of the mapping, so that
an edit which breaks it fails in a tenth of a second rather than in a smoke
test somebody has to remember to run.
"""

import json
import os
import struct

import pytest

from yue2_comfy import discovery, paths, repack


class Rows(list):
    """A stand-in for a weight matrix: rows, a shape, and enough of an equals.

    The test interpreter has neither torch nor numpy, and the conversion only
    ever slices along the first dimension and compares two tensors, so this is
    the whole of what repack.py asks a tensor to do.
    """

    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)

    def __getitem__(self, item):
        value = list.__getitem__(self, item)
        return Rows(value) if isinstance(item, slice) else value

    def __eq__(self, other):
        return _Verdict(list(self) == list(other))

    def __ne__(self, other):
        return _Verdict(list(self) != list(other))

    __hash__ = None


class _Verdict:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self.value


def matrix(rows, width=4, fill=None):
    return Rows([[fill if fill is not None else index] * width for index in range(rows)])


def ar_tree(prefix):
    """One layer of a repack tree: fused projections and the norms beside them."""
    return {
        prefix + "layers.0.self_attn.qkv_proj.weight": matrix(8),
        prefix + "layers.0.self_attn.o_proj.weight": matrix(4),
        prefix + "layers.0.self_attn.q_norm.weight": matrix(1),
        prefix + "layers.0.self_attn.k_norm.weight": matrix(1),
        prefix + "layers.0.mlp.gate_up_proj.weight": matrix(6),
        prefix + "layers.0.mlp.down_proj.weight": matrix(3),
        prefix + "layers.0.input_layernorm.weight": matrix(1),
        prefix + "layers.0.post_attention_layernorm.weight": matrix(1),
    }


def repack_state():
    state = {}
    state.update(ar_tree(repack.AR_PREFIX))
    state.update(ar_tree(repack.NAR_LAYER_PREFIX))
    state[repack.AR_PREFIX + "norm.weight"] = matrix(1, fill=7)
    state[repack.NAR_LAYER_PREFIX + "norm.weight"] = matrix(1, fill=7)
    state[repack.AR_PREFIX + "embed_tokens.weight"] = matrix(2)
    state[repack.AR_PREFIX + "lm_head.weight"] = matrix(2)
    state[repack.NAR_PREFIX + "llm2vae.weight"] = matrix(2)
    state[repack.NAR_PREFIX + "latent_pos_embed.pe"] = matrix(2)
    state["vae.decoder.layers.0.weight_v"] = matrix(2)
    return state


def write_repack(path, vocab):
    """A safetensors file carrying only the embedded tokenizer tensor."""
    blob = json.dumps({"model": {"vocab": vocab}}).encode("utf-8")
    header = {repack.TOKENIZER_KEY: {"dtype": "U8", "shape": [len(blob)],
                                     "data_offsets": [0, len(blob)]}}
    body = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(body)))
        handle.write(body)
        handle.write(blob)
    return str(path)


def write_marker_file(path, quantized=False):
    """A safetensors header with the keys that identify a repack, and no data.

    Enough for anything that reads headers, which is everything in discovery.
    The tensors are never mapped, so the offsets do not have to lead anywhere.
    """
    entry = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
    header = {
        "vae.decoder.layers.0.weight": dict(entry),
        repack.AR_PREFIX + "layers.0.mlp.down_proj.weight": dict(entry),
        repack.NAR_PREFIX + "model.layers.0.mlp.down_proj.weight": dict(entry),
        repack.TOKENIZER_KEY: dict(entry),
    }
    if quantized:
        header[repack.AR_PREFIX + "layers.0.mlp.down_proj.comfy_quant"] = dict(entry)
        header["__metadata__"] = {"quantization": "convrot_int8"}
    body = json.dumps(header).encode("utf-8")
    with open(str(path), "wb") as handle:
        handle.write(struct.pack("<Q", len(body)))
        handle.write(body)
        handle.write(b"\0\0\0\0")
    return str(path)


@pytest.fixture(autouse=True)
def forget_identifications():
    """The identification cache is keyed on path, and tmp_path gets reused."""
    discovery._identified.clear()
    yield
    discovery._identified.clear()


def test_a_repack_needs_all_three_trees_and_the_vocabulary():
    """Any one of them alone is some other checkpoint that happens to share a prefix."""
    full = {"vae.a": {}, "text_encoders.model.a": {}, "model.diffusion_model.a": {},
            repack.TOKENIZER_KEY: {}}
    assert repack.is_repack(full)
    for dropped in ("vae.a", "text_encoders.model.a", "model.diffusion_model.a",
                    repack.TOKENIZER_KEY):
        partial = {key: value for key, value in full.items() if key != dropped}
        assert not repack.is_repack(partial), dropped


def test_the_fused_projections_are_cut_where_o_proj_says():
    """q is as wide as the attention output; what is left is k and v in halves."""
    built = repack.lm_state(repack_state())

    query = built["model.layers.0.self_attn.q_proj.weight"]
    keys = built["model.layers.0.self_attn.k_proj.weight"]
    values = built["model.layers.0.self_attn.v_proj.weight"]
    assert query.shape[0] == 4
    assert keys.shape[0] == 2
    assert values.shape[0] == 2
    assert list(query) + list(keys) + list(values) == list(matrix(8))

    gate = built["model.layers.0.mlp.gate_proj.weight"]
    up = built["model.layers.0.mlp.up_proj.weight"]
    assert gate.shape[0] == up.shape[0] == 3
    assert list(gate) + list(up) == list(matrix(6))


def test_the_nar_tree_regains_the_names_upstream_uses():
    built = repack.lm_state(repack_state())
    for key in ("model.layers.0.nar_self_attn.q_proj.weight",
                "model.layers.0.nar_self_attn.o_proj.weight",
                "model.layers.0.nar_mlp.gate_proj.weight",
                "model.layers.0.nar_input_layernorm.weight",
                "model.layers.0.nar_pre_mlp_layernorm.weight"):
        assert key in built, key
    assert "model.layers.0.self_attn.q_proj.weight" in built
    assert "model.layers.0.input_layernorm.weight" in built


def test_the_top_level_pieces_land_where_our_checkpoint_keeps_them():
    built = repack.lm_state(repack_state())
    assert "lm_head.weight" in built
    assert "model.embed_tokens.weight" in built
    assert "model.norm.weight" in built
    assert "llm2vae.weight" in built
    assert "latent_pos_embed.pe" in built
    assert not any(key.startswith("model.diffusion_model") for key in built)
    assert not any(key.startswith("text_encoders") for key in built)


def test_two_final_norms_that_disagree_are_refused():
    """They are one tensor stored twice; a difference means a different model."""
    state = repack_state()
    state[repack.NAR_LAYER_PREFIX + "norm.weight"] = matrix(1, fill=9)
    with pytest.raises(ValueError) as error:
        repack.lm_state(state)
    assert "final norms differ" in str(error.value)


def test_a_qkv_that_cannot_be_split_evenly_is_refused():
    state = repack_state()
    state[repack.AR_PREFIX + "layers.0.self_attn.qkv_proj.weight"] = matrix(7)
    with pytest.raises(ValueError) as error:
        repack.lm_state(state)
    assert "cannot be split" in str(error.value)


def test_the_vae_block_only_loses_its_prefix():
    built = repack.vae_state(repack_state())
    assert list(built) == ["decoder.layers.0.weight_v"]


def test_the_vocabulary_comes_back_in_the_released_format(tmp_path):
    """Byte-level characters stand in for bytes; undoing that gives tiktoken rows."""
    vocab = {"!": 0, "A": 1, chr(0x0120): 2}
    path = write_repack(tmp_path / "repack.safetensors", vocab)

    produced = repack.tiktoken_lines(path)

    rows = [line.split() for line in produced.strip().split(b"\n")]
    assert [int(rank) for _, rank in rows] == [0, 1, 2]
    import base64
    assert [base64.b64decode(token) for token, _ in rows] == [b"!", b"A", b" "]


def test_an_unmappable_vocabulary_says_so(tmp_path):
    path = write_repack(tmp_path / "repack.safetensors", {chr(0x2603): 0})
    with pytest.raises(ValueError) as error:
        repack.tiktoken_lines(path)
    assert "outside the byte-level alphabet" in str(error.value)


def test_the_vocabulary_is_written_once_and_reused(tmp_path):
    path = write_repack(tmp_path / "repack.safetensors", {"!": 0})
    cache = str(tmp_path / "cache")

    first = repack.merges_beside(path, cache)
    stamp = os.stat(first).st_mtime_ns
    second = repack.merges_beside(path, cache)

    assert first == second
    assert os.stat(second).st_mtime_ns == stamp
    assert not [name for name in os.listdir(cache) if name.endswith(".part")]


def test_a_repack_is_found_when_the_released_files_are_not(tmp_path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    target = checkpoints / "yue2_3b_bf16.safetensors"
    target.write_bytes(b"placeholder")
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    found = discovery.locate("standard")

    assert found.repack == str(target)
    assert found.lm == ""


def test_the_released_files_win_over_a_repack(tmp_path, monkeypatch):
    """They need no conversion, and both were measured to give the same model."""
    from test_discovery import publish

    lm_dir, _ = publish(tmp_path)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "yue2_3b_bf16.safetensors").write_bytes(b"placeholder")
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    found = discovery.locate("standard")

    assert found.repack == ""
    assert found.lm == str(lm_dir / "model.safetensors")


def test_the_legacy_decoder_is_not_taken_from_a_repack(tmp_path, monkeypatch):
    """The repack carries the standard decoder only, so legacy must not match it."""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "yue2_3b_bf16.safetensors").write_bytes(b"placeholder")
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    with pytest.raises(FileNotFoundError):
        discovery.locate("legacy")


def test_a_quantized_repack_is_told_apart_from_the_released_one(tmp_path):
    """Loading one as the other is 1355 tensor mismatches, so name it early."""
    plain = write_marker_file(tmp_path / "plain.safetensors")
    quantized = write_marker_file(tmp_path / "quantized.safetensors", quantized=True)

    assert discovery.identify(plain) == "repack"
    assert discovery.identify(quantized) == "repack_int8"


def test_the_marker_survives_a_file_that_lost_its_metadata():
    """Metadata is the first thing a re-saving tool drops; the keys are not."""
    assert repack.is_quantized({"a.comfy_quant": {}})
    assert repack.is_quantized({"__metadata__": {"quantization": "convrot_int8"}})
    assert not repack.is_quantized({"a.weight": {}, "__metadata__": {"format": "pt"}})


def test_the_requested_build_wins_when_both_are_on_disk(tmp_path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "yue2_3b_bf16.safetensors").write_bytes(b"placeholder")
    (checkpoints / "yue2_3b_int8_convrot.safetensors").write_bytes(b"placeholder")
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    assert discovery.locate("standard", "bf16").repack.endswith("bf16.safetensors")
    assert discovery.locate("standard", "int8").repack.endswith("int8_convrot.safetensors")


def test_asking_for_int8_does_not_refuse_the_file_that_is_here(tmp_path, monkeypatch):
    """Downloading seven gigabytes to honour a switch would serve nobody."""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "yue2_3b_bf16.safetensors").write_bytes(b"placeholder")
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    assert discovery.locate("standard", "int8").repack.endswith("bf16.safetensors")


def test_a_missing_checkpoint_is_reported_with_the_link_to_the_build_asked_for(
        tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    with pytest.raises(FileNotFoundError) as error:
        discovery.locate("standard", "int8")
    message = str(error.value)
    assert "yue2_3b_int8_convrot.safetensors" in message
    assert "yue2_3b_bf16.safetensors" not in message
