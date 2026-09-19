"""Every spelling of a YuE2 LoRA published by 2026-09-19 lands on the right tensors, from its header alone."""

from __future__ import annotations

import json
import pathlib

import pytest

import lora_files as lf
from yue2_comfy import placement
from yue2_comfy.lora import formats

MODELS = pathlib.Path(__file__).resolve().parent.parent.parent / "models"


def targets_of(reading, half=None):
    return sorted({piece.target for part in reading.parts for piece in part.pieces
                   if half is None or piece.half == half})


def test_the_architecture_is_the_one_placement_moves():
    """Two modules describe one model; a disagreement would fold into the wrong rows."""
    assert formats.HIDDEN == placement.HEADS * placement.HEAD_DIM
    assert formats.KV == placement.KV_HEADS * placement.HEAD_DIM
    assert formats.LAYERS == placement.LAYERS


def test_fused_rows_are_cut_once_each_and_in_order():
    for path, total in (("model.layers.0.self_attn.qkv_proj", 4096),
                        ("model.layers.0.nar_mlp.gate_up_proj", 12288)):
        spans = formats.cuts(path)
        assert spans[0][1] == 0 and spans[-1][2] == total
        for before, after in zip(spans, spans[1:]):
            assert before[2] == after[1]
    assert [name.rsplit(".", 1)[1] for name, _a, _b in formats.cuts("x.qkv_proj")] == [
        "q_proj", "k_proj", "v_proj"]
    assert [name.rsplit(".", 1)[1] for name, _a, _b in formats.cuts("x.gate_up_proj")] == [
        "gate_proj", "up_proj"]


def test_map_layout_with_a_peft_sidecar_takes_its_scale(tmp_path):
    """industrial-rock: lora_A/lora_B with no .weight, alpha in adapter_config.json."""
    folder = tmp_path / "adapter-nar"
    folder.mkdir()
    lf.write(folder / "lora.safetensors", lf.map_layer(0, half="nar", rank=32))
    (folder / "adapter_config.json").write_text(json.dumps({"rank": 32, "alpha": 48.0}))
    reading = formats.read(str(folder / "lora.safetensors"))
    assert reading.usable and reading.layout == formats.MAP
    assert reading.halves == (formats.NAR,)
    assert {part.scale for part in reading.parts} == {1.5}
    assert reading.scale_source == formats.SIDECAR
    assert "model.layers.0.nar_self_attn.q_proj.weight" in targets_of(reading)


def test_rank_stabilised_peft_divides_by_the_root_of_the_rank(tmp_path):
    folder = tmp_path / "rs"
    folder.mkdir()
    lf.write(folder / "adapter_model.safetensors",
             lf.map_layer(0, prefix="base_model.model.model.", rank=16,
                          down="lora_A.weight", up="lora_B.weight"))
    (folder / "adapter_config.json").write_text(json.dumps(
        {"r": 16, "lora_alpha": 32, "use_rslora": True, "base_model_name_or_path": "m-a-p/YuE2-3B"}))
    reading = formats.read(str(folder / "adapter_model.safetensors"))
    assert reading.usable
    assert {round(part.scale, 6) for part in reading.parts} == {round(32 / 4.0, 6)}
    assert reading.halves == (formats.AR,)


def test_mothersuperior_native_heads_are_replacements(tmp_path):
    tensors = lf.map_layer(3, half="nar", prefix="", rank=8)
    tensors.update({"llm2vae.weight": ("F32", (64, 2048)), "llm2vae.bias": ("F32", (64,)),
                    "vae2llm.weight": ("F32", (2048, 64)), "vae2llm.bias": ("F32", (2048,))})
    lf.write(tmp_path / "nar.safetensors", tensors, {"lora_scale": "1.0", "rank": "8"})
    reading = formats.read(str(tmp_path / "nar.safetensors"))
    assert reading.usable
    kinds = {part.module: part.kind for part in reading.parts}
    assert kinds["llm2vae"] == formats.REPLACE and kinds["vae2llm"] == formats.REPLACE
    assert "model.layers.3.nar_mlp.down_proj.weight" in targets_of(reading)
    assert {"llm2vae.weight", "llm2vae.bias", "vae2llm.weight", "vae2llm.bias"} <= set(targets_of(reading))
    assert reading.scale_source == "1.0"


def test_kohya_names_on_the_map_layout_are_read(tmp_path):
    """Deathmetal v1: model.layers.* with lora_down/lora_up.weight and no alpha."""
    lf.write(tmp_path / "dm.safetensors",
             lf.map_layer(0, rank=64, down="lora_down.weight", up="lora_up.weight"),
             {"format": "fl-yue2-lora-v1", "branch": "ar",
              "acoustic_adapter": "pretrained/nar_lora_joint_v4.safetensors"})
    reading = formats.read(str(tmp_path / "dm.safetensors"))
    assert reading.usable and reading.halves == (formats.AR,)
    assert {part.scale for part in reading.parts} == {1.0}
    assert reading.companion == "nar_lora_joint_v4"


def test_comfyui_fused_nar_with_alpha_cuts_qkv_and_gate_up(tmp_path):
    """jpop-t4: block-diagonal fused pairs with alpha, the trigger word in the metadata."""
    tensors = lf.comfy_layer(5, rank=16, alpha=16.0)
    tensors["diffusion_model.llm2vae.diff"] = ("F32", (64, 2048))
    tensors["diffusion_model.llm2vae.diff_b"] = ("F32", (64,))
    lf.write(tmp_path / "jpop.safetensors", tensors,
             {"format": "comfyui-native-lora", "trigger_word": "jpstyle26",
              "base_model": "yue2_3b_bf16.safetensors"})
    reading = formats.read(str(tmp_path / "jpop.safetensors"))
    assert reading.usable and reading.layout == formats.COMFYUI
    assert reading.halves == (formats.NAR,)
    assert reading.triggers == ("jpstyle26",)
    assert {round(part.scale, 6) for part in reading.parts if part.kind == formats.LOWRANK} == {1.0}
    qkv = next(part for part in reading.parts if part.module.endswith("qkv_proj"))
    assert [(piece.target, piece.start, piece.stop) for piece in qkv.pieces] == [
        ("model.layers.5.nar_self_attn.q_proj.weight", 0, 2048),
        ("model.layers.5.nar_self_attn.k_proj.weight", 2048, 3072),
        ("model.layers.5.nar_self_attn.v_proj.weight", 3072, 4096)]
    gate_up = next(part for part in reading.parts if part.module.endswith("gate_up_proj"))
    assert [piece.target.rsplit(".", 2)[1] for piece in gate_up.pieces] == ["gate_proj", "up_proj"]
    diff = next(part for part in reading.parts if part.kind == formats.DIFF)
    assert {piece.target for piece in diff.pieces} == {"llm2vae.weight", "llm2vae.bias"}
    assert formats.summary(reading)["halves"]["nar"]["rank"] == 16


@pytest.mark.parametrize("dtype", ["F32", "BF16", "F16", "F64"])
def test_alpha_is_read_in_every_width_it_is_stored_in(tmp_path, dtype):
    tensors = lf.pair("text_encoders.model.layers.0.self_attn.o_proj", (2048, 2048), 8,
                      down="lora_A.weight", up="lora_B.weight", alpha=4.0, alpha_dtype=dtype)
    lf.write(tmp_path / "a.safetensors", tensors)
    reading = formats.read(str(tmp_path / "a.safetensors"))
    assert reading.parts[0].scale == pytest.approx(0.5)
    assert reading.halves == (formats.AR,)


def test_comfyui_ar_pairs_in_diffusers_spelling_land_on_the_ar_half(tmp_path):
    """ai-toolkit and the ntc-ai sliders: text_encoders.* with lora_A.weight/lora_B.weight."""
    lf.write(tmp_path / "at.safetensors",
             lf.comfy_layer(1, tree="text_encoders.", rank=32, down="lora_A.weight",
                            up="lora_B.weight", block_diagonal=False),
             {"ss_base_model_version": "yue2"})
    reading = formats.read(str(tmp_path / "at.safetensors"))
    assert reading.usable and reading.halves == (formats.AR,)
    assert "model.layers.1.self_attn.v_proj.weight" in targets_of(reading)
    assert formats.summary(reading)["halves"]["ar"]["rank"] == 32


def test_dashed_adapter_names_and_single_letter_roles_are_read(tmp_path):
    lf.write(tmp_path / "ntc.safetensors",
             lf.map_layer(2, prefix="adapters.", separator="-", down="lora_down.weight",
                          up="lora_up.weight", alpha=8.0))
    lf.write(tmp_path / "t8.safetensors", lf.map_layer(0, prefix="", rank=16, down="A", up="B"),
             {"schema": "yue2-ar-lora-v1", "metadata": json.dumps(
                 {"nar_companion_sha256": next(iter(formats.KNOWN_COMPANIONS))})})
    dashed = formats.read(str(tmp_path / "ntc.safetensors"))
    assert dashed.usable and "model.layers.2.self_attn.k_proj.weight" in targets_of(dashed)
    assert {round(part.scale, 6) for part in dashed.parts} == {2.0}
    lettered = formats.read(str(tmp_path / "t8.safetensors"))
    assert lettered.usable and lettered.halves == (formats.AR,)
    assert lettered.companion.startswith("nar_lora_joint_v4")


def test_image_loras_and_other_language_models_are_not_yue2(tmp_path):
    lf.write(tmp_path / "sdxl.safetensors", {
        "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight": ("F16", (4, 320)),
        "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight": ("F16", (320, 4))})
    lf.write(tmp_path / "flux.safetensors", {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": ("BF16", (4, 3072)),
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": ("BF16", (9216, 4))})
    lf.write(tmp_path / "qwen4b.safetensors", {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": ("BF16", (8, 2560)),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": ("BF16", (4096, 8))})
    (tmp_path / "broken.safetensors").write_bytes(b"\x00" * 3)
    for name in ("sdxl", "flux", "qwen4b", "broken"):
        with pytest.raises(formats.NotYuE2):
            formats.read(str(tmp_path / (name + ".safetensors")))


def test_a_lora_for_another_model_of_the_same_shape_says_so(tmp_path):
    """Qwen3-1.7B has YuE2's AR shapes; only the file's own word tells them apart."""
    folder = tmp_path / "qwen"
    folder.mkdir()
    lf.write(folder / "adapter_model.safetensors",
             lf.map_layer(0, prefix="base_model.model.model.", down="lora_A.weight", up="lora_B.weight"))
    (folder / "adapter_config.json").write_text(json.dumps(
        {"r": 4, "lora_alpha": 4, "base_model_name_or_path": "Qwen/Qwen3-1.7B"}))
    reading = formats.read(str(folder / "adapter_model.safetensors"))
    assert not reading.usable and "Qwen/Qwen3-1.7B" in reading.problem


def test_an_adapter_with_modules_of_its_own_is_refused_by_name(tmp_path):
    """Mothersuperior's hum-to-song carries hum_proj, which needs an input YuE2 does not have."""
    tensors = lf.map_layer(0, half="nar", prefix="", rank=8)
    tensors["hum_proj.0.weight"] = ("F32", (2048, 64))
    tensors["hum_proj.0.bias"] = ("F32", (2048,))
    lf.write(tmp_path / "hum.safetensors", tensors)
    reading = formats.read(str(tmp_path / "hum.safetensors"))
    assert not reading.usable and "hum_proj.0" in reading.problem


@pytest.mark.parametrize("extra,word", [
    ({"model.layers.0.self_attn.q_proj.dora_scale": ("F32", (2048, 1))}, "dora_scale"),
    ({"model.norm.diff": ("F32", (2048,))}, "does not change"),
    ({"lm_head.lora_A": ("F32", (4, 2048)), "lm_head.lora_B": ("F32", (184704, 4))}, "does not change"),
    ({"model.layers.0.mlp.down_proj.lora_A": ("F32", (4, 2048)),
      "model.layers.0.mlp.down_proj.lora_B": ("F32", (2048, 4))}, "where YuE2 has"),
])
def test_what_cannot_be_folded_is_named_rather_than_skipped(tmp_path, extra, word):
    tensors = {key: value for key, value in lf.map_layer(0).items() if "down_proj" not in key}
    tensors.update(extra)
    lf.write(tmp_path / "odd.safetensors", tensors)
    reading = formats.read(str(tmp_path / "odd.safetensors"))
    assert not reading.usable and word in reading.problem


def test_a_fused_pair_of_the_wrong_height_is_refused(tmp_path):
    tensors = lf.pair("diffusion_model.model.layers.0.self_attn.qkv_proj", (3072, 2048), 4,
                      down="lora_down.weight", up="lora_up.weight")
    tensors.update(lf.pair("diffusion_model.model.layers.0.self_attn.o_proj", (2048, 2048), 4,
                           down="lora_down.weight", up="lora_up.weight"))
    lf.write(tmp_path / "short.safetensors", tensors)
    reading = formats.read(str(tmp_path / "short.safetensors"))
    assert not reading.usable and "4096" in reading.problem


def test_comfyui_norms_of_the_acoustic_half_take_their_nar_names(tmp_path):
    lf.write(tmp_path / "norms.safetensors", {
        "diffusion_model.model.layers.4.input_layernorm.diff": ("F32", (2048,)),
        "diffusion_model.model.layers.4.post_attention_layernorm.diff": ("F32", (2048,)),
        "text_encoders.model.layers.4.post_attention_layernorm.diff": ("F32", (2048,))})
    reading = formats.read(str(tmp_path / "norms.safetensors"))
    assert reading.usable
    assert targets_of(reading) == ["model.layers.4.nar_input_layernorm.weight",
                                   "model.layers.4.nar_pre_mlp_layernorm.weight",
                                   "model.layers.4.post_attention_layernorm.weight"]
    assert reading.halves == (formats.AR, formats.NAR)


def test_triggers_come_as_text_or_as_a_list(tmp_path):
    lf.write(tmp_path / "t.safetensors", lf.map_layer(0),
             {"trigger_words": json.dumps(["xyzq", "in the style of xyzq"]),
              "modelspec.trigger_phrase": "xyzq, extra", "intended_cot": "full"})
    reading = formats.read(str(tmp_path / "t.safetensors"))
    assert reading.triggers == ("xyzq", "in the style of xyzq", "extra")
    assert reading.intended_cot == "full"


LOCAL = [
    "loras/yue2-industrial-rock-lora/adapter-ar-179/lora.safetensors",
    "loras/yue2-industrial-rock-lora/adapter-nar-179/lora.safetensors",
    "loras/yue2-jpop-t4-lora/yue2_jpop_t4.safetensors",
    "loras/YuE2_Deathmetalv1_lora/deathmetalv1_step-000600.safetensors",
    "mothersuperior-realaudio-tokenizer/nar_lora_joint_v4.safetensors",
    "mothersuperior-realaudio-tokenizer/nar_lora_joint_v4_comfyui.safetensors",
    "Mothersuperior-YuE2-instrumental-cot-full-loras/ar_lora_inst_v3abc.safetensors",
    "Mothersuperior-YuE2-instrumental-cot-full-loras/ar_lora_inst_v3abc_comfyui.safetensors",
]


@pytest.mark.parametrize("name", LOCAL)
def test_the_published_files_on_this_machine_fold_whole(name):
    """The real files, where a development checkout keeps them beside itself; skipped elsewhere."""
    path = MODELS / name
    if not path.is_file():
        pytest.skip("not on this machine")
    reading = formats.read(str(path))
    assert reading.usable, reading.problem
    assert sum(reading.tensors(half) for half in reading.halves) in (196, 198, 200)


@pytest.mark.parametrize("native,comfy", [(LOCAL[4], LOCAL[5]), (LOCAL[6], LOCAL[7])])
def test_a_native_file_and_its_comfyui_twin_change_the_same_tensors(native, comfy):
    if not (MODELS / native).is_file() or not (MODELS / comfy).is_file():
        pytest.skip("not on this machine")
    assert targets_of(formats.read(str(MODELS / native))) == targets_of(
        formats.read(str(MODELS / comfy)))
