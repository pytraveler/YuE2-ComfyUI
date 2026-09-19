"""Folding LoRAs into a model and taking them out, checked on the CPU on one layer of the real shapes.

The arithmetic has to be ComfyUI's and exact: a folded weight is the float32
sum rounded once, a fused pair is multiplied whole and then cut, and taking an
adapter out gives back the very tensors the model had. The bookkeeping has to
survive what the pack does between stages -- above all a Placement made again,
which is how a fold was once applied twice. On the card, the same was checked
against ComfyUI core bit for bit and song by song in every offload mode; the
record is in the plan of M22.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from yue2_comfy import placement, quantized  # noqa: E402
from yue2_comfy.lora import apply, choices, formats  # noqa: E402

H, F, KV, LAT = 2048, 6144, 1024, 64


def linear(out_features, in_features, bias=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    layer = torch.nn.Linear(in_features, out_features, bias=bias, device="meta")
    layer.weight = torch.nn.Parameter(
        (torch.randn(out_features, in_features, generator=generator) * 0.02).to(torch.bfloat16),
        requires_grad=False)
    if bias:
        layer.bias = torch.nn.Parameter(
            (torch.randn(out_features, generator=generator) * 0.02).to(torch.bfloat16),
            requires_grad=False)
    return layer


class Norm(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size, dtype=torch.bfloat16), requires_grad=False)


class Attention(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.q_proj = linear(H, H, seed=seed)
        self.k_proj = linear(KV, H, seed=seed + 1)
        self.v_proj = linear(KV, H, seed=seed + 2)
        self.o_proj = linear(H, H, seed=seed + 3)
        self.q_norm = Norm(128)
        self.k_norm = Norm(128)


class Mlp(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.gate_proj = linear(F, H, seed=seed)
        self.up_proj = linear(F, H, seed=seed + 1)
        self.down_proj = linear(H, F, seed=seed + 2)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = Norm(H)
        self.self_attn = Attention(10)
        self.post_attention_layernorm = Norm(H)
        self.mlp = Mlp(20)
        self.nar_input_layernorm = Norm(H)
        self.nar_self_attn = Attention(30)
        self.nar_pre_mlp_layernorm = Norm(H)
        self.nar_mlp = Mlp(40)


class Tops(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer()])
        self.embed_tokens = torch.nn.Embedding(8, H)
        self.norm = Norm(H)


class Timer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(linear(H, 256, bias=True, seed=50), torch.nn.SiLU(),
                                       linear(H, H, bias=True, seed=51))


class Positions(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pe", torch.zeros(4, H))


class Model(torch.nn.Module):
    """One layer of YuE2 with every name the real model has, on the CPU."""

    def __init__(self):
        super().__init__()
        self.model = Tops()
        self.lm_head = linear(8, H, seed=60)
        self.llm2vae = linear(LAT, H, bias=True, seed=61)
        self.vae2llm = linear(H, LAT, bias=True, seed=62)
        self.time_embedder = Timer()
        self.latent_pos_embed = Positions()


@pytest.fixture
def lm():
    return Model().eval()


def run(lm, *picked):
    return types.SimpleNamespace(lm=lm, device=torch.device("cpu"), offload="auto",
                                 loras=choices.from_settings(list(picked)))


def arrange(models, half):
    placement.arrange(models, half, 0, "test")


def pair(prefix, name, out_features, in_features, rank, seed, down="lora_A", up="lora_B"):
    generator = torch.Generator().manual_seed(seed)
    return {prefix + name + "." + down: torch.randn(rank, in_features, generator=generator) * 0.05,
            prefix + name + "." + up: torch.randn(out_features, rank, generator=generator) * 0.05}


def map_ar_file(path, rank=4, seed=1):
    tensors = {}
    for index, (name, shape) in enumerate({"self_attn.q_proj": (H, H), "self_attn.k_proj": (KV, H),
                                           "self_attn.v_proj": (KV, H), "self_attn.o_proj": (H, H),
                                           "mlp.gate_proj": (F, H), "mlp.up_proj": (F, H),
                                           "mlp.down_proj": (H, F)}.items()):
        tensors.update(pair("model.layers.0.", name, shape[0], shape[1], rank, seed + index))
    safetensors_torch.save_file(tensors, str(path))
    return {"name": path.name, "path": str(path), "sha256": path.name, "ar": 1.0, "nar": 0.0}


def comfy_nar_file(path, rank=4, seed=7):
    tensors = {}
    for index, (name, shape) in enumerate({"self_attn.qkv_proj": (H + 2 * KV, H),
                                           "mlp.gate_up_proj": (2 * F, H)}.items()):
        tensors.update(pair("diffusion_model.model.layers.0.", name, shape[0], shape[1], rank, seed + index,
                            down="lora_down.weight", up="lora_up.weight"))
    tensors["diffusion_model.model.layers.0.self_attn.qkv_proj.alpha"] = torch.tensor(float(rank * 2))
    generator = torch.Generator().manual_seed(seed + 9)
    tensors["diffusion_model.llm2vae.diff"] = torch.randn(LAT, H, generator=generator) * 0.01
    tensors["diffusion_model.llm2vae.diff_b"] = torch.randn(LAT, generator=generator) * 0.01
    safetensors_torch.save_file(tensors, str(path))
    return {"name": path.name, "path": str(path), "sha256": path.name, "ar": 0.0, "nar": 1.0}


def weight(lm, name):
    owner, attribute = apply._locate(lm, name)
    return getattr(owner, attribute).detach()


def expected(lm, original, path, strength, half):
    """W32 plus each part's strength * scale * (up @ down), cut after the product, rounded once."""
    out = {}
    with safetensors_torch.safe_open(path, framework="pt") as handle:
        for part in formats.read(path).parts:
            for piece in part.pieces:
                if piece.half != half:
                    continue
                if part.kind == formats.LOWRANK:
                    product = torch.mm(handle.get_tensor(part.up_key).float(),
                                       handle.get_tensor(part.down_key).float())
                    product.mul_(strength * part.scale)
                else:
                    key = part.weight_key if piece.target.endswith(".weight") else part.bias_key
                    product = handle.get_tensor(key).float() * strength
                rows = slice(piece.start, piece.stop) if piece.start is not None else slice(None)
                base = out.get(piece.target, original[piece.target].float())
                out[piece.target] = base + product[rows]
    return {name: value.to(torch.bfloat16) for name, value in out.items()}


def snapshot(lm):
    return {name: weight(lm, name).clone() for name in formats.TARGETS if name.startswith(
        ("model.layers.0.", "llm2vae", "vae2llm", "time_embedder"))}


def test_a_fold_is_the_float32_sum_rounded_once_and_comes_out_exactly(lm, tmp_path):
    original = snapshot(lm)
    storages = {name: weight(lm, name).data_ptr() for name in original}
    spec = dict(map_ar_file(tmp_path / "ar.safetensors"), ar=0.7)
    arrange(run(lm, spec), formats.AR)
    want = expected(lm, original, spec["path"], 0.7, formats.AR)
    assert len(want) == 7
    for name, value in want.items():
        assert torch.equal(weight(lm, name), value), name
    assert torch.equal(weight(lm, "model.layers.0.nar_mlp.up_proj.weight"),
                       original["model.layers.0.nar_mlp.up_proj.weight"])

    arrange(run(lm), formats.AR)
    for name in want:
        assert torch.equal(weight(lm, name), original[name]), name
        assert weight(lm, name).data_ptr() == storages[name], "the model's own tensor is back"


def test_a_new_strength_folds_from_the_original_not_on_top(lm, tmp_path):
    original = snapshot(lm)
    spec = map_ar_file(tmp_path / "ar.safetensors")
    arrange(run(lm, dict(spec, ar=0.5)), formats.AR)
    arrange(run(lm, dict(spec, ar=1.25)), formats.AR)
    for name, value in expected(lm, original, spec["path"], 1.25, formats.AR).items():
        assert torch.equal(weight(lm, name), value), name


def test_the_record_outlives_a_placement_made_again(lm, tmp_path):
    """vocabulary.narrowed drops the Placement at every stage; a fold must not happen twice."""
    original = snapshot(lm)
    spec = dict(map_ar_file(tmp_path / "ar.safetensors"), ar=0.9)
    models = run(lm, spec)
    arrange(models, formats.AR)
    lm._yue2_placement = None
    arrange(models, formats.AR)
    for name, value in expected(lm, original, spec["path"], 0.9, formats.AR).items():
        assert torch.equal(weight(lm, name), value), name


def test_a_fused_pair_is_multiplied_whole_then_cut_and_the_heads_take_differences(lm, tmp_path):
    original = snapshot(lm)
    spec = dict(comfy_nar_file(tmp_path / "nar.safetensors"), nar=0.6)
    arrange(run(lm, spec), formats.NAR)
    want = expected(lm, original, spec["path"], 0.6, formats.NAR)
    assert {"model.layers.0.nar_self_attn.q_proj.weight", "model.layers.0.nar_self_attn.v_proj.weight",
            "model.layers.0.nar_mlp.up_proj.weight", "llm2vae.weight", "llm2vae.bias"} <= set(want)
    for name, value in want.items():
        assert torch.equal(weight(lm, name), value), name
    qkv = next(part for part in formats.read(spec["path"]).parts if part.module.endswith("qkv_proj"))
    assert qkv.scale == 2.0

    arrange(run(lm), formats.NAR)
    for name in want:
        assert torch.equal(weight(lm, name), original[name]), name


def test_a_replacement_is_a_difference_against_the_model_own_weight(lm, tmp_path):
    original = snapshot(lm)
    generator = torch.Generator().manual_seed(3)
    new = {"llm2vae.weight": torch.randn(LAT, H, generator=generator) * 0.02,
           "llm2vae.bias": torch.randn(LAT, generator=generator) * 0.02}
    new.update(pair("layers.0.", "nar_mlp.down_proj", H, F, 4, 11))
    path = tmp_path / "ms.safetensors"
    safetensors_torch.save_file(new, str(path))
    arrange(run(lm, {"name": "ms", "path": str(path), "sha256": "ms", "ar": 0.0, "nar": 0.5}), formats.NAR)
    for name in ("llm2vae.weight", "llm2vae.bias"):
        halfway = (original[name].float() + 0.5 * (new[name] - original[name].float())).to(torch.bfloat16)
        assert torch.equal(weight(lm, name), halfway), name


def test_two_adapters_fold_the_same_whichever_row_comes_first(lm, tmp_path):
    first = dict(map_ar_file(tmp_path / "a.safetensors", seed=1), sha256="aaa", ar=0.4)
    second = dict(map_ar_file(tmp_path / "b.safetensors", seed=5), sha256="bbb", ar=0.8)
    arrange(run(lm, first, second), formats.AR)
    one = snapshot(lm)
    arrange(run(lm), formats.AR)
    arrange(run(lm, second, first), formats.AR)
    for name, value in one.items():
        assert torch.equal(weight(lm, name), value), name


def test_nothing_is_folded_again_when_nothing_changed(lm, tmp_path, monkeypatch):
    spec = map_ar_file(tmp_path / "ar.safetensors")
    models = run(lm, spec)
    arrange(models, formats.AR)
    calls = []
    monkeypatch.setattr(apply, "_fold_plain", lambda *args: calls.append(args) or [])
    arrange(models, formats.AR)
    arrange(models, formats.NAR)
    assert calls == []


def test_packed_layers_hold_the_adapter_beside_their_rows(lm, tmp_path):
    spec = dict(map_ar_file(tmp_path / "ar.safetensors"), ar=0.8)
    quantized.compress(lm)
    layer = lm.model.layers[0].self_attn.q_proj
    assert isinstance(layer, quantized.Packed)
    probe = torch.randn(3, H, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(2))
    before = layer(probe).float()
    arrange(run(lm, spec), formats.AR)
    assert layer.lora_down is not None and layer.lora_up.shape == (H, 4)
    with safetensors_torch.safe_open(spec["path"], framework="pt") as handle:
        down = handle.get_tensor("model.layers.0.self_attn.q_proj.lora_A").float()
        up = handle.get_tensor("model.layers.0.self_attn.q_proj.lora_B").float()
    term = 0.8 * (probe.float() @ down.T @ up.T)
    added = layer(probe).float() - before
    assert ((added - term).norm() / term.norm()).item() < 0.03
    assert torch.all((added - term).abs() <= before.abs().clamp(min=1.0) * 2 ** -7)
    arrange(run(lm), formats.AR)
    assert layer.lora_down is None and layer.lora_up is None
    assert torch.equal(layer(probe).float(), before)


def test_a_whole_matrix_difference_cannot_go_beside_packed_rows(lm, tmp_path):
    path = tmp_path / "diff.safetensors"
    safetensors_torch.save_file({"text_encoders.model.layers.0.self_attn.o_proj.diff": torch.zeros(H, H)},
                                str(path))
    quantized.compress(lm)
    with pytest.raises(ValueError, match="low_vram"):
        arrange(run(lm, {"name": "d", "path": str(path), "sha256": "d", "ar": 1.0, "nar": 0.0}),
                formats.AR)
