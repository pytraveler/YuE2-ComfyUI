"""The layers packed into INT8 rows, checked without a card.

What has to hold is arithmetic: the packing has to reconstruct a matrix to
about a percent, the packed layer has to answer what the original answered to
that same tolerance, the rows have to be half the bytes and travel with the
module the way ``placement`` moves it, and the stand-in that gets GraphAR past
its own guard has to put the packed layers back afterwards. The memory and the
time this costs on a real song were measured on a card and are recorded in the
module.
"""

from __future__ import annotations

import pytest

quantized = pytest.importorskip("yue2_comfy.quantized")
torch = pytest.importorskip("torch")

from yue2_comfy import placement

HIDDEN = 32


def linear(out_features=24, in_features=HIDDEN, seed=5):
    torch.manual_seed(seed)
    layer = torch.nn.Linear(in_features, out_features, bias=False)
    layer.weight.data = layer.weight.data.to(torch.bfloat16)
    return layer


class Block(torch.nn.Module):
    """One attention or MLP block: linears, and a norm that is not one."""

    def __init__(self):
        super().__init__()
        self.q_proj = linear()
        self.o_proj = linear(seed=6)
        self.q_norm = torch.nn.LayerNorm(HIDDEN)


class Layer(torch.nn.Module):
    """The four blocks a layer holds, and the four norms between them."""

    def __init__(self):
        super().__init__()
        for name in quantized.BLOCKS:
            setattr(self, name, Block())
        for name in placement.AR_LAYER + placement.NAR_LAYER:
            if name not in quantized.BLOCKS:
                setattr(self, name, torch.nn.LayerNorm(HIDDEN))


class Backbone(torch.nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList(Layer() for _ in range(layers))
        self.embed_tokens = torch.nn.Embedding(8, HIDDEN).to(torch.bfloat16)


class Model(torch.nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.model = Backbone(layers)


class Fake:
    """What ``captured`` reads off a Models tuple."""

    def __init__(self, lm):
        self.lm = lm


def test_packing_reconstructs_a_matrix_to_about_a_percent():
    weight = linear(out_features=64, in_features=256).weight.detach()
    rows, scales = quantized.pack(weight)
    back = rows.to(torch.float32) * scales.to(torch.float32)[:, None]
    straight = weight.float()
    cosine = torch.nn.functional.cosine_similarity(back.flatten(), straight.flatten(), dim=0)
    assert float(cosine) > 0.9999
    assert float((back - straight).abs().mean() / straight.abs().mean()) < 0.02


def test_a_row_with_its_own_range_keeps_its_own_scale():
    weight = torch.zeros(2, 8, dtype=torch.bfloat16)
    weight[0] = 1e-3
    weight[1] = 1.0
    rows, scales = quantized.pack(weight)
    assert float(scales[0]) < float(scales[1])
    back = rows.to(torch.float32) * scales.to(torch.float32)[:, None]
    assert torch.allclose(back[0], weight[0].float(), rtol=0.01, atol=0)


def test_a_row_of_zeros_stays_zero():
    rows, scales = quantized.pack(torch.zeros(3, 8, dtype=torch.bfloat16))
    assert int(rows.abs().sum()) == 0
    assert torch.isfinite(scales).all()


def test_the_packed_layer_answers_what_the_linear_answered():
    original = linear(out_features=64, in_features=256)
    hidden = torch.randn(4, 256, dtype=torch.bfloat16)
    packed = quantized.Packed(original)
    plain = torch.nn.functional.linear(hidden, original.weight.detach())
    answer = packed(hidden)
    assert answer.shape == plain.shape and answer.dtype == plain.dtype
    difference = (answer.float() - plain.float()).abs().mean()
    assert float(difference / plain.float().abs().mean()) < 0.02


def test_the_packed_layer_carries_a_bias_when_there_is_one():
    original = torch.nn.Linear(HIDDEN, 4, bias=True).to(torch.bfloat16)
    hidden = torch.randn(2, HIDDEN, dtype=torch.bfloat16)
    packed = quantized.Packed(original)
    plain = torch.nn.functional.linear(hidden, original.weight.detach(),
                                       original.bias.detach())
    assert float((packed(hidden).float() - plain.float()).abs().max()) < 0.2


def test_packing_halves_what_the_module_keeps():
    original = linear(out_features=64, in_features=256)
    before = sum(t.numel() * t.element_size() for t in original.parameters())
    packed = quantized.Packed(original)
    after = sum(t.numel() * t.element_size()
                for t in list(packed.parameters()) + list(packed.buffers()))
    assert after < before * 0.55


def test_compress_packs_every_matrix_of_both_halves_and_leaves_the_norms():
    model = Model()
    assert quantized.compress(model) == 2 * len(quantized.BLOCKS) * 2
    for layer in model.model.layers:
        for name in quantized.BLOCKS:
            block = getattr(layer, name)
            assert isinstance(block.q_proj, quantized.Packed)
            assert isinstance(block.q_norm, torch.nn.LayerNorm)
    assert quantized.compressed(model)


def test_compress_makes_placement_measure_the_halves_again():
    model = Model()
    model._yue2_placement = object()
    quantized.compress(model)
    assert model._yue2_placement is None


def test_an_uncompressed_model_says_so():
    assert not quantized.compressed(Model())
    assert not quantized.compressed(None)


def test_the_rows_travel_with_the_module():
    packed = quantized.Packed(linear())
    moved = packed.to(torch.device("cpu"))
    assert moved.rows.dtype == torch.int8
    assert [t.dtype for t in moved.buffers()].count(torch.int8) == 1


def test_placement_counts_the_packed_rows_as_the_half_it_moves():
    """The matrices halve; the norms beside them do not, and here they are a far
    larger share of a half than they are in the released model, where the NAR
    half was measured going from 2.63 GiB to 1.31."""
    model = Model()
    model.lm_head = torch.nn.Linear(HIDDEN, 4, bias=False)
    model.model.norm = torch.nn.LayerNorm(HIDDEN)
    for name in ("llm2vae", "vae2llm", "time_embedder", "latent_pos_embed"):
        setattr(model, name, torch.nn.LayerNorm(HIDDEN))
    device = torch.device("cpu")
    before = placement.Placement(model, device).sizes[placement.NAR]
    quantized.compress(model)
    after = placement.Placement(model, device).sizes[placement.NAR]
    assert after < before * 0.65


def test_a_packed_matrix_is_what_the_upstream_guard_rejects():
    """Why ``captured`` exists, written down: GraphAR asks exactly this question."""
    model = Model()
    quantized.compress(model)
    matrix = model.model.layers[0].self_attn.q_proj
    dtype = model.model.embed_tokens.weight.dtype
    assert not isinstance(matrix, torch.nn.Linear)
    with quantized._plain(model):
        shown = model.model.layers[0].self_attn.q_proj
        assert isinstance(shown, torch.nn.Linear) and shown.weight.dtype == dtype


def test_the_stand_in_looks_like_a_linear_of_the_model_dtype():
    model = Model()
    quantized.compress(model)
    with quantized._plain(model):
        shown = model.model.layers[0].self_attn.q_proj
        assert isinstance(shown, torch.nn.Linear)
        assert shown.weight.dtype == model.model.embed_tokens.weight.dtype
    assert isinstance(model.model.layers[0].self_attn.q_proj, quantized.Packed)


def test_the_stand_in_leaves_the_other_half_alone():
    model = Model()
    quantized.compress(model)
    with quantized._plain(model):
        assert isinstance(model.model.layers[0].nar_self_attn.q_proj, quantized.Packed)


def test_the_packed_layers_come_back_even_if_the_capture_fails():
    model = Model()
    quantized.compress(model)
    with pytest.raises(ValueError):
        with quantized._plain(model):
            raise ValueError("capture failed")
    assert isinstance(model.model.layers[0].self_attn.q_proj, quantized.Packed)


def test_captured_leaves_an_unpacked_model_untouched():
    from yue2_comfy.vendor.yue2 import cuda_graph

    original = cuda_graph.GraphAR
    with quantized.captured(Fake(Model())):
        assert cuda_graph.GraphAR is original
    assert cuda_graph.GraphAR is original


def test_captured_swaps_the_graph_class_and_puts_it_back():
    from yue2_comfy.vendor.yue2 import cuda_graph

    model = Model()
    quantized.compress(model)
    original = cuda_graph.GraphAR
    with quantized.captured(Fake(model)):
        assert cuda_graph.GraphAR is not original
        assert issubclass(cuda_graph.GraphAR, original)
    assert cuda_graph.GraphAR is original


def test_the_decode_asks_for_less_room_when_the_tiles_are_smaller():
    assert placement.decode_bytes(True) < placement.decode_bytes(False)
    assert placement.decode_bytes(False) == placement.DECODE_BYTES


def test_the_decode_retries_in_smaller_tiles_not_larger_ones():
    """A retry after running out of memory has to ask for less than the try that failed."""
    from yue2_comfy import generate

    assert generate.LOW_VRAM_FALLBACK_CORE_FRAMES < generate.LOW_VRAM_CORE_FRAMES
    assert generate.FALLBACK_CORE_FRAMES < 1024
