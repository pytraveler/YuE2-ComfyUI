"""What the speech models hold on the card, and what 'low_vram' changes about it.

Measured on 2026-09-23 (see ``asr.network.QUERY_BLOCK``, ``SMALL_CONV_CHUNKS``
and ``asr.model.pack``): a request's attention in blocks of queries is the
whole call's on the CPU to the bit, and on the card on eleven of twelve songs;
a small card packs the language model's layers and convolves the audio in
parts; the two listeners take turns on a small card, and what they heard is
kept apart by the shape they heard it with.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from yue2_comfy.asr import network, runtime  # noqa: E402


def grouped(length, keys=None, dtype=torch.bfloat16, seed=3):
    generator = torch.Generator().manual_seed(seed)
    keys = length if keys is None else keys
    q = torch.randn(1, 16, length, 8, generator=generator).to(dtype)
    k = torch.randn(1, 8, keys, 8, generator=generator).to(dtype)
    v = torch.randn(1, 8, keys, 8, generator=generator).to(dtype)
    return q, k, v


def test_a_request_in_blocks_is_attended_as_it_was_at_once(monkeypatch):
    """The same sum row by row: in bfloat16 on the CPU every row comes out the same, to the bit."""
    q, k, v = grouped(37)
    whole = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=network.SCALE,
                                                              enable_gqa=True)
    monkeypatch.setattr(network, "QUERY_BLOCK", 5)
    assert torch.equal(network.attend(q, k, v, 0), whole)


def test_a_block_sees_the_positions_up_to_its_own_and_no_further(monkeypatch):
    """A request that starts after positions the cache already holds: row i sees up to position 4 + i."""
    q, k, v = grouped(37, keys=41)
    mask = torch.arange(41)[None, :] <= (4 + torch.arange(37))[:, None]
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=network.SCALE,
                                                                 enable_gqa=True)
    monkeypatch.setattr(network, "QUERY_BLOCK", 5)
    assert torch.equal(network.attend(q, k, v, 4), expected)


def test_a_request_within_a_block_is_one_causal_call_as_before(monkeypatch):
    seen = []
    real = network.F.scaled_dot_product_attention

    def recorded(q, k, v, **kwargs):
        seen.append((q.shape[2], kwargs.get("is_causal"), kwargs.get("attn_mask") is not None))
        return real(q, k, v, **kwargs)

    monkeypatch.setattr(network.F, "scaled_dot_product_attention", recorded)
    monkeypatch.setattr(network, "QUERY_BLOCK", 8)
    network.attend(*grouped(8))
    assert seen == [(8, True, False)]
    seen.clear()
    network.attend(*grouped(19))
    assert seen == [(8, None, True), (8, None, True), (3, None, True)]


def test_the_convolutions_go_in_parts_of_the_towers_own_size(monkeypatch):
    """A small card's tower convolves forty seconds at a time; the rest of the tower is the same."""
    for name, value in (("CONV_WIDTH", 4), ("AUDIO_WIDTH", 8), ("AUDIO_HEADS", 2), ("AUDIO_LAYERS", 1),
                        ("AUDIO_FFN", 16)):
        monkeypatch.setattr(network, name, value)
    torch.manual_seed(5)
    tower = network.AudioTower()
    assert tower.conv_chunks == network.CONV_CHUNKS
    mel = torch.randn(network.MELS, 460)
    parts = []
    convolved = tower._convolved
    monkeypatch.setattr(tower, "_convolved", lambda x: parts.append(x.shape[0]) or convolved(x))
    with torch.inference_mode():
        whole = tower(mel)
        tower.conv_chunks = 2
        pieces = tower(mel)
    assert parts == [5, 2, 2, 1]
    assert pieces.shape == whole.shape and torch.allclose(pieces, whole, atol=1e-5)


def test_a_small_card_packs_the_language_models_layers_and_nothing_else():
    """Seven matrices in each of 28 layers; the encoder dropped a verse when packed, so it is not."""
    from yue2_comfy import quantized
    from yue2_comfy.asr import model

    with torch.device("meta"):
        net = network.Network()
    assert model.pack(net) == network.LAYERS * 7
    layer = net.language_model.layers[0]
    assert all(isinstance(getattr(layer.self_attn, name), quantized.Packed)
               for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
    assert all(isinstance(getattr(layer.mlp, name), quantized.Packed) for name in ("gate_proj", "up_proj", "down_proj"))
    assert isinstance(net.audio_tower.layers[0].fc1, torch.nn.Linear)
    assert isinstance(net.language_model.embed_tokens, torch.nn.Embedding)
    assert isinstance(net.multi_modal_projector.linear_2, torch.nn.Linear)


@pytest.fixture
def listeners(monkeypatch):
    """The two listeners stood in: what was loaded, with which shape, and what was let go."""
    from yue2_comfy.asr import aligner, tokenizer

    events = []
    fake_model = types.SimpleNamespace(
        load=lambda folder, device, low_vram=False: events.append(("speech", low_vram)) or types.SimpleNamespace(
            forget_steps=lambda: None))
    monkeypatch.setitem(sys.modules, "yue2_comfy.asr.model", fake_model)
    monkeypatch.setattr(sys.modules["yue2_comfy.asr"], "model", fake_model, raising=False)
    monkeypatch.setattr(tokenizer, "Tokenizer", lambda path: "tokenizer")
    monkeypatch.setattr(aligner, "load", lambda folder, reader, device, low_vram=False: events.append(
        ("aligner", low_vram)) or "held")
    monkeypatch.setattr(runtime, "stamp", lambda folder: ("weights",))
    monkeypatch.setattr(runtime, "_STATE", {"key": None, "net": None, "tokenizer": None})
    monkeypatch.setattr(runtime, "_ALIGNER", {"key": None, "held": None})
    return events


def test_a_small_card_loads_the_listeners_one_at_a_time(listeners):
    runtime.acquire("speech", "cpu", low_vram=True)
    assert runtime.is_loaded()
    runtime.acquire_aligner("aligner", "reader", "cpu", low_vram=True)
    assert runtime.aligner_loaded() and not runtime.is_loaded(), "the speech model made way"
    runtime.acquire("speech", "cpu", low_vram=True)
    assert runtime.is_loaded() and not runtime.aligner_loaded(), "and the aligner made way for it"
    assert listeners == [("speech", True), ("aligner", True), ("speech", True)]


def test_a_card_with_room_keeps_both_listeners_and_loads_them_as_they_are(listeners):
    runtime.acquire("speech", "cpu")
    runtime.acquire_aligner("aligner", "reader", "cpu")
    runtime.acquire("speech", "cpu")
    assert runtime.is_loaded() and runtime.aligner_loaded()
    assert listeners == [("speech", False), ("aligner", False)]


def test_the_packed_model_is_a_different_model_to_the_cache(listeners):
    runtime.acquire("speech", "cpu")
    runtime.acquire("speech", "cpu", low_vram=True)
    runtime.acquire("speech", "cpu", low_vram=True)
    assert listeners == [("speech", False), ("speech", True)]


def test_when_the_card_is_short_the_other_listener_goes_first(monkeypatch):
    """It is the quickest to load again; the song model, the slowest, goes last."""
    from yue2_comfy import loader

    management = types.SimpleNamespace(unload_all_models=lambda: None, soft_empty_cache=lambda force=False: None)
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=management))
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    gone = []
    free = [0]
    monkeypatch.setattr(runtime, "_free_bytes", lambda device: free[0])
    monkeypatch.setattr(loader, "is_loaded", lambda: True)
    monkeypatch.setattr(loader, "unload", lambda: gone.append("YuE2"))

    def let_go():
        gone.append("aligner")
        free[0] = runtime.ROOM_BYTES

    runtime._make_room(types.SimpleNamespace(type="cuda"), runtime.ROOM_BYTES, ("aligner", lambda: True, let_go))
    assert gone == ["aligner"]
    free[0] = 0
    gone.clear()
    runtime._make_room(types.SimpleNamespace(type="cuda"), runtime.ROOM_BYTES, ("aligner", lambda: False, let_go))
    assert gone == ["YuE2"]


def test_what_was_heard_is_kept_apart_by_the_shape_it_was_heard_with(monkeypatch):
    """The packed model can hear a word differently, so its answers are not the full model's."""
    asked = []
    fake_model = types.SimpleNamespace(recognise=lambda net, tokenizer, audio, language="", cancelled=None:
                                       asked.append(audio) or {"language": "English", "text": audio})
    monkeypatch.setitem(sys.modules, "yue2_comfy.asr.model", fake_model)
    monkeypatch.setattr(sys.modules["yue2_comfy.asr"], "model", fake_model, raising=False)
    net = types.SimpleNamespace(forget_steps=lambda: None)
    monkeypatch.setattr(runtime, "acquire", lambda folder, device, progress=None, low_vram=False: (net, "tok"))
    monkeypatch.setattr(runtime, "stamp", lambda folder: ("weights",))
    monkeypatch.setattr(runtime, "_HEARD", runtime.collections.OrderedDict())
    runtime.hear("f", "d", [("take", "a")], language="English")
    runtime.hear("f", "d", [("take", "a")], language="English")
    runtime.hear("f", "d", [("take", "a")], language="English", low_vram=True)
    assert asked == ["a", "a"]


def test_word_times_are_kept_apart_by_the_shape_too(monkeypatch):
    from yue2_comfy.asr import aligner

    measured = []
    fake_model = types.SimpleNamespace(mono_16k=lambda waveform, rate: waveform)
    monkeypatch.setitem(sys.modules, "yue2_comfy.asr.model", fake_model)
    monkeypatch.setattr(sys.modules["yue2_comfy.asr"], "model", fake_model, raising=False)
    monkeypatch.setattr(runtime, "acquire_aligner", lambda folder, reader, device, progress=None, low_vram=False:
                        measured.append(low_vram) or "held")
    monkeypatch.setattr(aligner, "align", lambda held, audio, text, cancelled=None, progress=None: [("a", 0.0, 0.4)])
    monkeypatch.setattr(runtime, "stamp", lambda folder: ("weights",))
    monkeypatch.setattr(runtime, "_TIMES", runtime.collections.OrderedDict())
    for low_vram in (False, False, True, True):
        runtime.word_times("f", "r", "d", "sound", 16000, "a", "song", low_vram=low_vram)
    assert measured == [False, True]
