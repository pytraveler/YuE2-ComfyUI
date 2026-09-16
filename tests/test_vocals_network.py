"""Mel-Band RoFormer's network and its chunking, on the CPU with made-up weights.

That the numbers match the reference implementation was measured on the real
weights (bit for bit in float32) and cannot be checked without them; what can
be checked here is that the weights would load, that the shapes flow, and that
cutting a recording into windows and putting it back loses nothing.
"""

from __future__ import annotations

import json
import pathlib

import pytest

torch = pytest.importorskip("torch")

from yue2_comfy.vocals import model, network  # noqa: E402

KEYS = json.loads((pathlib.Path(__file__).resolve().parent / "data" / "melbandroformer_keys.json")
                  .read_text(encoding="ascii"))


class Passthrough(torch.nn.Module):
    """A network that hears every chunk as all voice, so the pieces put back together must be the input."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, audio):
        return audio.clone()


def test_the_network_has_every_tensor_of_the_released_weights_by_name_and_shape():
    with torch.device("meta"):
        net = network.Network()
    assert {name: list(tensor.shape) for name, tensor in net.state_dict().items()} == KEYS["tensors"]


@pytest.mark.parametrize("seconds", [0.3, 5.0, 12.0, 21.3])
def test_windows_put_back_together_give_the_recording_back(seconds):
    """Short, under a window, past the mirrored edges, and over several windows."""
    audio = torch.randn(2, int(seconds * 44100), generator=torch.Generator().manual_seed(3))
    out = model.demix(Passthrough(), audio)
    assert out.shape == audio.shape
    assert torch.allclose(out, audio, atol=1e-5)


def test_a_cancel_between_windows_stops_the_separation():
    audio = torch.zeros(2, 20 * 44100)
    with pytest.raises(InterruptedError):
        model.demix(Passthrough(), audio, cancelled=lambda: True)


def test_progress_reaches_the_whole_after_the_last_window():
    seen = []
    model.demix(Passthrough(), torch.zeros(2, 20 * 44100), progress=seen.append)
    assert seen[-1] == 1.0 and seen == sorted(seen)


@pytest.mark.parametrize("channels", [1, 2])
def test_the_voice_comes_back_in_the_shape_and_rate_it_was_given(channels):
    waveform = torch.randn(2, channels, 48000 * 3 + 7, generator=torch.Generator().manual_seed(5)) * 0.1
    out = model.separate(Passthrough(), waveform, 48000)
    assert out.shape == waveform.shape
    assert out.dtype == torch.float32
    kept = torch.nn.functional.cosine_similarity(out.flatten(), waveform.flatten(), dim=0)
    assert kept > 0.9


def test_more_than_two_channels_are_refused():
    with pytest.raises(ValueError):
        model.separate(Passthrough(), torch.zeros(1, 3, 1000), 44100)


def test_empty_audio_comes_back_empty():
    out = model.separate(Passthrough(), torch.zeros(1, 2, 0), 48000)
    assert out.shape == (1, 2, 0)


def test_weights_that_do_not_fit_are_refused_by_name(tmp_path):
    path = tmp_path / "wrong.ckpt"
    torch.save({"band_split.to_features.0.0.gamma": torch.ones(3)}, path)
    with pytest.raises(ValueError) as refused:
        model.load(str(path), torch.device("cpu"))
    assert "do not fit" in str(refused.value)


def test_a_small_network_keeps_the_shape_of_the_audio(monkeypatch):
    """The same wiring at a toy width: every reshape between bands, frames and heads has to line up."""
    for name, value in (("WIDTH", 16), ("LAYERS", 1), ("HEADS", 2), ("HEAD", 8), ("FEED_FORWARD", 32),
                        ("MASK_HIDDEN", 32)):
        monkeypatch.setattr(network, name, value)
    torch.manual_seed(0)
    net = network.Network().eval()
    audio = torch.randn(1, 2, 11025) * 0.1
    with torch.inference_mode():
        out = net(audio)
    assert out.shape == audio.shape
    assert torch.isfinite(out).all()
