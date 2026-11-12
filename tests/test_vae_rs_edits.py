"""Edit windows must cover a sixty-frame decoder and both output lengths."""

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from yue2_comfy import generate  # noqa: E402
from yue2_comfy.inpaint import core, ops  # noqa: E402

HOP = 1920
HALO = 60


def decoded(latents, tail):
    """A finite receptive field of sixty frames either side, with no model download."""
    values = latents[:, :2].T.unsqueeze(0)
    kernel = torch.ones(2, 1, 2 * HALO + 1) / (2 * HALO + 1)
    audio = torch.nn.functional.conv1d(values, kernel, padding=HALO, groups=2)
    audio = audio.repeat_interleave(HOP, dim=-1)
    return audio if tail == 0 else audio[..., :-tail]


@pytest.mark.parametrize("tail", [0, 64])
@pytest.mark.parametrize("start,stop,count", [(300,330,30), (300,360,0), (300,300,40),
                                              (0,30,20), (660,700,40), (700,700,40)])
def test_wide_context_edits_match_the_whole_decode(monkeypatch, tail, start, stop, count):
    monkeypatch.setattr(generate, "decode", lambda models, latent, *args, **kwargs: (decoded(latent, tail), {}))
    models = types.SimpleNamespace(vae=types.SimpleNamespace(
        config=types.SimpleNamespace(downsampling_ratio=HOP, decode_halo_frames=HALO)))
    latents = torch.randn((700, 64), generator=torch.Generator().manual_seed(12))
    new = torch.ones(count, 64)
    edited = torch.cat([latents[:start], new, latents[stop:]])
    old = decoded(latents, tail)
    result, _timing = core.splice(models, old[0], edited, ops.Region(start, stop, count, 0), count)
    whole = decoded(edited, tail)
    assert result.shape == whole.shape
    assert torch.allclose(result, whole, atol=1e-6)


def test_the_previous_sixteen_frame_halo_would_fail_this_check(monkeypatch):
    monkeypatch.setattr(generate, "decode", lambda models, latent, *args, **kwargs: (decoded(latent, 0), {}))
    models = types.SimpleNamespace(vae=types.SimpleNamespace(
        config=types.SimpleNamespace(downsampling_ratio=HOP, decode_halo_frames=16)))
    latents = torch.zeros(700, 64)
    edited = latents.clone()
    edited[300:330] = 1
    result, _timing = core.splice(models, decoded(latents, 0)[0], edited, ops.Region(300,330,30,0), 30)
    assert float((result - decoded(edited, 0)).abs().max()) > 0.01


@pytest.mark.parametrize("tail", [0, 64])
def test_moving_a_region_repairs_the_full_receptive_field(monkeypatch, tail):
    monkeypatch.setattr(generate, "decode", lambda models, latent, *args, **kwargs: (decoded(latent, tail), {}))
    models = types.SimpleNamespace(vae=types.SimpleNamespace(
        config=types.SimpleNamespace(downsampling_ratio=HOP, decode_halo_frames=HALO)))
    latent = torch.randn((700, 64), generator=torch.Generator().manual_seed(3))
    old = decoded(latent, tail)
    pieces = [(0,200),(400,500),(200,400),(500,700)]
    moved = torch.cat([latent[first:last] for first,last in pieces])
    base = torch.cat([old[..., first*HOP:min(last*HOP,old.shape[-1])] for first,last in pieces], dim=-1)
    result, _timing = core.laid(models, base[0], moved, [(199,201),(299,301),(499,501)])
    assert torch.allclose(result, decoded(moved, tail), atol=1e-6)
