"""Optional tests on real decoder files; they never download models.

Set YUE2_TEST_VAE_ROOT to a folder holding YuE2-Vae/model.safetensors,
YuE2-Vae-legacy/model.safetensors and VAE-RS/YuE2-VAE-RS-x3_v{1,2}.safetensors.
Set YUE2_TEST_VAE_REFERENCE to the kit's yue2_vae_rs_reference.npz.
YUE2_TEST_VAE_DEVICE defaults to cuda. Ordinary CI skips this module.
"""

import gc
import os
from pathlib import Path
import types

import pytest

ROOT = os.environ.get("YUE2_TEST_VAE_ROOT")
REFERENCE = os.environ.get("YUE2_TEST_VAE_REFERENCE")
pytestmark = pytest.mark.skipif(not ROOT or not REFERENCE,
                                reason="real decoder files and the published reference are opt-in")


@pytest.fixture(scope="module", params=("standard", "legacy", "rs_x3_v1", "rs_x3_v2"))
def real_decoder(request):
    torch = pytest.importorskip("torch")
    from yue2_comfy import constants, loader

    device = os.environ.get("YUE2_TEST_VAE_DEVICE", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; explicitly choose cpu to run the slower reference checks")
    variant = request.param
    if variant in constants.VAE_RS_VARIANTS:
        path = Path(ROOT) / "VAE-RS" / constants.VAE_RS_VARIANTS[variant][0]
    else:
        path = Path(ROOT) / ("YuE2-Vae-legacy" if variant == "legacy" else "YuE2-Vae") / "model.safetensors"
    model = loader.load_vae(str(path), variant)
    models = types.SimpleNamespace(vae=model, device=torch.device(device), lm=None,
                                    low_vram=False, offload="on", variant=variant)
    yield models
    model.to("cpu")
    del models, model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def reference_latents():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    with np.load(REFERENCE) as reference:
        return torch.from_numpy(reference["latents"].copy())


def test_rs_matches_the_publishers_reference_including_both_tile_seams(real_decoder, reference_latents):
    np = pytest.importorskip("numpy")
    from yue2_comfy import generate, runtime

    if not real_decoder.variant.startswith("rs_x3_"):
        pytest.skip("the reference uses a different output-padding convention for Standard")
    with runtime.deterministic_math():
        try:
            real_decoder.vae.to(real_decoder.device)
            audio = generate._decode_tiled(real_decoder.vae, reference_latents, 750, None, None)
        finally:
            real_decoder.vae.to("cpu")
    with np.load(REFERENCE) as reference:
        excerpts = np.stack([audio[0, :, int(offset):int(offset) + 4096].numpy()
                             for offset in reference["offsets"]])
        branch = real_decoder.variant.rsplit("_", 1)[1]
        error = float(np.abs(excerpts - reference[branch + "_excerpts"]).max())
    assert error < 1e-4, error


@pytest.mark.parametrize("low_vram", [False, True])
def test_tiled_decode_matches_a_whole_decode_and_parks_on_cpu(real_decoder, reference_latents, low_vram):
    torch = pytest.importorskip("torch")
    from yue2_comfy import generate, runtime

    models = real_decoder
    models.low_vram = low_vram
    with runtime.deterministic_math():
        models.vae.to(models.device)
        whole = models.vae.decode(reference_latents).cpu().clone().float().clamp_(-1, 1)
        audio, timing = generate.decode(models, reference_latents[0].T.contiguous())
    assert audio.shape == whole.shape
    assert audio.dtype == torch.float32 and audio.device.type == "cpu"
    assert not audio.is_inference()
    assert all(parameter.device.type == "cpu" for parameter in models.vae.parameters())
    assert timing["decode"]["core_frames"] == (256 if low_vram else 1024)
    assert float((audio - whole).abs().max()) < 1e-4


def test_long_songs_keep_the_correct_tail_length(real_decoder):
    torch = pytest.importorskip("torch")
    from yue2_comfy import generate, runtime

    models = real_decoder
    models.low_vram = False
    latent = torch.randn((5500, 64), generator=torch.Generator().manual_seed(21))
    with runtime.deterministic_math():
        audio, _timing = generate.decode(models, latent)
    tail = 0 if models.variant.startswith("rs_x3_") else 64
    assert audio.shape == (1, 2, 5500 * 1920 - tail)
    assert torch.isfinite(audio).all()


@pytest.mark.parametrize("start,stop,count", [(350,380,30), (350,420,0), (0,30,40),
                                              (850,900,50), (900,900,80)])
def test_real_edit_windows_match_a_whole_decode(real_decoder, start, stop, count):
    torch = pytest.importorskip("torch")
    from yue2_comfy import generate, runtime
    from yue2_comfy.inpaint import core, ops

    models = real_decoder
    models.low_vram = False
    latent = torch.randn((900, 64), generator=torch.Generator().manual_seed(15))
    added = torch.randn((count, 64), generator=torch.Generator().manual_seed(77))
    edited = torch.cat([latent[:start], added, latent[stop:]])
    with runtime.deterministic_math():
        old, _ = generate.decode(models, latent)
        result, _ = core.splice(models, old[0], edited, ops.Region(start,stop,count,0), count)
        whole, _ = generate.decode(models, edited)
    assert result.shape == whole.shape
    assert float((result - whole).abs().max()) < 1e-4


def test_cancellation_parks_the_decoder_and_restores_math_flags(real_decoder, reference_latents):
    torch = pytest.importorskip("torch")
    from yue2_comfy import generate, runtime

    before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    with pytest.raises(InterruptedError), runtime.deterministic_math():
        generate.decode(real_decoder, reference_latents[0].T.contiguous(), cancelled=lambda: True)
    assert all(parameter.device.type == "cpu" for parameter in real_decoder.vae.parameters())
    assert (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) == before
