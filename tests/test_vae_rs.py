"""Selection of optional decoders never changes the existing model choices."""

import hashlib
import json
import struct
import types
import pytest

from yue2_comfy import checkpoints, constants, discovery, download, loader, paths, vae_rs


def rs_header(variant="rs_x3_v1"):
    """A header-only fixture; no real weights are needed for discovery."""
    header = {name: {"dtype": "F32", "shape": shape, "data_offsets": [0, 4]}
              for name, shape in vae_rs.SIGNATURES.items()}
    header.update({"unused.{}".format(index): {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
                   for index in range(427 - len(header))})
    header["__metadata__"] = {"variant": "x3", "step": constants.VAE_RS_VARIANTS[variant][1]}
    return header


def write_rs(path, variant="rs_x3_v1", header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(rs_header(variant) if header is None else header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(body)) + body)
    return str(path)


def no_download(*args, **kwargs):
    raise AssertionError("No download was expected")


def test_existing_variants_and_defaults_keep_their_positions():
    assert constants.VAE_CHOICES[:2] == ("standard", "legacy")
    assert constants.DEFAULT_OPTIONS["vae"] == "standard"


@pytest.mark.parametrize("variant", constants.VAE_RS_VARIANTS)
def test_a_renamed_rs_decoder_is_found_by_its_shapes_and_training_step(tmp_path, variant):
    path = write_rs(tmp_path / "vae" / "renamed.safetensors", variant)
    assert vae_rs.find([str(tmp_path)], variant) == path
    assert discovery.identify(path) == ""


def test_rs_v2_never_satisfies_a_request_for_v1(tmp_path):
    write_rs(tmp_path / "renamed.safetensors", "rs_x3_v2")
    assert vae_rs.find([str(tmp_path)], "rs_x3_v1") == ""


def test_a_wrong_file_under_the_published_name_is_refused(tmp_path):
    filename = constants.VAE_RS_VARIANTS["rs_x3_v1"][0]
    write_rs(tmp_path / filename, "rs_x3_v2")
    with pytest.raises(ValueError, match="rs_x3_v1"):
        vae_rs.find([str(tmp_path)], "rs_x3_v1")


@pytest.mark.parametrize("fault", ["half", "missing", "shape"])
def test_incompatible_rs_files_fail_before_a_large_load(tmp_path, fault):
    header = rs_header()
    if fault == "half":
        header["unused.0"]["dtype"] = "BF16"
    elif fault == "missing":
        del header["unused.0"]
    else:
        header[next(iter(vae_rs.SIGNATURES))]["shape"] = [1024, 64, 7]
    path = write_rs(tmp_path / "bad.safetensors", header=header)
    with pytest.raises(ValueError):
        vae_rs.validate(path, "rs_x3_v1")


def test_a_missing_external_decoder_does_not_use_the_embedded_standard_one(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "search_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(discovery, "find_repack", lambda *args: "backbone.safetensors")
    with pytest.raises(discovery.Missing) as error:
        discovery.locate("rs_x3_v1")
    assert error.value.backbone
    assert error.value.repack == "backbone.safetensors"
    assert constants.VAE_RS_VARIANTS["rs_x3_v1"][0] in str(error.value)


@pytest.mark.parametrize("kind", ["lm", "repack", "repack_int8"])
def test_an_explicit_checkpoint_keeps_its_backbone_with_rs(tmp_path, monkeypatch, kind):
    backbone = tmp_path / "chosen.safetensors"
    backbone.write_bytes(b"this test replaces backbone identification")
    (tmp_path / constants.MERGES_NAME).write_bytes(b"vocabulary")
    decoder = write_rs(tmp_path / "vae" / "renamed.safetensors")
    monkeypatch.setattr(checkpoints, "catalogue", lambda: {"chosen": str(backbone)})
    monkeypatch.setattr(paths, "search_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(discovery, "_identify_by_header", lambda path: kind)
    monkeypatch.setattr(download, "fetch_decoder", no_download)
    settings = dict(constants.DEFAULT_OPTIONS, checkpoint="chosen", vae="rs_x3_v1", download="off")
    files = download.ensure(settings)
    assert files.vae == decoder
    assert (files.lm if kind == "lm" else files.repack) == str(backbone)


def test_a_saved_rs_plan_preserves_its_checkpoint_and_decoder(tmp_path, monkeypatch):
    backbone = tmp_path / "chosen.safetensors"
    backbone.write_bytes(b"backbone")
    decoder = write_rs(tmp_path / "renamed.safetensors")
    files = discovery.Files(repack=str(backbone), vae=decoder)
    settings = dict(constants.DEFAULT_OPTIONS, checkpoint="chosen", vae="rs_x3_v1")
    checkpoints.remember(settings, files)
    monkeypatch.setattr(checkpoints, "roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(checkpoints, "catalogue", no_download)
    assert download.ensure(settings) == files
    backbone.write_bytes(b"different backbone")
    with pytest.raises(ValueError, match="missing or changed"):
        download.ensure(settings)


def test_decode_metadata_keeps_the_singing_backbone_when_options_name_another_one():
    original = {"name": "sung.safetensors", "choice": "original", "vae_variant": "standard",
                "files": {"lm": "sung.safetensors", "merges": "words", "vae": "old"},
                "stamps": {"lm": [12, 1], "vae": [24, 1]}}
    resolved = {"name": "other.safetensors", "files": {"lm": "other.safetensors", "vae": "rs"},
                "stamps": {"vae": [803071224, 7]}}
    source = checkpoints.decoded_source(original, resolved, "rs_x3_v1")
    assert source["name"] == "sung.safetensors"
    assert source["choice"] == "original"
    assert source["files"] == {"lm": "sung.safetensors", "merges": "words", "vae": "rs"}
    assert source["stamps"]["lm"] == [12, 1]
    assert source["stamps"]["vae"] == [803071224, 7]
    assert source["vae_variant"] == "rs_x3_v1"
    assert original["files"]["vae"] == "old"


def test_a_repack_cannot_be_silently_loaded_as_rs(tmp_path):
    with pytest.raises(ValueError, match="does not contain"):
        loader.load_repack(str(tmp_path / "absent.safetensors"), "cpu", "rs_x3_v1")


def test_switching_variants_never_reuses_a_different_decoder(tmp_path, monkeypatch):
    files = {}
    for name in ("lm", "vae", "merges"):
        path = tmp_path / name
        path.write_bytes(b"model identity")
        files[name] = str(path)
    files = discovery.Files(**files)
    loaded = []
    monkeypatch.setattr(loader, "_STATE", {"key": None, "lm": None, "vae": None, "tokenizer": None})
    monkeypatch.setattr(loader.devices, "resolve", lambda spec: "cpu")
    monkeypatch.setattr(loader, "_free_comfy_vram", lambda *args: None)
    monkeypatch.setattr(loader, "_empty_cache", lambda: None)
    monkeypatch.setattr(loader, "load_lm", lambda *args: object())
    monkeypatch.setattr(loader, "load_tokenizer", lambda *args: object())

    def load_vae(path, variant):
        loaded.append(variant)
        return types.SimpleNamespace(variant=variant)

    monkeypatch.setattr(loader, "load_vae", load_vae)
    for variant in ("standard", "rs_x3_v1", "rs_x3_v2", "legacy", "standard"):
        first = loader.acquire(files, "cpu", variant)
        again = loader.acquire(files, "cpu", variant)
        assert again.vae is first.vae
        assert first.vae.variant == variant
    assert loaded == ["standard", "rs_x3_v1", "rs_x3_v2", "legacy", "standard"]


def test_an_existing_backbone_downloads_only_the_missing_rs_decoder(monkeypatch):
    attempts = []
    calls = []

    def locate(variant, quantization):
        attempts.append(variant)
        if len(attempts) == 1:
            raise discovery.Missing("missing RS", repack="existing backbone")
        return discovery.Files(repack="existing backbone", vae="downloaded RS")

    monkeypatch.setattr(discovery, "locate", locate)
    monkeypatch.setattr(download, "fetch_decoder", lambda variant, progress: calls.append(variant))
    monkeypatch.setattr(download, "fetch_repack", no_download)
    monkeypatch.setattr(download, "fetch_original", no_download)
    result = download.ensure({"vae": "rs_x3_v1", "download": "comfy-org"})
    assert result.vae == "downloaded RS"
    assert calls == ["rs_x3_v1"]


def test_rs_downloads_are_pinned_and_checked(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(paths, "models_root", lambda: str(tmp_path))
    monkeypatch.setattr(download, "fetch", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(vae_rs, "check_download", lambda *args: calls.append(args))
    download.fetch_decoder("rs_x3_v2")
    args, kwargs = calls[0]
    assert args[0] == constants.VAE_RS_REPO
    assert kwargs["revision"] == constants.VAE_RS_REVISION
    assert next(iter(args[1])).endswith(constants.VAE_RS_VARIANTS["rs_x3_v2"][0])
    assert calls[1][1] == "rs_x3_v2"


def test_a_corrupt_download_is_refused_without_loading_it(tmp_path, monkeypatch):
    path = tmp_path / "wrong.safetensors"
    path.write_bytes(b"not the published file")
    monkeypatch.setattr(vae_rs, "validate", no_download)
    with pytest.raises(ValueError, match="SHA256"):
        vae_rs.check_download(str(path), "rs_x3_v1")
    with pytest.raises(InterruptedError):
        vae_rs.check_download(str(path), "rs_x3_v1", cancelled=lambda: True)


def test_an_unchanged_download_is_checked_before_use(tmp_path, monkeypatch):
    path = tmp_path / "decoder.safetensors"
    path.write_bytes(b"test release")
    row = (path.name, "161953", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setitem(vae_rs.VAE_RS_VARIANTS, "test_release", row)
    validated = []
    monkeypatch.setattr(vae_rs, "validate", lambda *args: validated.append(args))
    vae_rs.check_download(str(path), "test_release")
    assert validated == [(str(path), "test_release")]


def test_the_adapter_supplies_edit_context_without_replacing_the_architecture_config(monkeypatch):
    torch = pytest.importorskip("torch")
    from yue2_comfy.vendor.vae_rs import yue2_vae_rs as upstream

    calls = []

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = {"pre": 3}

        def required_halo(self, core=None):
            return 60

        def decode_tiled(self, latent, **kwargs):
            calls.append(kwargs)
            return latent

    inner = Inner()
    monkeypatch.setattr(vae_rs, "validate", lambda *args: None)
    monkeypatch.setattr(upstream, "load_decoder", lambda *args, **kwargs: inner)
    adapted = vae_rs.load("fixture.safetensors", "rs_x3_v1")
    assert inner.config == {"pre": 3}
    assert adapted.config.decode_halo_frames == 60
    assert adapted.config.downsampling_ratio == 1920
    adapted.decode_tiled(torch.ones(1, 64, 1), core_frames=256, halo_frames=16)
    assert calls[0]["halo_frames"] == 60
    assert calls[0]["core_frames"] == 256
    assert not next(adapted.parameters()).requires_grad


@pytest.mark.parametrize("low_vram", [False, True])
def test_rs_oom_fallback_keeps_the_required_halo(monkeypatch, low_vram):
    torch = pytest.importorskip("torch")
    from yue2_comfy import generate

    calls = []

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = types.SimpleNamespace(decode_core_frames=1024, decode_halo_frames=60)

        def decode_tiled(self, latent, **kwargs):
            calls.append((kwargs["core_frames"], kwargs["halo_frames"]))
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError("simulated tight card")
            return torch.zeros(1, 2, latent.shape[-1] * 1920)

    decoder = Decoder()
    models = types.SimpleNamespace(vae=decoder, lm=None, device=torch.device("cpu"), low_vram=low_vram)
    audio, _timing = generate.decode(models, torch.zeros(4, 64))
    assert calls == ([(256, 60), (64, 60)] if low_vram else [(1024, 60), (512, 60)])
    assert audio.shape == (1, 2, 4 * 1920)
    assert next(decoder.parameters()).device.type == "cpu"
