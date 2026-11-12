"""Optional Ruach Studio decoders, outside the standard and legacy paths.

The vendored module keeps its architecture config as a dict. An adapter adds
the small runtime config the pack's existing decode and edit code reads;
neither the old decoder nor its boundary calculations need to change.
"""

from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

from . import paths
from .constants import (
    VAE_RS_DIRNAME, VAE_RS_REPO, VAE_RS_REVISION, VAE_RS_VARIANTS,
)

SIGNATURES = {
    "conv1.parametrizations.weight.original1": [2048, 64, 7],
    "pre.2.conv1.parametrizations.weight.original1": [2048, 2048, 7],
    "blocks.5.units.5.conv1.parametrizations.weight.original1": [64, 64, 7],
    "conv2.parametrizations.weight.original1": [2, 64, 7],
}


def _header(path: str) -> dict:
    from .loader import read_header

    return read_header(path)


def variant_of(header: dict) -> str:
    """The published training branch, with architecture markers as a guard."""
    if not all((header.get(key) or {}).get("shape") == shape
               for key, shape in SIGNATURES.items()):
        return ""
    metadata = header.get("__metadata__") or {}
    if metadata.get("variant", "x3") != "x3":
        return ""
    step = str(metadata.get("step", ""))
    return next((name for name, (_file, expected, _sha) in VAE_RS_VARIANTS.items()
                 if step == expected), "")


def validate(path: str, variant: str) -> None:
    """A named file must be this release, never another VAE under its name."""
    header = _header(path)
    if variant_of(header) != variant:
        raise ValueError("The selected file is not the " + variant + " YuE2 VAE-RS release: "
                         + os.path.basename(path))
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    if len(tensors) != 427 or any(value.get("dtype") != "F32" for value in tensors):
        raise ValueError("YuE2 VAE-RS needs the released 427-tensor FP32 file: "
                         + os.path.basename(path))


def find(roots: list, variant: str) -> str:
    """Find a loose or renamed decoder within the same bounded model roots."""
    from .discovery import _homes

    filename = VAE_RS_VARIANTS[variant][0]
    homes = [home for root in roots for home in _homes(root)]
    for home in homes:
        candidate = os.path.join(home, filename)
        if os.path.isfile(candidate):
            validate(candidate, variant)
            return candidate
    for home in homes:
        try:
            entries = sorted(os.listdir(home))
        except OSError:
            continue
        for entry in entries:
            if not entry.lower().endswith(".safetensors"):
                continue
            candidate = os.path.join(home, entry)
            try:
                found = variant_of(_header(candidate))
            except (OSError, ValueError):
                continue
            if found == variant:
                validate(candidate, variant)
                return candidate
    return ""


def layout(roots: list, variant: str) -> dict:
    from . import discovery

    found = discovery._layout_pass(roots, "standard")
    found["vae"] = find(roots, variant)
    return found


def sweep(roots: list, found: dict, variant: str) -> None:
    """Ask the ordinary sweep only for the backbone and vocabulary."""
    from . import discovery

    backbone = dict(found, vae="not needed")
    discovery._sweep(roots, backbone, "standard")
    found.update(lm=backbone["lm"], merges=backbone["merges"])
    if not found["vae"]:
        found["vae"] = find(roots, variant)


def wanted(variant: str) -> dict:
    filename = VAE_RS_VARIANTS[variant][0]
    return {VAE_RS_DIRNAME + "/" + filename:
            os.path.join(paths.models_root(), VAE_RS_DIRNAME, filename)}


def locate(variant: str, quantization: str):
    """A normal YuE2 backbone beside an explicitly selected external decoder."""
    from . import discovery

    roots = paths.search_roots()
    found = layout(roots, variant)
    sweep(roots, found, variant)
    if quantization != "int8" and all(found.values()):
        return discovery.Files(**found)
    repacked = discovery.find_repack(roots, quantization)
    prefer_int8 = (quantization == "int8" and repacked
                   and discovery.identify(repacked) == "repack_int8")
    if found["vae"]:
        if prefer_int8:
            return discovery.Files(repack=repacked, vae=found["vae"])
        if all(found.values()):
            return discovery.Files(**found)
        if repacked:
            return discovery.Files(repack=repacked, vae=found["vae"])
    lines = ["The selected YuE2 decoder (" + variant + ") or its supporting files are missing."]
    if not found["vae"]:
        for remote, local in wanted(variant).items():
            lines += ["https://huggingface.co/" + VAE_RS_REPO + "/resolve/"
                      + VAE_RS_REVISION + "/" + remote, "  -> " + local]
    if not repacked and not (found["lm"] and found["merges"]):
        lines.append("A YuE2 backbone and its vocabulary are also needed. Select a local "
                     "checkpoint or enable downloads in YuE2 Options.")
    raise discovery.Missing("\n".join(lines), found, repacked)


def check_download(path: str, variant: str, cancelled=None) -> None:
    """Verify the published digest once, after a download of this release."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled while checking the VAE-RS download")
            digest.update(block)
    if digest.hexdigest() != VAE_RS_VARIANTS[variant][2]:
        raise ValueError("The YuE2 VAE-RS download does not match its published SHA256: "
                         + os.path.basename(path))
    validate(path, variant)


def load(path: str, variant: str):
    """Load only the optional decoder and adapt its runtime configuration."""
    import torch

    from .vendor.vae_rs.yue2_vae_rs import load_decoder

    validate(path, variant)
    decoder = load_decoder(path, device="cpu", allow_standard=False)

    class Decoder(torch.nn.Module):
        """The upstream arithmetic and a config compatible with the pack."""

        def __init__(self, inner):
            super().__init__()
            self.decoder = inner
            self.config = SimpleNamespace(
                decode_core_frames=1024,
                decode_halo_frames=max(16, *(inner.required_halo(core)
                                            for core in (64, 256, 512, 750, 1024))),
                downsampling_ratio=1920,
                release_variant=variant,
            )

        def required_halo(self, core_frames=None):
            return self.decoder.required_halo(core_frames)

        def natural_output_length(self, frames):
            return self.decoder.natural_output_length(frames)

        def decode(self, latent):
            return self.decoder.decode(latent)

        def decode_tiled(self, latent, core_frames=1024, halo_frames=None,
                         output_device="cpu", on_progress=None):
            required = self.required_halo(core_frames)
            halo = required if halo_frames is None else max(required, int(halo_frames))
            return self.decoder.decode_tiled(latent, core_frames=core_frames, halo_frames=halo,
                                             output_device=output_device, on_progress=on_progress)

    return Decoder(decoder).eval().requires_grad_(False)
