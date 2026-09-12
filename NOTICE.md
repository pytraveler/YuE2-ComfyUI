# Licenses and provenance

This pack carries three kinds of material under three different terms.

## The pack's own code

Apache-2.0. See LICENSE.

## Vendored YuE2 inference code

`yue2_comfy/vendor/yue2/` is a verbatim copy of part of
[multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE),
Apache-2.0, from `src/yue2/`. It is not modified. See
`yue2_comfy/vendor/README.md` for the file list and the reason each omitted file
was left out.

The VAE implementation inside that code is itself derived from
stable-audio-tools (Copyright (c) 2023 Stability AI, MIT) and from BigVGAN's
SnakeBeta (Copyright (c) 2022 NVIDIA CORPORATION, MIT). Those notices travel
with the model repository and are reproduced in the upstream
`THIRD_PARTY_NOTICES.md`.

## The model weights

**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).**

The YuE2-3B, YuE2-Vae and YuE2-Vae-legacy checkpoints are non-commercial. This
pack downloads them but does not redistribute them, and it does not relicense
them. If you intend to use the output commercially, read the model license at
https://huggingface.co/m-a-p/YuE2-3B before you do.

`qwen.tiktoken` is deliberately not shipped inside this pack for the same
reason: it comes from the same non-commercially licensed repository, and the
pack stays permissively licensed.
