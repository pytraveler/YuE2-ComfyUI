# Licenses and provenance

This pack carries four kinds of material under four different terms.

## The pack's own code

Apache-2.0. See LICENSE.

`yue2_comfy/gguf_meta.py`, `yue2_comfy/chat_template.py` and
`yue2_comfy/progress.py` were written for
[MiniMax-H3-Prompt-Rewriter-ComfyUI](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI)
by the same author and carried over. Same licence, same terms.

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

## The writer's language model

**Apache-2.0**, and nothing to do with the terms above.

`YuE2 Write Song` runs a separate instruction-following model to write the
style line and the lyrics. It uses whichever GGUF is already in your ComfyUI
model folders; when there is none it downloads `Qwen3.5-4B-Q4_K_M.gguf` from
[unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF), a
quantisation of Alibaba's Qwen3.5-4B. As with the YuE2 weights, this pack
downloads it and does not redistribute it.

A model you supply yourself carries whatever licence it came with, which this
pack neither reads nor enforces.
