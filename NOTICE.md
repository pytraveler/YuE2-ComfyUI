# Licenses and provenance

This pack carries eight kinds of material under their own terms.

## The pack's own code

Apache-2.0. See LICENSE.

`yue2_comfy/gguf_meta.py`, `yue2_comfy/chat_template.py`,
`yue2_comfy/progress.py`, `yue2_comfy/llamacpp.py`, `yue2_comfy/child.py` and
`yue2_comfy/cli.py` were written for
[MiniMax-H3-Prompt-Rewriter-ComfyUI](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI)
by the same author and carried over. Same licence, same terms.

`yue2_comfy/sheetsage/` and `yue2_comfy/asr/` are this pack's own
implementations of SheetSage2 and Qwen3-ASR, under the same terms. ComfyUI's
SheetSage2 code is GPL-3.0 and m-a-p's reference code carries no licence, so
neither is copied here; the transcriptions are checked against ComfyUI master's
instead, and match it to the byte on the pack's own songs. The speech model's
log-mel front end follows the arithmetic of OpenAI's Whisper (MIT) and of
Hugging Face's feature extractor (Apache-2.0), as Qwen3-ASR's own release does.

`yue2_comfy/vocals/` is this pack's own implementation of Mel-Band RoFormer
(Wang, Lu and Won, 2023), under the same terms. Its shapes and arithmetic follow
[lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) at commit
`93a07dd` (Copyright (c) 2023 Phil Wang, MIT), the implementation the vocal
model was trained with, and its output was checked against that code to the
last bit in float32; nothing of it is copied. The mel band plan follows the
arithmetic of librosa's `filters.mel` (ISC), the rotary positions that of
rotary-embedding-torch (MIT), and the chunking the settings published with the
model: eight-second windows overlapping by half.

## Vendored YuE2 code

`yue2_comfy/vendor/yue2/` is a verbatim copy of part of
[multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE),
Apache-2.0, from `src/yue2/`. It is not modified. See
`yue2_comfy/vendor/README.md` for the file list and the reason each omitted file
was left out.

`yue2_comfy/vendor/yue2_music/abc_tools.py` is a verbatim copy of
`skills/yue2-music/scripts/abc_tools.py` from the same repository, which ships
that skill with its own copy of the same Apache-2.0 licence. It reads the ABC
scores that `transpose` moves, and it is not modified either.

`yue2_comfy/notation.py`, the score editor's reader and writer, uses the same
file as the judge of what every token of a score means.

The VAE implementation inside that code is itself derived from
stable-audio-tools (Copyright (c) 2023 Stability AI, MIT) and from BigVGAN's
SnakeBeta (Copyright (c) 2022 NVIDIA CORPORATION, MIT). Those notices travel
with the model repository and are reproduced in the upstream
`THIRD_PARTY_NOTICES.md`.

## abcjs

**MIT.** Copyright (c) 2009-2026 Paul Rosen and Gregory Dyke.

`web/js/vendor/abcjs-basic-min.cjs` is `dist/abcjs-basic-min.js` from the npm
package [abcjs](https://github.com/paulrosen/abcjs) 6.7.0, unmodified, with its
licence beside it in `web/js/vendor/abcjs-LICENSE.md`. The score editor uses it
to draw a score as sheet music. Why the file does not end in `.js` is written
in `web/js/vendor/README.md`.

## The model weights

**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).**

The YuE2-3B, YuE2-Vae and YuE2-Vae-legacy checkpoints are non-commercial. This
pack downloads them but does not redistribute them, and it does not relicense
them. If you intend to use the output commercially, read the model license at
https://huggingface.co/m-a-p/YuE2-3B before you do.

SheetSage2's weights are under the same CC BY-NC 4.0. `YuE2 Transcribe`
downloads them as `sheetsage2_bf16.safetensors` from
[Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2), a single-file repack of
[m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2), and likewise does
not redistribute them.

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

## The speech model

**Apache-2.0.**

With `lyrics_auto_recognition` on, `YuE2 Transcribe` recognises the sung words
with Qwen3-ASR-1.7B from
[Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf), which
it downloads on first use and does not redistribute. The pack runs it with its
own code rather than with Qwen's package or transformers.

## The voice separator

**MIT.**

With `vocals_only` on, and in `YuE2 Vocals Only`, the voice is separated with
Kimberley Jensen's Mel-Band RoFormer vocal model from
[KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer),
which the pack downloads on first use, pinned to one revision, and does not
redistribute. A conversion of the same weights already on the machine, such as
kijai's `MelBandRoformer_fp16.safetensors`, is used in its place; that file is
the user's own and carries whatever terms it came with.

## The llama.cpp binaries

**MIT.** Copyright (c) 2023-2024 The ggml authors.

When `llama-cpp-python` is not installed, `YuE2 Write Song` downloads an
official release archive of [llama.cpp](https://github.com/ggml-org/llama.cpp)
and runs the writer model through it. Nothing from that archive is redistributed
here: the pack fetches it from the upstream release page at run time, pinned to
one tag, and unpacks it into ComfyUI's user folder. An llama.cpp you built or
installed yourself is used as it is and nothing is downloaded at all.
