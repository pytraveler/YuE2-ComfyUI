# Vendored upstream code -- do not edit

`yue2/` is a verbatim copy of `src/yue2/` from
https://github.com/multimodal-art-projection/YuE (Apache-2.0), version 0.1.6.

Keeping it unmodified is the point: updating means copying the new files over,
and every diff stays about what upstream changed.

## What is here

    protocol.py          prompting, sampling defaults, token layout
    modeling_yue2.py     the AR/NAR mixture-of-transformers backbone
    modeling_vae.py      the Oobleck VAE that turns latents into audio
    tokenization_yue2.py the text and score BPE
    sampling.py          the autoregressive decode loop
    nar.py               acoustic flow matching
    cuda_graph.py        pure-PyTorch CUDA graph decode

## What is not, and why

    cli.py          command line entry point; not used, and its example lyrics
                    are the only other file with non-ASCII content
    fast.py         the vLLM backend; fcntl, os.killpg and SIGKILL are POSIX
                    only and this pack does not offer that backend
    pipeline.py     YuE2Pipeline; it sets six torch globals process-wide, which
                    inside ComfyUI would degrade every other model until
                    restart. This pack orchestrates the four stages itself and
                    scopes those flags.
    storage.py      HF resolution and artifact hashing; this pack does its own
    progress.py     stderr progress; replaced by the ComfyUI node progress
    quantization.py opt-in FP8; not offered

`tokenization_yue2.py` imports `storage` inside `from_pretrained`, which this
pack never calls.

## The ASCII rule does not apply here

This project's own source is ASCII only. `modeling_yue2.py` contains 833
characters above U+007E -- section banners drawn with U+2500 and U+2550, a few
dashes and arrows, all in comments. Scrubbing them would mean maintaining a
fork and making every future upstream diff noisy, for no runtime benefit. The
ASCII check skips this directory.
