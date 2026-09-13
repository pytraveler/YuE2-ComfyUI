# YuE2 for ComfyUI

ComfyUI nodes for [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B). A style
description and lyrics go in; a readable score and a complete 48 kHz stereo
song come out, generated entirely on your own machine.

[Russian version](README_RU.md) | [Changelog](CHANGELOG.md)

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-blue"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <a href="https://huggingface.co/m-a-p/YuE2-3B"><img alt="YuE2-3B" src="https://img.shields.io/badge/%F0%9F%A4%97-YuE2--3B-yellow"></a>
  <a href="https://huggingface.co/m-a-p/YuE2-Vae"><img alt="YuE2-Vae" src="https://img.shields.io/badge/%F0%9F%A4%97-YuE2--Vae-yellow"></a>
  <a href="NOTICE.md"><img alt="Weights: CC BY-NC 4.0" src="https://img.shields.io/badge/Weights-CC%20BY--NC%204.0-lightgrey"></a>
</p>

![YuE2 Generate Song in ComfyUI: on the left a text node with Russian lyrics tagged [Verse], [Chorus] and [Outro], wired into the lyrics input; in the middle the node with its seed, the Edit song... button and the song summary -- Russian, 76 BPM, male rap vocal and breathy female voice, then rap, heavy bass, hip hop beat, rhythmic punchy melody, staccato flow, and lyrics from manual song style -- above the caption 75 seconds of audio; on the right Save Audio with the finished 1:15 track](docs/node_song_gen.png)

*The style is set on the node and the lyrics arrive through a wire from a plain
text node; the summary shows both. 75 seconds of song, and the node ran for
40.2 s on an RTX 5090.*

```text
  style:  "warm piano pop, expressive female voice, 88 BPM"
  lyrics: "[Verse] Neon fades along the lane ..."
                            |
                            v
              ABC score  (the model plans the tune)
                            |
                            v
              semantic codec tokens  (the performance)
                            |
                            v
              acoustic latents  (flow matching, 32 steps)
                            |
                            v
              VAE decode -> 48 kHz stereo AUDIO
```

    [YuE2 Generate Song]
      style   (multiline)
      lyrics  (multiline)
      seed    (int)
      -> AUDIO
      -> score_abc (STRING)

That is the whole node. There is no model picker and no path to fill in: the
weights are found wherever this machine already keeps them. Everything else
lives in an optional `YuE2 Options` node that you do not have to place.

If you would rather not write the style line and the lyrics yourself, there is
a second node that writes both from one sentence:

    [YuE2 Write Song]
      idea  "a sad song about winter, female vocal"
      -> style   -> [YuE2 Generate Song]
      -> lyrics  ->

See [YuE2 Write Song](#yue2-write-song).

## Contents

- [What you need before installing](#what-you-need-before-installing)
- [Install](#install)
- [Example workflows](#example-workflows)
- [Nodes](#nodes)
  - [YuE2 Generate Song](#yue2-generate-song)
  - [YuE2 Write Song](#yue2-write-song)
  - [YuE2 Options](#yue2-options)
  - [Staged nodes](#staged-nodes)
- [Song length](#song-length)
- [Writing lyrics](#writing-lyrics)
  - [Stress, and what capital letters really do](#stress-and-what-capital-letters-really-do)
- [Reproducibility](#reproducibility)
- [Where the weights go](#where-the-weights-go)
  - [The INT8 build](#the-int8-build)
  - [Using files you already have](#using-files-you-already-have)
  - [Environment variables](#environment-variables)
- [Notes](#notes)
- [Licence](#licence)

## What you need before installing

| Resource | Requirement |
|---|---|
| GPU | An NVIDIA GPU with BF16 support. CPU works and is roughly an hour per song |
| VRAM | **~5 GiB** for a 40-second song and **~11.5 GiB** for a four-minute one, with only the half of the model each stage needs on the card. A card with room to spare keeps the whole model on it and uses more: 7.6 and 15 GiB. See `offload` in [YuE2 Options](#yue2-options) |
| Disk | **7.26 GB**, as one file or as three. The INT8 build is 3.69 GB. `YuE2 Write Song` adds 2.55 GB unless you already have a GGUF, plus 32 MB of llama.cpp binaries if `llama-cpp-python` is not installed |
| Packages | `tiktoken`, which ComfyUI does not ship. It is the only thing this pack adds; `requests`, which the downloader uses, is already in every ComfyUI install. `llama-cpp-python` is optional -- `YuE2 Write Song` uses it when it is there and official llama.cpp binaries when it is not |

If `tiktoken` is missing the node says so, with the right pip line for the
interpreter ComfyUI is actually running on -- which is not the one a bare
`pip install` would reach in a portable build.

## Install

Clone into `ComfyUI/custom_nodes/` and install the requirement into the same
Python environment ComfyUI runs on:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/pytraveler/YuE2-ComfyUI
```

Then restart ComfyUI. Python caches imports, so an edit to this pack is never
live until the server restarts.

## Example workflows

Six workflows ship with the pack and appear in ComfyUI's template browser
(*Workflow -> Browse Templates*) under this node pack's name once it is
installed. Each is a card of its own and each runs on its own: nothing is
bypassed on open, and there is no second branch to mute before pressing Run.

| | Template | What it needs |
|---|---|---|
| 1 | **A song from one line** -- one sentence about the song in, a finished song out | a 2.7 GB writer and the 7.26 GB YuE2 weights |
| 2 | **A song from style and lyrics** -- describe the sound, write the words, press Run | the YuE2 weights |
| 3 | **A longer song** -- `YuE2 Options` in front of the same node | the YuE2 weights |
| 4 | **The score first** -- write the score, read it, edit it, then sing it | the YuE2 weights |
| 5 | **Four scores, one song** -- four takes on the same words, chosen by ear | the YuE2 weights |
| 6 | **One take, both decoders** -- one performance decoded twice | the YuE2 weights and the 0.53 GB legacy decoder |

The first three are the `YuE2` menu. The last three open a run up into its
stages, which is what `YuE2/Advanced` is for.

Every one carries a **Read me first** note: what it does, what it downloads,
what to set before pressing Run, and where to go next. Every seed is left on
`fixed` rather than `randomize`, so pressing Run twice returns the same song
instead of spending the time on a different one.

Template 6's two claims were checked on a real run rather than asserted. The
same latents through `standard` and through `legacy` gave different audio, and
putting the decode node back to `standard` gave audio identical to the render's
own output, to the last sample.

## Nodes

### YuE2 Generate Song

The whole pack in one node. It finds the weights, loads them, runs the four
stages, and releases the VRAM again.

**Outputs**

| Name | Contents |
|---|---|
| `audio` | 48 kHz stereo, ready for `SaveAudio` or anything else that takes `AUDIO` |
| `score_abc` | The ABC score the model wrote before it played anything -- moved to the new key when `transpose` is set, since that is the score it sang |

**Inputs**

- `style` -- genre, instruments, vocal character, language and tempo. This is
  the field that decides what the record sounds like; the lyrics decide what is
  sung, not how.
- `lyrics` -- the words, with `[Verse]` and `[Chorus]` section tags. See
  [Writing lyrics](#writing-lyrics).
- `seed` -- the same seed with the same settings gives the same song, byte for
  byte. See [Reproducibility](#reproducibility).
- `options` -- an optional socket. Leave it empty and every setting takes the
  value the model was released with.

Progress is reported per stage on the node, and Cancel stops a run in under a
second -- between tokens during generation, between tiles during decoding.

#### The song editor

`YuE2 Generate Song`, `YuE2 Plan` and `YuE2 Plan Batch` carry an
`Edit song...` button and a summary of the song instead of two bare text boxes.
The button opens a window over the canvas, the style on top and the lyrics
under it, and nothing is written to the node until Apply.

![The song editor over the canvas. Under STYLE: language Russian; tempo 60 BPM on a slider; two voices, male rap vocal and expressive female voice, with the hint about voices under them; five sound parts, rap, heavy bass, hip hop beat, rhythmic punchy melody and staccato flow, each with up, down and remove buttons; and the style line the model will read. Under LYRICS: 399 tokens in the lyrics, the + Section and Edit as text buttons, a blue [Verse] and an orange [Chorus] of four lines each, thin marks under the letters where the tokenizer cuts, and a token count at the end of every line. Cancel and Apply at the bottom](docs/song_editor.png)

*A Russian rap for two voices. The style line under the parts is the string the
model reads, and the marks under the letters are where the YuE2 tokenizer cuts
the words.*

- **Style** is built from parts: a language, a tempo or none at all, the
  voices, and any other part in a list to add to, edit and reorder. The line
  underneath can be edited directly. A line the editor has not touched is not
  rewritten, because a different string is a different song even when it reads
  the same.
- **Voices.** Naming two asks for both; it does not decide who sings where.
  Tags such as `[Chorus - male rap vocal]` and roles written into the style
  were both tried on twelve songs, and neither put a voice where it was asked
  for, so there is no per-section voice control.
- **Lyrics** come section by section. A click on a section tag steps it
  through Verse, Pre-Chorus, Chorus, Bridge, Outro and Intro; a right-click
  lists all ten. Lines and sections are added, duplicated, moved and deleted
  from the buttons on each row, a double-click types into a line, and
  `Edit as text` shows the plain text the model reads.
- **A click on a letter flips its case**, and the marks show where the real
  tokenizer cuts, with a token count on every line. A capital inside a word
  splits it where you clicked; it is not a stress mark, and like any change to
  the words it gives a different song on the same seed. See
  [Stress, and what capital letters really do](#stress-and-what-capital-letters-really-do).
- A style or lyrics that arrive through a wire, from `YuE2 Write Song` for
  instance, are shown by the name of the node they come from and are edited
  there.

The marks need the YuE2 weights on disk, and ComfyUI needs a restart after the
pack is installed or updated before they appear. The window is in English only.

### YuE2 Write Song

One line of intent in, a style description and tagged lyrics out, wired
straight into `YuE2 Generate Song`. It exists because the format is the part
that puts people off, not the idea: everybody has an idea.

**Outputs**

| Name | Contents |
|---|---|
| `style` | One line: language, genre, voice, instruments, melody, phrasing, BPM |
| `lyrics` | Sections tagged `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]` |

**Inputs**

- `idea` -- what the song is about, in one line, in any language. "a sad song
  about winter, female vocal" is enough.
- `model` -- `auto` uses a GGUF you already have and downloads a 2.7 GB one
  only if you have none. The other entries are the GGUFs found in your ComfyUI
  model folders, in the Hugging Face cache and in
  [Ollama's own store](#models-you-already-have), each with its size, because
  that is the fact that decides whether it fits on your card. LoRA adapters and
  mmproj files are left out, because they cannot answer on their own. Two files
  of the same name are told apart by where they came from.
- `language` -- `auto` follows whatever language your idea is written in.
- `length` -- how long the song should be, in words rather than in seconds:
  `short` asks for 8 sung lines, `normal` for 16, `long` for 24 and `very long`
  for 32, which is about as much as YuE2 sings in one pass. It is an aim, not a
  promise, and the node reports how many lines it actually got.
- `seed` -- the same seed with the same idea gives the same words.
- `keep_model_loaded` -- leave it off when YuE2 generates on the same card
  afterwards.
- `instructions` -- optional, in your own words: "no chorus", "first person",
  "end on a question". It goes after the writing rules, so it wins where the
  two disagree.

#### Models you already have

Anything the writer can run is offered, wherever it already lives, so nobody
downloads the same quant twice:

| Where | What is read |
|---|---|
| ComfyUI model folders | `LLM`, `llm`, `text_encoders`, `clip`, `transformers`, `diffusion_models`, `unet`, `unet_gguf`, `checkpoints`, two levels deep |
| Hugging Face cache | Every snapshot folder, labelled by repository rather than by commit hash |
| Ollama | Every model pulled into its store, listed as `ollama: name:tag` |

Ollama keeps its models as ordinary GGUFs under digest names, so they are read
in place -- no copy, no export, no second download. A store somewhere this
process cannot reach cheaply is not searched for: a WSL or container store is
named in `YUE2_OLLAMA_MODELS` instead, because reaching into `\\wsl$` would
start a stopped distribution every time ComfyUI refreshes a dropdown.

**How the model is run.** There is nothing to install and nothing to choose. If
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) is already in
the Python ComfyUI runs on, the model is loaded in that process -- the fastest
path, and the only one where `keep_model_loaded` means anything. If it is not,
the node fetches about 32 MB of
[official llama.cpp binaries](https://github.com/ggml-org/llama.cpp/releases)
on first use and writes the song in a subprocess instead, which is a few seconds
slower per song and no worse at writing. The wheel is not a dependency because
on Windows it usually builds from source, and half an hour with a compiler is a
poor price for one optional node.

An llama.cpp you already have is used as it is, and is looked for in this order:
`YUE2_LLAMA_BIN`, a path written into `llama_bin.txt` in ComfyUI's user folder,
the runtime this pack unpacked, then `PATH`. The two named ones are also the way
to a CUDA llama.cpp on Linux, where upstream publishes no CUDA archive at all.
Setting `download` to `off` in `YuE2 Options` stops the runtime download too,
and then the node prints the archive links and the folder to unpack them into.

**What to expect from a small model.** Measured on Qwen3.5-4B-Q4_K_M, a song
takes one to six seconds to write and the format survives sampling: over twenty
runs across five line budgets every answer parsed. The line count is an aim
rather than a promise. On one seed at each of the four settings it came back
exact at `short`, `normal` and `long`, and four lines over at `very long` --
overshooting is the safe direction, because the length ceiling can cut a song
short and cannot extend one -- so the node reports how many lines it actually
got. A bigger model holds the count better; any instruction-following GGUF will
do.

The two ways of running it were compared on one idea across six seeds: the
binaries wrote a usable song five times, the wheel six. The miss was an answer
that opened with a paragraph of prose instead of the style, which the node
notices and asks again for. Both read the same prompt as the same 329 tokens,
so they differ in arithmetic rather than in what they are told; the same seed
repeated within one of them gives the same words back, but the two do not agree
with each other.

### YuE2 Options

Everything the main node deliberately does not ask about. An unconnected socket
is never a special case: the defaults here are the same values the node uses
when this node is not on the graph at all.

![YuE2 Options wired into YuE2 Generate Song. Options, top to bottom: cot full, max_seconds 0, keep_model_loaded false, cfg_scale 0.00, vae standard, device auto, attention_backend sdpa, download auto, quantization bf16, ode_steps 32, abc_temperature 0.70, abc_top_p 0.90, abc_top_k 30, temperature 1.00, top_p 0.95, top_k 100, repetition_penalty 1.200, offload on. Generate Song after a 58.523 s run: the options, style and lyrics inputs, the seed, the Edit song... button and the summary -- Russian, 60 BPM, male rap vocal and expressive female voice; rap, heavy bass, hip hop beat, rhythmic punchy melody, staccato flow; the sections as [Verse] 4, [Chorus] 4 and on; the lyrics -- above the caption 99 seconds of audio](docs/node_options.png)

*Every setting at its default except `offload`, set to `on`. 99 seconds of song,
and the node ran for 58.5 s on an RTX 5090.*

- `cot` -- `full` plans melody and harmony, `melody` plans the tune only, `off`
  skips the score and generates directly. `full` is the default and the one the
  benchmark numbers come from.
- `cfg_scale` -- text guidance. `0` means the released default, which is 1.0
  for `full` and `melody` and 1.01 for `off`.
- `max_seconds` -- the length ceiling. `0` works it out from the lyrics. See
  [Song length](#song-length).
- `vae` -- `standard` for listening, `legacy` to reproduce published benchmark
  numbers.
- `device`, `keep_model_loaded` -- where the model runs and whether it stays
  resident between runs.
- `attention_backend` -- `sdpa` by default. `cudnn` is about 17 percent faster
  and **not reproducible**; see below.
- `download`, `quantization` -- where the weights come from when they are not
  on the machine yet, and which build to fetch. See
  [Where the weights go](#where-the-weights-go).
- `ode_steps` -- solver steps for the acoustic stage. `32` is what the model
  was released with. Fewer is faster and thinner; more costs time and changes
  the result rather than clearly improving it.
- Sampling for both stages: temperature, top-p, top-k, repetition penalty.
- `offload` -- how much of the model the card holds at once. YuE2 has one set
  of weights for the score and the performance and another for the audio, and
  no stage needs both. `on` keeps only the half the running stage needs, and
  neither during the decode; `off` keeps everything on the card; `auto`, the
  default, moves a half off only when a stage would not fit beside it. The song
  is identical in every mode, to the last byte. Measured, `on` took a 40-second
  song from 7.55 GiB to 4.94 and a four-minute one from 14.96 GiB to 11.54, for
  a second or two a run.
- `transpose` -- moves the song to another key, in semitones: `2` is a whole
  tone up, `-3` a minor third down. YuE2 has no key control of its own. A key
  named in the style line is ignored ('A minor', 'in the key of A minor' and
  'E major' changed the key the model wrote 0 times in 18), and editing `K:`
  does not move a song either, because every note is read relative to it. What
  the model follows is the score, so this moves the score -- every note, chord
  and key by the same step -- just before it is sung, and `score_abc` gives the
  moved one. Measured at -6, -5, -3, +2, +5, +6 and +12 on a pop song and a rap,
  the chroma of all 14 renders landed where asked, and the separated vocal
  moved by the requested amount to within 1.3 semitones in the pop song and 0.1
  in the rap, a whole octave included. It is a new take of the same tune rather
  than the old recording pitched. It needs a score, so not with `cot` set to
  `off`, and a score the parser cannot read is refused; none of the 52 the
  model wrote in testing was.

### Staged nodes

YuE2 writes an ABC score and then performs it. `YuE2 Generate Song` does both
and never shows you the middle. These five nodes open it up, and they sit one
level down in the menu, under `YuE2/Advanced`, because most people never need
them.

The score is the last point at which a change is cheap, and it is the only
honest lever on where the stress of a line falls -- capitalising a syllable
nudges the tokenizer, but editing the score moves the note.

- **YuE2 Plan** -- style, lyrics and a seed in; a `plan` and the score as text
  out. No audio is generated, so this is a small fraction of a full run.
- **YuE2 Render Plan** -- a `plan` in, audio and `latents` out. Leave
  `score_abc` empty and the model's own score is sung, which gives exactly the
  song `YuE2 Generate Song` makes from that seed. Paste an edited score into
  the box and that is what gets sung.
- **YuE2 Decode Latents** -- the `latents` from a render turned back into audio
  without singing anything again. This is how to hear one performance through
  both decoders, `standard` and `legacy`.
- **YuE2 Plan Batch** and **YuE2 Select Plan** -- several scores from
  consecutive seeds, and a separate node that picks one. Picking is a separate
  node on purpose: changing the index does not rewrite the batch, so trying the
  next take costs only the singing.

An options node connected to `YuE2 Render Plan` overrides what the plan carried
-- except `cot`, which always comes from the plan. `cot` decides what the model
was told before it wrote the score, so changing it at the render would sing one
score under the instructions written for another.

**Measured**, on an RTX 5090, two lines of lyrics and a 40-second ceiling: the
whole run took 8.9 s, of which the score was 2.2 s. Four scores cost 8.8 s
against 35.7 s for four songs. The saving grows with length, because it is the
singing that gets longer and the score that stays much the same.

Two things were checked rather than assumed. Rendering an untouched plan
produced audio **byte for byte identical** to `YuE2 Generate Song` on the same
seed. Changing one line of the score -- `K:C` to `K:G` -- produced a different
song, and a different length with it. Not the same tune in G, though: the
dialect reads every note relative to the key, so each F is sung as F-sharp and
nothing else moves. To move a whole song, use `transpose` in
[YuE2 Options](#yue2-options).

## Song length

The model decides when the song is over, and on a short lyric it usually stops
well inside a minute. `max_seconds` is the ceiling for when it does not: given
four lines and three minutes of room, it will sometimes sing on long after the
words have run out.

Left at `0`, the ceiling is worked out from the lyrics -- roughly a minute for
a verse and a chorus, and 180 seconds when there are no lyrics at all to count.
The resolved value is logged, so the console says `length ceiling 60 s, from 4
sung lines` rather than leaving you to guess.

The ceiling is not a free parameter. It sizes the static KV cache and the
captured CUDA graph, which changes the order the attention reduction runs in,
so the same seed and the same lyrics under two different ceilings give two
different songs. Measured on this pack: ceilings of 40 and 90 seconds agree for
79 tokens and diverge at the 80th, while 90 and 180 agree to the last token.
Nothing is wrong with either take, but if you are hunting for a seed, settle the
ceiling first.

## Writing lyrics

Section tags in square brackets on their own line; the words underneath. The
tags are directions rather than words, and the model sings none of them.

```text
[Verse]
Neon fades along the lane
Footsteps keep the time of rain

[Chorus]
Let the day come into view
Every road begins with you
```

Upstream documents nothing beyond this. The model card says nothing about line
length, syllable counts or capitalisation, and the official skill file in the
YuE repository says only to put genre, instruments, vocal character, language
and tempo in `style`, and section tags and actual words in `lyrics`.

### Stress, and what capital letters really do

A trick that circulates for other music models is to capitalise a syllable to
move the stress. It does something here, but not what it looks like.

Lyrics reach the model verbatim -- there is no lowercasing anywhere in the
chain -- and the tokenizer is a case-sensitive BPE. Measured on this pack's own
tokenizer:

```text
" record"   -> 1 token   [" record"]
" RECORD"   -> 1 token   [" RECORD"]         a different token, but equally whole
" recORD"   -> 2 tokens  [" rec", "ORD"]     split exactly at the stressed syllable
"CARRY EVERY SPARK OF WONDER" -> 8 tokens, split as C|ARRY, SP|ARK, WON|DER
```

So a capital in the middle of a word works by breaking the BPE merge at that
point, which puts a boundary where a syllable break belongs. A word in full
caps creates no boundary at all, and a line in full caps breaks in arbitrary
places and is more likely to hurt than help.

Two consequences worth knowing. Changing the case changes the tokens, which
changes the whole song the way `max_seconds` does -- you do not get the same
take with better stress, you get a different take. And the stronger lever is
not the text at all: in `full` and `melody` modes the model writes an ABC score
first, and that is where it decides which syllable lands on a strong beat.

## Reproducibility

The same seed with the same settings gives the same song, byte for byte. This
holds because the pack pins the decode attention kernel to `sdpa`. The `cudnn`
kernel is about 17 percent faster and is not reproducible -- over four runs of
one seed it produced four different songs -- so it is an opt-in in the options
node, clearly labelled.

"The same settings" includes `max_seconds` and the exact text of the lyrics,
capitalisation included, for the reasons above.

## Where the weights go

The node fetches them on first use, so on a fresh install there is nothing to
do but run it. Files that are already on the machine are used as they lie, and
nothing is downloaded twice.

| `download` | What it fetches | Where it puts it |
| --- | --- | --- |
| `auto` (default) | Comfy-Org's single checkpoint, 7.26 GB | `ComfyUI/models/checkpoints/` |
| `comfy-org` | the same, whatever else is set | `ComfyUI/models/checkpoints/` |
| `original` | the three files m-a-p released | `ComfyUI/models/YuE2/` |
| `off` | nothing, and says which files are missing, the link to each and the folder it goes in | -- |

`auto` prefers the repackaged checkpoint for one reason: it is the same file
ComfyUI's own YuE2 nodes read. One download serves both, and if the ComfyUI
model manager has already installed it there is nothing to download at all. It
falls back to the released files when `vae` is `legacy`, which is a decoder
Comfy-Org does not publish.

| File | Size | Repository |
| --- | --- | --- |
| `checkpoints/yue2_3b_bf16.safetensors` | 7.26 GB | [Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2) |
| `checkpoints/yue2_3b_int8_convrot.safetensors` | 3.69 GB | [Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2) |
| `YuE2-3B/model.safetensors` | 6.76 GB | [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) |
| `YuE2-Vae/model.safetensors` | 0.49 GB | [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) |
| `YuE2-3B/qwen.tiktoken` | 2.4 MB | [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) |

`YuE2 Write Song` has a model of its own, fetched the same way and only when
the machine has no GGUF at all:

| File | Size | Repository | Licence |
| --- | --- | --- | --- |
| `LLM/Qwen3.5-4B-Q4_K_M.gguf` | 2.55 GB | [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | Apache-2.0 |

Any instruction-following GGUF in your ComfyUI model folders is offered in the
node's `model` list and is used instead, so this download is for people who
have none rather than a second requirement.

`ComfyUI/models/YuE2/` is registered with ComfyUI, so `extra_model_paths.yaml`
can redirect it like any other model folder, and `checkpoints` is whatever that
file already says it is.

Transfers resume. Bytes land in `<name>.part` and are renamed into place only
once whole, so an interrupted download costs the last chunk rather than the
whole file, and a half-written checkpoint is never mistaken for a model --
by this pack or by ComfyUI's own nodes, which read the same folder. Free space
is checked before the first byte.

What arrives is checked against the checksums published with it. m-a-p ships a
`weights_manifest.json` beside each release, so the `original` layout is
verified once, right after the download -- 6.76 GB in 4.7 seconds, measured --
and a file that is the right length but wrong inside is named rather than
loaded. That is the failure size cannot see: a truncated transfer is refetched
on the next run, while a complete-but-rewritten one is skipped forever as
already present. Comfy-Org's repack publishes no manifest, so for it size is
all there is, and this pack says so instead of inventing a verdict.

### The INT8 build

`quantization` set to `int8` fetches Comfy-Org's quantized checkpoint: 3.69 GB
instead of 7.26 GB.

It saves the download and not the VRAM. This pack's layers are ordinary torch
linears, so the INT8 weights are restored to BF16 as the file loads -- the
rotation that ConvRot applies is a 256-point Hadamard matrix, which is
symmetric and orthogonal and therefore its own inverse -- and the card holds
the same 6.8 GB either way. It is also not quite the same model: the round trip
costs about one percent of each weight, measured over all 896 restored tensors
against the BF16 release, worst cosine 0.99993. The same seed gives a different
song from the two files.

Whatever is already on disk is used before anything is downloaded, so choosing
`int8` on a machine that has the BF16 file changes nothing.

### Using files you already have

Nothing has to be copied or renamed. The search looks, in this order, at every
root ComfyUI knows about -- `models/YuE2` first, then `diffusion_models`,
`vae`, `LLM` and `checkpoints` -- and then at the Hugging Face cache, so a copy
pulled by any other tool is found where it lies.

Names carry no authority. A file that matches the published folder layout is
believed on sight, and anything else is identified by reading its header: eight
bytes of length and about a hundred kilobytes of JSON, never the seven
gigabytes behind it. A renamed `yue2-music-model.safetensors` dropped into
`models/diffusion_models` is found; an unrelated Qwen-shaped checkpoint sitting
next to it is not mistaken for it, because the discriminator is a tensor that
only this architecture has. The repackaged checkpoint is recognised the same
way, and the INT8 build is told apart from the BF16 one by what is inside it
rather than by its name -- loading one as the other would be 1355 tensor
mismatches.

Whichever layout is found, it is the same model. The conversion from the
repackaged file was checked against the released checkpoints tensor by tensor:
628 of 628 in the language model and 435 of 435 in the VAE, bit for bit, and a
song generated from each has the same waveform hash.

If nothing is found, the refusal names every missing file at once -- the direct
link and the exact folder for each -- because being told about one missing file
per run turns one fix into three.

### Environment variables

| Variable | Effect |
|---|---|
| `YUE2_MODELS_ROOT` | Search this directory and nothing else. An override that still lets the search wander is not an override, so setting this makes the answer to "where did it load that from" exactly one directory |
| `YUE2_OLLAMA_MODELS` | Where an Ollama store lives, for the ones this process will not reach on its own -- a WSL or container store |
| `YUE2_LLAMA_BIN` | The llama.cpp binary to use, ahead of every other place it is looked for |
| `HF_HUB_CACHE`, `HF_HOME` | Honoured when looking through the Hugging Face cache |

## Notes

**VRAM accounting.** The loaded model is cached by this pack rather than
registered with ComfyUI's model manager. That means no other node can evict it
mid-run, and also that ComfyUI does not count its 6.8 GB when it decides whether
something else fits. With `keep_model_loaded` off, which is the default, the
window where this matters is one node's execution.

**Global torch state.** YuE2 needs deterministic math settings that upstream
applies process-wide and never puts back. This pack sets them for the length of
one run and restores them afterwards, so nothing else in your graph is quietly
changed until the next restart.

**Network paths are refused.** A UNC path reaching the pack from a downloaded
workflow or an API request is not followed: merely looking at one authenticates
this machine against whatever host it names. Map the share to a drive letter
and nothing is lost.

**Inspected TLS still downloads.** An antivirus or a company proxy that opens
HTTPS to look inside it installs its own certificate authority in the system
store, which `certifi` -- what `requests` verifies against -- knows nothing
about. When the handshake is refused for that reason the download is retried
against the store the rest of the machine uses, the same one the browser
trusts. Nothing is skipped: the certificate is still verified.

## Licence

The code here is Apache-2.0. The model weights are CC BY-NC 4.0, which is
non-commercial, and the vendored upstream inference code keeps its own licence.
The writer's default language model is Apache-2.0 and belongs to neither.
See [NOTICE.md](NOTICE.md).
