# Changelog

[Russian version](CHANGELOG_RU.md)

The version in `pyproject.toml`, the git tag and the release on GitHub always say
the same thing; the release workflow refuses a tag that disagrees with
`pyproject.toml`, or one that either changelog has no section for. The section it
finds is published as the release notes, English above Russian.

## 0.3.0 - 2026-09-14

### Added

- **`transpose` in YuE2 Options: the same tune in another key.** A number of
  semitones, from -12 to 12. YuE2 has no key control of its own. A key named in
  the style line is ignored -- measured on two styles and three seeds, 'A
  minor', 'in the key of A minor' and 'E major' changed the key the model wrote
  0 times in 18 -- and editing the `K:` line of a score does not move a song
  either, because the notes are read relative to it: `K:C` to `K:G` sings every
  F as F-sharp and leaves the rest where it was.

  What the model does follow is its score. Over twelve songs the pitches of each
  score's notes and chords lined up with the chroma of its audio at a shift of
  zero, every one of them. So `transpose` moves the score: every note by the same
  number of semitones, every chord symbol and key field with it, just before the
  score is sung. `YuE2 Generate Song` moves the score it has just written and
  hands the moved one out as `score_abc`; `YuE2 Render Plan` moves the plan's
  score, or the one pasted into it. The same move through either node gives the
  same song to the byte.

  Measured on a pop song and a Russian rap, sung with the same seed at -6, -5,
  -3, +2, +5, +6 and +12: the chroma of all 14 renders sat exactly the requested
  distance from the original score, and the vocal, separated with MelBandRoFormer
  and pitch-tracked, moved by the requested amount to within 1.3 semitones in the
  pop song and 0.1 in the rap -- an octave up included. A moved song is a new
  take of the same tune rather than the old recording pitched, because the model
  sings the moved score from its first note. At 0 nothing changes: the check
  song is the same to the byte as before.

  The moved score is read back with upstream's own parser and compared with the
  original note by note before anything is sung. A score that parser cannot read
  is refused rather than guessed at, with the two ways out -- a move of 0 or
  another seed -- and so is a move with `cot` set to `off`, which writes no score
  at all; both refusals come before the model loads. None of the 52 scores the
  model wrote in testing, in eight styles and both `cot` modes that write one,
  was refused, and every move of each from -12 to 12 checked out.

- **Upstream's ABC reader, vendored.** `yue2_comfy/vendor/yue2_music/abc_tools.py`
  is `skills/yue2-music/scripts/abc_tools.py` from the YuE repository, unchanged,
  under the same Apache-2.0 licence as the rest of the vendored code.

### Changed

- **What editing `K:` does is described correctly.** Template 4 and the README
  said that changing `K:C` to `K:G` gives a different song. That is true, and it
  suggested the song moves to G, which it does not. Both now say what happens
  and point at `transpose`. Template 3 lists `ode_steps`, `offload` and
  `transpose` with the rest of the options.

## 0.2.0 - 2026-09-13

### Added

- **A song editor on the nodes that take style and lyrics.**
  `YuE2 Generate Song`, `YuE2 Plan` and `YuE2 Plan Batch` show an `Edit song...`
  button and a summary of the song where the two bare text boxes used to be. The
  button opens a window over the canvas, as tall as the screen: the style on top,
  the lyrics under it. Nothing is written to the node until Apply. Cancel and
  Escape ask before throwing edits away, and a click on the dimmed backdrop does
  not close the window at all.

  Writing for YuE2 means getting a format right -- a style line of
  comma-separated parts, section tags in square brackets -- and what a capital
  letter changes, where the tokenizer cuts a word, cannot be seen in a text box.
  The window is in English only; the Russian locale does not reach it yet.

- **The style line is built from parts.** A language from the ten the writer
  offers. A tempo on a 40-200 BPM slider, with a dot that blinks at the chosen
  tempo and a tick to leave the tempo out and let the model pick one; the dot
  stays still when the system asks for reduced motion. The voices. And every
  other part -- "staccato phrasing", "string section" -- in a list to add to,
  edit, reorder and trim, with suggestions as you type. The line the model will
  read is shown under all of it and can be edited directly: four rows that can be
  dragged taller, where Enter does not break the line and text pasted over
  several lines is joined with commas.

  A line the editor has not touched is not rewritten. Parts are joined again only
  after a change, because a different string is a different song even when it
  reads the same.

- **Several voices, and what naming them does not do.** Each voice is a row of
  its own, and together they go into the style line as one part joined with
  "and": "male rap vocals and female melodic vocals". A part is split back into
  voices only when every piece names a voice by itself, so "male and female duet"
  stays one voice and a genre such as "rap" is not mistaken for one.

  Naming two voices asks for both. It does not decide who sings where, and there
  is no per-section voice control because nothing that was tried made that work.
  Checked on twelve songs -- three seeds, each with four variants of the same two
  verses and two choruses -- with all four variants of one seed listened to. On
  that seed, with plain section tags the male voice never came in; with
  `[Verse - breathy female voice]` and `[Chorus - male rap vocal]`, with
  meaningless tags of the same length, and with the roles written into the style
  instead, it sang the second verse every time -- the same place whether the text
  asked for that, asked for the opposite, or said nothing at all. What moved the
  voice was the text changing, not what the text asked for. The hint under the
  list says as much.

- **Lyrics section by section.** Every section header is a coloured chip. One
  click steps it through Verse, Pre-Chorus, Chorus, Bridge, Outro and Intro, so
  the most common change -- verse to chorus and back -- is a click; a right-click
  lists all ten tags the writer uses. Lines and sections are added, duplicated,
  moved and deleted, and stanzas duplicated and deleted, from buttons on each row
  and from a right-click menu. A line moved past the edge of its section lands in
  the next one. Double-click a line to type into it, Enter starts the next;
  `Edit as text` switches to the plain text the model reads, and back.

- **Click a letter to flip its case, and see where the tokenizer really cuts.**
  A second click flips it back. The thin marks under the letters come from the
  real YuE2 tokenizer, run over the whole prompt the model reads rather than over
  the lyrics alone: Qwen's pre-tokenizer glues a closing bracket or a comma to the
  newline after it, so a line cut on its own disagrees with the same line in
  context at exactly its edges. Every line shows its token count, and the header
  shows the total.

  The window calls this what it is. A capital inside a word splits the word where
  you clicked, which changes what the model reads; it is not a stress mark, and
  like any change to the words it gives a different song on the same seed. See
  [Stress, and what capital letters really do](README.md#stress-and-what-capital-letters-really-do).

  Nothing is downloaded for it: the vocabulary is part of the YuE2 weights, and
  either the Comfy-Org checkpoint or the released files will do. Until the weights
  are on disk the editor works without the marks and says why, and it says so too
  when the lyrics hold characters the tokenizer normalises first (NFC), rather
  than drawing the marks under the wrong letters. Measured on the real
  vocabulary, no letter was cut in two in Russian, Chinese, Japanese, Korean or
  accented Latin samples; an emoji was, and is drawn with a dotted underline. The
  marks come from a new route, `POST /yue2/tokens`, so ComfyUI needs a restart
  after the pack is installed or updated before they appear.

- **The summary on the node.** Language, tempo and voice on the first row, the
  rest of the style under them, the sections as coloured tags with their line
  counts, and then the lyrics themselves, as many lines as the node has room for
  -- the summary grows with the node when it is resized. A click opens the
  editor. A style or lyrics arriving through a wire, from `YuE2 Write Song` for
  instance, is shown by the name of the node it comes from and is not edited
  here: the window says it is written there, and with both wired the button is
  switched off. While the node runs, ComfyUI's progress caption keeps to one line
  under the summary instead of taking half the node.

- Screenshots in both READMEs: `YuE2 Generate Song` with the editor's button
  and summary at the top, with the time the run took beside it; the editor
  window, in a new section that describes the editor; and `YuE2 Options` with
  every setting in view.

- **`offload` in YuE2 Options: only the half of the model a stage needs on the
  card, and the same song to the last byte.** YuE2-3B keeps two sets of weights
  in every layer -- one writes the score and the performance, the other turns
  them into audio -- and no stage uses both: 4.03 GiB for the first half, 2.63
  GiB for the second. `on` keeps only the half the running stage needs, and
  neither during the decode. `off` keeps both, as the pack always did. `auto`,
  the default, moves a half off only when the next stage would not fit beside
  it, judged from the free memory before each stage, so a card with room ends
  up holding the whole model and moving nothing.

  Measured on an RTX 5090: with the whole model on the card a 40-second song
  needed 7.55 GiB up to its decode, and with `on` 4.94 GiB for the whole run; a
  240-second song needed 14.96 GiB against 11.54 GiB, and the decode, which uses
  none of the backbone, 3.2 GiB. The moves cost a second or two a run. Under an
  8 GiB memory cap `auto` sang the 40-second song and `off` ran out of memory
  within a second; under 13 GiB `auto` sang the 240-second one. In every mode
  and under every cap the audio was identical to the last byte, because the
  weights are the same tensors wherever they are kept.

  The call is made before each stage rather than by running out of memory and
  trying again: on Windows the NVIDIA driver can put what does not fit into
  shared system memory, and a run then crawls instead of stopping. Where the
  driver does raise, `auto` runs the stage once more with only the needed half
  on the card -- a stage starts from its seed, so the result is the same -- and
  `off` names the setting to change. The weights waiting their turn sit in
  system RAM, up to 6.7 GiB of them.

### Changed

- **Style and lyrics are no longer two text boxes on those three nodes, and the
  workflow did not change with them.** The boxes are hidden and their sockets are
  drawn as ordinary inputs under `options`, so `YuE2 Write Song` is wired in as
  before and a wire still wins over the text. A workflow stores the same two
  strings in the same places. The button and the summary are never saved, and
  they sit after every other widget because ComfyUI matches saved values to
  widgets by position: anything placed before `style` would move every value
  after it. The example workflows and anything saved with 0.1.1 open unchanged,
  and a workflow saved with the editor opens on a machine without it.

- **The backbone loads with its halves on the CPU, and the decode leaves it off
  the card.** Loading used to put all 6.8 GiB on the card at once, before any
  stage could decide anything; now only the 0.10 GiB the stages share goes
  there, and each half follows when a stage asks for it. The decode used to move
  the backbone off only when memory was short and then copy it straight back;
  now whatever it moves stays off until a stage needs it, so a run that unloads
  the model afterwards never pays for the trip back.

- **The VRAM figures in the README were measured again, per mode**, replacing
  7.1 GiB for a short song and 11 GiB for three and a half minutes.

## 0.1.1 - 2026-09-13

First public release.

- **`YuE2 Generate Song`**: a style line, tagged lyrics and a seed in; 48 kHz
  stereo audio and the ABC score the model wrote on the way out. No model picker
  and no paths -- the weights are found wherever the machine already keeps them.
  Progress is shown per stage on the node, and Cancel stops a run in under a
  second.
- **The weights are fetched on first use**, resumably, into ComfyUI's own model
  folders: Comfy-Org's single 7.26 GB checkpoint by default -- the file ComfyUI's
  own YuE2 nodes read -- or the three files m-a-p released, checked against the
  checksums published with them. `quantization` set to `int8` fetches the 3.69 GB
  build, which saves the download and not the VRAM. `download` set to `off`
  fetches nothing and names every missing file with its link and its folder.
- **Files already on the machine are used where they lie**, from every model
  folder ComfyUI knows and from the Hugging Face cache, and are recognised by what
  is inside them rather than by their names.
- **The same seed with the same settings gives the same song, byte for byte.**
  The `cudnn` attention kernel is about 17 percent faster and not reproducible,
  so it is an opt-in.
- **`YuE2 Options`** for everything the main node does not ask about: `cot`,
  `cfg_scale`, the length ceiling `max_seconds` -- worked out from the lyrics when
  left at 0 -- the `standard` or `legacy` decoder, the device,
  `keep_model_loaded`, where the weights come from, and sampling for both stages.
- **`YuE2 Write Song`**: one line about the song in, a style line and tagged
  lyrics out, in one to six seconds on a 4B model. It runs a GGUF the machine
  already has -- in ComfyUI's model folders, the Hugging Face cache or Ollama's
  store -- and downloads a 2.55 GB one only when there is none, through
  `llama-cpp-python` when that is installed and through about 32 MB of official
  llama.cpp binaries when it is not. `length` asks for 8, 16, 24 or 32 sung
  lines.
- **Staged nodes under `YuE2/Advanced`**: `YuE2 Plan`, `YuE2 Render Plan`,
  `YuE2 Decode Latents`, `YuE2 Plan Batch` and `YuE2 Select Plan`. The score can
  be read and edited before anything is sung. An untouched plan renders the song
  `YuE2 Generate Song` makes, byte for byte, and four scores cost 8.8 s against
  35.7 s for four songs on an RTX 5090.
- **Six example workflows** in ComfyUI's template browser, each with a card and a
  Read me first note.
- **A Russian locale** for the node descriptions and input tooltips.
