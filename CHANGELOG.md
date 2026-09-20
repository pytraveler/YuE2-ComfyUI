# Changelog

[Russian version](CHANGELOG_RU.md)

The version in `pyproject.toml`, the git tag and the release on GitHub always say
the same thing; the release workflow refuses a tag that disagrees with
`pyproject.toml`, or one that either changelog has no section for. The section it
finds is published as the release notes, English above Russian.

## 0.8.2 - 2026-09-20

### Added

- **The sections of a score can be edited.** The coloured strip along the top
  of the piano roll was a picture of the `[intro]`, `[verse]` and `[chorus]`
  the score names; now it is the control for them. Drag a boundary to move it,
  click a name to change it or to empty the box and join that section to the
  one before, and click anywhere else on the strip to start a new section at
  that bar. The list offers the names the transcriber itself writes, and a name
  of your own is accepted. A boundary lands on any bar: the format only allows
  a section to begin between groups of bars, so the group is cut in two where
  it has to be, and not a note moves -- which is checked by reading the score
  back after every edit. Asked for in issue #4.
- **A warning when the style and the score disagree about the tempo.** The
  model follows the style line, so a score written at 147 BPM sung with
  `127 BPM` in the style has every phrase stretched, loses its instrumental
  stretches first, and drifts away from its words. Measured on a real cover:
  the sung sections came out 6 to 16 per cent longer than written, a 36-second
  interlude vanished, and the run stopped at 54 per cent of the score. Nothing
  said a word about it before.
- **A warning when a song ends before its score does**, with the two lengths,
  and whether it was the model that stopped or `max_seconds`.
- **A score can be started from nothing.** An editor with no score to draw now
  offers "Make an empty score": sixteen bars of four four at 120 BPM, silent in
  both parts, which the node sings in place of whatever else it would have.
  Asked for by people who wanted a grid to draw on rather than a page telling
  them to run the workflow first.
- **A song can be made longer from the roll.** "Add bars" on the piano roll's
  toolbar adds 1 to 32 empty bars at the end. Nothing already written moves:
  the added groups only carry a voice line each, so the meter, the key and the
  last section run on into them.
- **Right-click removes a chord.** Clicking a chord in the lane still opens it
  for editing; the right button drops it in one go, and Ctrl+Z brings it back.
- **Right-click on the section strip moves the play cursor**, the way the bar
  strip below it always has. The left button still renames a section, so the
  strip does both without a modifier.
- **A warning when a transcription's beat is not the recording's.** YuE2
  Transcribe now measures the pulse of the recording itself -- onsets and their
  autocorrelation, no model, a second of work -- and compares it with the tempo
  it wrote. Measured on a five-minute song whose beat is a steady 130 BPM: the
  transcription was written at 147, and its notes, laid against the real beat,
  fall on it at chance. The melody, the chords and the sections of such a
  transcription are still worth having; its rhythm is not, and nothing
  downstream can recover it. A minute or two at a time is heard far more
  accurately than a whole long song.

### Changed

- **The piano roll can be put on thirty-second notes.** The grid step is one
  note length of the score and the format has no fractional lengths, so a score
  written on sixteenths could not hold a thirty-second note at all -- and a
  transcription of a song in 4/4 is always written on sixteenths. The snap list
  now offers "Thirty-second notes (rewrites the score)" for such a score, and
  choosing it writes the whole score again on thirty-seconds. The song is not
  touched: every time and every length is the same number of quarter notes, and
  the two texts are read back and compared before either is kept. Asked for in
  issue #4.
- **The bar strip of the piano roll is half again as tall**, 30 pixels instead
  of 20, because it is what you click to move the play cursor and it was easy
  to miss.

## 0.8.1 - 2026-09-20

### Added

- **A progress bar in the console, beside the one on the node.** ComfyUI draws
  a node's bar in the browser only, so the terminal the server was started from
  said nothing at all for the minutes a song takes. Every node of this pack now
  draws the same fraction and the same caption there as well --
  `YuE2 Generate Song: Composing |######----|  58% [01:12]` -- through tqdm,
  which every ComfyUI has, and through a line of its own where it is missing.
  The line closes when the node is done, so nothing ComfyUI prints next lands
  in the middle of it. There is no ETA on purpose: the stages do not take the
  shares of the bar they are given, so a remaining time worked out from the
  percentage would be a number this pack cannot stand behind. A download knows
  its own and writes it into the caption, as it always did.
  `YUE2_CONSOLE_PROGRESS=0` leaves a console that is really a log file alone.
- **A tempo for the score, in the score editor.** A slider and a box beside the
  play controls set the tempo the score is written at -- `Q:1/4` in the ABC --
  from 40 to 200 BPM, or from the score's own tempo when it came in outside
  that. Nothing is renotated: the notes keep their lengths in bars, and the
  whole song is sung faster or slower. The length on the facts line moves as
  the slider does, and so does the dashed line where the singing stops. The
  tempo is part of the edit like a moved note, with its own undo, and it
  reaches the node on Apply. It earns its keep most on `YuE2 Transcribe`, where
  the tempo was heard from a recording rather than chosen.
- **Example styles in the lyrics editor, taken from YuE2's own pages.** The
  style line is the hardest box in this pack to fill: nothing on the screen
  said whether the model wanted `sad piano` or five clauses about the room the
  drums were recorded in. `Examples` now offers twenty-eight lines that were
  really run -- ten written by the model's authors for their cover and editing
  demos, eighteen sent in for the genre gallery, in English, Chinese, Japanese
  and Russian. Picking one fills the style line, and an arrow puts the old one
  back. They are copied word for word, which is the point of them: one that
  never named its language leaves the `Language` box empty rather than being
  tidied up on the way in. The gallery's seventy genre names now complete in
  the sound box as well, and the voices and sounds it suggests are the demos'
  own vocabulary -- `no guitars`, `brushed drums`, `honking sax section`,
  `warm natural jazz-club recording` -- rather than phrases invented here. The
  list is written from a downloaded copy of the site by a tool that refuses any
  line it cannot reproduce exactly.

### Changed

- **Play sounds a real piano instead of a synth.** The roll played square and
  triangle waves, which say where a note is but not what it is: an edit that
  ruined a phrase sounded much like one that saved it. It now plays a grand
  piano, thirty recordings from Salamander Grand Piano (Alexander Holm,
  CC-BY 3.0), one every three semitones from A0 to C8, so no note is stretched
  by more than a semitone and every key of the roll has its own sound. They
  ship with the pack -- half a megabyte, loaded when the editor opens, nothing
  fetched from the internet -- and if they cannot be loaded at all, Play falls
  back to the old synth and says so rather than going silent.
- **The piano roll names C and the row under the pointer, not every white
  key.** Seven names an octave on rows fourteen pixels apart were a wall of
  text beside the notes. C stays, in bold; whichever row the mouse is on says
  what it is, black keys included, which had no name at all before -- theirs is
  written in white, on the key itself.

## 0.8.0 - 2026-09-20

### Added

- **LoRA adapters, on a node of their own.** `YuE2 LoRA` holds them as rows:
  one row is one file with a strength for each half of YuE2 -- AR, which
  writes the score and sings the performance, and NAR, which turns it into
  sound. ComfyUI's own `LoraLoader` calls the same two numbers `strength_clip`
  and `strength_model` for this model. The rows reach `YuE2 Generate Song`,
  `YuE2 Plan`, `YuE2 Plan Batch` and `YuE2 Render Plan` through a new `lora`
  input, last on each node so no saved workflow shifts, and one `YuE2 LoRA`
  node chains into another. The list offers only files that are for YuE2, from
  the LoRA folders ComfyUI already knows, and under each row says what the
  file changes, its rank, its trigger word -- shown, never written into the
  style for you -- the `cot` it was trained for, and the acoustic adapter it
  was trained beside. Files in m-a-p's layout are read as well as files in
  ComfyUI's, which matters: handed an m-a-p file ComfyUI's own loader patches
  nothing and only says so in the console, and of the eight YuE2 adapters this
  was measured against, three were in a layout it could apply.
- Adapters are folded into the weights as a half arrives on the card, 0.12 to
  0.47 seconds a half on an RTX 5090 with the file read, so a song with them
  runs at the speed of one without. With `low_vram` the rows are packed to
  INT8 and cannot be folded into, so the adapters ride beside them as factors
  instead, at 6 to 11 percent of the token speed. The fold is exact: on a file
  in ComfyUI's layout it equals core's `calculate_weight` to the bit, one seed
  gives one song in every `offload` mode, and the weights go back to the
  checkpoint bit for bit when the adapters come off. A plan carries the set it
  was made with, `YuE2 Render Plan` can replace it, and the song memory writes
  each adapter's name, file and strengths down beside the song.
- **Template 10, *A song with LoRA***: the same song node with the adapter
  node in front of it, its list empty and ready for a file of your own.

### Fixed

- **YuE2 Transcribe refused a score when the model numbered a bar's beats
  out of order** (issue [#3]). SheetSage2 sometimes repeats or skips a beat
  number inside a bar -- `[1, 2, 2]` in the report, 177 bars into a song -- and
  the node stopped with "counts its beats ..., not in order", blaming a missing
  beat or key. A bar is now as long as the beats between its downbeats, as
  ComfyUI's own SheetSage2 writes it: such a bar comes out as 5/4 or 3/4, and
  the score is the same to the byte as ComfyUI's.
- **A long, busy recording lost seconds at every seam, and that refused the
  score too.** A recording over 300 seconds is heard in parts, each answering
  for 200 seconds of it. A part that spends the decoder's 5120 tokens stops
  before the end of what it answers for -- 1.7 to 4 seconds early on the drum
  and bass track this was measured on -- and the next part dropped everything
  before the planned boundary, so those seconds fell out of the song: beats
  went missing, the grid around them stretched, and a short note in the gap
  refused the score with "shorter than a subbeat of the grid". The next part
  now takes over where the one before it really stopped, and a part that stops
  short says so in the log. Recordings whose parts reach their end, which is
  all five tracks measured here but one, are transcribed the same to the byte.

[#3]: https://github.com/pytraveler/YuE2-ComfyUI/issues/3

## 0.7.2 - 2026-09-19

### Changed

- **Moving a half of the model is faster.** The weights stay in the checkpoint
  file after they are loaded, so moving a half onto the card was really a read
  from disk through page faults, and moving it off was a copy into RAM. Onto the
  card now goes through two pinned buffers -- from 1.6 GB/s to 4.7 for a file
  not yet in the system's cache, from 4.2 to 7.6 for one that is -- and off the
  card is free: the weights are still in the file. On the reporting RTX 3070 Ti
  the moves had taken 12 s of a 60 s run.
- **The log says where a song's time went.** A second line after the summary
  gives the seconds of the score, the performance, the acoustic stage and the
  decode, and what loading added.

### Fixed

- **The decode ran an 8 GB card into its ceiling.** In `offload` `auto` the
  decode kept the NAR half (2.63 GB) on the card whenever 3.5 GB was free, which
  is what the decode allocates. What it holds is more: 4.0-4.5 GB in the pool
  ComfyUI uses on CUDA 13, 8 GB under the native allocator. An RTX 3070 Ti
  reached 7.1-7.8 GB of the 6.8 it had, and on Windows that is a card spilling
  into system memory, not an error. The decode now asks for the room it holds,
  so a card like that moves the NAR half off first -- with no copy, see above --
  and peaks at 4.5-5.4 GB. It also gives the allocator's cache back after every
  tile (8 GB down to 5 under the native allocator) and no longer leaves 0.25 GB
  of the decoder's weights on the card after it finishes. Songs are the same to
  the byte.
- **The legacy decoder downloaded the whole model again beside Comfy-Org's
  checkpoint.** With `vae` at `legacy`, a machine that had
  `yue2_3b_bf16.safetensors` fetched m-a-p's 6.76 GB backbone to get a 0.49 GB
  decoder, and with `download` at `comfy-org` it fetched nothing and refused.
  The checkpoint's backbone and vocabulary are m-a-p's, byte for byte, so a
  legacy run now takes them from the checkpoint and fetches the decoder alone;
  a song from either sounds the same to the byte. The INT8 checkpoint stands in
  only with `quantization` at `int8`. The same goes the other way: a machine with
  m-a-p's backbone that lacks only the standard decoder fetches that decoder,
  not the checkpoint.
- **ACE-Step 1.5's VAE could still be taken for YuE2's decoder, and 0.7.1 got
  issue [#1] wrong.** The file in #1 was not a video VAE but ACE-Step 1.5's
  diffusion model, which calls itself `decoder`; 0.7.1's tensor names stopped
  it. ACE-Step's own VAE in `models/vae` has those names too, with the same
  shapes, so a `standard` run with only m-a-p's backbone on the machine could
  still pick it up. A decoder is now also recognised by its output convolution,
  which only YuE2's six-block decoder has.
- **With `YUE2_MODELS_ROOT` set inside ComfyUI, downloads went elsewhere.** The
  search looked only in the override while the files were written to
  `ComfyUI/models`, so a download succeeded and the next run found nothing.
  Downloads now go into the override. If you had set it, the speech model and
  the vocal separator will be fetched once more, into the override; the old
  copies in `ComfyUI/models/YuE2` can be deleted.
- **A broken `qwen.tiktoken` said "not enough values to unpack" and nothing
  else.** The search takes a vocabulary beside the backbone by its name, so a
  damaged or stray one stopped the run in the parser's own words. It is now
  refused by its path and size, with what to do: put m-a-p's file back, move
  the folder out, or, for the pack's own copy of the checkpoint's vocabulary,
  delete it.

[#1]: https://github.com/pytraveler/YuE2-ComfyUI/issues/1

## 0.7.1 - 2026-09-18

### Fixed

- **A video VAE was taken for YuE2's decoder ([#1]).** The file search accepts
  a file as the decoder when it has tensors whose names start with `decoder.`
  and no `lm_head.weight` -- which is true of every video and image VAE ever
  published, and `models/vae` is one of the folders it sweeps. It only got the
  chance on an install that has the released backbone without the standard
  decoder beside it, which is exactly what template 6 leaves behind: it asks
  for the legacy decoder, Comfy-Org does not publish one, so the released files
  are fetched, and the next `standard` run finds the backbone, finds no decoder
  to match, goes looking, and comes back with somebody's video VAE. A decoder
  is now recognised by three tensor names only the released one has --
  `decoder.layers.0.weight_g`, its `weight_v` and the first SnakeBeta slope --
  so a stranger is not a candidate at all.
- **The refusal named a dtype and not a file.** Loading a decoder that is not
  FP32 said which tensors were narrow and left out the one thing worth knowing,
  which file they were in. It names the file now, and says what to do when that
  file is not YuE2's.

[#1]: https://github.com/pytraveler/YuE2-ComfyUI/issues/1

## 0.7.0 - 2026-09-17

### Added

- **`low_vram`: a song on a card of about 4 GB.** A new last switch in
  `YuE2 Options`, off by default. With it on, the 28 layers are kept on the card
  as INT8 rows rather than BF16 -- half their memory -- and each matrix becomes
  BF16 again only for the multiply that needs it, while the last stage decodes
  in 256-frame tiles instead of 1024. Measured on an RTX 5090 with `offload` at
  `on`: a 40-second song peaked at 3.11 GiB instead of 4.43 and a four-minute
  one at 3.14 instead of 4.50, and both sang with the card capped at 3.5 GiB,
  where the 40-second song had needed 4.75. The four-minute song took 110
  seconds instead of 104: the token loop is about twice as slow per step, and
  the acoustic stage is faster because there is half as much to carry.

  It is the one switch in this node that changes the song. An INT8 round trip
  is lossy -- measured on all 392 matrices of the released checkpoint, a cosine
  of 0.99990 or better and about one percent of relative error -- so the same
  seed writes the same score and a new performance of it. Both takes were read
  back by the pack's own speech recognition: the four-minute one sang the lyric
  through, and 98.6 percent of the words heard were words of the lyrics, which
  is exactly what the BF16 take scored.

### Changed

- **Half the video memory a song needs, and a quarter off its time.** The
  acoustic stage asked PyTorch for grouped-query attention, which the
  memory-efficient kernel refuses outright; with flash not compiled into the
  Windows wheels, every call fell back to the kernel that builds the whole
  attention matrix. Repeating the key heads and dropping the argument lets the
  fused kernel take the call: measured on an RTX 5090 with `offload` at `on`,
  a four-minute song peaked at 5.66 GiB instead of 10.72 and took 98 seconds
  instead of 122, and a 40-second one 4.67 GiB instead of 4.91, in 21.4 seconds
  instead of 28.6. It is the same attention -- repeating the key heads is what
  the refused argument does internally -- and it is what ComfyUI's own code
  does for its models.
- **Only the rows of the vocabulary the running phase can use.** YuE2 writes
  its score out of the text vocabulary and its song out of the codec one, and
  masks every other row away before it samples, so the rest of the two big
  tables never had to be on the card. They stay in system RAM now, with one
  window of them on the card: while the song is written that is 32770 of 184704
  rows, 0.125 GiB each instead of 0.705. Measured with `offload` at `on`: a
  40-second song peaked at 4.43 GiB instead of 4.67 and a four-minute one at
  4.50 instead of 5.66, both a second or two faster, and both the same song to
  the last byte, with or without CFG. `off` still keeps everything on the card.

- **Songs made before this version do not come back byte for byte.** The score
  and the sung notes are the same: the same seed writes the same score and the
  same performance, to the byte. Only the last stage, which turns them into
  sound, renders a little differently -- about 30 dB below the song's own level,
  which the author of this pack could not hear in a blind listen. The same seed
  still gives the same song from this version on.

## 0.6.1 - 2026-09-16

### Added

- **`vocals_only`: a song with nothing but the voice.** A new last switch in
  `YuE2 Options`, off by default. With it on, `YuE2 Generate Song`,
  `YuE2 Render Plan` and `YuE2 Decode Latents` make the song as always and then
  separate its voice from the band: `audio` carries the voice alone, at the
  song's length and rate, and the same seed still gives the same song with the
  switch off. YuE2 writes one stream for the whole mix and has no voice-only
  output of its own. Asked for "a cappella" in the style, it still left a soft
  held pad under the voice in 61 of 72 songs measured, and "no instruments"
  changed nothing -- so the voice is taken from the finished mix, and "a
  cappella" in the style keeps it free of the silences an ordinary song's intro
  and breaks leave.
- **`YuE2 Vocals Only`: the voice of any recording.** The same separation as a
  node of its own: audio from anywhere in, its voice out, in the shape and at
  the rate it came in.
- **Mel-Band RoFormer, run by this pack's own code.** Kimberley Jensen's vocal
  model (MIT, 0.85 GB) is downloaded on first use into `models/YuE2`, pinned to
  the revision the pack was checked against; a copy already on the machine,
  kijai's conversions included, is used instead. The network matches the
  reference code the model was trained with to the last bit in float32, a
  three-minute song takes about 7 seconds on an RTX 5090, and the words come
  through: 91 percent of the lyrics were heard in order both in the separated
  voices and in the full songs, over 72.
- **Template 9.** `9 - An a cappella song` sets the switch up with an a cappella
  style.

### Fixed

- **The bar under a song node stood full while the song was sung.** The stages
  reported percentages to a bar whose total is one, so it filled at the first
  percent past one and stayed full to the end. It fills with the stages now.
- **A song whose lyrics put a whole verse on one line was cut off at a minute.**
  With `max_seconds` at 0 the length ceiling counts sung lines, and it counted
  such a verse as one line; a line that began and ended with a direction, like
  `[Only bass] I don't need a map ... [Guitar stab]`, was taken for a section
  marker and not counted at all. One such song of about sixteen sung lines got
  four and a ceiling of 60 seconds. Directions in brackets are now taken out of
  a line before its words are counted, and a long line counts one line for
  every eight words, so that song gets 264 seconds. Lines of up to eleven words
  count as before, and the dashed line on the piano roll moves with it.
- **`YuE2 Transcribe` could hear nothing but "Ooh" in a song that opens with
  one.** On a song that begins with vocalising, the speech model could start
  writing "Ooh, ooh" and keep on to its token limit, and the node then gave
  almost no words. Such a loop begins where the model all but ties between two
  picks -- in bfloat16 they tie outright now and then -- so an answer that says
  the same few tokens twenty times over is now taken back to the narrowest
  choice where the repeating began, and the other pick is taken there. Measured
  on 20 loops, three songs heard at several volumes: every one came back with
  all the words the same song gave without the loop, in about a second rather
  than five, and over 150 recordings nothing else changed.

## 0.6.0 - 2026-09-16

### Added

- **`YuE2 Load MIDI`: a song from a MIDI file.** A new node reads a `.mid`,
  `.midi`, `.kar` or `.rmi` file from ComfyUI's `input` folder -- uploaded with
  its `Choose MIDI file...` button or dropped on the node -- and writes it as a
  score YuE2 sings from: the vocal line from one track and the instrumental line
  from another, chosen by number or left to the node, which lists every track
  and the one each line takes before anything runs. Wired into
  `YuE2 Generate Song` with `cot` at `melody`, the tune is sung in whatever style
  the style line describes; sung from two files and heard back, the songs kept
  76 to 100 percent of the melody's notes in order. With `mode` at `full`, chord
  symbols are guessed from what the file's tracks play together. A karaoke
  file's words become the `lyrics` output under section tags, a paragraph sung
  twice taken for the chorus. The score and lyrics editors sit on the node as
  they do on `YuE2 Transcribe`, and nothing is downloaded.
- **Words laid along a bare tune.** `YuE2 Generate Song` and `YuE2 Render Plan`
  lay the lyrics along a score that names no section -- which is how
  `YuE2 Load MIDI` hands one on, unless its `without_sections` switch is off --
  before singing it: each line on a phrase with about as many notes as it has
  syllables, riffs passed over, the tune repeated when the words outlast it, a
  chorus sung twice on the same bars, and the song stopped shortly after the
  tune ends. Sung over a MIDI tune as it stood, the template's words were lost,
  0 of 46 heard over the Pirates of the Caribbean theme; laid along the GTA San
  Andreas intro they were heard 97 percent in order on average over eight seeds,
  as clearly as over a score the model writes itself.
  The node says which bars each section is sung on, and warns when a tune has
  far more notes than the words have syllables.
- **Save as MIDI.** The score editor saves the score as it stands in the window,
  edits included, as a MIDI file: the voice, the instrument line and the chords
  on tracks of their own, with the tempo, meters, keys and sections. Loaded back
  into `YuE2 Load MIDI`, it gives the same notes in the same bars.
- **Two templates.** `7 - Sing a MIDI file` sings a MIDI file, and
  `8 - Cover a song` covers a recording with its own words, recognised.

### Changed

- **Brighter notes on the piano roll, and every white key named.** The notes
  are drawn brighter, with a dark outline and their names in bold, so they
  stand out from the grid. The keyboard names every white key -- C4, D4, E4 and
  on -- where it named only the Cs, which stay in bold.

### Fixed

- **The piano roll was drawn blurred.** It was drawn two pixels wider and taller
  than the space it sits in and shrunk back by the browser, which smeared every
  line and letter over the pixels next to it and drew notes near the right and
  bottom edges up to two pixels from where a click finds them. It is drawn pixel
  for pixel now.
- **A click on the chord lane opened no box to type a chord in.** The box opened
  and closed within the same click: as the mouse button went down, the browser
  moved the focus to the roll, and the box, left empty, closed. It stays open
  now.

## 0.5.1 - 2026-09-16

### Fixed

- **The tests failed on machines without torch or a language model**, such as
  GitHub's. They pass there again.

## 0.5.0 - 2026-09-15

### Added

- **`YuE2 Transcribe`: a cover from a recording.** A new node listens to a
  recording and writes what it hears as a score YuE2 sings from -- the vocal
  and instrumental lines, beats, key and sections, and with `mode` at `full`
  the chords too. Wired into `YuE2 Generate Song` with `cot` at `melody`, the
  tune is kept and the style, voice and instruments are whatever the style
  line says; on three real tracks, 95-97 percent of the melody's pitch order
  survived the cover. The transcription is this pack's own implementation of
  SheetSage2, checked against ComfyUI master's: master's tokens for the pack's
  own songs give the same ABC to the byte. Its weights (1.29 GB, CC BY-NC 4.0,
  like YuE2's) are downloaded into `models/audio_encoders` on first use. The
  `lyrics` output is the section tags; with `lyrics_auto_recognition` on, the
  sung words are recognised by Qwen3-ASR-1.7B (3.8 GB, Apache-2.0, downloaded
  into `models/YuE2`), cut into the sections and laid out in lines by the
  writer's language model, under a guard that keeps the words as heard. The
  score and lyrics editors sit on the node as they do on the song node, and an
  edit belongs to the recording it was made on. Transcription and recognition
  are kept per recording, so a new seed only lays the words out again. Inside
  ComfyUI the recognition's decoding step runs as a CUDA graph -- six seconds
  for a three-minute song where the plain loop took twenty-six -- and every
  sixty-fourth token of the replayed step is checked against the plain one,
  cache and all, with the plain step taking over if they ever disagree.
- **A warning when a score without chords is sung under `cot` `full`.** A
  melody-only score -- a transcription made for a cover looks like that -- says
  nothing about harmony, while `full` tells the model the score carries the
  harmony too. `YuE2 Generate Song` and `YuE2 Render Plan` still sing it, and
  say to set `cot` to `melody` so the accompaniment can follow the style.

### Fixed

- **Unload Models now frees the models this pack keeps.** With
  `keep_model_loaded` on, the YuE2 model and the writer's model sat in the
  pack's own cache, where ComfyUI's Unload Models button never reached them:
  the card stayed full until ComfyUI was restarted. The button releases them
  too now -- measured on the card, from 9.9 GB back down to 2.5 GB.
- **Lyrics typed in the song editor's text view were lost.** 'Edit as text'
  read the box only when switching back to sections, so Apply pressed straight
  from the text view wrote the lyrics as they were before, and Cancel closed
  without asking. Both see what was typed now.

## 0.4.1 - 2026-09-14

### Added

- **The score editor shows where the song ends.** The model writes a score for
  the whole song, but the singing stops at `max_seconds` -- at `0`, at the
  ceiling worked out from the lyrics -- so bars past that point were never
  heard, and nothing on screen said so: a 60-second ceiling under a score of
  1:45 left eight edited bars of a chorus silent. Now a dashed yellow line on
  the piano roll marks the ceiling and dims the bars after it, the facts above
  the roll say how much is sung, and both the window and the node's summary
  name edited bars that fall after the line. The editor reads `max_seconds`
  off the options node the singing node uses -- for `YuE2 Render Plan`, its own
  options or else the plan node's -- and counts the lyrics itself; lyrics that
  arrive through a wire are counted by the node that receives them, which hands
  the ceiling to the browser as `yue2_auto_seconds`. The rule lives in
  `constants.length_ceiling`: the singing stage stops by it, and a test holds
  the browser code to it.
- **The habits people bring from FL Studio and other desktop piano rolls.**
  Ctrl selects like Shift, with a click or a drag; Alt draws, moves and
  stretches a note off the grid, in steps of the shortest note the score is
  written in; Ctrl with the up and down arrows moves the selection by an
  octave, as Shift already did. The right Alt of a keyboard with AltGr works as
  Alt too: Windows reports it as Ctrl and Alt together, and read as Ctrl it
  selected instead. Every key that was there still works.
- **A chord drawn as stacked notes gets an answer.** A part sings one note at a
  time, so a note drawn on top of another is still refused, but the window now
  says why and where chords go, with a button that puts the chord the two notes
  suggest -- `C` for C and E, `Am` for A and C -- on the chord lane at that
  spot in one click. A click on the chord there changes or removes it.

### Changed

- **The piano roll takes the familiar dark look.** Light green notes with their
  names on them, selected notes in red, a blue-grey grid with strong bar lines,
  a keyboard of real keys and an orange play marker. The part being edited is
  green and the other part a faint grey behind it, where before each part had
  its own colour. The line where `max_seconds` ends the song is yellow and
  dashed, since red means selected.
- **A note stretched into the next one moves the boundary between them.** The
  stretch used to stop dead without a word, so in a bar where the notes touch,
  as they often do in the model's scores, no note could be made longer. Now the
  next note starts later and keeps its end, never squeezed out, and the
  stretched end lands on a grid line even when the note starts between two. A
  click in a gap shorter than the last note drawn draws a note that fills the
  gap, where it was refused.

### Fixed

- **A chord typed on the chord lane could vanish.** With the chord box open, a
  click on the grid set the chord for an instant, and the edit that click made
  then dropped it. The box now settles before the click does anything else, Esc
  closes it without a change, and it keeps its own width however other styles
  on the page size text boxes.

## 0.4.0 - 2026-09-14

### Added

- **A score editor on `YuE2 Render Plan`.** An `Edit score...` button and a
  summary of the score the node will sing stand where the bare `score_abc` box
  used to be. The button opens a window over the canvas with three views of one
  score, and nothing is written to the node until Apply; Cancel and Escape ask
  before throwing edits away. The summary says whether the node sings the
  model's score or an edited one, which bars were rewritten, and when the plan
  has written a new score since the edit was made.

  - **Piano roll.** The voice part and the instrument part, one edited at a
    time with the other drawn faintly behind it, the sections along the top and
    a chord lane under the bar numbers. A click draws a note, a drag moves it,
    its right edge stretches it, a right-click or Delete removes it; Shift
    selects several, the arrow keys move them by the grid, a semitone or an
    octave. Snap from whole notes down to the score's own unit, zoom, undo and
    redo. Play sounds the parts, and the chords if asked, through a plain synth
    in the browser that downloads nothing.
  - **Notes.** The score as sheet music, drawn by abcjs, with the bars the edit
    rewrites in red.
  - **ABC.** The text the model reads; a score pasted here loads into the other
    two views.
  - **Write the score** runs only the plan node feeding the render node, so a
    score can be edited before anything has been sung.

- **The same editor on `YuE2 Generate Song`.** After a run, `Edit score...` on
  the song node opens the score it wrote, and the next run sings the edit
  instead of writing a score, so a phrase can be fixed without rebuilding the
  workflow around `YuE2 Plan`. The edit is kept in a new optional `score_abc`
  input, the node's last widget, so a workflow saved before it existed loads
  unchanged; empty, the node works exactly as before. The score reaches the
  window through the node's `ui`, and the node stays an ordinary node, since
  running it on its own would mean singing the whole song.

  Checked on the card, sample for sample: with the box empty the node gives
  the same song as before; one edit through this node and through
  `YuE2 Render Plan` gives the same song, and the same as that edit rendered
  without a mark before marks existed; an edit marked for other words leaves
  each node its own song.

- **An edit belongs to the words it was made for.** On Apply the editor marks
  an edit with the style, lyrics and `cot` it was made for -- a comment line
  that every node takes off before comparing or singing, so the model never
  reads it -- and both singing nodes check the mark. A new seed with the same
  words sings the same edit as a new take, which is how a bar that did not take
  is tried again; for a new tune, a **Reset score** button, shown on the node
  while an edit is kept, throws the edit away. Other words leave the edit
  unsung: `YuE2 Generate Song`
  writes a new score for them and `YuE2 Render Plan` sings the plan's own,
  each with a warning, and the edit comes back with the words. A score pasted
  or wired in carries no mark and is sung whatever the words. The rule lives in
  `yue2_comfy/edits.py`.

- **Only the bars that change are written again.** `yue2_comfy/notation.py`
  reads a score into bars, notes and chords for the window and writes an edit
  back into the text, rewriting the changed bars and any bar tied to one of them
  and leaving every other byte where the model put it. A rewritten bar is spelled
  the way the model spells one: an accidental only where the key needs it,
  repeated in another octave of the same letter so that a person and abcjs read
  it as the parser does, and a whole bar of rest as `Z`. The result is read back
  with upstream's own parser and compared with the notes that were asked for,
  and an edit that does not survive that is refused rather than sung. A score
  opened and applied without a change is handed on as the model's own, so it
  still sings the song `YuE2 Generate Song` makes from the same seed.

  Tested on the eleven model scores in `tests/data/model_scores.json`: each one
  reads and writes back unchanged to the byte, and forty random edits of each
  read back note for note and touch only their own lines. The bars, meter, tempo
  and key are fixed in this version, and a bar with a key change inside it can be
  edited only as ABC text.

- **What the model does with an edit, measured before the window was built.**
  Four bars with every note changed, in three rap songs: where the new and the
  old line are two or more semitones apart, the new note was sung 16 times in 18
  and 11 in 18 and the old one never, while in the third song the old line won,
  21 to 14; the rest of each song stayed on its score. The lengths inside those
  bars reversed were followed in all three. A phrase twice as fast with a
  half-bar rest was not: the voice filled the rest with the words. A mark on
  every note, the way another pack writes scores, cost nothing measurable. After
  the window was built, one note in each of three phrases of a short piano pop
  song raised from D to A left the old D all three times, twice landing within a
  semitone of the A. The window says that an edit usually takes and not always,
  and what to do when it does not.

- **abcjs 6.7.0, vendored.** `web/js/vendor/abcjs-basic-min.cjs` is the npm
  package's `dist/abcjs-basic-min.js`, unmodified, MIT, with its licence beside
  it. It ends in `.cjs` because ComfyUI imports every `.js` file of a pack as a
  module, and this build is not one; `web/js/vendor/README.md` says so too.

### Changed

- **`YuE2 Plan` and `YuE2 Select Plan` are output nodes.** They hand their score
  to the browser, and ComfyUI runs a node on its own only when it is an output
  node. One consequence: a plan node with nothing connected to it now runs when
  the workflow does, where before it was skipped.
- **The `score_abc` box on `YuE2 Render Plan` is behind the editor.** A saved
  workflow is written exactly as before, a score pasted into the box in an
  earlier version opens in the editor as an edited score, and a score wired into
  the socket is still sung, with the summary naming the node it comes from.
- **Templates 1, 2 and 4 and the README describe the editor**, and template 4
  no longer copies the score out of a preview box to paste it back.

### Fixed

- **Warnings and refusals from the nodes reach the screen.** The nodes have
  always sent them as toasts -- why a run stopped, that `cot` set on a render
  node was ignored, that an index was past the end of a batch -- but nothing in
  the browser listened for them, and the frontend dropped each one into its
  console as an unhandled message. `web/js/yue2_notices.js` listens now and
  shows them as toasts, which is also where the warning about an edit made for
  other words appears.

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
