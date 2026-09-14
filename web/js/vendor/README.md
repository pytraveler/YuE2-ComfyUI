# Vendored browser code -- do not edit

`abcjs-basic-min.cjs` is `dist/abcjs-basic-min.js` from the npm package
[abcjs](https://github.com/paulrosen/abcjs) 6.7.0, byte for byte
(sha256 `b0cde4bc52bb33949181683a245005fff8a024a8c1f07ec6ce3222cd4bd72e51`).
MIT, Copyright (c) 2009-2026 Paul Rosen and Gregory Dyke; the licence is in
`abcjs-LICENSE.md` next to it. The score editor uses it for one thing: drawing
the score as notes in its Notes tab. The YuE2 demo page draws its scores with
the same library.

## Why the file does not end in .js

ComfyUI imports every `.js` file under a pack's web directory as an extension
module, and abcjs is a UMD bundle, not a module: imported that way it throws
before the page has finished loading. With another extension ComfyUI still
serves the file but leaves it alone, and `yue2_score.js` fetches it the first
time the Notes tab is opened and runs it as a classic script.

## The ASCII rule does not apply here

The bundle carries a copyright sign and musical symbols in its strings.
Rewriting them would mean a fork of a minified file. The source style check
skips this directory, as it skips `yue2_comfy/vendor/`.
