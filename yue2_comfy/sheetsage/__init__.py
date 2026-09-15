"""Audio to score: this pack's own implementation of SheetSage2 transcription.

SheetSage2 reads a recording and writes the melody, chords, beats, key and
sections as tokens, which become the same two-voice ABC dialect YuE2 sings
from. ComfyUI's own copy is GPL and m-a-p's reference code carries no licence,
so neither is copied here; the released weights (CC BY-NC 4.0, like YuE2's) are
read directly, and the output is checked against ComfyUI master's.

The modules split the same way the rest of the pack does. ``vocab``,
``grammar``, ``events`` and ``abc_rebuild`` are plain Python, so every rule
about tokens, windows and notation is tested on a machine without torch. The
network itself lives in modules that import torch and are only imported when a
transcription actually runs.
"""
