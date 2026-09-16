"""MIDI files in and out: a file's notes as a score YuE2 sings, and a score written back as a file.

Every module here is plain Python -- no numpy, no torch -- so the whole of it is
checked on a machine that has never downloaded a model. ``smf`` reads and writes
the file format, ``tracks`` describes a file's parts and picks the ones to sing,
``score`` turns them into the rows ``sheetsage.abc_rebuild`` writes a score
from, ``chords`` and ``karaoke`` add harmony and words, and ``export`` goes the
other way.
"""
