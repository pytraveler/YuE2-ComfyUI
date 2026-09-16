"""Mel-Band RoFormer in plain torch: the voice of a song, for 'vocals_only' and YuE2 Vocals Only.

``bands`` says which spectrogram bins each of the model's sixty mel bands
reads, with no dependencies. ``network``, ``model`` and ``runtime`` need torch
and are imported only when a voice is actually separated. The weights are
Kimberley Jensen's vocal model, ``KimberleyJSN/melbandroformer``, MIT; the
network follows the Mel-Band RoFormer paper (Wang, Lu and Won, 2023) and the
shapes of lucidrains' MIT implementation that model was trained with.
"""
