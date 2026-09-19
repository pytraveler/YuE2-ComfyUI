"""LoRA files for YuE2: what they hold, where they are, and folding them into the model.

``formats`` reads a file's header and says what it changes, with no
dependencies, so the list on the node is built and tested without torch.
``catalogue`` finds the files in ComfyUI's LoRA folders. ``apply`` needs torch
and is imported only when a song is actually sung with an adapter.
"""
