"""ComfyUI entry point for the YuE2 music generation nodes.

Module scope stays light on purpose: no torch, no transformers, none of the
vendored modeling code. ComfyUI imports this while it builds the node list, and
an exception here would take the whole pack out rather than one node.
"""

from .yue2_comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
