"""ComfyUI entry point for the YuE2 music generation nodes.

Module scope stays light on purpose: no torch, no transformers, none of the
vendored modeling code. ComfyUI imports this while it builds the node list, and
an exception here would take the whole pack out rather than one node.

The routes module registers the song editor's HTTP endpoint as it is imported,
and guards that itself, so a server it cannot reach costs the editor its token
cuts and nothing more. The memory module does the same for ComfyUI's Unload
Models button, so that it also lets go of the models this pack keeps loaded.
"""

from .yue2_comfy import memory as _memory  # noqa: F401
from .yue2_comfy import routes as _routes  # noqa: F401
from .yue2_comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
