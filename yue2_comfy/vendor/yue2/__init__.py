"""The seven upstream modules this pack calls, and nothing else.

This file is ours, not upstream's: the upstream package exposes YuE2Pipeline,
which lives in pipeline.py and is deliberately left behind (see ../README.md).
Everything else beside this file is a verbatim copy -- do not edit those.

Attribute access stays lazy on purpose. modeling_yue2 and modeling_vae are the
only vendored files that import names out of transformers, and those names move
between major versions. Keeping the import inside the loader means a future
break in transformers stops one node run with a clear traceback instead of
stopping the whole pack from registering.
"""

from importlib import import_module

UPSTREAM_VERSION = "0.1.6"
UPSTREAM_URL = "https://github.com/multimodal-art-projection/YuE"

_LAZY = {
    "YuE2Config": "modeling_yue2",
    "YuE2ForCausalLM": "modeling_yue2",
    "StaticKVCache": "modeling_yue2",
    "YuE2VAE": "modeling_vae",
    "YuE2VAEConfig": "modeling_vae",
    "YuE2TextTokenizer": "tokenization_yue2",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(import_module("." + module, __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
