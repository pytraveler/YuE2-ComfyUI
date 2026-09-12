"""Writing a song through the ``llama-completion`` binary, in a subprocess.

The backend of last resort, and a surprisingly good one. It needs nothing
installed into ComfyUI's Python: see ``llamacpp.py`` for why the wheel is worth
avoiding when it is not already there.

A subprocess reloads the model on every run, which sounds expensive and is not.
The node's own default is ``keep_model_loaded = False``, because the card is
wanted for the song the moment the words are written -- and in that mode the
in-process backend unloads after every run too. What this backend genuinely
cannot do is honour ``keep_model_loaded = True``; that is reported rather than
silently ignored.

Two things come free with the process boundary: VRAM is returned by the
operating system rather than by hoping a deallocator ran, and an llama.cpp crash
takes down a child process instead of ComfyUI and its queue.

The prompt travels through ``--file`` rather than the command line, so a
multi-line template full of quotes needs no shell escaping on any platform.

Nothing here watches for a model that has begun repeating itself, which the pack
this came from does. A song repeats: the same chorus is meant to come back, and
a detector that stops on it would cut the second half off every well-formed
answer. ``--predict`` is the limit instead.
"""

from __future__ import annotations

import logging
import os
import tempfile

from . import chat_template, child, devices, llamacpp
from .constants import WRITER_MAX_NEW_TOKENS

log = logging.getLogger(__name__)

ALL_LAYERS = 999
CHARS_PER_TOKEN = 4.0
PREVIEW_TAIL = 240


def available() -> bool:
    return llamacpp.available()


def render_prompt(model_path: str, messages: list) -> str:
    """The prompt string, from the chat template in the model's own header."""
    from . import llm

    template = llm.template(model_path)
    if not template:
        raise RuntimeError(
            "'" + os.path.basename(model_path) + "' carries no chat template, so there "
            "is no way to tell it which part of the prompt is an instruction.")
    return chat_template.render(template, messages, "", "", enable_thinking=False)


def build_command(binary: str, model_path: str, prompt_file: str, n_ctx: int, seed: int,
                  device: str = devices.AUTO, greedy: bool = False,
                  max_new_tokens: int = WRITER_MAX_NEW_TOKENS, temperature: float = 0.9,
                  top_p: float = 0.95, top_k: int = 40,
                  repetition_penalty: float = 1.05) -> list:
    """One-shot completion, with the interactive terminal UI switched off."""
    layers = devices.layers_for(device, -1)
    command = [
        binary,
        "--model", model_path,
        "--file", prompt_file,
        "--n-gpu-layers", str(ALL_LAYERS if layers < 0 else layers),
        "--ctx-size", str(int(n_ctx)),
        "--predict", str(int(max_new_tokens)),
        "--seed", str(int(seed) & 0xFFFFFFFF),
        "--repeat-penalty", "{:g}".format(float(repetition_penalty)),
        "-no-cnv",
        "-st",
        "--no-display-prompt",
        "--no-warmup",
        "--simple-io",
    ]
    command += devices.llama_arguments(device)
    if greedy:
        command += ["--temp", "0"]
    else:
        command += [
            "--temp", "{:g}".format(float(temperature)),
            "--top-p", "{:g}".format(float(top_p)),
            "--top-k", str(int(top_k)),
        ]
    return command


def generate(binary: str, model_path: str, messages: list, seed: int, n_ctx: int,
             device: str = devices.AUTO, progress=None, **sampling) -> str:
    """One completion, streamed so the caption moves and Cancel is answered."""
    from . import llm

    rendered = render_prompt(model_path, messages)
    max_new_tokens = int(sampling.get("max_new_tokens", WRITER_MAX_NEW_TOKENS))

    handle, prompt_file = tempfile.mkstemp(prefix="yue2_writer_", suffix=".txt")
    with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
        file.write(rendered)

    command = build_command(binary, model_path, prompt_file, n_ctx, seed, device, **sampling)
    llm.free_comfy_vram(device)

    if progress is not None:
        where = "" if device == devices.AUTO else " on " + device
        progress.set_total(max(max_new_tokens, 1))
        progress.text("Loading " + os.path.basename(model_path) + "\nllama.cpp binary"
                      + where, force=True)

    def report(whole: str) -> bool:
        if progress is not None:
            progress.update(
                min(len(whole) / CHARS_PER_TOKEN, float(max_new_tokens)),
                "Writing\n{} chars\n{}".format(len(whole), whole[-PREVIEW_TAIL:]))
        return False

    try:
        text, stderr_text = child.run(command, binary, report)
    except child.ChildFailed as error:
        raise RuntimeError(str(error)) from error
    finally:
        try:
            os.unlink(prompt_file)
        except OSError:
            log.debug("[yue2_comfy.cli] could not remove %s", prompt_file)

    if progress is not None:
        progress.finish("Written - {} chars{}".format(len(text), child.speed(stderr_text)))
    return text


def run(path: str, messages: list, seed: int, n_ctx: int, device: str,
        keep_loaded: bool = False, progress=None, settings=None, **sampling) -> str:
    """Fetch the runtime if it must, write once, and let the process go.

    A failure to get a runtime at all is the one place the wheel is worth
    mentioning: that reader has no llama.cpp, cannot download one, and
    installing the package is the road they have left.
    """
    from . import llm

    settings = settings or {}
    auto_download = str(settings.get("download") or "auto").lower() != "off"
    try:
        binary = llamacpp.ensure("auto", auto_download, progress)
    except RuntimeError as error:
        raise RuntimeError(str(error) + "\n\n" + llm.install_hint()) from error

    if keep_loaded:
        log.debug("[yue2_comfy.cli] keep_model_loaded has no effect here: the model "
                  "leaves with the subprocess")
    return generate(binary, path, messages, seed, n_ctx, device, progress, **sampling)
