"""Rendering a GGUF's own chat template, the way Transformers would.

llama-cpp-python has a chat formatter of its own, but it cannot forward
``enable_thinking=False``, and without that flag a reasoning model spends
hundreds of tokens deliberating before it writes a single line of the song. So
the template is taken out of the GGUF metadata and rendered here, in a sandbox
set up like the one Transformers uses.

Ported from the MiniMax-H3 Prompt Rewriter pack.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

TEMPLATE_KEY = "tokenizer.chat_template"
BOS_KEY = "tokenizer.ggml.bos_token_id"
EOS_KEY = "tokenizer.ggml.eos_token_id"


class TemplateError(RuntimeError):
    pass


def _environment():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    def tojson(value, indent=None, **_kwargs):
        return json.dumps(value, ensure_ascii=False, indent=indent)

    def strftime_now(fmt):
        """A fixed value, because the seed contract depends on a fixed prompt.

        Templates that stamp today's date into the system turn are rare, and a
        writer that answers differently tomorrow for the same seed is worse
        than one that does not know the date.
        """
        return ""

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    env.filters["tojson"] = tojson
    return env


def _fold_system(messages: list) -> list:
    """Move a leading system message into the first user turn.

    Not every chat template accepts a system role -- Gemma's calls
    ``raise_exception`` on one, and it is not alone. Since the writer puts its
    whole instruction sheet in that message, dropping it is not an option and
    refusing the model is a poor answer to "run this on anything". Prepending
    it to the first user turn is what those templates' own model cards
    prescribe.

    Returns ``None`` when there is nothing to fold, so the caller can re-raise
    the original error instead of pretending it tried something.
    """
    if not messages or messages[0].get("role") != "system":
        return None
    system = (messages[0].get("content") or "").strip()
    rest = [dict(message) for message in messages[1:]]
    for message in rest:
        if message.get("role") == "user":
            joined = system + "\n\n" + (message.get("content") or "")
            message["content"] = joined.strip()
            return rest
    return None


def render(template: str, messages: list, bos_token: str = "", eos_token: str = "",
           add_generation_prompt: bool = True, enable_thinking: bool = False) -> str:
    if not template:
        raise TemplateError("the model file carries no chat template")

    def attempt(payload):
        return _environment().from_string(template).render(
            messages=payload,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            bos_token=bos_token,
            eos_token=eos_token,
            tools=None,
        )

    try:
        return attempt(messages)
    except Exception as error:
        folded = _fold_system(messages)
        if folded is None:
            raise
        log.debug("[yue2_comfy.chat_template] template rejected the system role (%s), "
                  "folding it into the first user turn", error)
        try:
            return attempt(folded)
        except Exception:
            raise error


def from_metadata(metadata: dict, messages: list, **kwargs) -> str:
    """Render using the ``tokenizer.chat_template`` entry of a GGUF's metadata."""
    template = metadata.get(TEMPLATE_KEY) or metadata.get("chat_template")
    if not template:
        raise TemplateError(
            "this GGUF carries no chat template, so the writer cannot build its prompt"
        )
    return render(template, messages, **kwargs)
