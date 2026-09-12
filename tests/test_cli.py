"""The subprocess backend: the command it builds and when it is chosen.

Measured on 2026-09-12 against Qwen3.5-4B-Q4_K_M, one idea and six seeds: the
binaries wrote a usable song five times out of six and the wheel six out of six,
the miss being the one answer in either backend that began with a paragraph of
prose instead of the style -- which ``complete`` already catches and the node
already retries. Both backends tokenised the same 1339-character prompt to the
same 329 tokens, so the two roads differ in arithmetic, not in what they are
asked.
"""

import pytest

from yue2_comfy import child, cli, devices, llm

MODEL = "models/writer.gguf"
PROMPT = "prompts/prompt.txt"


def build(**kwargs):
    return cli.build_command("llama-completion", MODEL, PROMPT, 2048, 7, **kwargs)


def after(command, flag):
    return command[command.index(flag) + 1]


def test_the_prompt_travels_as_a_file_not_as_an_argument():
    """A multi-line template full of quotes needs no escaping this way."""
    command = build()
    assert after(command, "--file") == PROMPT
    assert not any("<|im_start|>" in part for part in command)


def test_the_interactive_terminal_ui_is_switched_off():
    """llama-cli draws a spinner to stdout and waits; a one-shot run must not."""
    command = build()
    for flag in ("-no-cnv", "-st", "--no-display-prompt", "--no-warmup", "--simple-io"):
        assert flag in command


def test_sampling_flags_are_passed_when_not_greedy():
    command = build(greedy=False, temperature=0.9, top_p=0.95, top_k=40)
    assert after(command, "--temp") == "0.9"
    assert after(command, "--top-p") == "0.95"
    assert after(command, "--top-k") == "40"


def test_greedy_asks_for_no_temperature_at_all():
    command = build(greedy=True)
    assert after(command, "--temp") == "0"
    assert "--top-p" not in command


def test_cpu_offloads_nothing_and_names_no_device():
    command = build(device="cpu")
    assert after(command, "--n-gpu-layers") == "0"
    assert after(command, "--device") == "none"


def test_a_named_card_is_passed_through():
    command = build(device="cuda:1")
    assert after(command, "--device") == "CUDA1"
    assert after(command, "--n-gpu-layers") == str(cli.ALL_LAYERS)


def test_auto_lets_llama_cpp_choose():
    """Only llama.cpp knows whether this build has a Vulkan device or a CUDA one."""
    command = build(device=devices.AUTO)
    assert "--device" not in command


def test_a_large_seed_is_narrowed_to_what_llama_cpp_takes():
    command = cli.build_command("llama-completion", MODEL, PROMPT, 2048, (1 << 62) + 5)
    assert after(command, "--seed") == "5"


def test_the_wheel_is_taken_when_it_is_importable(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "load", lambda *args, **kwargs: "llama")
    monkeypatch.setattr(llm, "generate", lambda *args, **kwargs: "wheel wrote this")
    monkeypatch.setattr(llm, "unload", lambda: None)
    assert llm.run(MODEL, [], 1, 512, "auto") == "wheel wrote this"
    assert llm.backend() == llm.WHEEL


def test_the_binaries_are_taken_when_it_is_not(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)
    seen = {}

    def fake(path, messages, seed, n_ctx, device, keep_loaded, progress, settings,
             **sampling):
        seen.update(path=path, settings=settings)
        return "the binary wrote this"

    monkeypatch.setattr(cli, "run", fake)
    answer = llm.run(MODEL, [], 1, 512, "auto", settings={"download": "off"})
    assert answer == "the binary wrote this"
    assert seen["path"] == MODEL
    assert seen["settings"] == {"download": "off"}
    assert llm.backend() == llm.BINARY


def test_downloading_off_reaches_the_runtime_fetch(monkeypatch):
    """'download' is one switch for weights and runtimes, not two."""
    asked = {}

    def fake_ensure(backend, auto_download, progress):
        asked["auto_download"] = auto_download
        raise RuntimeError("no runtime here")

    monkeypatch.setattr(cli.llamacpp, "ensure", fake_ensure)
    with pytest.raises(RuntimeError) as error:
        cli.run(MODEL, [], 1, 512, "auto", settings={"download": "off"})
    assert asked["auto_download"] is False
    assert "llama-cpp-python" in str(error.value)


def test_a_model_without_a_chat_template_is_named_in_the_refusal(monkeypatch):
    monkeypatch.setattr(llm, "template", lambda path: "")
    with pytest.raises(RuntimeError) as error:
        cli.render_prompt(MODEL, [])
    assert "writer.gguf" in str(error.value)


def test_the_speed_shown_is_the_one_that_wrote_the_song():
    """llama.cpp prints two, and the prompt one is thousands of tokens a second."""
    stderr = (
        "common_perf_print: prompt eval time = 64.62 ms / 329 tokens "
        "(    0.20 ms per token,  5091.46 tokens per second)\n"
        "common_perf_print:        eval time = 353.00 ms / 107 runs "
        "(    3.30 ms per token,   303.11 tokens per second)\n")
    assert child.speed(stderr) == " - 303.11 tok/s"


def test_no_speed_is_no_caption():
    assert child.speed("nothing useful here") == ""


def test_the_end_marker_is_not_part_of_the_song():
    assert child._END_MARKER.sub("", "my coat is heavy now [end of text]").strip() == (
        "my coat is heavy now")
