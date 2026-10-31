"""A completed result exports its own settings, including full integer seeds and Unicode lyrics."""

import json

import pytest

from yue2_comfy import checkpoints, constants, generation_info


def test_export_and_extract_keep_the_used_values_and_hide_local_paths():
    settings = dict(constants.DEFAULT_OPTIONS, ode_steps=24,
                    checkpoint_source={"name": "fine.safetensors", "files": {"lm": "/private/fine.safetensors"}},
                    loras=[{"name": "voice", "path": "/private/voice.safetensors", "ar": 0.5, "nar": 0.7}])
    text = generation_info.encode("piano", "\u041f\u0435\u0441\u043d\u044f", (1 << 63) - 1, settings, "X:1")
    settings["ode_steps"] = 64
    data = json.loads(text)
    assert data["steps"] == 24
    assert data["score_abc"] == "X:1"
    assert "/private" not in text
    out = generation_info.YuE2GenerationInfo().extract(text)
    assert out[:5] == ("\u041f\u0435\u0441\u043d\u044f", (1 << 63) - 1, "piano", "fine.safetensors", 24)
    assert "Seed: 9223372036854775807" in out[5]
    assert checkpoints.SOURCE not in data["settings"]


@pytest.mark.parametrize("text", ["", "not json", "[]", "{}", '{"format_version":2}'])
def test_bad_or_unknown_info_has_a_useful_error(text):
    with pytest.raises(ValueError, match="generation_info"):
        generation_info.YuE2GenerationInfo().extract(text)
