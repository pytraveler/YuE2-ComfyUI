"""Every translation names things that exist, and keeps the numbers honest.

A locale file is the one part of the pack that ComfyUI reads without the pack
ever looking at it, so nothing here fails at run time. A widget that was renamed
leaves a translation pointing at nothing, and the tooltip simply reverts to
English on that one row -- which nobody notices, because it looks like a row
that was never translated. These checks are the only thing that does notice.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from yue2_comfy import constants
from yue2_comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOCALES = sorted((ROOT / "locales").glob("*/nodeDefs.json"))


def widgets(name: str) -> set:
    spec = NODE_CLASS_MAPPINGS[name].INPUT_TYPES()
    return set(spec.get("required", {})) | set(spec.get("optional", {}))


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_there_is_at_least_one_translation():
    """Otherwise every check below passes by having nothing to check."""
    assert LOCALES


@pytest.mark.parametrize("path", LOCALES, ids=lambda p: p.parent.name)
def test_a_translation_covers_every_node_and_invents_none(path):
    defs = load(path)
    assert set(defs) == set(NODE_CLASS_MAPPINGS)


@pytest.mark.parametrize("path", LOCALES, ids=lambda p: p.parent.name)
def test_a_translated_input_is_a_widget_that_exists(path):
    for name, node in load(path).items():
        stray = sorted(set(node.get("inputs", {})) - widgets(name))
        assert not stray, "{}: {} has no input {}".format(path.parent.name, name, stray)


@pytest.mark.parametrize("path", LOCALES, ids=lambda p: p.parent.name)
def test_translated_outputs_sit_at_the_right_index(path):
    """ComfyUI keys outputs by position, so a wrong index renames the wrong wire."""
    for name, node in load(path).items():
        returned = NODE_CLASS_MAPPINGS[name].RETURN_NAMES
        outputs = node.get("outputs", {})
        assert set(outputs) == {str(i) for i in range(len(returned))}, name
        for index, expected in enumerate(returned):
            assert outputs[str(index)]["name"] == expected, (name, index)


@pytest.mark.parametrize("path", LOCALES, ids=lambda p: p.parent.name)
def test_a_translation_carries_a_display_name_and_a_description(path):
    for name, node in load(path).items():
        assert node.get("display_name"), name
        assert node.get("description", "").strip(), name
        assert name in NODE_DISPLAY_NAME_MAPPINGS


@pytest.mark.parametrize("path", LOCALES, ids=lambda p: p.parent.name)
def test_the_length_tooltip_keeps_the_same_numbers_in_every_language(path):
    """A tooltip is the only place these numbers are ever read.

    The English one is built from the table and cannot drift. A translation is
    typed by hand, so it can, and a tooltip promising eight lines where the
    prompt asks for sixteen is worse than no translation at all.
    """
    node = load(path).get("YuE2WriteSong", {})
    tooltip = node.get("inputs", {}).get("length", {}).get("tooltip", "")
    if not tooltip:
        pytest.skip("this locale does not translate 'length'")
    for choice, lines in constants.WRITER_LENGTH_LINES.items():
        assert choice in tooltip, choice
        assert str(lines) in tooltip, (choice, lines)
