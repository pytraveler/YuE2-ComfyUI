"""ComfyUI's Unload Models also releases the models this pack keeps loaded."""

import sys
import types

import pytest

from yue2_comfy import memory


class FakeQueue:
    """The two methods of ComfyUI's PromptQueue that the hook touches."""

    def __init__(self):
        self.flags = {}

    def set_flag(self, name, data):
        self.flags[name] = data

    def get_flags(self, reset=True):
        if reset:
            handed, self.flags = self.flags, {}
            return handed
        return dict(self.flags)


class Keeper:
    """A stand-in for a module of the pack that keeps a model: ``is_loaded`` and ``unload``."""

    def __init__(self, loaded=True, fails=False):
        self.loaded = loaded
        self.fails = fails
        self.unloads = 0

    def is_loaded(self):
        return self.loaded

    def unload(self):
        self.unloads += 1
        if self.fails:
            raise RuntimeError("stuck")
        self.loaded = False


def keep(monkeypatch, **modules):
    """Register stand-in keepers by name; returns them."""
    monkeypatch.setattr(memory, "_KEEPERS", [(name, "." + name) for name in modules])
    table = {"." + name: module for name, module in modules.items()}
    monkeypatch.setattr(memory, "importlib",
                        types.SimpleNamespace(import_module=lambda name, package=None: table[name]))
    return modules


def hooked_queue():
    """A fresh queue class with the hook installed on it, and one queue."""
    cls = type("Queue", (FakeQueue,), {})
    assert memory.install(cls)
    return cls()


def test_the_pack_registers_the_song_model_and_the_writer():
    assert ("YuE2", ".loader") in memory.keepers()
    assert ("writer", ".llm") in memory.keepers()
    from yue2_comfy import llm, loader

    for module in (loader, llm):
        assert callable(module.unload) and callable(module.is_loaded)


@pytest.mark.parametrize("flag", ["unload_models", "free_memory"])
def test_either_flag_releases_every_kept_model(monkeypatch, flag):
    kept = keep(monkeypatch, song=Keeper(), writer=Keeper())
    queue = hooked_queue()
    queue.set_flag(flag, True)
    assert queue.get_flags() == {flag: True}
    assert [k.unloads for k in kept.values()] == [1, 1]


@pytest.mark.parametrize("flags", [{}, {"unload_models": False}])
def test_a_quiet_queue_releases_nothing(monkeypatch, flags):
    kept = keep(monkeypatch, song=Keeper())
    queue = hooked_queue()
    queue.flags = dict(flags)
    queue.get_flags()
    assert kept["song"].unloads == 0


def test_a_look_at_the_flags_without_taking_them_releases_nothing(monkeypatch):
    kept = keep(monkeypatch, song=Keeper())
    queue = hooked_queue()
    queue.set_flag("unload_models", True)
    assert queue.get_flags(reset=False) == {"unload_models": True}
    assert queue.get_flags(False) == {"unload_models": True}
    assert kept["song"].unloads == 0
    queue.get_flags()
    assert kept["song"].unloads == 1


def test_one_stuck_model_does_not_keep_the_others_loaded(monkeypatch):
    kept = keep(monkeypatch, stuck=Keeper(fails=True), idle=Keeper(loaded=False), song=Keeper())
    assert memory.release_all() == ["song"]
    assert [k.unloads for k in kept.values()] == [1, 1, 1]


def test_installing_twice_wraps_the_queue_once():
    cls = type("Queue", (FakeQueue,), {})
    assert memory.install(cls) is True
    wrapped = cls.get_flags
    assert memory.install(cls) is False
    assert cls.get_flags is wrapped
    assert cls.get_flags.__wrapped__ is FakeQueue.get_flags


def test_outside_comfyui_there_is_no_queue_to_wrap(monkeypatch):
    """Importing the module must not fail there; installing on the real queue cannot succeed."""
    monkeypatch.setitem(sys.modules, "execution", None)
    with pytest.raises(ImportError):
        memory.install()


@pytest.mark.parametrize("flags, wanted", [
    ({"unload_models": True}, True),
    ({"free_memory": True}, True),
    ({"unload_models": False, "free_memory": True}, True),
    ({"unload_models": False}, False),
    ({}, False),
    (None, False),
])
def test_flags_are_read_the_way_comfyuis_main_loop_reads_them(flags, wanted):
    assert memory.asks_to_release(flags) is wanted
