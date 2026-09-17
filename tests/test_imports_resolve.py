"""Every module this pack imports is a file that ships with it.

Most of the pack's own imports happen inside functions rather than at the top
of a module -- nodes.py says why -- and that hides one whole kind of mistake
from everything else that guards a release. A module left out of a commit still
compiles, still loads, and still passes the suite: ``compileall`` never follows
an import, the node-loading check only reaches what nodes.py imports at module
scope, and the tests that would have touched it open with
``pytest.importorskip``, which skips rather than fails when torch is absent, as
it is in CI. The pack would break when somebody pressed Generate.

So this reads every relative import out of the syntax tree, the lazy ones
included, and checks that what it names is a file on disk. It runs where the
other house rules run: on a laptop, in CI, and against the release archive once
that is unpacked. Vendored modules are targets here too, even though their
contents are exempt from the style rules -- a release that shipped without one
would break the same way.

Two things it deliberately does not do. It imports nothing, so a module that
needs torch is checked on a machine that has none. And for
``from .package import name`` it checks the package, not the name, because
without importing there is no way to tell a submodule from a symbol. The form
that names a module unambiguously, ``from . import name``, is checked in full,
and that is the form in which a new module arrives.
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "yue2_comfy"
GUARD = ROOT / ".github" / "workflows" / "release.yml"

NOT_CARRIED = {
    ("yue2_comfy/vendor/yue2/tokenization_yue2.py", ".storage"),
}
"""The one import that names a module this pack deliberately does not carry.

``YuE2TextTokenizer.from_pretrained`` reaches for upstream's storage.py to
download a checkpoint, and storage.py is not one of the modules vendored here
-- see vendor/README.md. Nothing in this pack calls that constructor: the
tokenizer is built from a merges file already on disk. The line stays because
the vendored files are copied verbatim, and it is named here so that the check
below says nothing about it and everything about the next one."""


def _sources() -> list:
    """Every Python file of the pack, vendored code included."""
    return sorted(PACKAGE.rglob("*.py"))


def _module_file(target: pathlib.Path) -> bool:
    """Whether this path is a module: a .py file, or a directory that is a package."""
    return target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()


def _bound(package: pathlib.Path):
    """The names a package's ``__init__`` defines, or None when it defines any.

    A bare ``from . import name`` usually names a module, but it can also reach
    for something the package itself binds. All of this pack's own packages are
    a docstring and nothing else; the vendored one binds a few names and a
    ``__getattr__``, which under PEP 562 can answer for any name at all -- so
    for that one there is nothing left to check.
    """
    init = package / "__init__.py"
    if not init.is_file():
        return set()
    tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == "__getattr__":
                return None
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0]
                         for alias in node.names)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets
                         if isinstance(target, ast.Name))
    return names


def _base(path: pathlib.Path, level: int) -> pathlib.Path:
    """What ``.`` means in a file at ``path``: its own directory, one more up a dot."""
    base = path.parent
    for _ in range(level - 1):
        base = base.parent
    return base


def _unresolved(path: pathlib.Path, root: pathlib.Path = ROOT) -> list:
    """Every relative import in one file that names nothing on disk.

    root is what the report and the NOT_CARRIED keys are written against,
    so the check can be pointed at a stand-in tree as well as at this one.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(root).as_posix()
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        base = _base(path, node.level)
        dots = "." * node.level
        if node.module:
            target = base
            for part in node.module.split("."):
                target = target / part
            if not _module_file(target) and (relative, dots + node.module) not in NOT_CARRIED:
                missing.append("{}:{} imports {}".format(relative, node.lineno,
                                                         dots + node.module))
            continue
        bound = _bound(base)
        for alias in node.names:
            if bound is None or alias.name in bound or _module_file(base / alias.name):
                continue
            if (relative, dots + alias.name) in NOT_CARRIED:
                continue
            missing.append("{}:{} imports {}".format(relative, node.lineno,
                                                     dots + alias.name))
    return missing


def test_every_module_the_pack_imports_is_on_disk():
    missing = []
    for path in _sources():
        missing.extend(_unresolved(path))
    assert not missing, "these imports name a module that is not here: " + "; ".join(missing)


def test_the_check_above_reads_the_lazy_imports_too():
    """The ones that matter most: a module named only inside a function body.

    generate.py imports quantized and vocabulary that way, and nothing else in
    the release would notice if either went missing.
    """
    found = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.ImportFrom) and inner.level and not inner.module:
                    found.extend(alias.name for alias in inner.names)
    assert {"quantized", "vocabulary"} <= set(found)


def test_an_import_of_a_module_that_is_not_here_is_caught(tmp_path):
    """The check, run against a package that is missing one of its modules."""
    package = tmp_path / "yue2_comfy"
    package.mkdir()
    (package / "__init__.py").write_text('"""A stand-in."""\n', encoding="utf-8")
    source = package / "uses.py"
    source.write_text("def run():\n    from . import gone\n    return gone\n",
                      encoding="utf-8")
    assert len(_unresolved(source, tmp_path)) == 1

    (package / "gone.py").write_text('"""Now it is here."""\n', encoding="utf-8")
    assert _unresolved(source, tmp_path) == []


def test_the_release_guard_names_only_files_that_are_here():
    """The hand-written list in release.yml, checked before a tag rather than after.

    That list is what catches a new file committed without being added to git,
    which is the one thing the check above cannot see from a checkout. It earns
    that by being written by hand, and a hand-written list goes stale: a renamed
    module would fail a release for a reason that has nothing to do with the
    release. Here the same mistake fails a test instead.

    The archive does not carry .github, so unpacked there this finds nothing to
    check and says nothing.
    """
    if not GUARD.is_file():
        return
    text = GUARD.read_text(encoding="utf-8")
    start = text.find("for needed in")
    assert start > 0, "release.yml no longer lists the files an archive must hold"
    block = text[start:text.find("done", start)]
    named = []
    for word in block.replace("\\", " ").split():
        word = word.strip(';"')
        if "$" in word or "/" not in word and "." not in word:
            continue
        named.append(word)
    gone = [word for word in named if not (ROOT / word).exists()]
    assert not gone, "release.yml requires files this tree does not have: " + "; ".join(gone)
    assert len(named) > 40, "the guard list looks truncated: {} names".format(len(named))
