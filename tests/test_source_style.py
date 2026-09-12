"""Two house rules about this pack's own source, enforced where CI can see them.

ASCII only, and docstrings rather than inline comments. Both used to live in a
script beside the repository, which meant the rule was real on one machine and
absent everywhere else. Here they travel with the pack: the same check runs on
a laptop, in CI, and against the release archive after it is unpacked.

What each rule covers is written out below rather than inferred from a walk of
the working directory, which on a developer's machine also holds junctions,
scratch files and a virtualenv that no rule of this pack governs.
"""

from __future__ import annotations

import io
import pathlib
import tokenize
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parent.parent

SCANNED_DIRS = ("yue2_comfy", "tests", ".github", "example_workflows")
SCANNED_FILES = ("README.md", "NOTICE.md", "LICENSE", "pyproject.toml",
                 "requirements.txt", "__init__.py")
SUFFIXES = {".py", ".toml", ".txt", ".json", ".yml", ".yaml", ".md"}

SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "vendor", ".pytest_cache",
             "node_modules", "dist", "build", "locales"}
"""Vendored code is carried verbatim, so rewriting its punctuation would make
the diff against upstream unreadable. Translations are exempt for the opposite
reason: for README_RU.md and locales/ the other alphabet is the whole point."""

TRANSLATED = {"README_RU.md"}

DIRECTIVES = ("# noqa", "# pragma", "#!", "# type:", "# -*-")
COMMENT_DIRS = ("yue2_comfy", "tests")


def scanned():
    """Every file the ASCII rule governs, in a stable order."""
    found = []
    for name in SCANNED_FILES:
        path = ROOT / name
        if path.is_file():
            found.append(path)
    for name in SCANNED_DIRS:
        directory = ROOT / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUFFIXES:
                continue
            if SKIP_DIRS.intersection(path.parts) or path.name in TRANSLATED:
                continue
            found.append(path)
    return found


def non_ascii(path: pathlib.Path):
    """``(line, column, character)`` for everything above U+007E in one file."""
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for column, char in enumerate(line, 1):
            if ord(char) > 126:
                yield lineno, column, char


def described(path: pathlib.Path, lineno: int, column: int, char: str) -> str:
    """One offender, in the form an editor can jump to."""
    try:
        name = unicodedata.name(char)
    except ValueError:
        name = "unnamed"
    try:
        where = path.relative_to(ROOT).as_posix()
    except ValueError:
        where = path.as_posix()
    return "{}:{}:{}: U+{:04X} {}".format(where, lineno, column, ord(char), name)


def commented():
    """Every .py file the comment rule governs."""
    found = []
    for name in COMMENT_DIRS:
        directory = ROOT / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if not SKIP_DIRS.intersection(path.parts):
                found.append(path)
    return found


def prose_comments(path: pathlib.Path):
    """``(line, text)`` for every comment addressed to a reader, not a tool."""
    source = path.read_text(encoding="utf-8")
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        if not any(token.string.startswith(prefix) for prefix in DIRECTIVES):
            yield token.start[0], token.string


def test_there_is_something_to_scan():
    """A rule that governs no files passes for the wrong reason."""
    assert len(scanned()) > 20
    assert len(commented()) > 10


def test_the_source_is_ascii_only():
    """A typographic dash survives none of the ways this pack gets read.

    Console code pages that are not UTF-8, editors that guess encodings, issue
    reports retyped by hand: each of those loses one character silently, and
    the file still parses afterwards.
    """
    offenders = []
    for path in scanned():
        offenders.extend(described(path, *found) for found in non_ascii(path))
    assert not offenders, "non-ASCII characters:\n" + "\n".join(offenders)


def test_the_translations_are_exempt_and_really_are_translated():
    """Otherwise the exemption above is silently covering nothing."""
    for name in TRANSLATED:
        path = ROOT / name
        if not path.is_file():
            continue
        assert any(non_ascii(path)), name + " is on the exempt list but is ASCII"


def test_reasons_live_in_docstrings_rather_than_inline_comments():
    """A comment explains a line to whoever already found it.

    A docstring explains a function to everyone who calls it, shows up in
    help() and in an editor's hover, and cannot drift out of sight of what it
    describes. Tool directives are allowed, since moving those into a docstring
    would stop them working.
    """
    offenders = []
    for path in commented():
        for lineno, text in prose_comments(path):
            offenders.append("{}:{}: {}".format(
                path.relative_to(ROOT).as_posix(), lineno, text[:72]))
    assert not offenders, ("inline comments, widen the docstring instead:\n"
                           + "\n".join(offenders))


def test_a_tool_directive_is_not_reported_as_prose(tmp_path):
    """Looped rather than parametrized so this module imports without pytest.

    tools/check_ascii.py reads its rules from here, and a local shortcut that
    needed pytest installed in order to avoid running pytest would be a joke.
    """
    for directive in DIRECTIVES:
        path = tmp_path / "sample.py"
        path.write_text("x = 1  {} something\n".format(directive), encoding="utf-8")
        assert list(prose_comments(path)) == [], directive


def test_a_prose_comment_is_reported(tmp_path):
    """The checker has to be able to fail, or it is decoration."""
    path = tmp_path / "sample.py"
    path.write_text("x = 1  # a prose comment\n", encoding="utf-8")
    assert [text for _line, text in prose_comments(path)] == ["# a prose comment"]


def test_a_typographic_character_is_reported(tmp_path):
    """Written as escapes, because the rule above scans this file too."""
    dash, nbsp = chr(0x2013), chr(0x00A0)
    path = tmp_path / "sample.md"
    path.write_text("an en dash " + dash + " and a nbsp" + nbsp + "here",
                    encoding="utf-8")
    assert [char for _line, _column, char in non_ascii(path)] == [dash, nbsp]
    assert "EN DASH" in described(path, 1, 12, dash)
