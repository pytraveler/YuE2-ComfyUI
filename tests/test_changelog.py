"""The two changelogs, held to the shape the release workflow relies on.

The release job cuts the section for the tagged version out of CHANGELOG.md
and CHANGELOG_RU.md, publishes the two as the release notes, and refuses a tag
that either file has no section for. That only runs once a tag is pushed, after
the commit it describes is already public. The same rules are checked here, so
a missing section or a translation that fell behind shows up on the machine
where the entry is being written.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

CHANGELOGS = ("CHANGELOG.md", "CHANGELOG_RU.md")

HEADING = re.compile(r"^## (\d+)\.(\d+)\.(\d+) - (\d{4}-\d{2}-\d{2})$")
"""What release.yml looks for: ``## ``, the version, then `` - `` and a date."""


def sections(name: str) -> list:
    """``[(version, date)]`` for every release in one changelog, top down.

    Every second-level heading is a release. One that does not read as a
    version and a date is reported rather than skipped, because the release
    job would skip it too, and publish a tag with no notes.
    """
    found = []
    text = (ROOT / name).read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.startswith("## "):
            continue
        match = HEADING.match(line)
        assert match, "{}: {!r} is not '## X.Y.Z - YYYY-MM-DD'".format(name, line)
        major, minor, patch, date = match.groups()
        found.append(((int(major), int(minor), int(patch)), date))
    return found


def declared() -> tuple:
    """The version in pyproject.toml, read without tomllib, which 3.10 lacks."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', text, re.MULTILINE)
    assert match, "no version in pyproject.toml"
    return tuple(int(part) for part in match.groups())


def test_both_changelogs_have_releases_in_them():
    """A check over an empty list passes for the wrong reason."""
    for name in CHANGELOGS:
        assert sections(name), name + " has no release sections"


def test_both_changelogs_describe_the_same_releases():
    """Release notes in one language only are half a release."""
    english, russian = (sections(name) for name in CHANGELOGS)
    assert english == russian


def test_the_newest_release_comes_first_and_none_twice():
    """A changelog is read from the top; an entry filed lower down is missed."""
    for name in CHANGELOGS:
        versions = [version for version, _date in sections(name)]
        assert versions == sorted(set(versions), reverse=True), name


def test_the_declared_version_has_a_section():
    """Otherwise the release job refuses the tag, and only once it is pushed."""
    wanted = declared()
    for name in CHANGELOGS:
        versions = [version for version, _date in sections(name)]
        assert wanted in versions, "{} has no section for {}".format(
            name, ".".join(str(part) for part in wanted))
