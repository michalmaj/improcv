"""Verify that a release tag's version matches pyproject.toml's project.version.

Used by .github/workflows/release.yml before any build/publish step runs.
Reads the tag name from the GITHUB_REF_NAME environment variable (set by
GitHub Actions for a tag-triggered run); also runnable locally by setting
GITHUB_REF_NAME manually, e.g.:

    GITHUB_REF_NAME=v0.1.0a1 python3 scripts/verify_release_version.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tomllib


def main() -> int:
    tag = os.environ.get("GITHUB_REF_NAME", "")
    if not tag.startswith("v"):
        print(f"tag {tag!r} does not start with 'v'", file=sys.stderr)
        return 1
    tag_version = tag[1:]

    pyproject_path = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())
    project_version = pyproject["project"]["version"]

    if tag_version != project_version:
        print(
            f"tag version {tag_version!r} does not match pyproject.toml's "
            f"project.version {project_version!r}",
            file=sys.stderr,
        )
        return 1

    print(f"tag version matches pyproject.toml: {project_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
