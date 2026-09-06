#!/usr/bin/env python3
"""Regenerate README statistics from the current repository state."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def _count_lines(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            total += sum(1 for _ in f)
    return total


def _collect_test_count(repo_root: Path) -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = re.search(r"(\d+)\s+(?:tests?|items?)\s+collected", text, flags=re.IGNORECASE)
    if m is None:
        raise RuntimeError("Could not parse pytest collected test count.")
    return int(m.group(1))


def _apply(pattern: str, replacement: str, text: str) -> str:
    out, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError(f"Expected one match for pattern: {pattern!r}, got {n}")
    return out


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    readme = repo / "README.md"

    test_files = sorted((repo / "tests").glob("*.py"))

    test_lines = _count_lines(test_files)
    test_count = _collect_test_count(repo)

    test_lines_s = f"{test_lines:,}"
    test_count_s = f"{test_count:,}"

    text = readme.read_text(encoding="utf-8")
    bullet_pat = (
        r"- \*\*[0-9,+ ]+ tests\*\* — every optimization is benchmark-gated with "
        r"committed JSON artifacts"
    )
    bullet_rep = (
        f"- **{test_count_s} tests** — every optimization is benchmark-gated "
        "with committed JSON artifacts"
    )
    text = _apply(bullet_pat, bullet_rep, text)
    text = _apply(
        r"# Unit tests \(\d+ tests(?:,[^)]+)?\)",
        f"# Unit tests ({test_count_s} tests)",
        text,
    )
    text = _apply(
        r"tests/\s+# [\d,]+ lines, \d+ tests",
        f"tests/                    # {test_lines_s} lines, {test_count_s} tests",
        text,
    )
    text = _apply(
        r"pytest -q\s+# \d+ tests(?:,[^\n]+)?",
        f"pytest -q                 # {test_count_s} tests",
        text,
    )

    readme.write_text(text, encoding="utf-8")
    print(f"Updated README stats: tests={test_count_s}, test_lines={test_lines_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
