"""Execute the ``>>>`` examples embedded in the Markdown documentation.

Documentation numbers go stale silently. This test extracts every fenced
``python`` block that contains doctest prompts and runs it, sharing state between
blocks of one page in reading order, so a stale value in the docs fails CI.
"""

from __future__ import annotations

import doctest
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
FLAGS = doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE


def _pages() -> list[Path]:
    pages = []
    for path in sorted(DOCS.rglob("*.md")):
        blocks = FENCE.findall(path.read_text(encoding="utf-8"))
        if any(">>>" in b for b in blocks):
            pages.append(path)
    return pages


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(DOCS)))
def test_documentation_examples_run(page: Path) -> None:
    blocks = [b for b in FENCE.findall(page.read_text(encoding="utf-8")) if ">>>" in b]
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=FLAGS, verbose=False)
    globs: dict[str, object] = {}
    failures: list[str] = []
    for i, block in enumerate(blocks):
        test = parser.get_doctest(block, globs, f"{page.name}[block {i}]", str(page), 0)
        output: list[str] = []
        runner.run(test, out=output.append, clear_globs=False)
        failures.extend(output)
        globs = test.globs
    assert not failures, "\n".join(failures)
