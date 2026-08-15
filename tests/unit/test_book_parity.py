"""The acceptance test for the book's central claim.

The book tells the reader to run the code it prints. This test fails the build
if any symbol shown in a chapter listing is missing from the lab, so the gap
that once existed cannot silently reopen.
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EBOOK_DIR = PROJECT_ROOT.parent / "ebook"
CHECKER = PROJECT_ROOT / "scripts" / "check_book_parity.py"


@pytest.mark.skipif(
    not EBOOK_DIR.exists() or not list(EBOOK_DIR.glob("chapter_*.tex")),
    reason="ebook sources not checked out beside the lab",
)
def test_every_symbol_in_the_book_exists_in_the_lab():
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--ebook", str(EBOOK_DIR)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "The book shows code that the lab does not implement.\n\n"
        + result.stdout + result.stderr
    )


def test_the_checker_itself_detects_a_missing_symbol(tmp_path):
    """A checker that cannot fail is not a check."""
    ebook = tmp_path / "ebook"
    ebook.mkdir()
    (ebook / "chapter_01.tex").write_text(
        "\\begin{lstlisting}[style=python]\n"
        "def a_function_that_does_not_exist_anywhere():\n"
        "    pass\n"
        "\\end{lstlisting}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--ebook", str(ebook)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "a_function_that_does_not_exist_anywhere" in result.stdout
