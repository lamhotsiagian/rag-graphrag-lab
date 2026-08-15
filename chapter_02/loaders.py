"""Chapter 2: document loaders and safe normalization.

Book reference:
    Chapter 2, "Document Loaders by Format"
    Chapter 2, "Cleaning and Normalization Without Destroying Signal"

The normalization here is deliberately conservative. The version this replaces
stripped every character outside [A-Za-z0-9 .,!?-], which deletes currency
symbols, percent signs, mathematical operators, code punctuation, and every
non-Latin script. A finance corpus loses the difference between 15 and 15
percent. A codebase loses every bracket.
"""

import csv
import io
import re
import unicodedata
from typing import Dict, Tuple

# Control characters excluding tab, newline, and carriage return.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Ligature codepoints that PDF extraction commonly emits as single glyphs.
LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl",
}

# Hyphenation across a line break, for example "retrie-\nval".
SOFT_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")


def clean_text(text: str, preserve_layout: bool = True) -> str:
    """Normalize text without deleting semantically meaningful characters.

    preserve_layout keeps paragraph breaks, which structure aware chunkers
    depend on. Collapsing every newline into a space is the single most common
    way teams destroy their own chunk boundaries.
    """
    if not text:
        return ""

    for ligature, replacement in LIGATURES.items():
        text = text.replace(ligature, replacement)

    # Repair words split across a line break before collapsing whitespace.
    text = SOFT_HYPHEN_BREAK.sub(r"\1\2", text)

    # NFKC folds compatibility forms, for example full width Latin.
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHARS.sub("", text)

    if preserve_layout:
        # Collapse runs of spaces and tabs, keep at most one blank line.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------
def _decode(file) -> str:
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


def load_pdf(file) -> Tuple[str, Dict[str, str]]:
    """Extract a PDF's text layer, page by page.

    pypdf is a stream extractor: fast, dependency light, and blind to layout.
    Chapter 2 explains when to reach for a layout aware parser instead, which
    is any corpus where the answers live in tables.
    """
    from pypdf import PdfReader

    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return clean_text("\n\n".join(pages)), {"page_count": str(len(reader.pages))}


def load_text(file) -> Tuple[str, Dict[str, str]]:
    return clean_text(_decode(file)), {}


def load_markdown(file) -> Tuple[str, Dict[str, str]]:
    """Load Markdown, keeping the heading hierarchy intact.

    Headings are the natural chunk boundaries for this format, so they must
    survive normalization for structure aware chunking to work.
    """
    text = _decode(file)
    title = ""
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return clean_text(text), {"title": title} if title else {}


def load_html(file) -> Tuple[str, Dict[str, str]]:
    """Strip navigation before extracting text.

    Boilerplate dominance, described in Chapter 2, is what happens when the
    site menu is indexed alongside the content: thousands of near identical
    chunks cluster near the corpus centroid and win every weak query.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_decode(file), "lxml")
    for tag in soup(["nav", "header", "footer", "aside", "script", "style", "form"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return clean_text(main.get_text("\n")), {"title": title} if title else {}


def load_csv(file) -> Tuple[str, Dict[str, str]]:
    """Serialize each row with its column names.

    A bare grid of numbers loses the binding between a value and its header,
    which is why numeric questions fail on naively loaded tables. Repeating
    the column name into every row makes each row independently retrievable.
    """
    content = _decode(file).splitlines()
    reader = csv.reader(content)
    rows = list(reader)
    if not rows:
        return "", {}

    header, *body = rows
    lines = []
    for row in body:
        pairs = [
            f"{column.strip()}: {value.strip()}"
            for column, value in zip(header, row)
            if value.strip()
        ]
        if pairs:
            lines.append("; ".join(pairs))

    summary = f"Table with columns: {', '.join(c.strip() for c in header)}."
    return clean_text(summary + "\n\n" + "\n".join(lines)), {
        "row_count": str(len(body)),
        "columns": ", ".join(c.strip() for c in header),
    }


def load_docx(file) -> Tuple[str, Dict[str, str]]:
    """Walk the document tree, preserving heading structure."""
    import docx

    document = docx.Document(file)
    parts = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if paragraph.style.name.startswith("Heading"):
            parts.append(f"\n\n{text}\n")
        else:
            parts.append(text)
    return clean_text("\n".join(parts)), {}


LOADERS = {
    ".pdf": load_pdf,
    ".csv": load_csv,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".docx": load_docx,
    ".txt": load_text,
}


def load_document(file, with_metadata: bool = False):
    """Dispatch to the right loader based on the file extension.

    Returns text by default. Pass with_metadata=True to receive the
    (text, metadata) pair, which the ingestion pipeline needs so that
    filtering is possible at query time.
    """
    filename = getattr(file, "name", "").lower()
    for suffix, loader in LOADERS.items():
        if filename.endswith(suffix):
            text, metadata = loader(file)
            metadata = {"source": getattr(file, "name", ""), **metadata}
            return (text, metadata) if with_metadata else text

    raise ValueError(
        f"Unsupported file type: {filename!r}. "
        f"Supported: {', '.join(sorted(LOADERS))}"
    )
