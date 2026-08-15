"""Chapter 2: chunking strategies.

Book reference:
    Chapter 2, "Chunking Strategies"
    Chapter 2, "Chunk Size and Overlap Trade-offs"

Four strategies, from the deterministic baseline to the one that resolves the
tension between retrieval precision and generation context.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# ---------------------------------------------------------------------------
# Fixed size, the baseline
# ---------------------------------------------------------------------------
def fixed_size_chunking(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """Slice the character stream every `size` characters.

    Deterministic, trivially parallel, and completely blind to meaning. It
    splits sentences, tables, and code blocks without hesitation, and it is
    the baseline every other strategy must beat.
    """
    if size <= overlap:
        raise ValueError("size must exceed overlap")
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Recursive, the default for prose
# ---------------------------------------------------------------------------
def recursive_chunking(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """Try a hierarchy of separators, splitting on the coarsest that fits.

    Respects paragraph structure most of the time while still guaranteeing a
    maximum size, which makes it the correct default for prose.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text) if text else []


# ---------------------------------------------------------------------------
# Semantic, boundary detection by embedding distance
# ---------------------------------------------------------------------------
def semantic_chunking(
    text: str,
    embeddings_model,
    breakpoint_percentile: int = 85,
    min_chunk_chars: int = 200,
    max_chunk_chars: int = 1500,
) -> List[str]:
    """Split on semantic discontinuity rather than on character count.

    The algorithm embeds every sentence, measures cosine distance between
    consecutive sentences, and cuts wherever the distance exceeds the given
    percentile. A high percentile produces few large chunks, and a low
    percentile produces many small ones, so treat it as a tuned parameter.

    Costs one embedding call per sentence at ingestion, which is 20 to 60
    times the cost of recursive chunking. Chapter 2 sets out when that is
    worth paying and when it is not.
    """
    sentences = [s.strip() for s in SENTENCE_BOUNDARY.split(text) if s.strip()]
    if len(sentences) < 3:
        return [text.strip()] if text.strip() else []

    # One batched call, not one call per sentence.
    vectors = np.asarray(embeddings_model.embed_documents(sentences), dtype=np.float64)
    # L2 normalize so the dot product equals cosine similarity.
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10

    similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
    distances = 1.0 - similarities
    threshold = float(np.percentile(distances, breakpoint_percentile))

    chunks: List[str] = []
    current: List[str] = [sentences[0]]
    current_len = len(sentences[0])

    for index, sentence in enumerate(sentences[1:]):
        crosses_topic = distances[index] > threshold
        too_long = current_len + len(sentence) > max_chunk_chars
        long_enough = current_len >= min_chunk_chars

        if (crosses_topic and long_enough) or too_long:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Parent child, precision plus context
# ---------------------------------------------------------------------------
@dataclass
class ChildChunk:
    """A small, precisely embeddable unit that points back at its parent."""

    content: str
    parent_id: str
    child_id: str
    order: int
    metadata: Dict[str, str] = field(default_factory=dict)


def _stable_id(source_uri: str, text: str, position: int) -> str:
    """Content addressed identifier, which makes reingestion idempotent."""
    digest = hashlib.sha256(
        f"{source_uri}|{position}|{text}".encode("utf-8")
    ).hexdigest()
    return digest[:24]


def parent_child_chunking(
    text: str,
    source_uri: str = "",
    section_path: str = "",
    parent_chars: int = 2000,
    child_chars: int = 400,
    child_overlap: int = 60,
) -> Tuple[Dict[str, str], List[ChildChunk]]:
    """Produce a parent lookup table plus the child chunks that get embedded.

    Only children are embedded and searched. At query time the retriever
    resolves each hit to its parent, deduplicates parents, and passes the
    parent text to the generator. Precision comes from the child, and context
    comes from the parent.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chars,
        chunk_overlap=0,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chars,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parents: Dict[str, str] = {}
    children: List[ChildChunk] = []
    if not text.strip():
        return parents, children

    for parent_position, parent_text in enumerate(parent_splitter.split_text(text)):
        parent_id = _stable_id(source_uri, parent_text, parent_position)
        parents[parent_id] = parent_text

        for child_position, child_text in enumerate(
            child_splitter.split_text(parent_text)
        ):
            # The breadcrumb anchors a mid document chunk to its topic.
            anchored = f"{section_path}\n{child_text}" if section_path else child_text
            children.append(
                ChildChunk(
                    content=anchored,
                    parent_id=parent_id,
                    child_id=_stable_id(source_uri, child_text, child_position),
                    order=child_position,
                    metadata={"source_uri": source_uri, "section_path": section_path},
                )
            )

    return parents, children


# ---------------------------------------------------------------------------
# Structure aware, for formats that carry a tree
# ---------------------------------------------------------------------------
HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def structure_aware_chunking(text: str, max_chars: int = 1200) -> List[str]:
    """Split Markdown on headings and prefix each chunk with its breadcrumb.

    Chapter 2 argues this is the highest value change available per
    engineering hour on any corpus that carries real headings: it repairs the
    most common ingestion defect at almost no cost.
    """
    matches = list(HEADING.finditer(text))
    if not matches:
        return recursive_chunking(text, max_chars, max_chars // 10)

    breadcrumb: List[str] = []
    chunks: List[str] = []

    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        breadcrumb = breadcrumb[: level - 1]
        breadcrumb.append(title)

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue

        prefix = " > ".join(breadcrumb)
        for piece in recursive_chunking(body, max_chars, max_chars // 10):
            chunks.append(f"{prefix}\n{piece}")

    return chunks


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def get_chunk_stats(chunks: List[str]) -> Dict[str, Any]:
    """Summary statistics for the comparison lab."""
    if not chunks:
        return {"count": 0, "avg_size": 0, "min_size": 0,
                "max_size": 0, "total_chars": 0, "size_stddev": 0}

    sizes = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "avg_size": sum(sizes) / len(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "total_chars": sum(sizes),
        "size_stddev": float(np.std(sizes)),
    }


def overlap_ratio(chunks: List[str]) -> float:
    """Fraction of adjacent chunk pairs that share substantial text.

    Chapter 2 warns that high overlap fills the top K with near duplicates, so
    a comparison lab should be able to show the effect rather than assert it.
    """
    if len(chunks) < 2:
        return 0.0

    shared = 0
    for left, right in zip(chunks, chunks[1:]):
        left_terms = set(left.lower().split())
        right_terms = set(right.lower().split())
        if not left_terms or not right_terms:
            continue
        jaccard = len(left_terms & right_terms) / len(left_terms | right_terms)
        if jaccard >= 0.4:
            shared += 1
    return shared / (len(chunks) - 1)
