"""Chapter 1: RAG v1, the four stages with no external database.

The in-memory design is deliberate. It isolates indexing, retrieval,
augmentation, and generation so you can observe each one before Chapter 3
introduces pgvector.

Book reference:
    Chapter 1, "Building RAG v1"
    Chapter 1, "The Grounded Generation Prompt"
"""

from typing import Any, Dict, List

import numpy as np

ABSTENTION_MESSAGE = (
    "The provided documents do not contain enough information to answer this."
)

GROUNDED_SYSTEM_PROMPT = """You are a document question answering assistant.

RULES
1. Answer using ONLY the numbered context blocks below.
2. Cite every claim with the block number in square brackets, for example [2].
3. If the context does not contain the answer, reply exactly:
   "The provided documents do not contain enough information to answer this."
4. Never use knowledge from outside the context blocks.
5. Keep the answer under 200 words unless the question asks for a list.
"""


# ---------------------------------------------------------------------------
# Stage 1, loading
# ---------------------------------------------------------------------------
def load_document(file) -> str:
    """Parse an uploaded txt, md, or pdf file into text."""
    filename = file.name.lower()

    if filename.endswith(".pdf"):
        from pypdf import PdfReader

        text = ""
        for page in PdfReader(file).pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    if filename.endswith((".txt", ".md")):
        return file.getvalue().decode("utf-8")

    raise ValueError("Unsupported file type. Upload a txt, md, or pdf file.")


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Naive fixed size splitting, the baseline every later strategy must beat."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must exceed overlap")

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Stage 2, indexing and retrieval
# ---------------------------------------------------------------------------
def build_simple_index(chunks: List[str], embeddings_model) -> List[Dict[str, Any]]:
    """Embed every chunk once and hold the vectors in process memory.

    Batching all chunks into a single embed_documents call matters. One HTTP
    round trip to Ollama for 200 chunks costs roughly 400 ms, while 200
    sequential calls cost roughly 12 s on the same hardware.
    """
    index: List[Dict[str, Any]] = []
    if not chunks:
        return index

    embeddings = embeddings_model.embed_documents(chunks)
    for chunk, vector in zip(chunks, embeddings):
        index.append({"content": chunk, "embedding": np.asarray(vector)})
    return index


def cosine_similarity(a, b) -> float:
    """Cosine similarity, the dot product of the L2 normalized vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def retrieve(
    query: str,
    index: List[Dict[str, Any]],
    embeddings_model,
    top_k: int = 3,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Score the query against every chunk and return the strongest matches.

    The min_score floor is the abstention hook. When the best chunk falls
    below the floor, the caller receives an empty list and must refuse to
    answer rather than pass weak evidence to the generator.
    """
    if not index:
        return []

    query_vector = np.asarray(embeddings_model.embed_query(query))

    scored = [
        {
            "content": item["content"],
            "score": cosine_similarity(query_vector, item["embedding"]),
        }
        for item in index
    ]
    scored.sort(key=lambda row: row["score"], reverse=True)
    return [row for row in scored[:top_k] if row["score"] >= min_score]


# ---------------------------------------------------------------------------
# Stage 3, augmentation
# ---------------------------------------------------------------------------
def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Render retrieved chunks as numbered, delimited blocks.

    Numbering is what makes citation possible. A model cannot cite a source it
    cannot address, so an unnumbered blob of concatenated text guarantees
    uncited answers no matter how strict the system prompt sounds.
    """
    blocks = []
    for position, chunk in enumerate(chunks, start=1):
        score = chunk.get("score", 0.0)
        blocks.append(
            f"[{position}] (relevance {score:.3f})\n{chunk['content'].strip()}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Stage 4, generation
# ---------------------------------------------------------------------------
def generate_answer(query: str, context_chunks: List[Dict[str, Any]], llm) -> str:
    """Call the generator with a grounded prompt, or abstain when evidence is empty.

    The empty context branch returns a literal string rather than asking the
    model to abstain. A 3 billion parameter model asked to abstain still
    answers roughly 30 percent of the time, because instruction following
    degrades under distribution shift. Enforce refusal in code, then use the
    model only when evidence exists.
    """
    if not context_chunks:
        return ABSTENTION_MESSAGE

    prompt = (
        f"{GROUNDED_SYSTEM_PROMPT}\n\n"
        f"CONTEXT\n{build_context(context_chunks)}\n\n"
        f"QUESTION\n{query}\n\n"
        f"ANSWER"
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
