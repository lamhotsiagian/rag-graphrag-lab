"""Fakes that let the whole pipeline run with no Docker, Ollama, or network.

The point of these is not to simulate a language model. It is to make the
plumbing deterministic so a test can assert on behaviour the book claims:
that a parse failure falls back rather than scoring zero, that a repeated
query is suppressed, that fusion keys on identifiers, and so on.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.retrieval import RetrievalResult  # noqa: E402


class FakeResponse:
    """Mimics a LangChain message, which exposes .content."""

    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Scriptable language model.

    Pass `replies` for a fixed sequence, or `router` for a function that maps
    a prompt to a reply. Records every prompt so a test can assert on what was
    actually sent, which is how the batching and retry claims get verified.
    """

    def __init__(self, replies: Optional[List[str]] = None,
                 router: Optional[Callable[[str], str]] = None,
                 default: str = "ok"):
        self.replies = list(replies or [])
        self.router = router
        self.default = default
        self.prompts: List[str] = []
        self.batch_calls = 0

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if self.router is not None:
            return FakeResponse(self.router(prompt))
        if self.replies:
            return FakeResponse(self.replies.pop(0))
        return FakeResponse(self.default)

    def batch(self, prompts: List[str]):
        self.batch_calls += 1
        return [self.invoke(p) for p in prompts]


class NoBatchLLM(FakeLLM):
    """A model without .batch, so the loop fallback path gets exercised."""

    batch = None

    def __getattribute__(self, name):
        if name == "batch":
            raise AttributeError("batch")
        return object.__getattribute__(self, name)


class FakeEmbeddings:
    """Deterministic bag-of-characters embeddings.

    Similar strings produce similar vectors, which is all the retrieval and
    chunking tests need. No model, no network, fully reproducible.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim
        self.document_calls = 0
        self.query_calls = 0

    def _vector(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        for index, char in enumerate(text.lower()):
            vector[(ord(char) + index % 3) % self.dim] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.document_calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        self.query_calls += 1
        return self._vector(text)


def make_result(chunk_id: str, content: str, score: float = 0.9,
                retriever: str = "test", **kwargs) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=score,
        source_uri=kwargs.pop("source_uri", f"doc-{chunk_id}.md"),
        chunk_id=chunk_id,
        section_path=kwargs.pop("section_path", ""),
        metadata=kwargs.pop("metadata", {}),
        retriever=retriever,
    )


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


@pytest.fixture
def sample_results() -> List[RetrievalResult]:
    return [
        make_result("a", "Retrieval augmented generation grounds answers in sources.", 0.91),
        make_result("b", "HNSW is a layered proximity graph index for vector search.", 0.80),
        make_result("c", "BM25 ranks by term frequency adjusted for document length.", 0.71),
        make_result("d", "Neo4j stores entities and relationships as a property graph.", 0.62),
    ]
