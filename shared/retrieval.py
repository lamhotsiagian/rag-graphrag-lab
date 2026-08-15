"""The retriever contract.

Chapter 4 of the book introduces this module as the spine every later chapter
builds on. A production RAG system replaces its retriever several times: it
starts dense, gains BM25, gains reranking, gains a graph path, and eventually
gains an agentic router. Each replacement is cheap when the retriever is an
interface with a stable contract, and expensive when it is a function call
scattered across the codebase.

Book reference: Chapter 4, "The Retriever as an Interface, Not a Function".
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved unit, carrying everything the generator needs to cite it.

    Attributes:
        content: The chunk text that will enter the prompt.
        score: Relevance, normalized to [0, 1] by the retriever that produced
            it. Normalizing at the boundary is what lets a caller compare
            results from a dense retriever, a lexical retriever, and a graph
            traversal without knowing which produced them.
        source_uri: Where the chunk came from, used for citation and deletion.
        chunk_id: Stable identifier. Fusion, deduplication, and evaluation all
            key on this, never on the text.
        section_path: Heading breadcrumb, used for precise citation.
        metadata: Free-form attributes carried from ingestion.
        retriever: Which strategy produced this result, for tracing.
    """

    content: str
    score: float
    source_uri: str
    chunk_id: str
    section_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    retriever: str = "unknown"

    def with_score(self, score: float, retriever: Optional[str] = None) -> "RetrievalResult":
        """Return a copy carrying a new score, since the dataclass is frozen."""
        return RetrievalResult(
            content=self.content,
            score=score,
            source_uri=self.source_uri,
            chunk_id=self.chunk_id,
            section_path=self.section_path,
            metadata=dict(self.metadata),
            retriever=retriever or self.retriever,
        )


class Retriever(ABC):
    """Stable interface across dense, hybrid, graph, and agentic retrieval.

    A retriever accepts a query, a tenant, a K, and an optional filter. It
    returns an ordered list of RetrievalResult. It never returns raw database
    rows, it never returns unscored results, and it never returns more than K.
    """

    name: str = "base"

    @abstractmethod
    def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Return at most top_k results, ordered best first."""
        raise NotImplementedError


class FallbackRetriever(Retriever):
    """Try retrievers in order until one returns enough confident results.

    This composition pattern is why the contract matters. Adding a keyword
    fallback behind a dense retriever becomes a constructor argument rather
    than a change to every call site.
    """

    name = "fallback"

    def __init__(
        self,
        retrievers: List[Retriever],
        min_results: int = 1,
        min_score: float = 0.35,
    ):
        if not retrievers:
            raise ValueError("FallbackRetriever needs at least one retriever")
        self.retrievers = retrievers
        self.min_results = min_results
        self.min_score = min_score

    def retrieve(self, query, tenant_id, top_k=5, metadata_filter=None):
        for retriever in self.retrievers:
            results = retriever.retrieve(query, tenant_id, top_k, metadata_filter)
            confident = [r for r in results if r.score >= self.min_score]
            if len(confident) >= self.min_results:
                return confident[:top_k]
        return []   # every retriever failed, so the caller must abstain


class StaticRetriever(Retriever):
    """An in-memory retriever over a fixed result list.

    Present so the unit suite, the evaluation harness, and the interface
    walkthroughs can exercise the full pipeline without Postgres, Neo4j, or
    Ollama running. It scores by simple token overlap, which is deliberately
    crude: it exists to make the plumbing testable, not to retrieve well.
    """

    name = "static"

    def __init__(self, results: List[RetrievalResult]):
        self.results = list(results)

    def retrieve(self, query, tenant_id, top_k=5, metadata_filter=None):
        terms = set(query.lower().split())
        scored: List[RetrievalResult] = []

        for result in self.results:
            if metadata_filter and not all(
                result.metadata.get(k) == v for k, v in metadata_filter.items()
            ):
                continue
            content_terms = set(result.content.lower().split())
            if not content_terms:
                continue
            overlap = len(terms & content_terms) / max(1, len(terms))
            scored.append(result.with_score(round(overlap, 4), self.name))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def normalize_scores(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Rescale a result list so the best score is 1.0.

    Applied at a retriever boundary, never across retrievers. Normalizing
    across two retrievers is exactly the mistake reciprocal rank fusion exists
    to avoid, because the normalization depends on the retrieved batch rather
    than on the corpus.
    """
    if not results:
        return []
    best = max(r.score for r in results)
    if best <= 0:
        return list(results)
    return [r.with_score(r.score / best) for r in results]
