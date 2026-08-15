"""Chapter 5: advanced retrieval, hybrid fusion, and adaptive routing.

Book reference:
    Chapter 5, "Hybrid Retrieval and Reciprocal Rank Fusion"
    Chapter 5, "Query Transformation"
    Chapter 5, "Corrective RAG and Adaptive RAG"

Two bugs hide in naive reciprocal rank fusion implementations, and both are
fixed here. Ranks start at one, and fusion keys on a stable chunk identifier
rather than on the chunk text.
"""

import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from shared.retrieval import RetrievalResult, Retriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: Sequence[List[RetrievalResult]],
    weights: Sequence[float],
    k: int = 60,
    top_k: int = 10,
) -> List[RetrievalResult]:
    """Fuse several ranked lists without needing a shared score scale.

    Dense scores are cosine similarities in a narrow band. Lexical scores are
    unbounded sums. Adding them is meaningless, and min-max normalizing them
    per query is unstable because the normalization depends on the retrieved
    batch rather than on the corpus. Rank fusion discards magnitudes and keeps
    the only signal both retrievers agree on: order.

    Fusion keys on chunk_id, never on content, because two retrievers may
    return textually identical chunks that differ by whitespace and would
    otherwise be counted as separate documents.
    """
    if len(ranked_lists) != len(weights):
        raise ValueError("ranked_lists and weights must be the same length")

    fused: Dict[str, float] = defaultdict(float)
    by_id: Dict[str, RetrievalResult] = {}
    contributors: Dict[str, List[str]] = defaultdict(list)

    for results, weight in zip(ranked_lists, weights):
        for rank, result in enumerate(results, start=1):   # one-based
            fused[result.chunk_id] += weight / (k + rank)
            contributors[result.chunk_id].append(result.retriever)
            # Keep the instance carrying the richest metadata.
            if result.chunk_id not in by_id:
                by_id[result.chunk_id] = result

    ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
    if not ordered:
        return []

    best = ordered[0][1] or 1.0
    output: List[RetrievalResult] = []
    for chunk_id, score in ordered[:top_k]:
        original = by_id[chunk_id]
        output.append(
            RetrievalResult(
                content=original.content,
                score=score / best,          # normalized to [0, 1] per contract
                source_uri=original.source_uri,
                chunk_id=chunk_id,
                section_path=original.section_path,
                metadata={
                    **original.metadata,
                    "rrf_raw": score,
                    "found_by": contributors[chunk_id],
                },
                retriever="hybrid_rrf",
            )
        )
    return output


# ---------------------------------------------------------------------------
# First stage retrievers
# ---------------------------------------------------------------------------
def _row_to_result(row: Dict[str, Any], score: float, retriever: str) -> RetrievalResult:
    return RetrievalResult(
        content=row.get("content", ""),
        score=score,
        source_uri=row.get("source_uri", "") or "",
        chunk_id=str(row.get("content_hash") or row.get("id", "")),
        section_path=row.get("section_path", "") or "",
        metadata=row.get("metadata") or {},
        retriever=retriever,
    )


class DenseRetriever(Retriever):
    """Dense vector retrieval over pgvector."""

    name = "dense"

    def __init__(self, embeddings, chapter: str = "ch5", ef_search: int = 100):
        self.embeddings = embeddings
        self.chapter = chapter
        self.ef_search = ef_search

    def retrieve(self, query, tenant_id="default", top_k=5, metadata_filter=None):
        from shared.db_utils import vector_search

        try:
            vector = self.embeddings.embed_query(query)
            rows = vector_search(
                vector,
                tenant_id=tenant_id,
                top_k=top_k,
                chapter=self.chapter,
                metadata_filter=metadata_filter,
                ef_search=self.ef_search,
            )
            return [_row_to_result(r, float(r.get("similarity", 0.0)), self.name)
                    for r in rows]
        except Exception as exc:
            logger.error("dense.retrieve_failed %s", exc)
            return []


class LexicalRetriever(Retriever):
    """Term based retrieval using the materialized tsvector column.

    Chapter 5 argues against an in-process BM25 object: it requires the entire
    corpus in memory on every replica, it rebuilds on every restart, and it
    cannot reflect writes. Pushing lexical search into the database gives an
    incrementally maintained inverted index inside the same transaction as the
    vectors.
    """

    name = "lexical"

    def __init__(self, chapter: str = "ch5"):
        self.chapter = chapter

    def retrieve(self, query, tenant_id="default", top_k=5, metadata_filter=None):
        from shared.db_utils import lexical_search

        try:
            rows = lexical_search(
                query, tenant_id=tenant_id, top_k=top_k, chapter=self.chapter
            )
            if not rows:
                return []
            best = max(float(r.get("rank", 0.0)) for r in rows) or 1.0
            return [
                _row_to_result(r, float(r.get("rank", 0.0)) / best, self.name)
                for r in rows
            ]
        except Exception as exc:
            logger.error("lexical.retrieve_failed %s", exc)
            return []


class HybridRetriever(Retriever):
    """Dense and lexical retrieval fused by rank, executed concurrently."""

    name = "hybrid"

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        dense_weight: float = 0.6,
        k: int = 60,
        candidate_multiplier: int = 4,
    ):
        self.dense = dense
        self.lexical = lexical
        self.dense_weight = dense_weight
        self.k = k
        # Over-retrieve per branch so fusion has material to work with. Fusing
        # two top-K lists down to K means a document ranked eleventh by dense
        # and first by lexical never enters the fusion at all.
        self.candidate_multiplier = candidate_multiplier

    def retrieve(self, query, tenant_id="default", top_k=5, metadata_filter=None):
        candidates = top_k * self.candidate_multiplier

        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(
                self.dense.retrieve, query, tenant_id, candidates, metadata_filter
            )
            lexical_future = pool.submit(
                self.lexical.retrieve, query, tenant_id, candidates, metadata_filter
            )
            dense_results = dense_future.result()
            lexical_results = lexical_future.result()

        return reciprocal_rank_fusion(
            [dense_results, lexical_results],
            [self.dense_weight, 1.0 - self.dense_weight],
            k=self.k,
            top_k=top_k,
        )


class MultiQueryRetriever(Retriever):
    """Generate paraphrases, retrieve for each, and fuse the results."""

    name = "multi_query"

    VARIANT_PROMPT = (
        "Generate {n} alternative phrasings of the question. Vary the "
        "vocabulary. Output one per line, nothing else.\n\nQUESTION: {question}"
    )

    def __init__(self, base: Retriever, llm, variants: int = 3):
        self.base = base
        self.llm = llm
        self.variants = variants

    def _variants(self, question: str) -> List[str]:
        try:
            response = self.llm.invoke(
                self.VARIANT_PROMPT.format(n=self.variants, question=question)
            )
            raw = response.content if hasattr(response, "content") else str(response)
            lines = [line.strip(" -•\t") for line in raw.splitlines() if line.strip()]
            return [question] + lines[: self.variants]
        except Exception as exc:
            logger.error("multi_query.variants_failed %s", exc)
            return [question]

    def retrieve(self, query, tenant_id="default", top_k=5, metadata_filter=None):
        queries = self._variants(query)
        with ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
            lists = list(pool.map(
                lambda q: self.base.retrieve(q, tenant_id, top_k * 2, metadata_filter),
                queries,
            ))
        return reciprocal_rank_fusion(
            lists, [1.0] * len(lists), top_k=top_k
        )


class HyDERetriever(Retriever):
    """Hypothetical Document Embeddings.

    A hypothetical answer resembles a document far more than a question does,
    so the asymmetry between short queries and long passages disappears.

    The original question is always concatenated into the embedded text.
    Embedding the hypothetical alone lets a hallucinated entity dominate
    retrieval, which is HyDE's characteristic failure.
    """

    name = "hyde"

    HYDE_PROMPT = (
        "Write one short factual paragraph that would answer this question. "
        "Do not add caveats.\n\nQUESTION: {question}\n\nPARAGRAPH:"
    )

    def __init__(self, base: Retriever, llm):
        self.base = base
        self.llm = llm

    def expand(self, question: str) -> str:
        try:
            response = self.llm.invoke(self.HYDE_PROMPT.format(question=question))
            hypothetical = (
                response.content if hasattr(response, "content") else str(response)
            ).strip()
        except Exception as exc:
            logger.error("hyde.expand_failed %s", exc)
            return question
        return f"{question}\n\n{hypothetical}" if hypothetical else question

    def retrieve(self, query, tenant_id="default", top_k=5, metadata_filter=None):
        return self.base.retrieve(self.expand(query), tenant_id, top_k, metadata_filter)


# ---------------------------------------------------------------------------
# Adaptive routing and corrective retrieval
# ---------------------------------------------------------------------------
class QueryKind(str, Enum):
    FACTUAL = "factual"          # single fact lookup
    CONCEPTUAL = "conceptual"    # explanation, definition
    COMPARATIVE = "comparative"  # two or more entities compared
    RELATIONAL = "relational"    # needs entity relationships
    AGGREGATE = "aggregate"      # counting, ranking, summarizing


# Each policy is a budget decision, expressed declaratively so an operator can
# retune a route without a deployment.
ROUTING_POLICY: Dict[QueryKind, Dict[str, object]] = {
    QueryKind.FACTUAL:     {"transform": None,          "k": 5,  "rerank": False},
    QueryKind.CONCEPTUAL:  {"transform": "hyde",        "k": 12, "rerank": True},
    QueryKind.COMPARATIVE: {"transform": "multi_query", "k": 20, "rerank": True},
    QueryKind.RELATIONAL:  {"transform": "entities",    "k": 15, "rerank": True},
    QueryKind.AGGREGATE:   {"transform": "multi_query", "k": 30, "rerank": True},
}

CLASSIFY_PROMPT = """Classify the question into exactly one category.

factual      - asks for a single specific fact
conceptual   - asks for an explanation or definition
comparative  - compares two or more things
relational   - asks how entities are connected
aggregate    - asks to count, rank, or summarize across many items

Output the category name only.

QUESTION: {question}
CATEGORY:"""

# Layer one of the classifier cascade. Deterministic, costs microseconds, and
# resolves the large majority of production traffic without a model call.
_RULES = (
    (QueryKind.RELATIONAL,
     ("who reports to", "connected to", "relationship between", "related to",
      "linked to", "supplier of", "owns", "acquired")),
    (QueryKind.COMPARATIVE,
     ("versus", " vs ", "difference between", "compare", "better than",
      "instead of")),
    (QueryKind.AGGREGATE,
     ("how many", "list all", "count", "rank", "top ", "summarize", "across all")),
    (QueryKind.FACTUAL,
     ("when did", "what date", "what is the", "who is the", "where is")),
)


@dataclass
class AdaptiveOutcome:
    """The result of a routed, possibly corrected retrieval."""

    results: List[RetrievalResult] = field(default_factory=list)
    kind: QueryKind = QueryKind.CONCEPTUAL
    attempts: int = 0
    corrections: List[str] = field(default_factory=list)


class AdaptiveRetriever(Retriever):
    """Route by query kind, then correct when the graded context is weak.

    Corrective grading happens before generation. When the evidence cannot
    support an answer, the cheap remedies are still available: rewrite the
    query, widen the search, switch retrievers, or abstain. Correcting a weak
    retrieval costs one retrieval round. Repairing a confident wrong answer
    costs far more.
    """

    name = "adaptive"

    def __init__(
        self,
        hybrid: Retriever,
        reranker,
        llm,
        relevance_floor: float = 0.5,
        max_attempts: int = 2,
    ):
        self.hybrid = hybrid
        self.reranker = reranker
        self.llm = llm
        self.relevance_floor = relevance_floor
        self.max_attempts = max_attempts

    def classify(self, question: str) -> QueryKind:
        """Cascade: rules first, model only when the rules do not fire."""
        lowered = f" {question.lower()} "
        for kind, markers in _RULES:
            if any(marker in lowered for marker in markers):
                return kind

        try:
            response = self.llm.invoke(CLASSIFY_PROMPT.format(question=question))
            raw = (
                response.content if hasattr(response, "content") else str(response)
            ).strip().lower()
            for kind in QueryKind:
                if kind.value in raw:
                    return kind
        except Exception as exc:
            logger.error("adaptive.classify_failed %s", exc)

        return QueryKind.CONCEPTUAL   # safe default, never crash on routing

    def retrieve_adaptive(
        self,
        question: str,
        tenant_id: str = "default",
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> AdaptiveOutcome:
        kind = self.classify(question)
        policy = dict(ROUTING_POLICY[kind])
        corrections: List[str] = []
        query = question

        for attempt in range(1, self.max_attempts + 1):
            if policy["transform"] == "hyde" and hasattr(self.hybrid, "expand"):
                query = self.hybrid.expand(question)

            results = self.hybrid.retrieve(
                query, tenant_id, int(policy["k"]), metadata_filter
            )
            if policy["rerank"] and results and self.reranker is not None:
                results = self.reranker.rerank(question, results, self.llm)

            # Corrective grading, applied before the generator ever runs.
            strong = [r for r in results if r.score >= self.relevance_floor]
            if strong or attempt == self.max_attempts:
                return AdaptiveOutcome(strong or results, kind, attempt, corrections)

            # Weak context, so escalate rather than generate from noise.
            corrections.append(
                f"attempt {attempt}: weak context, widening search"
            )
            policy = {**policy, "k": int(policy["k"]) * 2,
                      "transform": "multi_query"}

        return AdaptiveOutcome([], kind, self.max_attempts, corrections)

    def retrieve(self, query, tenant_id="default", top_k=5, metadata_filter=None):
        outcome = self.retrieve_adaptive(query, tenant_id, metadata_filter)
        return outcome.results[:top_k]
