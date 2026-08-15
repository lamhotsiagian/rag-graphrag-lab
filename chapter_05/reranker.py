"""Chapter 5: second stage reranking.

Book reference:
    Chapter 5, "Reranking"

Retrieval optimizes recall. Reranking optimizes precision. Retrieve wide and
rerank narrow, because a reranker restricted to the top 5 can only reorder
what the first stage already ranked well.

The parser here is defensive on purpose. The version this replaces called
int(response.content.strip()), so a model answering "Score: 8" raised
ValueError and the handler assigned zero. A formatting variation silently
demoted a relevant chunk to last place.
"""

import logging
import re
from typing import List, Optional

from shared.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

SCORE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")

RERANK_PROMPT = """Rate how well the passage answers the question.

SCALE
0  = unrelated
5  = related topic, does not answer
10 = directly and completely answers

Output a single integer from 0 to 10. No words, no punctuation.

QUESTION: {question}

PASSAGE: {passage}

SCORE:"""


def _parse_score(raw: str, fallback: float) -> float:
    """Extract the first number and clamp it, or fall back to retrieval score.

    Falling back to the retrieval score rather than to zero matters. A parse
    failure is a formatting problem, not evidence that the chunk is
    irrelevant, and scoring it zero actively destroys a good candidate.
    """
    match = SCORE_PATTERN.search(raw or "")
    if not match:
        logger.warning("rerank.parse_failed raw=%r", (raw or "")[:80])
        return fallback
    try:
        return max(0.0, min(10.0, float(match.group(1)))) / 10.0
    except ValueError:
        return fallback


def _batch(llm, prompts: List[str]):
    """Call llm.batch when available, otherwise fall back to a loop.

    One batched call instead of N sequential calls is the difference between
    roughly 4 s and 0.9 s for 30 candidates on a local model.
    """
    if hasattr(llm, "batch"):
        try:
            return llm.batch(prompts)
        except Exception as exc:
            logger.warning("rerank.batch_unavailable %s", exc)
    return [llm.invoke(prompt) for prompt in prompts]


class LLMReranker:
    """Second stage precision ranking using the generator model."""

    def __init__(self, keep: int = 5, score_floor: float = 0.3,
                 passage_chars: int = 1200):
        self.keep = keep
        self.score_floor = score_floor
        self.passage_chars = passage_chars

    def rerank(
        self,
        question: str,
        candidates: List[RetrievalResult],
        llm,
    ) -> List[RetrievalResult]:
        if not candidates:
            return []

        # Truncate passages so a long chunk cannot exhaust the judge's context.
        prompts = [
            RERANK_PROMPT.format(
                question=question, passage=c.content[: self.passage_chars]
            )
            for c in candidates
        ]
        responses = _batch(llm, prompts)

        rescored: List[RetrievalResult] = []
        for candidate, response in zip(candidates, responses):
            raw = (
                response.content if hasattr(response, "content") else str(response)
            )
            score = _parse_score(raw, fallback=candidate.score)
            if score < self.score_floor:
                continue
            rescored.append(
                RetrievalResult(
                    content=candidate.content,
                    score=score,
                    source_uri=candidate.source_uri,
                    chunk_id=candidate.chunk_id,
                    section_path=candidate.section_path,
                    metadata={
                        **candidate.metadata,
                        "first_stage_score": candidate.score,
                    },
                    retriever=f"{candidate.retriever}+rerank",
                )
            )

        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[: self.keep]


def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max scale a score list to [0, 1].

    Kept for the interface comparison view. Note that Chapter 5 argues against
    using this to combine two retrievers, because the normalization depends on
    the retrieved batch rather than on the corpus.
    """
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if low == high:
        return [1.0] * len(scores)
    return [(s - low) / (high - low) for s in scores]


def maximal_marginal_relevance(
    results: List[RetrievalResult],
    keep: int = 5,
    diversity: float = 0.3,
) -> List[RetrievalResult]:
    """Trade a little relevance for diversity in the final context.

    A cheap defence against boilerplate dominance and near duplicate chunks
    crowding out genuinely different evidence.
    """
    if not results:
        return []

    selected: List[RetrievalResult] = [results[0]]
    remaining = list(results[1:])

    while remaining and len(selected) < keep:
        best, best_score = None, float("-inf")
        for candidate in remaining:
            terms = set(candidate.content.lower().split())
            overlap = 0.0
            for chosen in selected:
                chosen_terms = set(chosen.content.lower().split())
                if terms and chosen_terms:
                    overlap = max(
                        overlap,
                        len(terms & chosen_terms) / len(terms | chosen_terms),
                    )
            score = (1 - diversity) * candidate.score - diversity * overlap
            if score > best_score:
                best, best_score = candidate, score
        selected.append(best)
        remaining.remove(best)

    return selected
