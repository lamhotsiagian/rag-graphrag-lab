"""Chapter 6: evaluation, observability, and debugging.

Book reference:
    Chapter 6, "Retrieval Metrics"
    Chapter 6, "Generation Metrics"
    Chapter 6, "Building a Golden Dataset That Survives"

Two principles run through this module.

Retrieval and generation are measured separately, because they fail
independently and the correct engineering response differs by quadrant.

A judge failure is missing data, not a zero. The version this replaces called
float(res.content.strip()), so any judge reply other than a bare number raised
an exception that the handler recorded as 0.0. That makes the metric measure
the judge's formatting compliance rather than the system's groundedness.
"""

import json
import logging
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def precision_at_k(retrieved_ids: Sequence[str],
                   relevant_ids: Sequence[str], k: int) -> float:
    """Fraction of the top k that is relevant.

    Divides by the number actually retrieved, not by k. Dividing by k
    penalizes a query whose corpus simply contains fewer than k chunks.
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    top = list(retrieved_ids)[:k]
    hits = len(set(top) & set(relevant_ids))
    return hits / min(k, len(top))


def recall_at_k(retrieved_ids: Sequence[str],
                relevant_ids: Sequence[str], k: int) -> float:
    """Fraction of all relevant chunks appearing in the top k."""
    if not relevant_ids:
        return 1.0     # nothing to find, so nothing was missed
    hits = len(set(list(retrieved_ids)[:k]) & set(relevant_ids))
    return hits / len(set(relevant_ids))


def reciprocal_rank(retrieved_ids: Sequence[str],
                    relevant_ids: Sequence[str]) -> float:
    """Reciprocal of the rank of the first relevant result."""
    relevant = set(relevant_ids)
    for position, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str],
              relevance_grades: Dict[str, int], k: int) -> float:
    """Graded relevance ranking quality, normalized against the ideal order.

    relevance_grades maps chunk_id to a grade, conventionally
    2 = directly answers, 1 = supporting context, 0 = irrelevant.
    """
    def dcg(grades: Sequence[int]) -> float:
        return sum(
            (2 ** grade - 1) / math.log2(position + 1)
            for position, grade in enumerate(grades, start=1)
        )

    actual = [relevance_grades.get(cid, 0) for cid in list(retrieved_ids)[:k]]
    ideal = sorted(relevance_grades.values(), reverse=True)[:k]

    ideal_dcg = dcg(ideal)
    return dcg(actual) / ideal_dcg if ideal_dcg > 0 else 0.0


def hit_rate_at_k(retrieved_ids: Sequence[str],
                  relevant_ids: Sequence[str], k: int) -> float:
    """Binary: did any relevant chunk make the top k?

    This is the metric that best predicts end to end answer quality, because
    the generator needs one good chunk far more than it needs a perfectly
    ordered list.
    """
    return 1.0 if set(list(retrieved_ids)[:k]) & set(relevant_ids) else 0.0


# ---------------------------------------------------------------------------
# Generation metrics, judged
# ---------------------------------------------------------------------------
CLAIM_EXTRACT_PROMPT = """Break the answer into atomic factual claims.

RULES
1. One verifiable statement per claim.
2. Resolve pronouns so each claim stands alone.
3. Keep conditionals intact, do not split them into parts.
4. Ignore hedges, greetings, and meta commentary.
5. Output a JSON array of strings and nothing else.

ANSWER
{answer}

JSON"""

CLAIM_VERDICT_PROMPT = """Decide whether the context supports the claim.

Answer with exactly one word:
SUPPORTED    - the context states or directly entails the claim
CONTRADICTED - the context states the opposite
UNSUPPORTED  - the context is silent on this claim

CONTEXT
{context}

CLAIM
{claim}

VERDICT:"""

ANSWER_RELEVANCE_PROMPT = """Does the answer address the question that was asked?

Answer with exactly one word: YES or NO.

QUESTION
{question}

ANSWER
{answer}

VERDICT:"""

VERDICTS = ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED")

# Matched most specific first, with word boundaries. Plain substring matching
# is wrong here: "SUPPORTED" is a substring of "UNSUPPORTED", so a naive
# `if verdict in text` check silently scores every unsupported claim as
# supported and reports a faithfulness of 1.0 on an ungrounded answer.
_VERDICT_PATTERNS = (
    ("CONTRADICTED", re.compile(r"\bCONTRADICTED\b")),
    ("UNSUPPORTED", re.compile(r"\bUNSUPPORTED\b")),
    ("SUPPORTED", re.compile(r"\bSUPPORTED\b")),
)


def _match_verdict(text: str) -> Optional[str]:
    """Return the verdict named in the reply, or None when it is unparseable."""
    upper = (text or "").upper()
    for label, pattern in _VERDICT_PATTERNS:
        if pattern.search(upper):
            return label
    return None


def _extract_json_array(raw: str) -> Optional[list]:
    """Pull the first JSON array out of a possibly chatty model reply."""
    if not raw:
        return None
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start:end + 1])
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        logger.warning("judge.json_parse_failed raw=%r", raw[:120])
        return None


def _text_of(response) -> str:
    return response.content if hasattr(response, "content") else str(response)


def _batch(judge_llm, prompts: List[str]):
    if hasattr(judge_llm, "batch"):
        try:
            return judge_llm.batch(prompts)
        except Exception as exc:
            logger.warning("judge.batch_unavailable %s", exc)
    return [judge_llm.invoke(p) for p in prompts]


def faithfulness_score(answer: str, context: str, judge_llm) -> Dict[str, object]:
    """Claim level groundedness.

    Returns None for the score when the judge itself failed, so the caller
    excludes the sample from the average instead of recording a zero.

    Claim decomposition beats paragraph scoring on three axes: a binary
    entailment judgement on a short pair is far more stable than an aggregate
    score, the report names which claims failed, and a single fabricated
    sentence in an otherwise correct answer becomes visible instead of barely
    moving a paragraph score.
    """
    response = judge_llm.invoke(CLAIM_EXTRACT_PROMPT.format(answer=answer))
    claims = _extract_json_array(_text_of(response))

    if not claims:
        return {"score": None, "reason": "claim_extraction_failed",
                "claims": [], "verdicts": []}

    claims = [c for c in claims if isinstance(c, str) and c.strip()]
    if not claims:
        return {"score": None, "reason": "no_usable_claims",
                "claims": [], "verdicts": []}

    prompts = [
        CLAIM_VERDICT_PROMPT.format(context=context[:6000], claim=claim)
        for claim in claims
    ]
    responses = _batch(judge_llm, prompts)   # one round trip, not N

    verdicts: List[str] = []
    for response in responses:
        verdicts.append(_match_verdict(_text_of(response)) or "UNPARSEABLE")

    scored = [v for v in verdicts if v in VERDICTS]
    if not scored:
        return {"score": None, "reason": "all_verdicts_unparseable",
                "claims": claims, "verdicts": verdicts}

    supported = sum(1 for v in scored if v == "SUPPORTED")
    return {
        "score": supported / len(scored),
        "contradicted": sum(1 for v in scored if v == "CONTRADICTED"),
        "unsupported": sum(1 for v in scored if v == "UNSUPPORTED"),
        "claims": claims,
        "verdicts": verdicts,
    }


def answer_relevance_score(question: str, answer: str, judge_llm) -> Optional[float]:
    """Does the answer address the question? None when the judge fails.

    A faithful answer can still be irrelevant, for example when it correctly
    summarizes the wrong part of the context, so this must be measured
    alongside faithfulness rather than instead of it.
    """
    try:
        response = judge_llm.invoke(
            ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
        )
        text = _text_of(response).strip().upper()
    except Exception as exc:
        logger.warning("judge.relevance_failed %s", exc)
        return None

    if text.startswith("YES"):
        return 1.0
    if text.startswith("NO"):
        return 0.0
    return None


def context_relevance_score(question: str, contexts: List[str],
                            judge_llm) -> Optional[float]:
    """Fraction of retrieved chunks that address the question.

    The cheapest useful generation-side metric to add first, because it
    isolates retrieval quality without requiring chunk level labels.
    """
    if not contexts:
        return 0.0

    prompts = [
        f"Does this passage help answer the question? Answer YES or NO.\n\n"
        f"QUESTION\n{question}\n\nPASSAGE\n{c[:2000]}\n\nVERDICT:"
        for c in contexts
    ]
    try:
        responses = _batch(judge_llm, prompts)
    except Exception as exc:
        logger.warning("judge.context_relevance_failed %s", exc)
        return None

    votes = []
    for response in responses:
        text = _text_of(response).strip().upper()
        if text.startswith("YES"):
            votes.append(1.0)
        elif text.startswith("NO"):
            votes.append(0.0)
    return statistics.fmean(votes) if votes else None


# ---------------------------------------------------------------------------
# Golden dataset and the evaluation runner
# ---------------------------------------------------------------------------
@dataclass
class GoldenCase:
    """One labelled evaluation case."""

    case_id: str
    question: str
    expected_answer: str = ""
    relevant_chunk_ids: List[str] = field(default_factory=list)
    relevance_grades: Dict[str, int] = field(default_factory=dict)
    segment: str = "default"          # question type, doc type, language
    is_answerable: bool = True
    history: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], index: int = 0) -> "GoldenCase":
        """Tolerant loader so a hand written dataset file still works."""
        return cls(
            case_id=str(raw.get("case_id") or raw.get("id") or f"case-{index}"),
            question=raw.get("question") or raw.get("query") or "",
            expected_answer=raw.get("expected_answer", ""),
            relevant_chunk_ids=list(
                raw.get("relevant_chunk_ids")
                or raw.get("relevant_keywords")
                or []
            ),
            relevance_grades=dict(raw.get("relevance_grades") or {}),
            segment=raw.get("segment", "default"),
            is_answerable=bool(raw.get("is_answerable", True)),
            history=list(raw.get("history") or []),
        )


def load_golden_dataset(path) -> List[GoldenCase]:
    """Read a golden dataset file into GoldenCase objects."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    rows = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw
    return [GoldenCase.from_dict(row, i) for i, row in enumerate(rows)]


def evaluate_dataset(
    cases: List[GoldenCase],
    pipeline,
    judge_llm,
    tenant_id: str = "default",
    k: int = 5,
) -> Dict[str, Any]:
    """Run the full golden set and report metrics per segment.

    Segmentation is not optional. An aggregate improvement that hides a
    regression on one question type is the most common way a bad change
    passes evaluation and fails in production.
    """
    rows: List[Dict[str, Any]] = []

    for case in cases:
        response = pipeline.query(
            case.question, tenant_id=tenant_id, top_k=k, history=case.history
        )
        retrieved_ids = [source.chunk_id for source in response.sources]
        context = "\n\n".join(source.content for source in response.sources)

        row: Dict[str, Any] = {
            "case_id": case.case_id,
            "segment": case.segment,
            "abstained": response.abstained,
            "precision_at_k": precision_at_k(retrieved_ids, case.relevant_chunk_ids, k),
            "recall_at_k": recall_at_k(retrieved_ids, case.relevant_chunk_ids, k),
            "mrr": reciprocal_rank(retrieved_ids, case.relevant_chunk_ids),
            "ndcg_at_k": ndcg_at_k(retrieved_ids, case.relevance_grades, k),
            "hit_rate": hit_rate_at_k(retrieved_ids, case.relevant_chunk_ids, k),
            "question": case.question,
            "expected_answer": case.expected_answer,
            "generated_answer": response.answer,
        }

        # Unanswerable cases measure abstention behaviour, not faithfulness.
        if not case.is_answerable:
            row["correct_abstention"] = 1.0 if response.abstained else 0.0
            row["faithfulness"] = None
            row["answer_relevance"] = None
        elif response.abstained:
            row["correct_abstention"] = 0.0     # a false refusal
            row["faithfulness"] = None
            row["answer_relevance"] = None
        else:
            verdict = faithfulness_score(response.answer, context, judge_llm)
            row["faithfulness"] = verdict["score"]
            row["contradicted_claims"] = verdict.get("contradicted", 0)
            row["answer_relevance"] = answer_relevance_score(
                case.question, response.answer, judge_llm
            )
            row["correct_abstention"] = None

        rows.append(row)

    return {"rows": rows, "summary": summarize(rows)}


METRIC_NAMES = (
    "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k",
    "hit_rate", "faithfulness", "answer_relevance", "correct_abstention",
)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate overall and per segment, skipping None rather than zeroing."""

    def mean_of(values: List[Optional[float]]) -> Optional[float]:
        present = [v for v in values if v is not None]
        return statistics.fmean(present) if present else None

    overall = {name: mean_of([row.get(name) for row in rows])
               for name in METRIC_NAMES}
    overall["coverage"] = {
        name: sum(1 for row in rows if row.get(name) is not None)
        for name in METRIC_NAMES
    }
    overall["n"] = len(rows)

    by_segment: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)
    for segment in {row["segment"] for row in rows}:
        subset = [row for row in rows if row["segment"] == segment]
        for name in METRIC_NAMES:
            by_segment[segment][name] = mean_of([row.get(name) for row in subset])
        by_segment[segment]["n"] = len(subset)

    return {"overall": overall, "by_segment": dict(by_segment)}


class Evaluator:
    """Object wrapper kept so existing callers and the Chapter 6 app still work."""

    precision_at_k = staticmethod(precision_at_k)
    recall_at_k = staticmethod(recall_at_k)
    mrr = staticmethod(reciprocal_rank)
    ndcg_at_k = staticmethod(ndcg_at_k)
    hit_rate_at_k = staticmethod(hit_rate_at_k)
    faithfulness_score = staticmethod(faithfulness_score)
    answer_relevance_score = staticmethod(answer_relevance_score)
    context_relevance_score = staticmethod(context_relevance_score)
    evaluate_dataset = staticmethod(evaluate_dataset)
    summarize = staticmethod(summarize)
