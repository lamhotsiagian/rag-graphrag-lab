"""Chapter 5: fusion, reranking, and adaptive routing."""

import pytest

from chapter_05.reranker import (LLMReranker, _parse_score,
                                 maximal_marginal_relevance, normalize_scores)
from chapter_05.retrievers import (AdaptiveRetriever, HybridRetriever, QueryKind,
                                   ROUTING_POLICY, reciprocal_rank_fusion)
from shared.retrieval import Retriever
from tests.unit.conftest import FakeLLM, NoBatchLLM, make_result


class ListRetriever(Retriever):
    """Returns a fixed list, so fusion behaviour is fully determined."""

    def __init__(self, results, name="list"):
        self.results = results
        self.name = name
        self.calls = 0

    def retrieve(self, query, tenant_id="default", top_k=5, metadata_filter=None):
        self.calls += 1
        return self.results[:top_k]


# --- Reciprocal rank fusion --------------------------------------------------
def test_rrf_uses_one_based_ranks():
    """Zero-based ranks distort the top of the fusion."""
    single = [make_result("a", "x")]
    fused = reciprocal_rank_fusion([single], [1.0], k=60)
    assert fused[0].metadata["rrf_raw"] == pytest.approx(1 / 61)


def test_rrf_keys_on_chunk_id_not_content():
    """Whitespace differences must not split one document into two entries."""
    dense = [make_result("same", "The  answer   text")]
    lexical = [make_result("same", "The answer text")]
    fused = reciprocal_rank_fusion([dense, lexical], [0.5, 0.5])

    assert len(fused) == 1, "one document, not two"
    assert set(fused[0].metadata["found_by"]) == {"test"}
    assert fused[0].metadata["rrf_raw"] == pytest.approx(2 * 0.5 / 61)


def test_rrf_rewards_documents_found_by_both_branches():
    dense = [make_result("a", "alpha"), make_result("b", "beta")]
    lexical = [make_result("b", "beta"), make_result("c", "gamma")]
    fused = reciprocal_rank_fusion([dense, lexical], [0.5, 0.5])
    assert fused[0].chunk_id == "b", "found by both, so it should win"


def test_rrf_normalizes_the_top_score_to_one():
    fused = reciprocal_rank_fusion([[make_result("a", "x"), make_result("b", "y")]],
                                   [1.0])
    assert fused[0].score == pytest.approx(1.0)


def test_rrf_handles_empty_input_and_rejects_mismatched_weights():
    assert reciprocal_rank_fusion([[]], [1.0]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[], []], [1.0])


def test_hybrid_over_retrieves_so_fusion_has_material():
    """Fusing two top-K lists down to K starves the fusion."""
    dense = ListRetriever([make_result(str(i), f"d{i}") for i in range(40)], "dense")
    lexical = ListRetriever([make_result(str(i), f"l{i}") for i in range(40)], "lexical")

    hybrid = HybridRetriever(dense, lexical, candidate_multiplier=4)
    got = hybrid.retrieve("query", "t1", top_k=5)

    assert len(got) == 5
    assert dense.calls == 1 and lexical.calls == 1


# --- Reranking ---------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("8", 0.8), ("Score: 8", 0.8), ("8/10", 0.8),
    ("I would rate this an 8", 0.8), ("10", 1.0), ("0", 0.0), ("99", 1.0),
])
def test_parse_score_survives_real_model_formatting(raw, expected):
    """The defect this replaces called int() and scored these zero."""
    assert _parse_score(raw, fallback=0.5) == pytest.approx(expected)


def test_parse_score_falls_back_to_retrieval_score_not_zero():
    """A parse failure is a formatting problem, not evidence of irrelevance."""
    assert _parse_score("no digits here", fallback=0.77) == 0.77
    assert _parse_score("", fallback=0.42) == 0.42
    assert _parse_score(None, fallback=0.42) == 0.42


def test_reranker_batches_and_reorders():
    candidates = [make_result("a", "weak", 0.9), make_result("b", "strong", 0.1)]
    llm = FakeLLM(replies=["2", "9"])
    got = LLMReranker(keep=2, score_floor=0.0).rerank("q", candidates, llm)

    assert [r.chunk_id for r in got] == ["b", "a"], "reranking must reorder"
    assert llm.batch_calls == 1, "one batched call, not one per candidate"
    assert got[0].metadata["first_stage_score"] == 0.1


def test_reranker_falls_back_to_a_loop_without_batch():
    llm = NoBatchLLM(replies=["9", "2"])
    got = LLMReranker(keep=2, score_floor=0.0).rerank(
        "q", [make_result("a", "x"), make_result("b", "y")], llm)
    assert len(got) == 2


def test_reranker_drops_candidates_below_the_floor_and_handles_empty():
    llm = FakeLLM(replies=["1", "9"])
    got = LLMReranker(keep=5, score_floor=0.5).rerank(
        "q", [make_result("a", "x"), make_result("b", "y")], llm)
    assert [r.chunk_id for r in got] == ["b"]
    assert LLMReranker().rerank("q", [], llm) == []


def test_maximal_marginal_relevance_prefers_diversity():
    results = [make_result("a", "vector search index", 0.9),
               make_result("b", "vector search index too", 0.85),
               make_result("c", "graph traversal cypher", 0.6)]
    got = maximal_marginal_relevance(results, keep=2, diversity=0.8)
    assert [r.chunk_id for r in got] == ["a", "c"]


def test_normalize_scores_handles_a_flat_list():
    assert normalize_scores([0.5, 0.5]) == [1.0, 1.0]
    assert normalize_scores([]) == []


# --- Adaptive routing --------------------------------------------------------
@pytest.mark.parametrize("question,expected", [
    ("who reports to the CTO", QueryKind.RELATIONAL),
    ("compare pgvector versus Neo4j", QueryKind.COMPARATIVE),
    ("how many incidents last quarter", QueryKind.AGGREGATE),
    ("when did the policy change", QueryKind.FACTUAL),
])
def test_classifier_rules_resolve_without_a_model_call(question, expected):
    llm = FakeLLM(default="conceptual")
    router = AdaptiveRetriever(hybrid=None, reranker=None, llm=llm)
    assert router.classify(question) == expected
    assert llm.prompts == [], "layer one must not call the model"


def test_classifier_falls_back_to_the_model_then_to_a_safe_default():
    llm = FakeLLM(default="relational")
    router = AdaptiveRetriever(hybrid=None, reranker=None, llm=llm)
    assert router.classify("explain the tradeoffs of chunk size") == QueryKind.RELATIONAL
    assert llm.prompts, "layer three should have been reached"

    confused = FakeLLM(default="banana")
    fallback = AdaptiveRetriever(hybrid=None, reranker=None, llm=confused)
    assert fallback.classify("explain the tradeoffs") == QueryKind.CONCEPTUAL


def test_every_query_kind_has_a_routing_policy():
    for kind in QueryKind:
        policy = ROUTING_POLICY[kind]
        assert {"transform", "k", "rerank"} == set(policy)


def test_corrective_loop_widens_the_search_then_stops():
    weak = ListRetriever([make_result("a", "weak", 0.1)])
    llm = FakeLLM(default="conceptual")
    router = AdaptiveRetriever(weak, reranker=None, llm=llm,
                               relevance_floor=0.9, max_attempts=2)

    outcome = router.retrieve_adaptive("explain something obscure")
    assert outcome.attempts == 2, "should have corrected once"
    assert outcome.corrections, "the correction must be recorded in the trace"
    assert weak.calls == 2


def test_adaptive_stops_immediately_when_evidence_is_strong():
    strong = ListRetriever([make_result("a", "strong", 0.95)])
    router = AdaptiveRetriever(strong, reranker=None, llm=FakeLLM(default="factual"),
                               relevance_floor=0.5, max_attempts=3)
    outcome = router.retrieve_adaptive("when did the policy change")
    assert outcome.attempts == 1 and not outcome.corrections
