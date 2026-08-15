"""Chapter 6: retrieval metrics, judged metrics, and the runner."""

import math

import pytest

from chapter_06.evaluator import (GoldenCase, answer_relevance_score,
                                  evaluate_dataset, faithfulness_score,
                                  hit_rate_at_k, ndcg_at_k, precision_at_k,
                                  recall_at_k, reciprocal_rank, summarize)
from chapter_04.rag_complete import RAGPipeline
from shared.retrieval import StaticRetriever
from tests.unit.conftest import FakeLLM, make_result


# --- Retrieval metrics -------------------------------------------------------
def test_precision_divides_by_results_retrieved_not_by_k():
    """Dividing by k penalizes a query whose corpus has fewer than k chunks."""
    assert precision_at_k(["a"], ["a"], k=5) == 1.0
    assert precision_at_k(["a", "b"], ["a"], k=2) == 0.5
    assert precision_at_k([], ["a"], k=5) == 0.0


def test_recall_treats_nothing_to_find_as_perfect():
    assert recall_at_k(["a"], [], k=5) == 1.0
    assert recall_at_k(["a"], ["a", "b"], k=5) == 0.5


def test_reciprocal_rank_reports_the_first_hit():
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x"], ["a"]) == 0.0


def test_ndcg_rewards_the_ideal_order():
    grades = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "b", "a"], grades, 3) < 1.0
    assert ndcg_at_k(["a"], {}, 3) == 0.0


def test_hit_rate_is_binary():
    assert hit_rate_at_k(["x", "a"], ["a"], 2) == 1.0
    assert hit_rate_at_k(["x", "y"], ["a"], 2) == 0.0


# --- Judged metrics ----------------------------------------------------------
def _judge(claims, verdicts):
    """A judge that extracts `claims` then returns `verdicts` in order."""
    state = {"extracted": False}

    def router(prompt):
        if "atomic factual claims" in prompt:
            state["extracted"] = True
            import json
            return json.dumps(claims)
        return verdicts.pop(0) if verdicts else "UNSUPPORTED"

    return FakeLLM(router=router)


def test_faithfulness_is_the_supported_claim_ratio():
    judge = _judge(["c1", "c2", "c3", "c4"],
                   ["SUPPORTED", "SUPPORTED", "SUPPORTED", "UNSUPPORTED"])
    verdict = faithfulness_score("answer", "context", judge)
    assert verdict["score"] == pytest.approx(0.75)
    assert verdict["unsupported"] == 1


def test_faithfulness_separates_contradiction_from_silence():
    """Different failures, different fixes, so they must not collapse."""
    judge = _judge(["c1", "c2"], ["CONTRADICTED", "UNSUPPORTED"])
    verdict = faithfulness_score("answer", "context", judge)
    assert verdict["contradicted"] == 1 and verdict["unsupported"] == 1


def test_judge_failure_returns_none_not_zero():
    """The defect this replaces recorded a parse failure as an unfaithful answer."""
    verdict = faithfulness_score("answer", "context",
                                 FakeLLM(default="Sure! Here you go."))
    assert verdict["score"] is None
    assert verdict["reason"] == "claim_extraction_failed"


def test_unparseable_verdicts_return_none():
    judge = _judge(["c1"], ["I am not sure about that"])
    assert faithfulness_score("a", "c", judge)["score"] is None


def test_answer_relevance_parses_yes_no_and_nothing_else():
    assert answer_relevance_score("q", "a", FakeLLM(default="YES")) == 1.0
    assert answer_relevance_score("q", "a", FakeLLM(default="NO")) == 0.0
    assert answer_relevance_score("q", "a", FakeLLM(default="maybe")) is None


# --- Aggregation -------------------------------------------------------------
def test_summarize_skips_none_rather_than_zeroing():
    rows = [
        {"segment": "s1", "faithfulness": 1.0, "recall_at_k": 1.0},
        {"segment": "s1", "faithfulness": None, "recall_at_k": 0.0},
    ]
    summary = summarize(rows)
    assert summary["overall"]["faithfulness"] == 1.0, "None must be excluded"
    assert summary["overall"]["recall_at_k"] == 0.5
    assert summary["overall"]["coverage"]["faithfulness"] == 1


def test_summarize_reports_per_segment():
    rows = [{"segment": "factual", "recall_at_k": 1.0},
            {"segment": "relational", "recall_at_k": 0.0}]
    by_segment = summarize(rows)["by_segment"]
    assert by_segment["factual"]["recall_at_k"] == 1.0
    assert by_segment["relational"]["recall_at_k"] == 0.0


# --- The runner --------------------------------------------------------------
def test_evaluate_dataset_scores_abstention_separately():
    """An unanswerable case measures abstention, never faithfulness."""
    results = [make_result("a", "retrieval augmented generation grounds answers")]
    pipeline = RAGPipeline(FakeLLM(default="Answer [1]."), None,
                           StaticRetriever(results), min_score=0.0)

    cases = [
        GoldenCase(case_id="answerable", question="retrieval augmented generation",
                   relevant_chunk_ids=["a"], segment="factual"),
        GoldenCase(case_id="unanswerable", question="zzz nothing matches",
                   is_answerable=False, segment="unanswerable"),
    ]
    report = evaluate_dataset(cases, pipeline, FakeLLM(default="not json"), k=3)
    rows = {r["case_id"]: r for r in report["rows"]}

    assert rows["unanswerable"]["faithfulness"] is None
    assert rows["unanswerable"]["correct_abstention"] in (0.0, 1.0)
    assert rows["answerable"]["hit_rate"] == 1.0
    assert set(report["summary"]["by_segment"]) == {"factual", "unanswerable"}


def test_golden_case_loader_tolerates_a_hand_written_file():
    case = GoldenCase.from_dict(
        {"query": "what is RAG?", "expected_answer": "retrieval",
         "relevant_keywords": ["a", "b"]}, index=3)
    assert case.case_id == "case-3"
    assert case.question == "what is RAG?"
    assert case.relevant_chunk_ids == ["a", "b"]
