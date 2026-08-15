"""Chapter 4: context construction, citation validation, and the three gates."""

import pytest

from chapter_04.rag_complete import (ABSTENTION_TOKEN, RAGPipeline, build_context,
                                     condense_question, deduplicate,
                                     estimate_tokens, validate_citations)
from shared.retrieval import StaticRetriever
from tests.unit.conftest import FakeLLM, make_result


def test_estimate_tokens_never_returns_zero():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100


def test_deduplicate_drops_near_duplicates_keeping_the_first():
    results = [
        make_result("a", "the quick brown fox jumps over the lazy dog", 0.9),
        make_result("b", "the quick brown fox jumps over the lazy dog today", 0.8),
        make_result("c", "completely different content about vector indexes", 0.7),
    ]
    kept = deduplicate(results)
    assert [r.chunk_id for r in kept] == ["a", "c"]


def test_build_context_returns_the_admitted_blocks():
    """Returning only the string is what produces phantom citation failures."""
    results = [make_result(str(i), f"block {i} " * 40, 0.9 - i / 10) for i in range(6)]
    context, admitted = build_context(results, token_budget=200)

    assert len(admitted) < len(results), "budget must exclude some blocks"
    assert all(f"[{i}]" in context for i in range(1, len(admitted) + 1))
    # Numbers are assigned after reordering, so every citation is addressable.
    assert context.count("[") >= len(admitted)


def test_build_context_places_the_strongest_at_both_edges():
    results = [make_result(c, f"content {c}", s)
               for c, s in zip("abcde", [0.9, 0.8, 0.7, 0.6, 0.5])]
    _, admitted = build_context(results, token_budget=10_000)
    assert admitted[0].chunk_id == "a", "strongest first"
    assert admitted[-1].chunk_id == "b", "second strongest last"


def test_validate_citations_catches_uncited_and_fabricated_blocks():
    admitted = [make_result("a", "x"), make_result("b", "y")]

    ok, report = validate_citations("Claim one [1]. Claim two [2].", admitted)
    assert ok and report["citation_coverage"] == 1.0

    ok, report = validate_citations("Claim one [1]. Uncited claim.", admitted)
    assert not ok and len(report["uncited_sentences"]) == 1

    ok, report = validate_citations("Claim [7].", admitted)
    assert not ok and report["hallucinated_block_ids"] == [7]


def test_condense_question_skips_the_model_when_self_contained():
    llm = FakeLLM(default="rewritten")
    question = "What is the documented refund window for enterprise customers"
    assert condense_question(question, [{"role": "user", "content": "hi"}], llm) == question
    assert llm.prompts == [], "no model call for a standalone question"


def test_condense_question_rewrites_a_pronoun_follow_up():
    llm = FakeLLM(default="Does policy A cover flood damage?")
    history = [{"role": "user", "content": "Tell me about policy A"},
               {"role": "assistant", "content": "Policy A covers fire."}]
    out = condense_question("does it cover flood damage", history, llm)
    assert out == "Does policy A cover flood damage?"
    assert llm.prompts, "a dependent question must reach the model"


def test_condense_question_rejects_a_degenerate_rewrite():
    llm = FakeLLM(default="x" * 5000)
    history = [{"role": "user", "content": "a"}]
    assert condense_question("what about it", history, llm) == "what about it"


def test_condense_question_returns_input_with_no_history():
    llm = FakeLLM()
    assert condense_question("it", [], llm) == "it"
    assert llm.prompts == []


# --- The three gates ---------------------------------------------------------
def _pipeline(llm, results, **kwargs):
    return RAGPipeline(llm, None, StaticRetriever(results), **kwargs)


def test_gate_one_abstains_before_the_generator_runs():
    llm = FakeLLM(default="I would have invented something.")
    pipeline = _pipeline(llm, [make_result("a", "unrelated")], min_score=0.99)
    response = pipeline.query("nothing matches this query at all")

    assert response.abstained
    assert response.citation_report["reason"] == "no_confident_results"
    assert llm.prompts == [], "Gate 1 must fire before generation"


def test_gate_two_detects_the_abstention_token():
    llm = FakeLLM(default=ABSTENTION_TOKEN)
    pipeline = _pipeline(llm, [make_result("a", "retrieval augmented generation")],
                         min_score=0.0)
    response = pipeline.query("retrieval augmented generation")

    assert response.abstained
    assert response.citation_report["reason"] == "model_abstained"
    assert ABSTENTION_TOKEN not in response.answer, "the token is internal"


def test_gate_three_regenerates_once_then_gives_up():
    llm = FakeLLM(replies=["Uncited sentence.", "Still uncited."])
    pipeline = _pipeline(llm, [make_result("a", "retrieval augmented generation")],
                         min_score=0.0, max_regenerations=1)
    response = pipeline.query("retrieval augmented generation")

    assert len(llm.prompts) == 2, "exactly one regeneration"
    assert response.citation_report["regenerations"] == 1
    assert "FAILED VALIDATION" in llm.prompts[1], "the retry must name the failure"


def test_a_well_cited_answer_passes_without_regeneration():
    llm = FakeLLM(default="Grounded claim [1].")
    pipeline = _pipeline(llm, [make_result("a", "retrieval augmented generation")],
                         min_score=0.0)
    response = pipeline.query("retrieval augmented generation")

    assert not response.abstained
    assert response.citation_report["regenerations"] == 0
    assert len(llm.prompts) == 1


def test_pipeline_reports_per_stage_timings():
    llm = FakeLLM(default="Answer [1].")
    pipeline = _pipeline(llm, [make_result("a", "retrieval augmented generation")],
                         min_score=0.0)
    response = pipeline.query("retrieval augmented generation")
    assert {"condense", "retrieve", "context", "generate"} <= set(response.timings_ms)
