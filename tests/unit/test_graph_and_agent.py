"""Chapters 7, 8, 9, and 10: graph construction, GraphRAG, agent, production."""

import json
import time

import pytest

from chapter_07.kg_builder import (NODE_LABELS, RELATIONSHIP_TYPES, canonicalize,
                                   extract_graph, resolve_entities)
from chapter_08.graph_rag import GraphContext, triples_to_results
from chapter_09.agentic_rag import (evaluate_context, merge_unique, plan_next_hop,
                                    route_after_evaluation, verify_answer)
from chapter_10.system_design import (BreakerState, CircuitBreaker, LADDER,
                                      RetryWithBackoff, SemanticCache, TokenBudget,
                                      degrade, estimate_capacity, estimate_load,
                                      get_interview_scenarios)
from tests.unit.conftest import FakeLLM, make_result


# --- Chapter 7 ---------------------------------------------------------------
def _extraction(entities, relationships):
    return FakeLLM(default=json.dumps(
        {"entities": entities, "relationships": relationships}))


def test_extraction_rejects_out_of_vocabulary_types():
    """Unconstrained extraction is how a graph ends up with 340 relationship types."""
    llm = _extraction(
        [{"name": "Acme", "type": "Organization", "confidence": 0.9},
         {"name": "Zeta", "type": "Spaceship", "confidence": 0.9}],
        [{"source": "Acme", "target": "Zeta", "type": "TELEPORTS_TO",
          "confidence": 0.9, "evidence": "q"}])
    result = extract_graph("text", llm)

    assert [e["name"] for e in result.entities] == ["Acme"]
    assert result.relationships == []
    assert any("Spaceship" in r for r in result.rejected)


def test_extraction_drops_dangling_relationships():
    """An edge naming an unextracted entity would create a phantom node."""
    llm = _extraction(
        [{"name": "Acme", "type": "Organization", "confidence": 0.9}],
        [{"source": "Acme", "target": "Ghost", "type": "SUPPLIES",
          "confidence": 0.9, "evidence": "q"}])
    result = extract_graph("text", llm)
    assert result.relationships == []
    assert any("dangling" in r for r in result.rejected)


def test_extraction_filters_stop_entities_that_become_hubs():
    llm = _extraction(
        [{"name": "data", "type": "Concept", "confidence": 0.99}], [])
    result = extract_graph("text", llm)
    assert result.entities == []
    assert any("stop_entity" in r for r in result.rejected)


def test_extraction_survives_a_non_json_reply():
    result = extract_graph("text", FakeLLM(default="Sure, here are the entities!"))
    assert result.parse_failed and result.entities == []


def test_extraction_honours_the_confidence_floor():
    llm = _extraction(
        [{"name": "Acme", "type": "Organization", "confidence": 0.2}], [])
    assert extract_graph("t", llm, min_confidence=0.5).entities == []


@pytest.mark.parametrize("variant", [
    "Acme Corp", "ACME Corporation", "acme corp.", "Acme  Corp", "Ácme Corp",
])
def test_canonicalize_merges_surface_forms(variant):
    assert canonicalize(variant) == canonicalize("Acme")


def test_resolve_entities_applies_the_alias_dictionary():
    resolved = resolve_entities(["Big Blue", "IBM"], alias_map={"Big Blue": "IBM"})
    assert resolved["Big Blue"] == resolved["IBM"]


def test_schema_vocabularies_are_closed_and_non_empty():
    assert NODE_LABELS and RELATIONSHIP_TYPES
    assert all(t == t.upper() for t in RELATIONSHIP_TYPES)


# --- Chapter 8 ---------------------------------------------------------------
def test_triples_render_as_sentences_a_reranker_can_score():
    """A cross encoder scores prose correctly and raw triples poorly."""
    context = GraphContext(
        triples=["Acme -[SUPPLIES]-> Meridian"],
        evidence=["per the 2025 master agreement"],
        source_uris=["contract.pdf"],
        seed_entities=["Acme"])
    results = triples_to_results(context)

    assert results[0].content.startswith("Acme supplies Meridian.")
    assert "2025 master agreement" in results[0].content
    assert results[0].metadata["origin"] == "graph"
    assert results[0].source_uri == "contract.pdf"


def test_graph_context_renders_empty_when_nothing_linked():
    assert GraphContext().render() == ""


# --- Chapter 9 ---------------------------------------------------------------
def test_merge_unique_does_not_accumulate_the_same_chunk_twice():
    """A plain operator.add reducer inflates context across hops."""
    hop1 = [make_result("a", "x"), make_result("b", "y")]
    hop2 = [make_result("b", "y"), make_result("c", "z")]
    assert [r.chunk_id for r in merge_unique(hop1, hop2)] == ["a", "b", "c"]
    assert merge_unique(None, None) == []


def test_evaluator_suppresses_a_repeated_query():
    """A planner that repeats its query guarantees a wasted iteration."""
    llm = FakeLLM(default=json.dumps(
        {"sufficient": False, "gaps": ["g"], "next_query": "original question"}))
    update = evaluate_context(
        {"question": "original question", "evidence": [make_result("a", "x")],
         "hop_queries": []}, llm=llm)

    assert update["evidence_sufficient"] is True, "repetition must end the loop"
    assert "evaluate" in update["reasoning"][0]


def test_evaluator_accepts_a_genuinely_new_query():
    llm = FakeLLM(default=json.dumps(
        {"sufficient": False, "gaps": ["ownership"],
         "next_query": "who owns the Meridian subsidiary"}))
    update = evaluate_context(
        {"question": "original", "evidence": [make_result("a", "x")],
         "hop_queries": []}, llm=llm)
    assert update["evidence_sufficient"] is False
    assert update["active_query"] == "who owns the Meridian subsidiary"


def test_evaluator_does_not_trap_the_graph_on_a_parse_failure():
    update = evaluate_context(
        {"question": "q", "evidence": [make_result("a", "x")], "hop_queries": []},
        llm=FakeLLM(default="not json at all"))
    assert update["evidence_sufficient"] is True


def test_evaluator_escalates_when_there_is_no_evidence():
    update = evaluate_context({"question": "q", "evidence": []}, llm=FakeLLM())
    assert update["evidence_sufficient"] is False
    assert update["coverage_gaps"] == ["no evidence retrieved"]


def test_plan_next_hop_advances_the_counter_and_records_the_query():
    update = plan_next_hop({"iteration": 1, "active_query": "new query"})
    assert update["iteration"] == 2
    assert update["hop_queries"] == ["new query"]


@pytest.mark.parametrize("state,expected", [
    ({"evidence_sufficient": True}, "generate"),
    ({"evidence_sufficient": False, "iteration": 3, "max_iterations": 3}, "generate"),
    ({"evidence_sufficient": False, "iteration": 0, "active_query": ""}, "generate"),
    ({"evidence_sufficient": False, "iteration": 0, "max_iterations": 3,
      "active_query": "q", "deadline_ts": time.time() - 1}, "generate"),
    ({"evidence_sufficient": False, "iteration": 0, "max_iterations": 3,
      "active_query": "q"}, "continue"),
])
def test_every_loop_exit_is_reachable_and_bounded(state, expected):
    assert route_after_evaluation(state) == expected


def test_verify_answer_flags_low_confidence_for_human_review():
    update = verify_answer(
        {"answer": "Uncited claim.", "evidence": [make_result("a", "x")]},
        llm=None, confidence_floor=0.6)
    assert update["needs_human"] is True
    assert update["verification"]["passed"] is False


def test_verify_answer_passes_a_fully_cited_answer():
    update = verify_answer(
        {"answer": "Grounded claim [1].", "evidence": [make_result("a", "x")]},
        llm=None, confidence_floor=0.6)
    assert update["needs_human"] is False


# --- Chapter 10 --------------------------------------------------------------
def test_breaker_opens_on_errors_then_recovers_through_half_open():
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=0.01,
                             half_open_probes=1)

    def boom():
        raise RuntimeError("down")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            breaker.call(boom)
    assert breaker.state is BreakerState.OPEN

    with pytest.raises(RuntimeError, match="circuit_open"):
        breaker.call(lambda: "ok")

    time.sleep(0.02)
    assert breaker.call(lambda: "ok") == "ok"
    assert breaker.state is BreakerState.CLOSED


def test_breaker_opens_on_latency_alone():
    """An error-only breaker never opens on a slow endpoint, which is worse."""
    breaker = CircuitBreaker(failure_threshold=99, latency_threshold_ms=5,
                             slow_call_ratio=0.5, min_calls=5)
    for _ in range(4):
        breaker.call(lambda: time.sleep(0.01))
    assert breaker.state is BreakerState.CLOSED, "too few calls to judge a ratio"

    breaker.call(lambda: time.sleep(0.01))
    assert breaker.state is BreakerState.OPEN


def test_retry_gives_up_and_reraises():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        RetryWithBackoff(max_retries=3, base_delay=0.001).execute(flaky)
    assert calls["n"] == 3


def test_degradation_ladder_applies_rungs_in_order_with_stated_costs():
    config = {"ef_search": 100, "rerank": True, "top_k": 5}
    assert degrade(config, 0) == config
    assert degrade(config, 1)["ef_search"] == 40
    assert degrade(config, 2)["rerank"] is False
    assert degrade(config, 99)["top_k"] == 3
    assert all(rung.quality_cost for rung in LADDER), "every rung states its cost"


def test_semantic_cache_keys_on_tenant_and_corpus_version():
    cache = SemanticCache(threshold=0.99)
    vector = [1.0, 0.0, 0.0]
    cache.put(vector, "cached answer", tenant_id="t1", corpus_version="v1")

    assert cache.get(vector, "t1", "v1") == "cached answer"
    assert cache.get(vector, "t2", "v1") is None, "must not cross tenants"
    assert cache.get(vector, "t1", "v2") is None, "a corpus change invalidates"
    assert 0.0 < cache.hit_rate < 1.0


def test_token_budget_drops_whole_blocks_never_partial_ones():
    chunks = ["a" * 400, "b" * 400, "c" * 400]
    fitted = TokenBudget().fit_context(chunks, budget=150)
    assert all(c in chunks for c in fitted), "no truncated blocks"
    assert len(fitted) < len(chunks)


def test_capacity_and_load_arithmetic_matches_the_book():
    capacity = estimate_capacity(200_000)
    assert capacity["chunks"] == pytest.approx(5_600_000, rel=0.01)
    assert capacity["raw_vector_gb"] == pytest.approx(17.2, rel=0.05)

    load = estimate_load(5_000)
    assert load["queries_per_day"] == 40_000
    assert load["peak_qps"] == pytest.approx(6.9, rel=0.05)


def test_all_six_interview_scenarios_are_complete():
    scenarios = get_interview_scenarios()
    assert len(scenarios) == 6
    for scenario in scenarios:
        assert {"title", "requirements", "architecture", "follow_up", "hints"} \
            <= set(scenario)
        assert scenario["hints"]
