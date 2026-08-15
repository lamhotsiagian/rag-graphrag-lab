"""The retriever contract, Chapter 4."""

import pytest

from shared.retrieval import (FallbackRetriever, RetrievalResult, Retriever,
                              StaticRetriever, normalize_scores)
from tests.unit.conftest import make_result


def test_result_is_immutable_and_copies_with_new_score():
    result = make_result("a", "text", 0.5)
    with pytest.raises(Exception):
        result.score = 0.9              # frozen dataclass
    rescored = result.with_score(0.9, "reranked")
    assert rescored.score == 0.9
    assert rescored.chunk_id == result.chunk_id
    assert rescored.retriever == "reranked"
    assert result.score == 0.5          # original untouched


def test_retriever_is_abstract():
    with pytest.raises(TypeError):
        Retriever()                     # cannot instantiate without retrieve


def test_static_retriever_respects_top_k_and_filter(sample_results):
    retriever = StaticRetriever(sample_results)
    assert len(retriever.retrieve("graph", "t1", top_k=2)) <= 2

    tagged = [make_result("x", "alpha beta", metadata={"lang": "en"}),
              make_result("y", "alpha beta", metadata={"lang": "de"})]
    got = StaticRetriever(tagged).retrieve("alpha", "t1", 5, {"lang": "de"})
    assert [r.chunk_id for r in got] == ["y"]


def test_fallback_uses_second_retriever_when_first_is_weak(sample_results):
    weak = StaticRetriever([make_result("w", "zzz unrelated", 0.0)])
    strong = StaticRetriever(sample_results)

    chain = FallbackRetriever([weak, strong], min_results=1, min_score=0.2)
    got = chain.retrieve("retrieval augmented generation", "t1", 3)

    assert got, "fallback should have reached the second retriever"
    assert all(r.score >= 0.2 for r in got)


def test_fallback_returns_empty_when_every_retriever_fails():
    dead = StaticRetriever([make_result("w", "zzz", 0.0)])
    chain = FallbackRetriever([dead, dead], min_score=0.9)
    assert chain.retrieve("nothing matches this", "t1") == []


def test_fallback_rejects_empty_chain():
    with pytest.raises(ValueError):
        FallbackRetriever([])


def test_normalize_scores_scales_best_to_one(sample_results):
    normalized = normalize_scores(sample_results)
    assert max(r.score for r in normalized) == pytest.approx(1.0)
    assert normalize_scores([]) == []
