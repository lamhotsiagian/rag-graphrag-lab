"""Chapters 1 and 2: loading, normalization, and chunking."""

import io

import pytest

from chapter_01.rag_basic import (ABSTENTION_MESSAGE, build_simple_index,
                                  cosine_similarity, generate_answer, retrieve,
                                  split_text)
from chapter_02.chunkers import (ChildChunk, fixed_size_chunking, get_chunk_stats,
                                 overlap_ratio, parent_child_chunking,
                                 recursive_chunking, semantic_chunking,
                                 structure_aware_chunking)
from chapter_02.loaders import clean_text, load_csv, load_document
from tests.unit.conftest import FakeLLM


class Upload(io.BytesIO):
    """Minimal stand-in for a Streamlit UploadedFile."""

    def __init__(self, name: str, text: str):
        super().__init__(text.encode("utf-8"))
        self.name = name


# --- Chapter 2, normalization ------------------------------------------------
def test_clean_text_preserves_meaningful_characters():
    """The defect this replaces stripped currency, percent, and code punctuation."""
    raw = "Revenue rose 15% to $2.3M, see calc(a[0] + b) — done."
    cleaned = clean_text(raw)
    for token in ("15%", "$2.3M", "calc(a[0]", "+"):
        assert token in cleaned, f"{token!r} was destroyed by normalization"


def test_clean_text_repairs_hyphenation_and_ligatures():
    assert "retrieval" in clean_text("retrie-\nval")
    assert clean_text("eﬃcient") == "efficient"


def test_clean_text_preserves_paragraph_breaks_by_default():
    assert "\n\n" in clean_text("Para one.\n\n\n\nPara two.")
    assert "\n" not in clean_text("Para one.\n\nPara two.", preserve_layout=False)


def test_csv_loader_binds_values_to_column_names():
    text, meta = load_csv(Upload("t.csv", "region,revenue\nEMEA,120\nAPAC,90"))
    assert "region: EMEA" in text and "revenue: 120" in text
    assert meta["row_count"] == "2"


def test_load_document_rejects_unknown_formats():
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(Upload("data.xyz", "x"))


# --- Chapter 2, chunking -----------------------------------------------------
def test_fixed_size_chunking_overlaps_and_covers():
    chunks = fixed_size_chunking("abcdefghij" * 10, size=30, overlap=10)
    assert all(len(c) <= 30 for c in chunks)
    assert chunks[0][-10:] == chunks[1][:10]


def test_fixed_size_chunking_rejects_overlap_larger_than_size():
    with pytest.raises(ValueError):
        fixed_size_chunking("text", size=10, overlap=10)


def test_recursive_chunking_prefers_paragraph_boundaries():
    text = ("First paragraph about retrieval.\n\n"
            "Second paragraph about ranking.\n\n"
            "Third paragraph about evaluation.")
    chunks = recursive_chunking(text, size=60, overlap=0)
    assert len(chunks) >= 2
    assert not chunks[0].endswith("Sec")     # did not cut mid-word


def test_semantic_chunking_batches_embeddings_once(embeddings):
    text = " ".join(f"Sentence number {i} about retrieval systems." for i in range(12))
    chunks = semantic_chunking(text, embeddings)
    assert chunks
    assert embeddings.document_calls == 1, "must be one batched call, not one per sentence"


def test_semantic_chunking_handles_short_input(embeddings):
    assert semantic_chunking("One sentence.", embeddings) == ["One sentence."]
    assert semantic_chunking("", embeddings) == []


def test_parent_child_chunking_is_idempotent_and_linked():
    text = "\n\n".join(f"Paragraph {i} with enough words to matter here." * 4
                       for i in range(6))
    parents_a, children_a = parent_child_chunking(text, "doc.md", "Guide > Setup")
    parents_b, children_b = parent_child_chunking(text, "doc.md", "Guide > Setup")

    assert set(parents_a) == set(parents_b), "ids must be content addressed"
    assert [c.child_id for c in children_a] == [c.child_id for c in children_b]
    assert all(isinstance(c, ChildChunk) for c in children_a)
    assert all(c.parent_id in parents_a for c in children_a), "no orphan children"
    assert all(c.content.startswith("Guide > Setup") for c in children_a), \
        "breadcrumb must anchor each child"


def test_structure_aware_chunking_prefixes_the_breadcrumb():
    md = ("# Security Policy\n\nIntro.\n\n"
          "## Access Control\n\nBody text.\n\n"
          "### Contractor Accounts\n\nDisable within 24 hours.\n")
    chunks = structure_aware_chunking(md)
    assert any("Security Policy > Access Control > Contractor Accounts" in c
               for c in chunks)


def test_chunk_stats_and_overlap_ratio():
    stats = get_chunk_stats(["a" * 100, "b" * 200])
    assert stats["count"] == 2 and stats["avg_size"] == 150
    assert get_chunk_stats([])["count"] == 0

    heavy = fixed_size_chunking("word " * 200, size=200, overlap=100)
    assert overlap_ratio(heavy) > 0.5, "50 percent overlap should be visible"


# --- Chapter 1 ---------------------------------------------------------------
def test_split_text_rejects_bad_parameters():
    with pytest.raises(ValueError):
        split_text("text", chunk_size=10, overlap=10)


def test_cosine_similarity_edges():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0     # no division by zero


def test_retrieve_applies_the_abstention_floor(embeddings):
    index = build_simple_index(["retrieval augmented generation", "unrelated text"],
                               embeddings)
    assert retrieve("retrieval augmented generation", index, embeddings, 2)
    assert retrieve("q", index, embeddings, top_k=2, min_score=1.01) == []
    assert retrieve("q", [], embeddings) == []


def test_generate_answer_abstains_in_code_not_via_the_model():
    """The empty-context branch must never reach the model."""
    llm = FakeLLM(default="I will happily invent an answer.")
    assert generate_answer("q", [], llm) == ABSTENTION_MESSAGE
    assert llm.prompts == [], "the model must not be called with no evidence"


def test_generate_answer_numbers_the_context_blocks():
    llm = FakeLLM(default="Answer [1].")
    generate_answer("q", [{"content": "first", "score": 0.9},
                          {"content": "second", "score": 0.8}], llm)
    assert "[1]" in llm.prompts[0] and "[2]" in llm.prompts[0]
