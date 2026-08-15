"""Chapter 4: the complete basic RAG pipeline.

Book reference:
    Chapter 4, "Context Construction"
    Chapter 4, "Prompt Engineering for Grounded Generation"
    Chapter 4, "Source Attribution That Users Can Verify"
    Chapter 4, "Conversation-Aware Retrieval"
    Chapter 4, "The Complete Pipeline"

Three gates make retrieval failure observable rather than silent:
    Gate 1  a score floor, applied in code before the generator ever runs
    Gate 2  an explicit abstention token the model can emit and code can detect
    Gate 3  citation coverage on the finished answer, with one regeneration
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from shared.retrieval import RetrievalResult, Retriever

logger = logging.getLogger(__name__)

# Roughly four characters per token for English. Replace with the real
# tokenizer when the generator exposes one, since the estimate drifts badly on
# code, JSON, and non-Latin scripts.
CHARS_PER_TOKEN = 4

ABSTENTION_TOKEN = "INSUFFICIENT_EVIDENCE"

RAG_SYSTEM_PROMPT = """You answer questions using only the supplied evidence.

RULES
1. Use ONLY the numbered context blocks. Ignore all outside knowledge.
2. End every sentence with the block number that supports it, e.g. [2].
3. If no block supports an answer, output exactly: INSUFFICIENT_EVIDENCE
4. If blocks disagree, present both positions with their citations.
5. Answer in at most 150 words.
"""

CONDENSE_PROMPT = """Rewrite the follow-up question as a standalone question.

RULES
1. Resolve every pronoun and reference using the conversation.
2. Preserve the user's terminology exactly. Do not add new concepts.
3. If the question is already standalone, return it unchanged.
4. Output the rewritten question only, with no preamble.

CONVERSATION
{history}

FOLLOW-UP
{question}

STANDALONE QUESTION"""

# References that signal the question depends on prior turns.
DEPENDENT_MARKERS = (
    "it", "that", "this", "those", "these", "they", "them",
    "the second", "the first", "the former", "the latter",
    "he", "she", "there", "instead", "also", "same", "one",
)

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Cheap token estimate used for budgeting."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def deduplicate(
    results: List[RetrievalResult],
    overlap_threshold: float = 0.75,
) -> List[RetrievalResult]:
    """Drop near duplicate chunks, keeping the highest scoring instance.

    Overlapping chunks from the same source waste budget and bias the model
    toward whichever passage happens to appear most often.
    """
    kept: List[RetrievalResult] = []
    for candidate in results:
        candidate_terms = set(candidate.content.lower().split())
        duplicate = False
        for existing in kept:
            existing_terms = set(existing.content.lower().split())
            if not candidate_terms or not existing_terms:
                continue
            jaccard = len(candidate_terms & existing_terms) / len(
                candidate_terms | existing_terms
            )
            if jaccard >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def build_context(
    results: List[RetrievalResult],
    token_budget: int = 2400,
) -> Tuple[str, List[RetrievalResult]]:
    """Assemble numbered, delimited context blocks within a token budget.

    Returns both the rendered string and the results that actually made it in,
    because the citation validator must check the answer against the blocks
    the model saw rather than the blocks the retriever returned. Returning
    only the string is what produces phantom citation failures.

    Numbers are assigned after the final ordering. Assigning them before the
    edge-placement reorder is how citations end up pointing at the wrong
    source.
    """
    unique = deduplicate(results)

    admitted: List[RetrievalResult] = []
    spent = 0
    for result in unique:
        cost = estimate_tokens(result.content) + 20   # header overhead
        if spent + cost > token_budget:
            continue          # skip this block, keep trying smaller ones
        admitted.append(result)
        spent += cost

    # Strongest first, second strongest last. The middle holds the weakest,
    # which is exactly where attention is lowest.
    if len(admitted) > 2:
        reordered = [admitted[0]] + admitted[2:] + [admitted[1]]
    else:
        reordered = admitted

    blocks = []
    for position, result in enumerate(reordered, start=1):
        header = f"[{position}] source={result.source_uri}"
        if result.section_path:
            header += f" section={result.section_path}"
        header += f" relevance={result.score:.3f}"
        blocks.append(f"{header}\n{result.content.strip()}")

    return "\n\n".join(blocks), reordered


# ---------------------------------------------------------------------------
# Citation validation, Gate 3
# ---------------------------------------------------------------------------
def validate_citations(
    answer: str,
    admitted: List[RetrievalResult],
) -> Tuple[bool, Dict[str, object]]:
    """Check that every sentence carries a citation to a block that exists.

    A cheap, deterministic groundedness gate. It does not verify that the
    cited block actually supports the claim, which needs an LLM judge, but it
    catches fabricated block numbers and uncited assertions, which together
    account for most attribution defects on small models.
    """
    valid_ids = set(range(1, len(admitted) + 1))
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(answer) if s.strip()]

    uncited: List[str] = []
    invalid_refs: List[int] = []

    for sentence in sentences:
        cited = {int(m) for m in CITATION_PATTERN.findall(sentence)}
        if not cited:
            uncited.append(sentence)
            continue
        invalid_refs.extend(sorted(cited - valid_ids))

    report = {
        "sentence_count": len(sentences),
        "uncited_sentences": uncited,
        "hallucinated_block_ids": sorted(set(invalid_refs)),
        "citation_coverage": (
            1.0 - len(uncited) / len(sentences) if sentences else 0.0
        ),
    }
    passed = not uncited and not invalid_refs
    return passed, report


# ---------------------------------------------------------------------------
# Conversation-aware retrieval
# ---------------------------------------------------------------------------
def condense_question(question: str, history: List[Dict[str, str]], llm) -> str:
    """Rewrite a follow-up into a standalone query, skipping the LLM when possible.

    The marker check avoids an unnecessary model call on the large majority of
    turns, which removes 200 to 600 ms from the latency budget of every self
    contained question.

    Never retrieve on raw conversation history. Concatenating the whole
    conversation and embedding it produces a vector that averages every topic
    discussed so far, so retrieval drifts toward the conversation centroid and
    away from the current question.
    """
    if not history:
        return question

    lowered = f" {question.lower()} "
    depends = any(f" {marker} " in lowered for marker in DEPENDENT_MARKERS)
    if not depends and len(question.split()) > 6:
        return question

    recent = history[-4:]   # two exchanges, bounded by construction
    rendered = "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)

    response = llm.invoke(
        CONDENSE_PROMPT.format(history=rendered, question=question)
    )
    rewritten = (
        response.content if hasattr(response, "content") else str(response)
    ).strip()

    # Guard against a degenerate rewrite, which small models produce roughly
    # 2 to 5 percent of the time.
    if not rewritten or len(rewritten) > 4 * len(question) + 80:
        return question
    return rewritten


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
@dataclass
class RAGResponse:
    """Everything the interface and the evaluator need, in one object."""

    answer: str
    sources: List[RetrievalResult] = field(default_factory=list)
    abstained: bool = False
    citation_report: Dict[str, object] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    condensed_query: str = ""


class RAGPipeline:
    """Complete basic RAG with condensation, budgeting, and three gates."""

    def __init__(
        self,
        llm,
        embeddings,
        retriever: Retriever,
        min_score: float = 0.35,
        token_budget: int = 2400,
        max_regenerations: int = 1,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.retriever = retriever
        self.min_score = min_score
        self.token_budget = token_budget
        self.max_regenerations = max_regenerations

    def query(
        self,
        question: str,
        tenant_id: str = "default",
        top_k: int = 5,
        history: Optional[List[Dict[str, str]]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        timings: Dict[str, float] = {}
        history = history or []

        # --- Condensation ------------------------------------------------
        started = time.perf_counter()
        standalone = condense_question(question, history, self.llm)
        timings["condense"] = (time.perf_counter() - started) * 1000

        # --- Retrieval ---------------------------------------------------
        started = time.perf_counter()
        results = self.retriever.retrieve(
            standalone, tenant_id, top_k, metadata_filter
        )
        timings["retrieve"] = (time.perf_counter() - started) * 1000

        # --- Gate 1: score floor, enforced in code, not by the model ------
        confident = [r for r in results if r.score >= self.min_score]
        if not confident:
            logger.info(
                "abstain.no_evidence query=%r best=%.3f",
                standalone,
                max((r.score for r in results), default=0.0),
            )
            return RAGResponse(
                answer=("I could not find supporting information in the "
                        "indexed documents for that question."),
                sources=[],
                abstained=True,
                citation_report={"reason": "no_confident_results"},
                timings_ms=timings,
                condensed_query=standalone,
            )

        # --- Context construction ----------------------------------------
        started = time.perf_counter()
        context, admitted = build_context(confident, self.token_budget)
        timings["context"] = (time.perf_counter() - started) * 1000

        # --- Generation, with one regeneration on a citation failure ------
        started = time.perf_counter()
        answer, report = self._generate_and_validate(
            question, context, admitted, history
        )
        timings["generate"] = (time.perf_counter() - started) * 1000

        # --- Gate 2: explicit abstention token ---------------------------
        if ABSTENTION_TOKEN in answer:
            return RAGResponse(
                answer=("The documents I can access do not contain enough "
                        "information to answer that."),
                sources=admitted,
                abstained=True,
                citation_report={"reason": "model_abstained"},
                timings_ms=timings,
                condensed_query=standalone,
            )

        return RAGResponse(
            answer=answer,
            sources=admitted,
            abstained=False,
            citation_report=report,
            timings_ms=timings,
            condensed_query=standalone,
        )

    def _generate_and_validate(self, question, context, admitted, history):
        """Generate, then apply Gate 3 and regenerate once on failure."""
        recent = history[-4:]
        rendered_history = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in recent
        ) or "None"

        prompt = (
            f"{RAG_SYSTEM_PROMPT}\n\nCONTEXT\n{context}\n\n"
            f"RECENT CONVERSATION\n{rendered_history}\n\n"
            f"QUESTION\n{question}\n\nANSWER"
        )

        answer, report = "", {}
        for attempt in range(self.max_regenerations + 1):
            response = self.llm.invoke(prompt)
            answer = (
                response.content if hasattr(response, "content") else str(response)
            ).strip()

            if ABSTENTION_TOKEN in answer:
                return answer, {"reason": "model_abstained"}

            passed, report = validate_citations(answer, admitted)
            if passed or attempt == self.max_regenerations:
                report["regenerations"] = attempt
                return answer, report

            # Gate 3 failed. Make the failure explicit in the retry prompt.
            logger.info("citation.retry coverage=%.2f", report["citation_coverage"])
            prompt += (
                f"\n\nYOUR PREVIOUS ANSWER FAILED VALIDATION.\n"
                f"Uncited sentences: {len(report['uncited_sentences'])}.\n"
                f"Invalid block ids: {report['hallucinated_block_ids']}.\n"
                f"Rewrite the answer so every sentence ends with a valid "
                f"block citation, or output {ABSTENTION_TOKEN}."
            )

        return answer, report

    # -- ingestion and stats, used by the Streamlit app ---------------------
    def ingest_documents(self, files, tenant_id: str = "default",
                         chapter: str = "ch4") -> int:
        """Chunk, embed, and upsert uploaded files."""
        from chapter_02.chunkers import recursive_chunking
        from chapter_02.loaders import load_document
        from shared.db_utils import create_vector_table, store_embeddings

        create_vector_table()
        total = 0
        for handle in files:
            text, metadata = load_document(handle, with_metadata=True)
            chunks = recursive_chunking(text, size=600, overlap=72)
            if not chunks:
                continue
            vectors = self.embeddings.embed_documents(chunks)
            total += store_embeddings(
                chunks,
                vectors,
                metadatas=[{**metadata, "chunk": i} for i in range(len(chunks))],
                chapter=chapter,
                tenant_id=tenant_id,
                source_uri=getattr(handle, "name", ""),
            )
        return total

    def get_stats(self, tenant_id: str = "default",
                  chapter: str = "ch4") -> Dict[str, int]:
        from shared.db_utils import get_document_count

        try:
            return {"document_count": get_document_count(chapter=chapter,
                                                         tenant_id=tenant_id)}
        except Exception:
            return {"document_count": 0}
