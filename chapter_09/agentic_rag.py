"""Chapter 9: agentic GraphRAG with LangGraph.

Book reference:
    Chapter 9, "LangGraph Fundamentals"
    Chapter 9, "Designing the Node Set"
    Chapter 9, "Building the Graph"

Every node here makes a real decision. The version this replaces returned
hardcoded strings, which meant the workflow ran but the trace was theatre.

Two bounds guard every cycle: an iteration counter in state, which protects
answer quality, and a recursion limit in configuration, which protects the
process. Both are required, because the counter can be bypassed by a routing
bug and the recursion limit produces an exception rather than an answer.
"""

import json
import logging
import operator
import time
from functools import partial
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from shared.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def merge_unique(left: List[RetrievalResult],
                 right: List[RetrievalResult]) -> List[RetrievalResult]:
    """Accumulate evidence across hops without duplicating chunks.

    A plain operator.add reducer would accumulate the same chunk once per hop,
    which inflates the context, wastes budget, and biases the generator toward
    whichever chunk happened to be retrieved twice.
    """
    left = left or []
    right = right or []
    seen = {result.chunk_id for result in left}
    return left + [r for r in right if r.chunk_id not in seen]


class AgentState(TypedDict, total=False):
    # --- Inputs, written once -------------------------------------------
    question: str
    tenant_id: str
    history: List[Dict[str, str]]
    deadline_ts: float

    # --- Routing decisions, last write wins -----------------------------
    query_kind: Literal["factual", "conceptual", "comparative",
                        "relational", "aggregate"]
    active_query: str                 # the current, possibly rewritten query
    selected_route: Literal["vector", "graph", "hybrid", "direct"]

    # --- Accumulated across hops ----------------------------------------
    evidence: Annotated[List[RetrievalResult], merge_unique]
    entities: Annotated[List[str], operator.add]
    reasoning: Annotated[List[str], operator.add]
    hop_queries: Annotated[List[str], operator.add]

    # --- Loop control, never accumulated --------------------------------
    iteration: int
    max_iterations: int
    evidence_sufficient: bool
    coverage_gaps: List[str]

    # --- Outputs ---------------------------------------------------------
    answer: str
    verification: Dict[str, Any]
    confidence: float
    terminated_by: str                # why the graph stopped, for tracing
    needs_human: bool


ROUTE_BY_KIND = {
    "factual": "vector", "conceptual": "vector",
    "relational": "graph", "comparative": "hybrid", "aggregate": "hybrid",
}

EVALUATE_PROMPT = """Decide whether the evidence can answer the question.

Return JSON with exactly these keys:
  sufficient  - true or false
  gaps        - array of specific missing facts, empty when sufficient
  next_query  - a NEW search query targeting the largest gap, or ""

RULES
1. sufficient is true only if every part of the question is covered.
2. next_query must differ substantially from the queries already tried.
3. Do not answer the question. Only judge the evidence.

QUESTION
{question}

QUERIES ALREADY TRIED
{tried}

EVIDENCE
{evidence}

JSON"""

REWRITE_PROMPT = """Rewrite the question as a search query.

RULES
1. Expand abbreviations and add domain vocabulary.
2. Preserve every entity name exactly.
3. Output the query only.

QUESTION: {question}
QUERY:"""

VERIFY_PROMPT = """Is every claim in the answer supported by the evidence?

Reply with one word, GROUNDED or UNGROUNDED, then a confidence from 0.0 to 1.0.
Example: GROUNDED 0.85

EVIDENCE
{context}

ANSWER
{answer}

VERDICT:"""


def _text(response) -> str:
    return response.content if hasattr(response, "content") else str(response)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def classify_query(state: AgentState, llm=None) -> dict:
    """Choose the retrieval route before spending anything on retrieval."""
    from chapter_05.retrievers import AdaptiveRetriever

    question = state.get("question", "")
    classifier = AdaptiveRetriever(hybrid=None, reranker=None, llm=llm)
    kind = classifier.classify(question).value if llm is not None else "conceptual"
    route = ROUTE_BY_KIND.get(kind, "hybrid")

    return {
        "query_kind": kind,
        "selected_route": route,
        "active_query": question,
        "iteration": 0,
        "reasoning": [f"classified as {kind}, routing to {route}"],
    }


def rewrite_query(state: AgentState, llm=None) -> dict:
    """Expand the question into a retrievable form, once, at the start."""
    question = state.get("active_query") or state.get("question", "")
    if llm is None:
        return {"active_query": question, "reasoning": ["rewrite skipped, no llm"]}

    try:
        rewritten = _text(llm.invoke(REWRITE_PROMPT.format(question=question))).strip()
    except Exception as exc:
        logger.warning("agent.rewrite_failed %s", exc)
        return {"active_query": question, "reasoning": ["rewrite failed, using original"]}

    if not rewritten or len(rewritten) > 4 * len(question) + 80:
        return {"active_query": question, "reasoning": ["rewrite rejected as degenerate"]}
    return {"active_query": rewritten, "reasoning": [f"rewrote to {rewritten!r}"]}


def extract_entities(state: AgentState, llm=None) -> dict:
    """Find the graph anchors for the current query."""
    from chapter_08.graph_rag import EntityLinker

    query = state.get("active_query") or state.get("question", "")
    mentions = EntityLinker(llm=llm).mentions(query)
    return {
        "entities": mentions,
        "reasoning": [f"entities: {', '.join(mentions) if mentions else 'none'}"],
    }


def retrieve_node(state: AgentState, retriever=None, label: str = "vector") -> dict:
    """Run one retrieval hop with the retriever bound to this node."""
    query = state.get("active_query") or state.get("question", "")
    tenant_id = state.get("tenant_id", "default")

    if retriever is None:
        return {"evidence": [], "reasoning": [f"{label}: no retriever configured"]}

    try:
        results = retriever.retrieve(query, tenant_id, 8)
    except Exception as exc:
        logger.error("agent.retrieve_failed label=%s %s", label, exc)
        results = []

    return {
        "evidence": results,
        "reasoning": [f"{label} retrieved {len(results)} results for {query!r}"],
    }


def evaluate_context(state: AgentState, llm=None) -> dict:
    """Grade the accumulated evidence and propose a genuinely new query.

    This node is where agency actually lives. It must produce a next_query
    that differs from every query already tried, otherwise the cycle repeats a
    deterministic retrieval and burns the whole budget.
    """
    evidence = state.get("evidence", [])
    tried = list(state.get("hop_queries", [])) + [state.get("question", "")]

    if not evidence:
        return {
            "evidence_sufficient": False,
            "coverage_gaps": ["no evidence retrieved"],
            "reasoning": ["evaluate: zero evidence, escalating"],
        }

    if llm is None:
        return {"evidence_sufficient": True,
                "reasoning": ["evaluate: no judge configured, proceeding"]}

    rendered = "\n\n".join(
        f"[{i}] {r.content[:600]}" for i, r in enumerate(evidence[:12], start=1)
    )
    try:
        raw = _text(llm.invoke(EVALUATE_PROMPT.format(
            question=state.get("question", ""),
            tried="\n".join(f"- {q}" for q in tried),
            evidence=rendered,
        )))
        start, end = raw.find("{"), raw.rfind("}")
        verdict = json.loads(raw[start:end + 1])
    except (ValueError, json.JSONDecodeError, AttributeError) as exc:
        # A judge failure must not trap the graph in a loop. Treat an
        # unparseable verdict as sufficient and let verification catch it.
        logger.warning("agent.evaluate_parse_failed %s", exc)
        return {
            "evidence_sufficient": True,
            "reasoning": ["evaluate: parse failed, proceeding to generation"],
        }

    sufficient = bool(verdict.get("sufficient"))
    next_query = str(verdict.get("next_query", "")).strip()

    # Refuse a query that repeats a previous hop. Repetition is the signature
    # of a planner that has nothing new to try, and on a deterministic
    # retriever it guarantees a wasted iteration.
    if next_query and next_query.lower() in {q.lower() for q in tried}:
        logger.info("agent.repeated_query suppressed=%r", next_query)
        next_query = ""
        sufficient = True

    return {
        "evidence_sufficient": sufficient,
        "coverage_gaps": list(verdict.get("gaps", [])),
        "active_query": next_query or state.get("active_query", ""),
        "reasoning": [
            f"evaluate: sufficient={sufficient}, gaps={verdict.get('gaps', [])[:3]}"
        ],
    }


def plan_next_hop(state: AgentState) -> dict:
    """Advance the loop counter and record the query being tried next."""
    iteration = state.get("iteration", 0) + 1
    next_query = state.get("active_query", "")
    return {
        "iteration": iteration,
        "hop_queries": [next_query] if next_query else [],
        "reasoning": [f"hop {iteration}: retrying with {next_query!r}"],
    }


def generate_answer(state: AgentState, llm=None) -> dict:
    """Produce the answer from the accumulated evidence."""
    from chapter_04.rag_complete import RAG_SYSTEM_PROMPT, build_context

    evidence = state.get("evidence", [])
    if not evidence:
        return {"answer": "I could not find supporting information for that question.",
                "terminated_by": "no_evidence",
                "reasoning": ["generate: abstained, no evidence"]}

    context, _ = build_context(evidence, token_budget=2400)
    if llm is None:
        return {"answer": context[:400], "terminated_by": "no_llm",
                "reasoning": ["generate: no llm configured"]}

    prompt = (f"{RAG_SYSTEM_PROMPT}\n\nCONTEXT\n{context}\n\n"
              f"QUESTION\n{state.get('question', '')}\n\nANSWER")
    answer = _text(llm.invoke(prompt)).strip()

    terminated = ("sufficient_evidence" if state.get("evidence_sufficient")
                  else "iteration_cap")
    return {"answer": answer, "terminated_by": terminated,
            "reasoning": [f"generated {len(answer)} chars"]}


def verify_answer(state: AgentState, llm=None,
                  confidence_floor: float = 0.6) -> dict:
    """Check the answer against the evidence and decide whether to escalate."""
    from chapter_04.rag_complete import validate_citations

    evidence = state.get("evidence", [])
    answer = state.get("answer", "")

    passed, report = validate_citations(answer, evidence)
    confidence = float(report.get("citation_coverage", 0.0))

    if llm is not None and evidence:
        context = "\n\n".join(r.content for r in evidence[:8])
        try:
            raw = _text(llm.invoke(
                VERIFY_PROMPT.format(context=context[:6000], answer=answer)
            )).strip().upper()
            grounded = raw.startswith("GROUNDED")
            for token in raw.replace(",", " ").split():
                try:
                    confidence = min(confidence, float(token))
                    break
                except ValueError:
                    continue
            passed = passed and grounded
        except Exception as exc:
            logger.warning("agent.verify_failed %s", exc)

    return {
        "verification": {**report, "passed": passed},
        "confidence": confidence,
        "needs_human": (not passed) or confidence < confidence_floor,
        "reasoning": [f"verify: passed={passed}, confidence={confidence:.2f}"],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_query(state: AgentState) -> str:
    return state.get("selected_route", "hybrid")


def route_after_evaluation(state: AgentState) -> str:
    """Conditional edge. Every exit path must be reachable and bounded."""
    if state.get("evidence_sufficient"):
        return "generate"
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "generate"          # cap reached, answer with what we have
    if not state.get("active_query"):
        return "generate"          # planner produced nothing new
    deadline = state.get("deadline_ts")
    if deadline and time.time() >= deadline:
        return "generate"          # wall clock budget spent
    return "continue"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_agent_graph(llm=None, embeddings=None,
                      retrievers: Optional[Dict[str, Any]] = None,
                      checkpointer=None):
    """Compile the agentic GraphRAG workflow."""
    from langgraph.graph import END, START, StateGraph

    retrievers = retrievers or {}
    graph = StateGraph(AgentState)

    graph.add_node("classify_query", partial(classify_query, llm=llm))
    graph.add_node("rewrite_query", partial(rewrite_query, llm=llm))
    graph.add_node("extract_entities", partial(extract_entities, llm=llm))
    graph.add_node("vector_retrieve",
                   partial(retrieve_node, retriever=retrievers.get("vector"),
                           label="vector"))
    graph.add_node("graph_retrieve",
                   partial(retrieve_node, retriever=retrievers.get("graph"),
                           label="graph"))
    graph.add_node("hybrid_retrieve",
                   partial(retrieve_node, retriever=retrievers.get("hybrid"),
                           label="hybrid"))
    graph.add_node("evaluate_context", partial(evaluate_context, llm=llm))
    graph.add_node("plan_next_hop", plan_next_hop)
    graph.add_node("generate_answer", partial(generate_answer, llm=llm))
    graph.add_node("verify_answer", partial(verify_answer, llm=llm))

    graph.add_edge(START, "classify_query")
    graph.add_edge("classify_query", "rewrite_query")
    graph.add_edge("rewrite_query", "extract_entities")

    graph.add_conditional_edges(
        "extract_entities",
        route_query,
        {"vector": "vector_retrieve",
         "graph": "graph_retrieve",
         "hybrid": "hybrid_retrieve"},
    )

    for node in ("vector_retrieve", "graph_retrieve", "hybrid_retrieve"):
        graph.add_edge(node, "evaluate_context")

    graph.add_conditional_edges(
        "evaluate_context",
        route_after_evaluation,
        {"continue": "plan_next_hop", "generate": "generate_answer"},
    )
    # The cycle. It re-enters at entity extraction so a rewritten query gets
    # fresh graph anchors rather than reusing the original ones.
    graph.add_edge("plan_next_hop", "extract_entities")

    graph.add_edge("generate_answer", "verify_answer")
    graph.add_edge("verify_answer", END)

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


def run_agent(app, question: str, tenant_id: str = "default",
              thread_id: str = "lab", max_iterations: int = 3,
              deadline_seconds: float = 15.0) -> AgentState:
    """Invoke the compiled graph with every bound applied."""
    initial: AgentState = {
        "question": question,
        "tenant_id": tenant_id,
        "iteration": 0,
        "max_iterations": max_iterations,
        "deadline_ts": time.time() + deadline_seconds,
        "evidence": [],
        "reasoning": [],
        "hop_queries": [],
        "entities": [],
    }
    config = {
        "configurable": {"thread_id": thread_id},
        # Hard ceiling on total node executions regardless of routing bugs.
        "recursion_limit": 6 * max_iterations + 12,
    }
    return app.invoke(initial, config=config)
