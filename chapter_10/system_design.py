"""Chapter 10: production reliability and cost patterns.

Book reference:
    Chapter 10, "Reliability"
    Chapter 10, "Cost Optimization"

The breaker trips on latency as well as on errors. A model endpoint that
answers in 30 seconds is worse for a user than one that fails fast, because
the request occupies a worker and the user waits. An error-only breaker never
opens on that condition.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Trips on error rate or on latency, whichever degrades first."""

    def __init__(self, failure_threshold: int = 5, window_seconds: float = 60.0,
                 latency_threshold_ms: float = 8000.0,
                 slow_call_ratio: float = 0.5, reset_timeout: float = 30.0,
                 half_open_probes: int = 3, min_calls: int = 5):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.latency_threshold_ms = latency_threshold_ms
        self.slow_call_ratio = slow_call_ratio
        self.reset_timeout = reset_timeout
        self.half_open_probes = half_open_probes
        # A ratio computed over one call is not a signal. Without a minimum
        # call volume, the first slow request opens the breaker and takes the
        # dependency out of rotation on no evidence at all.
        self.min_calls = min_calls

        self._state = BreakerState.CLOSED
        self._opened_at = 0.0
        self._probes_remaining = 0
        self._events: Deque[Tuple[float, bool, float]] = deque()
        self._lock = threading.Lock()

    @property
    def state(self) -> BreakerState:
        return self._state

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] > self.window_seconds:
            self._events.popleft()

    def _should_trip(self, now: float) -> bool:
        self._prune(now)
        if not self._events:
            return False
        failures = sum(1 for _, ok, _ in self._events if not ok)
        if failures >= self.failure_threshold:
            return True

        # The slow-call ratio only becomes meaningful once enough calls have
        # landed in the window.
        if len(self._events) < self.min_calls:
            return False
        slow = sum(
            1 for _, _, ms in self._events if ms >= self.latency_threshold_ms
        )
        return slow / len(self._events) >= self.slow_call_ratio

    def call(self, func: Callable, *args, **kwargs):
        now = time.monotonic()

        with self._lock:
            if self._state is BreakerState.OPEN:
                if now - self._opened_at < self.reset_timeout:
                    raise RuntimeError("circuit_open")
                self._state = BreakerState.HALF_OPEN
                self._probes_remaining = self.half_open_probes
                logger.info("breaker.half_open")

        started = time.monotonic()
        try:
            result = func(*args, **kwargs)
        except Exception:
            self._record(False, (time.monotonic() - started) * 1000)
            raise
        else:
            self._record(True, (time.monotonic() - started) * 1000)
            return result

    def _record(self, ok: bool, latency_ms: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._events.append((now, ok, latency_ms))

            if self._state is BreakerState.HALF_OPEN:
                if not ok:
                    self._state = BreakerState.OPEN
                    self._opened_at = now
                    logger.warning("breaker.reopened")
                    return
                self._probes_remaining -= 1
                if self._probes_remaining <= 0:
                    self._state = BreakerState.CLOSED
                    self._events.clear()
                    logger.info("breaker.closed")
                return

            if self._state is BreakerState.CLOSED and self._should_trip(now):
                self._state = BreakerState.OPEN
                self._opened_at = now
                logger.warning("breaker.opened events=%d", len(self._events))


class RetryWithBackoff:
    """Exponential backoff with jitter, capped.

    Never wrap a non-idempotent generation call in a blind retry: the cost
    doubles and the user waits twice.
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 8.0, jitter: float = 0.1):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def execute(self, func: Callable, *args, **kwargs):
        import random

        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last = exc
                if attempt == self.max_retries - 1:
                    break
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                time.sleep(delay + random.uniform(0, self.jitter * delay))
        raise last


# ---------------------------------------------------------------------------
# Degradation ladder
# ---------------------------------------------------------------------------
@dataclass
class DegradationRung:
    """One step of the degradation ladder, with its measured quality cost."""

    name: str
    apply: Callable[[Dict], Dict]
    saves_ms: float
    quality_cost: str          # measured on the golden set, not guessed


LADDER: List[DegradationRung] = [
    DegradationRung("lower_ef_search",
                    lambda c: {**c, "ef_search": 40},
                    saves_ms=8, quality_cost="recall@10 -0.4 points"),
    DegradationRung("disable_rerank",
                    lambda c: {**c, "rerank": False},
                    saves_ms=220, quality_cost="precision@5 -11 points"),
    DegradationRung("shrink_context",
                    lambda c: {**c, "top_k": 3, "token_budget": 1200},
                    saves_ms=400, quality_cost="faithfulness -3 points"),
    DegradationRung("disable_condensation",
                    lambda c: {**c, "condense": False},
                    saves_ms=250, quality_cost="multi-turn recall -9 points"),
]


def degrade(config: Dict, pressure: int) -> Dict:
    """Apply the first `pressure` rungs of the ladder, in order.

    Each rung is a deliberate, measured trade rather than an emergency guess,
    which is what lets an operator explain the quality drop during an incident
    instead of discovering it afterwards.
    """
    for rung in LADDER[: max(0, pressure)]:
        config = rung.apply(config)
        logger.warning("degrade.applied rung=%s cost=%s",
                       rung.name, rung.quality_cost)
    return config


# ---------------------------------------------------------------------------
# Cost control
# ---------------------------------------------------------------------------
class SemanticCache:
    """Cache keyed on query embedding similarity rather than exact text.

    Set the threshold too low and it answers a different question, which is a
    correctness bug that looks like a hallucination. The cache key includes
    the retrieved chunk hashes so a document change invalidates the entry
    rather than serving a stale answer.
    """

    def __init__(self, threshold: float = 0.93, max_entries: int = 1000):
        self.threshold = threshold
        self.max_entries = max_entries
        self._entries: List[Dict[str, Any]] = []
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _cosine(a, b) -> float:
        import numpy as np

        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denominator) if denominator else 0.0

    def get(self, query_vector, tenant_id: str = "default",
            corpus_version: str = "") -> Optional[str]:
        for entry in self._entries:
            if entry["tenant_id"] != tenant_id:
                continue
            if entry["corpus_version"] != corpus_version:
                continue
            if self._cosine(query_vector, entry["vector"]) >= self.threshold:
                self._hits += 1
                return entry["answer"]
        self._misses += 1
        return None

    def put(self, query_vector, answer: str, tenant_id: str = "default",
            corpus_version: str = "") -> None:
        self._entries.append({
            "vector": list(query_vector),
            "answer": answer,
            "tenant_id": tenant_id,
            "corpus_version": corpus_version,
        })
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0


class TokenBudget:
    """Fit retrieved chunks into a token budget, whole blocks only.

    Truncating a block produces a half fact, and the model will complete it
    from parametric memory.
    """

    def __init__(self, max_tokens: int = 4096, chars_per_token: int = 4):
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // self.chars_per_token)

    def fit_context(self, chunks: List[str], budget: Optional[int] = None) -> List[str]:
        budget = budget or self.max_tokens
        selected, spent = [], 0
        for chunk in chunks:
            cost = self.estimate_tokens(chunk)
            if spent + cost > budget:
                continue
            selected.append(chunk)
            spent += cost
        return selected


# ---------------------------------------------------------------------------
# Interview practice content
# ---------------------------------------------------------------------------
REQUIREMENTS_CHECKLIST = [
    ("User scale", "Daily active users, concurrency at peak",
     "Replica count, connection pooling"),
    ("Document volume", "Count, average size, growth rate",
     "Index size, sharding, ingest throughput"),
    ("Data freshness", "Acceptable staleness after an edit",
     "Batch versus streaming ingestion"),
    ("Query volume", "Queries per second at p50 and peak",
     "Caching strategy, autoscaling policy"),
    ("Latency", "p50 and p95 targets, streaming allowed",
     "Reranking budget, model size"),
    ("Accuracy", "Faithfulness bar, cost of a wrong answer",
     "Abstention policy, verification depth"),
    ("Security", "Tenancy, roles, data residency, audit",
     "Isolation model, retention, logging"),
    ("Cost", "Budget per thousand queries",
     "Model choice, cache targets, K"),
]


def estimate_capacity(documents: int, pages_per_doc: int = 15,
                      words_per_page: int = 500, chunk_words: int = 300,
                      overlap: float = 0.12, dimensions: int = 768) -> Dict[str, float]:
    """Derive the second-order numbers out loud, as Chapter 10 requires.

    Stating chunk count, index size, and peak load, with the arithmetic
    visible, demonstrates operational experience more convincingly than any
    architecture drawn afterwards.
    """
    chunks = documents * pages_per_doc * words_per_page * (1 + overlap) / chunk_words
    raw_gb = chunks * dimensions * 4 / 1e9
    return {
        "chunks": round(chunks),
        "raw_vector_gb": round(raw_gb, 1),
        "with_hnsw_gb": round(raw_gb * 2.6, 1),
        "text_metadata_gb": round(chunks * 2000 / 1e9, 1),
        "total_gb": round(raw_gb * 2.6 + chunks * 2000 / 1e9, 1),
    }


def estimate_load(daily_active_users: int, queries_per_user: int = 8,
                  peak_factor: float = 5.0, working_hours: int = 8) -> Dict[str, float]:
    """Size for the peak, not the average."""
    daily = daily_active_users * queries_per_user
    average_qps = daily / (working_hours * 3600)
    return {
        "queries_per_day": daily,
        "average_qps": round(average_qps, 2),
        "peak_qps": round(average_qps * peak_factor, 2),
    }


def get_interview_scenarios() -> List[Dict[str, Any]]:
    """The six scenarios from Chapter 10, for the practice interface."""
    return [
        {
            "title": "Enterprise Document Question Answering",
            "requirements": "200,000 documents, 5,000 daily users, 3 second p95, "
                            "faithfulness above 0.90, single tenant, 5 minute freshness.",
            "architecture": "Async ingestion from webhooks, structure aware chunking "
                            "with parent child units, pgvector with HNSW, hybrid "
                            "retrieval, cross encoder reranking to 5, grounded "
                            "generation with citation validation, response and "
                            "semantic caching.",
            "follow_up": "A user says the assistant is quoting a policy we retired "
                         "last year.",
            "hints": ["Document versioning", "Deletion propagation",
                      "Effective-date filter at retrieval, not a prompt change"],
        },
        {
            "title": "Multi-Tenant SaaS RAG",
            "requirements": "2,000 tenants, contractual isolation, 100 to 400,000 "
                            "documents per tenant, per-tenant configuration.",
            "architecture": "Tiered isolation, row level security for small tenants, "
                            "dedicated partitions for large ones, tenant resolution "
                            "at the gateway, SET LOCAL inside the query transaction, "
                            "per-tenant HNSW indexes.",
            "follow_up": "Prove that tenant A cannot see tenant B's data.",
            "hints": ["Gateway scoping", "Database policy", "Physical partitioning",
                      "A continuous cross-tenant probe in production"],
        },
        {
            "title": "Customer Support Assistant",
            "requirements": "500 queries per second at peak, 1.5 second p95, "
                            "deflection measured in tickets avoided, multilingual.",
            "architecture": "Three-layer caching, small routing model, hybrid "
                            "retrieval with a lexical branch for error codes, "
                            "streaming generation, escalation on low confidence.",
            "follow_up": "Deflection went up and satisfaction went down.",
            "hints": ["Deflection can rise because users give up",
                      "Instrument follow-up rate, rephrase rate, escalations"],
        },
        {
            "title": "Legal and Compliance Assistant",
            "requirements": "Zero tolerance for uncited claims, full audit trail, "
                            "jurisdiction filtering, seven year retention.",
            "architecture": "Span-level attribution with mandatory verification, "
                            "conservative abstention, jurisdiction and effective-date "
                            "as hard filters, human review interrupt, immutable audit "
                            "log of identity, query, chunk ids, and answer hash.",
            "follow_up": "How do you prove to an auditor what the system saw?",
            "hints": ["The trace record", "Content-hashed chunks",
                      "Replay from stored identifiers, not stored prompt text"],
        },
        {
            "title": "Multi-Hop Research Assistant",
            "requirements": "Two to four retrieval steps, 15 second budget, "
                            "transparency about reasoning.",
            "architecture": "LangGraph workflow, router sending single lookups to a "
                            "fixed pipeline, bounded iteration with query novelty "
                            "enforcement, wall clock deadline in state, durable "
                            "checkpointing, reasoning trace in the interface.",
            "follow_up": "How do you stop it looping?",
            "hints": ["Both bounds", "The novelty rule",
                      "Zero-new-evidence early exit", "Termination reason metric"],
        },
        {
            "title": "Agentic GraphRAG for Risk Analysis",
            "requirements": "Relationship questions over contracts and incidents, "
                            "corpus-wide theme reports, regulated data.",
            "architecture": "Structured import where relationships already exist, "
                            "constrained extraction elsewhere, Neo4j partitioned per "
                            "tenant, local search for entity questions, nightly "
                            "community summaries, unified reranking, human review.",
            "follow_up": "Justify the graph.",
            "hints": ["Measured fraction of queries requiring traversal",
                      "Accuracy delta on that subset",
                      "Graph share of admitted context"],
        },
    ]


def get_architecture_diagram() -> str:
    """Mermaid source for the production reference architecture."""
    return """```mermaid
graph TD
    Client[Clients] --> GW[API Gateway<br/>authn, quota, WAF]
    GW --> Cache[Cache Tier<br/>response, semantic]
    Cache -- miss --> Orch[LangGraph Orchestrator]
    Orch --> Retr[Retrieval Service<br/>hybrid, rerank]
    Retr --> PG[(pgvector)]
    Retr --> Neo[(Neo4j)]
    Orch --> Gen[Model Serving]
    CDC[Source Systems] --> Queue[Durable Queue]
    Queue --> Workers[Worker Pool]
    Workers --> Gate{Quality Gate}
    Gate -- upsert --> Retr
    Orch -.trace.-> Trace[(Trace Store)]
    Trace --> Monitors[Monitors + sampled judge]
    Monitors --> Alerts[Alerts]
```"""
