"""Chapter 8: GraphRAG, local search, and hybrid context fusion.

Book reference:
    Chapter 8, "Local Search, Entity-Centric Retrieval"
    Chapter 8, "Hybrid Retrieval and Context Fusion"

The traversal is bounded three ways: by depth, by node degree, and by a
transaction timeout. A result LIMIT caps what is returned and not what is
computed, because the planner must enumerate matching paths before limiting
them. On a graph containing a hub, that enumeration exhausts heap before the
limit ever applies.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shared.retrieval import RetrievalResult, Retriever

logger = logging.getLogger(__name__)

# Degree above which a node is treated as a hub and skipped during traversal.
HUB_DEGREE_THRESHOLD = 500

LOCAL_SEARCH_CYPHER = """
MATCH (seed:Entity {tenant_id: $tenant_id, canonical_name: $canonical})
MATCH path = (seed)-[rels:RELATES*1..%(max_hops)d]-(reached:Entity)
WHERE all(r IN rels WHERE r.confidence >= $min_confidence)
  // Skip hubs at every hop. This is the bound that actually prevents
  // explosion, because it applies during expansion rather than after.
  AND all(n IN nodes(path)
          WHERE n = seed OR count { (n)--() } < $hub_threshold)
  AND reached.tenant_id = $tenant_id
RETURN [n IN nodes(path) | n.name]                   AS names,
       [r IN relationships(path) | r.rel_type]       AS types,
       [r IN relationships(path) | r.evidence]       AS evidence,
       [r IN relationships(path) | r.source_uri]     AS sources,
       reduce(s = 1.0, r IN relationships(path) | s * r.confidence)
                                                     AS path_confidence,
       length(path)                                  AS hops
ORDER BY path_confidence DESC, hops ASC
LIMIT $path_limit
"""


# ---------------------------------------------------------------------------
# Entity linking, the weak link in local search
# ---------------------------------------------------------------------------
class EntityLinker:
    """Resolve mentions in a question to graph nodes.

    Local search fails most often here. The user writes "the Meridian deal",
    the graph stores "Meridian Acquisition Project", and exact matching
    returns nothing. The cascade runs cheapest first: canonicalization, then
    an alias dictionary, then fuzzy matching constrained by type.

    Falling back to vector retrieval when linking fails is mandatory. An
    unlinked entity otherwise produces an empty graph context and a silent
    failure.
    """

    EXTRACT_PROMPT = (
        "List the named entities in the question. Output a JSON array of "
        'strings, for example ["Acme", "Meridian"]. Nothing else.\n\n'
        "QUESTION: {question}\n\nJSON:"
    )

    def __init__(self, llm=None, driver=None, fuzzy_threshold: float = 0.86,
                 alias_map: Optional[Dict[str, str]] = None):
        self.llm = llm
        self.driver = driver
        self.fuzzy_threshold = fuzzy_threshold
        self.alias_map = alias_map or {}

    def mentions(self, question: str) -> List[str]:
        """Extract candidate mentions, leniently.

        A false mention costs a failed lookup. A missed mention costs the
        whole query, so this errs toward over-extraction.
        """
        from chapter_06.evaluator import _extract_json_array

        if self.llm is not None:
            try:
                response = self.llm.invoke(
                    self.EXTRACT_PROMPT.format(question=question)
                )
                raw = (response.content if hasattr(response, "content")
                       else str(response))
                parsed = _extract_json_array(raw)
                if parsed:
                    return [str(p).strip() for p in parsed if str(p).strip()]
            except Exception as exc:
                logger.warning("linker.extract_failed %s", exc)

        # Fallback: capitalized runs, which catches most proper nouns.
        import re
        return re.findall(r"\b[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)*", question)

    def _candidates(self, tenant_id: str) -> List[Dict[str, Any]]:
        from shared.neo4j_utils import run_cypher

        return run_cypher(
            "MATCH (e:Entity {tenant_id: $tenant_id}) "
            "RETURN e.canonical_name AS canonical_name, e.name AS name, "
            "       e.entity_type AS entity_type",
            {"tenant_id": tenant_id},
        )

    def link(self, question: str, tenant_id: str = "default"):
        """Return (linked, unlinked) for the mentions found in the question."""
        from chapter_07.kg_builder import canonicalize
        from difflib import SequenceMatcher

        mentions = self.mentions(question)
        if not mentions:
            return [], []

        try:
            candidates = self._candidates(tenant_id)
        except Exception as exc:
            logger.error("linker.candidates_failed %s", exc)
            return [], mentions

        by_canonical = {c["canonical_name"]: c for c in candidates if c.get("canonical_name")}
        linked, unlinked = [], []

        for mention in mentions:
            key = canonicalize(self.alias_map.get(mention, mention))

            if key in by_canonical:                       # exact canonical hit
                linked.append(by_canonical[key])
                continue

            best, best_ratio = None, 0.0                  # fuzzy, constrained
            for canonical, row in by_canonical.items():
                ratio = SequenceMatcher(None, key, canonical).ratio()
                if ratio > best_ratio:
                    best, best_ratio = row, ratio

            if best is not None and best_ratio >= self.fuzzy_threshold:
                linked.append(best)
            else:
                unlinked.append(mention)

        return linked, unlinked


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------
@dataclass
class GraphContext:
    """Graph evidence rendered for a prompt, with provenance preserved."""

    triples: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    source_uris: List[str] = field(default_factory=list)
    seed_entities: List[str] = field(default_factory=list)
    unlinked_entities: List[str] = field(default_factory=list)
    path_count: int = 0

    def render(self, max_triples: int = 40) -> str:
        if not self.triples:
            return ""
        lines = [f"ENTITIES: {', '.join(self.seed_entities)}", "RELATIONSHIPS:"]
        lines += [f"  {t}" for t in self.triples[:max_triples]]
        if any(self.evidence):
            lines.append("SUPPORTING QUOTES:")
            lines += [f'  "{e}"' for e in self.evidence[:10] if e]
        return "\n".join(lines)


class LocalGraphRetriever:
    """Entity-centric retrieval over the knowledge graph."""

    def __init__(self, driver=None, entity_linker: Optional[EntityLinker] = None,
                 max_hops: int = 2, min_confidence: float = 0.5,
                 path_limit: int = 120, timeout_seconds: int = 5):
        self.driver = driver
        self.entity_linker = entity_linker or EntityLinker()
        self.max_hops = max_hops
        self.min_confidence = min_confidence
        self.path_limit = path_limit
        self.timeout_seconds = timeout_seconds

    def _driver(self):
        if self.driver is not None:
            return self.driver
        from shared.neo4j_utils import get_neo4j_driver

        return get_neo4j_driver()

    def retrieve(self, question: str, tenant_id: str = "default") -> GraphContext:
        linked, unlinked = self.entity_linker.link(question, tenant_id)
        context = GraphContext(
            seed_entities=[e.get("name") or e.get("canonical_name") for e in linked],
            unlinked_entities=unlinked,
        )
        if not linked:
            # No anchor means no traversal. The caller must fall back to
            # vector retrieval rather than return an empty context.
            logger.info("graph.no_entities_linked question=%r", question[:80])
            return context

        cypher = LOCAL_SEARCH_CYPHER % {"max_hops": self.max_hops}
        seen_triples = set()

        with self._driver().session() as session:
            for entity in linked:
                try:
                    records = session.run(
                        cypher,
                        tenant_id=tenant_id,
                        canonical=entity["canonical_name"],
                        min_confidence=self.min_confidence,
                        hub_threshold=HUB_DEGREE_THRESHOLD,
                        path_limit=self.path_limit,
                        timeout=self.timeout_seconds,  # never hold a worker
                    )
                except Exception as exc:
                    logger.error("graph.traversal_failed %s", exc)
                    continue

                for record in records:
                    names = record["names"] or []
                    types = record["types"] or []
                    evidence = record["evidence"] or []
                    sources = record["sources"] or []

                    for index, rel_type in enumerate(types):
                        if index + 1 >= len(names):
                            break
                        triple = (
                            f"{names[index]} -[{rel_type}]-> {names[index + 1]}"
                        )
                        if triple in seen_triples:
                            continue
                        seen_triples.add(triple)
                        context.triples.append(triple)
                        context.evidence.append(
                            evidence[index] if index < len(evidence) else ""
                        )
                        context.source_uris.append(
                            sources[index] if index < len(sources) else "graph"
                        )
                    context.path_count += 1

        return context


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def triples_to_results(context: GraphContext) -> List[RetrievalResult]:
    """Render graph triples as citable, rerankable natural language.

    A cross encoder scores "Acme supplies Meridian, per the 2025 master
    agreement" correctly. It scores "Acme -[SUPPLIES]-> Meridian" poorly,
    because that string resembles nothing in its training distribution.
    """
    results: List[RetrievalResult] = []
    for index, triple in enumerate(context.triples):
        subject, _, remainder = triple.partition(" -[")
        rel_type, _, obj = remainder.partition("]-> ")
        sentence = (
            f"{subject.strip()} {rel_type.strip().lower().replace('_', ' ')} "
            f"{obj.strip()}."
        )
        quote = context.evidence[index] if index < len(context.evidence) else ""
        if quote:
            sentence += f' Supporting text: "{quote}"'

        source = (
            context.source_uris[index]
            if index < len(context.source_uris) else "graph"
        )
        results.append(
            RetrievalResult(
                content=sentence,
                score=0.0,                      # the reranker assigns the score
                source_uri=source or "graph",
                chunk_id=f"graph:{index}",
                metadata={"origin": "graph", "triple": triple},
                retriever="graph_local",
            )
        )
    return results


class HybridGraphRAG:
    """Fuse graph and vector evidence into one ranked, budgeted context.

    Concatenating a graph context and a vector context doubles token usage,
    pushes both toward the middle of the prompt where attention is weakest,
    and performs no deduplication across sources because the two
    representations share no surface form. Rendering triples as sentences and
    reranking the pooled set fixes all of that at the cost of one extra call.
    """

    def __init__(
        self,
        vector_retriever: Retriever,
        graph_retriever: LocalGraphRetriever,
        reranker,
        token_budget: int = 2600,
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.reranker = reranker
        self.token_budget = token_budget

    def retrieve(self, question: str, tenant_id: str = "default", llm=None,
                 top_k: int = 6) -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        from chapter_04.rag_complete import build_context

        with ThreadPoolExecutor(max_workers=2) as pool:
            vector_future = pool.submit(
                self.vector_retriever.retrieve, question, tenant_id, top_k * 4
            )
            graph_future = pool.submit(
                self.graph_retriever.retrieve, question, tenant_id
            )
            vector_results = vector_future.result()
            graph_context = graph_future.result()

        graph_results = triples_to_results(graph_context)

        # Graph-guided expansion: when traversal found entities the query
        # never named, retrieve passages about them too.
        discovered = [
            name for name in graph_context.seed_entities
            if name and name.lower() not in question.lower()
        ]
        if discovered:
            expansion = " ".join(discovered[:5])
            vector_results = vector_results + self.vector_retriever.retrieve(
                f"{question} {expansion}", tenant_id, top_k * 2
            )

        pooled = vector_results + graph_results
        if not pooled:
            return {"results": [], "context": "", "graph_context": graph_context,
                    "fusion": "empty", "graph_share": 0.0}

        ranked = (
            self.reranker.rerank(question, pooled, llm)
            if (self.reranker is not None and llm is not None)
            else sorted(pooled, key=lambda r: r.score, reverse=True)
        )

        context, admitted = build_context(ranked, self.token_budget)
        graph_share = (
            sum(1 for r in admitted if r.metadata.get("origin") == "graph")
            / max(1, len(admitted))
        )
        return {
            "results": admitted,
            "context": context,
            "graph_context": graph_context,
            "fusion": "unified_rerank",
            # The metric that answers whether the graph is actually
            # contributing. A share near zero on relational questions means
            # the graph costs an extraction pipeline and a database while
            # adding nothing to answers.
            "graph_share": graph_share,
        }


class GraphRAGPipeline:
    """Four comparable architectures over one corpus, as the Chapter 8 lab needs."""

    MODES = ("basic", "advanced", "graphrag", "hybrid")

    def __init__(self, llm, embeddings, chapter: str = "ch8"):
        from chapter_05.reranker import LLMReranker
        from chapter_05.retrievers import DenseRetriever, HybridRetriever, LexicalRetriever

        self.llm = llm
        self.embeddings = embeddings
        self.chapter = chapter

        self.dense = DenseRetriever(embeddings, chapter=chapter)
        self.lexical = LexicalRetriever(chapter=chapter)
        self.advanced = HybridRetriever(self.dense, self.lexical)
        self.reranker = LLMReranker()
        self.graph = LocalGraphRetriever(entity_linker=EntityLinker(llm=llm))
        self.hybrid = HybridGraphRAG(self.advanced, self.graph, self.reranker)

    def ingest(self, text: str, tenant_id: str = "default", source_uri: str = ""):
        """Write the same text into both the vector store and the graph."""
        from chapter_02.chunkers import recursive_chunking
        from chapter_07.kg_builder import build_knowledge_graph
        from shared.db_utils import create_vector_table, store_embeddings

        create_vector_table()
        chunks = recursive_chunking(text, size=600, overlap=72)
        vectors = self.embeddings.embed_documents(chunks)
        store_embeddings(
            chunks, vectors,
            metadatas=[{"source": source_uri, "chunk": i} for i in range(len(chunks))],
            chapter=self.chapter, tenant_id=tenant_id, source_uri=source_uri,
        )
        return build_knowledge_graph(
            text, self.llm, tenant_id=tenant_id,
            source_uri=source_uri, chapter=self.chapter,
        )

    def query(self, question: str, mode: str = "hybrid",
              tenant_id: str = "default", top_k: int = 5) -> Dict[str, Any]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")

        graph_share = 0.0
        if mode == "basic":
            results = self.dense.retrieve(question, tenant_id, top_k)
            context = "\n\n".join(r.content for r in results)
        elif mode == "advanced":
            results = self.advanced.retrieve(question, tenant_id, top_k)
            results = self.reranker.rerank(question, results, self.llm)
            context = "\n\n".join(r.content for r in results)
        elif mode == "graphrag":
            graph_context = self.graph.retrieve(question, tenant_id)
            results = triples_to_results(graph_context)
            context = graph_context.render()
            graph_share = 1.0 if results else 0.0
        else:
            fused = self.hybrid.retrieve(question, tenant_id, self.llm, top_k)
            results, context = fused["results"], fused["context"]
            graph_share = fused["graph_share"]

        prompt = (
            "Answer the question using only the context below. Cite the block "
            "number after each sentence.\n\n"
            f"CONTEXT\n{context}\n\nQUESTION\n{question}\n\nANSWER"
        )
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return {"answer": answer, "results": results, "context": context,
                "mode": mode, "graph_share": graph_share}
