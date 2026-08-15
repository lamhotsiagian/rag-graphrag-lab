"""Chapter 7: knowledge graph construction from unstructured text.

Book reference:
    Chapter 7, "Building a Knowledge Graph From Documents"
    Chapter 7, "Entity Resolution"

Three properties separate a usable extracted graph from an unqueryable one:
a closed schema, entity resolution tuned for precision, and provenance on
every element.
"""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The closed vocabulary
#
# Free-form extraction produces WORKS_AT, WORKS_FOR, EMPLOYED_BY, and
# IS_EMPLOYEE_OF as four distinct types describing one concept. Every query
# then has to enumerate synonyms and every traversal misses edges.
# ---------------------------------------------------------------------------
NODE_LABELS = {
    "Person", "Organization", "Product", "Technology",
    "Location", "Project", "Concept", "Event",
}

RELATIONSHIP_TYPES = {
    "WORKS_AT", "FOUNDED", "ACQUIRED", "PARTNERS_WITH", "SUPPLIES",
    "COMPETES_WITH", "LOCATED_IN", "USES", "PART_OF", "MENTIONS",
    "REPORTS_TO", "SUBCONTRACTS_ON", "SUPERSEDES",
}

# Generic terms that appear in nearly every document. Extracting them creates
# hub nodes, and a hub is what turns a fast traversal into an outage.
STOP_ENTITIES = {
    "data", "system", "systems", "user", "users", "customer", "customers",
    "company", "companies", "team", "teams", "product", "products",
    "service", "services", "process", "information", "technology",
}

EXTRACT_PROMPT = """Extract entities and relationships from the text.

ALLOWED ENTITY TYPES
{node_labels}

ALLOWED RELATIONSHIP TYPES
{relationship_types}

RULES
1. Use ONLY the allowed types. Discard anything that does not fit.
2. Use the entity name exactly as written in the text.
3. source and target must both appear in the entities list.
4. confidence is 0.0 to 1.0 and reflects how explicit the text is.
5. evidence is the shortest verbatim quote supporting the relationship.
6. Output one JSON object and nothing else.

OUTPUT SHAPE
{{"entities": [{{"name": "...", "type": "...", "confidence": 0.0}}],
  "relationships": [{{"source": "...", "target": "...", "type": "...",
                      "confidence": 0.0, "evidence": "..."}}]}}

TEXT
{text}

JSON"""


@dataclass
class ExtractionResult:
    """A schema valid subgraph extracted from one chunk."""

    entities: List[Dict] = field(default_factory=list)
    relationships: List[Dict] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    parse_failed: bool = False

    def merge(self, other: "ExtractionResult") -> "ExtractionResult":
        """Combine two chunk extractions before writing.

        Merging per document before writing lets entities resolve within the
        document first, which is why relationships spanning a chunk boundary
        survive.
        """
        seen = {e["name"] for e in self.entities}
        self.entities.extend(e for e in other.entities if e["name"] not in seen)
        self.relationships.extend(other.relationships)
        self.rejected.extend(other.rejected)
        self.parse_failed = self.parse_failed or other.parse_failed
        return self


def _extract_json_object(raw: str) -> Optional[dict]:
    """Pull the first JSON object out of a possibly chatty model reply."""
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start:end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def extract_graph(text: str, llm, min_confidence: float = 0.5) -> ExtractionResult:
    """Extract a schema valid subgraph from one chunk of text."""
    prompt = EXTRACT_PROMPT.format(
        node_labels="\n".join(sorted(NODE_LABELS)),
        relationship_types="\n".join(sorted(RELATIONSHIP_TYPES)),
        text=text[:4000],
    )
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    payload = _extract_json_object(raw)

    if payload is None:
        logger.warning("kg.parse_failed raw=%r", (raw or "")[:150])
        return ExtractionResult(parse_failed=True)

    result = ExtractionResult()
    valid_names = set()

    for entity in payload.get("entities", []):
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name", "")).strip()
        label = str(entity.get("type", "")).strip()
        try:
            confidence = float(entity.get("confidence", 1.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        if not name or label not in NODE_LABELS:
            result.rejected.append(f"entity:{name}:{label}")
            continue
        if name.lower() in STOP_ENTITIES:
            result.rejected.append(f"stop_entity:{name}")
            continue
        if confidence < min_confidence:
            result.rejected.append(f"low_confidence_entity:{name}")
            continue

        result.entities.append(
            {"name": name, "type": label, "confidence": confidence}
        )
        valid_names.add(name)

    for relationship in payload.get("relationships", []):
        if not isinstance(relationship, dict):
            continue
        source = str(relationship.get("source", "")).strip()
        target = str(relationship.get("target", "")).strip()
        rel_type = (
            str(relationship.get("type", "")).strip().upper().replace(" ", "_")
        )
        try:
            confidence = float(relationship.get("confidence", 1.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        # Dangling edges are the most common extraction defect. A relationship
        # naming an entity the model never extracted would otherwise create a
        # phantom node with no type and no provenance.
        if source not in valid_names or target not in valid_names:
            result.rejected.append(f"dangling:{source}->{target}")
            continue
        if rel_type not in RELATIONSHIP_TYPES or confidence < min_confidence:
            result.rejected.append(f"relationship:{rel_type}")
            continue

        result.relationships.append({
            "source": source, "target": target, "type": rel_type,
            "confidence": confidence,
            "evidence": str(relationship.get("evidence", ""))[:400],
        })

    return result


# ---------------------------------------------------------------------------
# Entity resolution, stage one: deterministic canonicalization
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = re.compile(
    r"\b(inc|llc|ltd|limited|corp|corporation|gmbh|s\.a\.|plc|co)\b\.?",
    flags=re.IGNORECASE,
)


def canonicalize(name: str) -> str:
    """Deterministic canonical key for entity resolution stage one.

    A bad merge is worse than a missed merge. Failing to merge splits a
    subgraph, which degrades recall. Wrongly merging fuses unrelated subgraphs
    and creates false paths that traversal will confidently report as
    evidence, so this cascade is tuned for precision.
    """
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    folded = LEGAL_SUFFIXES.sub("", folded.lower())
    folded = re.sub(r"[^a-z0-9\s]", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def resolve_entities(
    names: List[str],
    alias_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Map every surface form to a canonical key.

    Stage one is normalization, stage two is a curated alias dictionary. The
    alias map is the highest precision signal available and it encodes domain
    knowledge no algorithm infers.
    """
    alias_map = {canonicalize(k): canonicalize(v)
                 for k, v in (alias_map or {}).items()}
    resolved: Dict[str, str] = {}
    for name in names:
        key = canonicalize(name)
        resolved[name] = alias_map.get(key, key)
    return resolved


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------
NODE_CYPHER = """
MERGE (e:Entity {tenant_id: $tenant_id, canonical_name: $canonical})
ON CREATE SET e.name = $name, e.created_at = datetime()
SET e.confidence = CASE
        WHEN e.confidence IS NULL OR $confidence > e.confidence
        THEN $confidence ELSE e.confidence END,
    e.entity_type = $label,
    e.chapter = $chapter,
    e.surface_forms = CASE
        WHEN $name IN coalesce(e.surface_forms, [])
        THEN e.surface_forms
        ELSE coalesce(e.surface_forms, []) + $name END
RETURN e
"""

# A single :RELATES type with the semantic type as a property keeps MERGE
# parameterizable and avoids Cypher string interpolation, which is a genuine
# injection risk when the type comes from a model. Chapter 7 sets out when to
# promote the property to a real relationship type instead.
REL_CYPHER = """
MATCH (a:Entity {tenant_id: $tenant_id, canonical_name: $source})
MATCH (b:Entity {tenant_id: $tenant_id, canonical_name: $target})
MERGE (a)-[r:RELATES {rel_type: $rel_type, source_uri: $source_uri}]->(b)
ON CREATE SET r.created_at = datetime()
SET r.confidence = $confidence,
    r.evidence   = $evidence,
    r.chunk_id   = $chunk_id,
    r.chapter    = $chapter
RETURN r
"""


def write_subgraph(
    result: ExtractionResult,
    tenant_id: str = "default",
    source_uri: str = "",
    chunk_id: str = "",
    driver=None,
    chapter: str = "ch7",
    alias_map: Optional[Dict[str, str]] = None,
) -> Dict[str, int]:
    """Write entities and relationships idempotently, with provenance.

    MERGE on (tenant_id, canonical_name) means reprocessing the same document
    never duplicates nodes. Provenance on every relationship means a document
    deletion can remove exactly the edges it created.
    """
    from shared.neo4j_utils import get_neo4j_driver

    driver = driver or get_neo4j_driver()
    written = {"nodes": 0, "relationships": 0}

    canonical = resolve_entities(
        [e["name"] for e in result.entities], alias_map
    )

    with driver.session() as session:
        for entity in result.entities:
            session.run(NODE_CYPHER, {
                "tenant_id": tenant_id,
                "canonical": canonical[entity["name"]],
                "name": entity["name"],
                "label": entity["type"],
                "confidence": entity["confidence"],
                "chapter": chapter,
            })
            written["nodes"] += 1

        for relationship in result.relationships:
            source = canonical.get(relationship["source"])
            target = canonical.get(relationship["target"])
            if not source or not target:
                continue
            session.run(REL_CYPHER, {
                "tenant_id": tenant_id,
                "source": source,
                "target": target,
                "rel_type": relationship["type"],
                "confidence": relationship["confidence"],
                "evidence": relationship["evidence"],
                "source_uri": source_uri,
                "chunk_id": chunk_id,
                "chapter": chapter,
            })
            written["relationships"] += 1

    return written


def build_knowledge_graph(
    text: str,
    llm,
    tenant_id: str = "default",
    source_uri: str = "",
    chapter: str = "ch7",
    chunk_chars: int = 3000,
    overlap: int = 300,
) -> Dict[str, Any]:
    """Chunk, extract, merge, and write a document's subgraph.

    Chunks overlap so an entity mentioned at a boundary is not split, and
    extractions merge per document before the write so relationships spanning
    a boundary still form.
    """
    combined = ExtractionResult()
    step = max(1, chunk_chars - overlap)
    for start in range(0, max(1, len(text)), step):
        piece = text[start:start + chunk_chars]
        if not piece.strip():
            continue
        combined.merge(extract_graph(piece, llm))

    try:
        written = write_subgraph(
            combined, tenant_id=tenant_id, source_uri=source_uri,
            chunk_id=source_uri, chapter=chapter,
        )
        status = "success"
    except Exception as exc:
        logger.error("kg.write_failed %s", exc)
        written = {"nodes": 0, "relationships": 0}
        status = f"error: {exc}"

    return {
        "entities": combined.entities,
        "relationships": combined.relationships,
        "rejected": combined.rejected,
        "written": written,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Inspection, the queries the walkthrough depends on
# ---------------------------------------------------------------------------
def query_graph(cypher: str, parameters: Optional[Dict[str, Any]] = None):
    from shared.neo4j_utils import run_cypher

    return run_cypher(cypher, parameters or {})


def get_graph_stats(tenant_id: str = "default", chapter: str = "ch7") -> Dict[str, Any]:
    """Node and relationship counts, plus the ratio that reveals fragmentation.

    A relationship to node ratio well below one means the graph is mostly
    isolated nodes, which almost always indicates entity resolution failure
    or dangling edge rejection.
    """
    params = {"tenant_id": tenant_id, "chapter": chapter}
    nodes = query_graph(
        "MATCH (n:Entity {tenant_id: $tenant_id, chapter: $chapter}) "
        "RETURN count(n) AS count", params)
    rels = query_graph(
        "MATCH (:Entity {tenant_id: $tenant_id})-[r:RELATES {chapter: $chapter}]->() "
        "RETURN count(r) AS count", params)
    types = query_graph(
        "MATCH (n:Entity {tenant_id: $tenant_id, chapter: $chapter}) "
        "RETURN n.entity_type AS type, count(n) AS count", params)

    node_count = nodes[0]["count"] if nodes else 0
    rel_count = rels[0]["count"] if rels else 0
    return {
        "nodes": node_count,
        "relationships": rel_count,
        "ratio": (rel_count / node_count) if node_count else 0.0,
        "entity_types": {t["type"]: t["count"] for t in types if t.get("type")},
    }


def get_degree_distribution(tenant_id: str = "default", limit: int = 20):
    """The query to run before trusting any traversal.

    Hubs are what make multi-hop traversal explode, and they form gradually
    during ingestion rather than all at once.
    """
    return query_graph(
        """
        MATCH (e:Entity {tenant_id: $tenant_id})
        RETURN e.name AS entity, count { (e)--() } AS degree
        ORDER BY degree DESC
        LIMIT $limit
        """,
        {"tenant_id": tenant_id, "limit": limit},
    )


def clear_graph(tenant_id: str = "default", chapter: str = "ch7") -> None:
    query_graph(
        "MATCH (n:Entity {tenant_id: $tenant_id, chapter: $chapter}) DETACH DELETE n",
        {"tenant_id": tenant_id, "chapter": chapter},
    )
