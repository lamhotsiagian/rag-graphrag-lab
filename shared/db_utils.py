"""PostgreSQL and pgvector utilities.

Book reference:
    Chapter 3, "Implementing Search With Correct Filtering"
    Chapter 5, "BM25 and Lexical Retrieval"

Three properties of this module matter more than the SQL itself:

1. Vectors are L2 normalized before they are written and before they are
   searched, so cosine similarity, dot product, and Euclidean ranking all
   agree and an entire class of bugs disappears.
2. Every write records the embedding model, and every search can assert it.
   Embedding version skew is the most expensive silent failure in RAG.
3. Filters are applied inside the query, never after it. Post-filtering
   collapses recall on selective predicates.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from shared.config import settings

# Dimension of nomic-embed-text. Kept here so the schema, the writer, and the
# reader cannot drift apart.
EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def get_pg_connection():
    """Get a PostgreSQL connection with pgvector type registration."""
    import psycopg2
    from pgvector.psycopg2 import register_vector

    conn = psycopg2.connect(**settings.postgres_connection_params)
    register_vector(conn)
    return conn


def check_postgres_health() -> bool:
    """Return True when PostgreSQL is reachable."""
    try:
        conn = get_pg_connection()
        conn.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers shared by the write and read paths
# ---------------------------------------------------------------------------
def l2_normalize(vector: Sequence[float]) -> np.ndarray:
    """Scale a vector to unit length.

    Normalizing once at write time is what lets the index operator class and
    the query agree. See Chapter 3, "Normalize Once at Write Time".
    """
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else array


def content_hash(text: str, source_uri: str = "", position: int = 0) -> str:
    """Deterministic identifier for a chunk.

    Ingestion keys on this, which turns an insert into an upsert and makes a
    job that crashes halfway through and restarts produce no duplicates.
    """
    payload = f"{source_uri}|{position}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def create_vector_table(table_name: str = "documents", dimension: int = EMBEDDING_DIM):
    """Create the chunk table and its indexes if they do not exist.

    scripts/init_pgvector.sql is the source of truth and runs on first
    container start. This function exists so a lab can be brought up against
    an already-running database that predates the schema.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id            BIGSERIAL PRIMARY KEY,
                    tenant_id     TEXT        NOT NULL DEFAULT 'default',
                    chapter       VARCHAR(20) NOT NULL DEFAULT 'default',
                    source_uri    TEXT        NOT NULL DEFAULT '',
                    section_path  TEXT        NOT NULL DEFAULT '',
                    content       TEXT        NOT NULL,
                    content_hash  CHAR(64)    NOT NULL,
                    metadata      JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding     vector({dimension}) NOT NULL,
                    embed_model   TEXT        NOT NULL DEFAULT 'nomic-embed-text',
                    is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT {table_name}_tenant_hash_key
                        UNIQUE (tenant_id, content_hash)
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_embedding_hnsw
                    ON {table_name} USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 96)
            """)
            conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------
def store_embeddings(
    contents: List[str],
    embeddings: List[List[float]],
    metadatas: Optional[List[dict]] = None,
    chapter: str = "default",
    tenant_id: str = "default",
    source_uri: str = "",
    section_paths: Optional[List[str]] = None,
    embed_model: Optional[str] = None,
    table_name: str = "documents",
) -> int:
    """Upsert chunks on the content hash.

    Returns the number of rows written or refreshed. Because the write is an
    upsert rather than an insert, reingesting the same document twice leaves
    the chunk count unchanged, which is the acceptance criterion the book
    states for the ingestion stage.
    """
    if len(contents) != len(embeddings):
        raise ValueError("contents and embeddings must be the same length")

    metadatas = metadatas or [{} for _ in contents]
    section_paths = section_paths or ["" for _ in contents]
    embed_model = embed_model or settings.EMBEDDING_MODEL

    conn = get_pg_connection()
    written = 0
    try:
        with conn.cursor() as cur:
            for position, (content, vector, metadata, section) in enumerate(
                zip(contents, embeddings, metadatas, section_paths)
            ):
                cur.execute(
                    f"""
                    INSERT INTO {table_name}
                        (tenant_id, chapter, source_uri, section_path, content,
                         content_hash, metadata, embedding, embed_model)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, content_hash) DO UPDATE
                        SET content    = EXCLUDED.content,
                            metadata   = EXCLUDED.metadata,
                            embedding  = EXCLUDED.embedding,
                            embed_model= EXCLUDED.embed_model,
                            is_active  = TRUE
                    """,
                    (
                        tenant_id,
                        chapter,
                        source_uri or str(metadata.get("source", "")),
                        section,
                        content,
                        content_hash(content, source_uri, position),
                        json.dumps(metadata),
                        l2_normalize(vector),
                        embed_model,
                    ),
                )
                written += 1
            conn.commit()
    finally:
        conn.close()
    return written


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------
def vector_search(
    query_embedding: List[float],
    tenant_id: str = "default",
    top_k: int = 5,
    chapter: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    ef_search: int = 100,
    min_similarity: float = 0.0,
    table_name: str = "documents",
) -> List[Dict[str, Any]]:
    """Cosine similarity search with pre-applied filters and tunable recall.

    ef_search is the single most useful runtime knob in HNSW. Raising it from
    40 to 200 typically lifts recall by 2 to 4 points and costs a few
    milliseconds, so route hard queries to a higher value rather than
    rebuilding the index at a higher m.
    """
    import psycopg2.extras

    vector = l2_normalize(query_embedding)

    predicates = ["is_active", "tenant_id = %(tenant_id)s"]
    params: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "query_vec": vector,
        "top_k": top_k,
        "min_similarity": min_similarity,
    }

    if chapter:
        predicates.append("chapter = %(chapter)s")
        params["chapter"] = chapter

    # Containment on jsonb uses the GIN index, unlike per-key equality.
    if metadata_filter:
        predicates.append("metadata @> %(metadata_filter)s::jsonb")
        params["metadata_filter"] = json.dumps(metadata_filter)

    where_sql = " AND ".join(predicates)
    sql = f"""
        SELECT id, content, metadata, source_uri, section_path, content_hash,
               embed_model,
               1 - (embedding <=> %(query_vec)s) AS similarity
        FROM {table_name}
        WHERE {where_sql}
          AND 1 - (embedding <=> %(query_vec)s) >= %(min_similarity)s
        ORDER BY embedding <=> %(query_vec)s
        LIMIT %(top_k)s
    """

    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Session scoped, so both settings apply to this query only.
            cur.execute("SET LOCAL hnsw.ef_search = %s", (ef_search,))
            cur.execute("SET LOCAL app.tenant_id = %s", (tenant_id,))
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def lexical_search(
    query: str,
    tenant_id: str = "default",
    top_k: int = 5,
    chapter: Optional[str] = None,
    table_name: str = "documents",
) -> List[Dict[str, Any]]:
    """Rank by term statistics using the materialized tsvector column.

    Chapter 5 argues the lexical branch belongs in the database rather than in
    an in-process BM25 object: it stays consistent with the vectors, it needs
    no rebuild on restart, and it reflects writes immediately.
    """
    import psycopg2.extras

    predicates = ["is_active", "tenant_id = %(tenant_id)s",
                  "content_tsv @@ plainto_tsquery('english', %(query)s)"]
    params: Dict[str, Any] = {"tenant_id": tenant_id, "query": query, "top_k": top_k}

    if chapter:
        predicates.append("chapter = %(chapter)s")
        params["chapter"] = chapter

    sql = f"""
        SELECT id, content, metadata, source_uri, section_path, content_hash,
               ts_rank_cd(content_tsv,
                          plainto_tsquery('english', %(query)s)) AS rank
        FROM {table_name}
        WHERE {" AND ".join(predicates)}
        ORDER BY rank DESC
        LIMIT %(top_k)s
    """

    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL app.tenant_id = %s", (tenant_id,))
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Maintenance and assertions
# ---------------------------------------------------------------------------
def assert_embedding_model(
    tenant_id: str = "default",
    chapter: Optional[str] = None,
    expected: Optional[str] = None,
    table_name: str = "documents",
) -> None:
    """Refuse to serve when the index holds vectors from another model.

    Chapter 3 calls embedding version skew the most expensive and most silent
    failure in this layer. This is the startup assertion that catches it.
    """
    expected = expected or settings.EMBEDDING_MODEL
    predicates = ["tenant_id = %s"]
    params: List[Any] = [tenant_id]
    if chapter:
        predicates.append("chapter = %s")
        params.append(chapter)

    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT embed_model FROM {table_name} "
                f"WHERE {' AND '.join(predicates)}",
                params,
            )
            present = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    unexpected = present - {expected}
    if unexpected:
        raise RuntimeError(
            f"Embedding model mismatch. Index holds {sorted(unexpected)} but the "
            f"query encoder is {expected!r}. Reindex before serving, see Chapter 3."
        )


def delete_chapter_documents(chapter: str, tenant_id: str = "default",
                             table_name: str = "documents") -> int:
    """Soft delete every chunk for a chapter, then hard delete it.

    Chapter 2 argues deletion deserves higher priority than insertion, because
    a stale chunk from a deleted document creates compliance exposure.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {table_name} SET is_active = FALSE "
                f"WHERE chapter = %s AND tenant_id = %s",
                (chapter, tenant_id),
            )
            cur.execute(
                f"DELETE FROM {table_name} WHERE chapter = %s AND tenant_id = %s",
                (chapter, tenant_id),
            )
            removed = cur.rowcount
            conn.commit()
            return removed
    finally:
        conn.close()


def get_document_count(chapter: Optional[str] = None, tenant_id: str = "default",
                       table_name: str = "documents") -> int:
    """Count active chunks, optionally scoped to a chapter."""
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            if chapter:
                cur.execute(
                    f"SELECT COUNT(*) FROM {table_name} "
                    f"WHERE chapter = %s AND tenant_id = %s AND is_active",
                    (chapter, tenant_id),
                )
            else:
                cur.execute(
                    f"SELECT COUNT(*) FROM {table_name} "
                    f"WHERE tenant_id = %s AND is_active",
                    (tenant_id,),
                )
            return int(cur.fetchone()[0])
    finally:
        conn.close()
