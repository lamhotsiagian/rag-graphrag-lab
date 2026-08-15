"""Chapter 3 lab helpers: index and search with tenant scoping.

Book reference: Chapter 3, "Implementing Search With Correct Filtering".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.db_utils import (create_vector_table, delete_chapter_documents,
                             get_document_count, lexical_search, vector_search)

CHAPTER = "ch3"


def create_chapter_table():
    """Ensure the vector table exists."""
    create_vector_table()


def embed_and_store(chunks, metadatas, embeddings_model, tenant_id="default",
                    source_uri=""):
    """Embed chunks and upsert them into pgvector for this tenant."""
    from shared.db_utils import store_embeddings

    try:
        vectors = embeddings_model.embed_documents(chunks)
        store_embeddings(chunks, vectors, metadatas, chapter=CHAPTER,
                         tenant_id=tenant_id, source_uri=source_uri)
        return True
    except Exception as exc:
        print(f"Error storing embeddings: {exc}")
        return False


def semantic_search(query, embeddings_model, tenant_id="default", top_k=5,
                    metadata_filter=None, ef_search=100, min_similarity=0.0):
    """Dense search, scoped to the tenant and tunable at query time."""
    try:
        query_embedding = embeddings_model.embed_query(query)
        rows = vector_search(
            query_embedding,
            tenant_id=tenant_id,
            top_k=top_k,
            chapter=CHAPTER,
            metadata_filter=metadata_filter,
            ef_search=ef_search,
            min_similarity=min_similarity,
        )
        return [
            {
                "content": r.get("content", ""),
                "score": r.get("similarity", 0),
                "metadata": r.get("metadata", {}),
                "source_uri": r.get("source_uri", ""),
                "embed_model": r.get("embed_model", ""),
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"Error in semantic search: {exc}")
        return []


def keyword_search(query, tenant_id="default", top_k=5):
    """Lexical search over the same rows, for the Chapter 5 comparison."""
    try:
        rows = lexical_search(query, tenant_id=tenant_id, top_k=top_k,
                              chapter=CHAPTER)
        return [{"content": r.get("content", ""), "score": r.get("rank", 0),
                 "metadata": r.get("metadata", {})} for r in rows]
    except Exception as exc:
        print(f"Error in keyword search: {exc}")
        return []


def get_stored_count(tenant_id="default"):
    try:
        return get_document_count(chapter=CHAPTER, tenant_id=tenant_id)
    except Exception:
        return 0


def clear_index(tenant_id="default"):
    try:
        delete_chapter_documents(CHAPTER, tenant_id)
        return True
    except Exception:
        return False
