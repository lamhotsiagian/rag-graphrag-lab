-- ===========================================================================
--  init_pgvector.sql
--  Production shaped schema for the RAG and GraphRAG lab.
--
--  Book reference:
--    Chapter 3, "Designing the pgvector Schema"
--    Chapter 5, "BM25 and Lexical Retrieval"
--
--  Runs automatically on first container start via docker-compose.
-- ===========================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fuzzy identifier matching, Chapter 5

-- ---------------------------------------------------------------------------
-- Chunk store
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      TEXT        NOT NULL DEFAULT 'default',
    chapter        VARCHAR(20) NOT NULL DEFAULT 'default',
    source_uri     TEXT        NOT NULL DEFAULT '',
    section_path   TEXT        NOT NULL DEFAULT '',
    content        TEXT        NOT NULL,
    content_hash   CHAR(64)    NOT NULL,
    metadata       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    embedding      vector(768) NOT NULL,
    -- Guards against the silent outage in Chapter 3: document vectors written
    -- by one model and query vectors produced by another still return numbers,
    -- they are just meaningless. Storing the model makes the mismatch checkable.
    embed_model    TEXT        NOT NULL DEFAULT 'nomic-embed-text',
    is_active      BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Idempotent ingestion: a rerun updates instead of duplicating.
    CONSTRAINT documents_tenant_hash_key UNIQUE (tenant_id, content_hash)
);

-- ---------------------------------------------------------------------------
-- Lexical column, Chapter 5. Materialized so it is indexed, not recomputed
-- on every query.
-- ---------------------------------------------------------------------------
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- Approximate nearest neighbour over cosine distance.
-- m controls graph degree and memory, ef_construction controls build quality.
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 96);

-- Selective B-tree indexes for the filters that appear in every query.
CREATE INDEX IF NOT EXISTS documents_tenant_active
    ON documents (tenant_id, chapter) WHERE is_active;
CREATE INDEX IF NOT EXISTS documents_source
    ON documents (tenant_id, source_uri);

-- Containment queries on metadata use jsonb_path_ops, which is smaller and
-- faster than the default operator class for the @> operator.
CREATE INDEX IF NOT EXISTS documents_metadata_gin
    ON documents USING gin (metadata jsonb_path_ops);

-- Lexical retrieval, Chapter 5.
CREATE INDEX IF NOT EXISTS documents_content_tsv
    ON documents USING gin (content_tsv);

-- Trigram index for fuzzy identifier matching, which catches typos in codes
-- that exact lexical search would miss entirely.
CREATE INDEX IF NOT EXISTS documents_content_trgm
    ON documents USING gin (content gin_trgm_ops);

-- ---------------------------------------------------------------------------
-- Tenant isolation enforced by the database, not by the application.
--
-- Chapter 3 and Chapter 10 both make the same argument: a single missing
-- predicate in one code path leaks another customer's documents. The policy
-- reads a session variable that the retrieval layer sets inside the same
-- transaction as the query.
-- ---------------------------------------------------------------------------
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON documents;
CREATE POLICY tenant_isolation ON documents
    USING (
        tenant_id = current_setting('app.tenant_id', TRUE)
        -- The lab runs single tenant by default. When app.tenant_id is unset,
        -- the policy allows the row so the walkthroughs work out of the box.
        -- Remove this clause for a real multi-tenant deployment.
        OR current_setting('app.tenant_id', TRUE) IS NULL
        OR current_setting('app.tenant_id', TRUE) = ''
    );

-- The table owner bypasses RLS by default, which would silently defeat the
-- policy during the lab. Forcing it makes the isolation walkthrough real.
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
