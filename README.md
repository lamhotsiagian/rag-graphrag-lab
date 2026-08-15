# rag-graphrag-lab

Runnable companion code for **Cracking RAG and GraphRAG System Design Interviews**
(AI Engineering Career Series).

Every chapter of the book maps to a module and a Streamlit lab here. The code
runs entirely on your own machine: Ollama for both models, PostgreSQL with
pgvector for the vector store, and Neo4j for the graph. No hosted API key and
no cloud account.

> Every class and function printed in the book exists in this repository, and
> `scripts/check_book_parity.py` fails the build if that ever stops being true.

## Quick start

```bash
git clone https://github.com/lamhotsiagian/rag-graphrag-lab.git
cd rag-graphrag-lab

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d                 # PostgreSQL with pgvector, and Neo4j
ollama pull qwen2.5vl:3b
ollama pull nomic-embed-text
cp .env.example .env

streamlit run chapter_01/app.py      # or any other chapter
```

The unit suite needs none of that:

```bash
./scripts/run_tests.sh unit          # 117 tests, no services, under a second
```

## Chapter map

| Chapter | Module | Lab | What it demonstrates |
| --- | --- | --- | --- |
| 1 | `chapter_01/rag_basic.py` | `app.py` | The four stages in memory, abstention enforced in code |
| 2 | `chapter_02/loaders.py`, `chunkers.py` | `app.py` | Safe normalization, four chunking strategies compared |
| 3 | `chapter_03/vector_search.py` | `app.py` | pgvector, HNSW, `ef_search`, metadata filtering, tenancy |
| 4 | `chapter_04/rag_complete.py` | `app.py` | Retriever contract, budgeted context, three reliability gates |
| 5 | `chapter_05/retrievers.py`, `reranker.py` | `app.py` | Rank fusion, reranking, HyDE, multi-query, adaptive routing |
| 6 | `chapter_06/evaluator.py` | `app.py` | Retrieval and generation metrics, segmented reporting |
| 7 | `chapter_07/kg_builder.py` | `app.py` | Constrained extraction, entity resolution, provenance |
| 8 | `chapter_08/graph_rag.py` | `app.py` | Local search, bounded traversal, unified reranking fusion |
| 9 | `chapter_09/agentic_rag.py` | `app.py` | LangGraph state, bounded cycles, verification |
| 10 | `chapter_10/system_design.py` | `app.py` | Circuit breaker, degradation ladder, capacity arithmetic |

Shared across chapters:

| Module | Role |
| --- | --- |
| `shared/retrieval.py` | `RetrievalResult` and the `Retriever` contract every chapter implements |
| `shared/db_utils.py` | pgvector reads and writes, normalization, idempotent upserts |
| `shared/neo4j_utils.py` | Graph driver and Cypher helpers |
| `shared/ollama_utils.py` | Local model and embedding clients |
| `shared/streamlit_utils.py` | Shared interface components, including the tenant selector |

## Multi-tenancy

Every lab that touches a store has a tenant selector in the sidebar. Index a
document under `default`, switch to `acme-corp`, and search again. Retrieval
returns nothing, which is the point: isolation becomes visible rather than
theoretical.

Isolation is enforced in three places, because each fails differently:

1. `scripts/init_pgvector.sql` enables row level security and forces it for the
   table owner, so the database refuses cross-tenant reads.
2. `shared/db_utils.vector_search` sets `app.tenant_id` inside the same
   transaction as the query.
3. `tenant_id` is a required argument on the `Retriever` interface, so omitting
   it is an error rather than a silent leak.

## Testing

```bash
./scripts/run_tests.sh            # unit suite plus the book parity check
./scripts/run_tests.sh unit       # no services required
./scripts/run_tests.sh parity     # assert the book and the lab agree
./scripts/run_tests.sh e2e        # Playwright, needs Docker and Ollama running
```

`tests/unit` uses fakes for the language model, the embeddings, and the stores,
so it runs anywhere in under a second. It asserts on behaviour the book claims,
not just on happy paths: that a reranker parse failure falls back to the
retrieval score rather than to zero, that a repeated agent query is suppressed,
that rank fusion keys on identifiers rather than on text, and that an
unsupported claim is never scored as supported.

`tests/e2e` drives the real Streamlit apps in a browser and requires the full
stack.

## Regenerating the golden dataset

Relevance judgements key on chunk identifiers, and identifiers change whenever
the chunking configuration changes:

```bash
python scripts/rebuild_golden_ids.py --write
```

## Requirements

- Python 3.11 or newer
- Docker and Docker Compose
- Ollama with `qwen2.5vl:3b` and `nomic-embed-text`
- Roughly 8 GB of free RAM for the models and both databases

## Configuration

Copy `.env.example` to `.env` and adjust if your services do not run on the
defaults. `.env` is gitignored; `.env.example` is the template.

The values shipped in `.env.example` are development defaults. Replace them
before any deployment that is reachable by anyone other than you.

---

AI Engineering Insider
[aiengineeringinsider.com](http://aiengineeringinsider.com) ·
[Substack](https://aiengineeringinsider.substack.com/subscribe)
# rag-graphrag-lab
