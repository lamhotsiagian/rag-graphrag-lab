"""Chapter 5 lab: compare retrieval strategies on identical inputs.

The point of this interface is to make the complementarity argument concrete.
Search for an exact identifier with Dense and watch it fail. Repeat with
Lexical and watch it succeed while a conceptual query fails. Then run Hybrid
and watch both work.

Run with:  streamlit run chapter_05/app.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from chapter_02.chunkers import recursive_chunking
from chapter_02.loaders import load_document
from chapter_05.reranker import LLMReranker
from chapter_05.retrievers import (AdaptiveRetriever, DenseRetriever,
                                   HybridRetriever, HyDERetriever,
                                   LexicalRetriever, MultiQueryRetriever)
from shared.db_utils import (check_postgres_health, create_vector_table,
                             store_embeddings)
from shared.ollama_utils import get_embeddings, get_llm
from shared.streamlit_utils import (health_badges, render_header, render_results,
                                    setup_page, tenant_selector)

CHAPTER = "ch5"

setup_page("Chapter 5: Advanced Retrieval and Hybrid RAG")
render_header(
    "Chapter 5: Advanced Retrieval and Hybrid RAG",
    "Dense, lexical, hybrid fusion, reranking, multi-query, HyDE, and adaptive routing.",
)

try:
    llm = get_llm()
    embeddings = get_embeddings()
except Exception as exc:  # pragma: no cover - interface path
    st.error(f"Could not reach Ollama: {exc}")
    st.stop()

postgres_ok = check_postgres_health()
health_badges({"Ollama": True, "PostgreSQL": postgres_ok})
tenant = tenant_selector("ch5")

# ---------------------------------------------------------------------------
# Retrievers, all sharing one contract
# ---------------------------------------------------------------------------
dense = DenseRetriever(embeddings, chapter=CHAPTER)
lexical = LexicalRetriever(chapter=CHAPTER)
reranker = LLMReranker(keep=5, score_floor=0.0)

st.sidebar.subheader("Indexing")
uploaded = st.sidebar.file_uploader(
    "Upload a document", type=["txt", "md", "pdf", "csv", "html", "docx"]
)
if st.sidebar.button("Index", use_container_width=True):
    if not uploaded:
        st.sidebar.warning("Upload a file first.")
    else:
        with st.spinner("Indexing..."):
            try:
                text, metadata = load_document(uploaded, with_metadata=True)
                chunks = recursive_chunking(text, size=600, overlap=72)
                vectors = embeddings.embed_documents(chunks)
                create_vector_table()
                written = store_embeddings(
                    chunks, vectors,
                    metadatas=[{**metadata, "chunk": i} for i in range(len(chunks))],
                    chapter=CHAPTER, tenant_id=tenant,
                    source_uri=uploaded.name,
                )
                st.sidebar.success(f"Successfully indexed {written} chunks.")
            except Exception as exc:
                st.sidebar.error(f"Error indexing: {exc}")

st.sidebar.subheader("Strategy")
strategy = st.sidebar.selectbox(
    "Strategy",
    ["Dense", "Lexical", "Hybrid", "Multi-Query", "HyDE", "Adaptive"],
)
top_k = st.sidebar.slider("Top K", 1, 20, 5)
dense_weight = st.sidebar.slider("Hybrid dense weight", 0.0, 1.0, 0.6, 0.05)
use_reranker = st.sidebar.checkbox("Rerank with the LLM")


def build(name: str):
    """Construct the selected strategy from the shared components."""
    hybrid = HybridRetriever(dense, lexical, dense_weight=dense_weight)
    return {
        "Dense": dense,
        "Lexical": lexical,
        "Hybrid": hybrid,
        "Multi-Query": MultiQueryRetriever(hybrid, llm),
        "HyDE": HyDERetriever(hybrid, llm),
        "Adaptive": AdaptiveRetriever(hybrid, reranker, llm),
    }[name]


def run(name: str, query: str):
    started = time.perf_counter()
    results = build(name).retrieve(query, tenant, top_k)
    if use_reranker and name != "Adaptive" and results:
        results = reranker.rerank(query, results, llm)
    return results, (time.perf_counter() - started) * 1000


tab_chat, tab_compare, tab_metrics = st.tabs(["Chat", "Compare", "Metrics"])

with tab_chat:
    query = st.chat_input("Ask a question")
    if query:
        st.chat_message("user").write(query)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving..."):
                try:
                    results, elapsed = run(strategy, query)
                except Exception as exc:
                    st.error(f"Retrieval failed: {exc}")
                    results, elapsed = [], 0.0

            st.write(
                f"Retrieved {len(results)} sources in {elapsed:.0f} ms "
                f"using strategy: {strategy}"
            )
            if strategy == "Adaptive":
                outcome = build("Adaptive").retrieve_adaptive(query, tenant)
                st.caption(
                    f"Classified as `{outcome.kind.value}`, "
                    f"{outcome.attempts} attempt(s)"
                )
                for correction in outcome.corrections:
                    st.warning(correction)
            render_results(results)

with tab_compare:
    st.subheader("Side by side")
    st.caption(
        "Try an exact identifier such as an error code, then a paraphrased "
        "conceptual question. The strategies should invert."
    )
    compare_query = st.text_input("Query for comparison")
    if st.button("Compare all strategies") and compare_query:
        rows = []
        columns = st.columns(3)
        for index, name in enumerate(["Dense", "Lexical", "Hybrid"]):
            with columns[index]:
                st.markdown(f"**{name}**")
                try:
                    results, elapsed = run(name, compare_query)
                except Exception as exc:
                    st.error(str(exc))
                    continue
                rows.append({"Strategy": name, "Results": len(results),
                             "Top score": round(results[0].score, 3) if results else 0,
                             "Latency (ms)": round(elapsed)})
                for result in results[:3]:
                    st.caption(f"`{result.score:.3f}` {result.content[:120]}")
        if rows:
            st.session_state.ch5_metrics = rows

with tab_metrics:
    st.subheader("Measured latency and yield")
    rows = st.session_state.get("ch5_metrics")
    if not rows:
        st.info("Run a comparison to populate this table with measured numbers.")
    else:
        frame = pd.DataFrame(rows)
        st.dataframe(frame, use_container_width=True)
        st.bar_chart(frame.set_index("Strategy")["Latency (ms)"])
