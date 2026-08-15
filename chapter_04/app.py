"""Chapter 4 lab: the complete basic RAG system.

Every stage is exposed so a bad answer can be attributed to a specific
component: the condensed query, the retrieved evidence with scores, which gate
fired, the citation report, and per-stage latency.

Run with:  streamlit run chapter_04/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from chapter_04.rag_complete import RAGPipeline
from chapter_05.retrievers import DenseRetriever
from shared.db_utils import check_postgres_health
from shared.ollama_utils import check_ollama_health, get_embeddings, get_llm
from shared.streamlit_utils import (health_badges, render_header, render_results,
                                    render_timings, setup_page, tenant_selector)

CHAPTER = "ch4"

setup_page("Chapter 4: Complete Basic RAG")
render_header(
    "Chapter 4: Complete Basic RAG",
    "Retriever contract, budgeted context, and three reliability gates.",
)

# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------
try:
    llm = get_llm()
    embeddings = get_embeddings()
    ollama_ok = True
except Exception as exc:  # pragma: no cover - interface path
    st.error(f"Could not reach Ollama: {exc}")
    st.stop()

postgres_ok = check_postgres_health()
health_badges({"Ollama": ollama_ok, "PostgreSQL": postgres_ok})
if not postgres_ok:
    st.warning("PostgreSQL is unreachable. Start it with `docker compose up -d`.")

tenant = tenant_selector("ch4")

retriever = DenseRetriever(embeddings, chapter=CHAPTER)
pipeline = RAGPipeline(llm, embeddings, retriever, min_score=0.0)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.subheader("Documents")
uploaded = st.sidebar.file_uploader(
    "Upload documents", accept_multiple_files=True,
    type=["txt", "md", "pdf", "csv", "html", "docx"],
)

if st.sidebar.button("Index documents", use_container_width=True):
    if not uploaded:
        st.sidebar.warning("Upload at least one file first.")
    else:
        with st.spinner("Chunking, embedding, and upserting..."):
            try:
                written = pipeline.ingest_documents(uploaded, tenant, CHAPTER)
                st.sidebar.success(f"Indexed {written} chunks for `{tenant}`.")
            except Exception as exc:
                st.sidebar.error(f"Indexing failed: {exc}")

st.sidebar.subheader("Retrieval")
top_k = st.sidebar.slider("Top-K", 1, 20, 5)
min_score = st.sidebar.slider(
    "Gate 1 score floor", 0.0, 1.0, 0.35, 0.01,
    help="Below this, the system abstains in code before the generator runs.",
)
token_budget = st.sidebar.slider("Context token budget", 400, 6000, 2400, 100)
pipeline.min_score = min_score
pipeline.token_budget = token_budget

st.sidebar.metric("Indexed chunks", pipeline.get_stats(tenant, CHAPTER)["document_count"])

if st.sidebar.button("Clear this tenant's index", use_container_width=True):
    from shared.db_utils import delete_chapter_documents

    try:
        removed = delete_chapter_documents(CHAPTER, tenant)
        st.sidebar.success(f"Removed {removed} chunks.")
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"Clear failed: {exc}")

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
if "ch4_messages" not in st.session_state:
    st.session_state.ch4_messages = []

for message in st.session_state.ch4_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_results(message["sources"])

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.ch4_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.ch4_messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            response = pipeline.query(
                prompt, tenant_id=tenant, top_k=top_k, history=history
            )

        st.markdown(response.answer)

        if response.condensed_query and response.condensed_query != prompt:
            st.caption(f"Condensed query: _{response.condensed_query}_")

        if response.abstained:
            st.warning(
                f"Abstained. Gate: `{response.citation_report.get('reason')}`"
            )
        else:
            coverage = response.citation_report.get("citation_coverage", 0.0)
            regenerations = response.citation_report.get("regenerations", 0)
            st.caption(
                f"Citation coverage {coverage:.0%} "
                f"after {regenerations} regeneration(s)"
            )
            uncited = response.citation_report.get("uncited_sentences") or []
            if uncited:
                st.warning(f"{len(uncited)} sentence(s) carry no citation.")

        render_results(response.sources)
        render_timings(response.timings_ms)

        st.session_state.ch4_messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        })
