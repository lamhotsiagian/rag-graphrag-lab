import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from shared.ollama_utils import get_embeddings
from chapter_03.vector_search import create_chapter_table, embed_and_store, semantic_search, get_stored_count, clear_index

st.set_page_config(page_title="Chapter 3: Embeddings & Vector Database", layout="wide")

st.title("Chapter 3: Embeddings & Vector Database")

st.sidebar.header("Settings")
st.sidebar.subheader("DB Health Status")
try:
    count = get_stored_count()
    st.sidebar.success("PostgreSQL Connected")
except Exception:
    st.sidebar.error("PostgreSQL Disconnected")

st.sidebar.subheader("Configuration")
top_k = st.sidebar.slider("Top-K", min_value=1, max_value=20, value=5)
metric = st.sidebar.selectbox("Similarity Metric", ["cosine", "dot product"])

st.sidebar.subheader("Upload Documents")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if st.sidebar.button("Index Documents"):
    if uploaded_file:
        with st.spinner("Indexing..."):
            create_chapter_table()
            content = uploaded_file.read().decode("utf-8")
            chunks = [content[i:i+500] for i in range(0, len(content), 500)]
            metadatas = [{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]
            embeddings_model = get_embeddings()
            success = embed_and_store(chunks, metadatas, embeddings_model)
            if success:
                st.sidebar.success("Documents indexed successfully!")
            else:
                st.sidebar.error("Failed to index documents.")
    else:
        st.sidebar.warning("Please upload a file first.")

st.sidebar.subheader("Index Stats")
st.sidebar.write(f"Documents: {get_stored_count()}")
if st.sidebar.button("Clear Index"):
    clear_index()
    st.sidebar.success("Index cleared!")
    st.rerun()

st.header("Semantic Search")
query = st.text_input("Enter your query:")
if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            embeddings_model = get_embeddings()
            results = semantic_search(query, embeddings_model, top_k=top_k)
            for res in results:
                st.markdown(f"**Score:** {res.get('score', 'N/A')}")
                st.info(res.get('content', ''))
                st.caption(f"Metadata: {res.get('metadata', {})}")
