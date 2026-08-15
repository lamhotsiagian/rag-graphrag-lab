import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import time
import tempfile
from pyvis.network import Network
from chapter_08.graph_rag import GraphRAGPipeline
from chapter_07.kg_builder import query_graph
from shared.ollama_utils import get_llm, get_embeddings
from shared.db_utils import get_pg_connection
from shared.neo4j_utils import get_neo4j_driver
from shared.streamlit_utils import setup_page, tenant_selector

setup_page("Chapter 8: GraphRAG & Hybrid Retrieval")

st.sidebar.header("Health Checks")
llm = None
embeddings = None
try:
    llm = get_llm()
    embeddings = get_embeddings()
    st.sidebar.success("Ollama: OK")
except Exception as e:
    st.sidebar.error(f"Ollama: Error - {e}")

try:
    if get_pg_connection():
        st.sidebar.success("PostgreSQL: OK")
    else:
        st.sidebar.error("PostgreSQL: Error")
except Exception as e:
    st.sidebar.error(f"PostgreSQL: Error - {e}")

try:
    if get_neo4j_driver():
        st.sidebar.success("Neo4j: OK")
    else:
        st.sidebar.error("Neo4j: Error")
except Exception as e:
    st.sidebar.error(f"Neo4j: Error - {e}")

tenant = tenant_selector("ch8")

uploaded_files = st.sidebar.file_uploader("Upload Documents", type=["txt", "md", "pdf"], accept_multiple_files=True)

pipeline = None
if llm and embeddings:
    pipeline = GraphRAGPipeline(llm, embeddings, chapter='ch8')

if st.sidebar.button("Ingest Documents") and uploaded_files and pipeline:
    with st.spinner("Ingesting into Vector and Graph databases..."):
        all_text = ""
        for f in uploaded_files:
            all_text += f.getvalue().decode("utf-8", errors="ignore") + "\n"
        pipeline.ingest(all_text, tenant_id=tenant, source_uri='upload')
        st.sidebar.success("Ingestion complete!")

mode = st.sidebar.selectbox("Mode", ["Basic RAG", "Advanced RAG", "GraphRAG", "Hybrid GraphRAG"])
mode_map = {
    "Basic RAG": "basic",
    "Advanced RAG": "advanced",
    "GraphRAG": "graphrag",
    "Hybrid GraphRAG": "hybrid"
}
top_k = st.sidebar.slider("Top K", 1, 10, 5)

tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Compare Modes", "Graph View", "Context Analysis"])

with tab1:
    st.subheader(f"Chat ({mode})")
    if 'ch8_messages' not in st.session_state:
        st.session_state.ch8_messages = []
        
    for msg in st.session_state.ch8_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.ch8_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if pipeline:
            with st.spinner("Thinking..."):
                result = pipeline.query(prompt, mode=mode_map[mode],
                                        tenant_id=tenant, top_k=top_k)
                answer = result['answer']
                st.session_state.ch8_messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)

with tab2:
    st.subheader("Compare Modes")
    compare_q = st.text_input("Enter a query to compare all modes:")
    if st.button("Run Comparison") and compare_q and pipeline:
        cols = st.columns(4)
        for i, (m_name, m_key) in enumerate(mode_map.items()):
            with cols[i]:
                st.markdown(f"**{m_name}**")
                start_t = time.time()
                ans = pipeline.query(compare_q, mode=m_key, tenant_id=tenant,
                                     top_k=top_k)['answer']
                elapsed = time.time() - start_t
                st.write(ans)
                st.caption(f"Time: {elapsed:.2f}s")

with tab3:
    st.subheader("Graph View")
    if st.button("Load Graph for ch8"):
        nodes = query_graph("MATCH (n {chapter: 'ch8'}) RETURN n.id AS id, labels(n)[0] AS type LIMIT 100")
        rels = query_graph("MATCH (a {chapter: 'ch8'})-[r]->(b {chapter: 'ch8'}) RETURN a.id AS source, b.id AS target, type(r) AS type LIMIT 100")
        
        if nodes:
            net = Network(height='600px', width='100%', directed=True)
            for n in nodes:
                net.add_node(n['id'], label=n['id'], title=n['type'])
            for r in rels:
                net.add_edge(r['source'], r['target'], title=r['type'], label=r['type'])
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, 'r') as f:
                    html_data = f.read()
                import streamlit.components.v1 as components
                components.html(html_data, height=650)
        else:
            st.info("Graph is empty.")

with tab4:
    st.subheader("Context Analysis")
    st.info("Shows what context comes from vector vs graph sources. Run a query in Chat or Compare Modes first.")
