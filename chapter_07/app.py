import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import json
from pyvis.network import Network
import tempfile
from chapter_07.kg_builder import build_knowledge_graph, query_graph, get_graph_stats, clear_graph
from shared.ollama_utils import get_llm
from shared.db_utils import get_pg_connection
from shared.neo4j_utils import get_neo4j_driver
from shared.streamlit_utils import (file_uploader_component, setup_page,
                                    tenant_selector)

setup_page("Chapter 7: Knowledge Graph Fundamentals")

st.sidebar.header("Health Checks")
llm = None
try:
    llm = get_llm()
    st.sidebar.success("Ollama: OK")
except Exception as e:
    st.sidebar.error(f"Ollama: Error - {e}")

try:
    if get_neo4j_driver():
        st.sidebar.success("Neo4j: OK")
    else:
        st.sidebar.error("Neo4j: Error")
except Exception as e:
    st.sidebar.error(f"Neo4j: Error - {e}")

import io
import pypdf
from chapter_02.loaders import clean_text

def extract_file_content(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()
    if filename.endswith(".pdf"):
        try:
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            extracted_pages = [p.extract_text() or "" for p in reader.pages]
            return clean_text("\n\n".join(extracted_pages))
        except Exception as err:
            st.sidebar.error(f"PDF Parse Error ({uploaded_file.name}): {err}")
            return ""
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")
        return clean_text(text)

tenant = tenant_selector("ch7")

uploaded_files = st.sidebar.file_uploader("Upload Documents", type=["txt", "md", "pdf"], accept_multiple_files=True)

if st.sidebar.button("Build Knowledge Graph") and uploaded_files:
    with st.spinner("Building Knowledge Graph..."):
        if llm:
            try:
                all_text = ""
                for f in uploaded_files:
                    txt = extract_file_content(f)
                    if txt:
                        all_text += txt + "\n\n"
                
                if not all_text.strip():
                    st.sidebar.error("No valid text found in uploaded document(s).")
                else:
                    source_name = uploaded_files[0].name if uploaded_files else 'upload'
                    res = build_knowledge_graph(
                        all_text, llm, tenant_id=tenant,
                        source_uri=source_name, chapter='ch7'
                    )
                    st.session_state['ch7_last_build'] = res
                    st.sidebar.success(f"Graph built! ({res.get('written', {}).get('nodes', 0)} nodes, {res.get('written', {}).get('relationships', 0)} rels)")
            except Exception as e:
                st.sidebar.error(f"Graph ingestion error: {e}")
        else:
            st.error("LLM not available.")

st.sidebar.subheader("Graph Stats")
stats = get_graph_stats(tenant_id=tenant, chapter='ch7')
st.sidebar.write(f"Nodes: {stats.get('nodes', 0)}")
st.sidebar.write(f"Relationships: {stats.get('relationships', 0)}")
st.sidebar.metric("Relationship to node ratio", f"{stats.get('ratio', 0):.2f}",
                  help="Below 1.0 means the graph is mostly isolated nodes, "
                       "which usually indicates entity resolution failure.")
st.sidebar.write("Entity Types:")
st.sidebar.json(stats.get('entity_types', {}))

if st.sidebar.button("Clear Graph"):
    clear_graph(tenant_id=tenant, chapter='ch7')
    if 'ch7_last_build' in st.session_state:
        del st.session_state['ch7_last_build']
    st.sidebar.success("Graph cleared!")

tab1, tab2, tab3, tab4 = st.tabs(["Graph Visualization", "Entity Explorer", "Cypher Query", "Extraction Log"])

with tab1:
    st.subheader("Interactive Graph")
    try:
        nodes = query_graph("MATCH (n {chapter: 'ch7'}) RETURN coalesce(n.name, n.canonical_name) AS id, coalesce(n.entity_type, labels(n)[0], 'Entity') AS type LIMIT 100")
        rels = query_graph("MATCH (a {chapter: 'ch7'})-[r]->(b {chapter: 'ch7'}) RETURN coalesce(a.name, a.canonical_name) AS source, coalesce(b.name, b.canonical_name) AS target, coalesce(r.rel_type, type(r), 'RELATES') AS type LIMIT 100")
        
        valid_nodes = [n for n in (nodes or []) if n.get('id')]
        valid_rels = [r for r in (rels or []) if r.get('source') and r.get('target')]
        
        if valid_nodes:
            net = Network(height='600px', width='100%', directed=True)
            colors = {"Person": "#ff9999", "Organization": "#99ff99", "Technology": "#9999ff", "Concept": "#ffff99", "Project": "#ffcc99", "Product": "#cc99ff"}
            for n in valid_nodes:
                nid = str(n['id'])
                ntype = str(n.get('type', 'Entity'))
                color = colors.get(ntype, "#cccccc")
                net.add_node(nid, label=nid, title=ntype, color=color)
            for r in valid_rels:
                src = str(r['source'])
                tgt = str(r['target'])
                rtype = str(r.get('type', 'RELATES'))
                net.add_edge(src, tgt, title=rtype, label=rtype)
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, 'r') as f:
                    html_data = f.read()
                import streamlit.components.v1 as components
                components.html(html_data, height=650)
        else:
            st.info("Graph is empty. Upload a file and build the graph.")
    except Exception as e:
        st.error(f"Error rendering graph: {e}")

with tab2:
    st.subheader("Entity Explorer")
    try:
        entities = query_graph("MATCH (n {chapter: 'ch7'}) RETURN coalesce(n.name, n.canonical_name) AS name, coalesce(n.entity_type, labels(n)[0], 'Entity') AS type")
        if entities:
            st.dataframe(entities)
        else:
            st.info("No entities found.")
    except Exception as e:
        st.error(f"Error fetching entities: {e}")

with tab3:
    st.subheader("Cypher Query")
    query = st.text_area("Enter Cypher Query", "MATCH (n {chapter: 'ch7'}) RETURN n LIMIT 10")
    if st.button("Run Query"):
        try:
            results = query_graph(query)
            st.write(results)
        except Exception as e:
            st.error(f"Error running query: {e}")

with tab4:
    st.subheader("Extraction Log")
    if 'ch7_last_build' in st.session_state:
        st.json(st.session_state['ch7_last_build'])
    else:
        st.info("No extraction run yet.")
