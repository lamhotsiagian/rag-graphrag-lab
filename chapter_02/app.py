import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from chapter_02.loaders import load_document
from chapter_02.chunkers import fixed_size_chunking, recursive_chunking, semantic_chunking, parent_child_chunking, get_chunk_stats
from shared.ollama_utils import get_embeddings
from shared.streamlit_utils import setup_page, render_header

def main():
    setup_page("Chapter 2: Document Ingestion & Chunking")
    render_header("Chapter 2: Chunking Strategies", "Compare different text chunking approaches.")
    
    with st.sidebar:
        st.header("Settings")
        
        try:
            embeddings_model = get_embeddings()
            st.success("Ollama embeddings loaded.")
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
            st.stop()
            
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "md", "pdf", "csv", "html", "docx"])
        
        st.subheader("Parameters")
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=500)
        overlap = st.slider("Overlap", min_value=0, max_value=200, value=50)
        
        if uploaded_file and st.button("Process"):
            with st.spinner("Loading and chunking..."):
                try:
                    text = load_document(uploaded_file)
                    
                    st.session_state['results'] = {
                        "Fixed Size": fixed_size_chunking(text, chunk_size, overlap),
                        "Recursive": recursive_chunking(text, chunk_size, overlap),
                        "Semantic (Mock)": semantic_chunking(text, embeddings_model)
                    }
                    st.success("Processing complete!")
                except Exception as e:
                    st.error(f"Error: {e}")

    if 'results' not in st.session_state:
        st.info("Please upload a document and click Process to see results.")
        return
        
    results = st.session_state['results']
    stats = {k: get_chunk_stats(v) for k, v in results.items()}
    
    tab1, tab2, tab3 = st.tabs(["Compare Strategies", "Chunk Inspector", "Statistics"])
    
    with tab1:
        st.subheader("Strategy Overview")
        for name, stat in stats.items():
            st.markdown(f"**{name}**: {stat['count']} chunks, avg size {stat['avg_size']:.0f} chars")
            
    with tab2:
        st.subheader("Inspect Chunks")
        strategy = st.selectbox("Select Strategy", list(results.keys()))
        chunks = results[strategy]
        
        for i, chunk in enumerate(chunks):
            with st.expander(f"Chunk {i+1} ({len(chunk)} chars)"):
                st.text(chunk)
                
    with tab3:
        st.subheader("Statistics")
        df_stats = pd.DataFrame(stats).T.reset_index().rename(columns={'index': 'Strategy'})
        
        fig1 = px.bar(df_stats, x='Strategy', y='count', title='Number of Chunks per Strategy')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.bar(df_stats, x='Strategy', y='avg_size', title='Average Chunk Size (chars)')
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
