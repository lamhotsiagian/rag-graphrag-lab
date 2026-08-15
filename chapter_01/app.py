import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from chapter_01.rag_basic import load_document, split_text, build_simple_index, retrieve, generate_answer
from shared.ollama_utils import get_llm, get_embeddings
from shared.streamlit_utils import setup_page, render_header, source_display

def main():
    setup_page("Chapter 1: RAG Foundations")
    render_header("Chapter 1: Basic RAG Implementation", "Upload a document and chat with it.")
    
    with st.sidebar:
        st.header("Setup")
        # Initialize models
        try:
            llm = get_llm()
            embeddings_model = get_embeddings()
            st.success("Ollama models loaded.")
        except Exception as e:
            st.error(f"Error loading Ollama: {e}")
            st.stop()
            
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "md", "pdf"])
        
        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    text = load_document(uploaded_file)
                    chunks = split_text(text)
                    index = build_simple_index(chunks, embeddings_model)
                    st.session_state['rag_index'] = index
                    st.session_state['chat_history'] = []
                    st.success(f"Document processed into {len(chunks)} chunks!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                    
    if 'rag_index' not in st.session_state:
        st.info("Please upload and process a document in the sidebar to start chatting.")
        return
        
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
        
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            if 'sources' in msg and msg['sources']:
                sources_for_display = [
                    {"content": s["content"], "similarity": s.get("score", 0), "metadata": {}}
                    for s in msg['sources']
                ]
                source_display(sources_for_display)
                    
    query = st.chat_input("Ask a question about your document")
    if query:
        st.session_state['chat_history'].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.spinner("Retrieving and generating..."):
            top_chunks = retrieve(query, st.session_state['rag_index'], embeddings_model)
            response = generate_answer(query, top_chunks, llm)
            # Handle LangChain response object
            answer = response.content if hasattr(response, 'content') else str(response)
            
            st.session_state['chat_history'].append({
                "role": "assistant",
                "content": answer,
                "sources": top_chunks
            })
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                if top_chunks:
                    sources_for_display = [
                        {"content": s["content"], "similarity": s.get("score", 0), "metadata": {}}
                        for s in top_chunks
                    ]
                    source_display(sources_for_display)

if __name__ == "__main__":
    main()

