import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import shared.streamlit_utils as st_utils
from chapter_09.agentic_rag import build_agent_graph, run_agent

st.set_page_config(page_title="Chapter 9: Agentic GraphRAG", layout="wide")

st.title("Chapter 9: Agentic GraphRAG")

with st.sidebar:
    st.header("Settings")
    max_iter = st.slider("Max iterations", 1, 5, 3)
    show_reasoning = st.toggle("Show reasoning", True)
    
    st.header("Ingest")
    uploaded_file = st.file_uploader("Upload document", type=["txt", "pdf"])
    if st.button("Ingest"):
        st.success("Ingestion successful")

tab1, tab2, tab3 = st.tabs(["Agent Chat", "Workflow Visualization", "Agent Decisions"])

with tab1:
    st.subheader("Chat")
    
    if "messages_ch9" not in st.session_state:
        st.session_state.messages_ch9 = []
        
    for msg in st.session_state.messages_ch9:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if show_reasoning and "reasoning" in msg:
                with st.expander("Reasoning Steps"):
                    for step in msg["reasoning"]:
                        st.write(f"- {step}")
            if "confidence" in msg:
                st.progress(msg["confidence"])
                
    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages_ch9.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    graph = build_agent_graph(llm=None, retrievers={})
                    result = run_agent(graph, prompt, thread_id="ch9",
                                       max_iterations=max_iter)
                    
                    answer = result.get("answer", "No answer generated.")
                    confidence = result.get("confidence", 0.0)
                    reasoning = result.get("reasoning", [])
                    
                    st.markdown(answer)
                    if show_reasoning:
                        with st.expander("Reasoning Steps"):
                            for step in reasoning:
                                st.write(f"- {step}")
                    st.progress(min(1.0, max(0.0, confidence)))
                    st.caption(f"Terminated by: `{result.get('terminated_by', 'unknown')}`")
                    
                    st.session_state.messages_ch9.append({
                        "role": "assistant",
                        "content": answer,
                        "confidence": confidence,
                        "reasoning": reasoning
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    st.subheader("Workflow Visualization")
    st.markdown("```mermaid\ngraph TD\nA[classify_query] --> B[rewrite_query]\nB --> C[extract_entities]\nC --> D{route_query}\nD -->|vector| E[vector_retrieve]\nD -->|graph| F[graph_retrieve]\nD -->|hybrid| G[hybrid_retrieve]\nE --> H[evaluate_context]\nF --> H\nG --> H\nH --> I{should_continue}\nI -->|continue| J[plan_next_hop]\nJ --> C\nI -->|generate| K[generate_answer]\nK --> L[verify_answer]\nL --> M[END]\n```")

with tab3:
    st.subheader("Agent Decisions")
    st.write("Decision logs will appear here based on the chat history.")
