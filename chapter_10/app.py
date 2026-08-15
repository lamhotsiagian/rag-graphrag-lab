import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import shared.streamlit_utils as st_utils
from chapter_10.system_design import get_architecture_diagram, get_interview_scenarios

st.set_page_config(page_title="Chapter 10: Production System Design", layout="wide")

st.title("Chapter 10: Production System Design")

with st.sidebar:
    st.header("Navigation")
    section = st.radio("Go to", ["Architecture", "Scalability", "Reliability", "Security", "Cost Optimization", "Interview Practice"])

if section == "Architecture":
    st.subheader("System Architecture")
    st.markdown(get_architecture_diagram())

elif section == "Scalability":
    st.subheader("Scalability Patterns")
    st.checkbox("Load Balancing LLM endpoints")
    st.checkbox("Asynchronous processing for ingestion")
    st.checkbox("Read Replicas for Vector/Graph DBs")

elif section == "Reliability":
    st.subheader("Reliability Patterns")
    st.button("Test Circuit Breaker")
    st.button("Test Retry with Backoff")

elif section == "Security":
    st.subheader("Security Patterns")
    st.checkbox("Role-Based Access Control (RBAC) in Neo4j")
    st.checkbox("Row-Level Security in PostgreSQL")
    st.checkbox("API Key Authentication")

elif section == "Cost Optimization":
    st.subheader("Cost Optimization Patterns")
    st.write("Semantic caching and Token Budgeting examples.")

elif section == "Interview Practice":
    st.subheader("System Design Interview Practice")
    scenarios = get_interview_scenarios()
    selected_title = st.selectbox("Select Scenario", [s["title"] for s in scenarios])
    
    scenario = next(s for s in scenarios if s["title"] == selected_title)
    st.write(f"**Requirements.** {scenario['requirements']}")
    with st.expander("Reference architecture", expanded=False):
        st.write(scenario["architecture"])
    st.info(f"**Follow-up the interviewer will ask.** {scenario['follow_up']}")
    st.caption("Hints: " + ", ".join(scenario["hints"]))
    
    answer = st.text_area("Your Design Answer", height=200)
    if st.button("Evaluate"):
        st.success("Evaluation placeholder. Good design consider caching and load balancing.")
