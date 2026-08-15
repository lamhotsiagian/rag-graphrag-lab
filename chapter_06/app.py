"""Chapter 6 lab: the evaluation dashboard.

Retrieval and generation are reported separately, because they fail
independently. Coverage counts are shown alongside every metric so a judge
failure reads as missing data rather than as a bad score.

Run with:  streamlit run chapter_06/app.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from chapter_04.rag_complete import RAGPipeline
from chapter_05.retrievers import DenseRetriever
from chapter_06.evaluator import GoldenCase, evaluate_dataset, load_golden_dataset
from shared.db_utils import check_postgres_health
from shared.ollama_utils import get_embeddings, get_llm
from shared.streamlit_utils import (health_badges, render_header, setup_page,
                                    tenant_selector)

DEFAULT_DATASET = Path(__file__).parent / "golden_dataset.json"

setup_page("Chapter 6: RAG Evaluation Dashboard")
render_header(
    "Chapter 6: Evaluation, Observability, and Debugging",
    "Measure retrieval and generation separately, and segment every result.",
)

try:
    llm = get_llm()
    embeddings = get_embeddings()
except Exception as exc:  # pragma: no cover - interface path
    st.error(f"Could not reach Ollama: {exc}")
    st.stop()

health_badges({"Ollama": True, "PostgreSQL": check_postgres_health()})
tenant = tenant_selector("ch6")

st.sidebar.subheader("Dataset")
uploaded = st.sidebar.file_uploader("Golden dataset (JSON)", type=["json"])
k = st.sidebar.slider("K", 1, 20, 5)
chapter = st.sidebar.selectbox("Index to evaluate", ["ch4", "ch5", "ch8"])

if "ch6_report" not in st.session_state:
    st.session_state.ch6_report = None

if st.sidebar.button("Run evaluation", use_container_width=True):
    with st.spinner("Replaying the golden set..."):
        try:
            if uploaded is not None:
                raw = json.load(uploaded)
                rows = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw
                cases = [GoldenCase.from_dict(r, i) for i, r in enumerate(rows)]
            else:
                cases = load_golden_dataset(DEFAULT_DATASET)

            pipeline = RAGPipeline(
                llm, embeddings,
                DenseRetriever(embeddings, chapter=chapter),
                min_score=0.0,
            )
            st.session_state.ch6_report = evaluate_dataset(
                cases, pipeline, llm, tenant_id=tenant, k=k
            )
            st.sidebar.success(f"Evaluated {len(cases)} cases.")
        except Exception as exc:
            st.sidebar.error(f"Evaluation failed: {exc}")

report = st.session_state.ch6_report
tab_overview, tab_cases, tab_segments = st.tabs(
    ["Overview", "Per-case results", "By segment"]
)

with tab_overview:
    if not report:
        st.info("Load a golden dataset and run the evaluation.")
    else:
        overall = report["summary"]["overall"]
        coverage = overall["coverage"]

        st.caption(
            "A metric reported as n/a had no scorable cases. That is missing "
            "data, not a zero, and collapsing the two is how an evaluator bug "
            "becomes three weeks of prompt engineering."
        )
        columns = st.columns(4)
        for column, name in zip(columns, ["recall_at_k", "hit_rate", "mrr", "ndcg_at_k"]):
            value = overall.get(name)
            column.metric(name.replace("_", " ").title(),
                          f"{value:.2f}" if value is not None else "n/a",
                          help=f"scored on {coverage.get(name, 0)} of {overall['n']} cases")

        columns = st.columns(3)
        for column, name in zip(columns, ["faithfulness", "answer_relevance",
                                          "correct_abstention"]):
            value = overall.get(name)
            column.metric(name.replace("_", " ").title(),
                          f"{value:.2f}" if value is not None else "n/a",
                          help=f"scored on {coverage.get(name, 0)} of {overall['n']} cases")

with tab_cases:
    if not report:
        st.info("Run the evaluation to inspect individual cases.")
    else:
        for row in report["rows"]:
            label = f"{row['case_id']} — {row['question'][:70]}"
            with st.expander(label):
                st.write("**Expected**", row.get("expected_answer") or "n/a")
                st.write("**Generated**", row.get("generated_answer"))
                st.json({key: value for key, value in row.items()
                         if key not in ("question", "expected_answer",
                                        "generated_answer")})

with tab_segments:
    if not report:
        st.info("Run the evaluation to see the segmented breakdown.")
    else:
        st.caption(
            "An aggregate gain that hides a segment regression is the most "
            "common way a bad change passes evaluation."
        )
        frame = pd.DataFrame(report["summary"]["by_segment"]).T
        st.dataframe(frame, use_container_width=True)
