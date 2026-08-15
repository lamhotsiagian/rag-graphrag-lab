"""Common Streamlit UI components and styling."""

import streamlit as st


def setup_page(title: str, icon: str = "🔍", layout: str = "wide"):
    """Configure Streamlit page with consistent styling."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )

    # Custom CSS for premium look
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }

    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .status-ok {
        background: #d4edda;
        color: #155724;
    }

    .status-error {
        background: #f8d7da;
        color: #721c24;
    }

    .chunk-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    div[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }

    .stChatMessage {
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str):
    """Render a styled page header."""
    st.markdown(f'<h1 class="main-header">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{subtitle}</p>', unsafe_allow_html=True)


def render_status_badge(label: str, is_ok: bool):
    """Render a status badge."""
    status_class = "status-ok" if is_ok else "status-error"
    icon = "✅" if is_ok else "❌"
    st.markdown(
        f'<span class="status-badge {status_class}">{icon} {label}</span>',
        unsafe_allow_html=True,
    )


def file_uploader_component(
    label: str = "Upload Documents",
    accepted_types: list[str] = None,
    key: str = "file_uploader",
):
    """Render a styled file uploader."""
    if accepted_types is None:
        accepted_types = ["pdf", "txt", "md", "docx", "csv", "html"]

    return st.file_uploader(
        label,
        type=accepted_types,
        accept_multiple_files=True,
        key=key,
        help="Upload one or more documents for processing",
    )


def chat_message(role: str, content: str, sources: list[dict] = None):
    """Display a chat message with optional source attribution."""
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                for i, source in enumerate(sources):
                    similarity = source.get("similarity", 0)
                    stars = "⭐" * min(5, max(1, int(similarity * 5)))
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>Source {i + 1}</strong> {stars} '
                        f'(similarity: {similarity:.3f})<br>'
                        f'{source.get("content", "")[:300]}...'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


def source_display(sources: list[dict]):
    """Display retrieved sources with scores."""
    if not sources:
        st.info("No sources retrieved.")
        return

    for i, source in enumerate(sources):
        similarity = source.get("similarity", 0)
        metadata = source.get("metadata", {})
        content = source.get("content", "")

        with st.expander(
            f"📄 Source {i + 1} — Score: {similarity:.4f}",
            expanded=i == 0,
        ):
            if metadata:
                cols = st.columns(len(metadata))
                for j, (key, value) in enumerate(metadata.items()):
                    cols[j % len(cols)].caption(f"**{key}**: {value}")

            st.markdown(
                f'<div class="chunk-box">{content}</div>',
                unsafe_allow_html=True,
            )


def render_metric_card(label: str, value: str, delta: str = None):
    """Render a styled metric card."""
    delta_html = f'<br><small style="color: #22c55e;">{delta}</small>' if delta else ""
    st.markdown(
        f'<div class="metric-card">'
        f'<div style="font-size: 0.85rem; color: #6b7280;">{label}</div>'
        f'<div style="font-size: 1.8rem; font-weight: 700; color: #1f2937;">{value}</div>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tenant selection
#
# Chapter 3 onward treats tenant_id as non-negotiable, so every lab that
# touches a store exposes it. Seeing retrieval return nothing after switching
# tenant is the point: it makes isolation visible rather than theoretical.
# ---------------------------------------------------------------------------
DEFAULT_TENANTS = ["default", "acme-corp", "meridian-ltd"]


def tenant_selector(key: str = "tenant", tenants=None) -> str:
    """Render the tenant picker and return the selected tenant id."""
    import streamlit as st

    tenants = tenants or DEFAULT_TENANTS
    st.sidebar.subheader("Tenant")
    tenant = st.sidebar.selectbox(
        "Active tenant", tenants, key=f"{key}_select",
        help=("Every read and write is scoped to this tenant. Switch it after "
              "indexing to watch isolation take effect."),
    )
    st.sidebar.caption(f"Queries run as `{tenant}`")
    return tenant


def health_badges(checks: dict) -> None:
    """Render a row of service health badges from {label: bool}."""
    import streamlit as st

    st.sidebar.subheader("Health")
    for label, ok in checks.items():
        render_status_badge(label, bool(ok))


def render_timings(timings: dict) -> None:
    """Show per-stage latency, which is what makes a p95 regression diagnosable."""
    import streamlit as st

    if not timings:
        return
    columns = st.columns(len(timings))
    for column, (stage, ms) in zip(columns, timings.items()):
        column.metric(stage.title(), f"{ms:.0f} ms")


def render_results(results, title: str = "Retrieved evidence") -> None:
    """Display RetrievalResult objects with scores and provenance."""
    import streamlit as st

    if not results:
        st.info("No evidence retrieved.")
        return

    with st.expander(f"{title} ({len(results)})", expanded=False):
        for position, result in enumerate(results, start=1):
            origin = result.metadata.get("origin", "vector")
            st.markdown(
                f"**[{position}]** score `{result.score:.3f}` "
                f"origin `{origin}` retriever `{result.retriever}`  \n"
                f"source `{result.source_uri or 'n/a'}`"
            )
            st.markdown(
                f'<div class="chunk-box">{result.content[:600]}</div>',
                unsafe_allow_html=True,
            )
