"""Streamlit UI for the Clinical Guidelines RAG bot."""
import streamlit as st

from src.rag import load_query_engine


st.set_page_config(
    page_title="Clinical Guidelines Q&A",
    page_icon="🏥",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading query engine (first run takes ~30s)…")
def get_engine():
    """Load the LlamaIndex query engine once per Streamlit session."""
    return load_query_engine()


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown(
        """
This Q&A bot is grounded in **4 AACAP Practice Parameters** covering:

- Major Depressive Disorder
- Anxiety Disorders
- Bipolar Disorder
- Autism Spectrum Disorder

Every answer cites the exact PDF page it came from. The system is
instructed to **refuse** questions outside this corpus rather than
extrapolate from adjacent material.
        """
    )
    st.divider()
    st.caption("Stack: LlamaIndex · ChromaDB · BGE-M3 · Llama-3.3-70B (Groq)")
    st.caption(
        "[GitHub repo](https://github.com/MaahirGit28/clinical-guidelines-rag)"
    )


# ── Main ──────────────────────────────────────────────────────────────
st.title("🏥 Clinical Guidelines Q&A")
st.caption(
    "Grounded answers from AACAP Practice Parameters with page-level citations."
)


# Session state init
if "query" not in st.session_state:
    st.session_state.query = ""


# Example questions — one per topic in the corpus
EXAMPLES = [
    "What are first-line pharmacotherapy options for adolescent MDD?",
    "How should clinicians screen for pediatric anxiety disorders?",
    "What is the recommended workup for suspected pediatric bipolar disorder?",
    "What evidence-based interventions exist for autism spectrum disorder?",
]


st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for i, q in enumerate(EXAMPLES):
    with cols[i]:
        if st.button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state.query = q


query = st.text_area(
    "Your question",
    key="query",
    height=80,
    placeholder=(
        "e.g. What is the recommended initial dose of fluoxetine "
        "for adolescent MDD?"
    ),
)


ask = st.button("Ask", type="primary", disabled=not query.strip())


if ask and query.strip():
    engine = get_engine()

    with st.spinner("Searching guidelines…"):
        try:
            response = engine.query(query)
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.stop()

    st.markdown("### Answer")
    st.markdown(response.response)

    n_sources = len(response.source_nodes)
    with st.expander(
        f"📄 Sources — {n_sources} chunks retrieved", expanded=False
    ):
        for i, node in enumerate(response.source_nodes, 1):
            meta = node.metadata or {}
            file_name = meta.get("file_name", "unknown")
            page_label = meta.get("page_label", "?")
            score = node.score if node.score is not None else 0.0

            st.markdown(
                f"**{i}. `{file_name}` — page {page_label}**  "
                f"· similarity `{score:.3f}`"
            )
            text = node.text or ""
            preview = text[:600] + ("…" if len(text) > 600 else "")
            st.text(preview)
            if i < n_sources:
                st.divider()
