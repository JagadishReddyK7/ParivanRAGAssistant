"""
app.py — Parivahan RAG Chatbot
Run: streamlit run app.py  (from project root)
"""
import sys, os
from pathlib import Path

# Ensure project root is on path so `src` package resolves correctly
_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parivahan RAG Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0f1117; }

.stChatMessage { border-radius: 12px; margin-bottom: 8px; }

.citation-box {
    background: #1e2130;
    border-left: 3px solid #f59e0b;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.82em;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 8px;
    color: #94a3b8;
}

.header-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75em;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.sidebar-section {
    background: #1e2130;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Index initialisation (cached) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def init_rag():
    """Load or build the FAISS index and embedding model."""
    import json
    from src.data_pipeline import build_chunks
    from src.embeddings import (
        load_model, load_index, embed_chunks, build_index, save_index
    )

    INDEX_FILE  = Path("data/faiss.index")
    META_FILE   = Path("data/metadata.pkl")
    BM25_FILE   = Path("data/bm25.pkl")
    CHUNKS_FILE = Path("data/chunks.json")

    model = load_model()

    if INDEX_FILE.exists() and META_FILE.exists() and BM25_FILE.exists():
        index, bm25, meta = load_index()
    else:
        # Try to read existing chunks; rebuild if missing or corrupt
        chunks = None
        if CHUNKS_FILE.exists():
            try:
                raw = CHUNKS_FILE.read_text(encoding="utf-8").strip()
                if raw:
                    chunks = json.loads(raw)
            except (json.JSONDecodeError, Exception):
                chunks = None  # corrupt file — rebuild

        if not chunks:
            chunks = build_chunks()

        embeddings = embed_chunks(model, chunks)
        index, bm25 = build_index(embeddings, chunks)
        save_index(index, bm25, chunks)
        meta = chunks

    return model, index, bm25, meta



# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 18px'>
        <span style='font-size:2.4em'>🚗</span><br>
        <span style='font-size:1.3em; font-weight:700; color:#f59e0b'>Parivahan</span><br>
        <span style='font-size:0.85em; color:#94a3b8'>RAG Assistant</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Settings")
    hindi_mode  = st.toggle("🇮🇳 Respond in Hindi", value=False)
    top_k       = st.slider("Context chunks (top-K)", 3, 8, 5)
    show_chunks = st.toggle("Show raw retrieved chunks", value=False)

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_qs = [
        "How do I apply for a Driving Licence?",
        "Documents needed for RC registration?",
        "How to transfer vehicle ownership?",
        "What is the fee for a learner's licence?",
        "How to apply for IDP (International Driving Permit)?",
    ]
    for q in sample_qs:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75em; color:#64748b; text-align:center'>
    Data: <a href='https://parivahan.gov.in' style='color:#f59e0b'>parivahan.gov.in</a><br>
    LLM: Gemini 1.5 Flash / Groq Llama3<br>
    Embeddings: all-MiniLM-L6-v2
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#f59e0b; margin-bottom:2px'>Parivahan RAG Assistant</h1>
<p style='color:#94a3b8; margin-top:0'>Ask anything about Driving Licence, RC, or vehicle services in India</p>
""", unsafe_allow_html=True)

# Load RAG components
try:
    model, index, bm25, meta = init_rag()
    rag_ready = True
except Exception as e:
    st.error(f"Failed to initialise RAG: {e}")
    rag_ready = False

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state["prefill"] = ""

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            st.markdown(
                f"<div class='citation-box'>📚 Sources<br>{msg['citations']}</div>",
                unsafe_allow_html=True
            )
        if show_chunks and msg.get("chunks"):
            with st.expander("🔍 Retrieved chunks"):
                for c in msg["chunks"]:
                    st.markdown(f"**[{c['source']} - page {c['page']}]** score={c.get('score',0):.3f}")
                    st.caption(c["text"][:400])

# Pre-fill from sidebar sample button (consume and clear)
prefill = st.session_state.get("prefill", "")
st.session_state["prefill"] = ""

# Chat input
user_input = st.chat_input("Ask about Driving Licence, RC, Sarathi, Vahan…") or prefill

if user_input and rag_ready:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve + generate
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            from src.embeddings import search
            from src.rag_pipeline import answer_query, format_citations

            retrieved = search(user_input, model, index, bm25, meta, top_k=top_k)
            answer, src_chunks = answer_query(user_input, retrieved, hindi=hindi_mode)
            citations = format_citations(src_chunks)

        st.markdown(answer)
        st.markdown(
            f"<div class='citation-box'>📚 Sources<br>{citations}</div>",
            unsafe_allow_html=True
        )

        if show_chunks:
            with st.expander("🔍 Retrieved chunks"):
                for c in src_chunks:
                    st.markdown(f"**[{c['source']} - page {c['page']}]** score={c.get('score',0):.3f}")
                    st.caption(c["text"][:400])

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "chunks": src_chunks
    })

# Clear button
if st.session_state.messages:
    if st.button("🗑️ Clear chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
