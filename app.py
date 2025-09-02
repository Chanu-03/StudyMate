# -------------------------
# app.py — StudyMate AI (with SEO & bright centered loader)
# -------------------------
import streamlit as st
from datetime import datetime
import pandas as pd

from utils import extract_text_from_uploaded_file, chunk_text, get_embeddings_batch, generate_answer
from vector_store import SimpleVectorStore

# -------------------------
# SEO & Page Configuration
# -------------------------
st.set_page_config(
    page_title="StudyMate AI - Smart Study Helper",  # Browser tab title
    page_icon="✍️",                                 # Tab icon
    layout="wide"                                   # Full-width layout
)

# Meta tags for SEO
st.markdown("""
    <meta name="description" content="StudyMate AI lets you upload PDFs, DOCX, PPTX, and TXT files and quickly get answers.">
    <meta name="keywords" content="AI, StudyMate, PDFs, DOCX, PPTX, TXT, Question Answering">
""", unsafe_allow_html=True)

st.title("✍️ StudyMate AI — Upload PDFs, DOCX, PPTX, TXT")

# -------------------------
# Session State Setup
# -------------------------
if "vs" not in st.session_state:
    st.session_state.vs = SimpleVectorStore()
if "documents" not in st.session_state:
    st.session_state.documents = []

st.sidebar.markdown("### Actions")
if st.sidebar.button("🗑️ Clear all"):
    st.session_state.vs.reset()
    st.session_state.documents = []
    st.sidebar.success("Cleared in-memory data")

# -------------------------
# Centered Loader Function
# -------------------------
def show_loader(message="Processing..."):
    """Show a bright centered loader with GIF."""
    loader_placeholder = st.empty()
    loader_placeholder.markdown(
        f"""
        <div style="
            display:flex; 
            justify-content:center; 
            align-items:center; 
            height:250px; 
            background-color:#e6f7ff;
            border-radius:12px;
            flex-direction:column;
        ">
            <img src="https://i.gifer.com/ZZ5H.gif" width="120" />
            <p style="font-size:18px; font-weight:bold; margin-top:12px;">
                {message}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    return loader_placeholder

# -------------------------
# File Upload & Processing
# -------------------------
uploaded_files = st.file_uploader(
    "Upload files (PDF, DOCX, PPTX, TXT)",
    accept_multiple_files=True,
    type=["pdf", "docx", "pptx", "txt"]
)

if uploaded_files:
    if st.button("Process files"):
        n = len(uploaded_files)
        processed_count = 0
        loader = show_loader("Processing uploaded files...")

        for uf in uploaded_files:
            st.info(f"Processing {uf.name}")
            text, ftype = extract_text_from_uploaded_file(uf)
            if not text:
                st.warning(f"No text found in {uf.name}")
                continue

            # Chunk text
            chunks = chunk_text(text)

            # Get embeddings
            embeddings = get_embeddings_batch(chunks, doc_name=uf.name)

            # Metadata
            metadatas = [{"document": uf.name, "chunk_index": idx, "text": chunk} for idx, chunk in enumerate(chunks)]

            # Add to vector store
            st.session_state.vs.add(embeddings, metadatas)

            # Document info
            st.session_state.documents.append({
                "name": uf.name,
                "length": len(text),
                "chunks": len(chunks),
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

            processed_count += 1

        loader.empty()  # remove loader
        st.success(f"Processed {processed_count}/{n} files")

# -------------------------
# Document Table + QnA
# -------------------------
if st.session_state.documents:
    st.subheader("Processed Documents")
    df = pd.DataFrame(st.session_state.documents)
    st.dataframe(df, use_container_width=True)

    st.subheader("Ask a question")
    question = st.text_area("Enter your question")
    top_k = st.slider("Number of retrieved chunks (top_k)", min_value=1, max_value=10, value=4)

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            # Search loader
            loader = show_loader("Searching relevant passages...")
            q_emb = get_embeddings_batch([question])[0]
            hits = st.session_state.vs.search(q_emb, top_k=top_k)
            loader.empty()

            if not hits:
                st.warning("No relevant passages found — upload documents first.")
            else:
                contexts = [{"text": h["metadata"]["text"], "source": h["metadata"].get("document","unknown")} for h in hits]

                # Answer generation loader
                loader = show_loader("Generating answer...")
                answer = generate_answer(question, contexts)
                loader.empty()

                st.markdown("### 🤖 Answer")
                st.write(answer)

                st.markdown("### 📖 Sources (retrieved)")
                for i, h in enumerate(hits, start=1):
                    md = h["metadata"]
                    st.markdown(f"**Source {i}:** {md.get('document')} — score {h['score']:.3f}")
                    snippet = md['text'][:400] + ("…" if len(md['text']) > 400 else "")
                    st.write(snippet)
