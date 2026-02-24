# app.py
# Streamlit UI for the Resume Analyzer RAG application

import streamlit as st
import os
from rag_pipeline import (
    process_uploaded_pdf,
    answer_with_rag,
    answer_without_rag,
    SUGGESTED_QUESTIONS
)

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Resume Analyzer — RAG Powered",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f7ff;
        border-left: 4px solid #2E86AB;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .without-rag-box {
        background-color: #fff5f5;
        border-left: 4px solid #e74c3c;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## 🤖 Resume Analyzer")
    st.markdown("**Powered by RAG + GPT-3.5**")
    st.divider()

    st.markdown("### How It Works")
    st.markdown("""
    1. **Upload** your resume PDF
    2. **Ask** any question about it
    3. **Get** specific, grounded answers

    The AI reads your actual resume —
    not generic career advice.
    """)

    st.divider()

    st.markdown("### What is RAG?")
    st.markdown("""
    **Retrieval Augmented Generation**

    Instead of asking an LLM to guess,
    RAG retrieves the relevant parts of
    your document first — then generates
    an answer grounded in your content.

    **Result:** Specific answers, not hallucinations.
    """)

    st.divider()

    show_comparison = st.toggle(
        "Show RAG vs No-RAG comparison",
        value=False,
        help="See the difference RAG makes"
    )

    st.divider()
    st.markdown("**Built by Kanupriya Guha**")
    st.markdown("*Data Science Portfolio 2026*")


# ── MAIN CONTENT ──
st.markdown('<div class="main-header">📄 Resume Analyzer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your resume and ask anything — powered by RAG</div>',
            unsafe_allow_html=True)

# ── SESSION STATE ──
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "resume_processed" not in st.session_state:
    st.session_state.resume_processed = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None
if "current_question" not in st.session_state:
    st.session_state.current_question = ""


# ── STEP 1: UPLOAD ──
st.markdown("## Step 1 — Upload Your Resume")

uploaded_file = st.file_uploader(
    "Upload your resume as a PDF",
    type=["pdf"],
    help="Processed locally — chunked, embedded, stored in ChromaDB"
)

if uploaded_file is not None:

    if not st.session_state.resume_processed:

        with st.spinner("Processing your resume... 10-15 seconds"):

            rag_chain, chunk_count, success, message = process_uploaded_pdf(
                uploaded_file
            )

            if success:
                st.session_state.rag_chain = rag_chain
                st.session_state.resume_processed = True
                st.session_state.chunk_count = chunk_count
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")

    if st.session_state.resume_processed:

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "✅ Ready")
        with col2:
            st.metric("Chunks Created", st.session_state.chunk_count)
        with col3:
            st.metric("Vector DB", "ChromaDB")

        st.divider()

        # ── STEP 2: ASK ──
        st.markdown("## Step 2 — Ask Questions About Your Resume")

        st.markdown("#### 💡 Suggested Questions — Click Any")

        cols = st.columns(2)
        for i, question in enumerate(SUGGESTED_QUESTIONS[:6]):
            with cols[i % 2]:
                if st.button(question, key=f"q_{i}",
                             use_container_width=True):
                    st.session_state.current_question = question

        st.markdown("#### Or Type Your Own")

        user_question = st.text_input(
            "Ask anything about your resume",
            value=st.session_state.current_question,
            placeholder="e.g. What ML skills do I have?"
        )

        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            ask_button = st.button("🔍 Get Answer", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.qa_history = []
                st.session_state.current_answer = None
                st.rerun()

        # ── GENERATE ANSWER ──
        if ask_button and user_question:

            with st.spinner("Searching resume and generating answer..."):

                rag_result = answer_with_rag(
                    st.session_state.rag_chain,
                    user_question
                )

                no_rag_answer = None
                if show_comparison:
                    no_rag_answer = answer_without_rag(user_question)

                st.session_state.current_answer = {
                    "question": user_question,
                    "rag_answer": rag_result["answer"],
                    "sources": rag_result["sources"],
                    "no_rag_answer": no_rag_answer
                }

                st.session_state.qa_history.append({
                    "question": user_question,
                    "answer": rag_result["answer"]
                })

                st.session_state.current_question = ""

        # ── DISPLAY ANSWER ──
        if st.session_state.current_answer:

            current = st.session_state.current_answer

            st.divider()
            st.markdown("## 💬 Answer")
            st.markdown(f"**Question:** *{current['question']}*")

            if show_comparison and current["no_rag_answer"]:

                col_rag, col_no_rag = st.columns(2)

                with col_rag:
                    st.markdown("### ✅ With RAG")
                    st.markdown("*Grounded in your actual resume*")
                    st.markdown(
                        f'<div class="answer-box">{current["rag_answer"]}</div>',
                        unsafe_allow_html=True
                    )

                with col_no_rag:
                    st.markdown("### ❌ Without RAG")
                    st.markdown("*Raw LLM — no resume context*")
                    st.markdown(
                        f'<div class="without-rag-box">{current["no_rag_answer"]}</div>',
                        unsafe_allow_html=True
                    )

                st.info("👆 The RAG answer is specific to YOUR resume. The non-RAG answer is generic and could apply to anyone.")

            else:
                st.markdown(
                    f'<div class="answer-box">{current["rag_answer"]}</div>',
                    unsafe_allow_html=True
                )

            with st.expander(
                f"📎 View {len(current['sources'])} Source Chunks Used"
            ):
                st.markdown(
                    "*Exact parts of your resume used to generate this answer:*"
                )
                for i, source in enumerate(current["sources"]):
                    st.markdown(
                        f'<div class="source-box"><strong>Chunk {i+1} — Page {source["page"] + 1}</strong><br><br>{source["content"]}</div>',
                        unsafe_allow_html=True
                    )

        # ── HISTORY ──
        if len(st.session_state.qa_history) > 1:
            st.divider()
            st.markdown("## 📋 Previous Questions")
            for qa in reversed(st.session_state.qa_history[:-1]):
                with st.expander(f"Q: {qa['question']}"):
                    st.markdown(qa["answer"])

else:
    st.info("👆 Upload your resume PDF above to get started")

    st.markdown("---")
    st.markdown("### 💡 What You Can Ask")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Skills Analysis**
        - What technical skills do I have?
        - What am I missing for ML Engineer roles?
        - How strong is my Python experience?

        **Experience Review**
        - How strong is my work experience?
        - What impact have I demonstrated?
        """)

    with col2:
        st.markdown("""
        **Improvement Advice**
        - What should I improve first?
        - What keywords am I missing?
        - How competitive am I for Data Science?

        **Content Generation**
        - Write me a professional summary
        - What roles suit my background best?
        """)


