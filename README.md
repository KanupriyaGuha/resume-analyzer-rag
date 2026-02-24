# 📄 Resume Analyzer — RAG Powered

Upload your resume PDF and ask anything about it.
Get specific, grounded answers powered by Retrieval Augmented Generation.

## How RAG Works

Your Resume PDF
→ Text Extraction (PyPDF)
→ Chunking (500 char chunks, 50 overlap)
→ Embedding (OpenAI text-embedding-ada-002)
→ Vector Storage (ChromaDB)
→ Question Asked → Chunks Retrieved → Grounded Answer

## Tech Stack

- LangChain — RAG pipeline orchestration
- ChromaDB — Local vector database
- OpenAI API — Embeddings + GPT-3.5-turbo
- Streamlit — Web UI
- PyPDF — PDF text extraction

## How to Run

git clone https://github.com/KanupriyaGuha/resume-analyzer-rag.git
cd resume-analyzer-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Add OPENAI_API_KEY to .env file
streamlit run app.py

## Author

Kanupriya Guha | Data Science Portfolio | 2026
```

Save ✅

---

## STEP 10 — Check Your File Structure

Your folder should look exactly like this:
```
resume-analyzer-rag/
├── venv/
├── .env
├── rag_pipeline.py
├── app.py
├── requirements.txt
└── README.md