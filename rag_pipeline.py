import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

# Load environment variables from .env file
load_dotenv()

# ── CONFIGURATION ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
COLLECTION_NAME = "resume_collection"
CHROMA_PERSIST_DIR = "./chroma_db"

def load_and_chunk_pdf(pdf_path: str) -> list:
    """
    Load a PDF and split it into overlapping chunks.
    
    Why chunking?
    - LLMs haven token limits - can't send entire document at once
    - Smaller chunks = more precise retrieval
    - Overlap ensures contect isn't lost at chunk boundaries
    """
    print(f"Loading PDF: {pdf_path}")

    # Load PDF - extracts text page by page
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print(f"Pages loaded: {len(pages)}")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(pages)
    print(f"Chunks created: {len(chunks)}")

    return chunks

def create_vector_store(chunks: list) -> Chroma:
    """
    Convert chunks to embeddingd and store in ChromaDB.
    
    Why embeddingd?
    - Embeddingd convert text to vectors (lists of numbers)
    - Similar meaning = similar vectors = close in vector spacce
    - Enable semantic search - finds meaning, not just keywords
    
    Why ChromaDB?
    - Local vector database - no external service needed
    - Persists to disk so you don't re-embed on every run
    - Fast similarity search
    """

    print("Creating embeddingd and storing in ChromaDB...")

    # OpenAI embeddingd - converts text to 1536-dimensional vectors
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-ada-002"
    )

    # Create vector store from chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )

    print(f"Vector store created with {len(chunks)} chunks")
    return vector_store

def load_existing_vector_store() -> Chroma:
    """
    Load an already-creataed vector store from disk."""
    embeddingd = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-ada-002"
    )

    vectore_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    return vectore_store

def build_rag_chain(vector_store: Chroma) -> RetrievalQA:
    """
    Build the RAG chain connecting retrieval to generation.
    
    How RAG works:
    1. User asks a question
    2. Question gets embedded into a vector
    3. ChromaDB finds the 4 most similar chunks (retrieval)
    4. Retrived chunks + question get sent to LLM (augmented generation)
    5. LLM generates answer grounded in retrieved content
    
    Why this reduce hallucinations:
    - LLM is given the actual docuemnt content
    - Instructed to answer ONLY from that content
    - If answer isn't in document, it says so
    """

    # The LLM that generates final answers
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo",
        temperature=0
    )

    # Custom prompt that keeps the LLM grounded
    prompt_template = """You are an expert career coach and resume analyst.
    
    Use ONLY the following context from the resume to answer the question.
    If the information is not in the context provided, say "I don't see that information in the resume."
    Do not make up or assume information that isn't explicitly in the resume.
    
    Be specific and actionable in your feedback.
    Reference specific sections, skills, or experiences from the resume when relevant.
    
    Context from resume:
    {context}
    
    Question: {question}
    
    Detailed Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Build the chain
    # k=4 means retrieve 4 most relevant chunks per question
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return rag_chain

def answer_with_rag(rag_chain: RetrievalQA, question: str) -> dict:
    """
    Answer a question using RAG.
    Returns the answer and the source chunks used.
    """
    result = rag_chain.invoke({"query": question})

    answer = result["result"]
    source_docs = result["source_documents"]

    sources = [
        {
            "content": doc.page_content,
            "page": doc.metadata.get("page", 0)
        }
        for doc in source_docs
    ]

    return {
        "answer": answer,
        "sources": sources
    }

def answer_without_rag(question: str) -> str:
    """
    Answer the same question WITHOUT RAG - just raw LLM.
    Used to demonstrate the difference RAG makes.
    The answer will be vague because the LLM has no resume contect.
    """

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo",
        temperature=0
    )

    prompt = f"""Answer this question about a resume: {question}

    Note: You don't have access to the actual resume content.
    Answer as best you can based on general knowledge only."""

    response = llm.invoke(prompt)
    return response.strip()

def process_uploaded_pdf(uploaded_file) -> tuple:
    """
    Handle a PDF uploaded through Streamlit.
    Saves to temp file, processes, returns chain and status.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        chunks = load_and_chunk_pdf(tmp_path)
        vector_store = create_vector_store(chunks)
        rag_chain = build_rag_chain(vector_store)

        return rag_chain, len(chunks), True, "Resume processed successfully!"

    except Exception as e:
        return None, 0, False, f"Error processing PDF: {str(e)}"

    finally:
        os.unlink(tmp_path)

# ── SUGGESTED QUESTIONS ──
SUGGESTED_QUESTIONS = [
    "What are my strongest technical skills?",
    "What skills am I missing for a Machine Learning Engineer role?",
    "How strong is my work experience section?",
    "What projects do I have and how impressive are they?",
    "What should I improve to be more competitive for Data Science roles?",
    "How does my education background support my target roles?",
    "What keywords am I missing that recruiters look for?",
    "What is my biggest weakness as a candidate based on this resume?",
    "What roles am I best qualified for right now?",
    "Write me a 3-sentence professional summary based on this resume",
]        