from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import os

from backend.config import DATA_DIR, INDEX_DIR, STATIC_DIR
from backend.document_processor import DocumentProcessor
from backend.vector_store import VectorStore
from backend.rag_engine import RAGEngine

app = FastAPI(
    title="College AI FAQ Assistant API",
    description="RAG-powered intelligent QA system for college academic regulations, fees, exams, and campus FAQs.",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Components
doc_processor = DocumentProcessor()
vector_store = VectorStore(index_dir=INDEX_DIR)
rag_engine = RAGEngine(vector_store=vector_store)

# Ensure sample documents are indexed on startup
@app.on_event("startup")
def startup_event():
    if not vector_store.chunks_data or vector_store.index is None:
        print("[Startup] Index not found. Auto-indexing initial college documents...")
        chunks = doc_processor.process_directory(DATA_DIR)
        vector_store.build_index(chunks)
        print(f"[Startup] Done. Indexed {len(chunks)} chunks.")

# Request / Response Schemas
class ChatRequest(BaseModel):
    message: str = Field(..., description="Student query or question")
    session_id: str = Field(default="default_session", description="Session identifier for conversation memory")

class ChatResponse(BaseModel):
    query: str
    session_id: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    latency_ms: float
    suggestions: List[str]

class ClearSessionRequest(BaseModel):
    session_id: str

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI-Powered College FAQ Chatbot",
        "vector_store_active": vector_store.index is not None,
        "indexed_chunks": len(vector_store.chunks_data),
        "documents_directory": str(DATA_DIR)
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")
    
    result = rag_engine.answer_query(request.message, session_id=request.session_id)
    return result

@app.get("/api/documents")
def list_documents():
    docs = []
    for file_path in DATA_DIR.glob("*.*"):
        if file_path.is_file():
            chunk_count = sum(1 for c in vector_store.chunks_data if c.get("source") == file_path.name)
            docs.append({
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "chunks_indexed": chunk_count
            })
    return {
        "total_documents": len(docs),
        "total_chunks": len(vector_store.chunks_data),
        "documents": docs
    }

@app.post("/api/reindex")
def reindex_knowledge_base():
    chunks = doc_processor.process_directory(DATA_DIR)
    vector_store.build_index(chunks)
    return {
        "status": "success",
        "message": f"Successfully processed {len(chunks)} document chunks into FAISS vector index.",
        "chunks_indexed": len(chunks)
    }

@app.get("/api/faqs")
def get_popular_faqs():
    return [
        {
            "category": "Academics",
            "question": "What is the minimum attendance required to write semester exams?",
            "tag": "Attendance"
        },
        {
            "category": "Examinations",
            "question": "What is the process and fee for semester exam revaluation?",
            "tag": "Revaluation"
        },
        {
            "category": "Fees & Deadlines",
            "question": "When is the tuition fee payment deadline for odd and even semesters?",
            "tag": "Tuition Fee"
        },
        {
            "category": "Placements",
            "question": "What is the minimum CGPA required to attend campus placements?",
            "tag": "Placements"
        },
        {
            "category": "Campus Life",
            "question": "What are the hostel in-out curfew and mess timings?",
            "tag": "Hostel"
        },
        {
            "category": "Academics",
            "question": "How many total credits are required for degree completion?",
            "tag": "Credits"
        }
    ]

@app.post("/api/clear-session")
def clear_session(req: ClearSessionRequest):
    rag_engine.clear_session(req.session_id)
    return {"status": "success", "session_id": req.session_id}

# Mount static frontend
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AI College FAQ Assistant API is running. Check /docs for Swagger API specification."}
