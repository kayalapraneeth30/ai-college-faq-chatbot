import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import DATA_DIR
from backend.document_processor import DocumentProcessor
from backend.vector_store import VectorStore
from backend.rag_engine import RAGEngine

def test_pipeline():
    print("\n--- [1] Testing Document Ingestion & Chunking ---")
    processor = DocumentProcessor()
    chunks = processor.process_directory(DATA_DIR)
    assert len(chunks) > 0, "No chunks generated from documents directory"
    print(f"[OK] Successfully processed {len(chunks)} chunks across college documents.")

    print("\n--- [2] Testing Vector Indexing & Search ---")
    store = VectorStore()
    store.build_index(chunks)
    assert len(store.chunks_data) == len(chunks), "Metadata length mismatch"
    print(f"[OK] FAISS vector store built with {len(store.chunks_data)} entries.")

    print("\n--- [3] Testing Semantic Retrieval & RAG Query ---")
    engine = RAGEngine(vector_store=store)
    
    test_queries = [
        "What is the minimum attendance required?",
        "How much is the revaluation fee for exams?",
        "When is the tuition fee payment deadline?",
        "What are the hostel curfew and mess timings?",
        "What is the minimum CGPA required for placements?"
    ]

    for q in test_queries:
        res = engine.answer_query(q)
        print(f"\nQuery: '{q}'")
        print(f"  -> Top Score: {res['confidence_score']}%")
        print(f"  -> Sources Found: {len(res['sources'])}")
        print(f"  -> Latency: {res['latency_ms']} ms")
        assert len(res['sources']) > 0, "No sources retrieved"
        assert len(res['answer']) > 20, "Answer too short"

    print("\n[SUCCESS] All pipeline tests passed successfully!")

if __name__ == "__main__":
    test_pipeline()
