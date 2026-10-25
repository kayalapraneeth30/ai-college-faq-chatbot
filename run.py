import uvicorn
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting AI-Powered College FAQ Chatbot (FastAPI + RAG)")
    print("📍 Local Web App:  http://127.0.0.1:8000")
    print("📖 API Swagger UI: http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
