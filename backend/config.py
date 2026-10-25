import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "college_docs"
INDEX_DIR = BASE_DIR / "data" / "vector_index"
STATIC_DIR = BASE_DIR / "frontend"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Parameters
CHUNK_SIZE = 450
CHUNK_OVERLAP = 60

# Model & Retrieval Settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K_RESULTS = 4
SIMILARITY_THRESHOLD = 0.25

# Optional LLM API Settings (Can be set via env vars)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
