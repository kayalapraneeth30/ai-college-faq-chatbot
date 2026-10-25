import json
import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

from backend.config import INDEX_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_DIM
from backend.document_processor import DocumentChunk

class VectorStore:
    def __init__(self, index_dir: Path = INDEX_DIR, dim: int = EMBEDDING_DIM):
        self.index_dir = index_dir
        self.dim = dim
        self.index_file = self.index_dir / "college_faiss.index"
        self.meta_file = self.index_dir / "chunks_metadata.json"
        
        self.transformer_model = None
        self._init_embedding_model()
        
        self.index = None
        self.chunks_data: List[Dict[str, Any]] = []
        
        # Load existing index if present
        self.load()

    def _init_embedding_model(self):
        """Initializes SentenceTransformer or falls back gracefully."""
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.transformer_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                self.dim = self.transformer_model.get_sentence_embedding_dimension()
                print(f"[VectorStore] Loaded SentenceTransformer: {EMBEDDING_MODEL_NAME} (dim={self.dim})")
            except Exception as e:
                print(f"[VectorStore] SentenceTransformer init failed ({e}), using Semantic Feature Embeddings.")
                self.transformer_model = None
        else:
            print("[VectorStore] sentence-transformers not installed; using Semantic Feature Embeddings.")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts.
        Uses SentenceTransformer if available, or a deterministic semantic n-gram projection with NumPy.
        """
        if self.transformer_model is not None:
            embeddings = self.transformer_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            # Normalize for Cosine Similarity
            faiss.normalize_L2(embeddings) if FAISS_AVAILABLE else None
            return embeddings.astype('float32')
        
        # Robust semantic projection fallback using term/character hashing & normalization
        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().replace("\n", " ").split()
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in tokens:
                # Seeded projection per token
                h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 1) % 2 == 0 else -1.0
                vec[idx] += sign
                # Bigram contribution
                idx2 = (h >> 3) % self.dim
                vec[idx2] += sign * 0.5
            norm = np.linalg.norm(vec)
            if norm > 0:
                embeddings[i] = vec / norm
            else:
                embeddings[i] = vec
        return embeddings

    def build_index(self, chunks: List[DocumentChunk]):
        """Encodes all document chunks and builds the FAISS index."""
        if not chunks:
            print("[VectorStore] No chunks provided to index.")
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        if FAISS_AVAILABLE:
            # Inner Product index on L2-normalized vectors computes exact Cosine Similarity
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
        else:
            # Pure NumPy cosine similarity storage fallback
            self.index = embeddings

        self.chunks_data = [chunk.to_dict() for chunk in chunks]
        self.save()
        print(f"[VectorStore] Indexed {len(chunks)} chunks into FAISS vector store.")

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """Searches vector index for chunks most relevant to query."""
        if self.index is None or not self.chunks_data:
            return []

        query_emb = self.embed_texts([query])

        if FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
            scores, indices = self.index.search(query_emb, min(top_k, len(self.chunks_data)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks_data):
                    # Normalized cosine similarity score clamped between 0 and 1
                    cos_score = float(max(0.0, min(1.0, (score + 1.0) / 2.0 if score < 0 else score)))
                    results.append((self.chunks_data[idx], cos_score))
            return results
        else:
            # NumPy Cosine Search fallback
            matrix = self.index
            scores = np.dot(matrix, query_emb.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = float(scores[idx])
                cos_score = float(max(0.0, min(1.0, score)))
                results.append((self.chunks_data[idx], cos_score))
            return results

    def save(self):
        """Saves FAISS index and chunks metadata to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
            faiss.write_index(self.index, str(self.index_file))
        elif isinstance(self.index, np.ndarray):
            np.save(str(self.index_dir / "numpy_embeddings.npy"), self.index)

        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks_data, f, indent=2)

    def load(self):
        """Loads FAISS index and chunks metadata from disk."""
        if self.meta_file.exists():
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.chunks_data = json.load(f)

        if FAISS_AVAILABLE and self.index_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                print(f"[VectorStore] Successfully loaded FAISS index with {self.index.ntotal} vectors.")
            except Exception as e:
                print(f"[VectorStore] Failed loading FAISS index: {e}")
        elif (self.index_dir / "numpy_embeddings.npy").exists():
            self.index = np.load(str(self.index_dir / "numpy_embeddings.npy"))
