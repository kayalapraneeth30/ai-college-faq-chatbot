import re
from pathlib import Path
from typing import List, Dict, Any

class DocumentChunk:
    def __init__(self, text: str, source: str, chunk_id: int, metadata: Dict[str, Any] = None):
        self.text = text.strip()
        self.source = source
        self.chunk_id = chunk_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata
        }

class DocumentProcessor:
    def __init__(self, chunk_size: int = 450, chunk_overlap: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: Path) -> str:
        """Loads text from file with robust encoding fallback."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

    def chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        """
        Recursive Character Chunking:
        Splits by double newline (paragraphs), single newline, sentence endings (.!?), or words.
        """
        # Normalize whitespace
        text = text.replace("\r\n", "\n")
        paragraphs = re.split(r'\n{2,}', text)
        
        raw_chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) > self.chunk_size:
                # Split large paragraph by single newline or sentences
                sentences = re.split(r'(?<=[.!?])\s+|\n', para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                        current_chunk = f"{current_chunk} {sentence}".strip()
                    else:
                        if current_chunk:
                            raw_chunks.append(current_chunk)
                        # Start new chunk with overlap if possible
                        if len(current_chunk) > self.chunk_overlap:
                            overlap = current_chunk[-self.chunk_overlap:]
                            current_chunk = f"{overlap} {sentence}".strip()
                        else:
                            current_chunk = sentence
            else:
                if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                    current_chunk = f"{current_chunk}\n\n{para}".strip()
                else:
                    if current_chunk:
                        raw_chunks.append(current_chunk)
                    current_chunk = para

        if current_chunk:
            raw_chunks.append(current_chunk)

        # Convert to DocumentChunk objects
        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) > 20:  # Ignore trivial noise
                chunks.append(
                    DocumentChunk(
                        text=chunk_text.strip(),
                        source=source,
                        chunk_id=idx,
                        metadata={
                            "char_count": len(chunk_text),
                            "word_count": len(chunk_text.split())
                        }
                    )
                )
        return chunks

    def process_directory(self, directory: Path) -> List[DocumentChunk]:
        """Processes all text/markdown files in a directory."""
        all_chunks: List[DocumentChunk] = []
        supported_extensions = [".txt", ".md", ".json"]
        
        for file_path in directory.glob("*.*"):
            if file_path.suffix.lower() in supported_extensions:
                content = self.load_document(file_path)
                chunks = self.chunk_text(content, source=file_path.name)
                all_chunks.extend(chunks)

        return all_chunks
