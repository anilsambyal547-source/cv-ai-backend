"""
Persistent vector store using ChromaDB.
Singleton shared across indexer, retriever, and CLI.
"""
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import config

class VectorStore:
    def __init__(self):
        self.db_path = Path(config.CHROMA_DB_DIR)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=config.CHROMA_TELEMETRY)
        )

        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            print(f"Loading embedding model: {config.EMBEDDING_MODEL}...")
            self._embedder = SentenceTransformer(config.EMBEDDING_MODEL)
            # Optimize Torch for low memory (Render 512MB limit)
            try:
                import torch
                torch.set_num_threads(1)
            except ImportError:
                pass
        return self._embedder

    def count(self) -> int:
        return self.collection.count()

# Singleton instance
vector_store = VectorStore()