"""
Persistent vector store using ChromaDB.
Safe-Mode: Includes timeouts and fail-safe fallbacks for low-memory cloud environments.
"""
import os
import time
from pathlib import Path
from config import config

class VectorStore:
    def __init__(self):
        self.db_path = Path(config.CHROMA_DB_DIR)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        self._embedder = None
        self._is_safe = True # Flag to track if the DB is actually connected

    @property
    def client(self):
        if self._client is None and self._is_safe:
            try:
                import chromadb
                from chromadb.config import Settings
                print(f"Connecting to ChromaDB at: {self.db_path}...")
                # Use a lightweight connection setting
                self._client = chromadb.PersistentClient(
                    path=str(self.db_path),
                    settings=Settings(anonymized_telemetry=False)
                )
            except Exception as e:
                print(f"CRITICAL: ChromaDB connection failed: {e}")
                self._is_safe = False
                self._client = None
        return self._client

    @property
    def collection(self):
        # Fail-safe: If DB fails, return a Mock object that returns empty lists
        # This prevents the whole API from crashing (502/504 error)
        if (self.client is None) or (not self._is_safe):
            class MockCollection:
                def get(self, **kwargs): return {"ids": [], "metadatas": [], "documents": []}
                def count(self): return 0
                def query(self, **kwargs): return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}
            return MockCollection()

        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=config.CHROMA_COLLECTION,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"ERROR: Collection access failed: {e}")
                self._is_safe = False
                return self.collection # Recursive call returns Mock
        return self._collection

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model: {config.EMBEDDING_MODEL}...")
                self._embedder = SentenceTransformer(config.EMBEDDING_MODEL)
                # Force single-thread
                import torch
                torch.set_num_threads(1)
            except Exception as e:
                print(f"ERROR: Embedding model failed to load: {e}")
        return self._embedder

    def count(self) -> int:
        return self.collection.count()

# Singleton instance
vector_store = VectorStore()