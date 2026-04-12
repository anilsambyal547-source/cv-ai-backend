import chromadb

_client = chromadb.Client(
    chromadb.Settings(persist_directory="data/chroma")
)

_collection = _client.get_or_create_collection("cvs")

def add_cv(doc_id: str, embedding: list, metadata: dict):
    _collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[metadata]
    )
