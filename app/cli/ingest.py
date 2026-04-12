# indexer.py
from app.storage.vector_store import vector_store
from embedder import embed_text

def index_cv(base_id: str, records: list):
    """Index a list of extracted CV records."""
    if not records:
        return
    for idx, record in enumerate(records):
        unique_id = f"{base_id}_{idx}"
        vector_store.add_cv(unique_id, record, json.dumps(record))
    print(f"📦 Indexed {len(records)} CVs from {base_id}. Total: {vector_store.count()}")