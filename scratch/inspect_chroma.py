import chromadb
from pathlib import Path
import json

def inspect_chroma():
    db_path = "data/chroma_db"
    if not Path(db_path).exists():
        print(f"Error: {db_path} does not exist")
        return

    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection("cvs")
        results = collection.get(limit=5, include=["metadatas"])
        
        print(f"Total entries: {collection.count()}")
        print("\nSample Entries (ID and file_path):")
        for i, id_val in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            print(f"ID: {id_val}")
            print(f"File Path: {meta.get('file_path', 'MISSING')}")
            if meta.get('file_path'):
                print(f"File exists on disk: {Path(meta['file_path']).exists()}")
            print("-" * 20)
            
    except Exception as e:
        print(f"Error accessing collection: {e}")

if __name__ == "__main__":
    inspect_chroma()
