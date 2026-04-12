"""
Skill clustering using K‑means on skill embeddings.
Provides functions to compute skill vectors, cluster, and assign labels.
"""
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.storage.vector_store import vector_store
from config import config

logger = logging.getLogger(__name__)

# Path to save cluster model and labels
CLUSTER_MODEL_PATH = Path("data/cluster_model.pkl")
CLUSTER_LABELS_PATH = Path("data/cluster_labels.pkl")

def compute_skill_vectors() -> np.ndarray:
    """
    Retrieve all candidates and compute skill vectors.
    For simplicity, we use the existing document embeddings (which already incorporate skills).
    Alternatively, we could embed only the skill string. Here we use the full document embedding.
    """
    all_docs = vector_store.collection.get(include=["embeddings", "metadatas"])
    if not all_docs["ids"]:
        return np.array([])
    embeddings = np.array(all_docs["embeddings"])
    return embeddings

def determine_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 10) -> int:
    """Use elbow method to suggest number of clusters (optional)."""
    # Simple heuristic: sqrt(n_samples) or at least 2
    n = embeddings.shape[0]
    return min(max(2, int(np.sqrt(n))), max_clusters)

def fit_clusters(embeddings: np.ndarray, n_clusters: Optional[int] = None) -> KMeans:
    """
    Fit K‑means on embeddings. If n_clusters not given, determine automatically.
    """
    if embeddings.size == 0:
        raise ValueError("No embeddings to cluster")
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(embeddings)
    logger.info(f"Clustering {embeddings.shape[0]} candidates into {n_clusters} clusters")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled)
    # Save scaler and model together
    with open(CLUSTER_MODEL_PATH, "wb") as f:
        pickle.dump({"kmeans": kmeans, "scaler": scaler, "n_clusters": n_clusters}, f)
    return kmeans

def assign_clusters(embeddings: np.ndarray, kmeans: KMeans, scaler: StandardScaler) -> List[int]:
    """Assign cluster labels to embeddings."""
    scaled = scaler.transform(embeddings)
    return kmeans.predict(scaled).tolist()

def update_candidate_metadata(ids: List[str], cluster_labels: List[int]):
    """
    Update each candidate's metadata with the new cluster label.
    We'll need to do a batch update: for each candidate, we modify metadata.
    Chroma doesn't support partial metadata update directly, so we fetch, modify, and upsert.
    """
    # Fetch current metadata and documents
    results = vector_store.collection.get(ids=ids, include=["metadatas", "documents", "embeddings"])
    updated_metadatas = []
    for i, meta in enumerate(results["metadatas"]):
        meta["tech_cluster"] = cluster_labels[i]
        updated_metadatas.append(meta)
    # Upsert with new metadata (keeping same ids, embeddings, documents)
    vector_store.collection.update(
        ids=ids,
        metadatas=updated_metadatas,
        embeddings=results["embeddings"],
        documents=results["documents"],
    )
    logger.info(f"Updated cluster labels for {len(ids)} candidates")

def run_clustering(force_recompute: bool = False):
    """
    Main entry point: compute embeddings, cluster, and update metadata.
    """
    if CLUSTER_MODEL_PATH.exists() and not force_recompute:
        logger.info("Cluster model already exists; use force_recompute=True to overwrite.")
        return

    embeddings = compute_skill_vectors()
    if embeddings.size == 0:
        logger.warning("No candidates to cluster")
        return

    # Fit clusters
    kmeans = fit_clusters(embeddings)
    with open(CLUSTER_MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    scaler = saved["scaler"]
    cluster_labels = assign_clusters(embeddings, kmeans, scaler)

    # Update all candidates
    all_ids = vector_store.collection.get()["ids"]
    update_candidate_metadata(all_ids, cluster_labels)

    # Optionally save labels separately
    with open(CLUSTER_LABELS_PATH, "wb") as f:
        pickle.dump({"ids": all_ids, "labels": cluster_labels}, f)

    logger.info("Clustering completed and metadata updated.")