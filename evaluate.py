# evaluate.py
import json
import logging
from pathlib import Path
from typing import List, Set, Dict, Any

import typer
from rich.console import Console
from rich.table import Table

from config import config
from app.retrieval.retriever import HybridCVRetriever
from app.storage.vector_store import vector_store

logger = logging.getLogger(__name__)
console = Console()

def load_golden_set(path: Path) -> List[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load golden set: {e}")
        return []

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)
    return relevant_retrieved / k

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)
    return relevant_retrieved / len(relevant)

def run_evaluation(golden_set_path: Path, alpha: float, top_k: int):
    test_cases = load_golden_set(golden_set_path)
    if not test_cases:
        console.print("[red]Golden set empty or failed to load.[/red]")
        return

    retriever = HybridCVRetriever(
        collection=vector_store.collection,
        embedder=vector_store.embedder,
        bm25_index_path=config.BM25_INDEX_PATH
    )
    # Ensure BM25 index is built
    retriever.build_bm25_index()

    precisions = []
    recalls = []

    for case in test_cases:
        query = case.get("query", "")
        relevant_ids = set(case.get("relevant_ids", []))
        if not query:
            logger.warning("Skipping case with empty query")
            continue

        results = retriever.hybrid_search(query, top_k=top_k, alpha=alpha)
        retrieved_ids = [r["id"] for r in results]

        p = precision_at_k(retrieved_ids, relevant_ids, top_k)
        r = recall_at_k(retrieved_ids, relevant_ids, top_k)
        precisions.append(p)
        recalls.append(r)
        logger.info(f"Query: {query[:50]}... P@{top_k}={p:.3f}, R@{top_k}={r:.3f}")

    if precisions:
        avg_p = sum(precisions) / len(precisions)
        avg_r = sum(recalls) / len(recalls)
        console.print(f"\n[bold]Average Precision@{top_k}: {avg_p:.3f}[/bold]")
        console.print(f"[bold]Average Recall@{top_k}: {avg_r:.3f}[/bold]")
    else:
        console.print("[yellow]No valid queries evaluated.[/yellow]")