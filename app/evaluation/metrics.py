"""
Retrieval metrics: Precision@K, Recall@K, MRR.
"""
from typing import List, Set, Dict, Any

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Precision@K = (# of relevant in top K) / K"""
    if k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    relevant_in_top = sum(1 for doc in retrieved_k if doc in relevant)
    return relevant_in_top / k

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Recall@K = (# of relevant in top K) / total relevant"""
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_in_top = sum(1 for doc in retrieved_k if doc in relevant)
    return relevant_in_top / len(relevant)

def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """Reciprocal rank = 1 / (rank of first relevant item), 0 if none found"""
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0

def mrr(results: List[Dict[str, Any]], relevant_map: Dict[str, Set[str]]) -> float:
    """Mean Reciprocal Rank over a list of query results"""
    total = 0.0
    count = 0
    for item in results:
        qid = item.get("query_id")
        retrieved = item.get("retrieved_ids", [])
        relevant = relevant_map.get(qid, set())
        total += reciprocal_rank(retrieved, relevant)
        count += 1
    return total / count if count > 0 else 0.0