"""
Golden dataset for retrieval evaluation.
Loads and validates a JSON file with queries and relevant candidate IDs.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)

class GoldenSet:
    def __init__(self, path: Path):
        self.path = path
        self.queries: List[Dict[str, Any]] = []
        self.relevant_map: Dict[str, Set[str]] = {}  # query_id -> set of relevant candidate IDs
        self.query_text_map: Dict[str, str] = {}     # query_id -> query text
        self._load()

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Golden set file not found: {self.path}")
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "queries" not in data:
            raise ValueError("Golden set must contain a 'queries' list")
        for q in data["queries"]:
            qid = q.get("id")
            text = q.get("text")
            relevant = q.get("relevant", [])
            if not qid or not text:
                raise ValueError("Each query must have 'id' and 'text'")
            self.queries.append(q)
            self.relevant_map[qid] = set(relevant)
            self.query_text_map[qid] = text
        logger.info(f"Loaded golden set with {len(self.queries)} queries")

    def get_query_text(self, qid: str) -> str:
        return self.query_text_map.get(qid, "")

    def get_relevant(self, qid: str) -> Set[str]:
        return self.relevant_map.get(qid, set())

    def get_all_queries(self) -> List[Dict]:
        return self.queries