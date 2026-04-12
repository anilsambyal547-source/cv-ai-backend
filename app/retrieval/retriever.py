"""
Hybrid search with cross‑encoder reranking, skill priority, fit breakdown, skill gap detection,
minimum experience filtering, result diversity control, advanced filters, query expansion,
caching (Redis + in‑memory), boolean query support, embedding cache, and email decryption.
"""
import pickle
import logging
import re
import hashlib
import json
import copy
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Callable

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import Collection
import redis
from redis import Redis

from config import config
from app.utils.encryption import decrypt

logger = logging.getLogger(__name__)

class BooleanQueryParser:
    """
    Simple parser for boolean expressions like "python AND docker OR (java NOT kotlin)".
    Supports AND, OR, NOT, and parentheses. Operands are skill names (case‑insensitive).
    """

    class Node:
        pass

    class Literal(Node):
        def __init__(self, skill: str):
            self.skill = skill.lower()

    class Not(Node):
        def __init__(self, child):
            self.child = child

    class And(Node):
        def __init__(self, left, right):
            self.left = left
            self.right = right

    class Or(Node):
        def __init__(self, left, right):
            self.left = left
            self.right = right

    def __init__(self, expression: str):
        self.tokens = self._tokenize(expression)
        self.pos = 0
        self.ast = self._parse_expression()

    def _tokenize(self, expr: str) -> List[str]:
        # Insert spaces around parentheses for easier splitting
        expr = expr.replace('(', ' ( ').replace(')', ' ) ')
        tokens = expr.split()
        result = []
        for t in tokens:
            if t.upper() in ('AND', 'OR', 'NOT') or t in ('(', ')'):
                result.append(t.upper())
            else:
                # assume it's a skill term
                result.append(t.lower())
        return result

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self, expected: Optional[str] = None):
        if expected and self._peek() != expected:
            raise ValueError(f"Expected '{expected}', got '{self._peek()}'")
        token = self._peek()
        self.pos += 1
        return token

    def _parse_expression(self):
        # Parse the lowest precedence: OR
        return self._parse_or()

    def _parse_or(self):
        left = self._parse_and()
        while self._peek() == 'OR':
            self._consume('OR')
            right = self._parse_and()
            left = self.Or(left, right)
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self._peek() == 'AND':
            self._consume('AND')
            right = self._parse_not()
            left = self.And(left, right)
        return left

    def _parse_not(self):
        if self._peek() == 'NOT':
            self._consume('NOT')
            child = self._parse_primary()
            return self.Not(child)
        return self._parse_primary()

    def _parse_primary(self):
        token = self._peek()
        if token == '(':
            self._consume('(')
            expr = self._parse_expression()
            self._consume(')')
            return expr
        elif token is not None and token not in ('AND', 'OR', 'NOT', '(', ')'):
            self._consume()
            return self.Literal(token)
        else:
            raise ValueError(f"Unexpected token: {token}")

    def evaluate(self, skills: Set[str]) -> bool:
        """
        Evaluate the boolean expression against a set of skills (lowercase).
        """
        def _eval(node) -> bool:
            if isinstance(node, self.Literal):
                return node.skill in skills
            elif isinstance(node, self.Not):
                return not _eval(node.child)
            elif isinstance(node, self.And):
                return _eval(node.left) and _eval(node.right)
            elif isinstance(node, self.Or):
                return _eval(node.left) or _eval(node.right)
            else:
                raise TypeError(f"Unknown node type: {type(node)}")
        return _eval(self.ast)


class HybridCVRetriever:
    def __init__(
        self,
        collection: Collection,
        embedder: SentenceTransformer,
        bm25_index_path: Path,
        cross_encoder: Optional[CrossEncoder] = None,
        redis_client: Optional[Redis] = None,
    ):
        self.collection = collection
        self.embedder = embedder
        self.bm25_index_path = bm25_index_path
        self.cross_encoder = cross_encoder
        self.redis_client = redis_client
        self.redis_ttl = getattr(config, 'REDIS_CACHE_TTL', 3600)
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_tokenized: List[List[str]] = []
        self.doc_metadatas: List[Dict] = []

        # In‑memory cache for search results (fallback)
        self._cache: Dict[str, List[Dict]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Cache for query embeddings (in‑memory only, per session)
        self._embedding_cache: Dict[str, List[float]] = {}

    def _cache_key(
        self,
        query: str,
        top_k: int,
        alpha: float,
        min_experience: Optional[float],
        desired_experience: Optional[float],
        query_skills: Optional[Set[str]],
        skill_priority: Optional[Dict[str, float]],
        required_education: Optional[str],
        required_seniority: Optional[str],
        use_rerank: bool,
        diversity_lambda: Optional[float],
        filter_skills: Optional[Set[str]],
        filter_min_exp: Optional[float],
        filter_max_exp: Optional[float],
        filter_edu: Optional[str],
        filter_seniority: Optional[str],
        expand_query: bool,
        boolean_query: Optional[str],
    ) -> str:
        query_skills_list = sorted(query_skills) if query_skills else []
        filter_skills_list = sorted(filter_skills) if filter_skills else []
        skill_priority_list = sorted(skill_priority.items()) if skill_priority else []

        data = {
            "query": query,
            "top_k": top_k,
            "alpha": alpha,
            "min_experience": min_experience,
            "desired_experience": desired_experience,
            "query_skills": query_skills_list,
            "skill_priority": skill_priority_list,
            "required_education": required_education,
            "required_seniority": required_seniority,
            "use_rerank": use_rerank,
            "diversity_lambda": diversity_lambda,
            "filter_skills": filter_skills_list,
            "filter_min_exp": filter_min_exp,
            "filter_max_exp": filter_max_exp,
            "filter_edu": filter_edu,
            "filter_seniority": filter_seniority,
            "expand_query": expand_query,
            "boolean_query": boolean_query,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Tokenization and skill extraction
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def extract_skills_from_text(self, text: str) -> Set[str]:
        text_lower = text.lower()
        found = set()
        for canon, variants in config.SKILL_SYNONYMS.items():
            if canon in text_lower:
                found.add(canon)
            else:
                for var in variants:
                    if var in text_lower:
                        found.add(canon)
                        break
        return found

    # ------------------------------------------------------------------
    # Query expansion using synonyms
    # ------------------------------------------------------------------
    def expand_query(self, query: str) -> str:
        expanded = query
        text_lower = query.lower()
        for canon, variants in config.SKILL_SYNONYMS.items():
            if canon in text_lower:
                expanded += " " + " ".join(variants)
            else:
                for var in variants:
                    if var in text_lower:
                        expanded += " " + canon + " " + " ".join(variants)
                        break
        return expanded

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------
    def build_bm25_index(self, force_rebuild: bool = False) -> None:
        if not force_rebuild and self.bm25_index_path.exists():
            try:
                with open(self.bm25_index_path, "rb") as f:
                    data = pickle.load(f)
                self.doc_ids = data["doc_ids"]
                self.doc_tokenized = data["doc_tokenized"]
                self.bm25 = BM25Okapi(self.doc_tokenized)
                logger.info(f"Loaded BM25 index with {len(self.doc_ids)} docs")
                return
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")

        logger.info("Building BM25 index from Chroma collection...")
        all_docs = self.collection.get(include=["documents", "metadatas"])
        if not all_docs["ids"]:
            logger.warning("No documents in collection, BM25 index empty")
            self.bm25 = None
            return

        self.doc_ids = all_docs["ids"]
        self.doc_texts = all_docs["documents"]
        self.doc_metadatas = all_docs["metadatas"]
        self.doc_tokenized = [self._tokenize(doc) for doc in self.doc_texts]
        self.bm25 = BM25Okapi(self.doc_tokenized)

        with open(self.bm25_index_path, "wb") as f:
            pickle.dump({
                "doc_ids": self.doc_ids,
                "doc_tokenized": self.doc_tokenized,
            }, f)
        logger.info(f"Built and saved BM25 index with {len(self.doc_ids)} docs")

    # ------------------------------------------------------------------
    # Scoring components
    # ------------------------------------------------------------------
    def _weighted_skill_score(
        self,
        query_skills: Set[str],
        candidate_skills_str: str,
        skill_priority: Optional[Dict[str, float]] = None,
    ) -> float:
        cand_skills = set(s.strip().lower() for s in candidate_skills_str.split(",") if s.strip())
        query_skills_lower = {s.lower() for s in query_skills}
        if not query_skills_lower or not cand_skills:
            return 0.0
        matched = query_skills_lower.intersection(cand_skills)
        if not matched:
            return 0.0
        if skill_priority:
            total_weight = sum(skill_priority.get(s, 1.0) for s in query_skills)
            matched_weight = sum(skill_priority.get(s, 1.0) for s in query_skills if s.lower() in matched)
            return matched_weight / total_weight
        else:
            union = query_skills_lower.union(cand_skills)
            return len(matched) / len(union)

    def _experience_score(self, query_exp: Optional[float], candidate_exp: float) -> float:
        if query_exp is None or query_exp <= 0:
            return 0.0
        if candidate_exp <= 0:
            return 0.0
        return min(candidate_exp / query_exp, 1.0)

    def _education_score(self, required_edu: Optional[str], candidate_edu: Optional[str]) -> float:
        if not required_edu:
            return 1.0
        if not candidate_edu:
            return 0.0
        levels = ["bachelor", "master", "phd"]
        if required_edu not in levels or candidate_edu not in levels:
            return 0.0
        req_idx = levels.index(required_edu)
        cand_idx = levels.index(candidate_edu)
        if cand_idx >= req_idx:
            return 1.0
        else:
            return 0.5

    def _seniority_score(self, required_seniority: Optional[str], candidate_seniority: Optional[str]) -> float:
        if not required_seniority:
            return 1.0
        if not candidate_seniority:
            return 0.0
        levels = ["junior", "mid", "senior", "lead", "principal"]
        if required_seniority not in levels or candidate_seniority not in levels:
            return 0.0
        req_idx = levels.index(required_seniority)
        cand_idx = levels.index(candidate_seniority)
        if cand_idx >= req_idx:
            return 1.0
        else:
            return 0.5

    # ------------------------------------------------------------------
    # Advanced filter application
    # ------------------------------------------------------------------
    def _apply_filters(
        self,
        candidates: List[Dict],
        filter_skills: Optional[Set[str]] = None,
        filter_min_exp: Optional[float] = None,
        filter_max_exp: Optional[float] = None,
        filter_edu: Optional[str] = None,
        filter_seniority: Optional[str] = None,
    ) -> List[Dict]:
        filtered = []
        for cand in candidates:
            meta = cand["metadata"]
            if filter_skills:
                cand_skills_str = meta.get("skills", "")
                cand_skills = set(s.strip().lower() for s in cand_skills_str.split(",") if s.strip())
                required_lower = {s.lower() for s in filter_skills}
                if not required_lower.issubset(cand_skills):
                    continue
            cand_exp = meta.get("years_experience", 0.0)
            if filter_min_exp is not None and cand_exp < filter_min_exp:
                continue
            if filter_max_exp is not None and cand_exp > filter_max_exp:
                continue
            if filter_edu is not None:
                cand_edu = meta.get("education_level")
                if cand_edu != filter_edu:
                    continue
            if filter_seniority is not None:
                cand_sen = meta.get("seniority")
                if cand_sen != filter_seniority:
                    continue
            filtered.append(cand)
        return filtered

    # ------------------------------------------------------------------
    # MMR diversity
    # ------------------------------------------------------------------
    def _mmr_diverse_rerank(
        self,
        candidates: List[Dict],
        query_embedding: List[float],
        lambda_param: float = 0.7,
    ) -> List[Dict]:
        if len(candidates) <= 1:
            return candidates

        ids = [c["id"] for c in candidates]
        fetched = self.collection.get(ids=ids, include=["embeddings"])
        id_to_embedding = {fetched["ids"][i]: fetched["embeddings"][i] for i in range(len(fetched["ids"]))}

        selected = []
        remaining = list(range(len(candidates)))

        first_idx = max(remaining, key=lambda i: candidates[i]["score"])
        selected.append(first_idx)
        remaining.remove(first_idx)

        while remaining and len(selected) < len(candidates):
            mmr_scores = []
            for i in remaining:
                rel = candidates[i]["score"]
                emb_i = id_to_embedding[candidates[i]["id"]]
                sim_max = 0.0
                for j in selected:
                    emb_j = id_to_embedding[candidates[j]["id"]]
                    sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-9)
                    sim_max = max(sim_max, sim)
                mmr = lambda_param * rel - (1 - lambda_param) * sim_max
                mmr_scores.append((i, mmr))
            next_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(next_idx)
            remaining.remove(next_idx)

        return [candidates[i] for i in selected]

    # ------------------------------------------------------------------
    # Main hybrid search with embedding cache and email decryption
    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        min_experience: Optional[float] = None,
        desired_experience: Optional[float] = None,
        query_skills: Optional[Set[str]] = None,
        skill_priority: Optional[Dict[str, float]] = None,
        required_education: Optional[str] = None,
        required_seniority: Optional[str] = None,
        use_rerank: bool = True,
        diversity_lambda: Optional[float] = None,
        filter_skills: Optional[Set[str]] = None,
        filter_min_exp: Optional[float] = None,
        filter_max_exp: Optional[float] = None,
        filter_edu: Optional[str] = None,
        filter_seniority: Optional[str] = None,
        expand_query: bool = True,
        boolean_query: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        # Parse boolean query once if provided
        boolean_filter: Optional[Callable[[Set[str]], bool]] = None
        if boolean_query:
            try:
                parser = BooleanQueryParser(boolean_query)
                boolean_filter = parser.evaluate
                logger.info(f"Parsed boolean query: {boolean_query}")
            except Exception as e:
                logger.error(f"Failed to parse boolean query '{boolean_query}': {e}")
                raise ValueError(f"Invalid boolean expression: {boolean_query}")

        # Generate cache key for result caching
        cache_key = None
        if use_cache:
            cache_key = self._cache_key(
                query, top_k, alpha, min_experience, desired_experience,
                query_skills, skill_priority, required_education, required_seniority,
                use_rerank, diversity_lambda, filter_skills, filter_min_exp,
                filter_max_exp, filter_edu, filter_seniority, expand_query, boolean_query
            )
            # Try Redis first if available
            if self.redis_client:
                try:
                    cached = self.redis_client.get(cache_key)
                    if cached:
                        self._cache_hits += 1
                        logger.debug(f"Redis cache hit for key {cache_key[:8]} (hits={self._cache_hits})")
                        return pickle.loads(cached)
                except Exception as e:
                    logger.warning(f"Redis error, falling back to memory cache: {e}")
            # Fallback to in‑memory
            if cache_key in self._cache:
                self._cache_hits += 1
                logger.debug(f"Memory cache hit for key {cache_key[:8]} (hits={self._cache_hits}, misses={self._cache_misses})")
                return copy.deepcopy(self._cache[cache_key])

        self._cache_misses += 1

        # Expand query if enabled
        final_query = query
        if expand_query:
            original_query = query
            final_query = self.expand_query(query)
            if final_query != original_query:
                logger.info(f"Expanded query: {original_query} -> {final_query}")

        # Embedding cache: use final_query as key
        embed_cache_key = f"embed:{final_query}"
        if embed_cache_key in self._embedding_cache:
            query_embedding = self._embedding_cache[embed_cache_key]
            logger.debug(f"Embedding cache hit for query: {final_query[:50]}")
        else:
            query_embedding = self.embedder.encode(final_query, convert_to_numpy=True).tolist()
            self._embedding_cache[embed_cache_key] = query_embedding
            logger.debug(f"Embedding cache miss for query: {final_query[:50]}")

        self.build_bm25_index()

        if query_skills is None:
            query_skills = self.extract_skills_from_text(final_query)

        # Vector search (now using cached embedding)
        initial_k = config.RERANK_TOP_K if use_rerank and self.cross_encoder else top_k
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_k * 2,
            include=["metadatas", "distances", "documents"]
        )

        # BM25 scores (use final_query for BM25)
        bm25_scores = None
        if self.bm25 and self.doc_ids:
            tokenized_query = self._tokenize(final_query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()
            else:
                bm25_scores = np.zeros_like(bm25_scores)

        # Build results map
        results_map = {}
        if vector_results["ids"] and vector_results["ids"][0]:
            for i, doc_id in enumerate(vector_results["ids"][0]):
                distance = vector_results["distances"][0][i]
                vector_score = 1.0 - min(distance, 1.0)
                metadata = vector_results["metadatas"][0][i]

                # Hard filter: min_experience
                if min_experience is not None:
                    cand_exp = metadata.get("years_experience", 0.0)
                    if cand_exp < min_experience:
                        continue

                results_map[doc_id] = {
                    "id": doc_id,
                    "metadata": metadata,
                    "document": vector_results["documents"][0][i],
                    "vector_score": vector_score,
                }

        if bm25_scores is not None:
            for idx, doc_id in enumerate(self.doc_ids):
                if doc_id in results_map:
                    results_map[doc_id]["bm25_score"] = bm25_scores[idx]

        # Convert to list
        candidates = list(results_map.values())

        # Apply advanced filters
        if filter_skills or filter_min_exp is not None or filter_max_exp is not None or filter_edu or filter_seniority:
            candidates = self._apply_filters(
                candidates,
                filter_skills=filter_skills,
                filter_min_exp=filter_min_exp,
                filter_max_exp=filter_max_exp,
                filter_edu=filter_edu,
                filter_seniority=filter_seniority,
            )

        # Apply boolean filter if present
        if boolean_filter:
            filtered = []
            for cand in candidates:
                skills_str = cand["metadata"].get("skills", "")
                skills_set = set(s.strip().lower() for s in skills_str.split(",") if s.strip())
                if boolean_filter(skills_set):
                    filtered.append(cand)
            candidates = filtered

        # Compute scores for each candidate
        for data in candidates:
            vector = data.get("vector_score", 0.0)
            bm25 = data.get("bm25_score", 0.0)

            cand_skills_str = data["metadata"].get("skills", "")
            skill_score = self._weighted_skill_score(query_skills, cand_skills_str, skill_priority)

            cand_exp = data["metadata"].get("years_experience", 0.0)
            exp_score = self._experience_score(desired_experience, cand_exp)

            cand_edu = data["metadata"].get("education_level")
            edu_score = self._education_score(required_education, cand_edu)

            cand_sen = data["metadata"].get("seniority")
            sen_score = self._seniority_score(required_seniority, cand_sen)

            recency = 1.0

            weights = config.SCORE_WEIGHTS
            final_score = (
                weights["semantic"] * vector +
                weights["skill_overlap"] * skill_score +
                weights["experience"] * exp_score +
                weights["education"] * edu_score +
                weights["seniority"] * sen_score +
                weights["bm25"] * bm25 +
                weights["recency"] * recency
            )

            cand_skills_lower = set(s.strip().lower() for s in cand_skills_str.split(",") if s.strip())
            query_skills_lower = {s.lower() for s in query_skills}
            missing_skills = list(query_skills_lower - cand_skills_lower) if query_skills_lower else []

            fit_breakdown = {
                "semantic": round(vector, 3),
                "skill_overlap": round(skill_score, 3),
                "experience": round(exp_score, 3),
                "education": round(edu_score, 3),
                "seniority": round(sen_score, 3),
                "bm25": round(bm25, 3),
                "recency": round(recency, 3),
            }

            data["fit_breakdown"] = fit_breakdown
            data["missing_skills"] = missing_skills
            data["final_score"] = final_score

        # Sort
        candidates.sort(key=lambda x: x["final_score"], reverse=True)

        # Rerank
        if use_rerank and self.cross_encoder and candidates:
            rerank_candidates = candidates[:config.RERANK_TOP_K]
            pairs = [(final_query, cand["document"]) for cand in rerank_candidates]
            rerank_scores = self.cross_encoder.predict(pairs)

            if len(rerank_scores) > 1 and rerank_scores.max() > rerank_scores.min():
                rerank_scores = (rerank_scores - rerank_scores.min()) / (rerank_scores.max() - rerank_scores.min())
            else:
                rerank_scores = np.where(rerank_scores > 0, 1.0, 0.0)

            for cand, score in zip(rerank_candidates, rerank_scores):
                cand["rerank_score"] = float(score)
                cand["score"] = float(score)
            remaining = candidates[config.RERANK_TOP_K:]
            for cand in remaining:
                cand["score"] = cand["final_score"]
            final_candidates = rerank_candidates + remaining
        else:
            for cand in candidates:
                cand["score"] = cand["final_score"]
            final_candidates = candidates

        # Diversity
        if diversity_lambda is not None and len(final_candidates) > 1:
            final_candidates = self._mmr_diverse_rerank(final_candidates, query_embedding, diversity_lambda)

        final_candidates.sort(key=lambda x: x["score"], reverse=True)
        results = final_candidates[:top_k]

        # Decrypt email addresses in results before caching and returning
        for res in results:
            if "metadata" in res and "email" in res["metadata"]:
                encrypted = res["metadata"]["email"]
                res["metadata"]["email"] = decrypt(encrypted)

        # Cache results (both Redis and in‑memory)
        if use_cache and cache_key is not None:
            result_serialized = pickle.dumps(results)
            if self.redis_client:
                try:
                    self.redis_client.setex(cache_key, self.redis_ttl, result_serialized)
                except Exception as e:
                    logger.warning(f"Redis set failed: {e}")
            self._cache[cache_key] = copy.deepcopy(results)

        return results