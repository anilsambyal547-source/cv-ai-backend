"""
Search and match endpoints.
"""
from typing import Optional, List, Dict, Any, Set
from fastapi import APIRouter, Depends, HTTPException, Query

from app.retrieval.retriever import HybridCVRetriever
from app.storage.vector_store import vector_store
from app.jd_parser import parse_job_description
from app.api.deps import get_current_active_user
from app.utils.audit import log_access
from config import config

router = APIRouter()

# Initialize retriever once (singleton in API)
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        # Use same retriever as CLI
        from sentence_transformers import CrossEncoder
        cross_encoder = None
        try:
            cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
        except Exception:
            pass
        # Attempt Redis only if configured (short timeout)
        redis_client = None
        if config.REDIS_HOST and config.REDIS_HOST not in ("localhost", "redis"):
            try:
                import redis
                redis_client = redis.Redis(
                    host=config.REDIS_HOST,
                    port=config.REDIS_PORT,
                    db=config.REDIS_DB,
                    password=config.REDIS_PASSWORD,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    decode_responses=False,
                )
                redis_client.ping()
            except Exception:
                redis_client = None
        _retriever = HybridCVRetriever(
            collection=vector_store.collection,
            embedder=vector_store.embedder,
            bm25_index_path=config.BM25_INDEX_PATH,
            cross_encoder=cross_encoder,
            redis_client=redis_client,
        )
        _retriever.build_bm25_index()
    return _retriever

@router.get("/search", summary="Search candidates by query")
async def search(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100),
    min_exp: Optional[float] = Query(None, description="Minimum experience"),
    filter_skills: Optional[str] = Query(None, description="Comma-separated required skills"),
    filter_edu: Optional[str] = Query(None, description="Required education level"),
    filter_seniority: Optional[str] = Query(None, description="Required seniority"),
    boolean: Optional[str] = Query(None, description="Boolean expression (e.g., python AND docker)"),
    user = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    """Hybrid search with optional filters. Accessible by any authenticated user."""
    filter_skills_set = None
    if filter_skills:
        filter_skills_set = {s.strip() for s in filter_skills.split(',') if s.strip()}

    retriever = get_retriever()
    results = retriever.hybrid_search(
        query=query,
        top_k=limit,
        alpha=0.5,
        min_experience=min_exp,
        filter_skills=filter_skills_set,
        filter_edu=filter_edu,
        filter_seniority=filter_seniority,
        boolean_query=boolean,
        use_rerank=True,
        expand_query=True,
    )

    # Log audit event
    log_access(
        user=user["username"],
        action="SEARCH",
        resource="QUERY",
        details={
            "query": query,
            "limit": limit,
            "result_count": len(results),
        }
    )

    return results

@router.post("/match", summary="Match candidates to a job description")
async def match(
    jd_text: str = Query(..., description="Job description text"),
    limit: int = Query(10, ge=1, le=100),
    user = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    """Parse job description and return ranked candidates."""
    parsed = parse_job_description(jd_text)
    skill_priority = {skill: 1.0 for skill in parsed['skills']} if parsed['skills'] else None

    retriever = get_retriever()
    results = retriever.hybrid_search(
        query=jd_text,
        top_k=limit,
        alpha=0.3,
        desired_experience=parsed['min_experience'],
        query_skills=parsed['skills'],
        skill_priority=skill_priority,
        required_education=parsed['education'],
        use_rerank=True,
    )

    # Log audit event
    log_access(
        user=user["username"],
        action="MATCH",
        resource="JOB_DESCRIPTION",
        details={
            "jd_length": len(jd_text),
            "skills_found": list(parsed['skills']),
            "result_count": len(results),
        }
    )

    return results