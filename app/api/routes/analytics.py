"""
Analytics & Data Intelligence API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from collections import Counter, defaultdict

from ...analytics import (
    enrich_candidate,
    compute_skill_vector,
    compute_tech_clusters,
    cosine_similarity,
)
from ..deps import get_current_active_user
from ...storage.vector_store import vector_store

router = APIRouter()

def _get_collection():
    return vector_store.collection

# ── Overview ──────────────────────────────────────────────────────────────────

@router.get("/analytics/overview")
async def analytics_overview(user=Depends(get_current_active_user)):
    """Aggregated stats: seniority, industry, top skills, clusters, avg exp."""
    try:
        result    = _get_collection().get(include=["metadatas"])
        metadatas = result.get("metadatas") or []

        if not metadatas:
            return {
                "total_candidates":     0,
                "avg_experience":       0,
                "seniority_dist":       {},
                "industry_dist":        {},
                "top_skills":           [],
                "cluster_dist":         {},
                "experience_histogram": [],
            }

        seniority_counter: Counter = Counter()
        industry_counter:  Counter = Counter()
        skill_counter:     Counter = Counter()
        cluster_totals:    dict    = defaultdict(float)
        total_exp = 0.0
        exp_buckets = {"0-1": 0, "1-3": 0, "3-5": 0, "5-8": 0, "8-12": 0, "12+": 0}

        for meta in metadatas:
            enriched = enrich_candidate(dict(meta))
            years    = float(meta.get("years_experience", 0) or 0)
            total_exp += years

            seniority_counter[enriched.get("seniority", "unknown")] += 1
            industry_counter[enriched.get("industry",  "general")]  += 1

            skills_raw = meta.get("skills", "") or ""
            skills = [s.strip() for s in skills_raw.split(",") if s.strip()]
            skill_counter.update(skills)

            for cluster, score in enriched.get("tech_clusters", {}).items():
                cluster_totals[cluster] += score

            if   years < 1:  exp_buckets["0-1"]  += 1
            elif years < 3:  exp_buckets["1-3"]  += 1
            elif years < 5:  exp_buckets["3-5"]  += 1
            elif years < 8:  exp_buckets["5-8"]  += 1
            elif years < 12: exp_buckets["8-12"] += 1
            else:            exp_buckets["12+"]  += 1

        n = len(metadatas)
        return {
            "total_candidates":     n,
            "avg_experience":       round(total_exp / n, 1),
            "seniority_dist":       dict(seniority_counter.most_common()),
            "industry_dist":        dict(industry_counter.most_common()),
            "top_skills":           [{"skill": s, "count": c} for s, c in skill_counter.most_common(20)],
            "cluster_dist":         {k: round(v / n, 2) for k, v in sorted(cluster_totals.items(), key=lambda x: x[1], reverse=True)},
            "experience_histogram": [{"range": k, "count": v} for k, v in exp_buckets.items()],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Similar Candidates ────────────────────────────────────────────────────────

@router.get("/analytics/similar/{candidate_id}")
async def similar_candidates(
    candidate_id: str,
    limit: int = Query(5, ge=1, le=20),
    user=Depends(get_current_active_user),
):
    try:
        result    = _get_collection().get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        ids       = result.get("ids") or []

        target_meta = next((m for i, m in zip(ids, metadatas) if i == candidate_id), None)
        if not target_meta:
            raise HTTPException(status_code=404, detail="Candidate not found")

        skills_raw    = target_meta.get("skills", "") or ""
        target_skills = [s.strip() for s in skills_raw.split(",") if s.strip()]
        target_vec    = compute_skill_vector(target_skills)

        scored = []
        for cid, meta in zip(ids, metadatas):
            if cid == candidate_id:
                continue
            s_raw  = meta.get("skills", "") or ""
            s_list = [s.strip() for s in s_raw.split(",") if s.strip()]
            sim    = cosine_similarity(target_vec, compute_skill_vector(s_list))
            if sim > 0:
                scored.append({
                    "id":         cid,
                    "name":       meta.get("name", "Unknown"),
                    "similarity": sim,
                    "skills":     s_raw,
                    "seniority":  meta.get("seniority", ""),
                })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return {"candidate_id": candidate_id, "similar": scored[:limit]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Tech Clusters ─────────────────────────────────────────────────────────────

@router.get("/analytics/clusters")
async def tech_clusters(user=Depends(get_current_active_user)):
    try:
        result    = _get_collection().get(include=["metadatas"])
        metadatas = result.get("metadatas") or []

        cluster_counts: dict = defaultdict(int)
        for meta in metadatas:
            skills_raw = meta.get("skills", "") or ""
            skills     = [s.strip() for s in skills_raw.split(",") if s.strip()]
            for cluster in compute_tech_clusters(skills):
                cluster_counts[cluster] += 1

        return {
            "clusters": [
                {"name": k, "count": v}
                for k, v in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Enrich Single Candidate ───────────────────────────────────────────────────

@router.get("/analytics/enrich/{candidate_id}")
async def enrich_single(
    candidate_id: str,
    user=Depends(get_current_active_user),
):
    try:
        result    = _get_collection().get(ids=[candidate_id], include=["metadatas", "documents"])
        metadatas = result.get("metadatas") or []

        if not metadatas:
            raise HTTPException(status_code=404, detail="Candidate not found")

        meta            = dict(metadatas[0])
        docs            = result.get("documents") or [""]
        meta["document"] = docs[0] if docs else ""

        enriched = enrich_candidate(meta)
        enriched.pop("skill_vector", None)
        return enriched

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))