"""
Candidate endpoints: list, detail, and AI insights.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException

from app.storage.vector_store import vector_store
from app.api.deps import get_current_active_user
from app.utils.audit import log_access
from app.utils.encryption import decrypt

router = APIRouter()

# ── Common skills/frameworks that should never be a candidate name ──
KNOWN_SKILLS = {
    "python", "javascript", "typescript", "java", "golang", "rust", "php",
    "ruby", "swift", "kotlin", "scala", "react", "angular", "vue", "django",
    "flask", "fastapi", "spring", "node", "express", "nextjs", "nuxtjs",
    "docker", "kubernetes", "aws", "gcp", "azure", "terraform", "jenkins",
    "postgresql", "mongodb", "redis", "mysql", "sqlite", "elasticsearch",
    "git", "linux", "html", "css", "graphql", "rest", "grpc", "kafka",
    "spark", "hadoop", "pytorch", "tensorflow", "scikit", "pandas", "numpy",
}

def safe_decrypt_email(email: str) -> str:
    """Decrypt email if encrypted, otherwise return as-is."""
    if not email:
        return ""
    # Fernet tokens start with 'gAAAAA'
    if email.startswith("gAAAAA"):
        try:
            return decrypt(email)
        except Exception:
            return ""
    return email

def is_valid_name(name: str) -> bool:
    """Check if extracted name is actually a person's name."""
    if not name or len(name.strip()) < 2:
        return False
    name_lower = name.lower().strip()
    # Reject if it's a known skill/framework
    if name_lower in KNOWN_SKILLS:
        return False
    # Reject if it's a single word that looks like a technology
    words = name.strip().split()
    if len(words) == 1 and name_lower in KNOWN_SKILLS:
        return False
    # A real name should have at least 2 characters and be mostly letters
    if not all(c.isalpha() or c in " .-'" for c in name.strip()):
        return False
    return True

def compute_quality_score(meta: dict) -> dict:
    """Score candidate data quality 0-100 and flag issues."""
    score  = 0
    issues = []

    name  = meta.get("name", "")
    email = meta.get("email", "")
    phone = meta.get("phone", "")
    skills_raw = meta.get("skills", "")
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else []
    years  = float(meta.get("years_experience", 0) or 0)

    # Name check (20 pts)
    if is_valid_name(name):
        score += 20
    else:
        issues.append("Invalid or missing name — may be a skill or framework")

    # Email check (20 pts)
    decrypted_email = safe_decrypt_email(email)
    if decrypted_email and "@" in decrypted_email:
        score += 20
    else:
        issues.append("Missing or invalid email")

    # Phone check (10 pts)
    if phone and len(phone) >= 7:
        score += 10
    else:
        issues.append("Missing phone number")

    # Skills check (25 pts)
    if len(skills) >= 5:
        score += 25
    elif len(skills) >= 2:
        score += 15
        issues.append("Few skills extracted — may be incomplete")
    else:
        issues.append("No skills extracted")

    # Experience check (15 pts)
    if years > 0:
        score += 15
    else:
        issues.append("No experience data")

    # Summary check (10 pts)
    if meta.get("summary"):
        score += 10

    # Quality label
    if score >= 80:
        quality = "excellent"
    elif score >= 60:
        quality = "good"
    elif score >= 40:
        quality = "fair"
    else:
        quality = "poor"

    return {
        "quality_score": score,
        "quality_label": quality,
        "quality_issues": issues,
        "needs_review":   score < 50,
    }


@router.get("/candidates", summary="List all candidates with pagination")
async def list_candidates(
    limit:  int = Query(20, ge=1, le=100),
    offset: int = Query(0,  ge=0),
    user=Depends(get_current_active_user),
) -> List[dict]:
    results = vector_store.collection.get(
        limit=limit,
        offset=offset,
        include=["metadatas"],
    )
    if not results["ids"]:
        return []

    candidates = []
    for i, doc_id in enumerate(results["ids"]):
        meta   = results["metadatas"][i]
        skills = meta.get("skills", "").split(",")
        skills_preview = ", ".join(s.strip() for s in skills[:3] if s.strip())

        quality = compute_quality_score(meta)
        name    = meta.get("name", "Unknown")
        if not is_valid_name(name):
            name = "Unknown (extraction failed)"

        candidates.append({
            "id":               doc_id,
            "name":             name,
            "email":            safe_decrypt_email(meta.get("email", "")),
            "phone":            meta.get("phone", ""),
            "location":         meta.get("location", ""),
            "years_experience": meta.get("years_experience", 0),
            "seniority":        meta.get("seniority", ""),
            "education_level":  meta.get("education_level", ""),
            "industry":         meta.get("industry", ""),
            "skills":           meta.get("skills", ""),
            "skills_preview":   skills_preview,
            "stage":            meta.get("stage", "Lead"),
            **quality,
        })

    # Sort by quality score descending
    candidates.sort(key=lambda x: x["quality_score"], reverse=True)

    log_access(
        user=user["username"],
        action="LIST_CANDIDATES",
        resource="CANDIDATE",
        details={"limit": limit, "offset": offset, "returned": len(candidates)},
    )
    return candidates


@router.get("/candidate/{candidate_id}", summary="Get full candidate profile")
async def get_candidate(
    candidate_id: str,
    user=Depends(get_current_active_user),
):
    result = vector_store.collection.get(
        ids=[candidate_id],
        include=["metadatas", "documents"],
    )
    if not result["ids"]:
        raise HTTPException(status_code=404, detail="Candidate not found")

    meta = result["metadatas"][0]
    doc  = result["documents"][0] if result.get("documents") else ""

    # Decrypt email
    decrypted_email = safe_decrypt_email(meta.get("email", ""))

    # Parse skills
    skills_raw = meta.get("skills", "") or ""
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

    # Parse work experience
    work_exp_raw = meta.get("work_experience", "") or ""
    work_experience = []
    if isinstance(work_exp_raw, list):
        work_experience = work_exp_raw
    elif isinstance(work_exp_raw, str) and work_exp_raw:
        for line in work_exp_raw.split(";"):
            line = line.strip()
            if line:
                work_experience.append({"description": line})

    # Parse education
    edu_raw = meta.get("education", "") or ""
    education = []
    if isinstance(edu_raw, list):
        education = edu_raw
    elif isinstance(edu_raw, str) and edu_raw:
        education = [e.strip() for e in edu_raw.split(";") if e.strip()]

    # Parse certifications
    cert_raw = meta.get("certifications", "") or ""
    certifications = []
    if isinstance(cert_raw, list):
        certifications = cert_raw
    elif isinstance(cert_raw, str) and cert_raw:
        certifications = [c.strip() for c in cert_raw.split(",") if c.strip()]

    # Validate name
    name = meta.get("name", "Unknown")
    if not is_valid_name(name):
        name = "Unknown (extraction failed)"

    # Quality score
    quality = compute_quality_score(meta)

    log_access(
        user=user["username"],
        action="VIEW_CANDIDATE",
        resource="CANDIDATE",
        details={"candidate_id": candidate_id},
    )

    return {
        "id":               candidate_id,
        "name":             name,
        "email":            decrypted_email,
        "phone":            meta.get("phone", ""),
        "location":         meta.get("location", ""),
        "years_experience": meta.get("years_experience", 0),
        "seniority":        meta.get("seniority", ""),
        "education_level":  meta.get("education_level", ""),
        "industry":         meta.get("industry", ""),
        "summary":          meta.get("summary", ""),
        "skills":           skills,
        "work_experience":  work_experience,
        "education":        education,
        "certifications":   certifications,
        "document":         doc,
        "has_file":         bool(meta.get("file_path")),
        "stage":            meta.get("stage", "Lead"),
        **quality,
    }


@router.get("/candidate/{candidate_id}/cv", summary="Serve original CV PDF")
async def serve_candidate_cv(
    candidate_id: str,
    user=Depends(get_current_active_user),
):
    from fastapi.responses import FileResponse
    from pathlib import Path

    result = vector_store.collection.get(
        ids=[candidate_id],
        include=["metadatas"],
    )
    if not result["ids"]:
        raise HTTPException(status_code=404, detail="Candidate not found")

    meta = result["metadatas"][0]
    file_path = meta.get("file_path")

    if not file_path:
        raise HTTPException(status_code=404, detail="Original CV file not available for this candidate")

    p = Path(file_path)
    if not p.exists():
         raise HTTPException(status_code=404, detail="File lost on server storage")

    return FileResponse(
        path=p,
        media_type="application/pdf",
        filename=f"CV_{candidate_id}.pdf"
    )


@router.post("/candidate/{candidate_id}/insights", summary="Generate AI insights")
async def generate_candidate_insights(
    candidate_id: str,
    body: dict = {},
    user=Depends(get_current_active_user),
):
    result = vector_store.collection.get(
        ids=[candidate_id],
        include=["metadatas", "documents"],
    )
    if not result["ids"]:
        raise HTTPException(status_code=404, detail="Candidate not found")

    meta             = dict(result["metadatas"][0])
    doc              = result["documents"][0] if result.get("documents") else ""
    meta["document"] = doc
    meta["email"]    = safe_decrypt_email(meta.get("email", ""))

    job_description = body.get("job_description", None)

    try:
        from app.insights.generator import generate_insights
        insights = generate_insights(meta, job_description)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Insights generation failed: " + str(e)
        )

    log_access(
        user=user["username"],
        action="GENERATE_INSIGHTS",
        resource="CANDIDATE",
        details={"candidate_id": candidate_id},
    )
    return insights


@router.patch("/candidate/{candidate_id}/stage", summary="Update candidate recruitment stage")
async def update_candidate_stage(
    candidate_id: str,
    body: dict,
    user=Depends(get_current_active_user),
):
    """Updates the 'stage' metadata for a candidate."""
    new_stage = body.get("stage")
    if not new_stage:
        raise HTTPException(status_code=400, detail="Missing 'stage' in request body")

    valid_stages = ["Lead", "Shortlisted", "Outreach", "Applied", "Screening", "Interview", "On Hold", "Offer", "Rejected", "Hired", "Contacted"]
    if new_stage not in valid_stages:
        raise HTTPException(status_code=400, detail=f"Invalid stage. Must be one of: {', '.join(valid_stages)}")

    # Fetch existing metadata to ensure we don't lose other fields
    result = vector_store.collection.get(ids=[candidate_id], include=["metadatas"])
    if not result["ids"]:
        raise HTTPException(status_code=404, detail="Candidate not found")

    meta = dict(result["metadatas"][0])
    meta["stage"] = new_stage

    # Update ChromaDB
    vector_store.collection.update(
        ids=[candidate_id],
        metadatas=[meta],
    )

    log_access(
        user=user["username"],
        action="UPDATE_CANDIDATE_STAGE",
        resource="CANDIDATE",
        details={"candidate_id": candidate_id, "stage": new_stage},
    )

    return {"message": "Candidate stage updated", "id": candidate_id, "stage": new_stage}