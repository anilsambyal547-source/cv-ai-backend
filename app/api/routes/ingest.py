"""
Ingestion endpoint – upload PDF and index with duplicate detection.
"""
import tempfile
import hashlib
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
import logging

from app.ingest.extractor import extract_structured_data
from app.ingest.indexer import index_cv
from app.storage.vector_store import vector_store
from app.api.deps import get_current_active_user
from app.utils.audit import log_access

router = APIRouter()
logger = logging.getLogger(__name__)


def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def find_duplicate(file_hash: str) -> bool:
    """Check if a CV with this hash already exists in ChromaDB."""
    try:
        result = vector_store.collection.get(include=["metadatas"])
        for meta in result.get("metadatas") or []:
            if meta.get("_hash") == file_hash:
                return True
    except Exception:
        pass
    return False

# Helper to process CV in background
def process_cv_background(tmp_path: str, filename: str, content: bytes, user_username: str, storage_path: Path):
    try:
        # Extract
        extracted = extract_structured_data(tmp_path)
        
        if not extracted:
            logger.warning(f"Background: No data from {filename}")
            return

        # Save original file permanently
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(storage_path, "wb") as f:
            f.write(content)

        # Index with the file path
        base_id = Path(filename).stem
        indexed = index_cv(base_id, extracted, file_path=str(storage_path))

        log_access(
            user=user_username,
            action="INGEST_BACKGROUND",
            resource="PDF",
            details={
                "filename": filename,
                "candidates": len(extracted),
                "indexed": indexed,
            },
        )
    except Exception as e:
        logger.error(f"Background extraction failed for {filename}: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/ingest", summary="Upload and index a CV PDF")
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user=Depends(get_current_active_user),
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(content) > 30 * 1024 * 1024:  # 30MB limit
        raise HTTPException(status_code=400, detail="File too large (max 30MB)")

    # Duplicate detection
    file_hash = compute_file_hash(content)
    if find_duplicate(file_hash):
        return {
            "status":     "duplicate",
            "filename":   file.filename,
            "message":    "This CV is already in the database",
            "skipped":    True,
        }

    # Prepare persistent storage path
    storage_dir = Path("data/cv_files")
    storage_path = storage_dir / f"{file_hash}.pdf"

    # Save to temp file for extraction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Add processing to background
    background_tasks.add_task(
        process_cv_background, 
        tmp_path, 
        file.filename, 
        content, 
        user["username"], 
        storage_path
    )

    return {
        "status":     "processing",
        "filename":   file.filename,
        "message":    "Ingestion started in background. Your dashboard will update soon.",
    }


@router.get("/ingest/stats", summary="Get database ingestion stats")
async def ingest_stats(user=Depends(get_current_active_user)):
    """Returns stats about what's in the database."""
    try:
        result = vector_store.collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        ids       = result.get("ids") or []

        total      = len(metadatas)
        needs_review = 0
        industries: dict = {}
        seniorities: dict = {}

        for meta in metadatas:
            conf = float(meta.get("_confidence", 1) or 1)
            if conf < 0.5:
                needs_review += 1
            ind = meta.get("industry", "general") or "general"
            industries[ind] = industries.get(ind, 0) + 1
            sen = meta.get("seniority", "unknown") or "unknown"
            seniorities[sen] = seniorities.get(sen, 0) + 1

        return {
            "total":        total,
            "needs_review": needs_review,
            "industries":   industries,
            "seniorities":  seniorities,
            "ids":          ids,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/ingest/{candidate_id}", summary="Delete a candidate from the database")
async def delete_candidate(
    candidate_id: str,
    user=Depends(get_current_active_user),
):
    """Delete a candidate by ID from ChromaDB."""
    try:
        result = vector_store.collection.get(ids=[candidate_id])
        if not result["ids"]:
            raise HTTPException(status_code=404, detail="Candidate not found")

        vector_store.collection.delete(ids=[candidate_id])

        log_access(
            user=user["username"],
            action="DELETE_CANDIDATE",
            resource="CANDIDATE",
            details={"candidate_id": candidate_id},
        )

        return {"message": "Candidate deleted successfully", "id": candidate_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))