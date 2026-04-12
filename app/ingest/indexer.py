"""
Index CV records into Chroma with deduplication and skill normalization.
"""
import logging
import hashlib
from typing import List, Dict, Any
from app.utils.encryption import encrypt
from tqdm import tqdm
from app.storage.vector_store import vector_store

logger = logging.getLogger("cv_indexer")

BATCH_SIZE = 32


def _build_embedding_text(record: Dict[str, Any]) -> str:
    """Build a single text string from normalized fields for embedding."""
    parts = [
        record.get("name", ""),
        record.get("email", ""),
        record.get("phone", ""),
        " ".join(record.get("skills", [])) if isinstance(record.get("skills"), list) else record.get("skills", ""),
        record.get("summary", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def _is_duplicate(collection, record_hash: str) -> bool:
    """Check if a record with this hash already exists in the collection."""
    try:
        results = collection.get(where={"_hash": record_hash})
        return len(results["ids"]) > 0
    except Exception:
        return False


def _serialize_work_experience(work_exp) -> str:
    """Serialize work experience list to string for ChromaDB storage."""
    if not work_exp:
        return ""
    if isinstance(work_exp, str):
        return work_exp
    if isinstance(work_exp, list):
        parts = []
        for w in work_exp:
            if not isinstance(w, dict):
                continue
            title   = w.get("job_title", "")
            company = w.get("company", "")
            start   = w.get("start_date", "")
            end     = w.get("end_date", "")
            desc    = w.get("description", "")

            entry = ""
            if title and company:
                entry = title + " at " + company
            elif title:
                entry = title
            elif company:
                entry = company

            if start:
                entry += " (" + start + ("-" + end if end else "") + ")"

            if desc:
                entry += ": " + desc[:200]

            if entry:
                parts.append(entry)
        return "; ".join(parts)
    return ""


def _serialize_list(items, delimiter="; ") -> str:
    """Safely serialize a list to string."""
    if not items:
        return ""
    if isinstance(items, str):
        return items
    if isinstance(items, list):
        return delimiter.join(str(i) for i in items if i)
    return str(items)


def index_cv(base_id: str, records: List[Dict[str, Any]], file_path: str = "") -> int:
    """
    Index multiple CV records, skipping duplicates based on content hash.
    Returns number of newly indexed records.
    """
    if not base_id:
        raise ValueError("base_id cannot be empty")

    if not records:
        logger.info("No records to index")
        return 0

    collection = vector_store.collection

    ids:       List[str]           = []
    texts:     List[str]           = []
    metadatas: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        # ── Compute hash ──
        record_hash = rec.get("_hash")
        if not record_hash:
            text_for_hash = (
                str(rec.get("name",    "")) +
                str(rec.get("email",   "")) +
                str(rec.get("skills",  "")) +
                str(rec.get("summary", ""))
            ).encode()
            record_hash = hashlib.sha256(text_for_hash).hexdigest()

        # ── Skip duplicates ──
        if _is_duplicate(collection, record_hash):
            logger.debug("Skipping duplicate record with hash %s...", record_hash[:8])
            continue

        # ── Build embedding text ──
        text = _build_embedding_text(rec)
        if not text:
            logger.debug("Record %d has no text, skipping", idx)
            continue

        # ── Serialize complex fields ──
        work_exp_str = _serialize_work_experience(rec.get("work_experience", []))
        edu_str      = _serialize_list(rec.get("education",      []))
        cert_str     = _serialize_list(rec.get("certifications", []), delimiter=", ")
        skills_raw   = rec.get("skills", [])
        skills_str   = ", ".join(skills_raw) if isinstance(skills_raw, list) else str(skills_raw or "")

        # ── Safe encrypt email ──
        raw_email = rec.get("email", "") or ""
        try:
            encrypted_email = encrypt(raw_email) if raw_email else ""
        except Exception:
            encrypted_email = ""

        ids.append(base_id + "_" + str(idx))
        texts.append(text)
        metadatas.append({
            "name":             str(rec.get("name",             "") or ""),
            "email":            encrypted_email,
            "phone":            str(rec.get("phone",            "") or ""),
            "skills":           skills_str,
            "years_experience": float(rec.get("years_experience", 0) or 0),
            "summary":          str(rec.get("summary",          "") or "")[:500],
            "work_experience":  work_exp_str[:1000],
            "education":        edu_str[:500],
            "certifications":   cert_str[:300],
            "location":         str(rec.get("location",         "") or ""),
            "seniority":        str(rec.get("seniority",        "") or ""),
            "education_level":  str(rec.get("education_level",  "") or ""),
            "industry":         str(rec.get("industry",         "") or ""),
            "remote_available": str(rec.get("remote_available", "") or ""),
            "file_path":        str(file_path or ""),
            "_hash":            record_hash,
            "_confidence":      float(rec.get("_confidence",    0)  or 0),
        })

    if not ids:
        logger.info("No new records to index (all duplicates or empty)")
        return 0

    # ── Generate embeddings in batches ──
    logger.info("Encoding %d new records...", len(texts))
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding", disable=None):
        batch      = texts[i:i + BATCH_SIZE]
        batch_embs = vector_store.embedder.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).tolist()
        embeddings.extend(batch_embs)

    # ── Upsert in batches ──
    logger.info("Upserting %d new records...", len(ids))
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="Upsert", disable=None):
        end = i + BATCH_SIZE
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            documents=texts[i:end],
        )

    collection.peek(1)  # force flush
    logger.info("Indexed %d new records for '%s'", len(ids), base_id)
    return len(ids)
