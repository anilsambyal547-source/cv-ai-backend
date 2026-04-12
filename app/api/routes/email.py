"""
Email endpoints – send email to candidate with custom body or template.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

from app.email.sender import get_sender
from app.email.tracker import EmailTracker
from app.storage.vector_store import vector_store
from app.api.deps import get_current_active_user
from app.utils.audit import log_access
from app.utils.encryption import decrypt
from config import config

router = APIRouter()
tracker = EmailTracker(config.EMAIL_TRACKER_DB)


class EmailRequest(BaseModel):
    candidate_id:    str
    candidate_email: Optional[str] = None   # frontend can pass directly
    candidate_name:  Optional[str] = None
    subject:         str
    body:            Optional[str] = None   # custom body from frontend
    template_variant: str = "default"
    schedule:        Optional[datetime] = None


def resolve_email(raw: str) -> str:
    """Decrypt email if encrypted."""
    if not raw:
        return ""
    if raw.startswith("gAAAAA"):
        try:
            return decrypt(raw)
        except Exception:
            return ""
    return raw


@router.post("/email", summary="Send or schedule an email to a candidate")
async def email_candidate(
    req: EmailRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_active_user),
):
    # Fetch candidate metadata
    result = vector_store.collection.get(
        ids=[req.candidate_id],
        include=["metadatas"],
    )
    if not result["ids"]:
        raise HTTPException(status_code=404, detail="Candidate not found")

    meta  = result["metadatas"][0]
    email = req.candidate_email or resolve_email(meta.get("email", ""))
    name  = req.candidate_name  or meta.get("name", "Candidate")

    if not email:
        raise HTTPException(status_code=400, detail="Candidate has no email address")

    # Build body — use custom body if provided, else use template file
    if req.body:
        body_text = req.body
    else:
        template_path = config.EMAIL_TEMPLATE_PATH
        if req.template_variant != "default":
            template_path = template_path.with_stem(
                "email_template_" + req.template_variant
            )
        if not template_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Template not found: " + str(template_path),
            )
        body_text = template_path.read_text(encoding="utf-8").replace(
            "{name}", name
        )

    if req.schedule:
        tracker.schedule_email(
            candidate_id=req.candidate_id,
            candidate_email=email,
            candidate_name=name,
            subject=req.subject,
            template_variant=req.template_variant,
            scheduled_time=req.schedule,
        )
        log_access(
            user=user["username"],
            action="SCHEDULE_EMAIL",
            resource="CANDIDATE",
            resource_id=req.candidate_id,
            details={"subject": req.subject},
        )
        return {"message": "Email scheduled for " + str(req.schedule)}

    # Send immediately in background
    def send():
        try:
            sender = get_sender(config.CREDENTIALS_PATH, config.TOKEN_PATH)
            sender.authenticate()
            msg_id = sender.send_message(email, req.subject, body_text)
            if msg_id:
                tracker.log_sent(
                    candidate_id=req.candidate_id,
                    candidate_email=email,
                    candidate_name=name,
                    subject=req.subject,
                    template_variant=req.template_variant,
                    message_id=msg_id,
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Email send failed: " + str(e))

    background_tasks.add_task(send)
    log_access(
        user=user["username"],
        action="SEND_EMAIL",
        resource="CANDIDATE",
        resource_id=req.candidate_id,
        details={"subject": req.subject, "to": email},
    )
    return {"message": "Email sending started", "to": email}


@router.get("/email/history", summary="Get email history")
async def email_history(user=Depends(get_current_active_user)):
    try:
        stats = tracker.get_stats()
        return stats
    except Exception:
        return []