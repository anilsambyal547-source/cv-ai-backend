"""
Audit logging utility for tracking access to candidate data.
Logs are written to a dedicated audit.log file with PII masking.
"""
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from config import config
from app.utils.log_filter import PIIMaskingFilter

# Create a dedicated audit logger
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False

# Add handler only if not already configured
if not audit_logger.handlers:
    log_file = config.LOGS_DIR / "audit.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)

    # Apply PII masking to audit logs
    pii_filter = PIIMaskingFilter()
    handler.addFilter(pii_filter)

    audit_logger.addHandler(handler)

def log_access(
    user: str,
    action: str,
    resource: str,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an audit event.

    Args:
        user: Username of the actor
        action: e.g., "VIEW", "SEARCH", "EMAIL", "INGEST"
        resource: e.g., "CANDIDATE", "QUERY", "PDF"
        resource_id: ID of the specific candidate (if applicable)
        details: Additional structured data (e.g., search query, filename)
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": user,
        "action": action,
        "resource": resource,
        "resource_id": resource_id,
        "details": details or {},
    }
    audit_logger.info(json.dumps(entry))