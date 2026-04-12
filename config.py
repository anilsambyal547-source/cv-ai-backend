"""
Production‑ready configuration with validation and path management.
Includes settings for hybrid extraction, skill normalization, scoring weights,
JD matching, cross‑encoder reranking, skill priority, email tracking, API, and Redis.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        logging.warning(f"Invalid integer for {name}, using default {default}")
        return default

def _clamp_float(name: str, value: float, lo: float, hi: float) -> float:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")
    return value

def _positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value

def _parse_skill_synonyms(env_var: str) -> Dict[str, List[str]]:
    import json
    val = _env_str(env_var)
    if val:
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            logging.warning(f"Invalid JSON in {env_var}, using default synonyms.")
    return {
        "python": ["python3", "py", "python2", "python programming"],
        "nodejs": ["node", "node.js", "node js", "express"],
        "javascript": ["js", "ecmascript"],
        "typescript": ["ts"],
        "java": ["java 8", "java 11", "j2ee"],
        "c++": ["cpp", "c plus plus"],
        "c#": ["csharp", "c sharp"],
        "sql": ["mysql", "postgresql", "sqlite", "oracle sql"],
        "nosql": ["mongodb", "cassandra", "dynamodb"],
        "aws": ["amazon web services", "ec2", "s3", "lambda"],
        "docker": ["container", "docker compose"],
        "kubernetes": ["k8s"],
        "react": ["reactjs", "react.js"],
        "angular": ["angularjs", "angular 2+"],
        "vue": ["vuejs", "vue.js"],
        "django": ["django framework"],
        "flask": ["flask framework"],
        "fastapi": ["fast api"],
        "git": ["github", "gitlab", "bitbucket"],
        "jenkins": ["jenkins ci"],
        "agile": ["scrum", "kanban"],
        "project management": ["pmp", "prince2"],
    }

def _parse_skill_priority(env_var: str) -> Dict[str, float]:
    import json
    val = _env_str(env_var)
    if val:
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            logging.warning(f"Invalid JSON in {env_var}, ignoring.")
    return {}

# ----------------------------------------------------------------------
# Config class
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Config:
    # -------------------- Base paths --------------------
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    DATA_DIR: Path = field(init=False)
    RAW_CVS_DIR: Path = field(init=False)
    PROCESSED_DIR: Path = field(init=False)
    EXPORTS_DIR: Path = field(init=False)
    LOGS_DIR: Path = field(init=False)

    # -------------------- Vector store --------------------
    CHROMA_DB_DIR: Path = field(init=False)
    BM25_INDEX_PATH: Path = field(init=False)
    GOLDEN_SET_PATH: Path = field(init=False)

    # -------------------- Email / Gmail --------------------
    CREDENTIALS_PATH: Path = field(init=False)
    TOKEN_PATH: Path = field(init=False)
    EMAIL_TEMPLATE_PATH: Path = field(init=False)
    EMAIL_TRACKER_DB: Path = field(init=False)          # SQLite DB for email tracking
    EMAIL_COOLDOWN_DAYS: int = 30                        # prevent duplicate outreach

    # -------------------- API keys --------------------
    GOOGLE_API_KEY: Optional[str] = field(
        default_factory=lambda: _env_str("GOOGLE_API_KEY") or _env_str("GEMINI_API_KEY")
    )

    # -------------------- JWT Authentication --------------------
    JWT_SECRET: str = field(
        default_factory=lambda: _env_str("JWT_SECRET", "change-this-secret-in-production")
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = field(
        default_factory=lambda: _env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
    )
    # -------------------- Encryption --------------------
    ENCRYPTION_KEY: str = field(default_factory=lambda: _env_str("ENCRYPTION_KEY", ""))
    # -------------------- Rate Limiting --------------------
    RATE_LIMIT_DEFAULT: str = field(
        default_factory=lambda: _env_str("RATE_LIMIT_DEFAULT", "100/minute")
    )

    # -------------------- Redis --------------------
    REDIS_HOST: str = field(default_factory=lambda: _env_str("REDIS_HOST", "localhost"))
    REDIS_PORT: int = field(default_factory=lambda: _env_int("REDIS_PORT", 6379))
    REDIS_DB: int = field(default_factory=lambda: _env_int("REDIS_DB", 0))
    REDIS_PASSWORD: Optional[str] = field(default_factory=lambda: _env_str("REDIS_PASSWORD", None))
    REDIS_CACHE_TTL: int = field(default_factory=lambda: _env_int("REDIS_CACHE_TTL", 3600))  # seconds

    # -------------------- Models --------------------
    PREFERRED_MODEL: Optional[str] = field(default_factory=lambda: _env_str("GEMINI_MODEL"))
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = 20

    # -------------------- Chroma --------------------
    CHROMA_COLLECTION: str = "cvs"
    CHROMA_TELEMETRY: bool = False

    # -------------------- Search defaults --------------------
    HYBRID_ALPHA_DEFAULT: float = 0.5
    MAX_SEARCH_LIMIT: int = 100
    MAX_QUERY_LENGTH: int = 500

    # -------------------- Extraction --------------------
    EXTRACTION_TIMEOUT: int = 30
    EXTRACTION_MAX_RETRIES: int = 3
    EXTRACTION_MIN_TEXT_LENGTH: int = 50
    EXTRACTION_MAX_TOKENS: int = 4096

    # -------------------- Ingestion --------------------
    INGESTION_BATCH_SIZE: int = 10
    INGESTION_RATE_LIMIT: float = 2.0

    # -------------------- Email --------------------
    EMAIL_SUBJECT: str = field(default_factory=lambda: _env_str("EMAIL_SUBJECT", "Job Opportunity"))

    # -------------------- Skill Normalization --------------------
    SKILL_SYNONYMS: Dict[str, List[str]] = field(
        default_factory=lambda: _parse_skill_synonyms("SKILL_SYNONYMS")
    )

    # -------------------- Skill Priority --------------------
    SKILL_PRIORITY: Dict[str, float] = field(
        default_factory=lambda: _parse_skill_priority("SKILL_PRIORITY")
    )

    # -------------------- Scoring Weights --------------------
    SCORE_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "semantic": 0.35,
            "skill_overlap": 0.25,
            "experience": 0.15,
            "education": 0.10,
            "seniority": 0.05,
            "bm25": 0.05,
            "recency": 0.05,
        }
    )

    # -------------------- Job Description Matching --------------------
    JD_MATCH_MIN_SKILLS: int = 2

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.DATA_DIR = self.BASE_DIR / "data"
        self.RAW_CVS_DIR = self.DATA_DIR / "raw_cvs"
        self.PROCESSED_DIR = self.DATA_DIR / "processed"
        self.EXPORTS_DIR = self.DATA_DIR / "exports"
        self.LOGS_DIR = self.BASE_DIR / "logs"

        self.CHROMA_DB_DIR = self.DATA_DIR / "chroma_db"
        self.BM25_INDEX_PATH = self.DATA_DIR / "bm25_index.pkl"
        self.GOLDEN_SET_PATH = self.BASE_DIR / "golden_set.json"

        self.CREDENTIALS_PATH = self.BASE_DIR / "credentials.json"
        self.TOKEN_PATH = self.DATA_DIR / "token.pickle"
        self.EMAIL_TEMPLATE_PATH = self.BASE_DIR / "email_template.txt"
        self.EMAIL_TRACKER_DB = self.DATA_DIR / "email_tracker.db"

        # Validate numeric fields
        self.HYBRID_ALPHA_DEFAULT = _clamp_float(
            "HYBRID_ALPHA_DEFAULT", self.HYBRID_ALPHA_DEFAULT, 0.0, 1.0
        )
        self.MAX_SEARCH_LIMIT = _positive_int("MAX_SEARCH_LIMIT", self.MAX_SEARCH_LIMIT)
        self.INGESTION_BATCH_SIZE = _positive_int("INGESTION_BATCH_SIZE", self.INGESTION_BATCH_SIZE)
        self.EXTRACTION_MAX_RETRIES = _positive_int("EXTRACTION_MAX_RETRIES", self.EXTRACTION_MAX_RETRIES)
        self.ACCESS_TOKEN_EXPIRE_MINUTES = _positive_int(
            "ACCESS_TOKEN_EXPIRE_MINUTES", self.ACCESS_TOKEN_EXPIRE_MINUTES
        )

        total = sum(self.SCORE_WEIGHTS.values())
        if not abs(total - 1.0) < 0.001:
            raise ValueError(f"SCORE_WEIGHTS must sum to 1.0, got {total}")

        # Create directories
        for d in (
            self.DATA_DIR,
            self.RAW_CVS_DIR,
            self.PROCESSED_DIR,
            self.EXPORTS_DIR,
            self.LOGS_DIR,
            self.CHROMA_DB_DIR,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def require_google_api_key(self) -> str:
        if not self.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is required for Gemini operations. "
                "Set it in your .env file."
            )
        return self.GOOGLE_API_KEY


config = Config()


def setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    log_file = config.LOGS_DIR / "system.log"

    # Create handlers
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add PII masking filter
    from app.utils.log_filter import PIIMaskingFilter
    pii_filter = PIIMaskingFilter()
    file_handler.addFilter(pii_filter)
    stream_handler.addFilter(pii_filter)

    # Add handlers to root logger
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root.setLevel(logging.INFO)